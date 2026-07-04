import warnings
warnings.filterwarnings("ignore")
import os
import hashlib
import wandb
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pdb
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import get_cosine_schedule_with_warmup

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from structure.data import SSDataset
from structure.lm import get_extractor
from structure.predictor import SSCNNPredictor
import scipy
from sklearn import metrics
import random
import json
import copy
import math


# ============================================================
# Utilities
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


def set_seed(seed):
    """Set seed for ALL random number generators to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"seed is fixed, seed = {seed}")


def bpe_position(texts, attn_mask, tokenizer):
    position_id = torch.zeros(attn_mask.shape)
    for i, text in enumerate(texts):
        text = tokenizer.tokenize(text)
        position_id[:, 0] = 1
        index = 0
        for j, token in enumerate(text):
            index = j + 1
            position_id[i, index] = len(token)
        position_id[i, index + 1] = 1
    return position_id


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze().astype(int)
    logits = logits.squeeze()
    probs = scipy.special.expit(logits)
    precision = precision_score(labels, probs > 0.5, average='binary')
    recall = recall_score(labels, probs > 0.5, average='binary')
    f1 = f1_score(labels, probs > 0.5, average='binary')
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def generate_wandb_run_id(args):
    """Deterministic wandb run ID from experiment config for resume support."""
    key_str = (
        f"{args.output_dir}|{args.al_strategy}|{args.seed}|"
        f"{args.model_type}|{args.token_type}|{args.lr}"
    )
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


# ============================================================
# AL State Checkpoint
#
# Design: a round is ATOMIC. The checkpoint is only written after
# a round fully completes (training + evaluation + acquisition).
# If a job is interrupted mid-round, that round restarts entirely.
# ============================================================

class ALStateManager:
    """
    Manages the active learning loop state for checkpointing and resuming.

    A round is treated as an atomic unit:
    - Training, evaluation, AND acquisition must ALL complete
    - Only then is the checkpoint updated
    - If interrupted mid-round, the entire round restarts on relaunch

    This eliminates partial-resume complexity and ensures the trained
    model used for acquisition is always the same one that produced
    the recorded test metrics.
    """

    def __init__(self, output_dir: str):
        self.checkpoint_path = os.path.join(output_dir, "al_state_checkpoint.json")

    def save(self, state: dict):
        """Save AL state atomically (write tmp then rename)."""
        tmp_path = self.checkpoint_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self.checkpoint_path)  # atomic on POSIX
        print(f"  [Checkpoint] Saved (round {state['last_completed_round'] + 1} done)")

    def load(self):
        """Load AL state if checkpoint exists, else return None."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r") as f:
                state = json.load(f)
            print(f"  [Checkpoint] Found checkpoint: {self.checkpoint_path}")
            print(f"  [Checkpoint] Rounds completed: {state['last_completed_round'] + 1}")
            return state
        return None

    @staticmethod
    def build_initial_state(all_indices_permutation, labeled_indices, unlabeled_indices):
        return {
            "last_completed_round": -1,
            "all_indices_permutation": all_indices_permutation,
            "labeled_indices": labeled_indices,
            "unlabeled_indices": unlabeled_indices,
            "al_results": [],
        }

    @staticmethod
    def validate_state(state, total_train_size):
        """Sanity-check a loaded checkpoint."""
        labeled = set(state["labeled_indices"])
        unlabeled = set(state["unlabeled_indices"])

        overlap = labeled & unlabeled
        assert len(overlap) == 0, \
            f"[Checkpoint ERROR] {len(overlap)} indices in BOTH labeled and unlabeled"

        all_covered = labeled | unlabeled
        expected = set(range(total_train_size))
        assert all_covered == expected, \
            f"[Checkpoint ERROR] Pools don't cover all indices: " \
            f"missing={len(expected - all_covered)}, extra={len(all_covered - expected)}"

        num_results = len(state["al_results"])
        expected_results = state["last_completed_round"] + 1
        assert num_results == expected_results, \
            f"[Checkpoint ERROR] {num_results} results but last_completed_round={state['last_completed_round']}"

        print(f"  [Checkpoint] Validated: {len(labeled)} labeled + "
              f"{len(unlabeled)} unlabeled = {total_train_size}, "
              f"{num_results} rounds done")


# ============================================================
# Collator
# ============================================================

class collator():
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        seqs = [x['seq'] for x in batch]
        struct = [x['struct'] for x in batch]

        max_len = max([len(seq) for seq in seqs])

        weight_mask = torch.ones((len(seqs), max_len + 2))
        if 'mer' in self.args.token_type:
            kmer = int(self.args.token_type[0])
            for i in range(1, kmer - 1):
                weight_mask[:, i + 1] = weight_mask[:, -i - 2] = 1 / (i + 1)
            weight_mask[:, kmer:-kmer] = 1 / kmer
            seqs = [generate_kmer_str(seq, kmer) for seq in seqs]

        data_dict = self.tokenizer(
            seqs,
            padding='longest',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        post_token_length = torch.zeros(data_dict['attention_mask'].shape)
        if self.args.token_type == 'bpe' or self.args.token_type == 'non-overlap':
            post_token_length = bpe_position(seqs, data_dict['attention_mask'], self.tokenizer)

        struct = np.array([
            np.pad(x, ((0, max_len - x.shape[0]), (0, max_len - x.shape[1])),
                   'constant', constant_values=-1)
            for x in struct
        ])
        struct = torch.tensor(struct)

        data_dict['struct'] = struct
        data_dict['weight_mask'] = weight_mask
        data_dict['post_token_length'] = post_token_length

        return data_dict


# ============================================================
# Evaluation
# ============================================================

def test(model, test_loader, accelerator):
    model.eval()
    outputs_list = []
    targets_list = []
    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            for key in data_dict:
                data_dict[key] = data_dict[key].to(accelerator.device)
            logits = model(data_dict)[:, 1:-1, 1:-1]
            labels = data_dict['struct']
            outputs_list.append(logits.detach().cpu().numpy().reshape(-1, 1))
            targets_list.append(labels.detach().cpu().numpy().reshape(-1, 1))
        logits = np.concatenate(outputs_list, axis=0)
        labels = np.concatenate(targets_list, axis=0)

    eval_metrics = calculate_metric_with_sklearn(logits, labels)
    print(f'\nTest: Precision: {eval_metrics["precision"]:.4f}, '
          f'Recall: {eval_metrics["recall"]:.4f}, F1: {eval_metrics["f1"]:.4f}')
    return eval_metrics


# ============================================================
# Acquisition Functions
# ============================================================

def _enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def _get_pool_logits(model, pool_loader, accelerator):
    model.eval()
    all_logits = []
    all_masks = []
    for data_dict in tqdm(pool_loader, desc="  [Acquisition] forward pass"):
        for key in data_dict:
            data_dict[key] = data_dict[key].to(accelerator.device)
        logits = model(data_dict)[:, 1:-1, 1:-1]
        labels = data_dict['struct']
        bsz = logits.shape[0]
        for b in range(bsz):
            all_masks.append((labels[b] != -1).cpu().numpy())
            all_logits.append(logits[b].cpu().numpy())
    return all_logits, all_masks


@torch.no_grad()
def _get_pool_logits_mc(model, pool_loader, accelerator, num_mc_samples: int):
    all_masks = []
    sample_shapes = []

    model.eval()
    for data_dict in pool_loader:
        for key in data_dict:
            data_dict[key] = data_dict[key].to(accelerator.device)
        labels = data_dict['struct']
        bsz = labels.shape[0]
        for b in range(bsz):
            all_masks.append((labels[b] != -1).cpu().numpy())
            sample_shapes.append(labels[b].shape)

    N = len(all_masks)
    mc_logits = [np.zeros((num_mc_samples,) + s, dtype=np.float32) for s in sample_shapes]

    for t in range(num_mc_samples):
        model.eval()
        _enable_mc_dropout(model)
        idx = 0
        for data_dict in pool_loader:
            for key in data_dict:
                data_dict[key] = data_dict[key].to(accelerator.device)
            logits = model(data_dict)[:, 1:-1, 1:-1]
            bsz = logits.shape[0]
            for b in range(bsz):
                mc_logits[idx][t] = logits[b].cpu().numpy()
                idx += 1

    model.eval()
    return mc_logits, all_masks


def acquire_random(pool_size: int, budget: int, **kwargs) -> np.ndarray:
    return np.random.choice(pool_size, size=budget, replace=False)


def acquire_entropy(model, pool_loader, accelerator, budget: int, **kwargs) -> np.ndarray:
    all_logits, all_masks = _get_pool_logits(model, pool_loader, accelerator)
    eps = 1e-10
    scores = []
    for logits_i, mask_i in zip(all_logits, all_masks):
        probs = scipy.special.expit(logits_i)
        entropy = -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))
        scores.append(entropy[mask_i].mean() if mask_i.sum() > 0 else 0.0)
    return np.argsort(scores)[::-1][:budget]


def acquire_margin(model, pool_loader, accelerator, budget: int, **kwargs) -> np.ndarray:
    all_logits, all_masks = _get_pool_logits(model, pool_loader, accelerator)
    scores = []
    for logits_i, mask_i in zip(all_logits, all_masks):
        probs = scipy.special.expit(logits_i)
        margin = np.abs(probs - 0.5)
        scores.append(margin[mask_i].mean() if mask_i.sum() > 0 else float('inf'))
    return np.argsort(scores)[:budget]


def acquire_bald(model, pool_loader, accelerator, budget: int,
                 num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    mc_logits, all_masks = _get_pool_logits_mc(model, pool_loader, accelerator, num_mc_samples)
    eps = 1e-10
    scores = []
    for mc_i, mask_i in zip(mc_logits, all_masks):
        mc_probs = scipy.special.expit(mc_i)
        mean_probs = mc_probs.mean(axis=0)
        pred_entropy = -(mean_probs * np.log(mean_probs + eps) +
                         (1 - mean_probs) * np.log(1 - mean_probs + eps))
        per_sample_entropy = -(mc_probs * np.log(mc_probs + eps) +
                               (1 - mc_probs) * np.log(1 - mc_probs + eps))
        expected_entropy = per_sample_entropy.mean(axis=0)
        bald = pred_entropy - expected_entropy
        scores.append(bald[mask_i].mean() if mask_i.sum() > 0 else 0.0)
    return np.argsort(scores)[::-1][:budget]


def acquire_variation_ratio(model, pool_loader, accelerator, budget: int,
                            num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    mc_logits, all_masks = _get_pool_logits_mc(model, pool_loader, accelerator, num_mc_samples)
    scores = []
    for mc_i, mask_i in zip(mc_logits, all_masks):
        mc_probs = scipy.special.expit(mc_i)
        mc_preds = (mc_probs > 0.5).astype(int)
        T = mc_preds.shape[0]
        sum_pos = mc_preds.sum(axis=0)
        mode_count = np.maximum(sum_pos, T - sum_pos)
        var_ratio = 1.0 - mode_count / T
        scores.append(var_ratio[mask_i].mean() if mask_i.sum() > 0 else 0.0)
    return np.argsort(scores)[::-1][:budget]


ACQUISITION_FUNCTIONS = {
    "random": acquire_random,
    "entropy": acquire_entropy,
    "margin": acquire_margin,
    "bald": acquire_bald,
    "variation_ratio": acquire_variation_ratio,
}


# ============================================================
# Single-round training
# ============================================================

def train_one_round(
    args, tokenizer, train_subset, val_dataset, test_dataset,
    accelerator, round_output_dir, scaled_epochs,
):
    """
    Train from scratch for one AL round.
    Returns: (best_test_metrics: dict, trained_model: nn.Module)
    """
    print(f"  Reloading extractor from {args.model_name_or_path}...")
    extractor_fresh, _ = get_extractor(args)

    model_config = extractor_fresh.config
    model = SSCNNPredictor(args, extractor_fresh, model_config, tokenizer, args.is_freeze)
    print(f"  Model params: {count_parameters(model)}")
    print(f"  Scaled epochs: {scaled_epochs}")

    collate_fn = collator(tokenizer, args)
    train_loader = DataLoader(
        train_subset, batch_size=args.per_device_train_batch_size,
        shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.per_device_eval_batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.per_device_eval_batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    per_steps_one_epoch = max(1,
        len(train_subset) // args.per_device_train_batch_size
        // accelerator.num_processes // args.gradient_accumulation_steps
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=per_steps_one_epoch * args.warmup_epos,
        num_training_steps=per_steps_one_epoch * scaled_epochs,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler,
    )

    criterion = nn.BCEWithLogitsLoss()
    best_val_f1 = 0.0
    best_test = {}
    # Initialize best_model_state to the initial weights
    unwrapped_model = accelerator.unwrap_model(model)
    best_model_state = copy.deepcopy(unwrapped_model.state_dict())
    # best_model_state = None  # <-- NEW: store best model weights
    last_val_f1 = 0.0
    early_stop_flag = 0

    # Evaluate every eval_every epochs, but always evaluate on the last epoch
    eval_every = max(1, scaled_epochs // args.al_epochs_per_round)
    print(f"  Eval every {eval_every} epochs (scaled_epochs={scaled_epochs}, base={args.al_epochs_per_round})")

    for epoch in range(scaled_epochs):
        model.train()
        train_loss_list = []
        start_time = time.time()

        for data_dict in tqdm(train_loader, desc=f"  Epoch {epoch}"):
            with accelerator.accumulate(model):
                logits = model(data_dict)[:, 1:-1, 1:-1]
                labels = data_dict['struct']
                label_mask = labels != -1
                loss = criterion(
                    logits[label_mask].reshape(-1, 1),
                    labels[label_mask].reshape(-1, 1),
                )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_scheduler.step()

            gather_loss = accelerator.gather(loss.detach().float()).mean().item()
            train_loss_list.append(gather_loss)
        
        # Skip evaluation unless it's an eval epoch or the last epoch
        is_last_epoch = (epoch == scaled_epochs - 1)
        if epoch % eval_every != 0 and not is_last_epoch:
            torch.cuda.empty_cache()
            end_time = time.time()
            if accelerator.is_main_process:
                print(f"  epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.6f}, "
                    f"loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.1f}s")
            continue

        print(f"  Epoch {epoch} — running evaluation:")
        val_metrics = test(model, val_loader, accelerator)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            print(f"  New best val F1: {best_val_f1:.4f} — evaluating test...")
            best_test = test(model, test_loader, accelerator)

            # Save best model weights in memory
            unwrapped_model = accelerator.unwrap_model(model)
            best_model_state = copy.deepcopy(unwrapped_model.state_dict())

        if val_metrics["f1"] > last_val_f1:
            early_stop_flag = 0
        else:
            early_stop_flag += 1

        if early_stop_flag >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

        last_val_f1 = val_metrics["f1"]

        end_time = time.time()
        if accelerator.is_main_process:
            print(f"  epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.6f}, "
                  f"loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.1f}s")

        torch.cuda.empty_cache()

    # Restore best model weights before returning
    if best_model_state is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(best_model_state)
        print(f"  Restored best model weights (val F1={best_val_f1:.4f})")

    # Save round results + model checkpoint
    if accelerator.is_main_process:
        results_path = os.path.join(round_output_dir, "results")
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(best_test, f, indent=4)
        with open(os.path.join(results_path, "val_best_f1.json"), "w") as f:
            json.dump({"best_val_f1": best_val_f1}, f, indent=4)

        # Save model checkpoint
        model_save_path = os.path.join(round_output_dir, "best_model.pt")
        
        if best_model_state is not None:
            torch.save(best_model_state, model_save_path)
            print(f"  Model saved to {model_save_path}")
    
    if not best_test:
        print("  No val improvement; evaluating test with best available model...")
        best_test = test(model, test_loader, accelerator)

    return best_test, model


# ============================================================
# Active Learning Main Loop
#
# Resume design:
#   - Checkpoint saved ONLY after a round fully completes
#     (training + evaluation + acquisition are atomic)
#   - On restart, incomplete rounds are re-executed from scratch
#   - Per-round seed ensures identical results if restarted
# ============================================================

def main(args):
    set_seed(args.seed)

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(
        kwargs_handlers=kwargs_handlers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        mixed_precision="fp16",
    )

    _, tokenizer = get_extractor(args)

    train_dataset = SSDataset(data_path=args.data_path, tokenizer=tokenizer, args=args, mode='train')
    val_dataset = SSDataset(data_path=args.data_path, tokenizer=tokenizer, args=args, mode='val')
    test_dataset = SSDataset(data_path=args.data_path, tokenizer=tokenizer, args=args, mode='test')

    total_train_size = len(train_dataset)
    print(f"Total training set: {total_train_size}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    # ---- AL schedule ----
    fractions = []
    frac = args.al_initial_fraction
    while frac <= args.al_target_fraction + 1e-9:
        fractions.append(frac)
        frac = round(frac + args.al_step_fraction, 10)

    round_sizes = [max(1, int(total_train_size * fr)) for fr in fractions]
    num_rounds = len(round_sizes)
    base_epochs = args.al_epochs_per_round

    print(f"\nActive Learning Configuration:")
    print(f"  Strategy        : {args.al_strategy}")
    print(f"  Fractions       : {[f'{fr:.1%}' for fr in fractions]}")
    print(f"  Round sizes     : {round_sizes}")
    print(f"  Base epochs     : {base_epochs}")
    print(f"  Patience        : {args.patience}")
    print(f"  MC samples      : {args.al_num_mc_samples}")

    print(f"\n  Epoch scaling preview:")
    for fr in fractions:
        n = max(1, int(total_train_size * fr))
        print(f"    frac={fr:.2f} ({n} samples) -> {int(round(base_epochs / fr))} epochs")

    # ---- Checkpoint / Resume ----
    os.makedirs(args.output_dir, exist_ok=True)
    state_mgr = ALStateManager(args.output_dir)
    saved_state = state_mgr.load()

    if saved_state is not None:
        # ---- RESUME ----
        ALStateManager.validate_state(saved_state, total_train_size)

        last_completed = saved_state["last_completed_round"]
        labeled_indices = saved_state["labeled_indices"]
        unlabeled_indices = saved_state["unlabeled_indices"]
        al_results = saved_state["al_results"]
        all_indices_permutation = saved_state["all_indices_permutation"]
        start_round = last_completed + 1

        if start_round >= num_rounds:
            print(f"\n  [Resume] All {num_rounds} rounds already completed.")
            if accelerator.is_main_process:
                aggregate_path = os.path.join(args.output_dir, "al_aggregate_results.json")
                with open(aggregate_path, "w") as f:
                    json.dump(al_results, f, indent=4)
            return

        print(f"  [Resume] Rounds 1-{start_round} done, starting round {start_round + 1}")
        print(f"  [Resume] Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")
    else:
        # ---- FRESH start ----
        all_indices = np.arange(total_train_size)
        np.random.shuffle(all_indices)
        all_indices_permutation = all_indices.tolist()

        labeled_indices = sorted(all_indices[:round_sizes[0]].tolist())
        unlabeled_indices = sorted(all_indices[round_sizes[0]:].tolist())
        al_results = []
        start_round = 0

        # Save initial state so the permutation is preserved even if round 1 crashes
        if accelerator.is_main_process:
            state_mgr.save(ALStateManager.build_initial_state(
                all_indices_permutation, labeled_indices, unlabeled_indices
            ))

    # ---- WandB ----
    if accelerator.is_main_process:
        wandb_name = (
            f'[RNA_SS_AL]{args.output_dir.split("/")[-1]}_{args.al_strategy}_'
            f'{args.model_type}_{args.token_type}_lr{args.lr}_seed{args.seed}'
        )
        wandb.init(
            project='SecondaryStructure_AL', mode='offline',
            name=wandb_name, id=generate_wandb_run_id(args), resume="allow",
        )

    # ---- Active Learning Loop ----
    for round_idx in range(start_round, num_rounds):
        current_labeled_size = len(labeled_indices)
        current_fraction = current_labeled_size / total_train_size
        scaled_epochs = int(round(base_epochs / current_fraction))
        round_seed = args.seed + round_idx * 1000

        # Set seed at the start of each round for reproducibility
        # If the round was interrupted and restarts, same seed -> same result
        set_seed(round_seed)

        print(f"\n{'=' * 70}")
        print(f"AL Round {round_idx + 1}/{num_rounds}")
        print(f"  Labeled   : {current_labeled_size}/{total_train_size} ({current_fraction * 100:.1f}%)")
        print(f"  Unlabeled : {len(unlabeled_indices)}")
        print(f"  Epochs    : {scaled_epochs} (base={base_epochs}, 1/{current_fraction:.2f})")
        print(f"  Seed      : {round_seed}")
        print(f"{'=' * 70}")

        labeled_subset = Subset(train_dataset, labeled_indices)
        round_output_dir = os.path.join(
            args.output_dir,
            f"round_{round_idx + 1}_frac_{current_fraction:.2f}",
        )

        # ---- STEP 1: Train ----
        best_test, trained_model = train_one_round(
            args=args, tokenizer=tokenizer,
            train_subset=labeled_subset,
            val_dataset=val_dataset, test_dataset=test_dataset,
            accelerator=accelerator,
            round_output_dir=round_output_dir,
            scaled_epochs=scaled_epochs,
        )

        print(f"  Test F1={best_test.get('f1', 'N/A')}, "
              f"P={best_test.get('precision', 'N/A')}, "
              f"R={best_test.get('recall', 'N/A')}")

        # ---- STEP 2: Acquire (if not last round) ----
        if round_idx < num_rounds - 1:
            next_size = round_sizes[round_idx + 1]
            budget = next_size - current_labeled_size

            if budget > 0 and len(unlabeled_indices) > 0:
                budget = min(budget, len(unlabeled_indices))
                print(f"  Acquiring {budget} samples via '{args.al_strategy}'...")

                # Deterministic acquisition seed
                acq_seed = args.seed + round_idx * 1000 + 500
                set_seed(acq_seed)

                pool_subset = Subset(train_dataset, unlabeled_indices)
                pool_loader = DataLoader(
                    pool_subset, batch_size=args.per_device_eval_batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    collate_fn=collator(tokenizer, args),
                )

                acquire_fn = ACQUISITION_FUNCTIONS[args.al_strategy]

                if args.al_strategy == "random":
                    selected = acquire_fn(pool_size=len(unlabeled_indices), budget=budget)
                elif args.al_strategy in ("bald", "variation_ratio"):
                    selected = acquire_fn(
                        model=trained_model, pool_loader=pool_loader,
                        accelerator=accelerator, budget=budget,
                        num_mc_samples=args.al_num_mc_samples,
                    )
                else:
                    selected = acquire_fn(
                        model=trained_model, pool_loader=pool_loader,
                        accelerator=accelerator, budget=budget,
                    )

                newly_selected = [unlabeled_indices[i] for i in selected]
                labeled_indices = sorted(labeled_indices + newly_selected)
                unlabeled_indices = sorted(set(unlabeled_indices) - set(newly_selected))
                print(f"  New labeled pool: {len(labeled_indices)}")

        # ---- STEP 3: Record & Checkpoint (round is now fully complete) ----
        round_record = {
            "round": round_idx + 1,
            "labeled_size": current_labeled_size,
            "labeled_fraction": current_fraction,
            "scaled_epochs": scaled_epochs,
            "round_seed": round_seed,
            "strategy": args.al_strategy,
            "test_results": best_test,
        }
        al_results.append(round_record)

        if accelerator.is_main_process:
            # Save round info
            info_path = os.path.join(round_output_dir, "results", "round_info.json")
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, "w") as f:
                json.dump({
                    "round": round_idx + 1,
                    "labeled_size": current_labeled_size,
                    "labeled_fraction": current_fraction,
                    "scaled_epochs": scaled_epochs,
                    "round_seed": round_seed,
                    "strategy": args.al_strategy,
                    "labeled_indices": labeled_indices,
                }, f, indent=4)

            # WandB
            log_dict = {
                "al_round": round_idx + 1,
                "labeled_fraction": current_fraction,
                "labeled_size": current_labeled_size,
                "scaled_epochs": scaled_epochs,
            }
            log_dict.update({f"test_{k}": v for k, v in best_test.items()})
            wandb.log(log_dict)

            # Checkpoint — this is the ONLY place state is saved per round
            # Everything above (training + acquisition) must succeed first
            state_mgr.save({
                "last_completed_round": round_idx,
                "all_indices_permutation": all_indices_permutation,
                "labeled_indices": labeled_indices,
                "unlabeled_indices": unlabeled_indices,
                "al_results": al_results,
            })

        # Free memory
        del trained_model
        torch.cuda.empty_cache()

    # ---- Aggregate results ----
    if accelerator.is_main_process:
        aggregate_path = os.path.join(args.output_dir, "al_aggregate_results.json")
        with open(aggregate_path, "w") as f:
            json.dump(al_results, f, indent=4)

        print(f"\n{'=' * 70}")
        print("Active Learning Complete")
        print(f"{'=' * 70}")
        for r in al_results:
            f1 = r["test_results"].get("f1", "N/A")
            prec = r["test_results"].get("precision", "N/A")
            rec = r["test_results"].get("recall", "N/A")
            print(f"  Round {r['round']}: frac={r['labeled_fraction']:.2f}, "
                  f"size={r['labeled_size']}, epochs={r['scaled_epochs']}, "
                  f"seed={r['round_seed']}, F1={f1}, P={prec}, R={rec}")
        print(f"\nResults: {aggregate_path}")

        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epos', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--model_scale', type=str, default='8m')
    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='bprna')
    parser.add_argument("--pretrained_lm_dir", type=str, default='')
    parser.add_argument('--data_path', default='')
    parser.add_argument('--model_name_or_path', default='output')
    parser.add_argument('--output_dir', default='./ckpts/')
    parser.add_argument('--model_type', type=str, default='rna')
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--bprna_dir', default='')
    parser.add_argument('--run_name', type=str, default="run")
    parser.add_argument('--token_type', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--train_from_scratch', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--data_train_path', default='')
    parser.add_argument('--data_val_path', default='')
    parser.add_argument('--data_test_path', default='')
    parser.add_argument('--attn_implementation', type=str, default="eager")
    parser.add_argument('--train_fraction', type=float, default=1.0)

    parser.add_argument('--al_strategy', type=str, default='entropy',
                        choices=['random', 'entropy', 'margin', 'bald', 'variation_ratio'])
    parser.add_argument('--al_initial_fraction', type=float, default=0.1)
    parser.add_argument('--al_target_fraction', type=float, default=0.5)
    parser.add_argument('--al_step_fraction', type=float, default=0.1)
    parser.add_argument('--al_epochs_per_round', type=int, default=100)
    parser.add_argument('--al_num_mc_samples', type=int, default=10)

    args = parser.parse_args()
    assert args.mode in ['bprna', 'pdb']

    main(args)