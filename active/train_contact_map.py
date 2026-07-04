import warnings
warnings.filterwarnings("ignore")
import os
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

from structure.data import SSDataset, ContactMapDataset
from structure.lm import get_extractor
from structure.predictor import SSCNNPredictor
import scipy
from sklearn import metrics
import random
import json
import copy
import math


# ============================================================
# Utilities (unchanged from original)
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed, seed = {args.seed}")


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


# ============================================================
# Metrics (unchanged from original contact map)
# ============================================================

def calculate_metric_with_sklearn(logits_list, labels_list):
    top_L_1_TP, top_L_1_FP = [], []
    top_L_2_TP, top_L_2_FP = [], []
    top_L_5_TP, top_L_5_FP = [], []
    top_L_10_TP, top_L_10_FP = [], []

    lengths = np.array([labels.shape[-1] for labels in labels_list])
    long_range_mask = np.zeros((lengths.max(), lengths.max()))
    long_range_mask[np.triu_indices(long_range_mask.shape[0], k=23)] = 1
    long_range_mask = long_range_mask.astype(bool)

    for logits, labels in zip(logits_list, labels_list):
        labels = labels.squeeze().astype(float)
        logits = logits.squeeze()
        logits = (logits + logits.T) / 2
        predictions = scipy.special.expit(logits)
        long_range_mask_tmp = long_range_mask[:labels.shape[-1], :labels.shape[-1]]

        long_range_labels = labels[long_range_mask_tmp].flatten()
        long_range_predictions = predictions[long_range_mask_tmp].flatten()

        L = labels.shape[-1]
        for factor in [1, 2, 5, 10]:
            length = L // factor
            top_L_indices = np.argsort(long_range_predictions)[-length:]
            top_L_predictions = long_range_predictions[top_L_indices]
            top_L_labels = long_range_labels[top_L_indices]
            top_L_predictions_over_threshold = top_L_predictions > 0.5

            true_positives = top_L_labels[top_L_predictions_over_threshold].sum()
            false_positives = (1 - top_L_labels[top_L_predictions_over_threshold]).sum()

            if factor == 1:
                top_L_1_TP.append(true_positives)
                top_L_1_FP.append(false_positives)
            elif factor == 2:
                top_L_2_TP.append(true_positives)
                top_L_2_FP.append(false_positives)
            elif factor == 5:
                top_L_5_TP.append(true_positives)
                top_L_5_FP.append(false_positives)
            elif factor == 10:
                top_L_10_TP.append(true_positives)
                top_L_10_FP.append(false_positives)

    top_L_1_precision = sum(top_L_1_TP) / (sum(top_L_1_TP) + sum(top_L_1_FP)) if (sum(top_L_1_TP) + sum(top_L_1_FP)) > 0 else 0.0
    top_L_2_precision = sum(top_L_2_TP) / (sum(top_L_2_TP) + sum(top_L_2_FP)) if (sum(top_L_2_TP) + sum(top_L_2_FP)) > 0 else 0.0
    top_L_5_precision = sum(top_L_5_TP) / (sum(top_L_5_TP) + sum(top_L_5_FP)) if (sum(top_L_5_TP) + sum(top_L_5_FP)) > 0 else 0.0
    top_L_10_precision = sum(top_L_10_TP) / (sum(top_L_10_TP) + sum(top_L_10_FP)) if (sum(top_L_10_TP) + sum(top_L_10_FP)) > 0 else 0.0

    return {
        "top_l_precision": top_L_1_precision,
        "top_l/2_precision": top_L_2_precision,
        "top_l/5_precision": top_L_5_precision,
        "top_l/10_precision": top_L_10_precision,
    }


# ============================================================
# Collator (unchanged from original)
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
# Evaluation (unchanged — keeps per-sample lists for top-L metric)
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
            outputs_list.append(logits.detach().cpu().numpy())
            targets_list.append(labels.detach().cpu().numpy())

    eval_metrics = calculate_metric_with_sklearn(outputs_list, targets_list)
    print(f'\nTest: Top-l precision: {eval_metrics["top_l_precision"]}, '
          f'Top-l/2 precision: {eval_metrics["top_l/2_precision"]}, '
          f'Top-l/5 precision: {eval_metrics["top_l/5_precision"]}, '
          f'Top-l/10 precision: {eval_metrics["top_l/10_precision"]}')
    return eval_metrics


# ============================================================
# Active Learning Acquisition Functions
# (Pairwise binary classification — same as SS task)
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
# Checkpoint helpers (same as SS v3)
# ============================================================

def load_al_checkpoint(output_dir):
    """Load AL checkpoint if it exists. Returns state dict or None."""
    path = os.path.join(output_dir, "al_checkpoint.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            state = json.load(f)
        print(f"  [Resume] Loaded checkpoint from {path}")
        print(f"  [Resume] Rounds completed: {state['last_completed_round'] + 1}")
        return state
    return None


def save_al_checkpoint(output_dir, state):
    """Save AL checkpoint atomically."""
    path = os.path.join(output_dir, "al_checkpoint.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)
    print(f"  [Checkpoint] Saved (round {state['last_completed_round'] + 1} complete)")


# ============================================================
# Single-round training (from scratch each round)
# ============================================================

def train_one_round(
    args, tokenizer, train_subset, val_dataset,
    test_dataset_list, test_dataloader_list, data_test_list,
    accelerator, round_output_dir, round_name, scaled_epochs,
):
    """
    Train from scratch for one AL round.
    - Extractor reloaded from original checkpoint.
    - Best model selected by val top_l_precision.
    - Evaluates on ALL test sets.
    Returns: (best_test_metrics_list: list[dict], trained_model: nn.Module)
    """
    print(f"  Reloading extractor from {args.model_name_or_path} (train from scratch)...")
    extractor_fresh, _ = get_extractor(args)

    model_config = extractor_fresh.config
    model = SSCNNPredictor(args, extractor_fresh, model_config, tokenizer, args.is_freeze)
    num_params = count_parameters(model)
    print(f"  Model params: {num_params}")
    print(f"  Scaled epochs for this round: {scaled_epochs}")

    collate_fn = collator(tokenizer, args)
    train_loader = DataLoader(
        train_subset, batch_size=args.per_device_train_batch_size,
        shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.per_device_eval_batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    per_steps_one_epoch = max(1,
        len(train_subset) // args.per_device_train_batch_size
        // accelerator.num_processes // args.gradient_accumulation_steps
    )
    num_warmup_steps = per_steps_one_epoch * args.warmup_epos
    num_training_steps = per_steps_one_epoch * scaled_epochs

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_metric = 0.0
    best_test = []
    # Initialize best_model_state to the initial weights
    unwrapped_model = accelerator.unwrap_model(model)
    best_model_state = copy.deepcopy(unwrapped_model.state_dict())
    last_val_metric = 0.0
    early_stop_flag = 0

    # Evaluate every eval_every epochs, but always evaluate on the last epoch
    eval_every = max(1, scaled_epochs // args.al_epochs_per_round)
    print(f"  Eval every {eval_every} epochs (scaled_epochs={scaled_epochs}, base={args.al_epochs_per_round})")

    for epoch in range(scaled_epochs):
        model.train()
        train_loss_list = []
        start_time = time.time()

        for data_dict in tqdm(train_loader, desc=f"  Round train epoch {epoch}"):
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
                    f"train_loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.1f}s")
            continue

        print(f"  Epoch {epoch} — running evaluation:")
        val_metrics = test(model, val_loader, accelerator)

        if val_metrics["top_l_precision"] > best_val_metric:
            best_val_metric = val_metrics["top_l_precision"]
            print(f"  New best val top_l_precision: {best_val_metric:.4f} — evaluating on test sets...")
            test_metrics_list = []
            for i, data_test in enumerate(data_test_list):
                print(f"    Evaluating on {data_test}:")
                test_metrics_list.append(test(model, test_dataloader_list[i], accelerator))
            best_test = test_metrics_list

            # Save best model weights in memory
            unwrapped_model = accelerator.unwrap_model(model)
            best_model_state = copy.deepcopy(unwrapped_model.state_dict())

        if val_metrics["top_l_precision"] > last_val_metric:
            early_stop_flag = 0
        else:
            early_stop_flag += 1

        if early_stop_flag >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

        last_val_metric = val_metrics["top_l_precision"]

        end_time = time.time()
        if accelerator.is_main_process:
            print(f"  epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.6f}, "
                  f"train_loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.1f}s")

        torch.cuda.empty_cache()
    
    # Restore best model weights before returning
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(best_model_state)
    print(f"  Restored best model weights (val top_l_precision={best_val_metric:.4f})")

    # Handle edge case: no epoch improved val metric
    if not best_test:
        print("  No val improvement; evaluating test with best available model...")
        best_test = []
        for i, data_test in enumerate(data_test_list):
            print(f"    Evaluating on {data_test}:")
            best_test.append(test(model, test_dataloader_list[i], accelerator))


    # Save round results
    if accelerator.is_main_process:
        results_path = os.path.join(round_output_dir, "results")
        os.makedirs(results_path, exist_ok=True)
        for i, data_test in enumerate(data_test_list):
            with open(os.path.join(results_path, f"{data_test}_results.json"), "w") as f:
                json.dump(best_test[i] if i < len(best_test) else {}, f, indent=4)
        with open(os.path.join(results_path, "val_best_top_l_precision.json"), "w") as f:
            json.dump({"best_val_top_l_precision": best_val_metric}, f, indent=4)

        # Save model checkpoint
        if best_model_state is not None:
            model_save_path = os.path.join(round_output_dir, "best_model.pt")
            torch.save(best_model_state, model_save_path)
            print(f"  Model saved to {model_save_path}")

    return best_test, model


# ============================================================
# Active Learning Main Loop
# ============================================================

def main(args):
    set_seed(args)

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(
        kwargs_handlers=kwargs_handlers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        mixed_precision="fp16",
    )

    # ---- Tokenizer only (extractor reloaded each round) ----
    _, tokenizer = get_extractor(args)

    # ---- Datasets ----
    train_dataset = ContactMapDataset(
        data_path=os.path.join(args.data_path, args.data_train_path),
        tokenizer=tokenizer, args=args,
    )
    val_dataset = ContactMapDataset(
        data_path=os.path.join(args.data_path, args.data_val_path),
        tokenizer=tokenizer, args=args,
    )

    # Multiple test sets
    data_test_list = args.data_test_path.replace(" ", "").split(",")
    test_dataset_list = []
    for data_test in data_test_list:
        data_test_name = data_test + ".csv"
        print(f"Loading test set: {data_test_name}")
        test_ds = ContactMapDataset(
            data_path=os.path.join(args.data_path, data_test_name),
            tokenizer=tokenizer, args=args,
        )
        test_dataset_list.append(test_ds)

    collate_fn = collator(tokenizer, args)
    test_dataloader_list = []
    for test_ds in test_dataset_list:
        test_dataloader_list.append(DataLoader(
            test_ds, batch_size=args.per_device_eval_batch_size,
            shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
        ))

    total_train_size = len(train_dataset)
    test_sizes = "+".join([str(len(ds)) for ds in test_dataset_list])
    print(f"Total training set: {total_train_size}, val: {len(val_dataset)}, test: {test_sizes}")

    # ---- AL schedule ----
    # AFTER (fixed — fractions are exact, counts derived from them)
    fractions = np.arange(
        args.al_initial_fraction,
        args.al_target_fraction + 1e-9,   # inclusive upper bound
        args.al_step_fraction,
    )
    round_sizes = [max(1, int(round(f * total_train_size))) for f in fractions]

    # Deduplicate while preserving order (edge case: tiny dataset)
    seen = set()
    round_sizes = [x for x in round_sizes if not (x in seen or seen.add(x))]

    num_rounds = len(round_sizes)
    initial_n  = round_sizes[0]
    target_n = round_sizes[-1]
    step_n = max(1, int(round(args.al_step_fraction * total_train_size)))

    base_epochs = args.al_epochs_per_round

    print(f"\nActive Learning Configuration:")
    print(f"  Strategy        : {args.al_strategy}")
    print(f"  Initial samples : {initial_n} ({args.al_initial_fraction * 100:.0f}%)")
    print(f"  Target samples  : {target_n} ({args.al_target_fraction * 100:.0f}%)")
    print(f"  Step size       : {step_n} ({args.al_step_fraction * 100:.0f}%)")
    print(f"  Rounds          : {num_rounds} -> sizes {round_sizes}")
    print(f"  Base epochs     : {base_epochs} (scaled per round to preserve total steps)")
    print(f"  MC samples      : {args.al_num_mc_samples}")
    print(f"  Test sets       : {data_test_list}")
    print(f"  Train from scratch each round: YES (extractor reloaded)")

    print(f"\n  Epoch scaling preview:")
    for rs in round_sizes:
        frac = rs / total_train_size
        scaled = int(round(base_epochs / frac))
        print(f"    frac={frac:.2f} ({rs} samples) -> {scaled} epochs")

    # ---- Try to resume from checkpoint ----
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = load_al_checkpoint(args.output_dir)

    if checkpoint is not None:
        labeled_indices = checkpoint["labeled_indices"]
        unlabeled_indices = checkpoint["unlabeled_indices"]
        al_results = checkpoint["al_results"]
        start_round = checkpoint["last_completed_round"] + 1
        print(f"  [Resume] Starting from round {start_round + 1}/{num_rounds}")
        print(f"  [Resume] Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")
    else:
        # Fresh start
        all_indices = np.arange(total_train_size)
        np.random.shuffle(all_indices)
        labeled_indices = sorted(all_indices[:initial_n].tolist())
        unlabeled_indices = sorted(all_indices[initial_n:].tolist())
        al_results = []
        start_round = 0

        # Save immediately so the initial split is preserved
        if accelerator.is_main_process:
            save_al_checkpoint(args.output_dir, {
                "last_completed_round": -1,
                "labeled_indices": labeled_indices,
                "unlabeled_indices": unlabeled_indices,
                "al_results": [],
            })

    # ---- WandB ----
    if accelerator.is_main_process:
        model_name = args.output_dir.split('/')[-1]
        wandb_name = (
            f'[ContactMap_AL]{model_name}_{args.al_strategy}_'
            f'{args.model_type}_{args.token_type}_'
            f'lr{args.lr}_seed{args.seed}'
        )
        wandb.init(project='ContactMap_AL', mode='offline', name=wandb_name)

    # ---- Active Learning Loop ----
    for round_idx in range(start_round, num_rounds):
        current_labeled_size = len(labeled_indices)
        current_fraction = current_labeled_size / total_train_size
        scaled_epochs = int(round(base_epochs / current_fraction))

        print(f"\n{'=' * 70}")
        print(f"AL Round {round_idx + 1}/{num_rounds}")
        print(f"  Labeled pool : {current_labeled_size}/{total_train_size} ({current_fraction * 100:.1f}%)")
        print(f"  Unlabeled    : {len(unlabeled_indices)}")
        print(f"  Epochs       : {scaled_epochs} (base={base_epochs}, scaled by 1/{current_fraction:.2f})")
        print(f"{'=' * 70}")

        labeled_subset = Subset(train_dataset, labeled_indices)

        round_output_dir = os.path.join(
            args.output_dir,
            f"round_{round_idx + 1}_frac_{current_fraction:.2f}",
        )
        round_name = f"{args.run_name}_AL_r{round_idx + 1}"

        # ---- Train from scratch ----
        best_test, trained_model = train_one_round(
            args=args,
            tokenizer=tokenizer,
            train_subset=labeled_subset,
            val_dataset=val_dataset,
            test_dataset_list=test_dataset_list,
            test_dataloader_list=test_dataloader_list,
            data_test_list=data_test_list,
            accelerator=accelerator,
            round_output_dir=round_output_dir,
            round_name=round_name,
            scaled_epochs=scaled_epochs,
        )

        # ---- Record ----
        test_results_by_set = {}
        for i, data_test in enumerate(data_test_list):
            test_results_by_set[data_test] = best_test[i] if i < len(best_test) else {}

        round_record = {
            "round": round_idx + 1,
            "labeled_size": current_labeled_size,
            "labeled_fraction": current_fraction,
            "scaled_epochs": scaled_epochs,
            "strategy": args.al_strategy,
            "test_results": test_results_by_set,
        }
        al_results.append(round_record)

        for data_test in data_test_list:
            tr = test_results_by_set.get(data_test, {})
            print(f"  [{data_test}] top_l={tr.get('top_l_precision', 'N/A')}, "
                  f"top_l/2={tr.get('top_l/2_precision', 'N/A')}, "
                  f"top_l/5={tr.get('top_l/5_precision', 'N/A')}, "
                  f"top_l/10={tr.get('top_l/10_precision', 'N/A')}")

        # Save round info
        if accelerator.is_main_process:
            info_path = os.path.join(round_output_dir, "results", "round_info.json")
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, "w") as f:
                json.dump({
                    "round": round_idx + 1,
                    "labeled_size": current_labeled_size,
                    "labeled_fraction": current_fraction,
                    "scaled_epochs": scaled_epochs,
                    "strategy": args.al_strategy,
                    "labeled_indices": labeled_indices,
                }, f, indent=4)

            log_dict = {
                "al_round": round_idx + 1,
                "labeled_fraction": current_fraction,
                "labeled_size": current_labeled_size,
                "scaled_epochs": scaled_epochs,
            }
            for data_test in data_test_list:
                tr = test_results_by_set.get(data_test, {})
                for k, v in tr.items():
                    log_dict[f"{data_test}_{k}"] = v
            wandb.log(log_dict)

        # ---- Acquisition (if not the last round) ----
        if round_idx < num_rounds - 1:
            next_size = round_sizes[round_idx + 1]
            budget = next_size - current_labeled_size

            if budget <= 0 or len(unlabeled_indices) == 0:
                print("  No more samples to acquire. Stopping.")
                if accelerator.is_main_process:
                    save_al_checkpoint(args.output_dir, {
                        "last_completed_round": round_idx,
                        "labeled_indices": labeled_indices,
                        "unlabeled_indices": unlabeled_indices,
                        "al_results": al_results,
                    })
                break

            budget = min(budget, len(unlabeled_indices))
            print(f"  Acquiring {budget} new samples via '{args.al_strategy}'...")

            pool_subset = Subset(train_dataset, unlabeled_indices)
            pool_collate_fn = collator(tokenizer, args)
            pool_loader = DataLoader(
                pool_subset, batch_size=args.per_device_eval_batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=pool_collate_fn,
            )

            acquire_fn = ACQUISITION_FUNCTIONS[args.al_strategy]

            if args.al_strategy == "random":
                selected_pool_indices = acquire_fn(
                    pool_size=len(unlabeled_indices), budget=budget,
                )
            elif args.al_strategy in ("bald", "variation_ratio"):
                selected_pool_indices = acquire_fn(
                    model=trained_model, pool_loader=pool_loader,
                    accelerator=accelerator, budget=budget,
                    num_mc_samples=args.al_num_mc_samples,
                )
            else:
                selected_pool_indices = acquire_fn(
                    model=trained_model, pool_loader=pool_loader,
                    accelerator=accelerator, budget=budget,
                )

            newly_selected = [unlabeled_indices[i] for i in selected_pool_indices]
            labeled_indices = sorted(labeled_indices + newly_selected)
            unlabeled_indices = sorted(set(unlabeled_indices) - set(newly_selected))

            print(f"  New labeled pool size: {len(labeled_indices)}")

        # ---- Save checkpoint after round fully completes ----
        if accelerator.is_main_process:
            save_al_checkpoint(args.output_dir, {
                "last_completed_round": round_idx,
                "labeled_indices": labeled_indices,
                "unlabeled_indices": unlabeled_indices,
                "al_results": al_results,
            })

        # Free GPU memory
        del trained_model
        torch.cuda.empty_cache()

    # ---- Save aggregate results ----
    if accelerator.is_main_process:
        aggregate_path = os.path.join(args.output_dir, "al_aggregate_results.json")
        with open(aggregate_path, "w") as f:
            json.dump(al_results, f, indent=4)

        print(f"\n{'=' * 70}")
        print("Active Learning Complete — Summary")
        print(f"{'=' * 70}")
        for r in al_results:
            line = (f"  Round {r['round']}: frac={r['labeled_fraction']:.2f}, "
                    f"size={r['labeled_size']}, epochs={r['scaled_epochs']}")
            for data_test in data_test_list:
                tr = r["test_results"].get(data_test, {})
                top_l = tr.get("top_l_precision", "N/A")
                line += f", {data_test}_top_l={top_l}"
            print(line)
        print(f"\nAggregate results: {aggregate_path}")

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # ---- Original arguments ----
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

    # ---- Active Learning arguments ----
    parser.add_argument('--al_strategy', type=str, default='entropy',
                        choices=['random', 'entropy', 'margin', 'bald', 'variation_ratio'],
                        help='Active learning acquisition strategy')
    parser.add_argument('--al_initial_fraction', type=float, default=0.1,
                        help='Fraction of training data for the initial labeled pool')
    parser.add_argument('--al_target_fraction', type=float, default=0.5,
                        help='Fraction of training data to reach by the final AL round')
    parser.add_argument('--al_step_fraction', type=float, default=0.1,
                        help='Fraction of total training data to acquire per AL round')
    parser.add_argument('--al_epochs_per_round', type=int, default=100,
                        help='Base training epochs (for 100%% data); scaled inversely by fraction')
    parser.add_argument('--al_num_mc_samples', type=int, default=10,
                        help='Number of MC-Dropout forward passes for BALD / variation_ratio')

    args = parser.parse_args()
    assert args.mode in ['bprna', 'pdb']

    main(args)