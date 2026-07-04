import os
import csv
import copy
import json
import logging
import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import random
from transformers import Trainer, TrainingArguments, BertTokenizer, EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback

import torch
import sklearn
import scipy
import transformers
import numpy as np
import re
from torch.utils.data import Dataset, Subset

import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.rnalm.modeling_rnalm import RnaLmForSequenceClassification
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.modeling_rnafm import RnaFmForSequenceClassification
from model.rnabert.modeling_rnabert import RnaBertForSequenceClassification
from model.rnamsm.modeling_rnamsm import RnaMsmForSequenceClassification
from model.splicebert.modeling_splicebert import SpliceBertForSequenceClassification
from model.utrbert.modeling_utrbert import UtrBertForSequenceClassification
from model.utrlm.modeling_utrlm import UtrLmForSequenceClassification
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer


# ============================================================
# Arguments
# ============================================================

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    tokenizer_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")  # CHANGE: explicit save_strategy to match evaluation_strategy
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    train_fraction: float = field(default=1.0, metadata={"help": "Fraction of training data to use (0.0-1.0)"})
    report_to: str = field(default="tensorboard")
    metric_for_best_model: str = field(default="r^2_mean")
    stage: str = field(default='0')
    model_type: str = field(default='rna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    patience: int = field(default=20, metadata={"help": "Early stopping patience"})  # CHANGE: configurable patience


# CHANGE: New dataclass for active learning arguments
@dataclass
class ActiveLearningArguments:
    """Arguments specific to the active learning loop."""
    al_strategy: str = field(
        default="entropy",
        metadata={"help": "Active learning acquisition strategy: random, entropy, margin, bald, variation_ratio"}
    )
    al_initial_fraction: float = field(
        default=0.1, metadata={"help": "Fraction of training data for initial labeled pool."}
    )
    al_target_fraction: float = field(
        default=0.5, metadata={"help": "Fraction of training data to reach by final AL round."}
    )
    al_step_fraction: float = field(
        default=0.1, metadata={"help": "Fraction of total training data to acquire per AL round."}
    )
    al_num_mc_samples: int = field(
        default=10, metadata={"help": "Number of MC dropout forward passes for BALD / variation_ratio."}
    )
    al_epochs_per_round: int = field(
        default=30, metadata={"help": "Base training epochs (for 100%% data); scaled inversely by fraction."}
    )


# ============================================================
# Utilities
# ============================================================

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed, seed = {args.seed}")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer


# ============================================================
# Dataset (Multi-label regression: 3 targets — ON, OFF, ON/OFF)
# ============================================================

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning (multi-label regression)."""

    def __init__(self, data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):
        super(SupervisedDataset, self).__init__()

        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 4:
            texts = [d[0].upper().replace("U", "T") for d in data]
            labels = [[float(d[i + 1]) for i in range(3)] for d in data]
        else:
            raise ValueError(f"Data format not supported. Expected 4 columns, got {len(data[0])}")

        labels = np.array(labels).tolist()

        if kmer != -1:
            # CHANGE: added distributed init guard
            if torch.distributed.is_initialized() and torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()
            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = 3  # ON, OFF, ON/OFF

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "attention_mask")
        )
        attention_mask = torch.stack(attention_mask)
        input_ids = torch.stack(input_ids)
        labels = torch.Tensor(labels).float()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


# ============================================================
# Metrics (multi-label regression: per-target R², MAE)
# ============================================================

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze()
    logits = logits.squeeze()
    labels_len = labels.shape[1]

    result = []
    for i in range(labels_len):
        correlation = scipy.stats.pearsonr(labels[:, i], logits[:, i])[0] ** 2
        result.append(correlation)
    result_mae = []
    for i in range(labels_len):
        mae = sklearn.metrics.mean_absolute_error(labels[:, i], logits[:, i])
        result_mae.append(mae)
    return {
        "r^2_mean": np.mean(result),
        "r^2_ON": result[0],
        "r^2_OFF": result[1],
        "r^2_ON_OFF": result[2],
        "MAE_mean": np.mean(result_mae),
        "MAE_ON": result_mae[0],
        "MAE_OFF": result_mae[1],
        "MAE_ON_OFF": result_mae[2],
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)


# ============================================================
# CHANGE: Active Learning Acquisition Functions
# (Multi-label regression: MC-Dropout variance averaged across
#  all 3 output dimensions for per-sample uncertainty)
# ============================================================

def _enable_mc_dropout(model):
    """Turn on dropout layers at inference time for MC-Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples: int):
    """
    Run multiple MC-Dropout forward passes over the unlabeled pool.
    Returns array of shape (num_mc_samples, N, num_labels).
    For this task: num_labels=3 (ON, OFF, ON/OFF).
    """
    loader = torch.utils.data.DataLoader(
        pool_dataset,
        batch_size=trainer.args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=trainer.args.dataloader_num_workers,
    )
    device = trainer.model.device
    all_mc_preds = []

    for _ in range(num_mc_samples):
        trainer.model.eval()
        _enable_mc_dropout(trainer.model)
        sample_preds = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = trainer.model(**batch)
            # logits shape: (B, 3) for multi-label regression
            sample_preds.append(outputs.logits.cpu())
        all_mc_preds.append(torch.cat(sample_preds, dim=0).numpy())

    trainer.model.eval()  # restore full eval mode
    return np.stack(all_mc_preds, axis=0)  # (T, N, 3)


def acquire_random(pool_size: int, budget: int, **kwargs) -> np.ndarray:
    """Random acquisition baseline."""
    return np.random.choice(pool_size, size=budget, replace=False)


def acquire_entropy(trainer, pool_dataset, data_collator, budget: int,
                    num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    For multi-label regression, 'entropy' = mean predictive variance across
    all output dimensions from MC-Dropout.
    Higher mean variance = more uncertain.
    """
    mc_preds = _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    # mc_preds shape: (T, N, 3)
    variance = mc_preds.var(axis=0)  # (N, 3) — per-target variance
    sample_scores = variance.mean(axis=1)  # (N,) — average across 3 targets
    return np.argsort(sample_scores)[::-1][:budget]


def acquire_margin(trainer, pool_dataset, data_collator, budget: int,
                   num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    For multi-label regression, 'margin' = mean predictive std across
    all output dimensions. Higher std = less confident.
    """
    mc_preds = _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    std = mc_preds.std(axis=0)  # (N, 3)
    sample_scores = std.mean(axis=1)  # (N,)
    return np.argsort(sample_scores)[::-1][:budget]


def acquire_bald(trainer, pool_dataset, data_collator, budget: int,
                 num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    BALD for multi-label regression via MC-Dropout.
    Reduces to mean predictive variance across output dims.
    """
    mc_preds = _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    variance = mc_preds.var(axis=0)  # (N, 3)
    sample_scores = variance.mean(axis=1)  # (N,)
    return np.argsort(sample_scores)[::-1][:budget]


def acquire_variation_ratio(trainer, pool_dataset, data_collator, budget: int,
                            num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    For multi-label regression, use mean coefficient of variation across
    all output dimensions: mean(std / |mean|) per sample.
    """
    mc_preds = _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    mean_pred = mc_preds.mean(axis=0)  # (N, 3)
    std_pred = mc_preds.std(axis=0)    # (N, 3)
    cv = std_pred / (np.abs(mean_pred) + 1e-10)  # (N, 3)
    sample_scores = cv.mean(axis=1)  # (N,)
    return np.argsort(sample_scores)[::-1][:budget]


ACQUISITION_FUNCTIONS = {
    "random": acquire_random,
    "entropy": acquire_entropy,
    "margin": acquire_margin,
    "bald": acquire_bald,
    "variation_ratio": acquire_variation_ratio,
}


# ============================================================
# CHANGE: Model Builder (extracted for per-round rebuild)
# ============================================================

def build_model(model_args, training_args, num_labels):
    """Instantiate a fresh model from the pretrained checkpoint."""
    if training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            config = RnaLmConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=num_labels,
                problem_type="regression",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
            )
            model = RnaLmForSequenceClassification(config)
        else:
            model = RnaLmForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=num_labels,
                trust_remote_code=True,
                problem_type="regression",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
            )
    elif training_args.model_type == 'rna-fm':
        model = RnaFmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    elif training_args.model_type == 'rnabert':
        model = RnaBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    elif training_args.model_type == 'rnamsm':
        model = RnaMsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    elif 'splicebert' in training_args.model_type:
        model = SpliceBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    elif 'utrbert' in training_args.model_type:
        model = UtrBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    elif 'utr-lm' in training_args.model_type:
        model = UtrLmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {training_args.model_type}")
    return model


# ============================================================
# CHANGE: Checkpoint helpers (atomic round design)
# ============================================================

def load_al_checkpoint(output_dir):
    path = os.path.join(output_dir, "al_checkpoint.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            state = json.load(f)
        print(f"  [Resume] Loaded checkpoint from {path}")
        print(f"  [Resume] Rounds completed: {state['last_completed_round'] + 1}")
        return state
    return None


def save_al_checkpoint(output_dir, state):
    path = os.path.join(output_dir, "al_checkpoint.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)
    print(f"  [Checkpoint] Saved (round {state['last_completed_round'] + 1} complete)")


# ============================================================
# CHANGE: Main Active Learning Loop
# ============================================================

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActiveLearningArguments)
    )
    model_args, data_args, training_args, al_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)

    # ---- Tokenizer ----
    if training_args.model_type == 'rnalm':
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right", use_fast=True, trust_remote_code=True,
        )
    elif training_args.model_type in [
        'rna-fm', 'rnabert', 'rnamsm',
        'splicebert-human510', 'splicebert-ms510', 'splicebert-ms1024',
        'utrbert-3mer', 'utrbert-4mer', 'utrbert-5mer', 'utrbert-6mer',
        'utr-lm-mrl', 'utr-lm-te-el',
    ]:
        tokenizer = OpenRnaLMTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right", use_fast=True, trust_remote_code=True,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right", use_fast=True, trust_remote_code=True,
        )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
    if 'mer' in training_args.token_type:
        data_args.kmer = int(training_args.token_type[0])

    # ---- Datasets ----
    full_train_dataset = SupervisedDataset(
        tokenizer=tokenizer, args=training_args,
        data_path=os.path.join(data_args.data_path, data_args.data_train_path),
        kmer=data_args.kmer,
    )
    val_dataset = SupervisedDataset(
        tokenizer=tokenizer, args=training_args,
        data_path=os.path.join(data_args.data_path, data_args.data_val_path),
        kmer=data_args.kmer,
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer, args=training_args,
        data_path=os.path.join(data_args.data_path, data_args.data_test_path),
        kmer=data_args.kmer,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, args=training_args)
    num_labels = full_train_dataset.num_labels  # 3

    total_train_size = len(full_train_dataset)
    print(f"Total training set size: {total_train_size}")
    print(f"Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    print(f"Number of labels (output dims): {num_labels}")

    # ---- AL schedule ----
    fractions = []
    frac = al_args.al_initial_fraction
    while frac <= al_args.al_target_fraction + 1e-9:
        fractions.append(frac)
        frac = round(frac + al_args.al_step_fraction, 10)

    round_sizes = [max(1, int(total_train_size * fr)) for fr in fractions]
    num_rounds = len(round_sizes)
    base_epochs = al_args.al_epochs_per_round

    print(f"\nActive Learning Configuration:")
    print(f"  Strategy       : {al_args.al_strategy}")
    print(f"  Fractions      : {[f'{fr:.1%}' for fr in fractions]}")
    print(f"  Round sizes    : {round_sizes}")
    print(f"  Base epochs    : {base_epochs} (scaled per round)")
    print(f"  Patience       : {training_args.patience}")
    print(f"  MC samples     : {al_args.al_num_mc_samples}")

    print(f"\n  Epoch scaling preview:")
    for fr in fractions:
        n = max(1, int(total_train_size * fr))
        print(f"    frac={fr:.2f} ({n} samples) -> {int(round(base_epochs / fr))} epochs")

    # ---- Checkpoint / Resume ----
    os.makedirs(training_args.output_dir, exist_ok=True)
    checkpoint = load_al_checkpoint(training_args.output_dir)

    if checkpoint is not None:
        labeled_indices = checkpoint["labeled_indices"]
        unlabeled_indices = checkpoint["unlabeled_indices"]
        al_results = checkpoint["al_results"]
        start_round = checkpoint["last_completed_round"] + 1
        print(f"  [Resume] Starting from round {start_round + 1}/{num_rounds}")
        print(f"  [Resume] Labeled: {len(labeled_indices)}, Unlabeled: {len(unlabeled_indices)}")
    else:
        all_indices = np.arange(total_train_size)
        np.random.shuffle(all_indices)
        labeled_indices = sorted(all_indices[:round_sizes[0]].tolist())
        unlabeled_indices = sorted(all_indices[round_sizes[0]:].tolist())
        al_results = []
        start_round = 0

        save_al_checkpoint(training_args.output_dir, {
            "last_completed_round": -1,
            "labeled_indices": labeled_indices,
            "unlabeled_indices": unlabeled_indices,
            "al_results": [],
        })

    # ---- Active Learning Loop ----
    for round_idx in range(start_round, num_rounds):
        current_labeled_size = len(labeled_indices)
        labeled_fraction = current_labeled_size / total_train_size
        scaled_epochs = int(round(base_epochs / labeled_fraction))

        print(f"\n{'=' * 60}")
        print(f"AL Round {round_idx + 1}/{num_rounds}")
        print(f"  Labeled pool size: {current_labeled_size}/{total_train_size} ({labeled_fraction * 100:.1f}%)")
        print(f"  Unlabeled pool   : {len(unlabeled_indices)}")
        print(f"  Epochs           : {scaled_epochs} (base={base_epochs}, scaled by 1/{labeled_fraction:.2f})")
        print(f"{'=' * 60}")

        labeled_subset = Subset(full_train_dataset, labeled_indices)

        model = build_model(model_args, training_args, num_labels)

        round_output_dir = os.path.join(
            training_args.output_dir,
            f"round_{round_idx + 1}_frac_{labeled_fraction:.2f}",
        )

        round_training_args = copy.deepcopy(training_args)
        round_training_args.output_dir = round_output_dir
        round_training_args.run_name = f"{training_args.run_name}_AL_r{round_idx + 1}"
        round_training_args.num_train_epochs = scaled_epochs

        # ---- Train ----
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=round_training_args,
            compute_metrics=compute_metrics,
            train_dataset=labeled_subset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.patience)],
        )
        trainer.train()

        # ---- Evaluate on val + test ----
        val_results = trainer.evaluate(eval_dataset=val_dataset)
        test_results = trainer.evaluate(eval_dataset=test_dataset)

        round_record = {
            "round": round_idx + 1,
            "labeled_size": current_labeled_size,
            "labeled_fraction": labeled_fraction,
            "scaled_epochs": scaled_epochs,
            "strategy": al_args.al_strategy,
            "val_results": val_results,
            "test_results": test_results,
        }
        al_results.append(round_record)

        print(f"  Val  R²_mean: {val_results.get('eval_r^2_mean', 'N/A')}")
        print(f"  Test R²_mean: {test_results.get('eval_r^2_mean', 'N/A')}, "
              f"R²_ON: {test_results.get('eval_r^2_ON', 'N/A')}, "
              f"R²_OFF: {test_results.get('eval_r^2_OFF', 'N/A')}, "
              f"R²_ON_OFF: {test_results.get('eval_r^2_ON_OFF', 'N/A')}")

        # Save round results
        results_path = os.path.join(round_output_dir, "results")
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "val_results.json"), "w") as f:
            json.dump(val_results, f, indent=4)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)
        with open(os.path.join(results_path, "round_info.json"), "w") as f:
            json.dump({
                "round": round_idx + 1,
                "labeled_size": current_labeled_size,
                "labeled_fraction": labeled_fraction,
                "scaled_epochs": scaled_epochs,
                "strategy": al_args.al_strategy,
                "labeled_indices": labeled_indices,
            }, f, indent=4)

        if training_args.save_model:
            trainer.save_state()

        # ---- Acquisition step (if not last round) ----
        if round_idx < num_rounds - 1:
            next_size = round_sizes[round_idx + 1]
            budget = next_size - current_labeled_size

            if budget <= 0 or len(unlabeled_indices) == 0:
                print("  No more samples to acquire. Stopping AL loop.")
                save_al_checkpoint(training_args.output_dir, {
                    "last_completed_round": round_idx,
                    "labeled_indices": labeled_indices,
                    "unlabeled_indices": unlabeled_indices,
                    "al_results": al_results,
                })
                break

            budget = min(budget, len(unlabeled_indices))
            print(f"  Acquiring {budget} new samples using '{al_args.al_strategy}' strategy...")

            pool_subset = Subset(full_train_dataset, unlabeled_indices)
            acquire_fn = ACQUISITION_FUNCTIONS[al_args.al_strategy]

            if al_args.al_strategy == "random":
                selected_pool_indices = acquire_fn(
                    pool_size=len(unlabeled_indices), budget=budget,
                )
            else:
                # All non-random strategies use MC-Dropout for regression
                selected_pool_indices = acquire_fn(
                    trainer=trainer,
                    pool_dataset=pool_subset,
                    data_collator=data_collator,
                    budget=budget,
                    num_mc_samples=al_args.al_num_mc_samples,
                )

            newly_selected = [unlabeled_indices[i] for i in selected_pool_indices]
            labeled_indices = sorted(labeled_indices + newly_selected)
            unlabeled_indices = sorted(set(unlabeled_indices) - set(newly_selected))

            print(f"  New labeled pool size: {len(labeled_indices)}")

        # ---- Checkpoint after round fully completes (atomic) ----
        save_al_checkpoint(training_args.output_dir, {
            "last_completed_round": round_idx,
            "labeled_indices": labeled_indices,
            "unlabeled_indices": unlabeled_indices,
            "al_results": al_results,
        })

        del model, trainer
        torch.cuda.empty_cache()

    # ---- Save aggregate AL results ----
    aggregate_path = os.path.join(training_args.output_dir, "al_aggregate_results.json")
    aggregate_summary = []
    for r in al_results:
        summary = {k: v for k, v in r.items() if k != "labeled_indices"}
        aggregate_summary.append(summary)
    with open(aggregate_path, "w") as f:
        json.dump(aggregate_summary, f, indent=4)

    print(f"\n{'=' * 60}")
    print("Active Learning complete. Summary:")
    print(f"{'=' * 60}")
    for r in aggregate_summary:
        tr = r["test_results"]
        print(f"  Round {r['round']}: frac={r['labeled_fraction']:.2f}, "
              f"size={r['labeled_size']}, epochs={r['scaled_epochs']}, "
              f"R²_mean={tr.get('eval_r^2_mean', 'N/A')}, "
              f"R²_ON={tr.get('eval_r^2_ON', 'N/A')}, "
              f"R²_OFF={tr.get('eval_r^2_OFF', 'N/A')}, "
              f"R²_ON_OFF={tr.get('eval_r^2_ON_OFF', 'N/A')}")
    print(f"\nAggregate results saved to: {aggregate_path}")


if __name__ == "__main__":
    train()