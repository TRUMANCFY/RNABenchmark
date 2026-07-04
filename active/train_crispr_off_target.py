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
import transformers
import sklearn
import scipy
import numpy as np
import re
from torch.utils.data import Dataset, Subset  # CHANGE: added Subset for AL pool management

import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
from model.rnalm.modeling_rnalm import RnaLmForCRISPROffTarget
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.modeling_rnafm import RnaFmForCRISPROffTarget
from model.rnabert.modeling_rnabert import RnaBertForCRISPROffTarget
from model.rnamsm.modeling_rnamsm import RnaMsmForCRISPROffTarget
from model.splicebert.modeling_splicebert import SpliceBertForCRISPROffTarget
from model.utrbert.modeling_utrbert import UtrBertForCRISPROffTarget
from model.utrlm.modeling_utrlm import UtrLmForCRISPROffTarget
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer

# CHANGE: removed global early_stopping — now created fresh per-round with configurable patience


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
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})


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
    evaluation_strategy: str = field(default="steps")  # CHANGE: removed trailing comma (was tuple bug in original)
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
    metric_for_best_model: str = field(default="spearman")  # unchanged: sequence-level Spearman
    stage: str = field(default='0')
    model_type: str = field(default='rna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    patience: int = field(default=10, metadata={"help": "Early stopping patience"})  # CHANGE: configurable patience (was hardcoded 10)


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
# Utilities (unchanged from original)
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
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from sequence."""
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each sequence."""
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
# Dataset (unchanged from original)
# ============================================================

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning (CRISPR off-target — sequence-level regression)."""

    def __init__(self, data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):
        super(SupervisedDataset, self).__init__()

        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 3:
            sgrna = [d[0].upper().replace("U", "T") for d in data]
            target = [d[1].upper().replace("U", "T") for d in data]
            labels = [float(d[2]) for d in data]
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")

        labels = np.array(labels)
        labels = labels.tolist()

        if kmer != -1:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()
            logging.warning(f"Using {kmer}-mer as input...")
            sgrna = load_or_generate_kmer(data_path.replace('.csv', '_sgrna.csv'), sgrna, kmer)
            target = load_or_generate_kmer(data_path.replace('.csv', '_target.csv'), target, kmer)
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        self.sgrna = sgrna
        self.target = target
        self.labels = labels
        self.num_labels = 1

    def __len__(self):
        return len(self.target)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.sgrna[i],
            target_input_ids=self.target[i],
            labels=self.labels[i],
        )


# ============================================================
# Data Collator (unchanged from original)
# Note: This collator produces labels as a simple (B,) tensor —
# NO struct mask, NO flattening. So logits will be (B, 1) and
# there is a 1:1 correspondence between samples and outputs.
# ============================================================

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sgrna, target, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "target_input_ids", "labels")
        )
        sgrna_output = self.tokenizer(
            sgrna, padding='longest',
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt',
        )
        target_output = self.tokenizer(
            target, padding='longest',
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt',
        )
        sgrna_input_ids = sgrna_output["input_ids"]
        sgrna_attention_mask = sgrna_output["attention_mask"]
        target_input_ids = target_output["input_ids"]
        target_attention_mask = target_output["attention_mask"]
        labels = torch.Tensor(labels).float()
        return dict(
            input_ids=sgrna_input_ids,
            labels=labels,
            attention_mask=sgrna_attention_mask,
            target_input_ids=target_input_ids,
            target_attention_mask=target_attention_mask,
        )


# ============================================================
# Metrics (sequence-level regression: Spearman + MSE, unchanged)
# ============================================================

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze()
    logits = logits.squeeze()
    return {
        "mse": sklearn.metrics.mean_squared_error(labels, logits),
        "spearman": scipy.stats.spearmanr(labels, logits)[0],
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)


# ============================================================
# CHANGE: Active Learning Acquisition Functions
#
# This is a SEQUENCE-LEVEL regression task (one scalar per sample).
# The collator produces labels as (B,) with no mask flattening,
# and the model returns logits of shape (B, 1).
# So unlike the structural imputation task, there is a guaranteed
# 1:1 correspondence between pool samples and logit entries.
#
# We still use per-sample MC-Dropout scoring for consistency and
# include safety assertions to prevent any IndexError.
# ============================================================

def _enable_mc_dropout(model):
    """Turn on dropout layers at inference time for MC-Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples: int):
    """
    CHANGE: Run multiple MC-Dropout forward passes over the unlabeled pool.

    For this sequence-level regression task, model returns logits of shape
    (B, 1) — exactly one scalar per sample. No mask flattening occurs.

    Returns:
        np.ndarray of shape (num_mc_samples, N) where N = len(pool_dataset).
        Each entry is the scalar prediction for that sample in that MC pass.
    """
    loader = torch.utils.data.DataLoader(
        pool_dataset,
        batch_size=trainer.args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=trainer.args.dataloader_num_workers,
    )
    device = trainer.model.device

    N = len(pool_dataset)
    all_mc_preds = np.zeros((num_mc_samples, N), dtype=np.float32)

    for t in range(num_mc_samples):
        trainer.model.eval()
        _enable_mc_dropout(trainer.model)

        sample_idx = 0
        for batch in loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = trainer.model(**batch_on_device)
            # logits shape: (B, 1) for sequence-level regression
            logits = outputs.logits.cpu().numpy().squeeze(-1)  # (B,)
            batch_size = logits.shape[0]

            all_mc_preds[t, sample_idx:sample_idx + batch_size] = logits
            sample_idx += batch_size

        # CHANGE: verify we filled exactly N entries
        assert sample_idx == N, \
            f"Expected {N} predictions, got {sample_idx}. DataLoader issue."

    trainer.model.eval()  # restore full eval mode
    return all_mc_preds  # shape: (num_mc_samples, N) — guaranteed one per sample


def acquire_random(pool_size: int, budget: int, **kwargs) -> np.ndarray:
    """Random acquisition baseline."""
    return np.random.choice(pool_size, size=budget, replace=False)


def acquire_entropy(trainer, pool_dataset, data_collator, budget: int,
                    num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    CHANGE: For sequence-level regression, 'entropy' = predictive variance
    from MC-Dropout. Higher variance = more uncertain.
    """
    mc_preds = _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    # mc_preds shape: (T, N) — one scalar per sample
    variance = mc_preds.var(axis=0)  # (N,)
    top_indices = np.argsort(variance)[::-1][:budget]
    # CHANGE: safety assertion
    assert top_indices.max() < mc_preds.shape[1], \
        f"Acquisition index {top_indices.max()} out of range for pool size {mc_preds.shape[1]}"
    return top_indices


def acquire_margin(trainer, pool_dataset, data_collator, budget: int,
                   num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    CHANGE: For sequence-level regression, 'margin' = predictive std
    from MC-Dropout. Higher std = less confident.
    """
    mc_preds = _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    std = mc_preds.std(axis=0)  # (N,)
    top_indices = np.argsort(std)[::-1][:budget]
    assert top_indices.max() < mc_preds.shape[1], \
        f"Acquisition index {top_indices.max()} out of range for pool size {mc_preds.shape[1]}"
    return top_indices


def acquire_bald(trainer, pool_dataset, data_collator, budget: int,
                 num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    CHANGE: BALD for sequence-level regression via MC-Dropout.
    Reduces to predictive variance (epistemic uncertainty).
    """
    mc_preds = _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    variance = mc_preds.var(axis=0)  # (N,)
    top_indices = np.argsort(variance)[::-1][:budget]
    assert top_indices.max() < mc_preds.shape[1], \
        f"Acquisition index {top_indices.max()} out of range for pool size {mc_preds.shape[1]}"
    return top_indices


def acquire_variation_ratio(trainer, pool_dataset, data_collator, budget: int,
                            num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    CHANGE: For sequence-level regression, use coefficient of variation
    (std / |mean|) as analog of variation ratio. Higher CV = more
    relative disagreement among MC samples.
    """
    mc_preds = _get_pool_predictions_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    mean_pred = mc_preds.mean(axis=0)  # (N,)
    std_pred = mc_preds.std(axis=0)    # (N,)
    cv = std_pred / (np.abs(mean_pred) + 1e-10)  # (N,)
    top_indices = np.argsort(cv)[::-1][:budget]
    assert top_indices.max() < mc_preds.shape[1], \
        f"Acquisition index {top_indices.max()} out of range for pool size {mc_preds.shape[1]}"
    return top_indices


# CHANGE: Registry of all acquisition functions
ACQUISITION_FUNCTIONS = {
    "random": acquire_random,
    "entropy": acquire_entropy,
    "margin": acquire_margin,
    "bald": acquire_bald,
    "variation_ratio": acquire_variation_ratio,
}


# ============================================================
# CHANGE: Model Builder
# Extracted from original train() so we can rebuild a fresh model each AL round.
# Note: uses *ForCRISPROffTarget model classes (not *ForStructuralimputation).
# ============================================================

def build_model(model_args, training_args, num_labels):
    """Instantiate a fresh model from the pretrained checkpoint."""
    if training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            print('Train from scratch')
            config = RnaLmConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=num_labels,
                problem_type="regression",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
            )
            model = RnaLmForCRISPROffTarget(config)
        else:
            print(f'Loading {training_args.model_type} model')
            model = RnaLmForCRISPROffTarget.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=num_labels,
                trust_remote_code=True,
                problem_type="regression",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
            )
    elif training_args.model_type == 'rna-fm':
        print(f'Loading {training_args.model_type} model')
        model = RnaFmForCRISPROffTarget.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            problem_type="regression",
        )
    elif training_args.model_type == 'rnabert':
        print(f'Loading {training_args.model_type} model')
        model = RnaBertForCRISPROffTarget.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            problem_type="regression",
        )
    elif training_args.model_type == 'rnamsm':
        print(f'Loading {training_args.model_type} model')
        model = RnaMsmForCRISPROffTarget.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    elif 'splicebert' in training_args.model_type:
        print(f'Loading {training_args.model_type} model')
        model = SpliceBertForCRISPROffTarget.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    elif 'utrbert' in training_args.model_type:
        print(f'Loading {training_args.model_type} model')
        model = UtrBertForCRISPROffTarget.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="regression",
            trust_remote_code=True,
        )
    elif 'utr-lm' in training_args.model_type:
        print(f'Loading {training_args.model_type} model')
        model = UtrLmForCRISPROffTarget.from_pretrained(
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
# CHANGE: Checkpoint helpers (atomic round design for resumability)
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
    """Save AL checkpoint atomically (write tmp then rename)."""
    path = os.path.join(output_dir, "al_checkpoint.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)
    print(f"  [Checkpoint] Saved (round {state['last_completed_round'] + 1} complete)")


# ============================================================
# CHANGE: Main Active Learning Loop
# Replaces the original single-run train() function.
#
# Key differences from original:
#   - Parses ActiveLearningArguments in addition to original three
#   - Builds an AL schedule of increasing labeled pool fractions
#   - Each round: fresh model, train, evaluate, acquire new samples
#   - Per-sample MC-Dropout scoring for acquisition
#   - Atomic checkpointing for crash-safe resumption
#   - Metric: Spearman correlation (sequence-level regression)
#   - Model classes: *ForCRISPROffTarget
#   - build_model() does NOT take tokenizer (CRISPROffTarget models don't need it)
# ============================================================

def train():
    # CHANGE: parse ActiveLearningArguments in addition to the original three
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActiveLearningArguments)
    )
    model_args, data_args, training_args, al_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)

    # ---- Tokenizer (unchanged from original) ----
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

    # ---- Datasets (unchanged from original) ----
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
    num_labels = full_train_dataset.num_labels

    total_train_size = len(full_train_dataset)
    print(f"Total training set size: {total_train_size}")
    print(f"Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # ---- CHANGE: AL schedule ----
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
    print(f"  Best metric    : {training_args.metric_for_best_model} (Spearman)")

    print(f"\n  Epoch scaling preview:")
    for fr in fractions:
        n = max(1, int(total_train_size * fr))
        print(f"    frac={fr:.2f} ({n} samples) -> {int(round(base_epochs / fr))} epochs")

    # ---- CHANGE: Checkpoint / Resume ----
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

    # ---- CHANGE: Active Learning Loop (replaces single trainer.train()) ----
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

        # CHANGE: Create subset for current labeled pool
        labeled_subset = Subset(full_train_dataset, labeled_indices)

        # CHANGE: Build fresh model each round (no tokenizer arg for CRISPROffTarget)
        model = build_model(model_args, training_args, num_labels)

        # CHANGE: Per-round output directory
        round_output_dir = os.path.join(
            training_args.output_dir,
            f"round_{round_idx + 1}_frac_{labeled_fraction:.2f}",
        )

        # CHANGE: Clone training args for this round
        round_training_args = copy.deepcopy(training_args)
        round_training_args.output_dir = round_output_dir
        round_training_args.run_name = f"{training_args.run_name}_AL_r{round_idx + 1}"
        round_training_args.num_train_epochs = scaled_epochs

        # ---- Train (uses HF Trainer, same as original) ----
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=round_training_args,
            compute_metrics=compute_metrics,
            train_dataset=labeled_subset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            # CHANGE: create fresh callback per round with configurable patience
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.patience)],
        )
        trainer.train()

        # ---- Evaluate on val + test ----
        # CHANGE: load_best_model_at_end=True ensures trainer.model has best-val weights
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

        # CHANGE: print Spearman + MSE (the primary metrics for this task)
        print(f"  Val  Spearman: {val_results.get('eval_spearman', 'N/A')}, MSE: {val_results.get('eval_mse', 'N/A')}")
        print(f"  Test Spearman: {test_results.get('eval_spearman', 'N/A')}, MSE: {test_results.get('eval_mse', 'N/A')}")

        # CHANGE: Save round results to per-round directory
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

        # ---- CHANGE: Acquisition step (if not last round) ----
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
                # CHANGE: All non-random strategies use per-sample MC-Dropout scoring.
                # _get_pool_predictions_mc returns shape (T, N) where N = len(pool_subset),
                # so acquisition indices are guaranteed in [0, len(unlabeled_indices)).
                selected_pool_indices = acquire_fn(
                    trainer=trainer,
                    pool_dataset=pool_subset,
                    data_collator=data_collator,
                    budget=budget,
                    num_mc_samples=al_args.al_num_mc_samples,
                )

            # CHANGE: Map pool-relative indices back to global dataset indices
            newly_selected = [unlabeled_indices[i] for i in selected_pool_indices]
            labeled_indices = sorted(labeled_indices + newly_selected)
            unlabeled_indices = sorted(set(unlabeled_indices) - set(newly_selected))

            print(f"  New labeled pool size: {len(labeled_indices)}")

        # ---- CHANGE: Checkpoint after round fully completes (atomic) ----
        save_al_checkpoint(training_args.output_dir, {
            "last_completed_round": round_idx,
            "labeled_indices": labeled_indices,
            "unlabeled_indices": unlabeled_indices,
            "al_results": al_results,
        })

        # CHANGE: Free GPU memory between rounds
        del model, trainer
        torch.cuda.empty_cache()

    # ---- CHANGE: Save aggregate AL results across all rounds ----
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
        test_spearman = r["test_results"].get("eval_spearman", "N/A")
        test_mse = r["test_results"].get("eval_mse", "N/A")
        print(f"  Round {r['round']}: frac={r['labeled_fraction']:.2f}, "
              f"size={r['labeled_size']}, epochs={r['scaled_epochs']}, "
              f"test_Spearman={test_spearman}, test_MSE={test_mse}")
    print(f"\nAggregate results saved to: {aggregate_path}")


if __name__ == "__main__":
    train()