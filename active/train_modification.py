import os
import csv
import copy
import json
import logging
import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import random
import transformers
import sklearn
import numpy as np
import scipy
from torch.utils.data import Dataset, Subset
import pandas as pd

os.environ["WANDB_DISABLED"] = "true"
from sklearn.metrics import roc_auc_score, matthews_corrcoef

from transformers import Trainer, TrainingArguments, BertTokenizer, EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback
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
    evaluation_strategy: str = field(default="steps"),
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
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    train_fraction: float = field(default=1.0, metadata={"help": "Fraction of training data to use (0.0-1.0)"})
    report_to: str = field(default="tensorboard")
    metric_for_best_model: str = field(default="mean_auc")
    stage: str = field(default='0')
    model_type: str = field(default='rna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    patience: int = field(default=20, metadata={"help": "Early stopping patience"})


@dataclass
class ActiveLearningArguments:
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
# Dataset (unchanged from original)
# ============================================================

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning (multi-label classification, 12 classes)."""

    def __init__(self, data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):
        super(SupervisedDataset, self).__init__()

        data = pd.read_csv(data_path, sep=",")
        data['targets'] = data['label'].apply(lambda x: np.array(x.split(), dtype=np.int8))
        data = data[['sequence', 'targets']]
        data.columns = ['seq', 'targets']

        self.num_labels = 12
        self.kmer = kmer
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.data["seq"].iloc[idx].upper()
        labels = self.data["targets"].iloc[idx].astype(np.float32)
        if self.kmer != -1:
            sample = generate_kmer_str(sample, self.kmer)

        output = self.tokenizer(
            sample,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = output["input_ids"][0]
        attention_mask = output["attention_mask"][0]
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


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
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.tensor(np.array(labels)).float()
        attention_mask = torch.stack(attention_mask)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


# ============================================================
# Metrics (unchanged from original)
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    metrics = {}
    p = sigmoid(logits)
    y = labels
    aucs = np.zeros(12, dtype=np.float32)
    mcc_scores = []
    label_names = ['hAm', 'hCm', 'hGm', 'hUm', 'hm1A', 'hm5C', 'hm5U', 'hm6A', 'hm6Am', 'hm7G', 'hPsi', 'Atol']
    for i in range(12):
        try:
            mcc_score = matthews_corrcoef(y[:, i], p[:, i] > 0.5)
            mcc_scores.append(mcc_score)
            aucs[i] = roc_auc_score(y[:, i], p[:, i])
        except ValueError:
            aucs[i] = 0.5
    for i, name in enumerate(label_names):
        metrics[f'{name}_auc'] = float(aucs[i])
    metrics['mean_auc'] = float(aucs.mean())
    metrics['mean_mcc'] = float(np.mean(mcc_scores)) if mcc_scores else 0.0
    return metrics


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


# ============================================================
# Active Learning Acquisition Functions
# (Multi-label classification with 12 binary labels)
#
# Model output: (N, 12) logits per sample.
# Each of the 12 outputs is an independent binary classification.
# Strategy: compute uncertainty per label, then average across labels
# to get one score per sample.
# ============================================================

def _enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def _get_pool_logits(trainer, pool_dataset, data_collator):
    """Single forward pass. Returns logits (N, 12)."""
    loader = torch.utils.data.DataLoader(
        pool_dataset,
        batch_size=trainer.args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=trainer.args.dataloader_num_workers,
    )
    device = trainer.model.device
    all_logits = []
    trainer.model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = trainer.model(**batch)
        all_logits.append(outputs.logits.cpu())
    return torch.cat(all_logits, dim=0).numpy()  # (N, 12)


@torch.no_grad()
def _get_pool_logits_mc(trainer, pool_dataset, data_collator, num_mc_samples: int):
    """MC-Dropout forward passes. Returns (num_mc_samples, N, 12)."""
    loader = torch.utils.data.DataLoader(
        pool_dataset,
        batch_size=trainer.args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=trainer.args.dataloader_num_workers,
    )
    device = trainer.model.device
    all_mc_logits = []

    for _ in range(num_mc_samples):
        trainer.model.eval()
        _enable_mc_dropout(trainer.model)
        sample_logits = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = trainer.model(**batch)
            sample_logits.append(outputs.logits.cpu())
        all_mc_logits.append(torch.cat(sample_logits, dim=0).numpy())

    trainer.model.eval()
    return np.stack(all_mc_logits, axis=0)  # (T, N, 12)


def acquire_random(pool_size: int, budget: int, **kwargs) -> np.ndarray:
    return np.random.choice(pool_size, size=budget, replace=False)


def acquire_entropy(trainer, pool_dataset, data_collator, budget: int, **kwargs) -> np.ndarray:
    """
    Binary entropy per label, averaged across 12 labels per sample.
    H(p) = -(p*log(p) + (1-p)*log(1-p))
    """
    logits = _get_pool_logits(trainer, pool_dataset, data_collator)  # (N, 12)
    probs = sigmoid(logits)
    eps = 1e-10
    entropy = -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))  # (N, 12)
    sample_scores = entropy.mean(axis=1)  # (N,)
    return np.argsort(sample_scores)[::-1][:budget]


def acquire_margin(trainer, pool_dataset, data_collator, budget: int, **kwargs) -> np.ndarray:
    """
    Margin = |p - 0.5| per label, averaged across 12 labels.
    Smaller margin = more uncertain.
    """
    logits = _get_pool_logits(trainer, pool_dataset, data_collator)
    probs = sigmoid(logits)
    margin = np.abs(probs - 0.5)  # (N, 12)
    sample_scores = margin.mean(axis=1)  # (N,)
    return np.argsort(sample_scores)[:budget]  # ascending: smallest margin first


def acquire_bald(trainer, pool_dataset, data_collator, budget: int,
                 num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    BALD for multi-label binary classification via MC-Dropout.
    Per label: I[y;w|x] = H[y|x] - E_w[H[y|x,w]]
    Then averaged across 12 labels.
    """
    mc_logits = _get_pool_logits_mc(trainer, pool_dataset, data_collator, num_mc_samples)  # (T, N, 12)
    mc_probs = sigmoid(mc_logits)  # (T, N, 12)
    eps = 1e-10

    # Predictive entropy from mean prediction
    mean_probs = mc_probs.mean(axis=0)  # (N, 12)
    pred_entropy = -(mean_probs * np.log(mean_probs + eps) +
                     (1 - mean_probs) * np.log(1 - mean_probs + eps))  # (N, 12)

    # Expected entropy across MC samples
    per_sample_entropy = -(mc_probs * np.log(mc_probs + eps) +
                           (1 - mc_probs) * np.log(1 - mc_probs + eps))  # (T, N, 12)
    expected_entropy = per_sample_entropy.mean(axis=0)  # (N, 12)

    bald_scores = (pred_entropy - expected_entropy).mean(axis=1)  # (N,)
    return np.argsort(bald_scores)[::-1][:budget]


def acquire_variation_ratio(trainer, pool_dataset, data_collator, budget: int,
                            num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    Variation ratio for multi-label binary classification via MC-Dropout.
    Per label: 1 - mode_count / T. Then averaged across 12 labels.
    """
    mc_logits = _get_pool_logits_mc(trainer, pool_dataset, data_collator, num_mc_samples)  # (T, N, 12)
    mc_preds = (sigmoid(mc_logits) > 0.5).astype(int)  # (T, N, 12)
    T = mc_preds.shape[0]
    sum_pos = mc_preds.sum(axis=0)  # (N, 12)
    mode_count = np.maximum(sum_pos, T - sum_pos)  # (N, 12)
    var_ratio = 1.0 - mode_count / T  # (N, 12)
    sample_scores = var_ratio.mean(axis=1)  # (N,)
    return np.argsort(sample_scores)[::-1][:budget]


ACQUISITION_FUNCTIONS = {
    "random": acquire_random,
    "entropy": acquire_entropy,
    "margin": acquire_margin,
    "bald": acquire_bald,
    "variation_ratio": acquire_variation_ratio,
}


# ============================================================
# Model Builder
# ============================================================

def build_model(model_args, training_args, num_labels):
    if training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            config = RnaLmConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=num_labels,
                problem_type="multi_label_classification",
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
                problem_type="multi_label_classification",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
            )
    elif training_args.model_type == 'rna-fm':
        model = RnaFmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=True,
        )
    elif training_args.model_type == 'rnabert':
        model = RnaBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=True,
        )
    elif training_args.model_type == 'rnamsm':
        model = RnaMsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=True,
        )
    elif 'splicebert' in training_args.model_type:
        model = SpliceBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=True,
        )
    elif 'utrbert' in training_args.model_type:
        model = UtrBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=True,
        )
    elif 'utr-lm' in training_args.model_type:
        model = UtrLmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {training_args.model_type}")
    return model


# ============================================================
# Checkpoint helpers (same as all other AL tasks)
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
# Main Active Learning Loop
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
    num_labels = full_train_dataset.num_labels

    total_train_size = len(full_train_dataset)
    print(f"Total training set: {total_train_size}")
    print(f"Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

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
    print(f"  Num labels     : {num_labels}")

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

        print(f"\n{'=' * 70}")
        print(f"AL Round {round_idx + 1}/{num_rounds}")
        print(f"  Labeled pool : {current_labeled_size}/{total_train_size} ({labeled_fraction * 100:.1f}%)")
        print(f"  Unlabeled    : {len(unlabeled_indices)}")
        print(f"  Epochs       : {scaled_epochs} (base={base_epochs}, scaled by 1/{labeled_fraction:.2f})")
        print(f"{'=' * 70}")

        labeled_subset = Subset(full_train_dataset, labeled_indices)

        # Build fresh model
        print(f"  Loading fresh model from {model_args.model_name_or_path}...")
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

        # ---- Evaluate ----
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

        print(f"  Val  mean_auc: {val_results.get('eval_mean_auc', 'N/A')}")
        print(f"  Test mean_auc: {test_results.get('eval_mean_auc', 'N/A')}")
        print(f"  Test mean_mcc: {test_results.get('eval_mean_mcc', 'N/A')}")

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

        # ---- Acquisition (if not the last round) ----
        if round_idx < num_rounds - 1:
            next_size = round_sizes[round_idx + 1]
            budget = next_size - current_labeled_size

            if budget <= 0 or len(unlabeled_indices) == 0:
                print("  No more samples to acquire. Stopping.")
                save_al_checkpoint(training_args.output_dir, {
                    "last_completed_round": round_idx,
                    "labeled_indices": labeled_indices,
                    "unlabeled_indices": unlabeled_indices,
                    "al_results": al_results,
                })
                break

            budget = min(budget, len(unlabeled_indices))
            print(f"  Acquiring {budget} samples via '{al_args.al_strategy}'...")

            pool_subset = Subset(full_train_dataset, unlabeled_indices)
            acquire_fn = ACQUISITION_FUNCTIONS[al_args.al_strategy]

            if al_args.al_strategy == "random":
                selected_pool_indices = acquire_fn(
                    pool_size=len(unlabeled_indices), budget=budget,
                )
            elif al_args.al_strategy in ("bald", "variation_ratio"):
                selected_pool_indices = acquire_fn(
                    trainer=trainer,
                    pool_dataset=pool_subset,
                    data_collator=data_collator,
                    budget=budget,
                    num_mc_samples=al_args.al_num_mc_samples,
                )
            else:
                selected_pool_indices = acquire_fn(
                    trainer=trainer,
                    pool_dataset=pool_subset,
                    data_collator=data_collator,
                    budget=budget,
                )

            newly_selected = [unlabeled_indices[i] for i in selected_pool_indices]
            labeled_indices = sorted(labeled_indices + newly_selected)
            unlabeled_indices = sorted(set(unlabeled_indices) - set(newly_selected))
            print(f"  New labeled pool size: {len(labeled_indices)}")

        # ---- Checkpoint after round fully completes ----
        save_al_checkpoint(training_args.output_dir, {
            "last_completed_round": round_idx,
            "labeled_indices": labeled_indices,
            "unlabeled_indices": unlabeled_indices,
            "al_results": al_results,
        })

        # Free GPU memory
        del model, trainer
        torch.cuda.empty_cache()

    # ---- Save aggregate results ----
    aggregate_path = os.path.join(training_args.output_dir, "al_aggregate_results.json")
    aggregate_summary = []
    for r in al_results:
        summary = {k: v for k, v in r.items() if k != "labeled_indices"}
        aggregate_summary.append(summary)
    with open(aggregate_path, "w") as f:
        json.dump(aggregate_summary, f, indent=4)

    print(f"\n{'=' * 70}")
    print("Active Learning Complete — Summary")
    print(f"{'=' * 70}")
    for r in aggregate_summary:
        test_auc = r["test_results"].get("eval_mean_auc", "N/A")
        test_mcc = r["test_results"].get("eval_mean_mcc", "N/A")
        print(f"  Round {r['round']}: frac={r['labeled_fraction']:.2f}, "
              f"size={r['labeled_size']}, epochs={r['scaled_epochs']}, "
              f"mean_auc={test_auc}, mean_mcc={test_mcc}")
    print(f"\nResults: {aggregate_path}")


if __name__ == "__main__":
    train()