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
from torch.utils.data import Dataset, Subset

import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from model.rnalm.modeling_rnalm import RnaLmForNucleotideLevel
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.modeling_rnafm import RnaFmForNucleotideLevel
from model.rnabert.modeling_rnabert import RnaBertForNucleotideLevel
from model.rnamsm.modeling_rnamsm import RnaMsmForNucleotideLevel
from model.splicebert.modeling_splicebert import SpliceBertForNucleotideLevel
from model.utrbert.modeling_utrbert import UtrBertForNucleotideLevel
from model.utrlm.modeling_utrlm import UtrLmForNucleotideLevel
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
    save_strategy: str = field(default="steps")
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
    report_to: str = field(default="tensorboard")
    metric_for_best_model: str = field(default="avg topk acc")
    stage: str = field(default='0')
    model_type: str = field(default='rna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)
    train_fraction: float = field(default=1.0, metadata={"help": "Fraction of training data to use (0.0-1.0)"})


@dataclass
class ActiveLearningArguments:
    """Arguments specific to the active learning loop."""
    al_strategy: str = field(
        default="entropy",
        metadata={"help": "Active learning acquisition strategy: random, entropy, margin, bald, variation_ratio"}
    )
    al_initial_fraction: float = field(
        default=0.1,
        metadata={"help": "Fraction of training data to use as the initial labeled pool."}
    )
    al_target_fraction: float = field(
        default=0.5,
        metadata={"help": "Fraction of training data to reach by the final AL round."}
    )
    al_step_fraction: float = field(
        default=0.1,
        metadata={"help": "Fraction of total training data to acquire per AL round."}
    )
    al_num_mc_samples: int = field(
        default=10,
        metadata={"help": "Number of MC dropout forward passes for BALD / variation_ratio."}
    )
    al_epochs_per_round: int = field(
        default=30,
        metadata={"help": "Number of training epochs per AL round."}
    )
    al_retrain_from_scratch: bool = field(
        default=True,
        metadata={"help": "If True, re-initialize model weights each AL round; else warm-start."}
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
# Dataset
# ============================================================

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer,
                 kmer: int = -1):
        super(SupervisedDataset, self).__init__()

        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 2:
            texts = [d[0].upper().replace("U", "T") for d in data]
            labels = np.array([list(map(float, d[1])) for d in data]).astype(np.float32)
        else:
            raise ValueError("Data format not supported.")

        seq_length = len(texts[0])
        if kmer != -1:
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
        self.weight_mask = torch.ones((self.input_ids.shape[0], seq_length + 2))
        if 'mer' in args.token_type:
            for i in range(1, kmer - 1):
                self.weight_mask[:, i + 1] = self.weight_mask[:, -i - 2] = 1 / (i + 1)
            self.weight_mask[:, kmer:-kmer] = 1 / kmer

        self.post_token_length = torch.zeros(self.attention_mask.shape)
        if args.token_type == 'bpe' or args.token_type == 'non-overlap':
            self.post_token_length = bpe_position(texts, self.attention_mask, tokenizer)

        self.num_labels = 3

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        targets = torch.tensor(self.labels[i, :], dtype=torch.float32)
        return dict(
            input_ids=self.input_ids[i],
            labels=targets,
            attention_mask=self.attention_mask[i],
            weight_mask=self.weight_mask[i],
            post_token_length=self.post_token_length[i],
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask, weight_mask, post_token_length = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "attention_mask", "weight_mask", "post_token_length")
        )
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            weight_mask=weight_mask,
            post_token_length=post_token_length,
        )


# ============================================================
# Metrics
# ============================================================

def top_k_accuracy_multidimensional(scores, true_labels, class_index):
    class_scores = scores[:, :, class_index].flatten()
    true_labels_flat = true_labels.flatten()
    one_hot_labels = np.zeros((true_labels_flat.size, 3))
    one_hot_labels[np.arange(true_labels_flat.size), true_labels_flat.astype(int)] = 1
    class_true_labels = one_hot_labels[:, class_index].flatten()
    k = int(np.sum(class_true_labels))
    if k == 0:
        raise ValueError("No positive instances for the specified class.")
    top_k_indices = np.argsort(class_scores)[::-1][:k]
    true_positives = np.sum(class_true_labels[top_k_indices])
    return true_positives / k


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze()
    logits = logits.squeeze()
    metrics = [top_k_accuracy_multidimensional(logits, labels, i) for i in range(logits.shape[-1])]
    return {
        "acceptor topk acc": metrics[1],
        "donor topk acc": metrics[2],
        "avg topk acc": np.mean(metrics[1:]),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)


# ============================================================
# Active Learning Acquisition Functions
# ============================================================

def _enable_mc_dropout(model):
    """Turn on dropout layers at inference time for MC-Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def _get_pool_logits(trainer, pool_dataset, data_collator):
    """Run a single forward pass over the unlabeled pool and return logits (N, seq_len, C)."""
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
    return torch.cat(all_logits, dim=0)  # (N, seq_len, C)


@torch.no_grad()
def _get_pool_logits_mc(trainer, pool_dataset, data_collator, num_mc_samples: int):
    """
    Run multiple MC-Dropout forward passes over the unlabeled pool.
    Returns tensor of shape (num_mc_samples, N, seq_len, C).
    """
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
        all_mc_logits.append(torch.cat(sample_logits, dim=0))

    trainer.model.eval()  # restore full eval mode
    return torch.stack(all_mc_logits, dim=0)  # (T, N, seq_len, C)


def acquire_random(pool_size: int, budget: int, **kwargs) -> np.ndarray:
    """Random acquisition baseline."""
    return np.random.choice(pool_size, size=budget, replace=False)


def acquire_entropy(trainer, pool_dataset, data_collator, budget: int, **kwargs) -> np.ndarray:
    """
    Select samples with highest predictive entropy, averaged over sequence positions.
    Entropy = -sum(p * log(p)) per position, then averaged over positions.
    """
    logits = _get_pool_logits(trainer, pool_dataset, data_collator)  # (N, seq, C)
    probs = torch.softmax(logits, dim=-1).numpy()
    eps = 1e-10
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)  # (N, seq)
    sample_scores = entropy.mean(axis=1)  # (N,)
    return np.argsort(sample_scores)[::-1][:budget]


def acquire_margin(trainer, pool_dataset, data_collator, budget: int, **kwargs) -> np.ndarray:
    """
    Select samples with the smallest margin between top-2 class probabilities
    (averaged over sequence positions). Smaller margin = more uncertain.
    """
    logits = _get_pool_logits(trainer, pool_dataset, data_collator)
    probs = torch.softmax(logits, dim=-1).numpy()  # (N, seq, C)
    sorted_probs = np.sort(probs, axis=-1)  # ascending
    margin = sorted_probs[:, :, -1] - sorted_probs[:, :, -2]  # (N, seq)
    sample_scores = margin.mean(axis=1)  # (N,) — lower = more uncertain
    return np.argsort(sample_scores)[:budget]  # ascending: pick smallest margins


def acquire_bald(trainer, pool_dataset, data_collator, budget: int,
                 num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    Bayesian Active Learning by Disagreement (BALD).
    I[y; w | x] = H[y|x] - E_w[H[y|x,w]]
    Uses MC-Dropout to approximate the posterior.
    """
    mc_logits = _get_pool_logits_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    mc_probs = torch.softmax(mc_logits, dim=-1).numpy()  # (T, N, seq, C)

    eps = 1e-10
    mean_probs = mc_probs.mean(axis=0)  # (N, seq, C)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)  # (N, seq)

    per_sample_entropy = -np.sum(mc_probs * np.log(mc_probs + eps), axis=-1)  # (T, N, seq)
    expected_entropy = per_sample_entropy.mean(axis=0)  # (N, seq)

    bald_scores = (predictive_entropy - expected_entropy).mean(axis=1)  # (N,)
    return np.argsort(bald_scores)[::-1][:budget]


def acquire_variation_ratio(trainer, pool_dataset, data_collator, budget: int,
                            num_mc_samples: int = 10, **kwargs) -> np.ndarray:
    """
    Variation ratio: 1 - (count of mode class) / T, averaged over positions.
    Higher variation ratio = more disagreement among MC samples.
    """
    mc_logits = _get_pool_logits_mc(trainer, pool_dataset, data_collator, num_mc_samples)
    mc_preds = mc_logits.argmax(dim=-1).numpy()  # (T, N, seq)
    T = mc_preds.shape[0]

    from scipy import stats
    mode_counts = stats.mode(mc_preds, axis=0, keepdims=False).count  # (N, seq)
    variation_ratio = 1.0 - mode_counts / T  # (N, seq)
    sample_scores = variation_ratio.mean(axis=1)  # (N,)
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

def build_model(model_args, training_args, num_labels, tokenizer):
    """Instantiate a fresh model from the pretrained checkpoint."""
    if training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            config = RnaLmConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=num_labels,
                problem_type="single_label_classification",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
            )
            model = RnaLmForNucleotideLevel(config, tokenizer=tokenizer)
        else:
            model = RnaLmForNucleotideLevel.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=num_labels,
                trust_remote_code=True,
                problem_type="single_label_classification",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
            )
    elif training_args.model_type == 'rna-fm':
        model = RnaFmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            tokenizer=tokenizer,
        )
    elif training_args.model_type == 'rnabert':
        model = RnaBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )
    elif training_args.model_type == 'rnamsm':
        model = RnaMsmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )
    elif 'splicebert' in training_args.model_type:
        model = SpliceBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )
    elif 'utrbert' in training_args.model_type:
        model = UtrBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )
    elif 'utr-lm' in training_args.model_type:
        model = UtrLmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
            problem_type="single_label_classification",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unknown model_type: {training_args.model_type}")
    return model


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
    print(f"Total training set size: {total_train_size}")
    print(f"Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # ---- Determine AL schedule ----
    # Build round sizes from fractions directly to avoid integer rounding drift
    fractions = []
    frac = al_args.al_initial_fraction
    while frac <= al_args.al_target_fraction + 1e-9:
        fractions.append(frac)
        frac = round(frac + al_args.al_step_fraction, 10)

    round_sizes = [max(1, int(total_train_size * fr)) for fr in fractions]
    num_rounds = len(round_sizes)

    print(f"Active Learning config:")
    print(f"  Strategy       : {al_args.al_strategy}")
    print(f"  Initial frac   : {al_args.al_initial_fraction*100:.0f}% ({round_sizes[0]} samples)")
    print(f"  Target frac    : {al_args.al_target_fraction*100:.0f}% ({round_sizes[-1]} samples)")
    print(f"  Step frac      : {al_args.al_step_fraction*100:.0f}%")
    print(f"  Fractions      : {[f'{fr:.1%}' for fr in fractions]}")
    print(f"  Rounds         : {num_rounds} -> sizes {round_sizes}")
    print(f"  Retrain from scratch each round: {al_args.al_retrain_from_scratch}")

    # ---- Initial labeled pool (random) ----
    all_indices = np.arange(total_train_size)
    np.random.shuffle(all_indices)
    labeled_indices = sorted(all_indices[:round_sizes[0]].tolist())
    unlabeled_indices = sorted(all_indices[round_sizes[0]:].tolist())

    # ---- Results collector ----
    al_results = []

    # ---- Active Learning Loop ----
    for round_idx, target_size in enumerate(round_sizes):
        current_labeled_size = len(labeled_indices)
        labeled_fraction = current_labeled_size / total_train_size
        print(f"\n{'='*60}")
        print(f"AL Round {round_idx + 1}/{num_rounds}")
        print(f"  Labeled pool size: {current_labeled_size}/{total_train_size} ({labeled_fraction*100:.1f}%)")
        print(f"  Unlabeled pool   : {len(unlabeled_indices)}")
        print(f"{'='*60}")

        # Create subset for current labeled pool
        labeled_subset = Subset(full_train_dataset, labeled_indices)

        # Build (or rebuild) model
        model = build_model(model_args, training_args, num_labels, tokenizer)

        # Scale epochs inversely with labeled fraction to maintain ~constant total training steps
        scaled_epochs = int(round(al_args.al_epochs_per_round / labeled_fraction))
        print(f"  Epoch scaling: base={al_args.al_epochs_per_round}, "
              f"fraction={labeled_fraction:.2f}, scaled={scaled_epochs}")

        # Per-round output directory
        round_output_dir = os.path.join(
            training_args.output_dir,
            f"round_{round_idx + 1}_frac_{labeled_fraction:.2f}"
        )

        # Clone training args for this round
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
        )
        trainer.train()

        # ---- Evaluate on val + test ----
        val_results = trainer.evaluate(eval_dataset=val_dataset)
        test_results = trainer.evaluate(eval_dataset=test_dataset)

        round_record = {
            "round": round_idx + 1,
            "labeled_size": current_labeled_size,
            "labeled_fraction": labeled_fraction,
            "strategy": al_args.al_strategy,
            "val_results": val_results,
            "test_results": test_results,
            "labeled_indices": labeled_indices.copy(),
        }
        al_results.append(round_record)

        print(f"  Val  avg topk acc: {val_results.get('eval_avg topk acc', 'N/A')}")
        print(f"  Test avg topk acc: {test_results.get('eval_avg topk acc', 'N/A')}")

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
                "strategy": al_args.al_strategy,
                "labeled_indices": labeled_indices,
            }, f, indent=4)

        if training_args.save_model:
            trainer.save_state()

        # ---- Acquisition step (if not the last round) ----
        if round_idx < num_rounds - 1:
            next_size = round_sizes[round_idx + 1]
            budget = next_size - current_labeled_size

            if budget <= 0 or len(unlabeled_indices) == 0:
                print("  No more samples to acquire. Stopping AL loop.")
                break

            budget = min(budget, len(unlabeled_indices))
            print(f"  Acquiring {budget} new samples using '{al_args.al_strategy}' strategy...")

            # Build pool subset from unlabeled indices
            pool_subset = Subset(full_train_dataset, unlabeled_indices)

            acquire_fn = ACQUISITION_FUNCTIONS[al_args.al_strategy]

            if al_args.al_strategy == "random":
                selected_pool_indices = acquire_fn(pool_size=len(unlabeled_indices), budget=budget)
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

            # Map pool-relative indices back to global indices
            newly_selected = [unlabeled_indices[i] for i in selected_pool_indices]
            labeled_indices = sorted(labeled_indices + newly_selected)
            unlabeled_indices = sorted(set(unlabeled_indices) - set(newly_selected))

            print(f"  New labeled pool size: {len(labeled_indices)}")

        # Free GPU memory
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

    print(f"\n{'='*60}")
    print("Active Learning complete. Summary:")
    print(f"{'='*60}")
    for r in aggregate_summary:
        test_acc = r["test_results"].get("eval_avg topk acc", "N/A")
        print(f"  Round {r['round']}: frac={r['labeled_fraction']:.2f}, "
              f"size={r['labeled_size']}, test_avg_topk_acc={test_acc}")
    print(f"\nAggregate results saved to: {aggregate_path}")


if __name__ == "__main__":
    train()