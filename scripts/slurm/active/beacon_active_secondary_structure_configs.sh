#!/usr/bin/env bash
# submit_beacon_ss_active_learning.sh
# Submit active learning experiments for BEACON on RNA Secondary Structure
# with structure-aware uncertainty aggregation sweep.
set -euo pipefail

# --- Global Paths ---
CODE_ROOT=/storage/ukp/work/cai_e/RNABenchmark
DATA_ROOT=${CODE_ROOT}/data
MODEL_ROOT=${CODE_ROOT}/checkpoint
OUT_ROOT=${CODE_ROOT}/outputs_slurm/ft/rna-all-active-learning
LOG_DIR=${CODE_ROOT}/logs/beacon_al
mkdir -p "$LOG_DIR"

SEED=666
NPROC_PER_NODE=1
NUM_GPUS=1

##########################
# Model definition
##########################
MODELS=(
  "BEACON-B    rnalm   single  1026    baseline"
)

##########################
# AL fractions (shared across all experiments)
##########################
AL_INITIAL_FRACTION=0.1
AL_TARGET_FRACTION=0.5
AL_STEP_FRACTION=0.1
AL_EPOCHS_PER_ROUND=50
AL_NUM_MC_SAMPLES=10

##########################
# Experiment configurations
#
# Format: "strategy|aggregation|alpha|nuc_topk_frac|stem_topk|min_stem_len"
#
# Design rationale:
#   - random+mean: baseline (no uncertainty used, aggregation irrelevant)
#   - {entropy,margin,bald} × mean: existing baselines (already run, included for completeness)
#   - {entropy,margin,bald} × {pos_reweight, nuc_marginal, graph_motif, pos_nuc}:
#       main ablation — isolate aggregation effect per strategy
#   - pos_reweight with alpha=2.0: sharper focus on high-confidence pairs
#   - nuc_marginal with topk_frac=0.1: more selective (top 10% nucleotides)
#   - pos_nuc with alpha=2.0: combo with sharper reweighting
##########################

CONFIGS=()

# --- Random baseline (aggregation is irrelevant, just one run) ---
# CONFIGS+=("random|mean|1.0|0.2|3|2")

# --- Main ablation: strategy × aggregation (default hyperparams) ---
for strategy in entropy margin bald; do
  # # Baseline aggregation (already run — skip if checkpoints exist)
  # CONFIGS+=("${strategy}|mean|1.0|0.2|3|2")

  # # Method 1: positive reweighting (alpha=1.0)
  # CONFIGS+=("${strategy}|pos_reweight|1.0|0.2|3|2")

  # # Method 2: nucleotide marginal (top 20%)
  # CONFIGS+=("${strategy}|nuc_marginal|1.0|0.2|3|2")

  # # Method 3: graph motif (top 3 stems, min len 2)
  # CONFIGS+=("${strategy}|graph_motif|1.0|0.2|3|2")

  # Combo: pos_reweight + nuc_marginal
  # CONFIGS+=("${strategy}|pos_nuc|1.0|0.2|3|2")
done

# --- Hyperparameter variants for the most promising methods ---
for strategy in entropy bald; do
  # pos_reweight with sharper alpha
  CONFIGS+=("${strategy}|pos_reweight|2.0|0.2|3|2")

  # nuc_marginal with more selective top-k
  CONFIGS+=("${strategy}|nuc_marginal|1.0|0.1|3|2")

  # pos_nuc with sharper alpha
  CONFIGS+=("${strategy}|pos_nuc|2.0|0.2|3|2")

  # graph_motif with longer minimum stems (more conservative motif detection)
  CONFIGS+=("${strategy}|graph_motif|1.0|0.2|5|3")
done

echo "============================================================"
echo "Total experiment configs: ${#CONFIGS[@]}"
echo "============================================================"

for entry in "${MODELS[@]}"; do
  read -r FOLDER_NAME MODEL_TYPE TOKEN_TYPE MAX_LENGTH CATEGORY <<< "$entry"

  PRETRAINED_PATH="${MODEL_ROOT}/${CATEGORY}/${FOLDER_NAME}/"
  MODEL_NAME="${FOLDER_NAME}"

  for config in "${CONFIGS[@]}"; do
    IFS='|' read -r AL_STRATEGY AL_AGG AL_ALPHA AL_NUC_TOPK AL_STEM_TOPK AL_MIN_STEM <<< "$config"

    # --- Build a unique, readable suffix for output dir and job name ---
    # Examples:
    #   entropy_mean
    #   entropy_pos_reweight_a1.0
    #   bald_nuc_marginal_topk0.1
    #   bald_graph_motif_stem5_minlen3
    #   entropy_pos_nuc_a2.0
    AGG_SUFFIX="${AL_AGG}"
    if [[ "$AL_AGG" == "pos_reweight" || "$AL_AGG" == "pos_nuc" ]]; then
      AGG_SUFFIX="${AGG_SUFFIX}_a${AL_ALPHA}"
    fi
    if [[ "$AL_AGG" == "nuc_marginal" || "$AL_AGG" == "pos_nuc" ]]; then
      AGG_SUFFIX="${AGG_SUFFIX}_topk${AL_NUC_TOPK}"
    fi
    if [[ "$AL_AGG" == "graph_motif" ]]; then
      AGG_SUFFIX="${AGG_SUFFIX}_stem${AL_STEM_TOPK}_minlen${AL_MIN_STEM}"
    fi

    EXPERIMENT_TAG="${AL_STRATEGY}_${AGG_SUFFIX}"
    JOB_NAME="al_ss_${MODEL_NAME}_bpRNA_${EXPERIMENT_TAG}"

    echo "Submitting: ${EXPERIMENT_TAG}"
    echo "  MODEL      : ${MODEL_NAME}"
    echo "  STRATEGY   : ${AL_STRATEGY}"
    echo "  AGGREGATION: ${AL_AGG} (alpha=${AL_ALPHA}, nuc_topk=${AL_NUC_TOPK}, stem_topk=${AL_STEM_TOPK}, min_stem=${AL_MIN_STEM})"

    sbatch <<EOT
#!/bin/bash
#SBATCH -p yolo
#SBATCH -q yolo
#SBATCH --job-name=${JOB_NAME}
#SBATCH --error=${LOG_DIR}/${JOB_NAME}.err
#SBATCH --output=${LOG_DIR}/${JOB_NAME}.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:${NUM_GPUS}
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fengyu.cai@tu-darmstadt.de

set -eo pipefail

echo "=== SS Active Learning started on \$(hostname) at \$(date) ==="
echo "=== Experiment: ${EXPERIMENT_TAG} ==="

source /storage/ukp/work/cai_e/anaconda3/bin/activate beacon
export LD_LIBRARY_PATH=/storage/ukp/work/cai_e/anaconda3/envs/beacon/lib/:\$LD_LIBRARY_PATH

CODE_ROOT="${CODE_ROOT}"
DATA_ROOT="${DATA_ROOT}"
MODEL_ROOT="${MODEL_ROOT}"
OUT_ROOT="${OUT_ROOT}"
MODEL_NAME="${MODEL_NAME}"
MODEL_TYPE="${MODEL_TYPE}"
TOKEN_TYPE="${TOKEN_TYPE}"
MAX_LENGTH="${MAX_LENGTH}"
PRETRAINED_PATH="${PRETRAINED_PATH}"
SEED="${SEED}"
NPROC_PER_NODE="${NPROC_PER_NODE}"
AL_STRATEGY="${AL_STRATEGY}"
AL_AGG="${AL_AGG}"
AL_ALPHA="${AL_ALPHA}"
AL_NUC_TOPK="${AL_NUC_TOPK}"
AL_STEM_TOPK="${AL_STEM_TOPK}"
AL_MIN_STEM="${AL_MIN_STEM}"

echo "CODE_ROOT       = \$CODE_ROOT"
echo "MODEL_NAME      = \$MODEL_NAME"
echo "PRETRAINED_PATH = \$PRETRAINED_PATH"
echo "AL_STRATEGY     = \$AL_STRATEGY"
echo "AL_AGGREGATION  = \$AL_AGG"
echo "AL_ALPHA        = \$AL_ALPHA"

cd "\$CODE_ROOT"
which python
python -V

export WANDB_DISABLED=true

MASTER_PORT=\$(shuf -i 10000-45000 -n 1)
EXEC_PREFIX="torchrun --nproc_per_node=\$NPROC_PER_NODE --master_port=\$MASTER_PORT"

# ---------------------------------------------------------------
# Secondary Structure Prediction — Active Learning
# ---------------------------------------------------------------
task='Secondary_structure_prediction'
batch_size=2
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}/bpRNA

# Output path includes strategy + aggregation tag for unique directories
# e.g., .../Secondary_structure_prediction/BEACON-B/entropy_pos_reweight_a1.0/
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}/${EXPERIMENT_TAG}

echo "--- [\$MODEL_NAME] Running \$task Active Learning (\$AL_STRATEGY + \$AL_AGG) at \$(date) ---"
echo "--- Output: \$OUTPUT_PATH ---"

\$EXEC_PREFIX active/train_secondary_structure_config.py \\
  --model_name_or_path \${PRETRAINED_PATH} \\
  --data_path \${DATA_PATH} \\
  --run_name \${MODEL_NAME}_\${task}_AL_${EXPERIMENT_TAG} \\
  --output_dir \${OUTPUT_PATH} \\
  --model_max_length \${MAX_LENGTH} \\
  --per_device_train_batch_size \${batch_size} \\
  --per_device_eval_batch_size 1 \\
  --gradient_accumulation_steps 4 \\
  --lr \${lr} \\
  --num_epochs ${AL_EPOCHS_PER_ROUND} \\
  --patience 60 \\
  --num_workers 1 \\
  --token_type \${TOKEN_TYPE} \\
  --model_type \${MODEL_TYPE} \\
  --seed \${SEED} \\
  --mode bprna \\
  --al_strategy \${AL_STRATEGY} \\
  --al_initial_fraction ${AL_INITIAL_FRACTION} \\
  --al_target_fraction ${AL_TARGET_FRACTION} \\
  --al_step_fraction ${AL_STEP_FRACTION} \\
  --al_epochs_per_round ${AL_EPOCHS_PER_ROUND} \\
  --al_num_mc_samples ${AL_NUM_MC_SAMPLES} \\
  --al_aggregation \${AL_AGG} \\
  --al_aggregation_alpha \${AL_ALPHA} \\
  --al_nuc_topk_frac \${AL_NUC_TOPK} \\
  --al_stem_topk \${AL_STEM_TOPK} \\
  --al_min_stem_len \${AL_MIN_STEM}

echo "=== SS Active Learning for \$MODEL_NAME (${EXPERIMENT_TAG}) completed at \$(date) ==="
EOT

    echo "  -> Submitted SLURM job: ${JOB_NAME}"
    echo ""
  done
done

echo "============================================================"
echo "All SS Active Learning jobs submitted: ${#CONFIGS[@]} total"
echo "============================================================"