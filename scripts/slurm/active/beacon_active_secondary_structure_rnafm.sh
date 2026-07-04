#!/usr/bin/env bash
# beacon_active_secondary_structure_rnafm.sh
# RNA-FM backbone replication of the SSP active-learning headline.
# Generalization test (M4): does the graph_motif > random advantage hold on a
# SECOND pretrained encoder (RNA-FM) instead of BEACON-B?
#
# RNA-FM needs NO code changes: get_extractor already dispatches model_type='rna-fm'
# (active/structure/lm.py), checkpoint lives at checkpoint/opensource/rna-fm/, and
# SSCNNPredictor auto-adapts to hidden_size=640. Verified by a forward-pass smoke test.
set -euo pipefail

# --- Global Paths ---
CODE_ROOT=/storage/ukp/work/cai_e/RNABenchmark
DATA_ROOT=${CODE_ROOT}/data
MODEL_ROOT=${CODE_ROOT}/checkpoint
OUT_ROOT=${CODE_ROOT}/outputs_slurm/ft/rna-all-active-learning
LOG_DIR=${CODE_ROOT}/logs/beacon_al
mkdir -p "$LOG_DIR"

NPROC_PER_NODE=1
NUM_GPUS=1

##########################
# Seeds — 4-seed set (matches the schedule-sweep seeds)
##########################
SEEDS=(42 666 1234 7)

##########################
# Model definition — RNA-FM
# fields: folder_name  model_type  token_type  max_length  category
#   checkpoint path = MODEL_ROOT/category/folder_name = checkpoint/opensource/rna-fm/
##########################
MODELS=(
  "rna-fm    rna-fm   single  1024    opensource"
)

##########################
# AL fractions (default headline schedule: 10% -> 50%)
##########################
AL_INITIAL_FRACTION=0.1
AL_TARGET_FRACTION=0.5
AL_STEP_FRACTION=0.1
AL_EPOCHS_PER_ROUND=100
AL_NUM_MC_SAMPLES=10

CONFIGS=()

##########################
# Explicit (config, seed) tuples to submit.
# Format: "strategy|aggregation|alpha|nuc_topk_frac|stem_topk|min_stem_len|seed"
#   (7 fields => default 10->50% schedule; add init|target|step for a custom schedule)
# The two key configs from the BEACON-B headline: random baseline + graph_motif winner.
##########################
RERUNS=(
  # Random jobs are running preemption-free on bob/a100 — do NOT resubmit them.
  # Only the graph_motif jobs are moved off a180 (heavy preemption) onto a100.
  # "random|mean|1.0|0.2|3|2|42"
  # "random|mean|1.0|0.2|3|2|666"
  # "random|mean|1.0|0.2|3|2|1234"
  # "random|mean|1.0|0.2|3|2|7"
  "entropy|graph_motif|1.0|0.2|3|2|42"
  "entropy|graph_motif|1.0|0.2|3|2|666"
  "entropy|graph_motif|1.0|0.2|3|2|1234"
  "entropy|graph_motif|1.0|0.2|3|2|7"
)

# Default lr=3e-5 (matches BEACON-B headline) — no override.
OVERRIDE_LR=""
LR_TAG=""

echo "============================================================"
if [ "${#RERUNS[@]}" -gt 0 ]; then
  echo "RERUN MODE — submitting ${#RERUNS[@]} explicit (config, seed) tuples"
else
  echo "SWEEP MODE — Total experiment configs: ${#CONFIGS[@]}"
  echo "             Total seeds: ${#SEEDS[@]}"
  echo "             Total jobs: $(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))"
fi
echo "============================================================"

# Build a unified list of (strategy|aggregation|alpha|nuc_topk|stem_topk|min_stem|seed[|init|target|step])
SUBMIT_LIST=()
if [ "${#RERUNS[@]}" -gt 0 ]; then
  SUBMIT_LIST=("${RERUNS[@]}")
else
  for s in "${SEEDS[@]}"; do
    for c in "${CONFIGS[@]}"; do
      SUBMIT_LIST+=("${c}|${s}")
    done
  done
fi

for entry in "${MODELS[@]}"; do
  read -r FOLDER_NAME MODEL_TYPE TOKEN_TYPE MAX_LENGTH CATEGORY <<< "$entry"

  PRETRAINED_PATH="${MODEL_ROOT}/${CATEGORY}/${FOLDER_NAME}/"
  MODEL_NAME="${FOLDER_NAME}"

  for item in "${SUBMIT_LIST[@]}"; do
    IFS='|' read -r AL_STRATEGY AL_AGG AL_ALPHA AL_NUC_TOPK AL_STEM_TOPK AL_MIN_STEM SEED AL_INIT AL_TGT AL_STEP <<< "$item"

      # --- Build a unique, readable suffix for output dir and job name ---
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

      # Per-job AL fraction schedule; fall back to globals if omitted.
      FRAC_SUFFIX=""
      if [ -z "${AL_INIT}" ]; then
        AL_INIT="${AL_INITIAL_FRACTION}"
        AL_TGT="${AL_TARGET_FRACTION}"
        AL_STEP="${AL_STEP_FRACTION}"
      else
        INIT_PCT=$(python3 -c "print(round(${AL_INIT}*100))")
        TGT_PCT=$(python3 -c "print(round(${AL_TGT}*100))")
        FRAC_SUFFIX="_frac${INIT_PCT}to${TGT_PCT}"
      fi

      LR_SUFFIX=""
      if [ -n "${LR_TAG:-}" ]; then
        LR_SUFFIX="_${LR_TAG}"
      fi
      EXPERIMENT_TAG="${AL_STRATEGY}_${AGG_SUFFIX}${FRAC_SUFFIX}${LR_SUFFIX}_seed${SEED}"
      JOB_NAME="al_ss_${MODEL_NAME}_bpRNA_${EXPERIMENT_TAG}"

      # All on a100 now — a180 was getting preempted heavily, a100 (bob) is stable
      GPU_TYPE="a100"

      JOB_LR="${OVERRIDE_LR:-3e-5}"

      echo "Submitting: ${EXPERIMENT_TAG}"
      echo "  MODEL      : ${MODEL_NAME}"
      echo "  STRATEGY   : ${AL_STRATEGY}"
      echo "  AGGREGATION: ${AL_AGG} (alpha=${AL_ALPHA}, nuc_topk=${AL_NUC_TOPK}, stem_topk=${AL_STEM_TOPK}, min_stem=${AL_MIN_STEM})"
      echo "  SEED       : ${SEED}"
      echo "  LR         : ${JOB_LR}"
      echo "  GPU        : ${GPU_TYPE}"

      sbatch <<EOT
#!/bin/bash
#SBATCH -p yolo
#SBATCH -q yolo
#SBATCH --job-name=${JOB_NAME}
#SBATCH --error=${LOG_DIR}/${JOB_NAME}.err
#SBATCH --output=${LOG_DIR}/${JOB_NAME}.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPUS}
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
echo "SEED            = \$SEED"

cd "\$CODE_ROOT"
which python
python -V

export WANDB_DISABLED=true

MASTER_PORT=\$(shuf -i 10000-45000 -n 1)
EXEC_PREFIX="torchrun --nproc_per_node=\$NPROC_PER_NODE --master_port=\$MASTER_PORT"

# ---------------------------------------------------------------
# Secondary Structure Prediction — Active Learning (RNA-FM backbone)
# ---------------------------------------------------------------
task='Secondary_structure_prediction'
batch_size=1
lr=${JOB_LR}
DATA_PATH=\${DATA_ROOT}/\${task}/bpRNA

# Output dir keyed by MODEL_NAME (=rna-fm) so it never collides with BEACON-B
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}/${EXPERIMENT_TAG}

checkpoint_and_requeue() {
    echo "=== Wall time approaching — sending SIGTERM to training process ==="
    kill -TERM \$TRAIN_PID 2>/dev/null || true
    wait \$TRAIN_PID 2>/dev/null || true
    echo "=== Requeueing job \$SLURM_JOB_ID at \$(date) ==="
    scontrol requeue \$SLURM_JOB_ID
    exit 0
}
trap checkpoint_and_requeue USR1

echo "--- [\$MODEL_NAME] Running \$task Active Learning (\$AL_STRATEGY + \$AL_AGG, seed=\$SEED) at \$(date) ---"
echo "--- Output: \$OUTPUT_PATH ---"

\$EXEC_PREFIX active/train_secondary_structure_config.py \\
  --model_name_or_path \${PRETRAINED_PATH} \\
  --data_path \${DATA_PATH} \\
  --run_name \${MODEL_NAME}_\${task}_AL_${EXPERIMENT_TAG} \\
  --output_dir \${OUTPUT_PATH} \\
  --model_max_length \${MAX_LENGTH} \\
  --per_device_train_batch_size \${batch_size} \\
  --per_device_eval_batch_size 1 \\
  --gradient_accumulation_steps 8 \\
  --lr \${lr} \\
  --num_epochs ${AL_EPOCHS_PER_ROUND} \\
  --patience 60 \\
  --num_workers 1 \\
  --token_type \${TOKEN_TYPE} \\
  --model_type \${MODEL_TYPE} \\
  --seed \${SEED} \\
  --mode bprna \\
  --al_strategy \${AL_STRATEGY} \\
  --al_initial_fraction ${AL_INIT} \\
  --al_target_fraction ${AL_TGT} \\
  --al_step_fraction ${AL_STEP} \\
  --al_epochs_per_round ${AL_EPOCHS_PER_ROUND} \\
  --al_num_mc_samples ${AL_NUM_MC_SAMPLES} \\
  --al_aggregation \${AL_AGG} \\
  --al_aggregation_alpha \${AL_ALPHA} \\
  --al_nuc_topk_frac \${AL_NUC_TOPK} \\
  --al_stem_topk \${AL_STEM_TOPK} \\
  --al_min_stem_len \${AL_MIN_STEM} &
TRAIN_PID=\$!
wait \$TRAIN_PID

echo "=== SS Active Learning for \$MODEL_NAME (${EXPERIMENT_TAG}) completed at \$(date) ==="
EOT

    echo "  -> Submitted SLURM job: ${JOB_NAME}"
    echo ""
  done
done

echo "============================================================"
echo "All RNA-FM SS Active Learning jobs submitted: ${#SUBMIT_LIST[@]} total"
echo "============================================================"
