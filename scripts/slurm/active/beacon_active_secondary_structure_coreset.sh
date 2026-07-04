#!/usr/bin/env bash
# beacon_active_secondary_structure_coreset.sh
# CoreSet (k-center-greedy, Sener & Savarese 2018) AL baseline on BEACON-B, bpRNA, 10%->50%.
#
# This is the published, diversity-based AL baseline (M3). It selects samples to
# maximize coverage of the encoder embedding space — it uses NO uncertainty and NO
# structure-aware aggregation. The point: does entropy_graph_motif's gain survive a
# competent off-the-shelf AL method, or was beating `random` too easy a bar?
#
# Implemented in active/train_secondary_structure_config.py as al_strategy='coreset'
# (acquire_coreset + _get_embeddings). Verified with a forward-pass smoke test.
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
# Seeds — 4-seed set (matches the RNA-FM / schedule-sweep convention)
##########################
SEEDS=(42 666 1234 7)

##########################
# Model definition — BEACON-B (same backbone as the headline)
##########################
MODELS=(
  "BEACON-B    rnalm   single  1026    baseline"
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
# Explicit (config, seed) tuples.
# Format: "strategy|aggregation|alpha|nuc_topk_frac|stem_topk|min_stem_len|seed"
# CoreSet ignores aggregation; we set agg=mean so the tag reads "coreset_mean_seedNN".
##########################
RERUNS=(
  "coreset|mean|1.0|0.2|3|2|42"
  "coreset|mean|1.0|0.2|3|2|666"
  "coreset|mean|1.0|0.2|3|2|1234"
  "coreset|mean|1.0|0.2|3|2|7"
)

# Default lr=3e-5 (matches the BEACON-B headline) — no override.
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

      # CoreSet runs on a6000 (a180 is congested)
      GPU_TYPE="a6000"

      JOB_LR="${OVERRIDE_LR:-3e-5}"

      echo "Submitting: ${EXPERIMENT_TAG}"
      echo "  MODEL      : ${MODEL_NAME}"
      echo "  STRATEGY   : ${AL_STRATEGY}"
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
echo "SEED            = \$SEED"

cd "\$CODE_ROOT"
which python
python -V

export WANDB_DISABLED=true

MASTER_PORT=\$(shuf -i 10000-45000 -n 1)
EXEC_PREFIX="torchrun --nproc_per_node=\$NPROC_PER_NODE --master_port=\$MASTER_PORT"

# ---------------------------------------------------------------
# Secondary Structure Prediction — CoreSet AL baseline (BEACON-B)
# ---------------------------------------------------------------
task='Secondary_structure_prediction'
batch_size=1
lr=${JOB_LR}
DATA_PATH=\${DATA_ROOT}/\${task}/bpRNA

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

echo "--- [\$MODEL_NAME] Running \$task CoreSet AL (seed=\$SEED) at \$(date) ---"
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

echo "=== SS CoreSet AL for \$MODEL_NAME (${EXPERIMENT_TAG}) completed at \$(date) ==="
EOT

    echo "  -> Submitted SLURM job: ${JOB_NAME}"
    echo ""
  done
done

echo "============================================================"
echo "All CoreSet SS Active Learning jobs submitted: ${#SUBMIT_LIST[@]} total"
echo "============================================================"
