#!/usr/bin/env bash
# submit_beacon_active_learning_crispr_on_target.sh
# Submit active learning experiments for BEACON on CRISPR On-Target
# CHANGE: adapted from structural imputation template for sequence-level
#         regression task (CRISPR on-target, evaluated by Spearman correlation)
#         Uses single-sequence input (not sgrna+target pair like off-target)
set -euo pipefail

# --- Global Paths ---
CODE_ROOT=/storage/ukp/work/cai_e/RNABenchmark
DATA_ROOT=${CODE_ROOT}/data
MODEL_ROOT=${CODE_ROOT}/checkpoint
OUT_ROOT=${CODE_ROOT}/outputs_slurm/ft/rna-all-active-learning
# CHANGE: separate log directory for CRISPR on-target AL experiments
LOG_DIR=${CODE_ROOT}/logs/beacon_al_crispr_on_target
mkdir -p "$LOG_DIR"

SEED=666
NPROC_PER_NODE=1
NUM_GPUS=1

##########################
# Define models to evaluate
# Format: "FolderName   ModelType   TokenType   MaxLength   Category"
##########################
MODELS=(
  "BEACON-B    rnalm   single  1026    baseline"
)

##########################
# Active learning strategies to run
##########################
AL_STRATEGIES=("random" "entropy" "margin" "bald")

# AL fractions: start at 10%, grow by 10% each round, up to 50%
AL_INITIAL_FRACTION=0.1
AL_TARGET_FRACTION=0.5
AL_STEP_FRACTION=0.1
# CHANGE: base epochs = 30 (same as original num_train_epochs=30 for CRISPROnTarget)
AL_EPOCHS_PER_ROUND=30
AL_NUM_MC_SAMPLES=10
# CHANGE: patience = 20 (matches original early_stopping_patience=20 for this task)
PATIENCE=20

for entry in "${MODELS[@]}"; do
  read -r FOLDER_NAME MODEL_TYPE TOKEN_TYPE MAX_LENGTH CATEGORY <<< "$entry"

  PRETRAINED_PATH="${MODEL_ROOT}/${CATEGORY}/${FOLDER_NAME}/"
  MODEL_NAME="${FOLDER_NAME}"

  for AL_STRATEGY in "${AL_STRATEGIES[@]}"; do

    # CHANGE: job name reflects CRISPR on-target task
    JOB_NAME="al_${MODEL_NAME}_CRISPROnTarget_${AL_STRATEGY}"

    echo "Submitting Active Learning job:"
    echo "  MODEL_NAME   : ${MODEL_NAME}"
    echo "  TASK         : CRISPROnTarget"
    echo "  AL_STRATEGY  : ${AL_STRATEGY}"
    echo "  FRACTIONS    : ${AL_INITIAL_FRACTION} -> ${AL_TARGET_FRACTION} (step ${AL_STEP_FRACTION})"
    echo "  LR           : 1e-5"
    echo "  PATIENCE     : ${PATIENCE}"

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
#SBATCH --gres=gpu:l40:${NUM_GPUS}
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fengyu.cai@tu-darmstadt.de

set -eo pipefail

echo "=== Active Learning Slurm job started on \$(hostname) at \$(date) ==="

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

cd "\$CODE_ROOT"
which python
python -V

export WANDB_DISABLED=true

MASTER_PORT=\$(shuf -i 10000-45000 -n 1)
EXEC_PREFIX="torchrun --nproc_per_node=\$NPROC_PER_NODE --master_port=\$MASTER_PORT"

# ---------------------------------------------------------------
# CHANGE: CRISPR On-Target — Active Learning
# Single-sequence regression task evaluated by Spearman correlation.
# Uses active_learning_rna_kmer_supervised.py (single input, *ForSequenceClassification).
# Key differences from CRISPR off-target:
#   - Single sequence input (2-col CSV), not dual sgrna+target (3-col CSV)
#   - Model classes: *ForSequenceClassification (not *ForCRISPROffTarget)
#   - lr = 1e-5 (off-target uses 3e-5)
#   - save_steps = 400, eval_steps = 200 (same as original)
#   - output_dir includes SEED subdirectory (matching original convention)
# ---------------------------------------------------------------
task='CRISPROnTarget'
# CHANGE: lr = 1e-5 (from original config, different from off-target's 3e-5)
batch_size=32
lr=1e-5
DATA_PATH=\${DATA_ROOT}/\${task}
# CHANGE: output includes /SEED/ subdirectory matching original convention
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}/\${AL_STRATEGY}/\${SEED}

echo "--- [\$MODEL_NAME] Running \$task Active Learning (\$AL_STRATEGY) at \$(date) ---"

# CHANGE: calls active learning version of CRISPR on-target training script
# CHANGE: uses active/train_crispr_on_target.py (single-sequence *ForSequenceClassification)
# CHANGE: added --al_* flags, --patience, --metric_for_best_model spearman
# CHANGE: removed --train_fraction (replaced by AL scheduling)
\$EXEC_PREFIX active/train_crispr_on_target.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv \
  --data_val_path val.csv \
  --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task}_AL_\${AL_STRATEGY} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs ${AL_EPOCHS_PER_ROUND} \
  --fp16 \
  --save_steps 200 \
  --output_dir \${OUTPUT_PATH} \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 200 \
  --overwrite_output_dir True \
  --log_level info \
  --seed \${SEED} \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --metric_for_best_model spearman \
  --patience ${PATIENCE} \
  --al_strategy \${AL_STRATEGY} \
  --al_initial_fraction ${AL_INITIAL_FRACTION} \
  --al_target_fraction ${AL_TARGET_FRACTION} \
  --al_step_fraction ${AL_STEP_FRACTION} \
  --al_epochs_per_round ${AL_EPOCHS_PER_ROUND} \
  --al_num_mc_samples ${AL_NUM_MC_SAMPLES}

echo "=== Active Learning for \$MODEL_NAME (\$AL_STRATEGY) on CRISPROnTarget completed at \$(date) ==="
EOT

    echo "Submitted SLURM job: ${JOB_NAME}"
    echo ""
  done
done

echo "All CRISPR On-Target Active Learning jobs submitted."