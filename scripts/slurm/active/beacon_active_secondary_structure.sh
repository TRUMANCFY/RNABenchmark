#!/usr/bin/env bash
# submit_beacon_ss_active_learning.sh
# Submit active learning experiments for BEACON on RNA Secondary Structure
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
# Active learning strategies
##########################
# AL_STRATEGIES=("random" "entropy" "margin" "bald")
AL_STRATEGIES=( "bald")
# AL_STRATEGIES=("random")
# AL_STRATEGIES=("entropy" "margin")

# AL fractions: 10% -> 50% in 10% steps
AL_INITIAL_FRACTION=0.1
AL_TARGET_FRACTION=0.5
AL_STEP_FRACTION=0.1
AL_EPOCHS_PER_ROUND=100
AL_NUM_MC_SAMPLES=10

for entry in "${MODELS[@]}"; do
  read -r FOLDER_NAME MODEL_TYPE TOKEN_TYPE MAX_LENGTH CATEGORY <<< "$entry"

  PRETRAINED_PATH="${MODEL_ROOT}/${CATEGORY}/${FOLDER_NAME}/"
  MODEL_NAME="${FOLDER_NAME}"

  for AL_STRATEGY in "${AL_STRATEGIES[@]}"; do

    JOB_NAME="al_ss_${MODEL_NAME}_bpRNA_${AL_STRATEGY}"

    echo "Submitting Active Learning job:"
    echo "  MODEL_NAME   : ${MODEL_NAME}"
    echo "  AL_STRATEGY  : ${AL_STRATEGY}"
    echo "  FRACTIONS    : ${AL_INITIAL_FRACTION} -> ${AL_TARGET_FRACTION} (step ${AL_STEP_FRACTION})"

    sbatch <<EOT
#!/bin/bash
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --job-name=${JOB_NAME}
#SBATCH --error=${LOG_DIR}/${JOB_NAME}.err
#SBATCH --output=${LOG_DIR}/${JOB_NAME}.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a180:${NUM_GPUS}
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fengyu.cai@tu-darmstadt.de

set -eo pipefail

echo "=== SS Active Learning started on \$(hostname) at \$(date) ==="

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

echo "CODE_ROOT       = \$CODE_ROOT"
echo "MODEL_NAME      = \$MODEL_NAME"
echo "PRETRAINED_PATH = \$PRETRAINED_PATH"
echo "AL_STRATEGY     = \$AL_STRATEGY"

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
batch_size=1
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}/bpRNA
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}/\${AL_STRATEGY}

echo "--- [\$MODEL_NAME] Running \$task Active Learning (\$AL_STRATEGY) at \$(date) ---"

\$EXEC_PREFIX active/train_secondary_structure.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \${DATA_PATH} \
  --run_name \${MODEL_NAME}_\${task}_AL_\${AL_STRATEGY} \
  --output_dir \${OUTPUT_PATH} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr \${lr} \
  --num_epochs ${AL_EPOCHS_PER_ROUND} \
  --patience 60 \
  --num_workers 1 \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --seed \${SEED} \
  --mode bprna \
  --al_strategy \${AL_STRATEGY} \
  --al_initial_fraction ${AL_INITIAL_FRACTION} \
  --al_target_fraction ${AL_TARGET_FRACTION} \
  --al_step_fraction ${AL_STEP_FRACTION} \
  --al_epochs_per_round ${AL_EPOCHS_PER_ROUND} \
  --al_num_mc_samples ${AL_NUM_MC_SAMPLES}

echo "=== SS Active Learning for \$MODEL_NAME (\$AL_STRATEGY) completed at \$(date) ==="
EOT

    echo "Submitted SLURM job: ${JOB_NAME}"
    echo ""
  done
done

echo "All SS Active Learning jobs submitted."

#SBATCH --nodelist=penelope