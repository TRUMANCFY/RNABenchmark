#!/usr/bin/env bash
# submit_beacon_eval.sh
set -euo pipefail

# --- Global Paths ---
CODE_ROOT=/storage/ukp/work/cai_e/RNABenchmark
DATA_ROOT=${CODE_ROOT}/data
MODEL_ROOT=${CODE_ROOT}/checkpoint
OUT_ROOT=${CODE_ROOT}/outputs_slurm/ft/rna-all
LOG_DIR=${CODE_ROOT}/logs/beacon
mkdir -p "$LOG_DIR"

SEED=666
NPROC_PER_NODE=1
NUM_GPUS=1

# Base training config
BASE_EPOCHS=30

# Fractions to sweep
# FRACTIONS=(0.01 0.1 0.2 0.5)
# FRACTIONS=(0.2 0.5)
# FRACTIONS=(0.01 0.1)
FRACTIONS=(0.02 0.05 0.4 0.8)


##########################
# Define models to evaluate
# Format: "FolderName   ModelType   TokenType   MaxLength   Category"
##########################

MODELS=(
  "BEACON-B    rnalm   single  1026    baseline"
#   "BEACON-B512 rnalm   single  512     baseline"
)

for entry in "${MODELS[@]}"; do
  read -r FOLDER_NAME MODEL_TYPE TOKEN_TYPE MAX_LENGTH CATEGORY <<< "$entry"

  PRETRAINED_PATH="${MODEL_ROOT}/${CATEGORY}/${FOLDER_NAME}/"
  MODEL_NAME="${FOLDER_NAME}"

  for FRAC in "${FRACTIONS[@]}"; do
    # Create a readable fraction tag for naming (e.g., 0.01 -> frac001)
    FRAC_TAG=$(echo "$FRAC" | sed 's/\.//g')

    echo "Submitting BEACON benchmark for:"
    echo "  MODEL_NAME      : ${MODEL_NAME}"
    echo "  MODEL_TYPE      : ${MODEL_TYPE}"
    echo "  TOKEN_TYPE      : ${TOKEN_TYPE}"
    echo "  MAX_LENGTH      : ${MAX_LENGTH}"
    echo "  PRETRAINED_PATH : ${PRETRAINED_PATH}"
    echo "  TRAIN_FRACTION  : ${FRAC}"

    sbatch <<EOT
#!/bin/bash
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --job-name=beacon_${MODEL_NAME}_frac${FRAC_TAG}_others
#SBATCH --error=${LOG_DIR}/beacon_${MODEL_NAME}_frac${FRAC_TAG}_others.err
#SBATCH --output=${LOG_DIR}/beacon_${MODEL_NAME}_frac${FRAC_TAG}_others.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a180:${NUM_GPUS}
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fengyu.cai@tu-darmstadt.de
#SBATCH --nodelist=penelope


set -eo pipefail

echo "=== BEACON Slurm job started on \$(hostname) at \$(date) ==="
echo "=== Train fraction: ${FRAC} ==="

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
FRAC="${FRAC}"
FRAC_TAG="${FRAC_TAG}"

echo "CODE_ROOT       = \$CODE_ROOT"
echo "MODEL_NAME      = \$MODEL_NAME"
echo "PRETRAINED_PATH = \$PRETRAINED_PATH"
echo "TRAIN_FRACTION  = \$FRAC"

cd "\$CODE_ROOT"
which python
python -V

export WANDB_DISABLED=true

# Generate a unique random port to avoid collisions
MASTER_PORT=\$(shuf -i 10000-45000 -n 1)
EXEC_PREFIX="torchrun --nproc_per_node=\$NPROC_PER_NODE --master_port=\$MASTER_PORT"


# ---------------------------------------------------------------
# 1. Secondary Structure Prediction
# ---------------------------------------------------------------
task='Secondary_structure_prediction'
batch_size=1
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}/bpRNA
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_secondary_structure.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \${DATA_PATH} \
  --run_name \${MODEL_NAME}_\${task} \
  --output_dir \${OUTPUT_PATH} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr \${lr} \
  --num_epochs 100 \
  --patience 60 \
  --num_workers 1 \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --seed \${SEED} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 2. Contact Map
# ---------------------------------------------------------------
task='ContactMap'
batch_size=1
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_contact_map.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \${DATA_PATH} \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test,RFAM19,DIRECT \
  --run_name \${MODEL_NAME}_\${task} \
  --output_dir \${OUTPUT_PATH} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr \${lr} \
  --num_epochs 100 \
  --patience 60 \
  --num_workers 1 \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --seed \${SEED} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 3. Distance Map
# ---------------------------------------------------------------
task='DistanceMap'
batch_size=1
lr=5e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_distance_map.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \${DATA_PATH} \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test,RFAM19,DIRECT \
  --run_name \${MODEL_NAME}_\${task} \
  --output_dir \${OUTPUT_PATH} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr \${lr} \
  --num_epochs 100 \
  --patience 60 \
  --num_workers 1 \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --seed \${SEED} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 4. Structural Score Imputation
# ---------------------------------------------------------------
task='StructuralScoreImputation'
batch_size=32
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_structural_score_imputation.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
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
  --train_fraction \${FRAC}


# ---------------------------------------------------------------
# 5. SpliceAI
# ---------------------------------------------------------------
task='SpliceAI'
batch_size=32
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_spliceai.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
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
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 5. Isoform
# ---------------------------------------------------------------
task='Isoform'
batch_size=32
lr=5e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_isoform.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
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
  --train_fraction \${FRAC}


# ---------------------------------------------------------------
# 6. NoncodingRNAFamily
# ---------------------------------------------------------------
task='NoncodingRNAFamily'
batch_size=16
lr=5e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_ncrna.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
  --output_dir \${OUTPUT_PATH}/\${SEED} \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 200 \
  --overwrite_output_dir True \
  --log_level info \
  --seed \${SEED} \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 7. Modification
# ---------------------------------------------------------------
task='Modification'
batch_size=32
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_modification.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task}_seed\${SEED}_lr\${lr} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
  --output_dir \${OUTPUT_PATH}/\${SEED} \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 200 \
  --overwrite_output_dir True \
  --log_level info \
  --seed \${SEED} \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 8. MeanRibosomeLoading
# ---------------------------------------------------------------
task='MeanRibosomeLoading'
batch_size=32
lr=1e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_mean_ribosome_loading.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
  --output_dir \${OUTPUT_PATH}/\${SEED} \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 200 \
  --overwrite_output_dir True \
  --log_level info \
  --seed \${SEED} \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 9. Degradation
# ---------------------------------------------------------------
task='Degradation'
batch_size=32
lr=5e-5
DATA_PATH=\${DATA_ROOT}/\${task}/train-val-test
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_degradation.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train_1.json --data_val_path val_1.json --data_test_path test_1.json \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 100 \
  --fp16 \
  --save_steps 400 \
  --output_dir \${OUTPUT_PATH}/\${SEED} \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 200 \
  --overwrite_output_dir True \
  --log_level info \
  --seed \${SEED} \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 10. ProgrammableRNASwitches
# ---------------------------------------------------------------
task='ProgrammableRNASwitches'
batch_size=32
lr=1e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_programmable_rna_switches.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
  --output_dir \${OUTPUT_PATH}/\${SEED} \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 200 \
  --overwrite_output_dir True \
  --log_level info \
  --seed \${SEED} \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 11. CRISPROnTarget
# ---------------------------------------------------------------
task='CRISPROnTarget'
batch_size=32
lr=1e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_crispr_on_target.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
  --output_dir \${OUTPUT_PATH}/\${SEED} \
  --evaluation_strategy steps \
  --eval_steps 200 \
  --warmup_steps 50 \
  --logging_steps 200 \
  --overwrite_output_dir True \
  --log_level info \
  --seed \${SEED} \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE} \
  --train_fraction \${FRAC}

# ---------------------------------------------------------------
# 12. CRISPROffTarget
# ---------------------------------------------------------------
task='CRISPROffTarget'
batch_size=32
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}_frac\${FRAC_TAG}
echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
\$EXEC_PREFIX downstream/train_crispr_off_target.py \
  --model_name_or_path \${PRETRAINED_PATH} \
  --data_path \$DATA_PATH \
  --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
  --run_name \${MODEL_NAME}_\${task} \
  --model_max_length \${MAX_LENGTH} \
  --per_device_train_batch_size \${batch_size} \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate \${lr} \
  --num_train_epochs 30 \
  --fp16 \
  --save_steps 400 \
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
  --train_fraction \${FRAC}

echo "=== All tasks for \$MODEL_NAME (fraction=\$FRAC) completed at \$(date) ==="
EOT

    echo "Submitted SLURM job for ${MODEL_NAME} with fraction=${FRAC}"
    echo ""

  done  # end fraction loop
done  # end model loop

echo "All BEACON benchmark jobs submitted."







# #!/usr/bin/env bash
# # submit_beacon_eval.sh
# set -euo pipefail

# # --- Global Paths ---
# CODE_ROOT=/storage/ukp/work/cai_e/RNABenchmark
# DATA_ROOT=${CODE_ROOT}/data
# MODEL_ROOT=${CODE_ROOT}/checkpoint
# OUT_ROOT=${CODE_ROOT}/outputs_slurm/ft/rna-all
# LOG_DIR=${CODE_ROOT}/logs/beacon
# mkdir -p "$LOG_DIR"

# SEED=666
# NPROC_PER_NODE=1
# NUM_GPUS=1

# # Base training config
# BASE_EPOCHS=30

# # Fractions to sweep
# FRACTIONS=(0.01 0.1 0.2 0.5)

# ##########################
# # Define models to evaluate
# # Format: "FolderName   ModelType   TokenType   MaxLength   Category"
# ##########################

# MODELS=(
#   "BEACON-B    rnalm   single  1026    baseline"
# #   "BEACON-B512 rnalm   single  512     baseline"
# )

# for entry in "${MODELS[@]}"; do
#   read -r FOLDER_NAME MODEL_TYPE TOKEN_TYPE MAX_LENGTH CATEGORY <<< "$entry"

#   PRETRAINED_PATH="${MODEL_ROOT}/${CATEGORY}/${FOLDER_NAME}/"
#   MODEL_NAME="${FOLDER_NAME}"

#   for FRAC in "${FRACTIONS[@]}"; do
#     # Create a readable fraction tag for naming (e.g., 0.01 -> frac001)
#     FRAC_TAG=$(echo "$FRAC" | sed 's/\.//g')

#     echo "Submitting BEACON benchmark for:"
#     echo "  MODEL_NAME      : ${MODEL_NAME}"
#     echo "  MODEL_TYPE      : ${MODEL_TYPE}"
#     echo "  TOKEN_TYPE      : ${TOKEN_TYPE}"
#     echo "  MAX_LENGTH      : ${MAX_LENGTH}"
#     echo "  PRETRAINED_PATH : ${PRETRAINED_PATH}"
#     echo "  TRAIN_FRACTION  : ${FRAC}"
#     echo "  ADJUSTED_EPOCHS : ${ADJUSTED_EPOCHS} (base=${BASE_EPOCHS})"

#     sbatch <<EOT
# #!/bin/bash
# #SBATCH -p gpu
# #SBATCH -q gpu
# #SBATCH --job-name=beacon_${MODEL_NAME}_frac${FRAC_TAG}
# #SBATCH --error=${LOG_DIR}/beacon_${MODEL_NAME}_frac${FRAC_TAG}.err
# #SBATCH --output=${LOG_DIR}/beacon_${MODEL_NAME}_frac${FRAC_TAG}.out
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=16
# #SBATCH --time=3-00:00:00
# #SBATCH --gres=gpu:a100:${NUM_GPUS}
# #SBATCH --mem-per-cpu=8000
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=fengyu.cai@tu-darmstadt.de

# set -eo pipefail

# echo "=== BEACON Slurm job started on \$(hostname) at \$(date) ==="
# echo "=== Train fraction: ${FRAC}, Adjusted epochs: ${ADJUSTED_EPOCHS} ==="

# source /storage/ukp/work/cai_e/anaconda3/bin/activate beacon
# export LD_LIBRARY_PATH=/storage/ukp/work/cai_e/anaconda3/envs/beacon/lib/:\$LD_LIBRARY_PATH

# CODE_ROOT="${CODE_ROOT}"
# DATA_ROOT="${DATA_ROOT}"
# MODEL_ROOT="${MODEL_ROOT}"
# OUT_ROOT="${OUT_ROOT}"
# MODEL_NAME="${MODEL_NAME}"
# MODEL_TYPE="${MODEL_TYPE}"
# TOKEN_TYPE="${TOKEN_TYPE}"
# MAX_LENGTH="${MAX_LENGTH}"
# PRETRAINED_PATH="${PRETRAINED_PATH}"
# SEED="${SEED}"
# NPROC_PER_NODE="${NPROC_PER_NODE}"
# FRAC="${FRAC}"
# ADJUSTED_EPOCHS="${ADJUSTED_EPOCHS}"
# FRAC_TAG="${FRAC_TAG}"

# echo "CODE_ROOT       = \$CODE_ROOT"
# echo "MODEL_NAME      = \$MODEL_NAME"
# echo "PRETRAINED_PATH = \$PRETRAINED_PATH"
# echo "TRAIN_FRACTION  = \$FRAC"
# echo "ADJUSTED_EPOCHS = \$ADJUSTED_EPOCHS"

# cd "\$CODE_ROOT"
# which python
# python -V

# export WANDB_DISABLED=true

# # Generate a unique random port to avoid collisions
# MASTER_PORT=\$(shuf -i 10000-45000 -n 1)
# EXEC_PREFIX="torchrun --nproc_per_node=\$NPROC_PER_NODE --master_port=\$MASTER_PORT"


# # ---------------------------------------------------------------
# # 1. Secondary Structure Prediction
# # ---------------------------------------------------------------
# task='Secondary_structure_prediction'
# batch_size=1
# lr=3e-5
# DATA_PATH=\${DATA_ROOT}/\${task}/bpRNA
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_secondary_structure.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \${DATA_PATH} \
#   --run_name \${MODEL_NAME}_\${task} \
#   --output_dir \${OUTPUT_PATH} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --lr \${lr} \
#   --num_epochs 100 \
#   --patience 60 \
#   --num_workers 1 \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE} \
#   --seed \${SEED}

# # ---------------------------------------------------------------
# # 2. Contact Map
# # ---------------------------------------------------------------
# task='ContactMap'
# batch_size=1
# lr=3e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_contact_map.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \${DATA_PATH} \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test,RFAM19,DIRECT \
#   --run_name \${MODEL_NAME}_\${task} \
#   --output_dir \${OUTPUT_PATH} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --lr \${lr} \
#   --num_epochs 100 \
#   --patience 60 \
#   --num_workers 1 \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE} \
#   --seed \${SEED}

# # ---------------------------------------------------------------
# # 3. Distance Map
# # ---------------------------------------------------------------
# task='DistanceMap'
# batch_size=1
# lr=5e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_distance_map.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \${DATA_PATH} \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test,RFAM19,DIRECT \
#   --run_name \${MODEL_NAME}_\${task} \
#   --output_dir \${OUTPUT_PATH} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --lr \${lr} \
#   --num_epochs 100 \
#   --patience 60 \
#   --num_workers 1 \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE} \
#   --seed \${SEED}

# # ---------------------------------------------------------------
# # 4. Structural Score Imputation
# # ---------------------------------------------------------------
# task='StructuralScoreImputation'
# batch_size=32
# lr=3e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_structural_score_imputation.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 5. SpliceAI
# # ---------------------------------------------------------------
# task='SpliceAI'
# batch_size=32
# lr=3e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_spliceai.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 6. Isoform
# # ---------------------------------------------------------------
# task='Isoform'
# batch_size=32
# lr=5e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_isoform.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 7. NoncodingRNAFamily
# # ---------------------------------------------------------------
# task='NoncodingRNAFamily'
# batch_size=16
# lr=5e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_ncrna.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 2 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH}/\${SEED} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 8. Modification
# # ---------------------------------------------------------------
# task='Modification'
# batch_size=32
# lr=3e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_modification.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task}_seed\${SEED}_lr\${lr} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH}/\${SEED} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 9. MeanRibosomeLoading
# # ---------------------------------------------------------------
# task='MeanRibosomeLoading'
# batch_size=32
# lr=1e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_mean_ribosome_loading.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH}/\${SEED} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 10. Degradation
# # ---------------------------------------------------------------
# task='Degradation'
# batch_size=32
# lr=5e-5
# DATA_PATH=\${DATA_ROOT}/\${task}/train-val-test
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_degradation.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train_1.json --data_val_path val_1.json --data_test_path test_1.json \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 100 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH}/\${SEED} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 11. ProgrammableRNASwitches
# # ---------------------------------------------------------------
# task='ProgrammableRNASwitches'
# batch_size=32
# lr=1e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_programmable_rna_switches.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH}/\${SEED} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 12. CRISPROnTarget
# # ---------------------------------------------------------------
# task='CRISPROnTarget'
# batch_size=32
# lr=1e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_crispr_on_target.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH}/\${SEED} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# # ---------------------------------------------------------------
# # 13. CRISPROffTarget
# # ---------------------------------------------------------------
# task='CRISPROffTarget'
# batch_size=32
# lr=3e-5
# DATA_PATH=\${DATA_ROOT}/\${task}
# OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# \$EXEC_PREFIX downstream/train_crispr_off_target.py \
#   --model_name_or_path \${PRETRAINED_PATH} \
#   --data_path \$DATA_PATH \
#   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
#   --run_name \${MODEL_NAME}_\${task} \
#   --model_max_length \${MAX_LENGTH} \
#   --per_device_train_batch_size \${batch_size} \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate \${lr} \
#   --num_train_epochs 30 \
#   --fp16 \
#   --save_steps 400 \
#   --output_dir \${OUTPUT_PATH} \
#   --evaluation_strategy steps \
#   --eval_steps 200 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}

# echo "=== All tasks for \$MODEL_NAME (fraction=\$FRAC) completed at \$(date) ==="
# EOT

#     echo "Submitted SLURM job for ${MODEL_NAME} with fraction=${FRAC}"
#     echo ""

#   done  # end fraction loop
# done  # end model loop

# echo "All BEACON benchmark jobs submitted."


