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

  echo "Submitting BEACON benchmark for:"
  echo "  MODEL_NAME      : ${MODEL_NAME}"
  echo "  MODEL_TYPE      : ${MODEL_TYPE}"
  echo "  TOKEN_TYPE      : ${TOKEN_TYPE}"
  echo "  MAX_LENGTH      : ${MAX_LENGTH}"
  echo "  PRETRAINED_PATH : ${PRETRAINED_PATH}"

  sbatch <<EOT
#!/bin/bash
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --job-name=beacon_${MODEL_NAME}_others
#SBATCH --error=${LOG_DIR}/beacon_${MODEL_NAME}_others.err
#SBATCH --output=${LOG_DIR}/beacon_${MODEL_NAME}_others.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100pcie:${NUM_GPUS}
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fengyu.cai@tu-darmstadt.de

set -eo pipefail

echo "=== BEACON Slurm job started on \$(hostname) at \$(date) ==="

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

echo "CODE_ROOT       = \$CODE_ROOT"
echo "MODEL_NAME      = \$MODEL_NAME"
echo "PRETRAINED_PATH = \$PRETRAINED_PATH"

cd "\$CODE_ROOT"
which python
python -V

export WANDB_DISABLED=true

# Generate a unique random port to avoid collisions
MASTER_PORT=\$(shuf -i 10000-45000 -n 1)
EXEC_PREFIX="torchrun --nproc_per_node=\$NPROC_PER_NODE --master_port=\$MASTER_PORT"


# ---------------------------------------------------------------
# 5. SpliceAI
# ---------------------------------------------------------------
task='SpliceAI'
batch_size=32
lr=3e-5
DATA_PATH=\${DATA_ROOT}/\${task}
OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
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
  --save_steps 5000 \
  --output_dir \${OUTPUT_PATH} \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --warmup_steps 50 \
  --logging_steps 200 \
  --overwrite_output_dir True \
  --log_level info \
  --seed \${SEED} \
  --token_type \${TOKEN_TYPE} \
  --model_type \${MODEL_TYPE}

echo "=== All tasks for \$MODEL_NAME completed at \$(date) ==="
EOT

  echo "Submitted SLURM job for ${MODEL_NAME}"
  echo ""

done

echo "All BEACON benchmark jobs submitted."


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
#   --save_steps 5000 \
#   --output_dir \${OUTPUT_PATH} \
#   --evaluation_strategy steps \
#   --eval_steps 5000 \
#   --warmup_steps 50 \
#   --logging_steps 200 \
#   --overwrite_output_dir True \
#   --log_level info \
#   --seed \${SEED} \
#   --token_type \${TOKEN_TYPE} \
#   --model_type \${MODEL_TYPE}


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

# # # ---------------------------------------------------------------
# # # 5. SpliceAI
# # # ---------------------------------------------------------------
# # task='SpliceAI'
# # batch_size=32
# # lr=3e-5
# # DATA_PATH=\${DATA_ROOT}/\${task}
# # OUTPUT_PATH=\${OUT_ROOT}/\${task}/\${MODEL_NAME}
# # echo "--- [\$MODEL_NAME] Running \$task at \$(date) ---"
# # \$EXEC_PREFIX downstream/train_spliceai.py \
# #   --model_name_or_path \${PRETRAINED_PATH} \
# #   --data_path \$DATA_PATH \
# #   --data_train_path train.csv --data_val_path val.csv --data_test_path test.csv \
# #   --run_name \${MODEL_NAME}_\${task} \
# #   --model_max_length \${MAX_LENGTH} \
# #   --per_device_train_batch_size \${batch_size} \
# #   --per_device_eval_batch_size 32 \
# #   --gradient_accumulation_steps 1 \
# #   --learning_rate \${lr} \
# #   --num_train_epochs 30 \
# #   --fp16 \
# #   --save_steps 400 \
# #   --output_dir \${OUTPUT_PATH} \
# #   --evaluation_strategy steps \
# #   --eval_steps 200 \
# #   --warmup_steps 50 \
# #   --logging_steps 200 \
# #   --overwrite_output_dir True \
# #   --log_level info \
# #   --seed \${SEED} \
# #   --token_type \${TOKEN_TYPE} \
# #   --model_type \${MODEL_TYPE}

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

#SBATCH --nodelist=penelope
