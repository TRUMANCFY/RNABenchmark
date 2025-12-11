#!/bin/bash

export WANDB_DISABLED=true
# --- Global Configuration ---
# Note: gpu_device is now handled dynamically in the execution loop
nproc_per_node=1

# Paths
data_root=./data
model_root=./checkpoint
seed=666

# --- Function: Run All Tasks for a Specific Model ---
run_benchmark() {
    local MODEL_NAME=$1      # e.g., SpliceBERT-H510
    local MODEL_TYPE=$2      # e.g., splicebert-human510
    local TOKEN_TYPE=$3      # e.g., single, 3mer
    local MAX_LENGTH=$4      # e.g., 510, 1024
    local PRETRAINED_PATH=$5 # Path to the pretrained checkpoint
    local GPU_ID=$6          # Specific GPU ID for this run

    # Generate a unique random port for this specific run to avoid collisions
    # between parallel processes
    local master_port=$(shuf -i 10000-45000 -n 1)

    # Define execution prefix locally for this GPU and Port
    local EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"

    echo "========================================================"
    echo "Running Benchmark for: $MODEL_NAME"
    echo "Type: $MODEL_TYPE | Token: $TOKEN_TYPE | Len: $MAX_LENGTH"
    echo "Path: $PRETRAINED_PATH"
    echo "Worker: GPU $GPU_ID | Port: $master_port"
    echo "========================================================"
    # 2. Contact Map
    task='ContactMap'
    # ls ContactMap showed files directly inside, no 'bpRNA' subfolder
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test,RFAM19,DIRECT
    batch_size=1
    lr=3e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}
    
    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_contact_map.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path ${DATA_PATH} \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --output_dir ${OUTPUT_PATH} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --lr ${lr} \
        --num_epochs 100 \
        --patience 60 \
        --num_workers 1 \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE} \
        --seed ${seed}
}


# ==============================================================================
# EXECUTION SECTION
# ==============================================================================

# Associative array to store model configurations
# Format: [FolderName]="ModelType,TokenType,MaxLength,Category"
declare -A models

# --- RNA-FM, RNABERT, RNA-MSM ---
# models["rna-fm"]="rna-fm,single,1024,opensource"
# models["rnabert"]="rnabert,single,440,opensource"
# models["rnamsm"]="rnamsm,single,1024,opensource"

# --- SpliceBERT ---
models["splicebert-human510"]="splicebert-human510,single,510,opensource"
models["splicebert-ms510"]="splicebert-ms510,single,510,opensource"
models["splicebert-ms1024"]="splicebert-ms1024,single,1024,opensource"

# --- UTR-LM ---
# models["utr-lm-mrl"]="utr-lm-mrl,single,1026,opensource"
# models["utr-lm-te-el"]="utr-lm-te-el,single,1026,opensource"

# # --- UTRBERT (k-mer models) ---
# models["utrbert-3mer"]="utrbert-3mer,3mer,512,opensource"
# models["utrbert-4mer"]="utrbert-4mer,4mer,512,opensource"
# models["utrbert-5mer"]="utrbert-5mer,5mer,512,opensource"
# models["utrbert-6mer"]="utrbert-6mer,6mer,512,opensource"

# # --- BEACON ---
# models["BEACON-B"]="rnalm,single,1026,baseline"
# models["BEACON-B512"]="rnalm,single,512,baseline"


# --- Parallel Distribution Logic ---
model_names=("${!models[@]}")
total_models=${#model_names[@]}
gpu_id=3

echo "Detected ${total_models} models to benchmark on GPU ${gpu_id} only."

# Loop over all models, always using GPU 1
for (( i=0; i<total_models; i++ )); do
    folder_name="${model_names[$i]}"
    
    # Read config
    IFS=',' read -r model_type token_type max_length category <<< "${models[$folder_name]}"
    chkpt_path="${model_root}/${category}/${folder_name}/"
    
    echo "[GPU $gpu_id] Starting benchmark for: $folder_name"
    
    # Run Benchmark with specific GPU ID (always 1)
    run_benchmark "$folder_name" "$model_type" "$token_type" "$max_length" "$chkpt_path" "$gpu_id"
    
    echo "[GPU $gpu_id] Finished benchmark for: $folder_name"
done



# Wait for all background processes (GPUs) to finish
wait
echo "All benchmarks completed successfully across all GPUs."
