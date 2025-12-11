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

    # 1. Secondary Structure Prediction
    task='Secondary_structure_prediction'
    # data='bpRNA' not needed as variable if hardcoded in path
    batch_size=1
    lr=3e-5
    # The ls showed 'bpRNA' inside Secondary_structure_prediction
    DATA_PATH=${data_root}/${task}/bpRNA
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}
    
    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_secondary_structure.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path ${DATA_PATH} \
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

    # 3. Distance Map
    task='DistanceMap'
    # Assuming similar structure to ContactMap (flat csvs)
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test,RFAM19,DIRECT
    batch_size=1
    lr=5e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}
    
    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_distance_map.py \
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

    # 4. Structural Score Imputation
    task='StructuralScoreImputation'
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=32
    lr=3e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_structural_score_imputation.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 5. SpliceAI
    task='SpliceAI'
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=32
    lr=3e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_spliceai.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 6. Isoform
    task='Isoform'
    # Corrected filenames based on ls: train.csv, test.csv
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=32
    lr=5e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_isoform.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 7. NoncodingRNAFamily
    task='NoncodingRNAFamily'
    # Corrected filenames based on ls: train.csv, test.csv
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=16
    lr=5e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_ncrna.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH}/${seed} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 8. Modification
    task='Modification'
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=32
    lr=3e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_modification.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task}_seed${seed}_lr${lr} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH}/${seed} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 9. MeanRibosomeLoading
    task='MeanRibosomeLoading'
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=32
    lr=1e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_mean_ribosome_loading.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH}/${seed} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 10. Degradation
    task='Degradation'
    data_file_train=train_1.json; data_file_val=val_1.json; data_file_test=test_1.json
    batch_size=32
    lr=5e-5
    # ls showed files are in train-val-test subdir
    DATA_PATH=${data_root}/${task}/train-val-test
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_degradation.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 100 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH}/${seed} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 11. ProgrammableRNASwitches
    task='ProgrammableRNASwitches'
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=32
    lr=1e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_programmable_rna_switches.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH}/${seed} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 12. CRISPROnTarget
    task='CRISPROnTarget'
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=32
    lr=1e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_crispr_on_target.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH}/${seed} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}

    # 13. CRISPROffTarget
    task='CRISPROffTarget'
    data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
    batch_size=32
    lr=3e-5
    DATA_PATH=${data_root}/${task}
    OUTPUT_PATH=./outputs/ft/rna-all/${task}/${MODEL_NAME}

    echo "--- [$MODEL_NAME] Running $task ---"
    $EXEC_PREFIX \
        downstream/train_crispr_off_target.py \
        --model_name_or_path ${PRETRAINED_PATH} \
        --data_path $DATA_PATH \
        --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test} \
        --run_name ${MODEL_NAME}_${task} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 30 \
        --fp16 \
        --save_steps 400 \
        --output_dir ${OUTPUT_PATH} \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --seed ${seed} \
        --token_type ${TOKEN_TYPE} \
        --model_type ${MODEL_TYPE}
}


# ==============================================================================
# EXECUTION SECTION
# ==============================================================================

# Associative array to store model configurations
# Format: [FolderName]="ModelType,TokenType,MaxLength,Category"
declare -A models

# --- RNA-FM, RNABERT, RNA-MSM ---
models["rna-fm"]="rna-fm,single,1024,opensource"
models["rnabert"]="rnabert,single,440,opensource"
models["rnamsm"]="rnamsm,single,1024,opensource"

# --- SpliceBERT ---
models["splicebert-human510"]="splicebert-human510,single,510,opensource"
models["splicebert-ms510"]="splicebert-ms510,single,510,opensource"
models["splicebert-ms1024"]="splicebert-ms1024,single,1024,opensource"

# --- UTR-LM ---
models["utr-lm-mrl"]="utr-lm-mrl,single,1026,opensource"
models["utr-lm-te-el"]="utr-lm-te-el,single,1026,opensource"

# --- UTRBERT (k-mer models) ---
models["utrbert-3mer"]="utrbert-3mer,3mer,512,opensource"
models["utrbert-4mer"]="utrbert-4mer,4mer,512,opensource"
models["utrbert-5mer"]="utrbert-5mer,5mer,512,opensource"
models["utrbert-6mer"]="utrbert-6mer,6mer,512,opensource"

# --- BEACON ---
models["BEACON-B"]="rnalm,single,1026,baseline"
models["BEACON-B512"]="rnalm,single,512,baseline"


# --- Parallel Distribution Logic ---
model_names=("${!models[@]}")
total_models=${#model_names[@]}
total_gpus=8

echo "Detected ${total_models} models to benchmark on ${total_gpus} GPUs."

# Loop over GPUs (0 to 7)
for (( gpu_id=0; gpu_id<total_gpus; gpu_id++ )); do
    (
        # Each GPU processes a subset of models (round-robin)
        # i starts at the gpu_id and steps by the number of GPUs
        for (( i=gpu_id; i<total_models; i+=total_gpus )); do
            folder_name="${model_names[$i]}"
            
            # Read config
            IFS=',' read -r model_type token_type max_length category <<< "${models[$folder_name]}"
            chkpt_path="${model_root}/${category}/${folder_name}/"
            
            echo "[GPU $gpu_id] Starting benchmark for: $folder_name"
            
            # Run Benchmark with specific GPU ID
            run_benchmark "$folder_name" "$model_type" "$token_type" "$max_length" "$chkpt_path" "$gpu_id"
            
            echo "[GPU $gpu_id] Finished benchmark for: $folder_name"
        done
    ) &
done

# Wait for all background processes (GPUs) to finish
wait
echo "All benchmarks completed successfully across all GPUs."