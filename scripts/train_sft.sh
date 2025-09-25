#!/bin/bash
# This script launches the SFT training process.
# It is designed to be called from the main train.sh script or run manually.

# --- Argument Parsing ---
SFT_DATA_PATH_ARG="$1"
CKPT_DIR_ARG="$2"
MODEL_PATH_ARG="$3"
LORA_CHECKPOINT_PATH_ARG="$4" # Optional: path to an existing LoRA checkpoint

# --- Path Resolution ---
# Resolve paths to be absolute BEFORE changing directory to ensure robustness.
SFT_DATA_PATH=$(python3 -c "import os; print(os.path.abspath('$SFT_DATA_PATH_ARG'))" 2>/dev/null || realpath "$SFT_DATA_PATH_ARG")
CKPT_DIR=$(python3 -c "import os; print(os.path.abspath('$CKPT_DIR_ARG'))" 2>/dev/null || realpath "$CKPT_DIR_ARG")

# Handle the model path: if it's a local directory, make it absolute. Otherwise, assume it's a Hub ID.
if [ -d "$MODEL_PATH_ARG" ]; then
    MODEL_PATH=$(python3 -c "import os; print(os.path.abspath('$MODEL_PATH_ARG'))" 2>/dev/null || realpath "$MODEL_PATH_ARG")
else
    MODEL_PATH="$MODEL_PATH_ARG"
fi

LORA_CHECKPOINT_PATH=""
if [ -n "$LORA_CHECKPOINT_PATH_ARG" ]; then
    LORA_CHECKPOINT_PATH=$(python3 -c "import os; print(os.path.abspath('$LORA_CHECKPOINT_PATH_ARG'))" 2>/dev/null || realpath "$LORA_CHECKPOINT_PATH_ARG")
fi

# --- Robust Pathing for Script Execution ---
# Now, change to the script's directory to ensure `train_sft.py` is found correctly.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# --- Argument Validation ---
if [ -z "$SFT_DATA_PATH_ARG" ]; then
    echo "Error: SFT data path must be provided as the first argument."
    exit 1
fi
if [ ! -f "$SFT_DATA_PATH" ]; then
    echo "Error: SFT data file not found at '$SFT_DATA_PATH'"
    exit 1
fi
if [ -z "$CKPT_DIR_ARG" ]; then
    echo "Error: Checkpoint directory must be provided as the second argument."
    exit 1
fi
if [ -z "$MODEL_PATH_ARG" ]; then
    echo "Error: Base model path must be provided as the third argument."
    exit 1
fi
mkdir -p "$CKPT_DIR"

echo "---"
echo "Starting SFT training with the following parameters:"
echo "Model Path: $MODEL_PATH"
echo "SFT Data Path: $SFT_DATA_PATH"
echo "Checkpoint Dir: $CKPT_DIR"
echo "LoRA Checkpoint Path: $LORA_CHECKPOINT_PATH"
echo "---"

# Runtime safety knobs
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Derive processes from GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  NUM_PROCS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  NUM_PROCS=1
fi

# Build the command arguments
CMD_ARGS=(
    ./train_sft.py
    --model_name_or_path "$MODEL_PATH"
    --learning_rate 1e-4
    --max_length 2048
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 4
    --num_train_epochs 10
    --eval_steps 5
    --sft_data_path "$SFT_DATA_PATH"
    --template_path qwen2.5-7b.jinja
    --output_dir "$CKPT_DIR"
    --save_strategy steps
    --save_steps 1000
)

# Conditionally add the LoRA checkpoint path only if it is provided
if [ -n "$LORA_CHECKPOINT_PATH" ]; then
    CMD_ARGS+=(--lora_checkpoint_path "$LORA_CHECKPOINT_PATH")
fi

CUDA_VISIBLE_DEVICES=8,9 accelerate launch \
  --num_processes "$NUM_PROCS" \
  --main_process_port 29512 \
    "${CMD_ARGS[@]}" 