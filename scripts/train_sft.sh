MODEL_PATH="${MODEL_PATH:-${AGENT_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}}"

SFT_DATA_PATH="$1"
CKPT_DIR="$2"

if [ -z "$SFT_DATA_PATH" ]; then
    echo "Error: SFT data path must be provided as the first argument."
    exit 1
fi
if [ ! -f "$SFT_DATA_PATH" ]; then
    echo "Error: SFT data file not found at '$SFT_DATA_PATH'"
    exit 1
fi
if [ -z "$CKPT_DIR" ]; then
    echo "Error: checkpoint dir must be provided as the second argument."
    exit 1
fi
mkdir -p "$CKPT_DIR"

# Derive processes from GPUs to avoid config mismatch
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
NUM_PROCS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

accelerate launch \
  --num_processes "$NUM_PROCS" \
  --main_process_port 29512 \
    ./train_sft.py \
    --model_name "$MODEL_PATH" \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 2 \
    --val_batch_size 1 \
    --accumulation_steps 8 \
    --num_epochs 500 \
    --use_lora \
    --use_qlora \
    --evaluation_steps 5 \
    --sft_data_path "$SFT_DATA_PATH" \
    --template_path ../configs/qwen2.5-7b.jinja \
    --checkpoint_dir "$CKPT_DIR"