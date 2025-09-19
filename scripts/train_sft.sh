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

# Runtime safety knobs
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Derive processes from GPUs to avoid config mismatch; don't override user's selection
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  NUM_PROCS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  # Safe default: single GPU/process if user didn't set devices upstream
  NUM_PROCS=1
fi

CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch \
  --num_processes "$NUM_PROCS" \
  --main_process_port 29512 \
    ./train_sft.py \
    --model_name "$MODEL_PATH" \
    --learning_rate 5e-5 \
    --max_length 2048 \
    --train_batch_size 4 \
    --val_batch_size 1 \
    --accumulation_steps 4 \
    --num_epochs 10 \
    --use_lora \
    --use_qlora \
    --evaluation_steps 5 \
    --sft_data_path "$SFT_DATA_PATH" \
    --template_path ../configs/qwen2.5-7b.jinja \
    --checkpoint_dir "$CKPT_DIR" 