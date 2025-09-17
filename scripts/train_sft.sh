export MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

SFT_DATA_PATH="$1"
if [ -z "$SFT_DATA_PATH" ]; then
    echo "Error: SFT data path must be provided as the first argument."
    exit 1
fi
if [ ! -f "$SFT_DATA_PATH" ]; then
    echo "Error: SFT data file not found at '$SFT_DATA_PATH'"
    exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6"} accelerate launch \
  --config_file ../configs/accelerate_config_sft.yaml \
  --main_process_port 29512 \
    ./train_sft.py \
    --model_name $MODEL_PATH \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --train_batch_size 2 \
    --val_batch_size 1 \
    --accumulation_steps 8 \
    --num_epochs 500 \
    --use_lora \
    --evaluation_steps 5 \
    --sft_data_path "$SFT_DATA_PATH" \
    --template_path ../configs/qwen2.5-7b.jinja \
    --checkpoint_dir "${CHECKPOINT_DIR_OVERRRIDE:-../sft_checkpoints_qwen2.5-7b}" \