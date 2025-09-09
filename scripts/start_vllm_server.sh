#!/bin/bash

# Read model configuration from config.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_READER="-m social_decipher.utils.config_reader"

# Change to project root for poetry command
cd "$PROJECT_ROOT"

echo "===================================="
echo "🐛 DEBUG: Checking config file content before reading:"
cat configs/config.yaml | grep "vllm_port:"
echo "===================================="

export GLOBAL_MODEL_B=$(poetry run python $CONFIG_READER models.model_b)
export VLLM_GPU=$(poetry run python $CONFIG_READER models.gpu)
export VLLM_PORT=$(poetry run python $CONFIG_READER models.vllm_port)

echo "===================================="
echo "🚀 Starting vLLM Server"
echo "===================================="
echo "Model: $GLOBAL_MODEL_B"
echo "Port: $VLLM_PORT"
echo "🐛 DEBUG: The port read into the script variable is: $VLLM_PORT"
echo "GPU: $VLLM_GPU"
echo ""

# Check if server is already running
if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "✅ vLLM server is already running on port $VLLM_PORT"
    exit 0
fi

echo "Starting vLLM server..."
CHAT_TEMPLATE=$(poetry run python $CONFIG_READER models.chat_template)
SERVED_MODEL_NAME=$(poetry run python $CONFIG_READER models.served_model_name)
MAX_MODEL_LEN=$(poetry run python $CONFIG_READER models.max_model_len)
TENSOR_PARALLEL_SIZE=$(poetry run python $CONFIG_READER models.tensor_parallel_size)

# If tensor_parallel_size not set, infer from number of GPUs listed
if [[ -z "$TENSOR_PARALLEL_SIZE" || "$TENSOR_PARALLEL_SIZE" == "0" ]]; then
  GPU_COUNT=$(awk -F',' '{print NF}' <<< "$VLLM_GPU")
  if [[ -z "$GPU_COUNT" || "$GPU_COUNT" == "0" ]]; then
    GPU_COUNT=1
  fi
  TENSOR_PARALLEL_SIZE=$GPU_COUNT
fi

echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"

CUDA_VISIBLE_DEVICES=$VLLM_GPU python -m vllm.entrypoints.openai.api_server \
    --model $GLOBAL_MODEL_B \
    --port $VLLM_PORT \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name $SERVED_MODEL_NAME \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE

echo ""
echo "✅ vLLM server started successfully!"
echo "   Server URL: http://localhost:$VLLM_PORT"
echo "   Health check: http://localhost:$VLLM_PORT/health"
echo ""
echo "💡 Keep this terminal open. Use Ctrl+C to stop the server." 