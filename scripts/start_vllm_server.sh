#!/bin/bash

# vLLM server launcher using values from configs/config.yaml

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_READER="-m social_decipher.utils.config_reader"

# Change to project root so poetry/env works consistently
cd "$PROJECT_ROOT"

# Read all parameters from config
CONFIG_FILE="$PROJECT_ROOT/configs/config.yaml"

# Extract server settings using yq (without 'e' for pip-installed yq)
GPU_IDS=$(yq '.models.gpu' "$CONFIG_FILE" | tr -d '"')
MODEL_PATH=$(yq '.models.model_b' "$CONFIG_FILE" | tr -d '"')
PORT=$(yq '.models.vllm_port' "$CONFIG_FILE")
CHAT_TEMPLATE_REL=$(yq '.models.chat_template' "$CONFIG_FILE" | tr -d '"')
SERVED_MODEL_NAME=$(yq '.models.served_model_name' "$CONFIG_FILE" | tr -d '"')
MAX_MODEL_LEN=$(yq '.models.max_model_len' "$CONFIG_FILE")
TENSOR_PARALLEL_SIZE=$(yq '.models.tensor_parallel_size' "$CONFIG_FILE")

# Convert chat template to an absolute path
CHAT_TEMPLATE="$PROJECT_ROOT/$CHAT_TEMPLATE_REL"

# Use number of specified GPUs if tensor_parallel_size is 0 or null
if [ -z "$TENSOR_PARALLEL_SIZE" ] || [ "$TENSOR_PARALLEL_SIZE" -eq 0 ]; then
    if [ -n "$GPU_IDS" ]; then
        # Remove quotes before counting to correctly determine the number of GPUs
        TENSOR_PARALLEL_SIZE=$(echo "$GPU_IDS" | tr -d '"' | awk -F, '{print NF}')
    else
        TENSOR_PARALLEL_SIZE=1 # Default to 1 if no GPUs are specified
    fi
fi

echo "===================================="
echo "🚀 Starting vLLM Server (from configs/config.yaml)"
echo "===================================="
echo "GPUs:            $GPU_IDS"
echo "Model:           $MODEL_PATH"
echo "Port:            $PORT"
echo "Chat template:   $CHAT_TEMPLATE"
echo "Served name:     $SERVED_MODEL_NAME"
echo "Max model len:   $MAX_MODEL_LEN"
echo "Tensor parallel: $TENSOR_PARALLEL_SIZE"
echo ""

# If already running, exit early
if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
  echo "✅ vLLM server is already running on port $PORT"
  exit 0
fi

export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false

echo "Starting vLLM server..."
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --port $PORT \
  --chat-template "$CHAT_TEMPLATE" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --max-model-len $MAX_MODEL_LEN \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \


echo ""
echo "✅ vLLM server started"
echo "   URL:   http://localhost:$PORT"
echo "   Health: http://localhost:$PORT/health"
echo ""
echo "💡 Keep this terminal open. Use Ctrl+C to stop the server." 