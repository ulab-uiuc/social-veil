#!/bin/bash

# vLLM server launcher using values from configs/config.yaml

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_READER="-m social_decipher.utils.config_reader"

# Change to project root so poetry/env works consistently
cd "$PROJECT_ROOT"

# Read all parameters from config
MODEL=$(poetry run python $CONFIG_READER models.model_b)
GPU=$(poetry run python $CONFIG_READER models.gpu)
PORT=$(poetry run python $CONFIG_READER models.vllm_port)
CHAT_TEMPLATE=$(poetry run python $CONFIG_READER models.chat_template)
SERVED_NAME=$(poetry run python $CONFIG_READER models.served_model_name)
MAX_LEN=$(poetry run python $CONFIG_READER models.max_model_len)
TP=$(poetry run python $CONFIG_READER models.tensor_parallel_size)

# Derive tensor parallel size from GPU list if not set or zero
if [[ -z "$TP" || "$TP" == "0" ]]; then
  GPU_COUNT=$(awk -F',' '{print NF}' <<< "$GPU")
  if [[ -z "$GPU_COUNT" || "$GPU_COUNT" == "0" ]]; then
    GPU_COUNT=1
  fi
  TP=$GPU_COUNT
fi

echo "===================================="
echo "🚀 Starting vLLM Server (from configs/config.yaml)"
echo "===================================="
echo "GPUs:            $GPU"
echo "Model:           $MODEL"
echo "Port:            $PORT"
echo "Chat template:   $CHAT_TEMPLATE"
echo "Served name:     $SERVED_NAME"
echo "Max model len:   $MAX_LEN"
echo "Tensor parallel: $TP"
echo ""

# If already running, exit early
if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
  echo "✅ vLLM server is already running on port $PORT"
  exit 0
fi

export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false

echo "Starting vLLM server..."
CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port $PORT \
  --chat-template "$CHAT_TEMPLATE" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len $MAX_LEN \
  --tensor-parallel-size $TP \


echo ""
echo "✅ vLLM server started"
echo "   URL:   http://localhost:$PORT"
echo "   Health: http://localhost:$PORT/health"
echo ""
echo "💡 Keep this terminal open. Use Ctrl+C to stop the server." 