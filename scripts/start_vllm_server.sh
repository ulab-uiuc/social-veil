#!/bin/bash

export GLOBAL_MODEL_B="/mnt/data_from_server1/models/Qwen2.5-7B-Instruct"
export VLLM_GPU=0
export VLLM_PORT=8010

echo "===================================="
echo "🚀 Starting vLLM Server"
echo "===================================="
echo "Model: $GLOBAL_MODEL_B"
echo "Port: $VLLM_PORT"
echo "GPU: $VLLM_GPU"
echo ""

# Check if server is already running
if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "✅ vLLM server is already running on port $VLLM_PORT"
    exit 0
fi

echo "Starting vLLM server..."
CUDA_VISIBLE_DEVICES=$VLLM_GPU python -m vllm.entrypoints.openai.api_server \
    --model $GLOBAL_MODEL_B \
    --port $VLLM_PORT \
    --chat-template ../configs/qwen2.5-7b.jinja \
    --served-model-name qwen2.5-7b-instruct \
    --max-model-len 4096 \
    --tensor-parallel-size 1

echo ""
echo "✅ vLLM server started successfully!"
echo "   Server URL: http://localhost:$VLLM_PORT"
echo "   Health check: http://localhost:$VLLM_PORT/health"
echo ""
echo "💡 Keep this terminal open. Use Ctrl+C to stop the server." 