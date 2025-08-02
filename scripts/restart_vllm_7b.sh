#!/bin/bash

echo "🔄 Restarting vLLM server with Qwen2.5-7B-Instruct..."

# Kill existing vLLM processes
echo "   Stopping existing vLLM server..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# Start vLLM server with Qwen2.5-7B-Instruct
echo "   Starting vLLM server with Qwen2.5-7B-Instruct..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --chat-template configs/qwen2.5-7b.jinja \
    --served-model-name qwen2.5-7b \
    --max-model-len 4096 \
    --tensor-parallel-size 1 &

echo "   Waiting for server to start..."
sleep 10

# Check if server is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ vLLM server started successfully!"
    echo "   Available models:"
    curl -s http://localhost:8000/v1/models | python -m json.tool
else
    echo "❌ Failed to start vLLM server"
    exit 1
fi 