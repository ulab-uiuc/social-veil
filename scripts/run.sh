#!/bin/bash

export GLOBAL_MODEL_A="gpt-4o-mini"
export GLOBAL_MODEL_B="/mnt/data_from_server1/models/Qwen2.5-7B-Instruct"
export DJANGO_GPU=1
export VLLM_PORT=8010

echo "===================================="
echo "🧪 Running Social Agent Experiment"
echo "===================================="
echo "Agent A: $GLOBAL_MODEL_A"
echo "Agent B: $GLOBAL_MODEL_B"
echo "GPU: $DJANGO_GPU"
echo ""

# Check if vLLM server is running
echo "Checking vLLM server..."
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "❌ vLLM server is not running on port $VLLM_PORT"
    echo ""
    echo "💡 Please start the server first:"
    echo "   ./scripts/start_vllm_server.sh"
    echo ""
    echo "   Then run this script in another terminal."
    exit 1
fi

echo "✅ vLLM server is running"
echo ""

# Run the experiment
echo "Starting experiment..."
CUDA_VISIBLE_DEVICES=$DJANGO_GPU VLLM_PORT=$VLLM_PORT python run.py --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B --episode_limit 3

echo ""
echo "✅ Experiment completed!" 