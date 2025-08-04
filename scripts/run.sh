#!/bin/bash

export GLOBAL_MODEL_A="gpt-4o-mini"
export GLOBAL_MODEL_B="/mnt/data_from_server1/models/Qwen2.5-7B-Instruct"
export GPU=1
export VLLM_PORT=6900
export SCENARIO_TYPE=${SCENARIO_TYPE:-"language_barrier"}           # normal, knowledge_barrier, language_barrier
export COMMUNICATION_MODALITY=${COMMUNICATION_MODALITY:-"text_only"}  # text_only, action_enabled, text_action_mix
export MEMORY_STRATEGY=${MEMORY_STRATEGY:-"off"}          # off, on
export EPISODE_LIMIT=${EPISODE_LIMIT:-3}                  # Number of episodes to run
TIMESTAMP=$(date +%m%d_%H%M)
export RESULTS_DIR=${RESULTS_DIR:-"../results/exp_${SCENARIO_TYPE}_${COMMUNICATION_MODALITY}_mem${MEMORY_STRATEGY}_${TIMESTAMP}"}

echo "===================================="
echo "🧪 Running Social Agent Experiment"
echo "===================================="
echo "Agent A: $GLOBAL_MODEL_A"
echo "Agent B: $GLOBAL_MODEL_B"
echo "GPU: $GPU"
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
CUDA_VISIBLE_DEVICES=$GPU VLLM_PORT=$VLLM_PORT python run.py \
    --model_a $GLOBAL_MODEL_A \
    --model_b $GLOBAL_MODEL_B \
    --scenario_type $SCENARIO_TYPE \
    --communication_modality $COMMUNICATION_MODALITY \
    --memory_strategy $MEMORY_STRATEGY \
    --results_dir $RESULTS_DIR

echo ""
echo "✅ Experiment completed!" 