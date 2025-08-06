#!/bin/bash

# Read model configuration from config.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_READER="$SCRIPT_DIR/config_reader.py"

# Change to project root for poetry command
cd "$PROJECT_ROOT"

export GLOBAL_MODEL_A=$(poetry run python "$CONFIG_READER" models.model_a)
export GLOBAL_MODEL_B=$(poetry run python "$CONFIG_READER" models.model_b)
export MODEL_NAME=$(poetry run python "$CONFIG_READER" models.served_model_name)
export GPU=$(poetry run python "$CONFIG_READER" models.gpu)
export VLLM_PORT=$(poetry run python "$CONFIG_READER" models.vllm_port)
export SCENARIO_TYPE=${SCENARIO_TYPE:-"normal"}           # normal, knowledge_barrier, language_barrier
export COMMUNICATION_MODALITY=${COMMUNICATION_MODALITY:-"action_enabled"}  # text_only, action_enabled, text_action_mix
export MEMORY_STRATEGY=${MEMORY_STRATEGY:-"off"}          # off, on
export EPISODE_LIMIT=${EPISODE_LIMIT:-3}                  # Number of episodes to run
TIMESTAMP=$(date +%m%d_%H%M)
export RESULTS_DIR=${RESULTS_DIR:-"results/exp_${SCENARIO_TYPE}_${COMMUNICATION_MODALITY}_mem${MEMORY_STRATEGY}_${MODEL_NAME}_${TIMESTAMP}"}

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
CUDA_VISIBLE_DEVICES=$GPU VLLM_PORT=$VLLM_PORT python scripts/run.py \
    --model_a $GLOBAL_MODEL_A \
    --model_b $GLOBAL_MODEL_B \
    --scenario_type $SCENARIO_TYPE \
    --communication_modality $COMMUNICATION_MODALITY \
    --memory_strategy $MEMORY_STRATEGY \
    --results_dir $RESULTS_DIR

echo ""
echo "✅ Experiment completed!" 