#!/bin/bash

# Read model configuration from config.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_READER="$SCRIPT_DIR/config_reader.py"

# Change to project root for poetry command
cd "$PROJECT_ROOT"

export GLOBAL_MODEL_A=$(poetry run python "$CONFIG_READER" models.model_a)
export GLOBAL_MODEL_B=$(poetry run python "$CONFIG_READER" models.model_b)
export DATA_NAME=$(poetry run python "$CONFIG_READER" data_dir)
# Derive a short tag from the data file (basename without extension), e.g., 'data/episode_hard.jsonl' -> 'episode_hard'
DATA_FILE_NAME=$(basename "$DATA_NAME")
DATA_TAG="${DATA_FILE_NAME%.*}"
export MODEL_NAME=$(poetry run python "$CONFIG_READER" models.served_model_name)
export GPU=$(poetry run python "$CONFIG_READER" models.gpu)
export VLLM_PORT=$(poetry run python "$CONFIG_READER" models.vllm_port)
export SCENARIO_TYPE=${SCENARIO_TYPE:-"normal"}           # normal, knowledge_barrier, language_barrier
export COMMUNICATION_MODALITY=${COMMUNICATION_MODALITY:-"text_only"}  # text_only, action_enabled, text_action_mix
export MEMORY_STRATEGY=${MEMORY_STRATEGY:-"off"}          # off, on
export BARRIER_RATIO=${BARRIER_RATIO:-1.0}                # For language_barrier: 0..1
export BARRIER_RATIOS=${BARRIER_RATIOS:-""}              # Optional list, e.g. "0.1 0.5 0.75 1"
TIMESTAMP=$(date +%m%d_%H%M)

# Set default RESULTS_DIR only for non-language-barrier; for language barrier we set per ratio below
if [ "$SCENARIO_TYPE" != "language_barrier" ]; then
    export RESULTS_DIR=${RESULTS_DIR:-"results/exp_${SCENARIO_TYPE}_${COMMUNICATION_MODALITY}_mem${MEMORY_STRATEGY}_${MODEL_NAME}_${DATA_TAG}_${TIMESTAMP}"}
fi

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

echo "Starting experiment..."

if [ "$SCENARIO_TYPE" = "language_barrier" ]; then
    # Determine ratios to sweep: prefer BARRIER_RATIOS if provided; else single BARRIER_RATIO; else default set
    if [ -n "$BARRIER_RATIOS" ]; then
        RATIOS="$BARRIER_RATIOS"
    elif [ -n "$BARRIER_RATIO" ]; then
        RATIOS="$BARRIER_RATIO"
    else
        RATIOS="0.1 0.5 0.75 1"
    fi

    for R in $RATIOS; do
        echo "\n➡️  Running language barrier with ratio=$R"
        RATIO_LABEL=$(python - "$R" <<'PY'
import sys
r=float(sys.argv[1])
print(int(round(r*100)))
PY
)
        RUN_RESULTS_DIR=${RESULTS_DIR:-"results"}/exp_${SCENARIO_TYPE}_${COMMUNICATION_MODALITY}_mem${MEMORY_STRATEGY}_ratio${RATIO_LABEL}_${MODEL_NAME}_${DATA_TAG}_${TIMESTAMP}
        CUDA_VISIBLE_DEVICES=$GPU VLLM_PORT=$VLLM_PORT python scripts/run.py \
            --model_a $GLOBAL_MODEL_A \
            --model_b $GLOBAL_MODEL_B \
            --episodes_file $DATA_NAME\
            --scenario_type $SCENARIO_TYPE \
            --communication_modality $COMMUNICATION_MODALITY \
            --memory_strategy $MEMORY_STRATEGY \
            --results_dir $RUN_RESULTS_DIR \
            --barrier_ratio $R
    done
else
    CUDA_VISIBLE_DEVICES=$GPU VLLM_PORT=$VLLM_PORT python scripts/run.py \
        --model_a $GLOBAL_MODEL_A \
        --model_b $GLOBAL_MODEL_B \
        --episodes_file $DATA_NAME\
        --scenario_type $SCENARIO_TYPE \
        --communication_modality $COMMUNICATION_MODALITY \
        --memory_strategy $MEMORY_STRATEGY \
        --results_dir $RESULTS_DIR
fi

echo ""
echo "✅ Experiment completed!" 