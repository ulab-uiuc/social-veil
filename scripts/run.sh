#!/bin/bash

# --- Preamble: Check for dependencies ---
if ! command -v yq &> /dev/null
then
    echo "❌ Error: 'yq' is not installed or not in your PATH."
    echo "   Please install it to proceed. For example, on Linux/macOS with pip:"
    echo "   pip install yq"
    exit 1
fi

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/configs/config.yaml"

# Read parameters from config file using yq
export GLOBAL_MODEL_A=$(yq '.models.model_a' "$CONFIG_FILE" | tr -d '"')
export GLOBAL_MODEL_B=$(yq '.models.model_b' "$CONFIG_FILE" | tr -d '"')
export DATA_NAME=$(yq '.data_dir' "$CONFIG_FILE" | tr -d '"')
export MODEL_NAME=$(yq '.models.served_model_name' "$CONFIG_FILE" | tr -d '"')
export GPU=$(yq '.models.gpu' "$CONFIG_FILE" | tr -d '"')
export VLLM_PORT=$(yq '.models.vllm_port' "$CONFIG_FILE")

# --- Derived Variables ---
# Derive a short tag from the data file (basename without extension)
DATA_FILE_NAME=$(basename "$DATA_NAME")
DATA_TAG="${DATA_FILE_NAME%.*}"

export CONCURRENCY=${CONCURRENCY:-1}
export PARTNER_REPAIR_MODE=${PARTNER_REPAIR_MODE:-"false"}
export PARTNER_COT_MODE=${PARTNER_COT_MODE:-"false"}

TIMESTAMP=$(date +%m%d_%H%M)

# Add a suffix based on the enabled mode
MODE_SUFFIX=""
if [[ "$PARTNER_COT_MODE" == "true" ]]; then
  MODE_SUFFIX="_cot"
elif [[ "$PARTNER_REPAIR_MODE" == "true" ]]; then
  MODE_SUFFIX="_repair"
fi

# Default results dir (run.py will create subfolders for baseline/semantic/cultural/emotional)
export RESULTS_DIR=${RESULTS_DIR:-"results/exp_${MODEL_NAME}_${DATA_TAG}${MODE_SUFFIX}"}

echo "===================================="
echo "🧪 Running Social Agent Experiment"
echo "===================================="
echo "Agent A: $GLOBAL_MODEL_A"
echo "Agent B: $GLOBAL_MODEL_B"
echo "GPU: $GPU"
echo ""

# Check vLLM server only if Agent B looks like a local/HF model path (contains a slash)
if [[ "$GLOBAL_MODEL_B" == *"/"* ]]; then
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
else
  echo "Skipping vLLM health check (Agent B = $GLOBAL_MODEL_B)"
fi

echo "Starting experiment..."

CUDA_VISIBLE_DEVICES=$GPU VLLM_PORT=$VLLM_PORT python scripts/run.py --disable-mcq \
    --model_a $GLOBAL_MODEL_A \
    --model_b $GLOBAL_MODEL_B \
    --episodes_file $DATA_NAME \
    --results_dir $RESULTS_DIR \
    --resume \
    --concurrency $CONCURRENCY \
    $( [[ "$PARTNER_COT_MODE" == "true" ]] && echo "--partner-cot-prompt" ) \
    $( [[ "$PARTNER_REPAIR_MODE" == "true" ]] && echo "--partner-repair-prompt" )

echo ""
echo "✅ Experiment completed!" 