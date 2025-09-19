#!/bin/bash

# Read model configuration from config.yaml
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_READER="-m social_decipher.utils.config_reader"

# Change to project root for poetry command
cd "$PROJECT_ROOT"

export GLOBAL_MODEL_A=$(poetry run python $CONFIG_READER models.model_a)
export GLOBAL_MODEL_B=$(poetry run python $CONFIG_READER models.model_b)
export DATA_NAME=$(poetry run python $CONFIG_READER data_dir)
# Derive a short tag from the data file (basename without extension), e.g., 'data/episode_hard.jsonl' -> 'episode_hard'
DATA_FILE_NAME=$(basename "$DATA_NAME")
DATA_TAG="${DATA_FILE_NAME%.*}"
export MODEL_NAME=$(poetry run python $CONFIG_READER models.served_model_name)
export GPU=$(poetry run python $CONFIG_READER models.gpu)
export VLLM_PORT=$(poetry run python $CONFIG_READER models.vllm_port)
export CONCURRENCY=${CONCURRENCY:-1}
export PARTNER_REPAIR_MODE=${PARTNER_REPAIR_MODE:-"false"}

TIMESTAMP=$(date +%m%d_%H%M)

# Add a suffix if repair mode is enabled
REPAIR_SUFFIX=""
if [[ "$PARTNER_REPAIR_MODE" == "true" ]]; then
  REPAIR_SUFFIX="_repair"
fi

# Default results dir (run.py will create subfolders for baseline/semantic/cultural/emotional)
export RESULTS_DIR=${RESULTS_DIR:-"results/exp_${MODEL_NAME}_${DATA_TAG}${REPAIR_SUFFIX}"}

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
    $( [[ "$PARTNER_REPAIR_MODE" == "true" ]] && echo "--partner-repair-prompt" )

echo ""
echo "✅ Experiment completed!" 