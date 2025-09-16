#!/bin/bash

# This script runs the LLaMA-Factory fine-tuning process on a pre-filtered,
# training-ready dataset. It skips all data collection and rating steps.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
SFT_DATASET_PATH="$1" # The first argument to the script is the path to the SFT data
EXPERIMENT_NAME="qwen-finetune-from-filtered-data"
AGENT_MODEL="models/Qwen2.5-0.5B-Instruct" # Make sure this matches your intended model

# --- Script Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/training_data/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
POLICY_UPDATER_SCRIPT="${PROJECT_ROOT}/social_decipher/training/policy_updater.py"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

cd "$PROJECT_ROOT"

# --- Argument Validation ---
if [ -z "$SFT_DATASET_PATH" ]; then
    echo "❌ Error: You must provide the path to the SFT dataset as the first argument."
    echo "   Usage: bash $0 /path/to/your/sft_data.json"
    exit 1
fi
if [ ! -f "$SFT_DATASET_PATH" ]; then
    echo "❌ Error: SFT dataset file not found at '$SFT_DATASET_PATH'"
    exit 1
fi

echo "===================================="
echo "🚀 Starting Fine-Tuning Run"
echo "===================================="
echo "SFT Dataset:    $SFT_DATASET_PATH"
echo "Agent Model:      $AGENT_MODEL"
echo "Output Dir:       $OUTPUT_DIR"
echo "Checkpoint Dir:   $CHECKPOINT_DIR"
echo "===================================="

# --- 1. Format the data for LLaMA-Factory ---
# We need to convert our conversation logs into the instruction/input/output format.
# A dedicated script would be best for this, but for now, we'll assume the manual_filter
# script will be updated to handle this formatting. Let's create a placeholder for the formatted data.
FORMATTED_DATA_PATH="${OUTPUT_DIR}/formatted_sft_data.json"
echo "Copying data to formatted path: $FORMATTED_DATA_PATH"
cp "$SFT_DATASET_PATH" "$FORMATTED_DATA_PATH"


# --- 2. Create LLaMA-Factory Config ---
# This step generates the llama_factory_config.yaml file required for training.
# We will create a small python script to call the existing function.
CONFIG_GENERATOR_SCRIPT=$(mktemp)
cat <<EOF > "$CONFIG_GENERATOR_SCRIPT"
from social_decipher.training.policy_updater import SocialPolicyUpdater
updater = SocialPolicyUpdater(output_dir="$OUTPUT_DIR")
updater.create_llama_factory_config(
    dataset_name="social_decipher_sft",
    model_name="$AGENT_MODEL",
    output_dir="$CHECKPOINT_DIR"
)
# Manually add the dataset path to the generated config
import yaml
config_path = f"{'$OUTPUT_DIR'}/llama_factory_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config['dataset_path'] = "$FORMATTED_DATA_PATH"
with open(config_path, 'w') as f:
    yaml.dump(config, f)
EOF

echo "\n🐍 Generating LLaMA-Factory config..."
python "$CONFIG_GENERATOR_SCRIPT"
rm "$CONFIG_GENERATOR_SCRIPT"
CONFIG_PATH="${OUTPUT_DIR}/llama_factory_config.yaml"
echo "   Config saved to $CONFIG_PATH"


# --- 3. Run Fine-Tuning ---
NUM_GPUS=$(nvidia-smi -L | wc -l)

CMD="llamafactory-cli train $CONFIG_PATH"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "\n🚀 Launching multi-GPU training ($NUM_GPUS GPUs)..."
    accelerate launch --config_file /path/to/accelerate/config.yaml --num_processes=$NUM_GPUS $(which llamafactory-cli) train $CONFIG_PATH
else
    echo "\n🚀 Launching single-GPU training..."
    $CMD
fi

echo "\n✅ Fine-tuning complete. Final model saved in $CHECKPOINT_DIR"