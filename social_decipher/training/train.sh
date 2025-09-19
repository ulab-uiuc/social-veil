#!/bin/bash
# Main training script for Social-Decipher, implementing the SOTOPIA-π iterative loop.

set -euo pipefail

# --- GPU Reservation ---
# Start a background process that allocates a small amount of VRAM on each visible GPU.
# This "reserves" the GPUs and prevents other processes from claiming them during the
# data collection phase, which does not keep the GPUs occupied on its own.
echo "INFO: Starting GPU placeholder to reserve devices specified by CUDA_VISIBLE_DEVICES..."
python -m social_decipher.utils.gpu_placeholder &
PLACEHOLDER_PID=$!
echo "INFO: GPU placeholder started with PID: $PLACEHOLDER_PID"

# Ensure the placeholder is killed automatically when this script exits for any reason.
trap "echo 'INFO: Shutting down GPU placeholder (PID: $PLACEHOLDER_PID)...'; kill $PLACEHOLDER_PID 2>/dev/null; exit" EXIT SIGINT SIGTERM

# --- CONFIGURATION ---
# Experiment
export EXPERIMENT_NAME="sotopia-pi-v1"
export WANDB_PROJECT="social-decipher"
export WANDB_ENTITY="kxtechds"
export NUM_IMPROVE_STEPS=20

# Data Preparation
export EPISODES_FILE="data/episode_all_neutralized.jsonl"
export EPISODE_LIMIT=1 # Number of unique scenarios per step
export BASE_OUTPUT_DIR="training_output/${EXPERIMENT_NAME}"

# SFT Training - Initial Model
# This is the starting point for the first iteration. Subsequent steps will use the checkpoint from the previous step.
CONFIG_READER_CMD="python3 -m social_decipher.utils.config_reader"
AGENT_MODEL_PATH=$($CONFIG_READER_CMD training_models.agent_model)
PARTNER_MODEL_PATH=$($CONFIG_READER_CMD training_models.partner_model)
EVALUATOR_MODEL=$($CONFIG_READER_CMD training_models.evaluator_model)
EXPERT_MODEL=$($CONFIG_READER_CMD training_models.expert_model)

export AGENT_OPENAI_API_KEY=$($CONFIG_READER_CMD AGENT_OPENAI_API_KEY)
export EVALUATOR_OPENAI_API_KEY=$($CONFIG_READER_CMD EVALUATOR_OPENAI_API_KEY)
export OPENAI_API_KEY=${AGENT_OPENAI_API_KEY}

export TEMPLATE_PATH="configs/qwen2.5-7b.jinja"
export TRAIN_BATCH_SIZE=4

# --- EXECUTION LOOP ---

for (( step=0; step<$NUM_IMPROVE_STEPS; step++ )); do
    echo "================================================="
    echo "🚀 STARTING IMPROVEMENT STEP $((step + 1)) / $NUM_IMPROVE_STEPS"
    echo "================================================="
    
    # --- Set Epochs and Data Mode based on Sotopia-pi Strategy ---
    if [ "$step" -eq 0 ]; then
        SFT_EPOCHS=20 # Longer training for the initial BC phase
        DATA_MODE="bc_and_sr"
        echo "INFO: Step 0, setting SFT epochs to 20 and data mode to 'bc_and_sr'."
    else
        SFT_EPOCHS=5  # Shorter training for subsequent SR phases
        DATA_MODE="sr_only"
        echo "INFO: Step > 0, setting SFT epochs to 5 and data mode to 'sr_only'."
    fi

    STEP_OUTPUT_DIR="${BASE_OUTPUT_DIR}/step_${step}"
    SFT_DATA_FILE="${STEP_OUTPUT_DIR}/sft_data.json"
    CHECKPOINT_DIR="${STEP_OUTPUT_DIR}/checkpoints"
    mkdir -p "$STEP_OUTPUT_DIR"
    mkdir -p "$CHECKPOINT_DIR"

    echo "Current Agent Model: ${AGENT_MODEL_PATH}"

    # --- STAGE 1: Preparing SFT Data ---
    echo "\n🔥 Preparing SFT Data for Step ${step}..."
    python -m social_decipher.training.prepare_data \
        --episodes_file "$EPISODES_FILE" \
        --episode_limit "$EPISODE_LIMIT" \
        --output_file "$SFT_DATA_FILE" \
        --agent_model "$AGENT_MODEL_PATH" \
        --partner_model "$PARTNER_MODEL_PATH" \
        --expert_model "$EXPERT_MODEL" \
        --evaluator_model "$EVALUATOR_MODEL" \
        --use_barrier_episodes \
        --data_collection_mode "$DATA_MODE"

    echo "✅ SFT data prepared at ${SFT_DATA_FILE}"

    if [ ! -f "$SFT_DATA_FILE" ]; then
        echo "❌ ERROR: SFT data file not found at ${SFT_DATA_FILE}. Cannot proceed to next step."
        exit 1
    fi

    # --- STAGE 2: Launching SFT Training ---
    echo "\n🔥 Launching SFT Training for Step ${step}..."
    # We must run the script from its directory for relative paths to work
    cd scripts
    
    # Pass absolute SFT path and explicit checkpoint dir
    SFT_DATA_FILE_ABS=$(pwd)/../$SFT_DATA_FILE
    CHECKPOINT_DIR_ABS=$(pwd)/../$CHECKPOINT_DIR
    bash ./train_sft.sh "$SFT_DATA_FILE_ABS" "$CHECKPOINT_DIR_ABS"
    
    cd ..
    
    # --- STAGE 3: Update model path for next iteration ---
    # Find the 'best-checkpoint' to use for the next SR round
    BEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -type d -name "best-checkpoint" | head -n 1)
    if [ -z "$BEST_CHECKPOINT" ]; then
        echo "❌ ERROR: Could not find 'best-checkpoint' in ${CHECKPOINT_DIR}. Cannot proceed to next step."
        exit 1
    fi
    
    AGENT_MODEL_PATH="$BEST_CHECKPOINT"
    echo "\n✅ Step ${step} complete. New agent model for next step is: ${AGENT_MODEL_PATH}"
done

echo "\n🎉 SOTOPIA-π training loop finished successfully!"