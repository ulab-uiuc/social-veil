#!/bin/bash

# Sotopia-π Style Training Script for Social-Decipher
# Following the training methodology from Sotopia-π but adapted for barrier scenarios

set -e # Exit on any error

################################################################################
### USER CONFIGURATION - EDIT THIS SECTION TO CONFIGURE YOUR TRAINING RUN    ###
################################################################################

# --- Wandb Authentication (MANDATORY for shared SSH environments) ---
export WANDB_API_KEY="606b4f0ccb0c2a098157d3055631930177d1aeac"
WANDB_ENTITY="kxtechds-university-of-illinois-urbana-champaign"
WANDB_PROJECT="social-decipher"

# --- Experiment Setup ---
# A name for the overall experiment. This will be used for checkpoint folders.
EXPERIMENT_NAME="qwen-finetune-barrier-run"
# A specific name for this particular run. Defaults to the experiment name + timestamp.
WANDB_RUN_NAME="${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M)"

# --- Model Configuration ---
AGENT_MODEL="models/Qwen2.5-0.5B-Instruct"
# The powerful model used to judge conversations and provide reward signals.
EVALUATOR_MODEL="gpt-4o"
EXPERT_MODEL="gpt-4.1"

# --- API Keys (read from config.yaml) ---
CONFIG_READER_CMD="python3 -m social_decipher.utils.config_reader"
export AGENT_OPENAI_API_KEY=$($CONFIG_READER_CMD AGENT_OPENAI_API_KEY)
export EVALUATOR_OPENAI_API_KEY=$($CONFIG_READER_CMD EVALUATOR_OPENAI_API_KEY)
export OPENAI_API_KEY=$AGENT_OPENAI_API_KEY

# --- Data & Training Loop Configuration ---
# The starting dataset of scenarios. Use a small sample for testing.
# For testing: "data/episode_test_sample.jsonl"
# For full run: "data/episode_all_neutralized.jsonl"
EPISODES_FILE="data/episode_test_sample.jsonl"

# How many times to repeat the "data collection -> training" cycle. 1-2 is good for testing.
NUM_IMPROVE_STEPS=1

# How many conversations to generate for each scenario. 1 is good for testing.
CONVERSATIONS_PER_EPISODE=1

# --- Advanced Settings ---
# Set to 'true' to skip BC data collection if bc_data.json already exists.
LOAD_EXISTING_DATA=true
# Set to 'true' to include the specialized barrier datasets in training.
USE_BARRIER_EPISODES=true
# The scoring logic to use for filtering conversations. "custom_barrier_focused" is the recommended default.
SCORING_STRATEGY="custom_barrier_focused"
# The quality score threshold for filtering conversations.
QUALITY_THRESHOLD=6.0
# The number of top conversations to keep per scenario type.
FILTER_TOP_K=2
# Which barrier types to include when USE_BARRIER_EPISODES is true.
BARRIER_TYPES="semantic cultural emotional"
OUTPUT_DIR="training_data"
CHECKPOINT_DIR="checkpoints"

################################################################################
### SCRIPT EXECUTION - NO NEED TO EDIT BELOW THIS LINE                       ###
################################################################################

# Environment setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Function to check if required files exist
check_requirements() {
    print_info "Checking requirements..."
    
    # Check if episodes file exists
    if [ ! -f "$EPISODES_FILE" ]; then
        print_error "Episodes file not found: $EPISODES_FILE"
        exit 1
    fi
    
    # Check if barrier episodes exist (if requested)
    if [ "$USE_BARRIER_EPISODES" = true ]; then
        for barrier_type in $BARRIER_TYPES; do
            barrier_file="data/episodes_${barrier_type}.json"
            if [ ! -f "$barrier_file" ]; then
                print_warning "Barrier episodes file not found: $barrier_file"
                print_info "Run barrier creation first or set USE_BARRIER_EPISODES=false"
            fi
        done
    fi
    
    # Check if Python dependencies are available
    python3 -c "import openai, yaml, json" 2>/dev/null || {
        print_error "Required Python packages not found. Please install: pip install openai pyyaml"
        exit 1
    }
    
    print_success "Requirements check passed"
}

# Function to setup directories
setup_directories() {
    print_info "Setting up directories..."
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "$OUTPUT_DIR/policy_updates"
    mkdir -p "$OUTPUT_DIR/training_data"
    mkdir -p "logs"
    
    print_success "Directories created"
}

# Function to run training
run_training() {
    print_info "Starting training pipeline..."
    print_info "Experiment: $EXPERIMENT_NAME"
    print_info "Improvement steps: $NUM_IMPROVE_STEPS"
    print_info "Episodes file: $EPISODES_FILE"
    print_info "Output directory: $OUTPUT_DIR"
    
    # Build training command
    TRAIN_CMD="python3 -m social_decipher.training.train"
    TRAIN_CMD="$TRAIN_CMD --experiment_name $EXPERIMENT_NAME"
    TRAIN_CMD="$TRAIN_CMD --num_improve_steps $NUM_IMPROVE_STEPS"
    TRAIN_CMD="$TRAIN_CMD --episodes_file $EPISODES_FILE"
    TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
    TRAIN_CMD="$TRAIN_CMD --checkpoint_dir $CHECKPOINT_DIR"
    TRAIN_CMD="$TRAIN_CMD --expert_model $EXPERT_MODEL"
    TRAIN_CMD="$TRAIN_CMD --agent_model $AGENT_MODEL"
    TRAIN_CMD="$TRAIN_CMD --evaluator_model $EVALUATOR_MODEL"
    TRAIN_CMD="$TRAIN_CMD --conversations_per_episode $CONVERSATIONS_PER_EPISODE"
    TRAIN_CMD="$TRAIN_CMD --quality_threshold $QUALITY_THRESHOLD"
    TRAIN_CMD="$TRAIN_CMD --filter_top_k $FILTER_TOP_K"
    TRAIN_CMD="$TRAIN_CMD --scoring_strategy $SCORING_STRATEGY"
    
    # Add wandb arguments
    TRAIN_CMD="$TRAIN_CMD --wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        TRAIN_CMD="$TRAIN_CMD --wandb_entity $WANDB_ENTITY"
    fi
    if [ -n "$WANDB_RUN_NAME" ]; then
        TRAIN_CMD="$TRAIN_CMD --wandb_run_name $WANDB_RUN_NAME"
    fi
    
    if [ "$LOAD_EXISTING_DATA" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --load_existing_data"
    fi

    if [ "$USE_BARRIER_EPISODES" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --use_barrier_episodes"
        TRAIN_CMD="$TRAIN_CMD --barrier_types $BARRIER_TYPES"
    fi
    
    print_info "Running command: $TRAIN_CMD"
    
    # Run training with logging
    LOG_FILE="logs/training_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
    $TRAIN_CMD 2>&1 | tee "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        print_success "Training completed successfully!"
        print_info "Log file: $LOG_FILE"
    else
        print_error "Training failed! Check log file: $LOG_FILE"
        exit 1
    fi
}

# Function to run data preprocessing only
run_preprocessing() {
    print_info "Running data preprocessing only..."
    
    # Check if input files exist
    CONVERSATIONS_FILE="$OUTPUT_DIR/conversations.json"
    RATINGS_FILE="$OUTPUT_DIR/ratings.json"
    
    if [ ! -f "$CONVERSATIONS_FILE" ] || [ ! -f "$RATINGS_FILE" ]; then
        print_error "Input files not found. Run training first to generate conversations and ratings."
        exit 1
    fi
    
    # Run preprocessing
    python3 -m social_decipher.training.data_preprocessing \
        --input_file "$CONVERSATIONS_FILE" \
        --ratings_file "$RATINGS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --quality_threshold "$QUALITY_THRESHOLD" \
        --balance_dataset
    
    print_success "Data preprocessing completed!"
}

# Function to show help
show_help() {
    echo "Sotopia-π Style Training Script for Social-Decipher"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h              Show this help message"
    echo "  --preprocess-only       Run data preprocessing only"
    echo "  --check-requirements    Check requirements and exit"
    echo ""
    echo "Environment Variables:"
    echo "  EXPERIMENT_NAME         Name of the training experiment (default: social_decipher_barrier_training)"
    echo "  NUM_IMPROVE_STEPS       Number of improvement steps (default: 3)"
    echo "  EPISODES_FILE           Path to episodes JSONL file (default: data/episode_sample.jsonl)"
    echo "  OUTPUT_DIR              Output directory (default: training_data)"
    echo "  CHECKPOINT_DIR          Checkpoint directory (default: checkpoints)"
    echo "  EXPERT_MODEL            Expert model for BC (default: gpt-4o)"
    echo "  AGENT_MODEL             Agent model for SR (default: gpt-4o-mini)"
    echo "  EVALUATOR_MODEL         Evaluator model (default: gpt-4o)"
    echo "  CONVERSATIONS_PER_EPISODE  Conversations per episode (default: 3)"
    echo "  QUALITY_THRESHOLD       Quality threshold (default: 6.0)"
    echo "  FILTER_TOP_K            Top-k filtering (default: 2)"
    echo "  SCORING_STRATEGY        Scoring strategy to use (default: custom_barrier_focused)"
    echo "  LOAD_EXISTING_DATA      Set to 'true' to reuse existing bc_data.json (default: false)"
    echo "  USE_BARRIER_EPISODES    Use barrier episodes (default: false)"
    echo "  BARRIER_TYPES           Barrier types to include (default: semantic cultural emotional)"
    echo "  WANDB_PROJECT           Wandb project name (default: social-decipher)"
    echo "  WANDB_ENTITY            Wandb entity name (optional)"
    echo "  WANDB_RUN_NAME          Wandb run name (optional, defaults to experiment name)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with default settings"
    echo "  $0 --preprocess-only                  # Run preprocessing only"
    echo "  EXPERIMENT_NAME=my_exp $0             # Run with custom experiment name"
    echo "  USE_BARRIER_EPISODES=true $0          # Run with barrier episodes"
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --preprocess-only)
                PREPROCESS_ONLY=true
                shift
                ;;
            --check-requirements)
                check_requirements
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check requirements
    check_requirements
    
    # Setup directories
    setup_directories
    
    # Run preprocessing only if requested
    if [ "$PREPROCESS_ONLY" = true ]; then
        run_preprocessing
    else
        # Run full training pipeline
        run_training
    fi
}

# Run main function
main "$@"
