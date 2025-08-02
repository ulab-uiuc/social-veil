#!/bin/bash

# Script to start vLLM server for local model serving
# Usage: ./start_local_model.sh [model_name] [port] [gpu]

set -e

# Default values
DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PORT=8000
DEFAULT_GPU=0
DEFAULT_TEMPLATE="configs/qwen2.5-7b.jinja"

# Parse arguments
MODEL_PATH=${1:-$DEFAULT_MODEL}
PORT=${2:-$DEFAULT_PORT}
GPU=${3:-$DEFAULT_GPU}
TEMPLATE_PATH=${4:-$DEFAULT_TEMPLATE}

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Resolve template path
if [[ "$TEMPLATE_PATH" != /* ]]; then
    TEMPLATE_PATH="$PROJECT_ROOT/$TEMPLATE_PATH"
fi

echo "🚀 Starting vLLM server for local model serving"
echo "================================================"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU: $GPU"
echo "Template: $TEMPLATE_PATH"
echo ""

# Check if template exists
if [[ ! -f "$TEMPLATE_PATH" ]]; then
    echo "⚠️  Warning: Template file not found at $TEMPLATE_PATH"
    echo "   Continuing without template..."
    TEMPLATE_PATH=""
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU

# Build vLLM command
VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --port "$PORT"
    --max-model-len 4096
    --tensor-parallel-size 1
)

# Add template if it exists
if [[ -n "$TEMPLATE_PATH" && -f "$TEMPLATE_PATH" ]]; then
    VLLM_CMD+=(--chat-template "$TEMPLATE_PATH")
fi

echo "Command: ${VLLM_CMD[*]}"
echo ""
echo "Starting server... (Press Ctrl+C to stop)"
echo ""

# Start the server
"${VLLM_CMD[@]}" 