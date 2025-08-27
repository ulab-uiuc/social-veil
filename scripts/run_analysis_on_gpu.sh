#!/bin/bash

# Minimal GPU runner for barrier analysis
# Usage:
#   ./scripts/run_analysis_on_gpu.sh [GPU_ID] [--extra-args ...]
# Examples:
#   ./scripts/run_analysis_on_gpu.sh 0 --num_episodes 5
#   ./scripts/run_analysis_on_gpu.sh 1 --model Qwen/Qwen2.5-7B-Instruct

GPU_ID="$1"
shift || true

# Default to GPU 0 if not provided
if [ -z "$GPU_ID" ]; then
  GPU_ID=0
fi

# Export GPU selection
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Run the analysis (forwards any additional CLI args)
python analysis/run_analysis.py "$@"