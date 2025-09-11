#!/bin/bash

# Minimal GPU runner for barrier analysis
# Usage:
#   ./scripts/run_analysis_on_gpu.sh [GPU_ID] [--extra-args ...]
# Configure episode filenames below or override via env before calling.

# ---------- User-configurable episode paths ----------
BASELINE_EPISODES=${BASELINE_EPISODES:-"data/episode_all_neutralized.jsonl"}
SEMANTIC_EPISODES=${SEMANTIC_EPISODES:-"data/episodes_all_semantic.json"}
CULTURAL_EPISODES=${CULTURAL_EPISODES:-"data/episodes_all_cultural.json"}
EMOTIONAL_EPISODES=${EMOTIONAL_EPISODES:-"data/episodes_all_emotional.json"}
# ----------------------------------------------------

GPU_ID="$1"
shift || true

# Default to GPU 0 if not provided
if [ -z "$GPU_ID" ]; then
  GPU_ID=4
fi

# Export GPU selection
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Export episode paths for analysis loader
export BASELINE_EPISODES
export SEMANTIC_EPISODES
export CULTURAL_EPISODES
export EMOTIONAL_EPISODES

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Baseline:  $BASELINE_EPISODES"
echo "Semantic:  $SEMANTIC_EPISODES"
echo "Cultural:  $CULTURAL_EPISODES"
echo "Emotional: $EMOTIONAL_EPISODES"

# Run the analysis (forwards any additional CLI args)
python analysis/internal_state/run_analysis.py "$@"