#!/bin/bash

# Simple 4-GPU parallel experiment runner
export GLOBAL_MODEL_A="gpt-4o-mini"
export GLOBAL_MODEL_B="/mnt/data_from_server1/models/Qwen2.5-7B-Instruct"
export VLLM_PORT=8020

echo "🚀 Starting 4 parallel experiments on GPUs 1,2,3,4"

# Check if vLLM server is running
if ! curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "❌ vLLM server not running. Start it first: ./scripts/start_vllm_server.sh"
    exit 1
fi

# Run 4 different experiments in parallel
CUDA_VISIBLE_DEVICES=1 VLLM_PORT=$VLLM_PORT python run.py \
    --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B \
    --scenario_type normal --communication_modality text_only --memory_strategy off \
    --results_base_dir "../results/exp_normal_text_only_$(date +%m%d_%H%M)" &

CUDA_VISIBLE_DEVICES=2 VLLM_PORT=$VLLM_PORT python run.py \
    --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B \
    --scenario_type normal --communication_modality action_enabled --memory_strategy off \
    --results_base_dir "results/gpu1_normal_action_$(date +%m%d_%H%M)" &

CUDA_VISIBLE_DEVICES=3 VLLM_PORT=$VLLM_PORT python run.py \
    --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B \
    --scenario_type language_barrier --communication_modality text_only --memory_strategy off \
    --results_base_dir "results/gpu2_lang_text_$(date +%m%d_%H%M)" &

CUDA_VISIBLE_DEVICES=4 VLLM_PORT=$VLLM_PORT python run.py \
    --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B \
    --scenario_type knowledge_barrier --communication_modality text_only --memory_strategy off \
    --results_base_dir "results/gpu3_know_text_$(date +%m%d_%H%M)" &

# Wait for all experiments to complete
wait

echo "✅ All 4 experiments completed!"