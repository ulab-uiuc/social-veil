#!/bin/bash

# Social Decipher Experiment Runner Script
# =======================================
# This script provides commands for running the 3×3×2 factorial experimental design
# 
# Experimental Design:
# 1. Social Scenario Types: Normal, Language Barrier, Knowledge Barrier
# 2. Communication Modality: Text-only, Action-enabled, Text-Action Mix  
# 3. Memory Strategies: Memory OFF, Memory ON
# 
# Total Experiments: 3 × 3 × 2 = 18 conditions
# 
# Global Model Configuration - Change these to set models for all experiments
GLOBAL_MODEL_A="gpt-4o-mini"
GLOBAL_MODEL_B="Qwen/Qwen2.5-7B-Instruct"

echo "🧪 Social Decipher Experiment Runner"
echo "===================================="
echo "🤖 Global Models: Agent A = $GLOBAL_MODEL_A, Agent B = $GLOBAL_MODEL_B"
echo ""

# List available language barrier pairs and models
echo "📋 Available language barrier pairs:"
python run.py --list_pairs
echo ""

echo "📋 Available models for agent configuration:"
python run.py --list_models
echo ""

# =============================================================================
# RUN ALL EXPERIMENTS (3×3×2 factorial design)
# =============================================================================
echo "🚀 Running all 18 experiment conditions..."
echo "Command: python run.py --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B --run_all --episode_limit 5 --max_round 10"
echo ""

# Uncomment the line below to run all experiments
# python run.py --run_all --episode_limit 5 --max_round 10

# =============================================================================
# BASIC EXPERIMENT COMMANDS
# =============================================================================

echo "📊 Basic Experiment Commands:"
echo "============================="

# Normal Scenario
echo ""
echo "🧩 Normal scenario:"
echo "python run.py --scenario_type normal --communication_modality text_only --memory_strategy memory_off --episode_limit 3"

# Language Barrier
echo ""
echo "🌐 Language barrier:"
echo "python run.py --scenario_type language_barrier --communication_modality text_only --memory_strategy memory_off --episode_limit 3 --pair 0"

# Knowledge Barrier
echo ""
echo "🧠 Knowledge barrier:"
echo "python run.py --scenario_type knowledge_barrier --communication_modality text_only --memory_strategy memory_off --episode_limit 3"

# Global Model Configuration (Change models in this script)
echo ""
echo "🤖 Global model configuration (change GLOBAL_MODEL_A/B in this script):"
echo "python run.py --model_a \$GLOBAL_MODEL_A --model_b \$GLOBAL_MODEL_B --episode_limit 3 --start_vllm"
echo "python run.py --model_a \$GLOBAL_MODEL_A --model_b \$GLOBAL_MODEL_B --run_all --episode_limit 5 --start_vllm"

echo ""
echo "🚀 Ready-to-run commands with global models:"
echo "python run.py --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B --episode_limit 3 --start_vllm"
echo "python run.py --model_a $GLOBAL_MODEL_A --model_b $GLOBAL_MODEL_B --run_all --episode_limit 5 --start_vllm"

echo ""
echo "💾 Memory persistence:"
echo "python run.py --scenario_type normal --communication_modality text_only --memory_strategy memory_on --episode_limit 3 --memory_path ./memory_data"

echo ""
echo "🎯 For more information, run:"
echo "python run.py --help" 