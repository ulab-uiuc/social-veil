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
# Note: Episodes are loaded from JSONL files. Use --episode_limit to limit processing.

echo "🧪 Social Decipher Experiment Runner"
echo "===================================="

# List available language barrier pairs
echo "📋 Available language barrier pairs:"
python run.py --list_pairs
echo ""

# =============================================================================
# RUN ALL EXPERIMENTS (3×3×2 factorial design)
# =============================================================================
echo "🚀 Running all 18 experiment conditions..."
echo "Command: python run.py --run_all --episode_limit 5 --max_round 10"
echo ""

# Uncomment the line below to run all experiments
# python run.py --run_all --episode_limit 5 --max_round 10

# =============================================================================
# INDIVIDUAL EXPERIMENT COMMANDS
# =============================================================================

echo "📊 Individual Experiment Commands:"
echo "=================================="

# Normal Scenario Experiments
echo ""
echo "🧩 NORMAL SCENARIO EXPERIMENTS:"
echo "python run.py --scenario_type normal --communication_modality text_only --memory_strategy memory_off --episode_limit 3"
echo "python run.py --scenario_type normal --communication_modality text_only --memory_strategy memory_on --episode_limit 3"
echo "python run.py --scenario_type normal --communication_modality action_enabled --memory_strategy memory_off --episode_limit 3"
echo "python run.py --scenario_type normal --communication_modality action_enabled --memory_strategy memory_on --episode_limit 3"
echo "python run.py --scenario_type normal --communication_modality text_action_mix --memory_strategy memory_off --episode_limit 3"
echo "python run.py --scenario_type normal --communication_modality text_action_mix --memory_strategy memory_on --episode_limit 3"

# Language Barrier Experiments
echo ""
echo "🌐 LANGUAGE BARRIER EXPERIMENTS:"
echo "python run.py --scenario_type language_barrier --communication_modality text_only --memory_strategy memory_off --episode_limit 3 --pair 0"
echo "python run.py --scenario_type language_barrier --communication_modality text_only --memory_strategy memory_on --episode_limit 3 --pair 0"
echo "python run.py --scenario_type language_barrier --communication_modality action_enabled --memory_strategy memory_off --episode_limit 3 --pair 0"
echo "python run.py --scenario_type language_barrier --communication_modality action_enabled --memory_strategy memory_on --episode_limit 3 --pair 0"
echo "python run.py --scenario_type language_barrier --communication_modality text_action_mix --memory_strategy memory_off --episode_limit 3 --pair 0"
echo "python run.py --scenario_type language_barrier --communication_modality text_action_mix --memory_strategy memory_on --episode_limit 3 --pair 0"

# Knowledge Barrier Experiments
echo ""
echo "🧠 KNOWLEDGE BARRIER EXPERIMENTS:"
echo "python run.py --scenario_type knowledge_barrier --communication_modality text_only --memory_strategy memory_off --episode_limit 3"
echo "python run.py --scenario_type knowledge_barrier --communication_modality text_only --memory_strategy memory_on --episode_limit 3"
echo "python run.py --scenario_type knowledge_barrier --communication_modality action_enabled --memory_strategy memory_off --episode_limit 3"
echo "python run.py --scenario_type knowledge_barrier --communication_modality action_enabled --memory_strategy memory_on --episode_limit 3"
echo "python run.py --scenario_type knowledge_barrier --communication_modality text_action_mix --memory_strategy memory_off --episode_limit 3"
echo "python run.py --scenario_type knowledge_barrier --communication_modality text_action_mix --memory_strategy memory_on --episode_limit 3"

# =============================================================================
# EXPERIMENT SUBSETS
# =============================================================================
echo ""
echo "📋 EXPERIMENT SUBSETS:"
echo "======================"

# Run only normal scenarios
echo ""
echo "🧩 Normal scenarios only (6 experiments):"
echo "python run.py --run_all --experiment_subset normal --episode_limit 3"

# Run only language barrier scenarios  
echo ""
echo "🌐 Language barrier scenarios only (6 experiments):"
echo "python run.py --run_all --experiment_subset language_barrier --episode_limit 3 --pair 0"

# Run only knowledge barrier scenarios
echo ""
echo "🧠 Knowledge barrier scenarios only (6 experiments):"
echo "python run.py --run_all --experiment_subset knowledge_barrier --episode_limit 3"

# Run multiple scenario types
echo ""
echo "🔀 Multiple scenario types:"
echo "python run.py --run_all --experiment_subset normal knowledge_barrier --episode_limit 3"

# =============================================================================
# MEMORY PERSISTENCE EXPERIMENTS
# =============================================================================
echo ""
echo "💾 MEMORY PERSISTENCE EXPERIMENTS:"
echo "=================================="

# Run with memory persistence
echo ""
echo "💾 Run all experiments with memory persistence:"
echo "python run.py --run_all --episode_limit 5 --memory_path ./agent_memories"

# Run specific experiment with memory persistence
echo ""
echo "💾 Run specific experiment with memory persistence:"
echo "python run.py --scenario_type knowledge_barrier --communication_modality text_action_mix --memory_strategy memory_on --episode_limit 5 --memory_path ./agent_memories"

# =============================================================================
# DRY RUN AND TESTING
# =============================================================================
echo ""
echo "🔍 DRY RUN AND TESTING:"
echo "======================="

# Dry run to see experiment configuration
echo ""
echo "🔍 Dry run to see experiment configuration:"
echo "python run.py --run_all --dry_run"

# Dry run for specific experiment
echo ""
echo "🔍 Dry run for specific experiment:"
echo "python run.py --scenario_type knowledge_barrier --communication_modality text_action_mix --memory_strategy memory_on --dry_run"

# =============================================================================
# QUICK TEST COMMANDS
# =============================================================================
echo ""
echo "⚡ QUICK TEST COMMANDS:"
echo "======================"

# Quick single episode test
echo ""
echo "⚡ Quick single episode test:"
echo "python run.py --scenario_type normal --communication_modality text_only --memory_strategy memory_off --episode_limit 1 --max_round 5"

# Quick test with different models
echo ""
echo "⚡ Quick test with different models:"
echo "python run.py --scenario_type normal --communication_modality text_only --memory_strategy memory_off --episode_limit 1 --model gpt-4o-mini"
echo "python run.py --scenario_type normal --communication_modality text_only --memory_strategy memory_off --episode_limit 1 --model gpt-4o"

# =============================================================================
# LEGACY COMMANDS (for backward compatibility)
# =============================================================================
echo ""
echo "📜 LEGACY COMMANDS (for backward compatibility):"
echo "================================================"
echo "# python run.py --max_round 10 --list_pairs"
echo "# python run.py --encryption --nature_language --pair 2"
echo "# python run.py --encryption"
echo "# python run.py --episode_limit 3 --max_round 5 --encryption_enabled --nature_language --action --pair 2"

echo ""
echo "🎯 To run all experiments, uncomment the line below:"
echo "# python run.py --run_all --episode_limit 5 --max_round 10"
echo ""
echo "✅ Script completed. Choose your experiment command above!" 