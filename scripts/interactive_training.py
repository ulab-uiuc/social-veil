#!/usr/bin/env python3
"""
Interactive Training Pipeline for Social-Decipher
Implements Sotopia-π inspired training methodology for barrier-aware social intelligence.

Usage:
    python scripts/interactive_training.py --mode collect_bc
    python scripts/interactive_training.py --mode collect_sr  
    python scripts/interactive_training.py --mode rate_conversations
    python scripts/interactive_training.py --mode prepare_training
    python scripts/interactive_training.py --mode full_pipeline
"""

import argparse
import os
import sys
import json
from typing import List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from social_decipher.training.data_collector import BarrierDataCollector
from social_decipher.training.conversation_rater import ConversationRater  
from social_decipher.training.policy_updater import SocialPolicyUpdater


def load_episodes(episodes_file: str) -> List[dict]:
    """Load episode data from JSON or JSONL file"""
    print(f"📁 Loading episodes from {episodes_file}")
    
    if episodes_file.endswith('.jsonl'):
        episodes = []
        with open(episodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
    else:
        with open(episodes_file, 'r', encoding='utf-8') as f:
            episodes = json.load(f)
            
    print(f"✅ Loaded {len(episodes)} episodes")
    return episodes


def collect_behavior_cloning_data(args):
    """Step 1: Collect expert demonstration data (BC)"""
    print("\n🎓 === STEP 1: Collecting Behavior Cloning Data ===")
    
    episodes = load_episodes(args.episodes_file)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        print(f"🔢 Limited to {len(episodes)} episodes")
    
    collector = BarrierDataCollector(
        expert_model=args.expert_model,
        agent_model=args.agent_model,
        evaluator_model=args.evaluator_model,
        output_dir=args.output_dir
    )
    
    bc_conversations = collector.collect_behavior_cloning_data(
        episodes=episodes,
        num_conversations_per_episode=args.bc_conversations_per_episode,
        max_rounds=args.max_rounds
    )
    
    print(f"✅ Collected {len(bc_conversations)} BC conversations")
    

def collect_self_reinforcement_data(args):
    """Step 2: Collect self-play data (SR)"""
    print("\n🤖 === STEP 2: Collecting Self-Reinforcement Data ===")
    
    episodes = load_episodes(args.episodes_file)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        print(f"🔢 Limited to {len(episodes)} episodes")
    
    collector = BarrierDataCollector(
        expert_model=args.expert_model,
        agent_model=args.agent_model,
        evaluator_model=args.evaluator_model,
        output_dir=args.output_dir
    )
    
    sr_conversations = collector.collect_self_reinforcement_data(
        episodes=episodes,
        num_conversations_per_episode=args.sr_conversations_per_episode,
        max_rounds=args.max_rounds
    )
    
    print(f"✅ Collected {len(sr_conversations)} SR conversations")


def rate_conversations(args):
    """Step 3: Rate conversation quality"""
    print("\n🎯 === STEP 3: Rating Conversation Quality ===")
    
    # Load conversations
    collector = BarrierDataCollector(output_dir=args.output_dir)
    collector.load_conversations("bc_data.json", "sr_data.json")
    
    bc_conversations, sr_conversations = collector.get_all_conversations()
    all_conversations = bc_conversations + sr_conversations
    
    if not all_conversations:
        print("❌ No conversations found to rate. Run data collection first.")
        return
        
    print(f"📊 Rating {len(all_conversations)} conversations...")
    
    # Rate conversations
    rater = ConversationRater(
        model=args.evaluator_model,
        temperature=args.rating_temperature
    )
    
    ratings = rater.rate_conversations(
        conversations=all_conversations,
        quality_threshold=args.quality_threshold
    )
    
    # Save ratings
    ratings_file = os.path.join(args.output_dir, "conversation_ratings.json")
    rater.save_ratings(ratings, ratings_file)
    
    # Analyze ratings
    analysis = rater.analyze_ratings(ratings)
    
    print(f"✅ Rated conversations with {analysis['positive_rate']:.1%} positive rate")


def prepare_training_data(args):
    """Step 4: Prepare fine-tuning data"""
    print("\n🔄 === STEP 4: Preparing Training Data ===")
    
    # Load conversations and ratings
    collector = BarrierDataCollector(output_dir=args.output_dir)
    collector.load_conversations("bc_data.json", "sr_data.json")
    
    bc_conversations, sr_conversations = collector.get_all_conversations()
    all_conversations = bc_conversations + sr_conversations
    
    # Load ratings
    ratings_file = os.path.join(args.output_dir, "conversation_ratings.json")
    if not os.path.exists(ratings_file):
        print("❌ No ratings found. Run rating step first.")
        return
        
    with open(ratings_file, 'r', encoding='utf-8') as f:
        ratings_data = json.load(f)
    
    from social_decipher.training.conversation_rater import ConversationRating
    ratings = [ConversationRating(**r) for r in ratings_data]
    
    # Prepare training data
    updater = SocialPolicyUpdater(output_dir=args.output_dir)
    
    training_examples = updater.prepare_training_data(
        conversations=all_conversations,
        ratings=ratings,
        focus_on_agent_b=args.focus_on_agent_b,
        min_quality_score=args.quality_threshold
    )
    
    # Format for LLaMA-Factory
    formatted_data = updater.format_for_llama_factory(training_examples)
    
    # Save training data
    updater.save_training_data(formatted_data, "barrier_training_data.json")
    
    # Create LLaMA-Factory config
    config = updater.create_llama_factory_config(
        dataset_name="barrier_social_intelligence",
        model_name=args.model_to_train,
        output_dir="saves/barrier_social_agent"
    )
    
    print(f"✅ Prepared {len(formatted_data)} training samples")
    

def run_full_pipeline(args):
    """Run complete training pipeline"""
    print("\n🚀 === RUNNING FULL INTERACTIVE TRAINING PIPELINE ===")
    
    # Step 1: Collect BC data
    collect_behavior_cloning_data(args)
    
    # Step 2: Collect SR data  
    collect_self_reinforcement_data(args)
    
    # Step 3: Rate conversations
    rate_conversations(args)
    
    # Step 4: Prepare training data
    prepare_training_data(args)
    
    print("\n🎉 === PIPELINE COMPLETE ===")
    print(f"📁 All outputs saved to: {args.output_dir}")
    print("\n📋 Next steps:")
    print("1. Review training data quality in barrier_training_data.json")
    print("2. Set up LLaMA-Factory with the generated config")
    print("3. Run fine-tuning: llamafactory-cli train llama_factory_config.yaml")
    print("4. Evaluate improved model on barrier scenarios")


def main():
    parser = argparse.ArgumentParser(description="Interactive Training Pipeline for Social-Decipher")
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["collect_bc", "collect_sr", "rate_conversations", "prepare_training", "full_pipeline"],
        required=True,
        help="Training pipeline step to run"
    )
    
    # Data inputs
    parser.add_argument(
        "--episodes_file", 
        type=str, 
        default="data/episode_all.jsonl",
        help="Path to episodes file (JSON or JSONL)"
    )
    
    parser.add_argument(
        "--max_episodes", 
        type=int, 
        default=None,
        help="Maximum number of episodes to process (default: all)"
    )
    
    # Model configuration
    parser.add_argument(
        "--expert_model", 
        type=str, 
        default="gpt-4o",
        help="Expert model for BC data collection"
    )
    
    parser.add_argument(
        "--agent_model", 
        type=str, 
        default="gpt-4o-mini",
        help="Current agent model for SR data collection"
    )
    
    parser.add_argument(
        "--evaluator_model", 
        type=str, 
        default="gpt-4o",
        help="Model for conversation evaluation and rating"
    )
    
    parser.add_argument(
        "--model_to_train", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to fine-tune (for LLaMA-Factory config)"
    )
    
    # Data collection parameters
    parser.add_argument(
        "--bc_conversations_per_episode", 
        type=int, 
        default=3,
        help="Number of BC conversations per episode"
    )
    
    parser.add_argument(
        "--sr_conversations_per_episode", 
        type=int, 
        default=5,
        help="Number of SR conversations per episode"
    )
    
    parser.add_argument(
        "--max_rounds", 
        type=int, 
        default=20,
        help="Maximum conversation rounds"
    )
    
    # Rating parameters
    parser.add_argument(
        "--quality_threshold", 
        type=float, 
        default=6.0,
        help="Minimum quality score for positive conversations"
    )
    
    parser.add_argument(
        "--rating_temperature", 
        type=float, 
        default=0.3,
        help="Temperature for rating model"
    )
    
    # Training data parameters
    parser.add_argument(
        "--focus_on_agent_b", 
        action="store_true", 
        default=True,
        help="Focus training on Agent B's adaptive responses"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="training_data",
        help="Output directory for training data"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected mode
    if args.mode == "collect_bc":
        collect_behavior_cloning_data(args)
    elif args.mode == "collect_sr":
        collect_self_reinforcement_data(args)
    elif args.mode == "rate_conversations":
        rate_conversations(args)
    elif args.mode == "prepare_training":
        prepare_training_data(args)
    elif args.mode == "full_pipeline":
        run_full_pipeline(args)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()