#!/usr/bin/env python3
"""
Main Training Script for Social-Decipher
Following Sotopia-π training methodology but adapted for barrier-aware scenarios.

Usage:
    python -m social_decipher.training.train --config configs/training_config.yaml
    python -m social_decipher.training.train --experiment_name my_experiment --num_improve_steps 2
"""

import argparse
import os
import sys
import yaml
import json
import wandb
import random
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from social_decipher.training.sotopia_style_trainer import (
    SotopiaStyleTrainer, 
    TrainingConfig, 
    create_training_config
)
from social_decipher.training.data_collector import load_barrier_episode_sets


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_episodes(episodes_file: str) -> List[Dict[str, Any]]:
    """Load episode data from JSONL file"""
    episodes = []
    with open(episodes_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def create_training_config_from_dict(config_dict: Dict[str, Any]) -> TrainingConfig:
    """Create TrainingConfig from dictionary"""
    
    # Extract configuration sections
    experiment = config_dict.get("experiment", {})
    models = config_dict.get("models", {})
    training = config_dict.get("training", {})
    data_collection = config_dict.get("data_collection", {})
    quality_filtering = config_dict.get("quality_filtering", {})
    
    # Create TrainingConfig
    config = TrainingConfig(
        # Experiment settings
        experiment_name=experiment.get("name", "social_decipher_barrier_training"),
        num_improve_steps=experiment.get("num_improve_steps", 3),
        checkpoint_dir=experiment.get("checkpoint_dir", "checkpoints"),
        output_dir=experiment.get("output_dir", "training_data"),
        
        # Model settings
        expert_model=models.get("expert_model", "gpt-4o"),
        agent_model=models.get("agent_model", "gpt-4o-mini"),
        evaluator_model=models.get("evaluator_model", "gpt-4o"),
        
        # Training settings
        num_train_epochs=training.get("num_train_epochs", 20.0),
        per_device_train_batch_size=training.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        learning_rate=training.get("learning_rate", 5e-5),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training.get("warmup_ratio", 0.03),
        
        # Data collection settings
        conversations_per_episode=data_collection.get("conversations_per_episode", 3),
        max_rounds=data_collection.get("max_rounds", 20),
        
        # Quality filtering settings
        quality_threshold=quality_filtering.get("quality_threshold", 6.0),
        filter_top_k=quality_filtering.get("filter_top_k", 2),
        
        # LoRA settings
        finetuning_type=training.get("finetuning_type", "lora"),
        lora_target=training.get("lora_target", "q_proj,v_proj"),
        quantization_bit=training.get("quantization_bit", 4),
        quantization_type=training.get("quantization_type", "nf4"),
    )
    
    return config


def setup_environment(config_dict: Dict[str, Any]):
    """Setup environment variables from config"""
    env_vars = config_dict.get("environment", {})
    
    for key, value in env_vars.items():
        if value is not None:
            os.environ[key] = value
            print(f"Set {key} environment variable")


def main():
    parser = argparse.ArgumentParser(
        description="Train Social-Decipher models using Sotopia-π methodology"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="social_decipher_barrier_training",
        help="Name of the training experiment"
    )
    parser.add_argument(
        "--num_improve_steps", 
        type=int, 
        default=3,
        help="Number of improvement steps"
    )
    parser.add_argument(
        "--episodes_file", 
        type=str, 
        default="data/episode_sample.jsonl",
        help="Path to episodes JSONL file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="training_data",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="checkpoints",
        help="Directory for model checkpoints"
    )
    
    # Wandb options
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="social-decipher",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=None,
        help="Wandb entity name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name"
    )
    
    # Model options
    parser.add_argument(
        "--expert_model", 
        type=str, 
        default="gpt-4o",
        help="Expert model for BC demonstrations"
    )
    parser.add_argument(
        "--agent_model", 
        type=str, 
        default="gpt-4o-mini",
        help="Agent model for SR self-play"
    )
    parser.add_argument(
        "--partner_model",
        type=str,
        default="gpt-4o-mini",
        help="Partner model for SR self-play"
    )
    parser.add_argument(
        "--evaluator_model", 
        type=str, 
        default="gpt-4o",
        help="Model for conversation evaluation"
    )
    
    # Training options
    parser.add_argument(
        "--conversations_per_episode", 
        type=int, 
        default=3,
        help="Number of conversations per episode"
    )
    parser.add_argument(
        "--quality_threshold", 
        type=float, 
        default=6.0,
        help="Quality threshold for conversation filtering"
    )
    parser.add_argument(
        "--filter_top_k", 
        type=int, 
        default=2,
        help="Top-k filtering per episode type"
    )
    
    parser.add_argument(
        "--scoring_strategy", 
        type=str, 
        default="default",
        help="Scoring strategy to use (e.g., default, weighted, custom_barrier_focused)"
    )
    
    # Data options
    parser.add_argument(
        "--use_barrier_episodes", 
        action="store_true",
        help="Use barrier-specific episode sets"
    )
    parser.add_argument(
        "--episode_limit",
        type=int,
        default=None,
        help="Limit the number of episodes to process for faster runs."
    )
    parser.add_argument(
        "--load_existing_data",
        action="store_true",
        help="Load existing BC/SR data if available, instead of regenerating it."
    )
    parser.add_argument(
        "--barrier_types", 
        nargs="+", 
        default=["semantic", "cultural", "emotional"],
        help="Barrier types to include"
    )
    
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.experiment_name,
        config=vars(args),
        settings=wandb.Settings(init_timeout=300)
    )
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config_dict = load_config(args.config)
        setup_environment(config_dict)
        config = create_training_config_from_dict(config_dict)
    else:
        print("Using command-line configuration")
        config = create_training_config(
            experiment_name=args.experiment_name,
            num_improve_steps=args.num_improve_steps,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            expert_model=args.expert_model,
            agent_model=args.agent_model,
            partner_model=args.partner_model,
            evaluator_model=args.evaluator_model,
            conversations_per_episode=args.conversations_per_episode,
            quality_threshold=args.quality_threshold,
            filter_top_k=args.filter_top_k,
            scoring_strategy=args.scoring_strategy
        )
    
    # Pass the data loading preference to the trainer
    config.load_existing_data = args.load_existing_data
    
    # Load episodes
    print(f"Loading episodes from {args.episodes_file}")
    if not os.path.exists(args.episodes_file):
        print(f"ERROR: Episodes file not found: {args.episodes_file}")
        sys.exit(1)
    
    episodes = load_episodes(args.episodes_file)
    print(f"Loaded {len(episodes)} base episodes")
    
    # Load barrier episodes if requested
    if args.use_barrier_episodes:
        print("Loading barrier-specific episodes...")
        barrier_episodes = load_barrier_episode_sets()
        
        # Combine all episodes into a dictionary first
        all_episode_sets = {"neutral": episodes, **barrier_episodes}

        # Apply episode limit by random sampling to EACH category if specified
        if args.episode_limit is not None:
            print(f"Randomly sampling {args.episode_limit} episodes per category...")
            for category, eps in all_episode_sets.items():
                original_count = len(eps)
                if original_count > args.episode_limit:
                    all_episode_sets[category] = random.sample(eps, args.episode_limit)
                # If the category has fewer episodes than the limit, just use all of them
                else:
                    all_episode_sets[category] = eps
                print(f"   {category}: {original_count} → {len(all_episode_sets[category])} episodes")
        
        # Combine all limited episodes into the final training list
        final_episodes = []
        for category_eps in all_episode_sets.values():
            final_episodes.extend(category_eps)
        
        episodes = final_episodes
        print(f"Total episodes for training: {len(episodes)}")
    
    # Handle the case where only a limit on neutral episodes is desired
    elif args.episode_limit is not None:
        original_count = len(episodes)
        if original_count > args.episode_limit:
            episodes = random.sample(episodes, args.episode_limit)
        print(f"Randomly sampling to {len(episodes)} neutral episodes based on --episode_limit.")

    # Initialize trainer
    print("Initializing Sotopia-π style trainer...")
    trainer = SotopiaStyleTrainer(config)
    
    # Run training pipeline
    print("Starting training pipeline...")
    try:
        trainer.run_training_pipeline(episodes)
        print("Training completed successfully!")
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
