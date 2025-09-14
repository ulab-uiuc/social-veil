"""
Sotopia-π Style Training Pipeline for Social-Decipher
Reimplements the training architecture following Sotopia-π methodology
but adapted for barrier-aware social intelligence scenarios.
"""

import json
import os
import multiprocessing
import subprocess
import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from datetime import datetime

from .data_collector import BarrierDataCollector, TrainingConversation
from .conversation_rater import ConversationRater, ConversationRating
from .policy_updater import SocialPolicyUpdater, TrainingExample
from .scoring_strategy import (
    ScoringManager, 
    ScoringConfig, 
    get_default_scoring_config,
    get_barrier_focused_config,
    get_balanced_config,
    get_custom_barrier_focused_config
)


@dataclass
class TrainingConfig:
    """Training configuration following Sotopia-π style"""
    # Experiment settings
    experiment_name: str = "social_decipher_barrier_training"
    num_improve_steps: int = 3
    checkpoint_dir: str = "checkpoints"
    
    # Model settings
    expert_model: str = "gpt-4o"
    agent_model: str = "gpt-4o-mini"
    partner_model: str = "gpt-4o-mini"
    evaluator_model: str = "gpt-4o"
    
    # Training settings
    num_train_epochs: float = 20.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    
    # Data settings
    conversations_per_episode: int = 3
    max_rounds: int = 20
    
    # Scoring strategy settings
    scoring_strategy: str = "default"  # "default", "weighted", "adaptive", "custom"
    scoring_config: Optional[ScoringConfig] = None
    
    # Legacy settings (for backward compatibility)
    quality_threshold: float = 6.0
    filter_top_k: int = 2
    
    # LoRA settings
    finetuning_type: str = "lora"
    lora_target: str = "q_proj,v_proj"
    quantization_bit: int = 4
    quantization_type: str = "nf4"
    
    # Output settings
    output_dir: str = "training_data"
    save_steps: int = 1000000000
    save_total_limit: int = 5
    logging_steps: int = 1

    # New feature: Load existing data to speed up iteration
    load_existing_data: bool = False


class SotopiaStyleTrainer:
    """
    Main training orchestrator following Sotopia-π methodology.
    
    Implements the complete training pipeline:
    1. Data Collection (BC + SR)
    2. Quality Filtering
    3. Data Preprocessing
    4. Model Training
    5. Evaluation & Deployment
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_directories()
        
        # Initialize components
        self.data_collector = BarrierDataCollector(
            expert_model=config.expert_model,
            agent_model=config.agent_model,
            partner_model=config.partner_model,
            evaluator_model=config.evaluator_model,
            output_dir=config.output_dir
        )
        self.conversation_rater = ConversationRater()
        self.policy_updater = SocialPolicyUpdater(
            output_dir=os.path.join(config.output_dir, "policy_updates")
        )
        
        # Initialize scoring strategy
        self.scoring_manager = self._initialize_scoring_strategy()
        
        # Training state
        self.current_step = 0
        self.training_history = []
        
    def setup_directories(self):
        """Setup training directories"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "policy_updates"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "training_data"), exist_ok=True)
        
    def _initialize_scoring_strategy(self) -> ScoringManager:
        """Initialize scoring strategy based on configuration"""
        
        # Use provided scoring config or create default based on legacy settings
        if self.config.scoring_config is not None:
            scoring_config = self.config.scoring_config
        else:
            # Create config from legacy settings for backward compatibility
            scoring_config = ScoringConfig(
                quality_threshold=self.config.quality_threshold,
                filter_top_k=self.config.filter_top_k
            )
        
        print(f"Initializing {self.config.scoring_strategy} scoring strategy")
        print(f"   Quality threshold: {scoring_config.quality_threshold}")
        print(f"   Filter top-K: {scoring_config.filter_top_k}")
        print(f"   Scoring dimensions: {scoring_config.scoring_dimensions}")
        
        return ScoringManager(
            strategy_name=self.config.scoring_strategy,
            config=scoring_config
        )
        
    def run_training_pipeline(self, episodes: List[Dict[str, Any]]):
        """
        Run the complete training pipeline following Sotopia-π methodology.
        
        Args:
            episodes: List of episode data for training
        """
        print("Starting Sotopia-π Style Training Pipeline")
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Improvement Steps: {self.config.num_improve_steps}")
        
        for improve_step in range(self.config.num_improve_steps):
            print(f"\nImprovement Step {improve_step + 1}/{self.config.num_improve_steps}")
            
            # Step 1: Data Collection
            conversations = self.collect_training_data(episodes, improve_step)
            
            # Step 2: Quality Filtering
            filtered_conversations = self.filter_high_quality_data(conversations)
            
            # Step 3: Data Preprocessing
            training_data = self.preprocess_training_data(filtered_conversations)
            
            # Step 4: Model Training
            if training_data:
                self.train_model(training_data, improve_step)
            else:
                print("WARNING: No training data available, skipping training step")
                
            # Step 5: Evaluation (optional)
            if improve_step < self.config.num_improve_steps - 1:
                self.evaluate_model(improve_step)
                
        print("\nTraining pipeline completed!")
        self.save_training_summary()
        
    def collect_training_data(
        self, 
        episodes: List[Dict[str, Any]], 
        improve_step: int
    ) -> List[TrainingConversation]:
        """Collect training data using BC and SR methods, with an option to load existing BC data."""
        
        print(f"\nStep 1: Data Collection (Improvement Step {improve_step + 1})")

        # Behavior Cloning (Expert demonstrations)
        bc_data_path = os.path.join(self.config.output_dir, "bc_data.json")
        if self.config.load_existing_data and os.path.exists(bc_data_path):
            print(f"♻️ Loading existing Behavior Cloning data from {bc_data_path}...")
            self.data_collector.load_conversations(bc_file="bc_data.json")
            bc_conversations, _ = self.data_collector.get_all_conversations()
            print(f"   Loaded {len(bc_conversations)} BC conversations.")
        else:
            print("🎓 Collecting new Behavior Cloning data...")
            bc_conversations = self.data_collector.collect_behavior_cloning_data(
                episodes, 
                self.config.conversations_per_episode,
                self.config.max_rounds
            )
        
        # Self-Reinforcement (Self-play) should always be regenerated as it depends on the current agent
        print("Collecting new Self-Reinforcement data...")
        sr_conversations = self.data_collector.collect_self_reinforcement_data(
            episodes,
            self.config.conversations_per_episode,
            self.config.max_rounds
        )
        
        all_conversations = bc_conversations + sr_conversations
        
        print(f"\nCollected {len(all_conversations)} total conversations for this step:")
        print(f"   - BC: {len(bc_conversations)} conversations")
        print(f"   - SR: {len(sr_conversations)} conversations")
        
        return all_conversations
        
    def filter_high_quality_data(
        self, 
        conversations: List[TrainingConversation]
    ) -> List[TrainingConversation]:
        """Filter conversations using extensible scoring strategy"""
        
        print(f"\nStep 2: Quality Filtering ({self.config.scoring_strategy} strategy)")
        print(f"Rating {len(conversations)} conversations...")
        
        # Rate conversations
        ratings = self.conversation_rater.rate_conversations(
            conversations, 
            self.config.quality_threshold  # For backward compatibility
        )
        
        # Use scoring manager for filtering
        filtered_conversations = self.scoring_manager.filter_conversations(
            conversations, ratings
        )
        
        # Apply top-k filtering per episode type
        top_k_conversations = self.scoring_manager.apply_top_k_filtering(
            filtered_conversations, ratings
        )
        
        print(f"Final result: {len(conversations)} → {len(top_k_conversations)} conversations")
        
        # Save ratings and scores for analysis
        self.conversation_rater.save_ratings(
            ratings, 
            os.path.join(self.config.output_dir, f"ratings_step_{self.current_step}.json")
        )
        
        # Save scores for analysis
        scores = self.scoring_manager.get_conversation_scores(conversations, ratings)
        scores_file = os.path.join(self.config.output_dir, f"scores_step_{self.current_step}.json")
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        print(f"Saved scores to {scores_file}")
        
        return top_k_conversations
        
    def preprocess_training_data(
        self, 
        conversations: List[TrainingConversation]
    ) -> List[Dict[str, Any]]:
        """Preprocess conversations into training format"""
        
        print(f"\nStep 3: Data Preprocessing")
        
        # Rate conversations for training data preparation
        ratings = self.conversation_rater.rate_conversations(conversations)
        
        # Prepare training examples
        training_examples = self.policy_updater.prepare_training_data(
            conversations, ratings, focus_on_agent_b=True
        )
        
        # Format for LLaMA-Factory
        formatted_data = self.policy_updater.format_for_llama_factory(
            training_examples
        )
        
        # Save training data
        training_file = os.path.join(
            self.config.output_dir, 
            f"training_data_step_{self.current_step}.json"
        )
        self.policy_updater.save_training_data(formatted_data, training_file)
        
        print(f"Preprocessed {len(formatted_data)} training examples")
        
        return formatted_data
        
    def train_model(self, training_data: List[Dict[str, Any]], improve_step: int):
        """Train model using the collected data"""
        
        print(f"\nStep 4: Model Training (Improvement Step {improve_step + 1})")
        
        # Create training configuration
        output_dir = os.path.join(
            self.config.checkpoint_dir, 
            f"{self.config.experiment_name}_step_{improve_step}"
        )
        
        # Create LLaMA-Factory config
        config = self.policy_updater.create_llama_factory_config(
            dataset_name=f"social_decipher_step_{improve_step}",
            output_dir=output_dir
        )
        
        # Update config with current settings
        config.update({
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "lr_scheduler_type": self.config.lr_scheduler_type,
            "warmup_ratio": self.config.warmup_ratio,
            "finetuning_type": self.config.finetuning_type,
            "lora_target": self.config.lora_target,
            "quantization_bit": self.config.quantization_bit,
            "quantization_type": self.config.quantization_type,
            "save_steps": self.config.save_steps,
            "save_total_limit": self.config.save_total_limit,
            "logging_steps": self.config.logging_steps,
            "report_to": "wandb"
        })
        
        print(f"Training output directory: {output_dir}")
        print(f"Training configuration created")
        
        # Note: Actual training would be triggered here
        # For now, we just save the configuration
        config_path = os.path.join(output_dir, "training_config.yaml")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"Training configuration saved to {config_path}")
        print("Ready for model training (use LLaMA-Factory to execute)")
        
    def evaluate_model(self, improve_step: int):
        """Evaluate the trained model"""
        
        print(f"\nStep 5: Model Evaluation (Improvement Step {improve_step + 1})")
        
        # This would typically involve:
        # 1. Deploying the trained model
        # 2. Running evaluation on test scenarios
        # 3. Comparing performance metrics
        
        print("Model evaluation completed")
        
    def save_training_summary(self):
        """Save training summary and history"""
        
        summary = {
            "experiment_name": self.config.experiment_name,
            "config": asdict(self.config),
            "training_history": self.training_history,
            "completion_time": datetime.now().isoformat(),
            "total_steps": self.config.num_improve_steps
        }
        
        summary_path = os.path.join(self.config.output_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"Training summary saved to {summary_path}")


def create_training_config(
    experiment_name: str = "social_decipher_barrier_training",
    scoring_strategy: str = "default",
    scoring_config: Optional[ScoringConfig] = None,
    **kwargs
) -> TrainingConfig:
    """Create training configuration with custom parameters"""
    
    config = TrainingConfig(
        experiment_name=experiment_name,
        scoring_strategy=scoring_strategy,
        scoring_config=scoring_config
    )
    
    # Update with custom parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"WARNING: Unknown config parameter: {key}")
    
    return config


def create_barrier_focused_training_config(**kwargs) -> TrainingConfig:
    """Create training configuration focused on barrier handling"""
    scoring_config = get_barrier_focused_config()
    return create_training_config(
        scoring_strategy="weighted",
        scoring_config=scoring_config,
        **kwargs
    )


def create_custom_barrier_focused_training_config(**kwargs) -> TrainingConfig:
    """Create the user-defined custom barrier-focused training configuration."""
    scoring_config = get_custom_barrier_focused_config()
    return create_training_config(
        scoring_strategy="weighted",
        scoring_config=scoring_config,
        **kwargs
    )


def create_balanced_training_config(**kwargs) -> TrainingConfig:
    """Create training configuration with balanced scoring"""
    scoring_config = get_balanced_config()
    return create_training_config(
        scoring_strategy="weighted", 
        scoring_config=scoring_config,
        **kwargs
    )


def create_adaptive_training_config(**kwargs) -> TrainingConfig:
    """Create training configuration with adaptive scoring"""
    scoring_config = get_default_scoring_config()
    return create_training_config(
        scoring_strategy="adaptive",
        scoring_config=scoring_config,
        **kwargs
    )


def run_sotopia_style_training(
    episodes: List[Dict[str, Any]],
    config: Optional[TrainingConfig] = None,
    **config_kwargs
):
    """
    Run Sotopia-π style training pipeline.
    
    Args:
        episodes: List of episode data for training
        config: Training configuration (optional)
        **config_kwargs: Additional configuration parameters
    """
    
    if config is None:
        config = create_training_config(**config_kwargs)
    
    trainer = SotopiaStyleTrainer(config)
    trainer.run_training_pipeline(episodes)
    
    return trainer


# Example usage
if __name__ == "__main__":
    # Load episodes (example)
    episodes = []  # Load your episode data here
    
    # Create custom configuration
    config = create_training_config(
        experiment_name="barrier_adaptation_v1",
        num_improve_steps=2,
        conversations_per_episode=5,
        quality_threshold=7.0
    )
    
    # Run training
    trainer = run_sotopia_style_training(episodes, config)
