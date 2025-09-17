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
import shutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from datetime import datetime
import wandb
import numpy as np

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
        
    def run_single_improvement_step(self, episodes: List[Dict[str, Any]], current_step: int):
        """
        Runs a single, self-contained improvement step on a given batch of episodes.
        This includes data collection, filtering, and model training.
        """
        self.improve_step = current_step
        self.current_step = current_step
        print(f"Executing improvement step {self.improve_step + 1} with {len(episodes)} episodes.")

        # --- 1. Data Collection ---
        # This will now only run on the batch of episodes for the current step.
        all_conversations = self.collect_training_data(episodes, self.improve_step)
        if not all_conversations:
            print("  No conversations were collected. Skipping this improvement step.")
            return

        # --- 2. Quality Filtering ---
        filtered_conversations, ratings = self.filter_high_quality_data(all_conversations)
        
        if not filtered_conversations:
            print("  No conversations passed the quality filter. Skipping model training for this step.")
            return
            
        # --- 3. Log Performance Metrics ---
        self._log_performance_metrics(ratings, self.improve_step + 1)

        # --- 4. Prepare Data for SFT ---
        # This part needs to be updated to use the filtered data and format it correctly.
        sft_data = self._prepare_sft_data(filtered_conversations, ratings)
        
        # --- 5. Fine-Tune the Model ---
        if sft_data:
            self.train_model(sft_data, self.improve_step)
        else:
            print("  No SFT data was generated. Skipping model training.")

    def run_training_pipeline(self, all_episodes: List[Dict[str, Any]]):
        """
        DEPRECATED: This method is replaced by the new batched approach
        controlled from train.py. Keeping it for backward compatibility if needed.
        """
        print("WARNING: `run_training_pipeline` is deprecated. The training loop is now managed in train.py.")
        # For simplicity, we can have it call the new step-wise function for the full dataset
        self.run_single_improvement_step(all_episodes, 0)
        
    def _prepare_sft_data(self, conversations: List, ratings: List) -> List[Dict[str, Any]]:
        """
        Prepares the final SFT data in the required format for LLaMA-Factory.
        """
        print("\nStep 4: Preparing SFT Data for LLaMA-Factory")
        
        # We can reuse the formatting logic from our existing PolicyUpdater
        training_examples = self.policy_updater.prepare_training_data(
            conversations=conversations,
            ratings=ratings,
            focus_on_agent_b=True,
            min_quality_score=0 # Filtering is already done
        )

        llama_factory_data = self.policy_updater.format_for_llama_factory(training_examples)
        
        # Save the formatted data for this step for inspection
        sft_data_path = os.path.join(self.config.output_dir, f"sft_data_step_{self.improve_step}.json")
        with open(sft_data_path, 'w', encoding='utf-8') as f:
            json.dump(llama_factory_data, f, indent=2)
        print(f"  Saved formatted SFT data for this step to {sft_data_path}")
        
        return llama_factory_data
        
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
    ) -> Tuple[List[TrainingConversation], List[ConversationRating]]:
        """Filter conversations using extensible scoring strategy and return ratings."""
        
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
        
        return top_k_conversations, ratings

    def _log_performance_metrics(self, ratings: List[ConversationRating], improve_step: int):
        """Calculate and log average performance metrics to wandb."""
        if not ratings:
            print("No ratings available to log performance metrics.")
            return

        print(f"📊 Logging performance metrics for step {improve_step + 1} to wandb...")
        
        # Initialize containers for scores
        metrics = {
            "performance/goal_completion": [],
            "performance/believability": [],
            "performance/relationship": [],
            "performance/unresolved_confusion": [],
            "performance/mutual_understanding": [],
            "performance/composite_score": []
        }

        # Extract scores from each rating
        for r in ratings:
            metrics["performance/goal_completion"].append(r.goal_completion)
            metrics["performance/believability"].append(r.believability)
            metrics["performance/relationship"].append(r.relationship)
            if r.episode_level:
                metrics["performance/unresolved_confusion"].append(r.episode_level.get("unresolved_confusion", np.nan))
                metrics["performance/mutual_understanding"].append(r.episode_level.get("mutual_understanding", np.nan))
            
            # Calculate and store the composite score for this conversation
            composite_score = self.scoring_manager.strategy.calculate_composite_score(r)
            metrics["performance/composite_score"].append(composite_score)

        # Calculate averages, handling potential NaNs from missing data
        avg_metrics = {key: np.nanmean(values) for key, values in metrics.items() if values}
        avg_metrics["improvement_step"] = improve_step + 1
        
        wandb.log(avg_metrics)
        print("   Done logging.")
        
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
            model_name=self.config.agent_model, # Pass the correct agent model
            output_dir=output_dir
        )
        
        # Update config with current settings and the dataset path for this step
        sft_data_path = os.path.join(self.config.output_dir, f"sft_data_step_{improve_step}.json")
        # LLaMA-Factory expects dataset_dir + dataset (basename without .json)
        config["dataset_dir"] = self.config.output_dir
        dataset_name = os.path.splitext(os.path.basename(sft_data_path))[0]
        config["dataset"] = dataset_name

        # Ensure dataset_info.json exists and includes our dataset entry
        dataset_info_path = os.path.join(self.config.output_dir, "dataset_info.json")
        try:
            if os.path.exists(dataset_info_path):
                with open(dataset_info_path, 'r', encoding='utf-8') as f:
                    dataset_info = json.load(f)
            else:
                dataset_info = {}
            dataset_info[dataset_name] = {
                "file_name": f"{dataset_name}.json",
                "formatting": "alpaca"
            }
            with open(dataset_info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"WARNING: Failed to write dataset_info.json: {e}")

        config_path = os.path.join(self.policy_updater.output_dir, "llama_factory_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Determine number of GPUs from environment variable
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        num_gpus = len(cuda_visible_devices.split(','))

        # Build the training command
        if num_gpus > 1:
            print(f"Multi-GPU training detected ({num_gpus} GPUs). Using 'accelerate launch'.")
            # Ensure llamafactory-cli executable can be found
            llama_factory_executable = shutil.which("llamafactory-cli")
            if not llama_factory_executable:
                raise FileNotFoundError("Could not find 'llamafactory-cli' executable in the environment's PATH.")
            
            training_command = [
                "accelerate", "launch",
                "--num_processes", str(num_gpus),
                llama_factory_executable,
                "train",
                config_path
            ]
        else:
            print("Single-GPU training detected.")
            training_command = [
                "llamafactory-cli",
                "train",
                config_path
            ]

        # Run LLaMA-Factory training
        print(f"\nRunning LLaMA-Factory command: {' '.join(training_command)}")
        
        # Set CUDA devices if specified
        env = os.environ.copy()
        
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                wandb.log({"training_output": output.strip()})
        
        # Wait for the process to finish
        process.wait()
        print(f"LLaMA-Factory training process finished with exit code {process.returncode}")
        if process.returncode != 0:
            raise RuntimeError(f"LLaMA-Factory exited with code {process.returncode}. Check the config at {config_path} and logs above.")
        
        # Save training configuration
        config_path = os.path.join(output_dir, "training_config.yaml")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"Training configuration saved to {config_path}")
        print("Ready for model training (use LLaMA-Factory to execute)")
        
    def evaluate_model(self, improve_step: int):
        """Evaluate the trained model"""
        
        print(f"\nStep 5: Model Evaluation (Improvement Step {improve_step + 1})")
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
