"""
Policy Update for Interactive Training  
Implements fine-tuning pipeline for barrier-aware social intelligence
following Sotopia-π methodology but adapted for communication barrier scenarios.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
from datetime import datetime

from .data_collector import TrainingConversation
from .conversation_rater import ConversationRating


@dataclass
class TrainingExample:
    """Single training example for fine-tuning"""
    conversation_id: str
    episode_type: str
    barrier_type: Optional[str]
    agent_role: str  # "agent_a" or "agent_b"
    conversation_history: List[str]
    target_response: str
    quality_score: float
    context: Dict[str, Any]


class SocialPolicyUpdater:
    """
    Updates agent policy through fine-tuning on high-quality conversations.
    
    Inspired by Sotopia-π policy update but specialized for:
    1. Barrier Adaptation: Learning to communicate despite Agent A's barriers
    2. Social Intelligence: Maintaining relationships under communication stress
    3. Adaptive Strategies: Developing flexible communication approaches
    4. Quality Focus: Training only on high-rated conversation examples
    """
    
    def __init__(self, output_dir: str = "training_data/policy_updates"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_training_data(
        self,
        conversations: List[TrainingConversation],
        ratings: List[ConversationRating],
        focus_on_agent_b: bool = True,
        min_quality_score: float = 0.0
    ) -> List[TrainingExample]:
        """
        Prepare fine-tuning data from rated conversations.
        
        Args:
            conversations: Training conversations
            ratings: Quality ratings for conversations  
            focus_on_agent_b: Whether to focus on Agent B's adaptive responses
            min_quality_score: Minimum quality score to include
        """
        
        print(f"Preparing training data (focus_on_agent_b={focus_on_agent_b})...")
        
        # Create rating lookup
        rating_map = {r.conversation_id: r for r in ratings}
        
        training_examples = []
        
        for conversation in conversations:
            rating = rating_map.get(conversation.conversation_id)
            
            # Skip if a rating is not found. Filtering by score is now handled
            # upstream in scripts like manual_filter.py, so the check for
            # overall_quality is removed here.
            if not rating:
                continue
                
            # Extract training examples from conversation
            examples = self._extract_training_examples(
                conversation, rating, focus_on_agent_b
            )
            training_examples.extend(examples)
            
        print(f"Prepared {len(training_examples)} training examples")
        return training_examples
    
    def _extract_training_examples(
        self,
        conversation: TrainingConversation,
        rating: ConversationRating,
        focus_on_agent_b: bool
    ) -> List[TrainingExample]:
        """Extract training examples from a single conversation"""
        
        examples = []
        conversation_log = conversation.conversation_log
        
        # Identify agent names from conversation
        agent_names = self._extract_agent_names(conversation_log)
        if len(agent_names) < 2:
            return examples
            
        agent_a_name, agent_b_name = agent_names[0], agent_names[1]
        
        # Extract turn-by-turn examples
        for i in range(1, len(conversation_log)):
            current_line = conversation_log[i]
            
            # Determine which agent is speaking
            if current_line.startswith(f"{agent_a_name}:"):
                agent_role = "agent_a"
                speaker_name = agent_a_name
            elif current_line.startswith(f"{agent_b_name}:"):
                agent_role = "agent_b"
                speaker_name = agent_b_name
            else:
                continue
                
            # Skip if focusing on Agent B and this is Agent A
            if focus_on_agent_b and agent_role == "agent_a":
                continue
                
            # Extract response text
            target_response = current_line[len(f"{speaker_name}:"):].strip()
            if not target_response:
                continue
                
            # Get conversation history up to this point
            conversation_history = conversation_log[:i]
            
            # Create training example
            example = TrainingExample(
                conversation_id=conversation.conversation_id,
                episode_type=conversation.episode_type,
                barrier_type=self._get_barrier_type(conversation),
                agent_role=agent_role,
                conversation_history=conversation_history,
                target_response=target_response,
                quality_score=rating.agent_2.get('overall', 0.0),
                context=self._create_example_context(conversation, rating, agent_role)
            )
            
            examples.append(example)
            
        return examples
    
    def _extract_agent_names(self, conversation_log: List[str]) -> List[str]:
        """Extract agent names from conversation log"""
        names = []
        for line in conversation_log:
            if ':' in line:
                name = line.split(':')[0].strip()
                if name not in names:
                    names.append(name)
                if len(names) >= 2:
                    break
        return names
    
    def _get_barrier_type(self, conversation: TrainingConversation) -> Optional[str]:
        """Get barrier type from conversation"""
        if conversation.barrier_info:
            return conversation.barrier_info.get("barrier_type")
        return None
    
    def _create_example_context(
        self,
        conversation: TrainingConversation,
        rating: ConversationRating,
        agent_role: str
    ) -> Dict[str, Any]:
        """Create context information for training example"""
        
        context = {
            "episode_type": conversation.episode_type,
            "trajectory_type": conversation.trajectory_type,
            "barrier_info": conversation.barrier_info,
            "quality_metrics": {
                "agent_1_scores": rating.agent_1,
                "agent_2_scores": rating.agent_2,
                "interaction_quality": rating.interaction_quality,
                "episode_level_scores": rating.episode_level,
            },
            "agent_role": agent_role,
            "timestamp": conversation.timestamp
        }
        
        return context
    
    def format_for_llama_factory(
        self,
        training_examples: List[TrainingExample],
        instruction_template: str = None
    ) -> List[Dict[str, Any]]:
        """
        Format training examples for LLaMA-Factory fine-tuning.
        
        Uses instruction-following format compatible with QLoRA training.
        """
        
        print("Formatting for LLaMA-Factory...")
        
        if instruction_template is None:
            instruction_template = self._get_default_instruction_template()
        
        formatted_data = []
        
        for example in training_examples:
            # Create instruction based on barrier context
            instruction = self._create_instruction(example, instruction_template)
            
            # Format conversation history
            conversation_text = "\n".join(example.conversation_history)
            
            # Create training sample
            sample = {
                "instruction": instruction,
                "input": conversation_text,
                "output": example.target_response,
                "metadata": {
                    "conversation_id": example.conversation_id,
                    "episode_type": example.episode_type,
                    "barrier_type": example.barrier_type,
                    "quality_score": example.quality_score,
                    "agent_role": example.agent_role
                }
            }
            
            formatted_data.append(sample)
        
        print(f"Formatted {len(formatted_data)} samples for training")
        return formatted_data
    
    def format_for_sotopia_sft(
        self,
        training_examples: List[TrainingExample],
    ) -> List[Dict[str, Any]]:
        """
        Format training examples for the SotopiaSFTTrainer.
        It expects a list of dictionaries with "input" and "output" keys.
        """
        print("Formatting for Sotopia SFT...")
        
        formatted_data = []
        for example in training_examples:
            # The "input" is the conversation history.
            conversation_text = "\n".join(example.conversation_history)
            
            # The "output" is the target response.
            sample = {
                "input": conversation_text,
                "output": example.target_response,
            }
            formatted_data.append(sample)
            
        print(f"Formatted {len(formatted_data)} samples for Sotopia SFT training")
        return formatted_data

    def _create_instruction(self, example: TrainingExample, template: str) -> str:
        """Create instruction for training sample"""
        
        # Get barrier description
        barrier_desc = ""
        if example.barrier_type:
            barrier_descriptions = {
                "semantic_structure": "The other person uses vague, ambiguous language with complex sentences. Adapt your communication to maintain understanding.",
                "cultural_style": "The other person uses indirect, high-context communication with hedges and politeness. Match their communication style appropriately.",
                "emotional_influence": "The other person has a negative emotional tone and uses clipped responses. Stay patient and maintain the conversation constructively."
            }
            barrier_desc = barrier_descriptions.get(example.barrier_type, "")
        
        # Focus on Agent B learning
        if example.agent_role == "agent_b":
            role_instruction = "You are responding in a conversation where you need to adapt to your partner's communication style."
        else:
            role_instruction = "You are communicating in a social conversation."
        
        instruction = template.format(
            role_instruction=role_instruction,
            barrier_description=barrier_desc,
            episode_type=example.episode_type
        )
        
        return instruction
    
    def _get_default_instruction_template(self) -> str:
        """Get default instruction template for training"""
        return """You are a socially intelligent conversational agent. {role_instruction}

{barrier_description}

Continue the conversation naturally while maintaining social appropriateness and working toward productive communication. Show emotional intelligence and adaptability in your response.

Episode type: {episode_type}"""
    
    def save_training_data(
        self,
        formatted_data: List[Dict[str, Any]],
        filename: str = "barrier_training_data.json"
    ):
        """Save formatted training data"""
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved training data to {filepath}")
        
        # Also save metadata summary
        self._save_training_summary(formatted_data, filepath.replace('.json', '_summary.json'))
    
    def _save_training_summary(self, formatted_data: List[Dict[str, Any]], filepath: str):
        """Save training data summary"""
        
        # Analyze training data
        total_samples = len(formatted_data)
        
        # Count by episode type
        episode_counts = {}
        barrier_counts = {}
        quality_scores = []
        
        for sample in formatted_data:
            metadata = sample.get("metadata", {})
            
            episode_type = metadata.get("episode_type", "unknown")
            episode_counts[episode_type] = episode_counts.get(episode_type, 0) + 1
            
            barrier_type = metadata.get("barrier_type")
            if barrier_type:
                barrier_counts[barrier_type] = barrier_counts.get(barrier_type, 0) + 1
            
            quality_score = metadata.get("quality_score")
            if quality_score:
                quality_scores.append(quality_score)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        summary = {
            "total_samples": total_samples,
            "episode_type_distribution": episode_counts,
            "barrier_type_distribution": barrier_counts,
            "average_quality_score": avg_quality,
            "quality_score_range": [min(quality_scores), max(quality_scores)] if quality_scores else [0, 0],
            "generation_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f"Training Data Summary:")
        print(f"   Total samples: {total_samples}")
        print(f"   Episode types: {episode_counts}")
        print(f"   Barrier types: {barrier_counts}")
        print(f"   Average quality: {avg_quality:.1f}/10")
    
    def create_llama_factory_config(
        self,
        dataset_name: str,
        model_name: str, # This should be the agent_model from the main config
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Create LLaMA-Factory training configuration based on Sotopia-π paper.
        This configuration is used by the llamafactory-cli tool, which leverages
        Hugging Face Accelerate for distributed and mixed-precision training.
        """
        
        config = {
            # Core settings
            "stage": "sft",
            "do_train": True,
            "model_name_or_path": model_name,
            "dataset": dataset_name,
            "template": "qwen",
            "finetuning_type": "lora",
            "lora_target": "all",
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            "plot_loss": True,

            # Sotopia-π Training Hyperparameters
            "num_train_epochs": 20.0,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4, # Kept from original, adjust if needed
            "learning_rate": 5.0e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "cutoff_len": 4096,

            # Sotopia-π QLoRA Hyperparameters
            "quantization_bit": 4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,

            # Other settings
            "logging_steps": 10,
            "save_steps": 500,
            "max_grad_norm": 1.0,
            "use_unsloth": True,
            "ddp_timeout": 180000000,
            "include_num_input_tokens_seen": True,
        }
        
        config_path = os.path.join(self.output_dir, "llama_factory_config.yaml")
        
        # Save as YAML for LLaMA-Factory
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"Saved LLaMA-Factory config to {config_path}")
        return config