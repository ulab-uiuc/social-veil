"""
Data Preprocessing for Social-Decipher Training
Following Sotopia-π data processing methodology but adapted for barrier scenarios.
"""

import json
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from .data_collector import TrainingConversation
from .conversation_rater import ConversationRating


class SotopiaStyleDataProcessor:
    """
    Data processor following Sotopia-π methodology.
    
    Implements:
    1. Episode filtering and selection
    2. Quality-based data filtering
    3. Training data format conversion
    4. Dataset balancing and augmentation
    """
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def filter_episodes_by_quality(
        self,
        conversations: List[TrainingConversation],
        ratings: List[ConversationRating],
        quality_threshold: float = 6.0,
        filter_strategy: str = "absolute"
    ) -> List[TrainingConversation]:
        """
        Filter episodes based on quality ratings following Sotopia-π methodology.
        
        Args:
            conversations: List of training conversations
            ratings: Quality ratings for conversations
            quality_threshold: Minimum quality score
            filter_strategy: "absolute" or "relative" filtering
        """
        
        print(f"Filtering episodes by quality (threshold: {quality_threshold})")
        
        # Create rating lookup
        rating_map = {r.conversation_id: r for r in ratings}
        
        if filter_strategy == "absolute":
            # Absolute threshold filtering
            filtered_conversations = []
            for conv in conversations:
                rating = rating_map.get(conv.conversation_id)
                if rating and rating.overall_quality >= quality_threshold:
                    filtered_conversations.append(conv)
                    
        elif filter_strategy == "relative":
            # Relative filtering (top-k per episode type)
            filtered_conversations = self._relative_quality_filtering(
                conversations, ratings, quality_threshold
            )
        else:
            raise ValueError(f"Unknown filter strategy: {filter_strategy}")
        
        print(f"Filtered {len(conversations)} → {len(filtered_conversations)} conversations")
        return filtered_conversations
    
    def _relative_quality_filtering(
        self,
        conversations: List[TrainingConversation],
        ratings: List[ConversationRating],
        top_k_ratio: float = 0.5
    ) -> List[TrainingConversation]:
        """Apply relative quality filtering per episode type"""
        
        # Group conversations by episode type
        conversations_by_type = defaultdict(list)
        rating_map = {r.conversation_id: r for r in ratings}
        
        for conv in conversations:
            episode_type = conv.episode_type
            conversations_by_type[episode_type].append(conv)
        
        # Apply top-k filtering per type
        filtered_conversations = []
        for episode_type, convs in conversations_by_type.items():
            # Sort by quality score
            convs_with_scores = [
                (conv, rating_map[conv.conversation_id].overall_quality)
                for conv in convs
            ]
            convs_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top-k
            top_k = max(1, int(len(convs) * top_k_ratio))
            top_k_convs = convs_with_scores[:top_k]
            filtered_conversations.extend([conv for conv, _ in top_k_convs])
            
            print(f"   {episode_type}: {len(convs)} → {len(top_k_convs)} (top-{top_k})")
        
        return filtered_conversations
    
    def balance_dataset(
        self,
        conversations: List[TrainingConversation],
        target_distribution: Optional[Dict[str, int]] = None
    ) -> List[TrainingConversation]:
        """
        Balance dataset across episode types following Sotopia-π methodology.
        
        Args:
            conversations: List of conversations
            target_distribution: Target distribution per episode type
        """
        
        print("Balancing dataset across episode types...")
        
        # Count current distribution
        current_dist = defaultdict(int)
        conversations_by_type = defaultdict(list)
        
        for conv in conversations:
            episode_type = conv.episode_type
            current_dist[episode_type] += 1
            conversations_by_type[episode_type].append(conv)
        
        print("Current distribution:")
        for episode_type, count in current_dist.items():
            print(f"   {episode_type}: {count}")
        
        if target_distribution is None:
            # Use uniform distribution
            min_count = min(current_dist.values())
            target_distribution = {episode_type: min_count for episode_type in current_dist.keys()}
        
        # Balance dataset
        balanced_conversations = []
        for episode_type, target_count in target_distribution.items():
            if episode_type in conversations_by_type:
                convs = conversations_by_type[episode_type]
                if len(convs) >= target_count:
                    # Randomly sample target_count conversations
                    import random
                    sampled_convs = random.sample(convs, target_count)
                    balanced_conversations.extend(sampled_convs)
                else:
                    # Use all available conversations
                    balanced_conversations.extend(convs)
                    print(f"WARNING: {episode_type}: Only {len(convs)} available, target was {target_count}")
        
        print("Balanced distribution:")
        balanced_dist = defaultdict(int)
        for conv in balanced_conversations:
            balanced_dist[conv.episode_type] += 1
        
        for episode_type, count in balanced_dist.items():
            print(f"   {episode_type}: {count}")
        
        return balanced_conversations
    
    def convert_to_sft_format(
        self,
        conversations: List[TrainingConversation],
        ratings: List[ConversationRating],
        template: str = "qwen"
    ) -> List[Dict[str, Any]]:
        """
        Convert conversations to SFT training format following Sotopia-π methodology.
        
        Args:
            conversations: List of training conversations
            ratings: Quality ratings
            template: Template format for training
        """
        
        print(f"Converting to SFT format (template: {template})")
        
        rating_map = {r.conversation_id: r for r in ratings}
        sft_data = []
        
        for conv in conversations:
            rating = rating_map.get(conv.conversation_id)
            if not rating:
                continue
            
            # Extract conversation turns
            conversation_log = conv.conversation_log
            
            # Create training examples for each agent turn
            for i in range(1, len(conversation_log)):
                current_line = conversation_log[i]
                
                # Skip if not a proper agent turn
                if ':' not in current_line:
                    continue
                
                # Extract agent name and response
                agent_name, response = current_line.split(':', 1)
                agent_name = agent_name.strip()
                response = response.strip()
                
                if not response:
                    continue
                
                # Get conversation history
                history = conversation_log[:i]
                history_text = "\n".join(history)
                
                # Create instruction based on barrier context
                instruction = self._create_instruction(conv, rating, agent_name)
                
                # Create SFT example
                sft_example = {
                    "instruction": instruction,
                    "input": history_text,
                    "output": response,
                    "metadata": {
                        "conversation_id": conv.conversation_id,
                        "episode_type": conv.episode_type,
                        "barrier_type": conv.barrier_info.get("barrier_type") if conv.barrier_info else None,
                        "agent_name": agent_name,
                        "quality_score": rating.overall_quality,
                        "trajectory_type": conv.trajectory_type,
                        "timestamp": conv.timestamp
                    }
                }
                
                sft_data.append(sft_example)
        
        print(f"Converted to {len(sft_data)} SFT examples")
        return sft_data
    
    def _create_instruction(
        self,
        conversation: TrainingConversation,
        rating: ConversationRating,
        agent_name: str
    ) -> str:
        """Create instruction for training example"""
        
        # Base instruction
        instruction = "You are a socially intelligent conversational agent."
        
        # Add barrier-specific context
        if conversation.barrier_info:
            barrier_type = conversation.barrier_info.get("barrier_type")
            if barrier_type:
                barrier_instructions = {
                    "semantic_structure": "The other person uses vague, ambiguous language with complex sentences. Adapt your communication to maintain understanding and clarity.",
                    "cultural_style": "The other person uses indirect, high-context communication with hedges and politeness. Match their communication style appropriately while being clear.",
                    "emotional_influence": "The other person has a negative emotional tone and uses clipped responses. Stay patient and maintain the conversation constructively."
                }
                instruction += f" {barrier_instructions.get(barrier_type, '')}"
        
        # Add episode context
        instruction += f" Continue the conversation naturally while maintaining social appropriateness and working toward productive communication. Show emotional intelligence and adaptability in your response."
        
        return instruction
    
    def create_dataset_splits(
        self,
        sft_data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create train/validation/test splits following Sotopia-π methodology.
        
        Args:
            sft_data: SFT training data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        
        print(f"Creating dataset splits (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})")
        
        # Group by episode type for stratified splitting
        data_by_type = defaultdict(list)
        for example in sft_data:
            episode_type = example["metadata"]["episode_type"]
            data_by_type[episode_type].append(example)
        
        train_data, val_data, test_data = [], [], []
        
        for episode_type, examples in data_by_type.items():
            # Shuffle examples
            import random
            random.shuffle(examples)
            
            # Calculate split indices
            n_examples = len(examples)
            train_end = int(n_examples * train_ratio)
            val_end = train_end + int(n_examples * val_ratio)
            
            # Split data
            train_data.extend(examples[:train_end])
            val_data.extend(examples[train_end:val_end])
            test_data.extend(examples[val_end:])
            
            print(f"   {episode_type}: {n_examples} → train: {train_end}, val: {val_end - train_end}, test: {n_examples - val_end}")
        
        print(f"Dataset splits created:")
        print(f"   Train: {len(train_data)} examples")
        print(f"   Validation: {len(val_data)} examples")
        print(f"   Test: {len(test_data)} examples")
        
        return train_data, val_data, test_data
    
    def save_processed_data(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        filename_prefix: str = "social_decipher_sft"
    ):
        """Save processed data to files"""
        
        # Save training data
        train_file = os.path.join(self.output_dir, f"{filename_prefix}_train.json")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # Save validation data
        val_file = os.path.join(self.output_dir, f"{filename_prefix}_val.json")
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        # Save test data
        test_file = os.path.join(self.output_dir, f"{filename_prefix}_test.json")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved processed data:")
        print(f"   Train: {train_file}")
        print(f"   Validation: {val_file}")
        print(f"   Test: {test_file}")
        
        # Save dataset summary
        summary = {
            "total_examples": len(train_data) + len(val_data) + len(test_data),
            "train_examples": len(train_data),
            "val_examples": len(val_data),
            "test_examples": len(test_data),
            "episode_type_distribution": self._get_episode_type_distribution(train_data + val_data + test_data),
            "barrier_type_distribution": self._get_barrier_type_distribution(train_data + val_data + test_data)
        }
        
        summary_file = os.path.join(self.output_dir, f"{filename_prefix}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset summary: {summary_file}")
    
    def _get_episode_type_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get episode type distribution"""
        dist = defaultdict(int)
        for example in data:
            episode_type = example["metadata"]["episode_type"]
            dist[episode_type] += 1
        return dict(dist)
    
    def _get_barrier_type_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get barrier type distribution"""
        dist = defaultdict(int)
        for example in data:
            barrier_type = example["metadata"]["barrier_type"]
            if barrier_type:
                dist[barrier_type] += 1
        return dict(dist)


def main():
    parser = argparse.ArgumentParser(description="Preprocess training data following Sotopia-π methodology")
    parser.add_argument("--input_file", type=str, required=True, help="Input conversations JSON file")
    parser.add_argument("--ratings_file", type=str, required=True, help="Input ratings JSON file")
    parser.add_argument("--output_dir", type=str, default="training_data", help="Output directory")
    parser.add_argument("--quality_threshold", type=float, default=6.0, help="Quality threshold")
    parser.add_argument("--filter_strategy", type=str, default="absolute", choices=["absolute", "relative"], help="Filtering strategy")
    parser.add_argument("--balance_dataset", action="store_true", help="Balance dataset across episode types")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading conversations from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        conversations_data = json.load(f)
    conversations = [TrainingConversation(**conv) for conv in conversations_data]
    
    print(f"Loading ratings from {args.ratings_file}")
    with open(args.ratings_file, 'r', encoding='utf-8') as f:
        ratings_data = json.load(f)
    ratings = [ConversationRating(**rating) for rating in ratings_data]
    
    # Initialize processor
    processor = SotopiaStyleDataProcessor(args.output_dir)
    
    # Filter conversations
    filtered_conversations = processor.filter_episodes_by_quality(
        conversations, ratings, args.quality_threshold, args.filter_strategy
    )
    
    # Balance dataset if requested
    if args.balance_dataset:
        filtered_conversations = processor.balance_dataset(filtered_conversations)
    
    # Convert to SFT format
    sft_data = processor.convert_to_sft_format(filtered_conversations, ratings)
    
    # Create dataset splits
    train_data, val_data, test_data = processor.create_dataset_splits(
        sft_data, args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    # Save processed data
    processor.save_processed_data(train_data, val_data, test_data)
    
    print("Data preprocessing completed!")


if __name__ == "__main__":
    main()
