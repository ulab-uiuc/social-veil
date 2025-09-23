"""
Scoring Strategy Module for Social-Decipher Training
Provides extensible scoring and filtering strategies for training data selection.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from .data_collector import TrainingConversation
from .conversation_rater import ConversationRating


@dataclass
class ScoringConfig:
    """Configuration for scoring strategy"""
    # Basic filtering
    quality_threshold: float = 6.0
    filter_top_k: int = 2
    min_conversation_length: int = 5
    max_conversation_length: int = 50
    
    # Scoring dimensions and weights
    scoring_dimensions: List[str] = None
    dimension_weights: Dict[str, float] = None
    
    # Custom scoring functions
    custom_score_function: Optional[Callable] = None
    composite_score_function: Optional[Callable] = None
    
    # Filtering strategies
    filter_strategy: str = "absolute"  # "absolute", "relative", "adaptive"
    
    def __post_init__(self):
        if self.scoring_dimensions is None:
            self.scoring_dimensions = [
                "overall_quality",
                "barrier_handling", 
                "social_intelligence",
                "communication_effectiveness",
                "goal_achievement"
            ]
        
        if self.dimension_weights is None:
            # Default: equal weights, but overall_quality has higher weight
            self.dimension_weights = {
                "overall_quality": 1.0,
                "barrier_handling": 0.8,
                "social_intelligence": 0.6,
                "communication_effectiveness": 0.6,
                "goal_achievement": 0.6
            }


class ScoringStrategy(ABC):
    """Abstract base class for scoring strategies"""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
    
    @abstractmethod
    def calculate_composite_score(self, rating: ConversationRating) -> float:
        """Calculate a composite score from multiple dimensions"""
        pass
    
    @abstractmethod
    def should_include_conversation(self, rating: ConversationRating, context: Dict[str, Any] = None) -> bool:
        """Determine if a conversation should be included in training"""
        pass
    
    @abstractmethod
    def rank_conversations(
        self, 
        conversations: List[TrainingConversation], 
        ratings: List[ConversationRating]
    ) -> List[Tuple[TrainingConversation, float]]:
        """Rank conversations by score for top-k selection"""
        pass


class DefaultScoringStrategy(ScoringStrategy):
    """Default scoring strategy using overall_quality as primary filter"""
    
    def calculate_composite_score(self, rating: ConversationRating) -> float:
        """
        Default: use overall_quality as the main score
        Can be easily overridden to use weighted combination
        """
        if self.config.composite_score_function:
            return self.config.composite_score_function(rating)
        
        # Default implementation: use overall_quality
        return rating.overall_quality
    
    def should_include_conversation(self, rating: ConversationRating, context: Dict[str, Any] = None) -> bool:
        """
        Default: filter by quality threshold from the main config.
        The context dictionary is ignored in the default strategy.
        """
        composite_score = self.calculate_composite_score(rating)
        return composite_score >= self.config.quality_threshold
    
    def rank_conversations(
        self, 
        conversations: List[TrainingConversation], 
        ratings: List[ConversationRating]
    ) -> List[Tuple[TrainingConversation, float]]:
        """
        Default: rank by composite score
        """
        rating_map = {r.conversation_id: r for r in ratings}
        
        ranked_conversations = []
        for conv in conversations:
            rating = rating_map.get(conv.conversation_id)
            if rating:
                score = self.calculate_composite_score(rating)
                ranked_conversations.append((conv, score))
        
        # Sort by score (descending)
        ranked_conversations.sort(key=lambda x: x[1], reverse=True)
        return ranked_conversations


class WeightedScoringStrategy(ScoringStrategy):
    """Weighted combination of multiple scoring dimensions"""
    
    def calculate_composite_score(self, rating: ConversationRating) -> float:
        """
        Calculate a normalized, weighted composite score.
        This version correctly handles metrics with different scales (e.g., [1, 5], [1, 10], [-5, 5])
        by normalizing them to a 0-1 range before applying weights.
        """
        if self.config.composite_score_function:
            return self.config.composite_score_function(rating)
        
        total_score = 0.0
        total_weight = 0.0
        
        # Define the min/max scale for each metric
        metric_scales = {
            "goal_completion":      {"min": 1, "max": 10},
            "believability":        {"min": 1, "max": 10},
            "relationship":         {"min": -5, "max": 5},
            "unresolved_confusion": {"min": 1, "max": 5},
            "mutual_understanding": {"min": 1, "max": 5}
        }

        for dimension in self.config.scoring_dimensions:
            score = None
            if hasattr(rating, dimension):
                score = getattr(rating, dimension)
            elif rating.episode_level and dimension in rating.episode_level:
                score = rating.episode_level[dimension]

            if score is not None:
                scale = metric_scales.get(dimension, {"min": 1, "max": 10})
                min_score, max_score = scale["min"], scale["max"]
                
                # Universal normalization to a 0-1 range
                if (max_score - min_score) == 0:
                    normalized_score = 0.5 # Avoid division by zero for constant-value metrics
                else:
                    normalized_score = (score - min_score) / (max_score - min_score)
                
                weight = self.config.dimension_weights.get(dimension, 1.0)
                total_score += normalized_score * weight
                total_weight += abs(weight)
        
        # Return the final score, scaled to a 1-10 range for interpretability
        # A score of 0 on the normalized scale becomes 1, 0.5 becomes 5.5, and 1 becomes 10.
        final_score = (total_score / total_weight) * 9 + 1 if total_weight > 0 else 1.0
        return final_score
    
    def should_include_conversation(self, rating: ConversationRating, context: Dict[str, Any] = None) -> bool:
        """
        For weighted scoring, the decision is based on the composite score, 
        which already takes multiple dimensions into account.
        The context dictionary is ignored here as well.
        """
        composite_score = self.calculate_composite_score(rating)
        return composite_score >= self.config.quality_threshold
    
    def rank_conversations(
        self, 
        conversations: List[TrainingConversation], 
        ratings: List[ConversationRating]
    ) -> List[Tuple[TrainingConversation, float]]:
        """
        Rank by weighted composite score
        """
        rating_map = {r.conversation_id: r for r in ratings}
        
        ranked_conversations = []
        for conv in conversations:
            rating = rating_map.get(conv.conversation_id)
            if rating:
                score = self.calculate_composite_score(rating)
                ranked_conversations.append((conv, score))
        
        ranked_conversations.sort(key=lambda x: x[1], reverse=True)
        return ranked_conversations


class AdaptiveScoringStrategy(ScoringStrategy):
    """Adaptive scoring that adjusts based on data distribution"""
    
    def __init__(self, config: ScoringConfig):
        super().__init__(config)
        self.score_statistics = None
    
    def _update_statistics(self, ratings: List[ConversationRating]):
        """Update scoring statistics for adaptive thresholding"""
        if not ratings:
            return
        
        scores = [self.calculate_composite_score(rating) for rating in ratings]
        self.score_statistics = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": np.median(scores),
            "q75": np.percentile(scores, 75),
            "q90": np.percentile(scores, 90)
        }
    
    def calculate_composite_score(self, rating: ConversationRating) -> float:
        """
        Calculate composite score with custom weighting
        """
        if self.config.composite_score_function:
            return self.config.composite_score_function(rating)
        
        # Focus on barrier handling and overall quality
        barrier_weight = 0.4
        quality_weight = 0.6
        
        return (rating.barrier_handling * barrier_weight + 
                rating.overall_quality * quality_weight)
    
    def should_include_conversation(self, rating: ConversationRating, context: Dict[str, Any] = None) -> bool:
        """
        Adaptive filtering based on data distribution
        """
        composite_score = self.calculate_composite_score(rating)
        
        if self.score_statistics is None:
            # Fallback to threshold
            return composite_score >= self.config.quality_threshold
        
        # Adaptive threshold: use 75th percentile or configured threshold, whichever is lower
        adaptive_threshold = min(
            self.config.quality_threshold,
            self.score_statistics["q75"]
        )
        
        return composite_score >= adaptive_threshold
    
    def rank_conversations(
        self, 
        conversations: List[TrainingConversation], 
        ratings: List[ConversationRating]
    ) -> List[Tuple[TrainingConversation, float]]:
        """
        Rank with adaptive scoring
        """
        # Update statistics first
        self._update_statistics(ratings)
        
        rating_map = {r.conversation_id: r for r in ratings}
        
        ranked_conversations = []
        for conv in conversations:
            rating = rating_map.get(conv.conversation_id)
            if rating:
                score = self.calculate_composite_score(rating)
                ranked_conversations.append((conv, score))
        
        ranked_conversations.sort(key=lambda x: x[1], reverse=True)
        return ranked_conversations


class CustomScoringStrategy(ScoringStrategy):
    """Fully customizable scoring strategy"""
    
    def __init__(self, config: ScoringConfig, custom_functions: Dict[str, Callable] = None):
        super().__init__(config)
        self.custom_functions = custom_functions or {}
    
    def calculate_composite_score(self, rating: ConversationRating) -> float:
        if "composite_score" in self.custom_functions:
            return self.custom_functions["composite_score"](rating, self.config)
        elif self.config.composite_score_function:
            return self.config.composite_score_function(rating)
        else:
            return rating.overall_quality
    
    def should_include_conversation(self, rating: ConversationRating, context: Dict[str, Any] = None) -> bool:
        """
        Use the custom filtering logic that leverages the context dictionary for specific thresholds.
        """
        if "filter_function" in self.custom_functions:
            return self.custom_functions["filter_function"](rating, self.config, context)
        
        # Fallback for custom strategies that don't define a specific filter function.
        # This is where we implement the direct metric thresholding.
        context = context or {}
        goal_threshold = context.get("goal_threshold", 5.5)
        understanding_threshold = context.get("understanding_threshold", 3.0)
        confusion_threshold = context.get("confusion_threshold", 2.0)

        if not rating.episode_level:
            return False

        goal_score = float(rating.episode_level.get("goal_completion", 0.0))
        mutual_understanding = float(rating.episode_level.get("mutual_understanding", 0.0))
        unresolved_confusion = float(rating.episode_level.get("unresolved_confusion", 0.0))

        # Apply the direct filtering logic
        return (goal_score >= goal_threshold and
                mutual_understanding >= understanding_threshold and
                unresolved_confusion >= confusion_threshold)
    
    def rank_conversations(
        self, 
        conversations: List[TrainingConversation], 
        ratings: List[ConversationRating]
    ) -> List[Tuple[TrainingConversation, float]]:
        """
        Use custom ranking function if provided
        """
        if "ranking_function" in self.custom_functions:
            return self.custom_functions["ranking_function"](conversations, ratings, self.config)
        
        # Default ranking
        rating_map = {r.conversation_id: r for r in ratings}
        
        ranked_conversations = []
        for conv in conversations:
            rating = rating_map.get(conv.conversation_id)
            if rating:
                score = self.calculate_composite_score(rating)
                ranked_conversations.append((conv, score))
        
        ranked_conversations.sort(key=lambda x: x[1], reverse=True)
        return ranked_conversations


class ScoringManager:
    """Manager class for different scoring strategies"""
    
    STRATEGIES = {
        "default": DefaultScoringStrategy,
        "weighted": WeightedScoringStrategy,
        "adaptive": AdaptiveScoringStrategy,
        "custom": CustomScoringStrategy,
        "custom_barrier_focused": CustomScoringStrategy,
    }
    
    def __init__(self, strategy_name: str = "default", config: ScoringConfig = None, **kwargs):
        self.config = config or ScoringConfig()
        self.strategy_name = strategy_name
        
        if strategy_name not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(self.STRATEGIES.keys())}")
        
        strategy_class = self.STRATEGIES[strategy_name]
        self.strategy = strategy_class(self.config, **kwargs)
    
    def filter_conversations(
        self, 
        conversations: List[TrainingConversation], 
        ratings: List[ConversationRating],
        context: Dict[str, Any] = None
    ) -> List[TrainingConversation]:
        """
        Filter conversations using the selected strategy's logic.
        """
        print(f"Filtering with {self.strategy_name} strategy")
        
        rating_map: Dict[str, ConversationRating] = {r.conversation_id: r for r in ratings}
        filtered_conversations = []

        for conv in conversations:
            rating = rating_map.get(conv.conversation_id)
            if not rating:
                continue

            # Defer the filtering decision to the specific strategy
            if self.strategy.should_include_conversation(rating, context):
                filtered_conversations.append(conv)
        
        print(f"Filtered {len(conversations)} -> {len(filtered_conversations)} conversations.")
        return filtered_conversations
    
    def apply_top_k_filtering(
        self, 
        conversations: List[TrainingConversation], 
        ratings: List[ConversationRating]
    ) -> List[TrainingConversation]:
        """
        Apply top-k filtering per episode type
        """
        print(f"🔝 Applying top-{self.config.filter_top_k} filtering per episode type")
        
        # Group by episode type
        conversations_by_type = {}
        for conv in conversations:
            episode_type = conv.episode_type
            if episode_type not in conversations_by_type:
                conversations_by_type[episode_type] = []
            conversations_by_type[episode_type].append(conv)
        
        # Apply top-k per type
        filtered_conversations = []
        for episode_type, convs in conversations_by_type.items():
            # Rank conversations for this episode type
            ranked_convs = self.strategy.rank_conversations(convs, ratings)
            
            # Take top-k
            top_k_convs = ranked_convs[:self.config.filter_top_k]
            filtered_conversations.extend([conv for conv, _ in top_k_convs])
            
            print(f"   {episode_type}: {len(convs)} → {len(top_k_convs)} (top-{self.config.filter_top_k})")
        
        return filtered_conversations
    
    def get_conversation_scores(
        self, 
        conversations: List[TrainingConversation], 
        ratings: List[ConversationRating]
    ) -> Dict[str, float]:
        """
        Get composite scores for all conversations
        """
        scores = {}
        rating_map = {r.conversation_id: r for r in ratings}
        
        for conv in conversations:
            rating = rating_map.get(conv.conversation_id)
            if rating:
                score = self.strategy.calculate_composite_score(rating)
                scores[conv.conversation_id] = score
        
        return scores


# Pre-defined scoring configurations
def get_default_scoring_config() -> ScoringConfig:
    """Get default scoring configuration"""
    return ScoringConfig(
        quality_threshold=6.0,
        filter_top_k=2,
        scoring_dimensions=["overall_quality", "barrier_handling", "social_intelligence"],
        dimension_weights={"overall_quality": 1.0, "barrier_handling": 0.8, "social_intelligence": 0.6}
    )


def get_barrier_focused_config() -> ScoringConfig:
    """Get barrier-focused scoring configuration"""
    return ScoringConfig(
        quality_threshold=6.0,
        filter_top_k=3,
        scoring_dimensions=["barrier_handling", "overall_quality", "communication_effectiveness"],
        dimension_weights={"barrier_handling": 1.0, "overall_quality": 0.8, "communication_effectiveness": 0.6}
    )


def get_balanced_config() -> ScoringConfig:
    """Get balanced scoring configuration"""
    return ScoringConfig(
        quality_threshold=6.5,
        filter_top_k=2,
        scoring_dimensions=["overall_quality", "barrier_handling", "social_intelligence", "communication_effectiveness", "goal_achievement"],
        dimension_weights={
            "overall_quality": 0.3,
            "barrier_handling": 0.25,
            "social_intelligence": 0.2,
            "communication_effectiveness": 0.15,
            "goal_achievement": 0.1
        }
    )


def get_custom_barrier_focused_config() -> ScoringConfig:
    """
    Get a custom, barrier-focused scoring configuration.
    This version uses corrected weights and prioritizes barrier-handling metrics.
    """
    return ScoringConfig(
        quality_threshold=5.5,
        filter_top_k=3,
        scoring_dimensions=[
            "goal_completion",
            "believability",
            "relationship",
            "unresolved_confusion",
            "mutual_understanding"
        ],
        dimension_weights={
            "goal_completion": 1.0,
            "mutual_understanding": 1.2, # Increased weight
            "unresolved_confusion": 1.2,  # Corrected from negative and increased weight
            "believability": 0.8,
            "relationship": 0.6,
        }
    )


# Example custom scoring functions
def barrier_priority_score(rating: ConversationRating) -> float:
    """Custom scoring function that prioritizes barrier handling"""
    return (rating.barrier_handling * 0.5 + 
            rating.overall_quality * 0.3 + 
            rating.social_intelligence * 0.2)


def adaptive_quality_filter(rating: ConversationRating, config: ScoringConfig, context: Dict[str, Any] = None) -> bool:
    """Custom filter that adapts based on conversation context"""
    base_score = rating.overall_quality
    
    # Bonus for good barrier handling
    if rating.barrier_handling >= 7.0:
        base_score += 0.5
    
    # Penalty for poor communication
    if rating.communication_effectiveness < 5.0:
        base_score -= 1.0
    
    return base_score >= config.quality_threshold


if __name__ == "__main__":
    # Example usage
    config = get_barrier_focused_config()
    manager = ScoringManager("weighted", config)
    
    # Example with custom functions
    custom_functions = {
        "composite_score": barrier_priority_score,
        "filter_function": adaptive_quality_filter
    }
    custom_manager = ScoringManager("custom", config, custom_functions=custom_functions)
