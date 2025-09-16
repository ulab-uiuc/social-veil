"""
Conversation Rating for Interactive Training
Implements GPT-4 based conversation quality assessment
following Sotopia-π methodology but adapted for barrier-aware social intelligence.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
import os
import yaml

from .data_collector import TrainingConversation


@dataclass
class ConversationRating:
    """Rating for a single conversation"""
    conversation_id: str
    overall_quality: float  # 0-10 scale
    barrier_handling: float  # 0-10 scale 
    social_intelligence: float  # 0-10 scale
    communication_effectiveness: float  # 0-10 scale
    goal_achievement: float  # 0-10 scale
    explanation: str
    is_positive: bool  # Whether to include in training data
    
    # New metrics for barrier-focused scoring strategy
    goal_completion: float = 0.0  # 0-10 scale
    believability: float = 0.0  # 0-10 scale
    relationship: float = 0.0  # -5 to 5 scale
    episode_level: Dict[str, Any] = None  # Contains unresolved_confusion and mutual_understanding


class ConversationRater:

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        # Load main config to get the evaluator-specific API key
        main_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")
        with open(main_config_path) as config_file:
            main_config = yaml.safe_load(config_file)
        
        evaluator_api_key = main_config.get("EVALUATOR_OPENAI_API_KEY")
        if not evaluator_api_key:
            evaluator_api_key = main_config.get("AGENT_OPENAI_API_KEY")
        
        # Keep client available, but we will avoid re-rating by default
        self.client = OpenAI(api_key=evaluator_api_key) if evaluator_api_key else None
        self.model = model
        self.temperature = temperature
        
    def rate_conversations(
        self, 
        conversations: List[TrainingConversation],
        quality_threshold: float = 6.0
    ) -> List[ConversationRating]:
        """
        Build ratings from existing eval_result when available.
        Skips external re-rating; conversations without eval_result are ignored.
        """
        ratings: List[ConversationRating] = []
        print(f"Rating {len(conversations)} conversations (no re-rating; using existing eval_result)...")
        
        for i, conversation in enumerate(conversations):
            print(f"Rating conversation {i+1}/{len(conversations)}")
            try:
                rating = self._rating_from_eval(conversation)
                if rating is None:
                    print("  WARNING: Missing eval_result; skipping.")
                    continue
                rating.is_positive = rating.overall_quality >= quality_threshold
                ratings.append(rating)
                quality_emoji = "PASS" if rating.is_positive else "FAIL"
                print(f"  {quality_emoji} Quality: {rating.overall_quality:.1f}/10")
            except Exception as e:
                print(f"  WARNING: Failed to build rating from eval_result: {e}")
                continue
        
        positive_count = sum(1 for r in ratings if r.is_positive)
        print(f"\nResults: {positive_count}/{len(ratings)} conversations above threshold ({quality_threshold})")
        
        return ratings
    
    def _rating_from_eval(self, conversation: TrainingConversation) -> Optional[ConversationRating]:
        """Construct ConversationRating from conversation.eval_result; return None if unavailable."""
        ev = conversation.eval_result
        if not ev or not isinstance(ev, dict):
            return None
        agg = ev.get("aggregated_scores", {})
        agent2 = agg.get("agent_2", {})
        if not agent2:
            return None
        # Episode-level metrics
        episode_level = agg.get("episode_level", {}) or {}
        
        # Pull primary metrics from agent_2 (Sotopia evaluator output)
        overall = float(agent2.get("overall", 0) or 0)
        goal_completion = float(agent2.get("goal_completion", 0) or 0)
        believability = float(agent2.get("believability", 0) or 0)
        relationship = float(agent2.get("relationship", 0) or 0)  # expected [-5,5]
        
        # Heuristic mappings for legacy fields used by default strategy
        # barrier_handling: use interaction_quality if available, else overall
        interaction_quality = float(agg.get("interaction_quality", 0) or 0)
        barrier_handling = interaction_quality if interaction_quality > 0 else overall
        # social_intelligence: use believability
        social_intelligence = believability
        # communication_effectiveness: map mutual_understanding (1..5) to 1..10
        mu = episode_level.get("mutual_understanding")
        if isinstance(mu, (int, float)) and mu >= 1:
            communication_effectiveness = (float(mu) - 1.0) / 4.0 * 9.0 + 1.0
        else:
            # fallback: scale relationship (-5..5) to ~1..10
            communication_effectiveness = ((relationship + 5.0) / 10.0) * 9.0 + 1.0
        # goal_achievement aligns with goal_completion
        goal_achievement = goal_completion
        
        explanation = "derived from eval_result"
        
        return ConversationRating(
            conversation_id=conversation.conversation_id,
            overall_quality=overall,
            barrier_handling=barrier_handling,
            social_intelligence=social_intelligence,
            communication_effectiveness=communication_effectiveness,
            goal_achievement=goal_achievement,
            explanation=explanation,
            is_positive=False,
            goal_completion=goal_completion,
            believability=believability,
            relationship=relationship,
            episode_level={
                "unresolved_confusion": episode_level.get("unresolved_confusion"),
                "mutual_understanding": episode_level.get("mutual_understanding")
            }
        )

    # The old _rate_single_conversation is retained for optional future use but unused now.
    def _rate_single_conversation(self, conversation: TrainingConversation) -> ConversationRating:
        """Deprecated in no-rerating mode; kept for compatibility if needed."""
        raise RuntimeError("Re-rating is disabled. Use existing eval_result instead.")

    def _prepare_conversation_context(self, conversation: TrainingConversation) -> Dict[str, Any]:
        """Prepare conversation context for rating (unused in no-rerating mode)"""
        context = {
            "episode_type": conversation.episode_type,
            "trajectory_type": conversation.trajectory_type,
            "conversation_length": len(conversation.conversation_log),
            "barrier_info": conversation.barrier_info or {},
            "eval_metrics": conversation.eval_result
        }
        if conversation.barrier_info:
            barrier_type = conversation.barrier_info.get("barrier_type", "")
            if barrier_type:
                context["barrier_description"] = self._get_barrier_description(barrier_type)
        return context

    def _get_barrier_description(self, barrier_type: str) -> str:
        descriptions = {
            "semantic_structure": "Agent A uses vague, ambiguous language with complex sentence structures",
            "cultural_style": "Agent A uses indirect, high-context communication style with hedges and politeness",
            "emotional_influence": "Agent A maintains negative emotional tone with clipped, sharp responses"
        }
        return descriptions.get(barrier_type, "No specific barrier")

    def filter_positive_conversations(
        self, 
        conversations: List[TrainingConversation],
        ratings: List[ConversationRating]
    ) -> List[TrainingConversation]:
        rating_map = {r.conversation_id: r for r in ratings}
        positive_conversations = []
        for conv in conversations:
            rating = rating_map.get(conv.conversation_id)
            if rating and rating.is_positive:
                positive_conversations.append(conv)
        print(f"Filtered to {len(positive_conversations)}/{len(conversations)} positive conversations")
        return positive_conversations
    
    def save_ratings(self, ratings: List[ConversationRating], filepath: str):
        """Save ratings to file"""
        rating_data = []
        for rating in ratings:
            rating_data.append({
                "conversation_id": rating.conversation_id,
                "overall_quality": rating.overall_quality,
                "barrier_handling": rating.barrier_handling,
                "social_intelligence": rating.social_intelligence,
                "communication_effectiveness": rating.communication_effectiveness,
                "goal_achievement": rating.goal_achievement,
                "explanation": rating.explanation,
                "is_positive": rating.is_positive
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rating_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(ratings)} ratings to {filepath}")
    
    def analyze_ratings(self, ratings: List[ConversationRating]) -> Dict[str, Any]:
        if not ratings:
            return {}
        avg_overall = sum(r.overall_quality for r in ratings) / len(ratings)
        avg_barrier = sum(r.barrier_handling for r in ratings) / len(ratings)
        avg_social = sum(r.social_intelligence for r in ratings) / len(ratings)
        avg_communication = sum(r.communication_effectiveness for r in ratings) / len(ratings)
        avg_goal = sum(r.goal_achievement for r in ratings) / len(ratings)
        positive_count = sum(1 for r in ratings if r.is_positive)
        positive_rate = positive_count / len(ratings)
        analysis = {
            "total_conversations": len(ratings),
            "positive_conversations": positive_count,
            "positive_rate": positive_rate,
            "average_ratings": {
                "overall_quality": avg_overall,
                "barrier_handling": avg_barrier,
                "social_intelligence": avg_social,
                "communication_effectiveness": avg_communication,
                "goal_achievement": avg_goal
            }
        }
        print("\nRating Analysis:")
        print(f"   Total conversations: {analysis['total_conversations']}")
        print(f"   Positive rate: {positive_rate:.1%}")
        print(f"   Average ratings:")
        print(f"     Overall Quality: {avg_overall:.1f}/10")
        print(f"     Barrier Handling: {avg_barrier:.1f}/10")
        print(f"     Social Intelligence: {avg_social:.1f}/10")
        print(f"     Communication: {avg_communication:.1f}/10")
        print(f"     Goal Achievement: {avg_goal:.1f}/10")
        return analysis