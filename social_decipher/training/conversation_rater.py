"""
Conversation Rating for Interactive Training
Implements GPT-4 based conversation quality assessment
following Sotopia-π methodology but adapted for barrier-aware social intelligence.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from openai import OpenAI
import os
import yaml

from .data_collector import TrainingConversation


@dataclass
class ConversationRating:
    """Rating for a single conversation, mirroring eval_result structure"""
    conversation_id: str
    agent_1: Dict[str, Any]
    agent_2: Dict[str, Any]
    interaction_quality: float = 0.0
    episode_level: Dict[str, Any] = None


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
    ) -> List[Dict[str, Any]]:
        """
        Build ratings from existing eval_result and return as dictionaries.
        Skips external re-rating; conversations without eval_result are ignored.
        """
        ratings: List[Dict[str, Any]] = []
        print(f"Rating {len(conversations)} conversations (no re-rating; using existing eval_result)...")
        
        rated_count = 0
        for i, conversation in enumerate(conversations):
            print(f"Rating conversation {i+1}/{len(conversations)}")
            try:
                rating_obj = self._rating_from_eval(conversation)
                if rating_obj is None:
                    print("  WARNING: Missing or malformed eval_result; skipping.")
                    continue
                
                # Convert the object to a dictionary before appending
                ratings.append(asdict(rating_obj))
                rated_count += 1
                
                # Use agent_2's overall score for a summary printout
                overall_score = rating_obj.agent_2.get('overall', 0.0)
                print(f"  Overall Score (Agent 2): {overall_score:.1f}/10")

            except Exception as e:
                print(f"  WARNING: Failed to build rating from eval_result: {e}")
                continue
        
        print(f"\nResults: Successfully rated {rated_count}/{len(conversations)} conversations.")
        
        return ratings
    
    def _rating_from_eval(self, conversation: TrainingConversation) -> Optional[ConversationRating]:
        """Construct ConversationRating from conversation.eval_result; return None if unavailable."""
        ev = conversation.eval_result
        if not ev or not isinstance(ev, dict):
            return None
        agg = ev.get("aggregated_scores", {})
        if not agg:
            return None
        
        agent1_data = agg.get("agent_1", {})
        agent2_data = agg.get("agent_2", {})
        ep_level_data = agg.get("episode_level", {})

        # Ensure required nested structures are present
        if not agent1_data or not agent2_data or not ep_level_data:
            return None
        
        return ConversationRating(
            conversation_id=conversation.conversation_id,
            agent_1=agent1_data,
            agent_2=agent2_data,
            interaction_quality=float(agg.get("interaction_quality", 0.0)),
            episode_level=ep_level_data
        )

    def save_ratings(self, ratings: List[Dict[str, Any]], filepath: str):
        """Save ratings to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ratings, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(ratings)} ratings to {filepath}")
    
    def analyze_ratings(self, ratings: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not ratings:
            return {}
        
        num_convs = len(ratings)

        # Analyze Agent 2 scores, as it's typically the agent being trained
        agent2_scores = [r['agent_2'] for r in ratings if 'agent_2' in r and isinstance(r['agent_2'], dict)]
        avg_goal = 0.0
        avg_believability = 0.0
        avg_relationship = 0.0
        avg_overall = 0.0
        if agent2_scores:
            avg_goal = sum(s.get('goal_completion', 0.0) for s in agent2_scores) / len(agent2_scores)
            avg_believability = sum(s.get('believability', 0.0) for s in agent2_scores) / len(agent2_scores)
            avg_relationship = sum(s.get('relationship', 0.0) for s in agent2_scores) / len(agent2_scores)
            avg_overall = sum(s.get('overall', 0.0) for s in agent2_scores) / len(agent2_scores)

        # Analyze episode level scores
        ep_level_scores = [r['episode_level'] for r in ratings if 'episode_level' in r and isinstance(r['episode_level'], dict)]
        avg_understanding = 0.0
        avg_confusion = 0.0
        if ep_level_scores:
            avg_understanding = sum(s.get('mutual_understanding', 0.0) for s in ep_level_scores) / len(ep_level_scores)
            avg_confusion = sum(s.get('unresolved_confusion', 0.0) for s in ep_level_scores) / len(ep_level_scores)
        
        analysis = {
            "total_conversations": num_convs,
            "average_agent_2_scores": {
                "overall": avg_overall,
                "goal_completion": avg_goal,
                "believability": avg_believability,
                "relationship": avg_relationship,
            },
            "average_episode_scores": {
                "mutual_understanding": avg_understanding,
                "unresolved_confusion": avg_confusion,
            }
        }
        
        print("\nRating Analysis (Agent 2 & Episode):")
        print(f"   Total conversations: {analysis['total_conversations']}")
        print(f"   Average Agent 2 Scores:")
        print(f"     Overall: {avg_overall:.1f}/10")
        print(f"     Goal Completion: {avg_goal:.1f}/10")
        print(f"     Believability: {avg_believability:.1f}/10")
        print(f"   Average Episode Scores:")
        print(f"     Mutual Understanding: {avg_understanding:.1f}/5")
        print(f"     Unresolved Confusion: {avg_confusion:.1f}/5")
        
        return analysis