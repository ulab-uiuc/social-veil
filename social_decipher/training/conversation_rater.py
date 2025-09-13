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


class ConversationRater:
    """
    Rates conversation quality for training data filtering.
    
    Inspired by Sotopia-π rating system but specialized for:
    1. Barrier Communication Quality: How well Agent B adapts to Agent A's barriers
    2. Social Intelligence: Maintaining relationship despite communication challenges  
    3. Goal Achievement: Success in reaching conversation objectives
    4. Learning Value: Whether conversation provides good training signal
    """
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        # Load main config to get the evaluator-specific API key
        main_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")
        with open(main_config_path) as config_file:
            main_config = yaml.safe_load(config_file)
        
        evaluator_api_key = main_config.get("EVALUATOR_OPENAI_API_KEY")
        if not evaluator_api_key:
            # Fallback to the agent key if the evaluator key is not found
            evaluator_api_key = main_config.get("AGENT_OPENAI_API_KEY")
        
        if not evaluator_api_key:
            raise ValueError("No OpenAI API key found in config.yaml (checked for EVALUATOR_OPENAI_API_KEY and AGENT_OPENAI_API_KEY)")

        self.client = OpenAI(api_key=evaluator_api_key)
        self.model = model
        self.temperature = temperature
        
    def rate_conversations(
        self, 
        conversations: List[TrainingConversation],
        quality_threshold: float = 6.0
    ) -> List[ConversationRating]:
        """Rate a batch of conversations for training data filtering"""
        
        ratings = []
        print(f"Rating {len(conversations)} conversations...")
        
        for i, conversation in enumerate(conversations):
            print(f"Rating conversation {i+1}/{len(conversations)}")
            
            try:
                rating = self._rate_single_conversation(conversation)
                rating.is_positive = rating.overall_quality >= quality_threshold
                ratings.append(rating)
                
                quality_emoji = "PASS" if rating.is_positive else "FAIL"
                print(f"  {quality_emoji} Quality: {rating.overall_quality:.1f}/10")
                
            except Exception as e:
                print(f"  WARNING: Rating failed: {e}")
                continue
                
        positive_count = sum(1 for r in ratings if r.is_positive)
        print(f"\nResults: {positive_count}/{len(ratings)} conversations above threshold ({quality_threshold})")
        
        return ratings
    
    def _rate_single_conversation(self, conversation: TrainingConversation) -> ConversationRating:
        """Rate a single conversation using GPT-4"""
        
        # Prepare conversation context
        context = self._prepare_conversation_context(conversation)
        
        # Create rating prompt
        prompt = self._create_rating_prompt(conversation, context)
        
        # Get GPT-4 rating
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_rating_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        
        # Parse rating response
        rating_data = json.loads(response.choices[0].message.content)
        
        return ConversationRating(
            conversation_id=conversation.conversation_id,
            overall_quality=float(rating_data.get("overall_quality", 0)),
            barrier_handling=float(rating_data.get("barrier_handling", 0)),
            social_intelligence=float(rating_data.get("social_intelligence", 0)),
            communication_effectiveness=float(rating_data.get("communication_effectiveness", 0)),
            goal_achievement=float(rating_data.get("goal_achievement", 0)),
            explanation=rating_data.get("explanation", ""),
            is_positive=False  # Will be set by caller based on threshold
        )
    
    def _prepare_conversation_context(self, conversation: TrainingConversation) -> Dict[str, Any]:
        """Prepare conversation context for rating"""
        
        # Extract key information
        context = {
            "episode_type": conversation.episode_type,
            "trajectory_type": conversation.trajectory_type,
            "conversation_length": len(conversation.conversation_log),
            "barrier_info": conversation.barrier_info or {},
            "eval_metrics": conversation.eval_result
        }
        
        # Add barrier-specific context
        if conversation.barrier_info:
            barrier_type = conversation.barrier_info.get("barrier_type", "")
            if barrier_type:
                context["barrier_description"] = self._get_barrier_description(barrier_type)
        
        return context
    
    def _get_barrier_description(self, barrier_type: str) -> str:
        """Get description of barrier type for context"""
        descriptions = {
            "semantic_structure": "Agent A uses vague, ambiguous language with complex sentence structures",
            "cultural_style": "Agent A uses indirect, high-context communication style with hedges and politeness",
            "emotional_influence": "Agent A maintains negative emotional tone with clipped, sharp responses"
        }
        return descriptions.get(barrier_type, "No specific barrier")
    
    def _create_rating_prompt(self, conversation: TrainingConversation, context: Dict[str, Any]) -> str:
        """Create rating prompt for GPT-4"""
        
        # Format conversation
        conversation_text = "\n".join(conversation.conversation_log)
        
        # Get barrier context
        barrier_desc = context.get("barrier_description", "No barriers present")
        
        return f"""
Please rate this social conversation on multiple dimensions (0-10 scale):

**Conversation Context:**
- Episode Type: {conversation.episode_type}
- Trajectory Type: {conversation.trajectory_type} 
- Barrier Situation: {barrier_desc}
- Conversation Length: {context['conversation_length']} turns

**Conversation:**
{conversation_text}

**Rating Criteria:**

1. **Overall Quality (0-10)**: General conversation quality, naturalness, and coherence
2. **Barrier Handling (0-10)**: How well Agent B adapts to and manages Agent A's communication barriers
3. **Social Intelligence (0-10)**: Maintenance of social relationship despite communication challenges
4. **Communication Effectiveness (0-10)**: Success in exchanging information and understanding
5. **Goal Achievement (0-10)**: Progress toward conversation objectives despite barriers

**Instructions:**
- Rate each dimension 0-10 (higher = better)
- Focus on Agent B's adaptive communication skills when barriers are present
- Consider whether this conversation would be valuable for training social intelligence
- Provide brief explanation of ratings

Output your rating as JSON with this exact format:
{{
    "overall_quality": <0-10>,
    "barrier_handling": <0-10>,
    "social_intelligence": <0-10>, 
    "communication_effectiveness": <0-10>,
    "goal_achievement": <0-10>,
    "explanation": "<brief explanation of ratings>"
}}
"""
    
    def _get_rating_system_prompt(self) -> str:
        """System prompt for conversation rating"""
        return """You are an expert evaluator of social conversation quality, specializing in barrier-aware communication.

Your task is to rate conversations where one agent (Agent A) has communication barriers (cognitive biases, cultural differences, emotional states) and the other agent (Agent B) must adapt their communication strategy.

Focus on:
1. How well Agent B recognizes and adapts to Agent A's communication style
2. Maintenance of social relationship despite communication challenges
3. Effectiveness of information exchange under constraints
4. Overall conversation quality and naturalness
5. Achievement of conversation goals despite barriers

Rate objectively and consistently. High-quality conversations show adaptive communication, emotional intelligence, and successful goal pursuit despite barriers."""
    
    def filter_positive_conversations(
        self, 
        conversations: List[TrainingConversation],
        ratings: List[ConversationRating]
    ) -> List[TrainingConversation]:
        """Filter conversations to keep only high-quality ones for training"""
        
        # Create rating lookup
        rating_map = {r.conversation_id: r for r in ratings}
        
        # Filter positive conversations
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
        """Analyze rating patterns"""
        if not ratings:
            return {}
        
        # Calculate averages
        avg_overall = sum(r.overall_quality for r in ratings) / len(ratings)
        avg_barrier = sum(r.barrier_handling for r in ratings) / len(ratings)
        avg_social = sum(r.social_intelligence for r in ratings) / len(ratings)
        avg_communication = sum(r.communication_effectiveness for r in ratings) / len(ratings)
        avg_goal = sum(r.goal_achievement for r in ratings) / len(ratings)
        
        # Count positive conversations
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