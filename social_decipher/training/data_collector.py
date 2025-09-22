"""
Data Collection for Interactive Training
Implements BC (Behavior Cloning) and SR (Self-Reinforcement) data collection
following Sotopia-π methodology but adapted for barrier-based social scenarios.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from datetime import datetime

from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.agent.social_agent import SocialAgent
from social_decipher.communication import run_single_scenario_simulation
from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.evaluate import ConversationEvaluator


@dataclass
class TrainingConversation:
    """Training conversation data structure"""
    conversation_id: str
    episode_type: str  # "original", "semantic", "cultural", "emotional"
    agent_a_model: str
    agent_b_model: str
    conversation_log: List[str]
    mcq_logs: List[Dict[str, Any]]
    eval_result: Dict[str, Any]
    barrier_info: Optional[Dict[str, Any]]
    timestamp: str
    trajectory_type: str  # "BC" or "SR"


class BarrierDataCollector:
    """
    Collects training conversations for barrier-aware social intelligence.
    
    Inspired by Sotopia-π data collection but specialized for:
    1. Barrier Communication: How agents handle cognitive biases
    2. Adaptive Strategies: Learning to communicate despite barriers
    3. Social Intelligence: Improving conversation quality under constraints
    """
    
    def __init__(
        self,
        expert_model: str = "gpt-4o",
        agent_model: str = "gpt-4o-mini", 
        partner_model: str = "gpt-4o-mini",
        evaluator_model: str = "gpt-4o",
        output_dir: str = "training_data"
    ):
        self.expert_model = expert_model
        self.agent_model = agent_model
        self.partner_model = partner_model
        self.evaluator = ConversationEvaluator(evaluator_model)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.bc_conversations: List[TrainingConversation] = []
        self.sr_conversations: List[TrainingConversation] = []
        
    def collect_behavior_cloning_data(
        self,
        episodes: List[Dict[str, Any]],
        num_conversations_per_episode: int = 3,
        max_rounds: int = 20
    ) -> List[TrainingConversation]:
        """
        Collect expert demonstrations (BC data).
        Both agents use expert models to show optimal barrier handling.
        """
        print("🎓 Collecting Behavior Cloning (BC) data...")
        
        # Load existing conversations to append new data, preventing overwrites
        bc_filepath = os.path.join(self.output_dir, "bc_data.json")
        if os.path.exists(bc_filepath):
            print(f"🔄 Found existing BC data at {bc_filepath}. Loading to append new conversations.")
            with open(bc_filepath, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    conversations = [TrainingConversation(**conv) for conv in existing_data]
                    print(f"   Loaded {len(conversations)} existing conversations.")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"⚠️  Could not load existing BC data, starting fresh. Error: {e}")
                    conversations = []
        else:
            conversations = []

        for episode_idx, episode_data in enumerate(episodes):
            episode_type = self._get_episode_type(episode_data)
            print(f"Episode {episode_idx + 1}/{len(episodes)} ({episode_type})")
            
            for conv_idx in range(num_conversations_per_episode):
                try:
                    conversation = self._run_expert_conversation(
                        episode_data, episode_idx, conv_idx, max_rounds, episode_type
                    )
                    if conversation:
                        conversations.append(conversation)
                        # Save after each new conversation to ensure progress is not lost
                        self._save_conversations(conversations, "bc_data.json")
                        
                except Exception as e:
                    print(f"ERROR: Failed BC conversation {conv_idx}: {e}")
                    continue
                    
        self.bc_conversations = conversations # Update the instance variable at the end
        return conversations
    
    def collect_self_reinforcement_data(
        self,
        episodes: List[Dict[str, Any]],
        num_conversations_per_episode: int = 5,
        max_rounds: int = 20
    ) -> List[TrainingConversation]:
        """
        Collect self-play data (SR data).
        Current agent model plays against itself to generate improvement targets.
        """
        print("Collecting Self-Reinforcement (SR) data...")

        # Load existing conversations to append new data
        sr_filepath = os.path.join(self.output_dir, "sr_data.json")
        if os.path.exists(sr_filepath):
            print(f"🔄 Found existing SR data at {sr_filepath}. Loading to append new conversations.")
            with open(sr_filepath, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    conversations = [TrainingConversation(**conv) for conv in existing_data]
                    print(f"   Loaded {len(conversations)} existing conversations.")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"⚠️  Could not load existing SR data, starting fresh. Error: {e}")
                    conversations = []
        else:
            conversations = []

        for episode_idx, episode_data in enumerate(episodes):
            episode_type = self._get_episode_type(episode_data)
            print(f"Episode {episode_idx + 1}/{len(episodes)} ({episode_type})")
            
            for conv_idx in range(num_conversations_per_episode):
                try:
                    conversation = self._run_self_play_conversation(
                        episode_data, episode_idx, conv_idx, max_rounds, episode_type
                    )
                    if conversation:
                        conversations.append(conversation)
                        self.sr_conversations.append(conversation)
                        
                except Exception as e:
                    print(f"ERROR: Failed SR conversation {conv_idx}: {e}")
                    continue
                    
        self._save_conversations(conversations, "sr_data.json")
        return conversations
    
    def _run_expert_conversation(
        self,
        episode_data: Dict[str, Any],
        episode_idx: int,
        conv_idx: int,
        max_rounds: int,
        episode_type: str
    ) -> Optional[TrainingConversation]:
        """Run conversation with expert models"""
        
        # Agent A (barrier agent) is the non-expert, Agent B (partner) is the expert
        profile_a = self._build_profile(episode_data, 0, self.partner_model) # Non-expert
        profile_b = self._build_profile(episode_data, 1, self.expert_model)  # Expert
        
        # Create environment
        env = self._create_environment(episode_data)
        
        # Create expert agents
        agent_a = SocialAgent(
            profile_a.first_name, profile_a, profile_b, env, 
            role_num=0, 
        )
        agent_b = SocialAgent(
            profile_b.first_name, profile_b, profile_a, env, 
            role_num=1, 
        )
        
        # Run conversation
        conversation_log, eval_result, mcq_logs = run_single_scenario_simulation(
            personA=agent_a,
            personB=agent_b,
            environment=env,
            evaluator=self.evaluator,
            num_turns=max_rounds,
            run_mcq_tests=False,
        )
        
        # Package training conversation
        conversation = TrainingConversation(
            conversation_id=f"bc_{episode_idx}_{conv_idx}_{int(time.time())}",
            episode_type=episode_type,
            agent_a_model=self.partner_model, # Correctly log the non-expert model
            agent_b_model=self.expert_model,   # Correctly log the expert model
            conversation_log=conversation_log,
            mcq_logs=mcq_logs,
            eval_result=eval_result,
            barrier_info=self._extract_barrier_info(episode_data),
            timestamp=datetime.now().isoformat(),
            trajectory_type="BC"
        )
        
        return conversation
    
    def _run_self_play_conversation(
        self,
        episode_data: Dict[str, Any],
        episode_idx: int,
        conv_idx: int,
        max_rounds: int,
        episode_type: str
    ) -> Optional[TrainingConversation]:
        """Run conversation with current agent model against a fixed partner."""
        
        # Agent A (the partner) uses the partner_model
        # Agent B (the one being trained) uses the agent_model
        print(f"Running SR conversation with Agent A: {self.partner_model}, Agent B: {self.agent_model}")

        profile_a = self._build_profile(episode_data, 0, self.partner_model)
        profile_b = self._build_profile(episode_data, 1, self.agent_model)
        
        # Create environment
        env = self._create_environment(episode_data)
        
        # Create agent instances
        agent_a = SocialAgent(
            profile_a.first_name, profile_a, profile_b, env, 
            role_num=0, 
        )
        agent_b = SocialAgent(
            profile_b.first_name, profile_b, profile_a, env, 
            role_num=1, 
        )
        
        # Run conversation
        conversation_log, eval_result, mcq_logs = run_single_scenario_simulation(
            personA=agent_a,
            personB=agent_b,
            environment=env,
            evaluator=self.evaluator,
            num_turns=max_rounds,
            run_mcq_tests=False,
        )
        
        # Package training conversation with correct model names
        conversation = TrainingConversation(
            conversation_id=f"sr_{episode_idx}_{conv_idx}_{int(time.time())}",
            episode_type=episode_type,
            agent_a_model=self.partner_model,
            agent_b_model=self.agent_model,
            conversation_log=conversation_log,
            mcq_logs=mcq_logs,
            eval_result=eval_result,
            barrier_info=self._extract_barrier_info(episode_data),
            timestamp=datetime.now().isoformat(),
            trajectory_type="SR"
        )
        
        return conversation
    
    def _get_episode_type(self, episode_data: Dict[str, Any]) -> str:
        """Determine episode type from barrier information"""
        barrier_type = episode_data.get("barrier_type", "")
        if barrier_type == "semantic_structure":
            return "semantic"
        elif barrier_type == "cultural_style":
            return "cultural"
        elif barrier_type == "emotional_influence":
            return "emotional"
        else:
            return "original"
    
    def _build_profile(self, episode_data: Dict[str, Any], agent_idx: int, model_id: str) -> AgentProfile:
        """Build agent profile from episode data"""
        agent_profile_data = episode_data["agent_profiles"][agent_idx]
        return AgentProfile.from_dict(agent_profile_data, model_id)
    
    def _create_environment(self, episode_data: Dict[str, Any]) -> EnvironmentProfile:
        """Create environment from episode data"""
        env = EnvironmentProfile(
            scenario=episode_data["scenario"],
            agent_goals=episode_data["agent_goals"],
            agent_reasons=[
                episode_data.get("agent1_reason", ""), 
                episode_data.get("agent2_reason", "")
            ],
            agent_goals_mcqas=episode_data.get("agent_goals_mcqas", []),
            agent_reasons_mcqas=episode_data.get("agent_reasons_mcqas", []),
            agent_knowledge_mcqas=episode_data.get("agent_knowledge_mcqas", []),
            agent_relationship=episode_data.get("agent_relationship", "friend")
        )
        
        # Add barrier information if present
        if "barrier_prompts" in episode_data:
            env.env["barrier_prompts"] = episode_data["barrier_prompts"]
        if "barrier_cues" in episode_data:
            env.env["barrier_cues"] = episode_data["barrier_cues"]
        if "barrier_type" in episode_data:
            env.env["barrier_type"] = episode_data["barrier_type"]
            
        return env
    
    def _extract_barrier_info(self, episode_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract barrier information from episode"""
        barrier_info = {}
        
        if "barrier_type" in episode_data:
            barrier_info["barrier_type"] = episode_data["barrier_type"]
        if "barrier_prompts" in episode_data:
            barrier_info["barrier_prompts"] = episode_data["barrier_prompts"]
        if "barrier_cues" in episode_data:
            barrier_info["barrier_cues"] = episode_data["barrier_cues"]
            
        return barrier_info if barrier_info else None
    
    def _save_conversations(self, conversations: List[TrainingConversation], filename: str):
        """Save conversations to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to serializable format
        serializable_data = [asdict(conv) for conv in conversations]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(conversations)} conversations to {filepath}")
    
    def get_all_conversations(self) -> Tuple[List[TrainingConversation], List[TrainingConversation]]:
        """Get all collected conversations"""
        return self.bc_conversations, self.sr_conversations
    
    def load_conversations(self, bc_file: str = None, sr_file: str = None):
        """Load previously collected conversations"""
        if bc_file:
            bc_path = os.path.join(self.output_dir, bc_file)
            if os.path.exists(bc_path):
                with open(bc_path, 'r', encoding='utf-8') as f:
                    bc_data = json.load(f)
                    self.bc_conversations = [TrainingConversation(**conv) for conv in bc_data]
                    print(f"Loaded {len(self.bc_conversations)} BC conversations")
        
        if sr_file:
            sr_path = os.path.join(self.output_dir, sr_file)
            if os.path.exists(sr_path):
                with open(sr_path, 'r', encoding='utf-8') as f:
                    sr_data = json.load(f)
                    self.sr_conversations = [TrainingConversation(**conv) for conv in sr_data]
                    print(f"Loaded {len(self.sr_conversations)} SR conversations")


def load_barrier_episode_sets(data_dir: str = "data") -> Dict[str, List[Dict[str, Any]]]:
    """Load pre-generated barrier episode sets"""
    
    episode_sets = {}
    
    barrier_files = {
        "semantic": "episodes_all_semantic.json",
        "cultural": "episodes_all_cultural.json", 
        "emotional": "episodes_all_emotional.json"
    }
    
    for barrier_type, filename in barrier_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                episode_sets[barrier_type] = json.load(f)
            print(f"Loaded {len(episode_sets[barrier_type])} {barrier_type} episodes")
        else:
            print(f"WARNING: {filepath} not found - run barrier creation first")
            episode_sets[barrier_type] = []
    
    return episode_sets