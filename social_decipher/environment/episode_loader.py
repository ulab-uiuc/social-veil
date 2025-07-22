# social_decipher/environment/episode_loader.py

import json
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .env_profile import EnvironmentProfile


@dataclass
class SotopiaAgentProfile:
    """Agent profile from Sotopia episode format"""
    pk: str
    first_name: str
    last_name: str
    age: int
    occupation: str
    gender: str
    gender_pronoun: str
    public_info: str
    big_five: str
    moral_values: List[str]
    schwartz_personal_values: List[str]
    personality_and_values: str
    decision_making_style: str
    secret: str
    mbti: str = ""
    tag: str = ""


@dataclass
class MCQQuestion:
    """Multiple choice question format"""
    question: str
    options: Dict[str, str]
    correct_answer: str


@dataclass
class SotopiaEpisode:
    """Complete Sotopia episode with all data"""
    episode_id: str
    environment_id: str
    scenario: str
    codename: str
    agent_profiles: List[SotopiaAgentProfile]
    agent_goals: List[str]
    agent_relationship: str
    agent1_reason: str
    agent2_reason: str
    agent1_private_knowledge: str
    agent2_private_knowledge: str
    agent_goals_mcqas: List[MCQQuestion]
    agent_reasons_mcqas: List[MCQQuestion]
    agent_knowledge_mcqas: Optional[List[MCQQuestion]] = None


class EpisodeLoader:
    """Loads and manages Sotopia episodes"""
    
    def __init__(self, episodes_file: str):
        self.episodes_file = Path(episodes_file)
        self.episodes: List[SotopiaEpisode] = []
        self.load_episodes()
    
    def load_episodes(self):
        """Load all episodes from JSON or JSONL file"""
        if not self.episodes_file.exists():
            print(f"Warning: Episodes file {self.episodes_file} not found")
            return
        
        # Handle both JSON and JSONL formats
        if self.episodes_file.suffix == '.json':
            with open(self.episodes_file, 'r') as f:
                episodes_data = json.load(f)
                if isinstance(episodes_data, list):
                    for episode_data in episodes_data:
                        try:
                            episode = self._parse_episode(episode_data)
                            self.episodes.append(episode)
                        except Exception as e:
                            print(f"Error parsing episode: {e}")
                            continue
        else:  # JSONL format
            with open(self.episodes_file, 'r') as f:
                for line in f:
                    try:
                        episode_data = json.loads(line.strip())
                        episode = self._parse_episode(episode_data)
                        self.episodes.append(episode)
                    except Exception as e:
                        print(f"Error parsing episode: {e}")
                        continue
        
        print(f"Loaded {len(self.episodes)} episodes")
    
    def _parse_episode(self, data: Dict[str, Any]) -> SotopiaEpisode:
        """Parse episode data into SotopiaEpisode object"""
        
        # Parse agent profiles
        agent_profiles = []
        for profile_data in data["agent_profiles"]:
            profile = SotopiaAgentProfile(
                pk=profile_data["pk"],
                first_name=profile_data["first_name"],
                last_name=profile_data["last_name"],
                age=profile_data["age"],
                occupation=profile_data["occupation"],
                gender=profile_data["gender"],
                gender_pronoun=profile_data["gender_pronoun"],
                public_info=profile_data["public_info"],
                big_five=profile_data["big_five"],
                moral_values=profile_data.get("moral_values", []),
                schwartz_personal_values=profile_data.get("schwartz_personal_values", []),
                personality_and_values=profile_data["personality_and_values"],
                decision_making_style=profile_data["decision_making_style"],
                secret=profile_data["secret"],
                mbti=profile_data.get("mbti", ""),
                tag=profile_data.get("tag", "")
            )
            agent_profiles.append(profile)
        
        # Parse MCQ questions
        def parse_mcqs(mcq_list):
            return [MCQQuestion(
                question=mcq["question"],
                options=mcq["options"],
                correct_answer=mcq["correct_answer"]
            ) for mcq in mcq_list]
        
        return SotopiaEpisode(
            episode_id=data["episode_id"],
            environment_id=data["environment_id"],
            scenario=data["scenario"],
            codename=data["codename"],
            agent_profiles=agent_profiles,
            agent_goals=data["agent_goals"],
            agent_relationship=data["agent_relationship"],
            agent1_reason=data["agent1_reason"],
            agent2_reason=data["agent2_reason"],
            agent1_private_knowledge=data.get("agent1_private_knowledge", ""),
            agent2_private_knowledge=data.get("agent2_private_knowledge", ""),
            agent_goals_mcqas=parse_mcqs(data["agent_goals_mcqas"]),
            agent_reasons_mcqas=parse_mcqs(data["agent_reasons_mcqas"]),
            agent_knowledge_mcqas=parse_mcqs(data["agent_knowledge_mcqas"]) if "agent_knowledge_mcqas" in data else None
        )
    
    def get_episode(self, episode_id: Optional[str] = None, index: Optional[int] = None) -> Optional[SotopiaEpisode]:
        """Get episode by ID or index"""
        if episode_id:
            for episode in self.episodes:
                if episode.episode_id == episode_id:
                    return episode
            return None
        elif index is not None:
            if 0 <= index < len(self.episodes):
                return self.episodes[index]
            return None
        else:
            # Return first episode if no specification
            return self.episodes[0] if self.episodes else None
    
    def get_random_episode(self) -> Optional[SotopiaEpisode]:
        """Get a random episode"""
        if not self.episodes:
            return None
        return random.choice(self.episodes)
    
    def get_multiple_episodes(self, count: int, start_index: int = 0) -> List[SotopiaEpisode]:
        """Get multiple episodes starting from an index"""
        end_index = min(start_index + count, len(self.episodes))
        return self.episodes[start_index:end_index]
    
    def to_environment_profiles(self, episodes: List[SotopiaEpisode]) -> List[EnvironmentProfile]:
        """Convert Sotopia episodes to EnvironmentProfile format"""
        env_profiles = []
        
        for episode in episodes:
            # Convert MCQs to dictionary format
            goals_mcqas = [{
                "question": mcq.question,
                "options": mcq.options,
                "correct_answer": mcq.correct_answer
            } for mcq in episode.agent_goals_mcqas]
            
            reasons_mcqas = [{
                "question": mcq.question,
                "options": mcq.options,
                "correct_answer": mcq.correct_answer
            } for mcq in episode.agent_reasons_mcqas]
            
            knowledge_mcqas = None
            if episode.agent_knowledge_mcqas:
                knowledge_mcqas = [{
                    "question": mcq.question,
                    "options": mcq.options,
                    "correct_answer": mcq.correct_answer
                } for mcq in episode.agent_knowledge_mcqas]
            
            env_profile = EnvironmentProfile(
                scenario=episode.scenario,
                agent_goals=episode.agent_goals,
                agent_reasons=[episode.agent1_reason, episode.agent2_reason],
                agent_relationship=episode.agent_relationship,
                agent_goals_mcqas=goals_mcqas,
                agent_reasons_mcqas=reasons_mcqas
            )
            
            # Add additional episode information
            env_profile.env.update({
                "episode_id": episode.episode_id,
                "codename": episode.codename,
                "agent_profiles": episode.agent_profiles,
                "agent1_private_knowledge": episode.agent1_private_knowledge,
                "agent2_private_knowledge": episode.agent2_private_knowledge,
                "agent_knowledge_mcqas": knowledge_mcqas
            })
            
            env_profiles.append(env_profile)
        
        return env_profiles