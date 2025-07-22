import json
from social_decipher.environment.env_profile import EnvironmentProfile

def load_environments(file_path):
    """Load pre-processed environments from JSON file"""
    try:
        with open(file_path, 'r') as f:
            env_data = json.load(f)
            
        environments = []
        for env_item in env_data:
            agent_relationship = env_item.get("agent_relationship", None)
            
            profile = EnvironmentProfile(
                scenario=env_item["scenario"],
                agent_goals=env_item["agent_goals"],
                agent_reasons=env_item["agent_reasons"],
                agent_goals_mcqas=env_item["agent_goals_mcqas"],
                agent_reasons_mcqas=env_item["agent_reasons_mcqas"],
                agent_relationship=agent_relationship 
            )
                
            environments.append(profile)
            
        print(f"Successfully loaded {len(environments)} scenarios from {file_path}")
        return environments
    
    except Exception as e:
        print(f"Error loading environments from {file_path}: {e}")
        return []
