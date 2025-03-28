
from typing import List


class Agent_Profile():
    def __init__(self, 
                 first_name: str,
                 last_name: str,
                 age: int,
                 gender: str,
                 gender_pronoun: str,
                 occupation: str,
                 public_info: str,
                 personality_and_values: str,
                 model_id: str
                ):
        
        self.profile = {
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'gender': gender,
            'gender_pronoun': gender_pronoun,
            'occupation': occupation,
            'public_info': public_info,
            'personality_and_values': personality_and_values,
            'model_id': model_id
        }
        
class Environment_Profile():
    def __init__(self,
                 scenario: str,
                 agent_goals: List[str],
            
                ):
        
        self.env = {
            'scenario': scenario,
            'agent_goals': agent_goals
        }