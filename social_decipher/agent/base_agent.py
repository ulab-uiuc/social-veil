import random
import uuid
from openai import swarm
from encryption.encryption import Encryptor
from social_decipher.agent.profile import Agent_Profile

class BaseAgent(swarm.Agent):
    def __init__(self, 
                 first_name: str, 
                 last_name: str,
                 age: int,
                 gender: str,
                 gender_pronoun: str,
                 occupation: str,
                 public_info: str,
                 personality_and_values: str,
                 agent_profile: Agent_Profile, 
                 encryptor: Encryptor):
        self.encryption = encryptor  
        self.profile = agent_profile(
                        first_name,
                        last_name,
                        age,
                        gender,
                        gender_pronoun,
                        occupation,
                        public_info,
                        personality_and_values,
                        uuid.uuid4()
                    )
        self.memory = {} 

    def goal(self) -> str:
        pass

    def response(self, message: str) -> str:
        return self.encryption.run(message)

    def act(self) -> dict:
        pass
        

