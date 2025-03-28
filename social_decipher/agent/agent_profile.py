
from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile, RelationshipProfile, EnvironmentList

class Agent_Profile(AgentProfile):
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
        
        self.profile = AgentProfile(
            first_name,
            last_name,
            age,
            gender,
            gender_pronoun,
            occupation,
            public_info,
            personality_and_values,
            model_id
        )
        