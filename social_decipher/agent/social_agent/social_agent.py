from agency_swarm.agents import Agent
from ..profile import Agent_Profile, Environment_Profile
import yaml

class SocialAgent(Agent):
    def __init__(self, 
                 name, 
                 profile: Agent_Profile,
                 env: Environment_Profile,
                 role_num: int):
        
        description = ""
        instruction = self.set_instruction(env, profile, role_num)

        super().__init__(
            name=name,
            description=description,
            instructions=instruction,
            files_folder="./files",
            schemas_folder="./schemas",
            tools=[],
            tools_folder="./tools",
            temperature=0.3,
            max_prompt_tokens=25000,
        )

    
    def set_instruction(self, 
                        env: Environment_Profile, 
                        profile: Agent_Profile,
                        agent_role: int
                        ) -> str:

        with open("../configs/social_task.yaml", "r") as template_file:
            template_sections = yaml.safe_load(template_file)

        profile_dict = profile.profile
        env_dict = env.env

        if agent_role == 0:
            agent_role_1 = env_dict["agent_goals"][0]
            agent_role_2 = env_dict["agent_goals"][1]
        else:
            agent_role_1 = env_dict["agent_goals"][1]
            agent_role_2 = env_dict["agent_goals"][0]

        merged = {
            **profile_dict,
            "scenario": env_dict["scenario"],
            "agent_goal_1": agent_role_1,
            "agent_goal_2": agent_role_2,
            "agent_goal": env_dict["agent_goals"][agent_role],
        }

        instruction = template_sections["agent_profile"] + "\n\n" \
                    + template_sections["profile_description"] + "\n\n" \
                    + template_sections["social_task_instructions"]

        for key, value in merged.items():
            instruction = instruction.replace(f"{{{{ {key} }}}}", str(value))

        return instruction

        
    def response_validator(self, message):
        return message
