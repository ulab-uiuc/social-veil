from agency_swarm import Agent, Agency
from .profile import Agent_Profile, Environment_Profile
import yaml
from encryption import BaseEncryption

class SocialAgent(Agent):
    def __init__(self, 
                 name, 
                 profile: Agent_Profile,
                 env: Environment_Profile,
                 role_num: int):
        
        description = ""
        instruction = self.set_instruction(env, profile, role_num)
        self.agency = None
        self.encryption = None
        self.log = []
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

    def set_agency(self, agency: Agency):
        self.agency = agency

    def set_encryption(self, encryption: BaseEncryption):
        self.encryption = encryption

    def inference(self, message):
        response = self.agency.get_completion(message, recipient_agent=self)
        return response

    def act(self, message=None, initial: bool = False):
        assert self.agency is not None, "Agent must be assigned to an agency before acting."

        if initial:
            # Use the system instruction to generate an opening message
            response = self.agency.get_completion("Now, generate your initial message to start the conversation, try to be concise", recipient_agent=self)
            print(f"**{self.name} INITIAL RESPONSE: {response}")

            encrypted_response = self.encryption(response) if self.encryption else response
            print(f"**{self.name} ENCRYPTED RESPONSE: {encrypted_response}")
            self.log.append({
                "initial": True,
                "response_raw": response,
                "response_encrypted": encrypted_response
            })
            return encrypted_response

        received = message
        print(f"**{self.name} RECEIVED MESSAGE: {received}")
        if self.encryption is not None:
            message = self.encryption.decrypt(message)
            print(f"**{self.name} DECRYPTED MESSAGE: {message}\n")

        response = self.agency.get_completion(message, recipient_agent=self)
        print(f"**{self.name} ORIGINAL RESPONSE: {response}")

        encrypted_response = self.encryption(response) if self.encryption else response
        print(f"**{self.name} ENCRYPTED RESPONSE: {encrypted_response}")

        self.log.append({
            "received_raw": received,
            "received_decrypted": message if self.encryption else received,
            "response_raw": response,
            "response_encrypted": encrypted_response
        })


        return encrypted_response
        
    def response_validator(self, message):
        return message
