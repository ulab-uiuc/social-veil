import json
import yaml
from agency_swarm import Agency, Agent
from rich import print
from typing import Dict, Any

from ..encryption import BaseEncryption
from ..environment.env_profile import EnvironmentProfile
from .agent_profile import AgentProfile


class SocialAgent(Agent):
    def __init__(
        self, 
        name, 
        profile: AgentProfile, 
        partner_profile: AgentProfile, 
        env: EnvironmentProfile, role_num: int, use_action: bool = False
    ):

        self.agency = None
        self.env = env
        self.encryption = None
        self.log = []
        self.role_num = role_num
        self.profile = profile
        self.partner_profile = partner_profile

        initial_instruction = self.set_static_instruction(use_action)

        super().__init__(
            name=name,
            description="",
            instructions=initial_instruction,
            files_folder="./files",
            schemas_folder="./schemas",
            tools=[],
            tools_folder="./tools",
            temperature=0.2,
            max_prompt_tokens=50000,
        )
        self.instructions = initial_instruction

    def set_static_instruction(self, use_action=False) -> str:
        return self.build_instruction(transcript="", turn_number=0, use_action=use_action)

    def build_instruction(self, transcript: str, turn_number: int, use_action: bool = False) -> str:
        with open("../configs/social_task.yaml") as f:
            templates = yaml.safe_load(f)

        profile = self.profile.profile
        partner = self.partner_profile.profile
        env_dict = self.env.env

        if self.role_num == 0:
            agent_goal = env_dict["agent_goals"][0]
            agent_reason = env_dict["agent_reasons"][0]
            partner_goal = env_dict["agent_goals"][1]
        else:
            agent_goal = env_dict["agent_goals"][1]
            agent_reason = env_dict["agent_reasons"][1]
            partner_goal = env_dict["agent_goals"][0]

        template_key = "social_task_instructions_action" if use_action else "social_task_instructions"
        template = templates[template_key]

        mapping = {
            "agent_name": profile["first_name"],
            "partner_name": partner["first_name"],
            "scenario": env_dict["scenario"],
            "agent_background": profile["public_info"],
            "partner_background": partner["public_info"],
            "agent_goal": agent_goal,
            "agent_reason": agent_reason,
            "transcript": transcript,
            "turn_number": turn_number,
            "agent_age": profile["age"],
            "agent_gender": profile["gender"],
            "agent_occupation": profile["occupation"],
            "agent_public_info": profile["public_info"],
            "partner_age": partner["age"],
            "partner_gender": partner["gender"],
            "partner_occupation": partner["occupation"],
            "partner_public_info": partner["public_info"],
        }

        for key, value in mapping.items():
            template = template.replace(f"{{{{ {key} }}}}", str(value))

        return template

    def update_instruction(self, transcript: list[str], turn_number: int, use_action: bool = False):
        short_transcript = transcript[-4:] if len(transcript) > 4 else transcript
        transcript_text = "\n".join(short_transcript)

        self.instructions = self.build_instruction(transcript=transcript_text,
                                                    turn_number=turn_number,
                                                    use_action=use_action
                                                    )
        
    def set_agency(self, agency: Agency):
        self.agency = agency

    def set_encryption(self, encryption: BaseEncryption):
        self.encryption = encryption

    def inference(self, message):
        response = self.agency.get_completion(message, recipient_agent=self)
        return response

    def act(self, message=None, initial: bool = False, use_action: bool = False) -> str:
        assert (
            self.agency is not None
        ), "Agent must be assigned to an agency before acting."

        if initial:
            response = self.agency.get_completion(
                "Now, generate your initial message to start the conversation, try to be concise",
                recipient_agent=self,
            )

            print(f"**{self.name} INITIAL RESPONSE: {response}")

            if use_action:
                response = json.loads(response)
                response["argument"] = (
                    self.encryption(response["argument"]) if self.encryption else response["argument"]
                )
                encrypted_response = response
            else:
                encrypted_response = (
                    self.encryption(response) if self.encryption else response
                )

            if self.encryption is not None:
                print(f"**{self.name} ENCRYPTED MESSAGE: {encrypted_response}")
   
            self.log.append(
                {
                    "initial": True,
                    "response_raw": response,
                    "response_encrypted": encrypted_response,
                }
            )
            return encrypted_response

        received = message
        
        if use_action:
            response = self.agency.get_completion(message['argument'], recipient_agent=self) 
        else:
            response = self.agency.get_completion(message, recipient_agent=self) 

        if use_action:
            response = json.loads(response)
            response["argument"] = (
                self.encryption(response["argument"]) if self.encryption else response["argument"]
            )
            encrypted_response = response
        else:   
            encrypted_response = self.encryption(response) if self.encryption else response

        if self.encryption is not None:
            print(f"[green]**{self.name} ENCRYPTED RESPONSE: {encrypted_response}")

        self.log.append(
            {
                "received_raw": received,
                # "received_decrypted": message if self.encryption else received,
                "response_raw": response,
                "response_encrypted": encrypted_response,
            }
        )
        return encrypted_response

    def predict_mcq_answer(self, 
                           transcript: list[str], 
                           mcqa: Dict[str, Any], 
                           test_prompt: Dict[str, str],
                           task_type: str) -> Dict[str, Any]:
        assert self.agency is not None, "Agent must be assigned to an agency before acting."
        assert task_type in {"goal", "reason"}, "task_type must be 'goal' or 'reason'"
        
        if len(transcript) > 4:
            short_transcript = transcript[-4:]
        else:
            short_transcript = transcript

        formatted_options = "\n".join([f"{k}: {v}" for k, v in mcqa["options"].items()])
        conversation_str = "\n".join(short_transcript)

        prompt = test_prompt[
            "MCQ_Goal_Prediction_Prompt" if task_type == "goal" else "MCQ_Reason_Prediction_Prompt"
        ].format(
            question=mcqa["question"],
            options=formatted_options,
            transcript=conversation_str
        )

        response = self.agency.get_completion(prompt, recipient_agent=self).strip()
        selected = None
        confidence = 0.0
        try:
            for line in response.split("\n"):
                if line.lower().startswith("selected:"):
                    selected = line.split(":")[1].strip().upper()
                elif line.lower().startswith("confidence:"):
                    confidence = float(line.split(":")[1].strip())
        except Exception as e:
            print(f"Error parsing MCQ response from agent {self.name}: {e}")

        return {
            "selected": selected if selected in mcqa["options"] else "Invalid",
            "confidence": max(0.0, min(confidence, 1.0)),  # clamp between 0-1
            "correct": selected == mcqa["correct_answer"]
        }




    def response_validator(self, message):
        return message
