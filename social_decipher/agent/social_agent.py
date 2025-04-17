import yaml
from agency_swarm import Agency, Agent
from rich import print
from typing import Dict, Any

from ..encryption import BaseEncryption
from ..environment.env_profile import EnvironmentProfile
from .agent_profile import AgentProfile


class SocialAgent(Agent):
    def __init__(
        self, name, profile: AgentProfile, env: EnvironmentProfile, role_num: int
    ):
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
            temperature=0.2,
            max_prompt_tokens=50000,
        )

    def set_instruction(
        self, env: EnvironmentProfile, profile: AgentProfile, agent_role: int
    ) -> str:
        with open("../configs/social_task.yaml") as template_file:
            template_sections = yaml.safe_load(template_file)

        profile_dict = profile.profile
        env_dict = env.env

        if agent_role == 0:
            agent_goal = env_dict["agent_goals"][0]
            partner_goal = env_dict["agent_goals"][1]
            agent_reason = env_dict["agent_reasons"][0]
        else:
            agent_goal = env_dict["agent_goals"][1]
            partner_goal = env_dict["agent_goals"][0]
            agent_reason = env_dict["agent_reasons"][1]

        merged = {
            **profile_dict,
            "scenario": env_dict["scenario"],
            "agent_goal": agent_goal,
            "partner_goal": partner_goal,
            "agent_reason": agent_reason,
        }

        instruction = (
            template_sections["profile_description"]
            + "\n\n"
            + template_sections["social_task_instructions"]
        )

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
        assert (
            self.agency is not None
        ), "Agent must be assigned to an agency before acting."

        if initial:
            # Use the system instruction to generate an opening message
            response = self.agency.get_completion(
                "Now, generate your initial message to start the conversation, try to be concise",
                recipient_agent=self,
            )
            print(f"**{self.name} INITIAL RESPONSE: {response}")

            encrypted_response = (
                self.encryption(response) if self.encryption else response
            )

            if self.encryption is not None:
                print(f"**{self.name} ENCRYPTED RESPONSE: {encrypted_response}")
            self.log.append(
                {
                    "initial": True,
                    "response_raw": response,
                    "response_encrypted": encrypted_response,
                }
            )
            return encrypted_response

        received = message
        print(f"[bold cyan]**{self.name} RECEIVED MESSAGE: {received}")
        if self.encryption is not None:
            message = self.encryption.decrypt(message)
            print(f"**{self.name} DECRYPTED MESSAGE: {message}\n")

        response = self.agency.get_completion(message, recipient_agent=self)
        print(f"[yellow]**{self.name} ORIGINAL RESPONSE: {response}")

        encrypted_response = self.encryption(response) if self.encryption else response
        if self.encryption is not None:
            print(f"**{self.name} ENCRYPTED RESPONSE: {encrypted_response}")

        self.log.append(
            {
                "received_raw": received,
                "received_decrypted": message if self.encryption else received,
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
        
        if len(transcript) > 6:
            short_transcript = transcript[-6:]
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
