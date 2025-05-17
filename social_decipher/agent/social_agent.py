import json
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from rich import print

from social_decipher.encryption import BaseEncryption
from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.utils.utils import chinese_to_pinyin, direct_completion

from .agent_memory import AgentMemory
from .agent_profile import AgentProfile


class SocialAgent:
    def __init__(
        self,
        name: str,
        profile: AgentProfile,
        partner_profile: AgentProfile,
        env: EnvironmentProfile,
        role_num: int,
        use_action: bool = False,
        memory: Optional[AgentMemory] = None,
    ):
        self.name = name
        self.env = env
        self.encryption = None
        self.log = []
        self.role_num = role_num
        self.profile = profile
        self.partner_profile = partner_profile

        if memory is None:
            self.memory = AgentMemory(name, partner_profile.profile["first_name"])
        else:
            self.memory = memory

        self.instructions = self.set_static_instruction(use_action)

    def set_static_instruction(self, use_action=False) -> str:
        return self.build_instruction(transcript="", turn_number=0, use_action=use_action)
    
    def build_instruction(
        self, transcript: str, turn_number: int, use_action: bool = False
    ) -> str:

        with open("../configs/social_task.yaml") as f:
            templates = yaml.safe_load(f)
        
        profile = self.profile.profile
        partner = self.partner_profile.profile
        
        if self.env is None:
            return "Instructions not available yet."
            
        env_dict = self.env.env
        
        if self.role_num == 0:
            agent_goal = env_dict["agent_goals"][0]
            agent_reason = env_dict["agent_reasons"][0]
        else:
            agent_goal = env_dict["agent_goals"][1]
            agent_reason = env_dict["agent_reasons"][1]
        
        template_key = (
            "social_task_instructions_action"
            if use_action
            else "social_task_instructions"
        )
        
        template = templates[template_key]
        
        memory_context = self.memory.get_memory_context(detailed=(turn_number == 0))
        
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
            "memory_context": memory_context,
        }
        
        for key, value in mapping.items():
            template = template.replace(f"{{{{ {key} }}}}", str(value))
        
        return template
    
    def update_instruction(
        self, transcript: List[str], turn_number: int, use_action: bool = False
    ):
        short_transcript = transcript[-6:] if len(transcript) > 6 else transcript
        transcript_text = "\n".join(short_transcript)
        
        self.instructions = self.build_instruction(
            transcript=transcript_text, turn_number=turn_number, use_action=use_action
        )
    
    def set_encryption(self, encryption: Optional[BaseEncryption]):
        self.encryption = encryption
    
    def act(
        self, message=None, initial: bool = False, use_action: bool = False
    ) -> Union[str, Dict[str, Any]]:
        if initial:
            prompt = "Now, generate your initial message to start the conversation, try to be concise"
            response = direct_completion(self, message=prompt, use_action=use_action)
            
            print(f"[green]**{self.name} INITIAL RESPONSE: {response}")
            
            if use_action:
                try:
                    if isinstance(response, str):
                        response_json = json.loads(response)
                    else:
                        response_json = response
                        
                    original_response = response_json
                    
                    if response_json["action_type"] == "speak":
                        # Always check if encryption should be applied
                        if self.encryption is not None:
                            response_json["argument"] = self.encryption(response_json["argument"])
                
                        response_json["argument"] = chinese_to_pinyin(response_json["argument"])
                    
                    encrypted_response = response_json
                except (json.JSONDecodeError, KeyError) as e:
                    # Handle case where response is not valid JSON
                    print(f"[red]Error processing action response: {e}")
                    original_response = response
                    # Always check if encryption should be applied
                    encrypted_response = self.encryption(response) if self.encryption is not None else response
            else:
                # Handle text-based communication
                original_response = response
                # Always check if encryption should be applied
                encrypted_response = self.encryption(response) if self.encryption is not None else response
                encrypted_response = chinese_to_pinyin(encrypted_response)
            
            if self.encryption is not None:
                print(f"[yellow]**{self.name} ENCRYPTED RESPONSE: {encrypted_response}")
            
            # Log the response
            self.log.append(
                {
                    "initial": True,
                    "response_raw": original_response,
                    "response_encrypted": encrypted_response,
                }
            )
            return encrypted_response
        
        # Handle non-initial messages
        received = message
        print(f"[blue]**{self.name} RECEIVED: {received}")
        
        if use_action:
            # Extract argument from message for action-based communication
            if isinstance(message, dict) and "action_type" in message:
                action_type = message.get("action_type", "")
                argument = message.get("argument", "")
                
                partner_name = self.partner_profile.profile["first_name"]
                
                if action_type == "speak":
                    response = direct_completion(self, message=argument)
                elif action_type in ["non-verbal communication", "action"]:
                    response = direct_completion(self, message=f"{partner_name} {argument}")
                else:
                    response = direct_completion(self, message=str(message))
            else:
                response = direct_completion(self, message=str(message))
            
            try:
                if isinstance(response, str):
                    response_json = json.loads(response)
                else:
                    response_json = response
                    
                original_response = response_json
                
                if response_json["action_type"] == "speak":
                    # Always check if encryption should be applied
                    if self.encryption is not None:
                        response_json["argument"] = self.encryption(response_json["argument"])
                    response_json["argument"] = chinese_to_pinyin(response_json["argument"])
                
                encrypted_response = response_json
            except (json.JSONDecodeError, KeyError) as e:
                # Handle case where response is not valid JSON
                print(f"[red]Error processing action response: {e}")
                original_response = response
                # Always check if encryption should be applied
                encrypted_response = self.encryption(response) if self.encryption is not None else response
        else:
            # Handle text-based communication
            response = direct_completion(self, message=message)
            original_response = response
            # Always check if encryption should be applied
            encrypted_response = self.encryption(response) if self.encryption is not None else response
            encrypted_response = chinese_to_pinyin(encrypted_response)
        
        print(f"[green]**{self.name} RESPONSE: {encrypted_response}")
        
        if self.encryption is not None:
            print(f"[yellow]**{self.name} ENCRYPTED RESPONSE: {encrypted_response}")
        # Log the response
        self.log.append(
            {
                "received_raw": received,
                "response_raw": original_response,
                "response_encrypted": encrypted_response,
            }
        )
        
        return encrypted_response

    def predict_mcq_answer(
        self,
        agent_name: str,
        partner_name: str,
        transcript: List[str],
        mcqa: Dict[str, Any],
        test_prompt: Dict[str, str],
        task_type: str,
    ) -> Dict[str, Any]:
   
        assert task_type in {"goal", "reason"}, "task_type must be 'goal' or 'reason'"
        
        if len(transcript) > 6:
            short_transcript = transcript[-6:]
        else:
            short_transcript = transcript
        
        formatted_options = "\n".join([f"{k}: {v}" for k, v in mcqa["options"].items()])
        conversation_str = "\n".join(short_transcript)
        
        # Customize question based on agent roles
        question = mcqa["question"]
        if self.role_num == 0:
            question = question.replace("Agent 1", agent_name).replace("Agent 2", partner_name)
        else:
            question = question.replace("Agent 1", partner_name).replace("Agent 2", agent_name)
        
        # Create prompt for MCQ prediction
        prompt = test_prompt[
            "MCQ_Goal_Prediction_Prompt"
            if task_type == "goal"
            else "MCQ_Reason_Prediction_Prompt"
        ].format(
            agent_name=agent_name,
            partner_name=partner_name,
            question=question,
            options=formatted_options,
            transcript=conversation_str,
        )
        
        # Generate response using direct completion
        response = direct_completion(self, prompt, use_action=False).strip()
        
        # Parse the response to extract selected option and confidence
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
        
        # Return prediction results
        return {
            "selected": selected if selected in mcqa["options"] else "Invalid",
            "confidence": max(0.0, min(confidence, 1.0)),  # clamp between 0-1
            "correct": selected == mcqa["correct_answer"],
        }
    
    def update_memory_after_scenario(
        self,
        scenario_log: List[str],
        scenario_results: Dict[str, Any],
        encryption_enabled: bool = False,
    ):
        """
        Update agent memory after completing a scenario.
        
        Args:
            scenario_log: Conversation log
            scenario_results: Scenario evaluation results
            encryption_enabled: Whether encryption was enabled
        """
        # Get the agent's goal from the environment
        if self.role_num == 0:
            agent_goal = self.env.env["agent_goals"][0]
            goal_achieved = scenario_results.get("agent0_goal_achieved", False)
        else:
            agent_goal = self.env.env["agent_goals"][1]
            goal_achieved = scenario_results.get("agent1_goal_achieved", False)
        
        # Update memory
        self.memory.update_after_scenario(
            scenario_log=scenario_log,
            scenario_results=scenario_results,
            agent_goal=agent_goal,
            goal_achieved=goal_achieved,
            encryption_enabled=encryption_enabled,
        )
        
        print('Updated memory:')
        print(f"- Key memories: {self.memory.key_memories}")
        print(f"- Language barrier: {self.memory.language_barrier}")
    
    def save_memory(self, output_dir: str):
        """
        Save agent memory to file.
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        self.memory.save(os.path.join(output_dir, f"{self.name}_memory.json"))
    
    def load_memory(self, filepath: str) -> bool:
        """
        Load agent memory from file.
        
        Args:
            filepath: Path to memory file
            
        Returns:
            True if memory loaded successfully, False otherwise
        """
        if os.path.exists(filepath):
            self.memory = AgentMemory.load(filepath)
            return True
        return False