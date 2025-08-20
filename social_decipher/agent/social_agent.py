import json
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from rich import print

from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.utils.base import direct_completion

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
        mix: bool = False,
        template_path: str = None,
    ):
        self.name = name
        self.env = env
        self.log = []
        self.role_num = role_num
        self.profile = profile
        self.partner_profile = partner_profile
        # Optional preface injected before the core social instruction (e.g., barrier prompts)
        self.extra_instruction_preface: Optional[str] = None
        
        # Set default template path if not provided
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "social_task.yaml")
            
        self.template_path = template_path
        if memory is None:
            self.memory = AgentMemory(name, partner_profile.first_name)
        else:
            self.memory = memory
        self.instructions = self.set_static_instruction(use_action, mix)

    

    def set_static_instruction(self, use_action=False, mix=False) -> str:
        return self.build_instruction(transcript="", turn_number=0, use_action=use_action, mix=mix)
    
    def build_instruction(
        self, transcript: str, turn_number: int, use_action: bool = False, mix: bool = False
    ) -> str:

        with open(self.template_path) as f:
            templates = yaml.safe_load(f)
        profile = self.profile
        partner = self.partner_profile
        if self.env is None:
            return "Instructions not available yet."
        env_dict = self.env.env
        if self.role_num == 0:
            agent_goal = env_dict["agent_goals"][0]
            agent_reason = env_dict["agent_reasons"][0]
            agent_private_knowledge = env_dict.get("agent1_private_knowledge", "")
        else:
            agent_goal = env_dict["agent_goals"][1]
            agent_reason = env_dict["agent_reasons"][1]
            agent_private_knowledge = env_dict.get("agent2_private_knowledge", "")
        # Choose barrier-aware templates if barrier prompts are present
        barrier_present = isinstance(env_dict.get("barrier_prompts"), dict)
        if mix:
            template_key = "social_task_instructions_action_mix_barrier" if barrier_present else "social_task_instructions_action_mix"
        else:
            if use_action:
                template_key = "social_task_instructions_action_barrier" if barrier_present else "social_task_instructions_action"
            else:
                template_key = "social_task_instructions_barrier" if barrier_present else "social_task_instructions"
        template = templates[template_key]
        memory_context = self.memory.get_memory_context(detailed=(turn_number == 0))
        # Extract private barrier notes per agent (do not expose in public info)
        barrier_cues = env_dict.get("barrier_cues") if isinstance(env_dict.get("barrier_cues"), dict) else None
        barrier_private_note = ""
        if barrier_cues:
            if self.role_num == 0:
                barrier_private_note = (barrier_cues.get("profile_note_A") or "").strip()
            else:
                barrier_private_note = (barrier_cues.get("profile_note_B") or "").strip()

        mapping = {
            "agent_name": profile.first_name,
            "partner_name": partner.first_name,
            "scenario": env_dict["scenario"],
            "agent_background": profile.public_info,
            "partner_background": partner.public_info,
            "agent_goal": agent_goal,
            "agent_reason": agent_reason,
            "agent_private_knowledge": agent_private_knowledge,
            "history": transcript,
            "turn_number": turn_number,
            "agent_age": profile.age,
            "agent_gender": profile.gender,
            "agent_occupation": profile.occupation,
            "agent_public_info": profile.public_info,
            "partner_age": partner.age,
            "partner_gender": partner.gender,
            "partner_occupation": partner.occupation,
            "partner_public_info": partner.public_info,
            "memory_context": memory_context,
            "barrier_private_note": barrier_private_note,
        }

        # Format the template with the mapping
        formatted_template = template.format(**mapping)
 
        
        # Remove private knowledge section if no private knowledge exists
        if not agent_private_knowledge.strip():
            # Remove the private knowledge section from the formatted template
            private_knowledge_section = f"IMPORTANT: You have private knowledge that {partner.first_name} does not know: {agent_private_knowledge}\nThis private knowledge should influence your strategy and communication, but you should not explicitly reveal it unless it serves your goal.\n\n"
            formatted_template = formatted_template.replace(private_knowledge_section, "")
        
        # Prepend any extra preface (e.g., barrier prompts) so it persists across turns
        if self.extra_instruction_preface:
            return f"{self.extra_instruction_preface}\n\n{formatted_template}"
        return formatted_template

    def set_extra_instruction_preface(self, preface: Optional[str]):
        self.extra_instruction_preface = (preface or '').strip() or None
    
    def update_instruction(
        self, transcript: List[str], turn_number: int, use_action: bool = False, mix: bool = False
    ):
        short_transcript = transcript[-6:] if len(transcript) > 6 else transcript
        transcript_text = "\n".join(short_transcript)
        
        self.instructions = self.build_instruction(
            transcript=transcript_text, turn_number=turn_number, use_action=use_action, mix=mix
        )

    
    
    def act(
        self, message=None, initial: bool = False, use_action: bool = False
    ) -> Union[str, Dict[str, Any]]:
        if initial:
            prompt = "Now, generate your initial message to start the conversation, try to be concise"
            response = direct_completion(self, message=prompt, use_action=use_action)
            
            # Normalize outputs
            if use_action:
                try:
                    response_json = json.loads(response) if isinstance(response, str) else response
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"❌ Error processing action response: {e}")
                    response_json = response
                original_response = json.loads(json.dumps(response_json))
                encrypted_response = json.loads(json.dumps(response_json))
            else:
                original_response = response
                encrypted_response = response

            # Log the response
            self.log.append(
                {
                    "initial": True,
                    "response_raw": original_response,
                    "response_encrypted": encrypted_response,
                }
            )
            return encrypted_response
        
        received = message

        if use_action:
            # Extract argument from message for action-based communication
            if isinstance(message, dict) and "action_type" in message:
                action_type = message.get("action_type", "")
                argument = message.get("argument", "")
                partner_name = self.partner_profile.first_name
                if action_type == "speak":
                    response = direct_completion(self, message=argument)
                elif action_type in ["non-verbal communication", "action"]:
                    response = direct_completion(self, message=f"{partner_name} {argument}")
                else:
                    response = direct_completion(self, message=str(message))
            else:
                response = direct_completion(self, message=str(message))
            try:
                response_json = json.loads(response) if isinstance(response, str) else response
            except (json.JSONDecodeError, KeyError) as e:
                print(f"❌ Error processing action response: {e}")
                response_json = response
            original_response = json.loads(json.dumps(response_json))
            encrypted_response = json.loads(json.dumps(response_json))
        else:
            # Handle text-based communication
            response = direct_completion(self, message=message)
            original_response = response
            encrypted_response = response
        
        print(f"💬 {self.name}: {str(original_response)[:100]}{'...' if len(str(original_response)) > 100 else ''}")
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
   
        assert task_type in {"goal", "reason", "knowledge"}, "task_type must be 'goal', 'reason', or 'knowledge'"
        
        if len(transcript) > 6:
            short_transcript = transcript[-6:]
        else:
            short_transcript = transcript
        
        formatted_options = "\n".join([f"{k}: {v}" for k, v in mcqa["options"].items()])
        conversation_str = "\n".join(short_transcript)
        
        question = mcqa["question"]
        if self.role_num == 0:
            question = question.replace("Agent 1", agent_name).replace("Agent 2", partner_name)
        else:
            question = question.replace("Agent 1", partner_name).replace("Agent 2", agent_name)

        agent_goal = ""
        agent_reason = ""
        if self.env and hasattr(self.env, "env"):
            env_dict = self.env.env
            if "agent_goals" in env_dict and len(env_dict["agent_goals"]) > self.role_num:
                agent_goal = env_dict["agent_goals"][self.role_num]
            if "agent_reasons" in env_dict and len(env_dict["agent_reasons"]) > self.role_num:
                agent_reason = env_dict["agent_reasons"][self.role_num]
        
        # Create prompt for MCQ prediction
        if task_type == "goal":
            prompt_key = "MCQ_Goal_Prediction_Prompt"
        elif task_type == "reason":
            prompt_key = "MCQ_Reason_Prediction_Prompt"
        else:  # knowledge
            prompt_key = "MCQ_Knowledge_Prediction_Prompt"
        
        prompt = test_prompt[prompt_key].format(
            agent_name=agent_name,
            partner_name=partner_name,
            question=question,
            options=formatted_options,
            transcript=conversation_str,
            agent_goal=agent_goal,
            agent_reason=agent_reason,
            scenario=self.env.env.get("scenario", "") if self.env and hasattr(self.env, "env") else "",
        )

        # Generate response using direct completion
        response = direct_completion(self, prompt, use_action=False).strip()
    
        selected = None
        confidence = 0.0
        reasoning = ""
        
        try:
            for line in response.split("\n"):
                if line.lower().startswith("selected:"):
                    selected = line.split(":")[1].strip().upper()
                elif line.lower().startswith("confidence:"):
                    confidence = float(line.split(":")[1].strip())
                elif line.lower().startswith("reasoning:"):
                    reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
        except Exception as e:
            print(f"❌ Error parsing MCQ response from {self.name}: {e}")
        
        # Clamp confidence to 0-1 range
        confidence = max(0.0, min(confidence, 1.0))
        
        # Determine confidence class using the binning system
        from ..utils.metrics import get_confidence_bin
        confidence_class = get_confidence_bin(confidence)
        
        # Print MCQ result summary with reasoning
        is_correct = selected == mcqa.get('correct_answer')
        print(f"🎯 {self.name} MCQ ({task_type}): {selected} (confidence: {confidence:.2f}) - {'✅' if is_correct else '❌'}")
        if reasoning:
            print(f"   💭 Reasoning: {reasoning}")
        
        return {
            "question": question,
            "selected": selected,
            "confidence": confidence,
            "confidence_class": confidence_class,
            "correct_answer": mcqa.get("correct_answer"),
            "correct": is_correct,  # Add this for metrics compatibility
            "is_correct": is_correct,
            "reasoning": reasoning,
            "options": mcqa["options"],
        }
    
    def reset_memory_for_scenario(self):
        """Reset memory for independent scenario simulation"""
        self.memory.reset_for_new_scenario()
    
    def update_memory_from_exchange(self, agent_message: str, partner_response: str, turn_number: int):
        """Update memory from a single exchange within the current conversation"""
        self.memory.update_from_exchange(agent_message, partner_response, turn_number)
