import json
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from rich import print

from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.utils.state import build_dynamic_rules_from_state, init_barrier_state
from social_decipher.utils.base import direct_completion
from ..utils.metrics import get_confidence_bin
from .agent_profile import AgentProfile
from ..utils.utils import parse_mcq_response_text

class SocialAgent:
    def __init__(
        self,
        name: str,
        profile: AgentProfile,
        partner_profile: AgentProfile,
        env: EnvironmentProfile,
        role_num: int,
        template_path: str = None,
        use_repair_prompt: bool = False,
    ):
        self.name = name
        self.env = env
        self.log = []
        self.role_num = role_num
        self.profile = profile
        self.partner_profile = partner_profile
        self.use_repair_prompt = use_repair_prompt
        
        # Set default template path if not provided
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "social_task.yaml")
            
        self.template_path = template_path
        self.instructions = self.set_static_instruction()

    def set_static_instruction(self) -> str:
        return self.build_instruction(transcript="", turn_number=0)
    
    def build_instruction(
        self, transcript: str, turn_number: int
    ) -> str:
        with open(self.template_path) as f:
            templates = yaml.safe_load(f)

        env_dict = self.env.env
        profile, partner = self.profile, self.partner_profile

        is_agent_a = self.role_num == 0
        agent_goal = env_dict["agent_goals"][0 if is_agent_a else 1]
        agent_reason = env_dict["agent_reasons"][0 if is_agent_a else 1]
        agent_private_knowledge = env_dict.get("agent1_private_knowledge" if is_agent_a else "agent2_private_knowledge", "")

        # Use the pre-formatted profile string if it exists, as it may contain barrier notes
        agent_background_str = env_dict.get("agent1_profile" if is_agent_a else "agent2_profile")
        if not agent_background_str:
            agent_background_str = profile.public_info

        partner_background_str = env_dict.get("agent2_profile" if is_agent_a else "agent1_profile")
        if not partner_background_str:
            partner_background_str = partner.public_info

        agent_key = "agentA" if is_agent_a else "agentB"
        barrier_for_this_agent = bool((env_dict.get("barrier_prompts") or {}).get(agent_key))
        barrier_type = env_dict.get("barrier_type") if barrier_for_this_agent else None
  
        template_key = (
            "social_task_instructions_barrier_semantic" if barrier_type == "semantic_structure" else
            "social_task_instructions_barrier_cultural" if barrier_type == "cultural_style" else
            "social_task_instructions_barrier_emotional" if barrier_type == "emotional_influence" else
            ("social_task_instructions_barrier" if barrier_for_this_agent else "social_task_instructions")
        )

        # Override for Agent B if repair prompt is enabled
        if not is_agent_a and self.use_repair_prompt:
            # Agent B should not have a barrier, so this path is safe
            if not barrier_for_this_agent:
                template_key = "social_task_instructions_repair"

        template = templates[template_key]

        if is_agent_a and barrier_for_this_agent and not isinstance(env_dict.get("barrier_state"), dict):
            try:
                init_barrier_state(env_dict)
            except Exception:
                pass
        # barrier_state is a dict with keys: semantic_strength, style_strength, affect_strength
        barrier_state = env_dict.get("barrier_state") if isinstance(env_dict.get("barrier_state"), dict) else None
        barrier_dynamic_rules: str = ""
  
        lines: List[str] = []

        if barrier_for_this_agent:
            # Prepend concise severity-driven rules so they dominate
            if is_agent_a and isinstance(env_dict, dict):
                dyn_map = build_dynamic_rules_from_state(env_dict, is_agent_a=True)
                sev_lines: List[str] = []
                if barrier_type == "semantic_structure":
                    for k in [
                        "univ_safety_guardrail",
                        "univ_adversarial_stance",
                        "univ_core_tactics",
                        "univ_content_integrity",
                        "sem_narrative",
                        "sem_tactics",
                        "sem_confusion",
                        "sem_examples",
                    ]:
                        v = dyn_map.get(k)
                        if isinstance(v, str) and v.strip():
                            sev_lines.append(v)
                elif barrier_type == "cultural_style":
                    for k in [
                        "univ_safety_guardrail",
                        "univ_adversarial_stance",
                        "univ_core_tactics",
                        "univ_content_integrity",
                        "cul_narrative",
                        "cul_tactics",
                        "cul_confusion",
                        "cul_examples",
                    ]:
                        v = dyn_map.get(k)
                        if isinstance(v, str) and v.strip():
                            sev_lines.append(v)
                elif barrier_type == "emotional_influence":
                    for k in [
                        "univ_safety_guardrail",
                        "univ_adversarial_stance",
                        "univ_core_tactics",
                        "univ_content_integrity",
                        "emo_narrative",
                        "emo_tactics",
                        "emo_confusion",
                        "emo_examples",
                    ]:
                        v = dyn_map.get(k)
                        if isinstance(v, str) and v.strip():
                            sev_lines.append(v)
                else:
                    for v in dyn_map.values():
                        if isinstance(v, str) and v.strip():
                            sev_lines.append(v)
                if sev_lines:
                    lines = sev_lines + lines

        if lines:
            # Deduplicate lines by content order-preserving
            seen = set()
            deduped: List[str] = []
            for ln in lines:
                key = ln.strip()
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(ln)
            barrier_dynamic_rules = "\n".join(deduped)

        # Build action list: if Agent A has a barrier, restrict to speak/leave only
        default_action_list = templates.get("action_list", "")
  
        if is_agent_a and barrier_type:
            action_list_str = (
                '\n- "speak": Say something verbally.\n'
                '- "leave": End the conversation.'
            )
        else:
            action_list_str = default_action_list

        mapping = {
            "agent_name": profile.first_name,
            "partner_name": partner.first_name,
            "scenario": env_dict["scenario"],
            "agent_background": agent_background_str,
            "partner_background": partner_background_str,
            "agent_goal": agent_goal,
            "agent_reason": agent_reason,
            "agent_private_knowledge": agent_private_knowledge,
            "history": transcript,
            "turn_number": turn_number,
            "agent_age": profile.age,
            "agent_gender": profile.gender,
            "agent_occupation": profile.occupation,
            "agent_public_info": agent_background_str,  # Use the potentially enriched string here
            "partner_age": partner.age,
            "partner_gender": partner.gender,
            "partner_occupation": partner.occupation,
            "partner_public_info": partner_background_str, # And here for the partner
            "action_list": action_list_str,
            "barrier_prompt": (env_dict.get("barrier_prompts") or {}).get(agent_key, ""),
            "barrier_dynamic_rules": barrier_dynamic_rules,
        }
        
        # Format the template with the mapping
        formatted_template = template.format(**mapping)
        return formatted_template

    def update_instruction(
        self, transcript: List[str], turn_number: int
    ):
        short_transcript = transcript[-6:] if len(transcript) > 6 else transcript
        transcript_text = "\n".join(short_transcript)
        
        self.instructions = self.build_instruction(
            transcript=transcript_text, turn_number=turn_number
        )
    
    def act(
        self, message=None, initial: bool = False
    ) -> Union[str, Dict[str, Any]]:
        
        if initial:
            self.instructions = self.build_instruction(transcript="", turn_number=0)
            prompt = "Now, generate your initial message to start the conversation, try to be concise"
            response = direct_completion(self, message=prompt)
            
            print(f"💬 {self.name}: {response}")
            try:
                response_json = json.loads(response) if isinstance(response, str) else response
            except (json.JSONDecodeError, KeyError) as e:
                print(f"❌ Error processing action response: {e}")
                response_json = response
            original_response = json.loads(json.dumps(response_json))

            # Log the response
            self.log.append(
                {
                    "initial": True,
                    "response_raw": original_response,
                }
            )
            return original_response
        
        received = message

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
            # Attempt to sanitize common backslash escapes (e.g., LaTeX) before retrying JSON parsing
            try:
                sanitized = response.replace("\\(", "(").replace("\\)", ")").replace("\\", "") if isinstance(response, str) else response
                response_json = json.loads(sanitized) if isinstance(sanitized, str) else sanitized
            except Exception:
                print(f"❌ Error processing action response: {e}")
                response_json = response
        original_response = json.loads(json.dumps(response_json))
        
        print(f"💬 {self.name}: {str(original_response)}")
        
        self.log.append(
            {
                "received_raw": received,
                "response_raw": original_response,
            }
        )
        
        return original_response


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
        # Generate response using direct completion, but extract text from JSON if needed
        response = direct_completion(self, prompt).strip()
        
        # If the response is JSON (due to action mode), extract the argument
        if response.startswith('{') and response.endswith('}'):
            try:
                import json
                response_json = json.loads(response)
                if isinstance(response_json, dict) and 'argument' in response_json:
                    response = response_json['argument']
            except json.JSONDecodeError:
                pass  # Keep original response if JSON parsing fails
    
        # Parse MCQ triple using utils to keep this class light
        selected, confidence, reasoning = parse_mcq_response_text(response)
    
        confidence = max(0.0, min(confidence, 1.0))
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
    
    
