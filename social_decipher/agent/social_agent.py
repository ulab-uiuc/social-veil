import json
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from rich import print

from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.utils.state import build_dynamic_rules_from_state, init_barrier_state
from social_decipher.utils.base import direct_completion

from .agent_profile import AgentProfile


class SocialAgent:
    def __init__(
        self,
        name: str,
        profile: AgentProfile,
        partner_profile: AgentProfile,
        env: EnvironmentProfile,
        role_num: int,
        mix: bool = False,
        template_path: str = None,
    ):
        self.name = name
        self.env = env
        self.log = []
        self.role_num = role_num
        self.profile = profile
        self.partner_profile = partner_profile
        
        # Set default template path if not provided
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "social_task.yaml")
            
        self.template_path = template_path
        self.instructions = self.set_static_instruction(mix)

    def set_static_instruction(self, mix=False) -> str:
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

        barrier_prompts_present = isinstance(env_dict.get("barrier_prompts"), dict)
        agent_key = "agentA" if is_agent_a else "agentB"
        barrier_for_this_agent = barrier_prompts_present and bool((env_dict.get("barrier_prompts") or {}).get(agent_key))
        barrier_type = env_dict.get("barrier_type") if barrier_for_this_agent else None

        template_key = (
            "social_task_instructions_barrier_semantic" if barrier_type == "semantic_structure" else
            "social_task_instructions_barrier_cultural" if barrier_type == "cultural_style" else
            "social_task_instructions_barrier_emotional" if barrier_type == "emotional_influence" else
            ("social_task_instructions_barrier" if barrier_for_this_agent else "social_task_instructions")
        )

        template = templates[template_key]

        # Build private cues block from barrier_cues
        barrier_cues = env_dict.get("barrier_cues") if isinstance(env_dict.get("barrier_cues"), dict) else None
        # Ensure barrier_state exists for Agent A when in barrier mode
        if is_agent_a and barrier_for_this_agent and not isinstance(env_dict.get("barrier_state"), dict):
            try:
                init_barrier_state(env_dict)
            except Exception:
                pass
        # barrier_state is a dict with keys: semantic_strength, style_strength, affect_strength
        barrier_state = env_dict.get("barrier_state") if isinstance(env_dict.get("barrier_state"), dict) else None
        barrier_private_note: str = ""
        barrier_dynamic_rules: str = ""
  
        if barrier_for_this_agent and barrier_cues:
            barrier_private_note = (barrier_cues.get("profile_note_A") or "").strip() if is_agent_a else ""

            lines: List[str] = []
            def _fmt_list(key: str, label: str):
                vals = barrier_cues.get(key)
                if isinstance(vals, list) and vals:
                    filtered = [str(v).strip() for v in vals if isinstance(v, str) and v.strip()]
                    if filtered:
                        lines.append(f"- {label}: " + ", ".join(filtered[:8]))

            def _fmt_scalar(key: str, label: str):
                val = barrier_cues.get(key)
                if isinstance(val, (int, float, str)) and str(val).strip():
                    lines.append(f"- {label}: {val}")

            # Keep only minimal cues that severity scales
            _fmt_list("ambiguity_devices", "Use ambiguity devices")            # semantic
            _fmt_scalar("question_rate_hint", "Target question rate")          # cultural (high-context)
            _fmt_scalar("imperative_rate_hint", "Target imperative rate")      # cultural (low-context)
            _fmt_scalar("exclamation_bias", "Exclamation bias")               # emotional
            _fmt_scalar("turn_length_max", "Max sentences per turn")          # emotional

            # Inject dynamic barrier state (A-only) using episode barrier_cues (no severity text)
            if is_agent_a:
                if barrier_type == "semantic_structure":
                    allow_exact = barrier_cues.get("allow_exact_numbers_only_if_partner_requests")
                    if isinstance(allow_exact, bool) and allow_exact:
                        lines.append("- Provide exact numbers/names only if the partner explicitly requests them.")
                    # Avoid duplicating lexicon/sentence bias and numeric targets covered by dynamic rules
                    # Keep devices hint minimal via the earlier summary
                elif barrier_type == "cultural_style":
                    style = str(barrier_cues.get("style", "high_context")).strip().lower()
                    if style:
                        lines.append(f"- Private style: {style.replace('_','-')}")
                    directness = barrier_cues.get("directness_level")
                    if isinstance(directness, (int, float)):
                        lines.append(f"- Directness level: {float(directness):.2f}")
                    shared_refs = barrier_cues.get("shared_background_refs")
                    if isinstance(shared_refs, list) and shared_refs:
                        examples = [str(v).strip() for v in shared_refs if isinstance(v, str) and v.strip()]
                        if examples:
                            lines.append("- Shared background cues: " + ", ".join(examples[:6]))
                    # Drop hedge lexicon / politeness / frames here (dynamic rules already include)
                elif barrier_type == "emotional_influence":
                    affect_lex = barrier_cues.get("affect_lexicon")
                    if isinstance(affect_lex, list) and affect_lex:
                        examples = [str(v).strip() for v in affect_lex if isinstance(v, str) and v.strip()]
                        if examples:
                            lines.append("- Affect lexicon: " + ", ".join(examples[:6]))
                    # Avoid duplicating sentence_length_bias here (dynamic rules include)

            # Prepend concise severity-driven rules so they dominate
            if is_agent_a and isinstance(env_dict, dict):
                dyn_map = build_dynamic_rules_from_state(env_dict, is_agent_a=True)
                sev_lines: List[str] = []
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
            "action_list": templates.get("action_list", ""),
            "barrier_prompt": (env_dict.get("barrier_prompts") or {}).get(agent_key, ""),
            "barrier_private_note": barrier_private_note,
            "barrier_dynamic_rules": barrier_dynamic_rules,
        }

        # Format the template with the mapping
        formatted_template = template.format(**mapping)
 
        # Remove the private knowledge section if absent
        if not agent_private_knowledge.strip():
            # Remove the private knowledge section from the formatted template
            private_knowledge_section = f"IMPORTANT: You have private knowledge that {partner.first_name} does not know: {agent_private_knowledge}\nThis private knowledge should influence your strategy and communication, but you should not explicitly reveal it unless it serves your goal.\n\n"
            formatted_template = formatted_template.replace(private_knowledge_section, "")
         
        # Get barrier prompt from environment if this agent has one
        barrier_preface = (env_dict.get("barrier_prompts") or {}).get(agent_key)
        
        # Place Additional barrier context at the end of the BARRIER section to keep dynamic rules dominant
        if (isinstance(barrier_preface, str) and barrier_preface.strip() and 
            template_key.startswith("social_task_instructions_barrier_")):
            addl = f"\n- Additional barrier context: {barrier_preface.strip()}"
            # Remove any prior header injection remnants
            formatted_template = formatted_template.replace(addl, "")
            # Insert just before the leave note if present; otherwise append
            marker = "Note: You can \"leave\""
            if marker in formatted_template:
                formatted_template = formatted_template.replace(marker, addl + "\n\n" + marker)
            else:
                formatted_template = formatted_template + addl
        elif isinstance(barrier_preface, str) and barrier_preface.strip():
            return f"{barrier_preface.strip()}\n\n{formatted_template}"
            
        return formatted_template

    def update_instruction(
        self, transcript: List[str], turn_number: int, mix: bool = False
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
            # Ensure latest barrier preface and cues are reflected in the very first turn
     
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
            print(f"❌ Error processing action response: {e}")
            response_json = response
        original_response = json.loads(json.dumps(response_json))
        
        print(f"💬 {self.name}: {str(original_response)}")
        
        # Log the response
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
    

