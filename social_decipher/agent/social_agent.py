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

        agent_key = "agentA" if is_agent_a else "agentB"
        barrier_for_this_agent = bool((env_dict.get("barrier_prompts") or {}).get(agent_key))
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

            # Keep only minimal cues and avoid injecting multi-dimensional semantic hints
            if barrier_type == "cultural_style":
                _fmt_scalar("question_rate_hint", "Target question rate")      # cultural (high-context)
                _fmt_scalar("imperative_rate_hint", "Target imperative rate")  # cultural (low-context)
            elif barrier_type == "emotional_influence":
                _fmt_scalar("exclamation_bias", "Exclamation bias")           # emotional
                _fmt_scalar("turn_length_max", "Max sentences per turn")      # emotional

            # Inject dynamic barrier state (A-only) using episode barrier_cues (no severity text)
            if is_agent_a:
                if barrier_type == "semantic_structure":
                    # Lightweight keyword harvester to support implicit referencing (vagueness of referents and intent)
                    def _harvest_keywords(hist: str, env: Dict[str, Any], goal_text: str, reason_text: str, max_k: int = 6) -> List[str]:
                        import re
                        candidates: List[str] = []
                        text_sources = [hist or "", str((env or {}).get("scenario", "")), goal_text or "", reason_text or ""]
                        # 1) Quoted strings
                        for src in text_sources:
                            for m in re.findall(r"['\"]([^'\"]{2,60})['\"]", src):
                                s = m.strip()
                                if s:
                                    candidates.append(s)
                        # 2) Capitalized multi-word tokens (likely names/titles)
                        for src in text_sources:
                            for m in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", src):
                                s = m.strip()
                                if s:
                                    candidates.append(s)
                        # 3) Single capitalized tokens that look like product names
                        for src in text_sources:
                            for m in re.findall(r"\b([A-Z][A-Za-z]{2,})\b", src):
                                s = m.strip()
                                if s:
                                    candidates.append(s)
                        # 4) Content words from goal/reason (simple heuristic, exclude common stopwords)
                        stop = set(["the","a","an","to","for","and","or","of","in","on","with","by","at","from","is","are","was","were","be","been","being","this","that","these","those","it","its","as","about","into","over","under","you","your","my","our","their","his","her","him","she","he","they","them"]) 
                        for src in [goal_text or "", reason_text or ""]:
                            for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", src):
                                lw = w.lower()
                                if lw not in stop:
                                    candidates.append(w)
                        # Deduplicate and remove agent names
                        seen = set()
                        out_kw: List[str] = []
                        agent_names = {
                            self.profile.first_name,
                            self.partner_profile.first_name,
                            f"{self.profile.first_name} {self.profile.last_name}".strip(),
                            f"{self.partner_profile.first_name} {self.partner_profile.last_name}".strip(),
                        }
                        # Score candidates by how much they reveal intent
                        scores: Dict[str, float] = {}
                        def add_score(key: str, val: float):
                            scores[key] = scores.get(key, 0.0) + val
                        # Normalize pool
                        norm_cands: List[str] = []
                        for c in candidates:
                            c = c.strip()
                            if not c or len(c) < 2:
                                continue
                            if c in agent_names:
                                continue
                            low = c.lower()
                            if low in seen:
                                continue
                            seen.add(low)
                            norm_cands.append(c)
                        # Scoring heuristics
                        hist_low = (hist or "").lower()
                        scen_low = str((env or {}).get("scenario", "")).lower()
                        goal_low = (goal_text or "").lower()
                        reason_low = (reason_text or "").lower()
                        for c in norm_cands:
                            low = c.lower()
                            # Base
                            add_score(c, 0.5)
                            # Presence in goal/intent → hide these first
                            if low in goal_low:
                                add_score(c, 3.0)
                            if low in reason_low:
                                add_score(c, 1.5)
                            # Presence in scenario → contextual but useful to mask
                            if low in scen_low:
                                add_score(c, 0.8)
                            # Frequency in recent transcript
                            freq = hist_low.count(low)
                            if freq > 0:
                                add_score(c, min(1.0 + 0.2 * freq, 2.0))
                            # Capitalized multi-word bonuses (likely names/brands)
                            if any(ch.isupper() for ch in c) and (" " in c):
                                add_score(c, 1.0)
                        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                        top = [k for k, _ in ranked[:max_k]]
                        return top

                    kws = _harvest_keywords(transcript, env_dict, agent_goal, agent_reason)

                    if kws:
                        lines.append("- Must‑mask keywords (refer implicitly; do not say these strings verbatim): " + ", ".join(kws))
                        lines.append("- Refer using shells only (that one/this thing/that flavor); avoid explicit labels unless repeatedly pressed.")
                        lines.append("- Name‑reveal policy: 1st press → deflect descriptor; 2nd press → give minimal descriptor; 3rd press → reveal one minimal label then pivot.")
                        lines.append("- Per‑turn quota: keep at least two salient referents implicit (prefer those tied to your goal).")
                
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

        # Build action list: if Agent A has a barrier, restrict to speak/leave only
        default_action_list = templates.get("action_list", "")
  
        if is_agent_a and barrier_type:
            action_list_str = (
                '  - "speak": Say something verbally.\n'
                '  - "leave": End the conversation.'
            )
        else:
            action_list_str = default_action_list

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
            "action_list": action_list_str,
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
            template_key.startswith("social_task_instructions_barrier_") and
            barrier_type != "semantic_structure"):
            addl = f"\n- Additional barrier context: {barrier_preface.strip()}"
            # Remove any prior header injection remnants
            formatted_template = formatted_template.replace(addl, "")
            # Insert just before the leave note if present; otherwise append
            marker = "Note: You can \"leave\""
            if marker in formatted_template:
                formatted_template = formatted_template.replace(marker, addl + "\n\n" + marker)
            else:
                formatted_template = formatted_template + addl
        elif isinstance(barrier_preface, str) and barrier_preface.strip() and barrier_type != "semantic_structure":
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
            # Attempt to sanitize common backslash escapes (e.g., LaTeX) before retrying JSON parsing
            try:
                sanitized = response.replace("\\(", "(").replace("\\)", ")").replace("\\", "") if isinstance(response, str) else response
                response_json = json.loads(sanitized) if isinstance(sanitized, str) else sanitized
            except Exception:
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
    
        # Parse MCQ triple using utils to keep this class light
        selected, confidence, reasoning = parse_mcq_response_text(response)
        
        # Clamp confidence to 0-1 range
        confidence = max(0.0, min(confidence, 1.0))
        
        # Determine confidence class using the binning system
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
    
    
