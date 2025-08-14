import json
import re
from typing import Dict, Any
import random
import argparse
import os

from openai import OpenAI

class SotopiaMCQGenerator:
    """Class to generate MCQs for Sotopia scenarios"""
    
    def __init__(self, client=None):
        """Initialize the generator with an OpenAI client"""
        self.client = client
 
        # Focused prompt: generate only reasons and reason MCQs (more constrained and grounded)
        self.reason_generation_prompt = """
You are an expert in social cognition and conversational analysis.

Task: Generate ONLY (1) each agent's underlying motivation (reason) and (2) two reason MCQs.

Grounding and constraints:
- Reasons must be causally tied to the agent's goal (use "because", "in order to", or "so that").
- Reasons must be grounded in the agent's profile (personality/values/decision-making/public info) and appropriate to the relationship.
- Reasons for Agent 1 and Agent 2 must be distinct and not paraphrases of each other.
- Do NOT leak or invent private knowledge; use only the scenario, goals, relationship, and profile.
- Avoid generic/vague wording (e.g., "to get along", "to be happy").
- Style: one concise sentence (max 35 words) per reason.

MCQ construction rules (STRICT):
- Q1 asks: "What motivates Agent 2 to pursue their goal?"; Option A must be EXACTLY the same text as agent2_reason.
- Q2 asks: "What motivates Agent 1 to pursue their goal?"; Option A must be EXACTLY the same text as agent1_reason.
- For each question: B = the OTHER agent's reason; C = a generic/vague motivation; D = a contradictory/implausible motivation. Set correct_answer to "A".

Input context:
SCENARIO: {scenario}
AGENT RELATIONSHIP: {agent_relationship}
AGENT 1 GOAL: {agent1_goal}
AGENT 2 GOAL: {agent2_goal}
AGENT 1 PROFILE: {agent1_profile}
AGENT 2 PROFILE: {agent2_profile}

Output strictly in this JSON format:
```json
{{
  "agent1_reason": "Agent 1's motivation (<= 35 words, grounded, causal)",
  "agent2_reason": "Agent 2's motivation (<= 35 words, grounded, causal)",
  "mcqs": {{
    "reasons": [
      {{
        "question": "What motivates Agent 2 to pursue their goal?",
        "options": {{
          "A": "<EXACTLY agent2_reason>",
          "B": "<agent1_reason>",
          "C": "<generic/vague motivation>",
          "D": "<contradictory/implausible motivation>"
        }},
        "correct_answer": "A"
      }},
      {{
        "question": "What motivates Agent 1 to pursue their goal?",
        "options": {{
          "A": "<EXACTLY agent1_reason>",
          "B": "<agent2_reason>",
          "C": "<generic/vague motivation>",
          "D": "<contradictory/implausible motivation>"
        }},
        "correct_answer": "A"
      }}
    ]
  }}
}}
```
"""

        # Focused prompt: generate only private knowledge and knowledge MCQs
        self.knowledge_generation_prompt = """
You are an expert in designing evaluation questions for social agents in simulation environments.

Task: Only generate private knowledge for each agent and the knowledge MCQs. Do not include goal MCQs or reason MCQs.

Input context:
SCENARIO: {scenario}
AGENT RELATIONSHIP: {agent_relationship}
AGENT 1 PROFILE: {agent1_profile}
AGENT 2 PROFILE: {agent2_profile}
AGENT 1 GOAL: {agent1_goal}
AGENT 2 GOAL: {agent2_goal}

Output strictly in this JSON format:
```json
{{
  "agent1_private_knowledge": "Private info only Agent 1 knows",
  "agent2_private_knowledge": "Private info only Agent 2 knows",
  "mcqs": {{
    "knowledge": [
      {{
        "question": "Which of the following is something only Agent 2 knows?",
        "options": {{
          "A": "Agent 2's private knowledge.",
          "B": "Agent 1's private knowledge.",
          "C": "Public/shared information.",
          "D": "Incorrect or fabricated info."
        }},
        "correct_answer": "A"
      }},
      {{
        "question": "Which of the following is something only Agent 1 knows?",
        "options": {{
          "A": "Agent 1's private knowledge.",
          "B": "Agent 2's private knowledge.",
          "C": "Public/shared information.",
          "D": "Incorrect or fabricated info."
        }},
        "correct_answer": "A"
      }}
    ]
  }}
}}
```
"""

        # Focused prompt: generate only goal MCQs (no reasons or knowledge)
        self.goal_generation_prompt = """
You are an expert in designing evaluation questions for social agents in simulation environments.

Task: ONLY generate two goal MCQs that test whether each agent can infer the partner's true goal from the scenario and profiles. Do NOT generate reasons or knowledge items.

Construction rules:
- Q1 asks: "What is Agent 2's goal in this scenario?"
  - Option A MUST be EXACTLY Agent 2's true goal from input.
  - Option B MUST be Agent 1's goal from input (to test confusion).
  - Option C MUST be a plausible but incorrect goal grounded in the scenario.
  - Option D MUST be a generic/irrelevant goal.
- Q2 mirrors Q1 for Agent 1.
- Set correct_answer to "A" for both questions.

Input context:
SCENARIO: {scenario}
AGENT RELATIONSHIP: {agent_relationship}
AGENT 1 GOAL: {agent1_goal}
AGENT 2 GOAL: {agent2_goal}
AGENT 1 PROFILE: {agent1_profile}
AGENT 2 PROFILE: {agent2_profile}

Output strictly in this JSON format:
```json
{{
  "mcqs": {{
    "goals": [
      {{
        "question": "What is Agent 2's goal in this scenario?",
        "options": {{
          "A": "<EXACTLY agent2_goal>",
          "B": "<agent1_goal>",
          "C": "<plausible but incorrect goal>",
          "D": "<generic or irrelevant goal>"
        }},
        "correct_answer": "A"
      }},
      {{
        "question": "What is Agent 1's goal in this scenario?",
        "options": {{
          "A": "<EXACTLY agent1_goal>",
          "B": "<agent2_goal>",
          "C": "<plausible but incorrect goal>",
          "D": "<generic or irrelevant goal>"
        }},
        "correct_answer": "A"
      }}
    ]
  }}
}}
```
"""

    def _call_openai(self, prompt: str, temperature: float = 0.7, model: str = "gpt-4o-mini") -> str:
        if not self.client:
            self.client = OpenAI()
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
        )
        return response.choices[0].message.content

    def _extract_json(self, content: str) -> Dict[str, Any]:
        m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        json_str = m.group(1) if m else content.strip()
        return json.loads(json_str)

    def _shuffle_mcq_options(self, mcq_list):
        for mcq in mcq_list:
            options = mcq.get("options", {})
            if not options:
                continue
            correct_value = options.get("A")
            items = list(options.items())
            random.shuffle(items)
            new_options = {}
            correct_key = None
            for idx, (_k, v) in enumerate(items):
                new_key = chr(ord('A') + idx)
                new_options[new_key] = v
                if v == correct_value:
                    correct_key = new_key
            mcq["options"] = new_options
            if correct_key:
                mcq["correct_answer"] = correct_key

    def generate_goal_mcqs(self, sotopia_data: Dict[str, Any], temperature: float = 0.2, model: str = "gpt-4o") -> Dict[str, Any]:
        scenario = sotopia_data.get("scenario", "")
        agent1_goal = sotopia_data.get("agent1_goal", "")
        agent2_goal = sotopia_data.get("agent2_goal", "")
        relationship = sotopia_data.get("relationship", "")
        agent1_profile = sotopia_data.get("agent1_profile", "")
        agent2_profile = sotopia_data.get("agent2_profile", "")

        prompt = self.goal_generation_prompt.format(
            scenario=scenario,
            agent1_goal=agent1_goal,
            agent2_goal=agent2_goal,
            agent_relationship=relationship,
            agent1_profile=agent1_profile,
            agent2_profile=agent2_profile,
        )

        try:
            content = self._call_openai(prompt, temperature=temperature, model=model)
            data = self._extract_json(content)
            goals = data.get("mcqs", {}).get("goals", [])

            # Enforce Option A exact match with provided true goals
            if isinstance(goals, list):
                if len(goals) >= 1 and isinstance(goals[0], dict):
                    opts0 = goals[0].setdefault("options", {})
                    opts0["A"] = agent2_goal
                    goals[0]["correct_answer"] = "A"
                if len(goals) >= 2 and isinstance(goals[1], dict):
                    opts1 = goals[1].setdefault("options", {})
                    opts1["A"] = agent1_goal
                    goals[1]["correct_answer"] = "A"

            # Do NOT shuffle here; allow caller to randomize positions if desired
            return {"mcqs": {"goals": goals}}
        except Exception as e:
            print(f"Error during goal MCQ generation: {e}")
            return {"mcqs": {"goals": []}}

    def generate_reasons(self, sotopia_data: Dict[str, Any], temperature: float = 0.2, model: str = "gpt-4o") -> Dict[str, Any]:
        scenario = sotopia_data.get("scenario", "")
        agent1_goal = sotopia_data.get("agent1_goal", "")
        agent2_goal = sotopia_data.get("agent2_goal", "")
        relationship = sotopia_data.get("relationship", "")
        agent1_profile = sotopia_data.get("agent1_profile", "")
        agent2_profile = sotopia_data.get("agent2_profile", "")

        prompt = self.reason_generation_prompt.format(
            scenario=scenario,
            agent1_goal=agent1_goal,
            agent2_goal=agent2_goal,
            agent_relationship=relationship,
            agent1_profile=agent1_profile,
            agent2_profile=agent2_profile,
        )

        try:
            content = self._call_openai(prompt, temperature=temperature, model=model)
            data = self._extract_json(content)
            # Extract reasons
            agent1_reason = (data.get("agent1_reason") or "").strip()
            agent2_reason = (data.get("agent2_reason") or "").strip()
            reasons = data.get("mcqs", {}).get("reasons", [])

            # Enforce MCQ Option A alignment with generated reasons and keep A as the correct answer
            if isinstance(reasons, list):
                if len(reasons) >= 1 and isinstance(reasons[0], dict):
                    opts0 = reasons[0].setdefault("options", {})
                    opts0["A"] = agent2_reason
                    reasons[0]["correct_answer"] = "A"
                if len(reasons) >= 2 and isinstance(reasons[1], dict):
                    opts1 = reasons[1].setdefault("options", {})
                    opts1["A"] = agent1_reason
                    reasons[1]["correct_answer"] = "A"

            # Do NOT shuffle reasons, to preserve Option A strictness
            return {
                "agent1_reason": agent1_reason,
                "agent2_reason": agent2_reason,
                "mcqs": {"reasons": reasons},
            }
        except Exception as e:
            print(f"Error during reason generation: {e}")
            return {"agent1_reason": "", "agent2_reason": "", "mcqs": {"reasons": []}}

    def generate_knowledge(self, sotopia_data: Dict[str, Any], temperature: float = 0.7, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        scenario = sotopia_data.get("scenario", "")
        agent1_goal = sotopia_data.get("agent1_goal", "")
        agent2_goal = sotopia_data.get("agent2_goal", "")
        relationship = sotopia_data.get("relationship", "")
        agent1_profile = sotopia_data.get("agent1_profile", "")
        agent2_profile = sotopia_data.get("agent2_profile", "")

        prompt = self.knowledge_generation_prompt.format(
            scenario=scenario,
            agent1_goal=agent1_goal,
            agent2_goal=agent2_goal,
            agent_relationship=relationship,
            agent1_profile=agent1_profile,
            agent2_profile=agent2_profile,
        )

        try:
            content = self._call_openai(prompt, temperature=temperature, model=model)
            data = self._extract_json(content)
            knowledge = data.get("mcqs", {}).get("knowledge", [])
            self._shuffle_mcq_options(knowledge)
            return {
                "agent1_private_knowledge": data.get("agent1_private_knowledge", ""),
                "agent2_private_knowledge": data.get("agent2_private_knowledge", ""),
                "mcqs": {"knowledge": knowledge},
            }
        except Exception as e:
            print(f"Error during knowledge generation: {e}")
            return {
                "agent1_private_knowledge": "",
                "agent2_private_knowledge": "",
                "mcqs": {"knowledge": []},
            }

