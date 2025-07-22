import json
import re
from typing import Dict, Any
import random

from openai import OpenAI

class SotopiaMCQGenerator:
    """Class to generate MCQs for Sotopia scenarios"""
    
    def __init__(self, client=None):
        """Initialize the generator with an OpenAI client"""
        self.client = client
        self.mcq_generation_prompt = """
You are an expert in designing evaluation questions for social agents in simulation environments.

You will generate a **set of multiple-choice questions (MCQs)** to evaluate agent understanding to social interaction.

Each agent should receive:
- Goal MCQ: about the other agent’s **true social goal**
- Reason MCQ: about the other agent’s **underlying motivation**
- Knowledge MCQ: about the other agent’s **private information**

### Important Design Constraints:
- The correct answer must be **clearly and uniquely correct** based on the agent's goal, value, or private knowledge.
- Avoid distractors that are just **emotionally softened** or **diplomatic paraphrases** of the correct goal (especially for goal MCQs).
- Make sure the distractors differ not only in **intent** but also in **commitment level** or **core behavior**.

### Distractor Construction Format:

#### 1. Goal MCQ:
- **Correct answer (A)**: The agent's actual goal.
- **B**: The other agent’s goal (to test goal confusion).
- **C**: A plausible but incorrect goal (related to setting).
- **D**: A generic or irrelevant goal.

#### 2. Reason MCQ:
- **Correct answer (A)**: Motivation that reflects this agent’s values and explains their goal (must align with `agentX_reason`)
- **B**: The other agent's motivation.
- **C**: Generic or vague motivation (e.g., "to get along").
- **D**: Contradictory or implausible motivation (e.g., “to gain fame” when agent is shy).

#### 3. Private Knowledge MCQ:
- **Correct answer (A)**: Private belief/plan of the agent.
- **B**: The other agent’s private belief.
- **C**: Public/shared knowledge.
- **D**: Incorrect or fabricated statement.

===========================
SCENARIO: {scenario}

AGENT 1 NAME: {agent1_name}
AGENT 2 NAME: {agent2_name}

AGENT RELATIONSHIP: {agent_relationship}

AGENT 1 GOAL: {agent1_goal}
AGENT 2 GOAL: {agent2_goal}

===========================

# IMPORTANT: The `agentX_reason` fields must reflect a plausible internal motivation for each agent’s goal.
# These will be used to construct the MCQ correct answer.
# Make sure they are:
# - Specific to the agent’s personality, values, or background
# - Causally tied to the stated goal
# - Not overly vague or philosophical (e.g., avoid "seeking peace" or "being a good person")
# - Distinct from the partner's reasoning
# - Should be inferable through the agent’s behavior in interaction

Please output your results in the following JSON format:

```json
{{
  "agent1_reason": "Agent 1's motivation for their goal",
  "agent2_reason": "Agent 2's motivation for their goal",
  "agent1_private_knowledge": "Private info only Agent 1 knows",
  "agent2_private_knowledge": "Private info only Agent 2 knows",
  "mcqs": {{
    "goals": [
      {{
        "question": "What is Agent 2's goal in this scenario?",
        "options": {{
          "A": "Agent 2's actual goal.",
          "B": "Agent 1's goal.",
          "C": "Plausible but incorrect goal.",
          "D": "Generic or irrelevant goal."
        }},
        "correct_answer": "A"
      }},
      {{
        "question": "What is Agent 1's goal in this scenario?",
        "options": {{
          "A": "Agent 1's actual goal.",
          "B": "Agent 2's goal.",
          "C": "Plausible but incorrect goal.",
          "D": "Generic or irrelevant goal."
        }},
        "correct_answer": "A"
      }}
    ],
    "reasons": [
      {{
        "question": "What motivates Agent 2 to pursue their goal?",
        "options": {{
          "A": "Motivation for agent.",
          "B": "Motivation that fits Agent 1, not Agent 2.",
          "C": "Generic or vague motivation.",
          "D": "Contradictory or implausible motivation."
        }},
        "correct_answer": "A"
      }},
      {{
        "question": "What motivates Agent 1 to pursue their goal?",
        "options": {{
          "A": "Motivation that reflects Agent 1's values.",
          "B": "Motivation that fits Agent 2, not Agent 1.",
          "C": "Generic or vague motivation.",
          "D": "Contradictory or implausible motivation."
        }},
        "correct_answer": "A"
      }}
    ],
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
"""

    def set_client(self, client):
        """Set the OpenAI client"""
        self.client = client

    def generate_mcqs_for_sotopia(self, sotopia_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MCQs for a Sotopia scenario"""
        if not self.client:
            raise ValueError("Client not set. Please call set_client() first.")
            
        # Extract data from sotopia format
        scenario = sotopia_data.get("scenario", "")
        agent1_goal = sotopia_data.get("agent1_goal")
        agent2_goal = sotopia_data.get("agent2_goal")
        relationship = sotopia_data.get("relationship", "")
        agent1_profile = sotopia_data.get("agent1_profile", "")
        agent2_profile = sotopia_data.get("agent2_profile", "")

 
        agent1_name = agent1_profile.split(",")[0].strip() if agent1_profile else "Agent 1"
        agent2_name = agent2_profile.split(",")[0].strip() if agent2_profile else "Agent 2"

        # Format prompt
        prompt = self.mcq_generation_prompt.format(
            scenario=scenario,
            agent1_goal=agent1_goal,
            agent2_goal=agent2_goal,
            agent_relationship=relationship,
            agent1_name=agent1_name,
            agent2_name=agent2_name
        )
  
        try:
            # Generate MCQs using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates detailed MCQs for social scenarios.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            
            # Extract content
            response_text = response.choices[0].message.content
        
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            json_str = json_match.group(1) if json_match else response_text.strip()  
            mcq_data = json.loads(json_str)

            def shuffle_mcq_options(mcq_list):
                for mcq in mcq_list:
                    options = mcq["options"]
                    # Find the original correct answer value (always 'A' in the template)
                    correct_value = options["A"]
                    # Shuffle the options
                    items = list(options.items())
                    random.shuffle(items)
                    # Assign new keys A/B/C/D
                    new_options = {}
                    correct_key = None
                    for idx, (k, v) in enumerate(items):
                        new_key = chr(ord('A') + idx)
                        new_options[new_key] = v
                        if v == correct_value:
                            correct_key = new_key
                    mcq["options"] = new_options
                    mcq["correct_answer"] = correct_key
            # Shuffle options for all MCQs
            for section in ["goals", "reasons", "knowledge"]:
                if section in mcq_data.get("mcqs", {}):
                    shuffle_mcq_options(mcq_data["mcqs"][section])

            print(len(mcq_data.get("mcqs", {}).get("goals", [])), "goal questions generated")
            print(len(mcq_data.get("mcqs", {}).get("reasons", [])), "reason questions generated")
            print(len(mcq_data.get("mcqs", {}).get("knowledge", [])), "knowledge questions generated")
            return {
                "agent1_reason": mcq_data.get("agent1_reason", ""),
                "agent2_reason": mcq_data.get("agent2_reason", ""),
                "agent1_private_knowledge": mcq_data.get("agent1_private_knowledge", ""),
                "agent2_private_knowledge": mcq_data.get("agent2_private_knowledge", ""),
                "mcqs": mcq_data.get("mcqs", {"goals": [], "reasons": [], "knowledge": []})
            }
                    
        except Exception as e:
            print(f"Error during MCQ generation: {e}")
            return {
                "agent1_reason": "",
                "agent2_reason": "",
                "agent1_private_knowledge": "",
                "agent2_private_knowledge": "",
                "mcqs": {"goals": [], "reasons": [], "knowledge": []}
            }