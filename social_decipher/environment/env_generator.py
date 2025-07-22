import json
import re

from ..environment.env_profile import EnvironmentProfile

class EnvironmentGenerator:
    def __init__(self, client=None):
        self.client = client
        # Using triple braces to escape the JSON braces in the format string
        self.prompt_template = """
Generate a social scenario involving two individuals with different goals and motivations. The scenario should be realistic, specific, and contain enough context to understand the social dynamics.

Important guidelines:
- Do not mention specific names; just generally describe the scenarios and roles, do not mention their jobs
- Goals should be general descriptions of what each person wants to achieve
- Reasons should explain the underlying motivations
- Make sure the scenario involves social goals that could lead to different behaviors
- The scenario should involve some tension or potential conflict between the goals

IMPORTANT: Generate your response as a valid JSON object with the following structure:

```
{{
  "scenarios": [
    {{
      "scenario": "1-2 sentence description of a realistic social scenario involving two people",
      "agent1": {{
        "goal": "Clear, concise goal for the first person in the scenario",
        "reason": "Underlying motivation or reason for Agent 1's goal"
      }},
      "agent2": {{
        "goal": "Clear, concise goal for the second person that creates an interesting social dynamic",
        "reason": "Underlying motivation or reason for Agent 2's goal"
      }},
      "mcqs": {{
        "goals": [
          {{
            "question": "What is Agent 1's goal in this scenario?",
            "options": {{
              "A": "Agent 1's actual goal",
              "B": "Agent 2's goal",
              "C": "Plausible but incorrect goal",
              "D": "Plausible but incorrect goal"
            }},
            "correct_answer": "A"
          }},
          {{
            "question": "What is Agent 2's goal in this scenario?",
            "options": {{
              "A": "Agent 2's actual goal",
              "B": "Agent 1's goal",
              "C": "Plausible but incorrect goal",
              "D": "Plausible but incorrect goal"
            }},
            "correct_answer": "A"
          }}
        ],
        "reasons": [
          {{
            "question": "What is the reason behind Agent 1's goal?",
            "options": {{
              "A": "Agent 1's actual reason",
              "B": "Agent 2's reason",
              "C": "Plausible but incorrect reason",
              "D": "Plausible but incorrect reason"
            }},
            "correct_answer": "A"
          }},
          {{
            "question": "What is the reason behind Agent 2's goal?",
            "options": {{
              "A": "Agent 2's actual reason",
              "B": "Agent 1's reason",
              "C": "Plausible but incorrect reason",
              "D": "Plausible but incorrect reason"
            }},
            "correct_answer": "A"
          }}
        ]
      }}
    }}
  ]
}}
```

Ensure your response is properly formatted, valid JSON that can be parsed by Python's json.loads() function. Do not include any text outside of the JSON object.

{additional_instructions}
"""

    def set_client(self, client):
        self.client = client

    def generate_prompt(self, num_scenarios: int = 1, domain: str = None) -> str:
        additional_instructions = f"Please generate {num_scenarios} different scenario(s) in the 'scenarios' array."
        if domain:
            additional_instructions += f" Focus on scenarios in the {domain} domain."
        return self.prompt_template.format(
            additional_instructions=additional_instructions
        )

    def parse_response(self, response: str) -> list[EnvironmentProfile]:
        # Extract JSON from response (handling cases where there might be text outside the JSON)
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_str = response.strip()

        try:
            # Parse the JSON
            data = json.loads(json_str)

            environment_profiles = []
            for scenario_data in data.get("scenarios", []):
                try:
                    # Extract data from JSON structure
                    scenario = scenario_data.get("scenario", "")
                    agent1_goal = scenario_data.get("agent1", {}).get("goal", "")
                    agent1_reason = scenario_data.get("agent1", {}).get("reason", "")
                    agent2_goal = scenario_data.get("agent2", {}).get("goal", "")
                    agent2_reason = scenario_data.get("agent2", {}).get("reason", "")

                    # Get MCQs
                    goal_mcqas = scenario_data.get("mcqs", {}).get("goals", [])
                    reason_mcqas = scenario_data.get("mcqs", {}).get("reasons", [])

                    profile = EnvironmentProfile(
                        scenario=scenario,
                        agent_goals=[agent1_goal, agent2_goal],
                        agent_reasons=[agent1_reason, agent2_reason],
                        agent_goals_mcqas=goal_mcqas,
                        agent_reasons_mcqas=reason_mcqas,
                    )

                    environment_profiles.append(profile)
                except Exception as e:
                    print(f"Error parsing scenario: {e}")
                    continue

            return environment_profiles
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Received response: {response}")
            return []

    def generate_environments(
        self,
        num_scenarios: int = 1,
        domain: str = None,
        temperature: float = 0.7,
        model: str = None,
    ) -> list[EnvironmentProfile]:
        if not self.client:
            raise ValueError("Client not set")

        prompt = self.generate_prompt(num_scenarios, domain)

        # Create the OpenAI chat completion request
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates detailed social scenarios in valid JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
        )

        # Extract text content from the response
        response_text = response.choices[0].message.content
     
        return self.parse_response(response_text)

