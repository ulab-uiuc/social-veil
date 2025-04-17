from typing import Dict

class EnvironmentProfile:
    def __init__(
        self,
        scenario: str,
        agent_goals: list[str],
        agent_reasons: list[str],
        agent_goals_mcqas: list[Dict[str, str]] = None,
        agent_reasons_mcqas: list[Dict[str, str]] = None,
    ):
        self.env = {
            "scenario": scenario,
            "agent_goals": agent_goals,
            "agent_reasons": agent_reasons,
            "agent_goals_mcqas": agent_goals_mcqas,
            "agent_reasons_mcqas": agent_reasons_mcqas,
        }
