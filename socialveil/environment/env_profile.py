from typing import Optional


class EnvironmentProfile:
    def __init__(
        self,
        scenario: str,
        agent_goals: list[str],
        agent_reasons: list[str],
        agent_goals_mcqas: Optional[list] = None,
        agent_reasons_mcqas: Optional[list] = None,
        agent_knowledge_mcqas: Optional[list] = None,
        agent_relationship: str = "friend",
        agent1_private_knowledge: str = "",
        agent2_private_knowledge: str = "",
        agent1_profile: Optional[str] = None,
        agent2_profile: Optional[str] = None,
    ):
        self.env = {
            "scenario": scenario,
            "agent_goals": agent_goals,
            "agent_reasons": agent_reasons,
            "agent_relationship": agent_relationship,
            "agent_goals_mcqas": agent_goals_mcqas or [],
            "agent_reasons_mcqas": agent_reasons_mcqas or [],
            "agent_knowledge_mcqas": agent_knowledge_mcqas or [],
            "agent1_private_knowledge": agent1_private_knowledge,
            "agent2_private_knowledge": agent2_private_knowledge,
            "agent1_profile": agent1_profile,
            "agent2_profile": agent2_profile,
        }
