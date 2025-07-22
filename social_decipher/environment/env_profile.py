class EnvironmentProfile:
    def __init__(
        self,
        scenario: str,
        agent_goals: list[str],
        agent_reasons: list[str],
        agent_goals_mcqas: list[dict[str, str]] = None,
        agent_reasons_mcqas: list[dict[str, str]] = None,
        agent_knowledge_mcqas: list[dict[str, str]] = None,
        agent_relationship: str = "friend",
        agent1_private_knowledge: str = "",
        agent2_private_knowledge: str = "",
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
        }
