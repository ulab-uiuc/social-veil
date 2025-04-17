class EnvironmentProfile:
    def __init__(
        self,
        scenario: str,
        agent_goals: list[str],
        agent_reasons: list[str],
    ):
        self.env = {
            "scenario": scenario,
            "agent_goals": agent_goals,
            "agent_reasons": agent_reasons,
        }
