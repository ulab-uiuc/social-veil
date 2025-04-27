import json
from typing import Any


class AgentMemory:
    """
    Memory system for social agents to maintain knowledge across multiple scenarios
    """

    def __init__(self, agent_name: str, partner_name: str):
        self.agent_name = agent_name
        self.partner_name = partner_name
        self.scenarios_participated = 0
        self.past_interactions = []  # Stores important exchanges
        self.partner_observations = {}  # Observations about partner
        self.communication_strategies = {"successful": [], "unsuccessful": []}
        # Track goals and achievements
        self.goals_history = []
        # Track repair strategies for language barriers
        self.repair_strategies = {
            "used": {},  # Strategy -> frequency
            "effective": {},  # Strategy -> success rate
        }

    def update_after_scenario(
        self,
        scenario_log: list[str],
        scenario_results: dict[str, Any],
        agent_goal: str,
        goal_achieved: bool,
        encryption_enabled: bool = False,
    ):
        """Update memory after a completed scenario"""
        self.scenarios_participated += 1

        # Store the goal and whether it was achieved
        self.goals_history.append(
            {
                "scenario_num": self.scenarios_participated,
                "goal": agent_goal,
                "achieved": goal_achieved,
            }
        )

        # Extract key interactions (keep last 3-5 exchanges as key memories)
        key_exchanges = scenario_log[-min(6, len(scenario_log)) :]
        self.past_interactions.append(
            {
                "scenario_num": self.scenarios_participated,
                "key_exchanges": key_exchanges,
            }
        )

        # Extract communication strategies if encryption/language barriers were present
        if encryption_enabled:
            repair_strategies = self._extract_repair_strategies(scenario_log)
            for strategy, count in repair_strategies.items():
                if strategy in self.repair_strategies["used"]:
                    self.repair_strategies["used"][strategy] += count
                else:
                    self.repair_strategies["used"][strategy] = count

            # Update effectiveness based on goal achievement
            effectiveness_value = 1.0 if goal_achieved else 0.0
            for strategy in repair_strategies:
                if strategy in self.repair_strategies["effective"]:
                    # Running average of effectiveness
                    current = self.repair_strategies["effective"].get(strategy, 0)
                    count = self.repair_strategies["used"].get(strategy, 1)
                    self.repair_strategies["effective"][strategy] = (
                        current * (count - 1) + effectiveness_value
                    ) / count
                else:
                    self.repair_strategies["effective"][strategy] = effectiveness_value

        # Update partner observations
        self._update_partner_observations(scenario_log, scenario_results)

    def _extract_repair_strategies(self, scenario_log: list[str]) -> dict[str, int]:
        """Extract repair strategies used in the conversation"""
        strategies = {
            "simplification": 0,
            "repetition": 0,
            "gesturing": 0,
            "visual_description": 0,
            "clarification": 0,
            "key_words": 0,
        }

        # Simple keyword-based detection
        strategy_keywords = {
            "simplification": ["simple", "simpler", "simplified", "basic", "easier"],
            "repetition": ["repeat", "again", "reiterate", "resay"],
            "gesturing": ["gesture", "pointing", "wave", "hand", "motion"],
            "visual_description": ["show", "draw", "picture", "image", "visual"],
            "clarification": ["clarify", "understand", "mean", "unclear", "confused"],
            "key_words": ["key", "important", "essential", "main", "critical"],
        }

        # Count strategy appearances in the log
        for log_entry in scenario_log:
            if self.agent_name in log_entry:  # Only analyze this agent's messages
                message = log_entry.lower()
                for strategy, keywords in strategy_keywords.items():
                    for keyword in keywords:
                        if keyword in message:
                            strategies[strategy] += 1
                            break

        return strategies

    def _update_partner_observations(
        self, scenario_log: list[str], scenario_results: dict[str, Any]
    ):
        """Update observations about partner based on interactions"""
        # This would be more sophisticated in a real implementation
        # Simple implementation: extract some basic observations

        # Extract info about partner's communication style
        if len(scenario_log) > 4:
            # Check partner message patterns
            partner_messages = [msg for msg in scenario_log if self.partner_name in msg]

            avg_length = sum(len(msg.split()) for msg in partner_messages) / max(
                1, len(partner_messages)
            )

            if avg_length < 10:
                self.partner_observations["communication_style"] = "concise"
            elif avg_length > 20:
                self.partner_observations["communication_style"] = "verbose"
            else:
                self.partner_observations["communication_style"] = "balanced"

    def get_memory_context(self, detailed: bool = False) -> str:
        """Return formatted memory context for agent instructions"""
        if self.scenarios_participated == 0:
            return "This is your first interaction with " + self.partner_name + "."

        memory_lines = []
        memory_lines.append(
            f"You have interacted with {self.partner_name} across {self.scenarios_participated} different scenarios."
        )

        # Add goal achievements
        successes = sum(1 for goal in self.goals_history if goal["achieved"])
        if self.scenarios_participated > 0:
            memory_lines.append(
                f"You have achieved {successes} out of {self.scenarios_participated} social goals in past interactions."
            )

        # Add partner observations
        if self.partner_observations:
            style = self.partner_observations.get("communication_style")
            if style:
                memory_lines.append(
                    f"You've noticed {self.partner_name} tends to be {style} in their communication style."
                )

        # Add successful repair strategies if any (for language barriers)
        if self.repair_strategies["effective"]:
            # Get the most effective strategies
            effective = sorted(
                self.repair_strategies["effective"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            if (
                effective and effective[0][1] > 0.5
            ):  # Only include if somewhat effective
                strategy, _ = effective[0]
                memory_lines.append(
                    f"When communication was difficult, {strategy} seemed to help you get your point across."
                )

        # Add recent key exchanges (only if detailed memory requested)
        if detailed and self.past_interactions:
            memory_lines.append("\nRecent key exchanges:")
            last_scenario = self.past_interactions[-1]
            for exchange in last_scenario["key_exchanges"][
                -2:
            ]:  # Just the last couple exchanges
                memory_lines.append(exchange)

        return "\n".join(memory_lines)

    def to_dict(self) -> dict:
        """Convert memory to dictionary for serialization"""
        return {
            "agent_name": self.agent_name,
            "partner_name": self.partner_name,
            "scenarios_participated": self.scenarios_participated,
            "past_interactions": self.past_interactions,
            "partner_observations": self.partner_observations,
            "communication_strategies": self.communication_strategies,
            "goals_history": self.goals_history,
            "repair_strategies": self.repair_strategies,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMemory":
        """Create memory from dictionary"""
        memory = cls(data["agent_name"], data["partner_name"])
        memory.scenarios_participated = data["scenarios_participated"]
        memory.past_interactions = data["past_interactions"]
        memory.partner_observations = data["partner_observations"]
        memory.communication_strategies = data["communication_strategies"]
        memory.goals_history = data["goals_history"]
        memory.repair_strategies = data["repair_strategies"]
        return memory

    def save(self, filepath: str):
        """Save memory to file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AgentMemory":
        """Load memory from file"""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)
