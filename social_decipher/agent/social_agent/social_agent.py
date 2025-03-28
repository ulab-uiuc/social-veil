from agency_swarm.agents import Agent


class SocialAgent(Agent):
    def __init__(self, name, description="Intellective agent designed to simulate human-like conversation and interactions to solve certain difficult social tasks."):
        super().__init__(
            name=name,
            description=description,
            instructions="./instructions.md",
            files_folder="./files",
            schemas_folder="./schemas",
            tools=[],
            tools_folder="./tools",
            temperature=0.3,
            max_prompt_tokens=25000,
        )

    def response_validator(self, message):
        return message
