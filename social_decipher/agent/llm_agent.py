from agent.base_agent import BaseAgent

class MultiLingualAgent(BaseAgent):
    def __init__(self, name: str, language: str):
        super().__init__(name, language)

    def translate(self, message: str) -> str:
        translations = {
            "English": {"Start": "Begin", "Complete Task": "Finish Work"},
            "French": {"Start": "Commencer", "Complete Task": "Terminer le Travail"},
            "Spanish": {"Start": "Empezar", "Complete Task": "Terminar Tarea"}
        }
        return translations.get(self.language, {}).get(message, message)

    def generate_message(self) -> dict:
        messages = {
            "English": "Let's work together!",
            "French": "Travaillons ensemble!",
            "Spanish": "¡Trabajemos juntos!"
        }
        return {"message": messages.get(self.language, "Hello"), "language": self.language}

