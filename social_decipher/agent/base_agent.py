import random
from openai import swarm
from language.language_proxy import LanguageProxy

class BaseAgent(swarm.Agent):
    def __init__(self, name: str, native_language: str, conlang: str):
        super().__init__(name)
        self.native_language = native_language
        self.conlang = conlang
        self.language_proxy = LanguageProxy(self.native_language, self.conlang)
        self.memory = {"vocabulary": set(), "syntax_rules": set()}  # Language learning memory

    def perceive(self, message: dict):
        translated_message = self.language_proxy.decrypt(message.get("speech", ""))
        return translated_message

    def act(self):
        possible_actions = ["gesture", "speak", "barter", "cooperate"]
        chosen_action = random.choice(possible_actions)
        print(f"{self.name} ({self.native_language}) chooses to {chosen_action}")
        
        if chosen_action == "speak":
            return self.generate_message()
        elif chosen_action == "gesture":
            return self.perform_gesture()
        elif chosen_action == "barter":
            return self.attempt_barter()
        return None

    def generate_message(self) -> dict:
        native_text = "Let's collaborate!"
        conlang_text = self.language_proxy.encrypt(native_text)
        return {"speech": conlang_text, "language": self.conlang}

    def perform_gesture(self) -> dict:
        gestures = ["wave", "point", "nod", "shake head"]
        return {"gesture": random.choice(gestures)}

    def attempt_barter(self) -> dict:
        return {"action": "offer_trade", "item": "Product", "price": random.randint(10, 50)}
