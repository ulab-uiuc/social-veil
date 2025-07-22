# social_decipher/agent/agent_profile.py (Updated)

from dataclasses import dataclass, field
from typing import List, Any, Optional

@dataclass
class AgentProfile:
    pk: str
    first_name: str
    last_name: str
    age: int
    gender: str
    gender_pronoun: str
    occupation: str
    public_info: str
    big_five: str = ""
    moral_values: List[str] = field(default_factory=list)
    schwartz_personal_values: List[str] = field(default_factory=list)
    personality_and_values: str = ""
    decision_making_style: str = ""
    secret: str = ""
    mbti: str = ""
    tag: str = ""
    model_id: str = ""
    private_knowledge: str = ""

    @classmethod
    def from_dict(cls, d: dict, model_id: str = ""):
        # Normalize list fields
        def ensure_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                # Try to parse string representation of list
                if val.startswith("[") and val.endswith("]"):
                    import ast
                    try:
                        return ast.literal_eval(val)
                    except Exception:
                        return [val]
                return [val]
            return []
        return cls(
            pk=d.get("pk", ""),
            first_name=d.get("first_name", ""),
            last_name=d.get("last_name", ""),
            age=d.get("age", 0),
            gender=d.get("gender", ""),
            gender_pronoun=d.get("gender_pronoun", ""),
            occupation=d.get("occupation", ""),
            public_info=d.get("public_info", ""),
            big_five=d.get("big_five", ""),
            moral_values=ensure_list(d.get("moral_values", [])),
            schwartz_personal_values=ensure_list(d.get("schwartz_personal_values", [])),
            personality_and_values=d.get("personality_and_values", ""),
            decision_making_style=d.get("decision_making_style", ""),
            secret=d.get("secret", ""),
            mbti=d.get("mbti", ""),
            tag=d.get("tag", ""),
            model_id=model_id,
            private_knowledge=d.get("private_knowledge", ""),
        )

    def get_public_profile_info(self) -> str:
        info_parts = [
            f"Name: {self.first_name} {self.last_name}",
            f"Age: {self.age}",
            f"Gender: {self.gender} ({self.gender_pronoun})",
            f"Occupation: {self.occupation}",
            f"Public Information: {self.public_info}",
        ]
        if self.big_five:
            info_parts.append(f"Personality Traits: {self.big_five}")
        if self.personality_and_values:
            info_parts.append(f"Personality and Values: {self.personality_and_values}")
        if self.decision_making_style:
            info_parts.append(f"Decision Making Style: {self.decision_making_style}")
        return "\n".join(info_parts)

    def get_private_profile_info(self) -> str:
        public_info = self.get_public_profile_info()
        private_parts = []
        if self.secret:
            private_parts.append(f"Secret: {self.secret}")
        if self.private_knowledge:
            private_parts.append(f"Private Knowledge: {self.private_knowledge}")
        if private_parts:
            return f"{public_info}\n" + "\n".join(private_parts)
        else:
            return public_info

    def has_private_knowledge(self) -> bool:
        return bool(self.private_knowledge.strip())

    def has_secret(self) -> bool:
        return bool(self.secret.strip())