from social_decipher.encryption import BaseEncryption
from social_decipher.utils.model import ModelManager


class LanguageModelEncryption(BaseEncryption):
    def __init__(
        self, target_language: str, model_id: str, source_language: str = "English"
    ):
 
        self.target_language = target_language
        self.source_language = source_language
        self.model_id = model_id

        # Check if model understands the target language 
        self.can_understand = ModelManager.can_model_understand_language(
            model_id, target_language
        )

    def __call__(self, text: str) -> str:
        if not text:
            return text

        if self.can_understand and self.source_language:
            result = ModelManager.translate_text(
                text,
                source_language=self.source_language,
                target_language=self.target_language,
                model_id="gpt-4o-mini",
            )
            # print(f"- Translated to {self.target_language}: {result}")
            return result

        return text
