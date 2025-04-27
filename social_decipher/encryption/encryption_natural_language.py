from social_decipher.encryption import BaseEncryption
from social_decipher.utils.model import ModelManager


class LanguageModelEncryption(BaseEncryption):
    def __init__(
        self, target_language: str, model_id: str, source_language: str = "English"
    ):
        """
        Initialize language model encryption

        Args:
            target_language: Language to translate to/from
            model_id: Model ID used by this agent
            source_language: Original language (default English)
        """
        self.target_language = target_language
        self.source_language = source_language
        self.model_id = model_id

        # Check if model understands the target language
        self.can_understand = ModelManager.can_model_understand_language(
            model_id, target_language
        )

    def __call__(self, text: str) -> str:
        """
        Encrypt message by translating to target language if needed

        For models that understand the barrier language, messages are translated to that language
        For models that don't understand the barrier language, messages stay in source language
        """
        if not text:
            return text

        # IMPORTANT: Only translate if this model can understand the target language
        if self.can_understand:
            result = ModelManager.translate_text(
                text,
                source_language=self.source_language,
                target_language=self.target_language,
                model_id="gpt-4o-mini",
            )
            print(f"- Translated to {self.target_language}: {result}")
            return result

        # If model doesn't understand target language, keep in source language
        print(
            f"- Keeping in {self.source_language} (model doesn't understand {self.target_language})"
        )
        return text
