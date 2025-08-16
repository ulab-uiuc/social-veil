from social_decipher.encryption import BaseEncryption
from social_decipher.utils.model import ModelManager


class LanguageModelEncryption(BaseEncryption):
    def __init__(
        self, target_language: str, model_id: str, source_language: str = "English"
    ):
 
        self.target_language = target_language
        self.source_language = source_language
        self.model_id = model_id

        # Keep capability info (not used to gate translation)
        self.can_understand = ModelManager.can_model_understand_language(
            model_id, target_language
        )

    def __call__(self, text: str) -> str:
        if not text:
            return text

        # Always translate from source_language -> target_language for barrier output
        if self.source_language:
            result = ModelManager.translate_text(
                text,
                source_language=self.source_language,
                target_language=self.target_language,
                model_id="gpt-4o-mini",
            )
            return result

        return text
