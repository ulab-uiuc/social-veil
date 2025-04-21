# Add to your encryption.py file
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from social_decipher.utils.model import ModelManager

from .encryption import BaseEncryption


class LanguageModelEncryption(BaseEncryption):
    def __init__(self, target_language: str, model_id: str, source_language: str = "English"):
        self.target_language = target_language
        self.source_language = source_language
        self.model_id = model_id

        self.can_understand = ModelManager.can_model_understand_language(
            model_id, target_language
        )

        model1, model2, _ = ModelManager.language_barrier_pair()
        self.translator_model = model1  # model1 is always the one that understands the barrier language

        # Debug logging
        print(f"LanguageModelEncryption for {model_id}: " 
              f"Can understand {target_language}: {self.can_understand}")
        
    def __call__(self, text: str) -> str:
        """
        Encrypt message by translating to target language if needed
        
        For models that understand the barrier language, messages are translated to that language
        For models that don't understand the barrier language, messages stay in source language
        """
        if not text:
            return text
            
        # If this model understands the target language, translate to it
        if self.can_understand:
            result = ModelManager.translate_text(
                text, 
                source_language=self.source_language,
                target_language=self.target_language, 
                model_id=self.translator_model
            )
            print(f"Translated to {self.target_language}: {result[:50]}...")
            return result
        
        # If model doesn't understand target language, keep in source language
        # The message will be translated if needed during relay_communication
        print(f"Keeping in {self.source_language} (model doesn't understand {self.target_language})")
        return text
