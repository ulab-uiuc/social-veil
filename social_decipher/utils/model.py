import os
from typing import Any

import anthropic
from openai import OpenAI


class ModelManager:
    """Manages different language models for language barrier experiments"""

    _openai_client = None
    _anthropic_client = None
    _hf_api_url = "https://api-inference.huggingface.co/models/"

    # Enhanced model capabilities with more precise language support
    MODEL_CAPABILITIES = {
        "gpt-4o-mini": {
            "languages": [
                "English",
                "Chinese",
                "Spanish",
                "French",
                "Japanese",
                "Korean",
                "Arabic",
                "Russian",
                "Portuguese",
                "German",
                "Italian",
                "Hindi",
                "Bengali",
                "Vietnamese",
                "Thai"
            ],
            "strength": "high",
            "provider": "openai",
        },
        "gpt-3.5-turbo": {
            "languages": [
                "English",
                "Chinese",
                "Spanish",
                "French",
                "Japanese",
                "German",
                "Russian",
            ],
            "strength": "medium",
            "provider": "openai",
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "languages": ["English"],
            "strength": "low",
            "provider": "huggingface",
        },
        "mistral-small-latest": {
            "languages": ["English", "French", "German", "Spanish"],
            "strength": "medium",
            "provider": "mistral",  # changed from huggingface
            "description": "7B parameter model with good instruction following",
        },
        "mistral-3b-latest": {
            "languages": ["English", "French", "German", "Spanish"],
            "strength": "medium",
            "provider": "mistral",  # changed from huggingface
            "description": "3B parameter model with good instruction following",
        },
        "mistral-small-2506": {
            "languages": ["English", "Spanish", "French", "Dutch", "German", "Russian", "Italian", "Hindi"],
        },
        "ministral-8b-latest": {
            "languages": ["English", "French", "German", "Spanish"],
            "strength": "medium",
            "provider": "mistral",  # changed from huggingface
            "description": "8B parameter model with good instruction following",
        },
        "microsoft/phi-2": {
            "languages": ["English"],
            "strength": "medium",
            "provider": "huggingface",
            "description": "2.7B parameter model with strong instruction capabilities",
        },
        "Qwen/Qwen2.5-7B-Instruct": {
            "languages": ["English"],  # This model actually has some Chinese ability
            "strength": "medium",
            "provider": "huggingface",
        },
        "claude-3-opus-20240229": {
            "languages": ["English","Chinese","Spanish","French","Japanese","German"],
            "strength": "high",
            "provider": "anthropic",
        },
        "claude-3-sonnet-20240229": {
            "languages": ["English", "Spanish", "French", "German"],
            "strength": "medium",
            "provider": "anthropic",
        },
        "claude-3-haiku-20240307": {
            "languages": ["English", "Spanish", "French"],
            "strength": "low",
            "provider": "anthropic",
        },
        "mistralai/Mistral-7B-v0.1": {
            "languages": ["English"],
            "strength": "medium",
            "provider": "huggingface",
        },
    }

    # Map actual models to their provider and API format
    MODEL_PROVIDERS = {
        "gpt-4o-mini": {"provider": "openai", "api_format": "openai"},
        "gpt-3.5-turbo": {"provider": "openai", "api_format": "openai"},
        "claude-3-opus-20240229": {"provider": "anthropic", "api_format": "anthropic"},
        "claude-3-sonnet-20240229": {
            "provider": "anthropic",
            "api_format": "anthropic",
        },
        "claude-3-haiku-20240307": {"provider": "anthropic", "api_format": "anthropic"},
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "provider": "huggingface",
            "api_format": "huggingface",
        },
        "mistralai/Mistral-7B-v0.1": {"provider": "huggingface", "api_format": "huggingface"},
        "mistral-small-latest": {"provider": "mistral", "api_format": "mistral"},
        "mistral-small-2506": {"provider": "mistral", "api_format": "mistral"},
        "ministral-3b-latest": {"provider": "mistral", "api_format": "mistral"},
        "Qwen/Qwen2.5-7B-Instruct": {
            "provider": "huggingface",
            "api_format": "huggingface",
        },
        "microsoft/phi-2": {"provider": "huggingface", "api_format": "huggingface"},
        
    }

    # Define language barrier pairs with explicit incompatibility
    LANGUAGE_BARRIER_PAIRS = [
        ("gpt-4o-mini", "Qwen/Qwen2.5-7B-Instruct", "Chinese"),
        ("gpt-4o-mini", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Chinese"),
        ("gpt-4o-mini", "ministral-3b-latest","Chinese"),
        ("gpt-4o-mini", "mistral-small-latest", "Chinese"),
        ("gpt-4o-mini", "microsoft/phi-2", "Chinese"),
        ("gpt-4o-mini", "claude-3-sonnet-20240229", "Chinese"),
        ("gpt-4o-mini", "claude-3-haiku-20240307", "Japanese"),
    ]

    @classmethod
    def _normalize_model_id(cls, model_id: str) -> str:
        """Normalize model id for capability/provider lookup.

        For HF-style ids like "org/model", keep the full id only for TinyLlama
        (which is explicitly listed). Otherwise use the model suffix for lookup.
        """
        if "/" in model_id and not model_id.startswith("TinyLlama/"):
            return model_id.split("/")[-1]
        return model_id

    @classmethod
    def get_openai_client(cls):
        """Get or initialize the OpenAI client"""
        if cls._openai_client is None:
            # For using a proxy like Astra Assistants API
            if os.environ.get("USE_ASTRA_PROXY", "false").lower() == "true":
                from astra_assistants import patch

                cls._openai_client = patch(OpenAI())
                print("Using Astra Assistants API proxy for multiple model support")
            else:
                cls._openai_client = OpenAI()
        return cls._openai_client

    @classmethod
    def get_anthropic_client(cls):
        """Get or initialize the Anthropic client"""
        if cls._anthropic_client is None:
            cls._anthropic_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        return cls._anthropic_client

    @classmethod
    def language_barrier_pair(cls, pair_index: Any = 0) -> tuple[str, str, str]:
        pair_index = (
            int(pair_index)
            if isinstance(pair_index, str) and pair_index.isdigit()
            else pair_index
        )

        pair = list(cls.LANGUAGE_BARRIER_PAIRS[pair_index])

        print(f"- Model 1: {pair[0]} (understands {pair[2]})")
        print(f"- Model 2: {pair[1]} (does NOT understand {pair[2]})")
        print(f"- Barrier language: {pair[2]}")

        return tuple(pair)

    @classmethod
    def can_model_understand_language(cls, model_id: str, language: str) -> bool:
        model_id = cls._normalize_model_id(model_id)

        supported_languages = cls.MODEL_CAPABILITIES.get(model_id, {}).get(
            "languages", []
        )

        return language in supported_languages

    @classmethod
    def translate_text(
        cls, text: str, source_language: str, target_language: str, model_id: str
    ) -> str:
        model_id = cls._normalize_model_id(model_id)

        model_info = cls.MODEL_PROVIDERS.get(model_id, {})
        provider = model_info.get("provider", "openai")

        if provider == "huggingface":
            model_id = "gpt-4o-mini"
            provider = "openai"

        if provider == "openai" or provider == "openai_proxy":
            result = cls._translate_with_openai(
                text, source_language, target_language, model_id
            )
        elif provider == "anthropic":
            result = cls._translate_with_anthropic(
                text, source_language, target_language, model_id
            )
        else:
            print(f"Unsupported provider: {provider}")
            return text

        return result

    @classmethod
    def _translate_with_openai(cls, text: str, source_language: str, target_language: str, model_id: str) -> str:
        # First, check if text is already in the target language
        client = cls.get_openai_client()
        
        # Language detection step
        detect_prompt = f"Identify if this text is in {source_language} or {target_language}. Just respond with ONLY the language name, no explanations:\n\n{text}"
        
        try:
            detect_response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a language detection tool. Respond with ONLY the language name."},
                    {"role": "user", "content": detect_prompt},
                ],
                temperature=0.0,
            )
            detected_language = detect_response.choices[0].message.content.strip().lower()
            
            # If already in target language or not in source language, don't translate
            if target_language.lower() in detected_language:
                print(f"Text already in {target_language}, skipping translation")
                return text
                
            # Only translate if text is in source language
            if source_language.lower() not in detected_language:
                print(f"Text not in {source_language}, might be {detected_language}. Skipping translation.")
                return text
                
            # Otherwise proceed with translation
            prompt = f"Translate the following {source_language} text to {target_language}. Maintain the original structure and format. Return only the translated text without explanations or metadata.\n\n{text}"
            
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            
            translated_text = response.choices[0].message.content
            
            # Verify translation actually happened
            if translated_text.strip() == text.strip():
                print(f"Warning: Translation returned identical text. Possible failure.")
            
            return translated_text
            
        except Exception as e:
            print(f"Translation error with {model_id}: {e}")
            return text

    @classmethod
    def _translate_with_anthropic(
        cls, text: str, source_language: str, target_language: str, model_id: str
    ) -> str:
        """Translate text using Anthropic models"""
        client = cls.get_anthropic_client()
        prompt = f"Translate the following {source_language} text to {target_language}. Maintain the original structure and format. Return only the translated text without explanations or metadata.\n\n{text}"

        try:
            response = client.messages.create(
                model=model_id,
                system="You are a professional translator.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.content[0].text
        except Exception as e:
            print(f"Translation error with {model_id}: {e}")
            return text

    @classmethod
    def list_available_pairs(cls):
        """List all available language barrier pairs (by index)."""
        print("\n=== Available Language Barrier Pairs ===")
        print("\nAll Pairs (use index with --pair parameter):")
        for idx, (model1, model2, language) in enumerate(cls.LANGUAGE_BARRIER_PAIRS):
            model1_understands = cls.can_model_understand_language(model1, language)
            model2_understands = cls.can_model_understand_language(model2, language)

            print(f"  {idx}: {model1} <-> {model2} ({language})")
            print(f"     - {model1} understands {language}: {model1_understands}")
            print(f"     - {model2} understands {language}: {model2_understands}")
