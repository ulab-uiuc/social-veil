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
