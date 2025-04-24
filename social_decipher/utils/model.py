import os
from typing import Any, Dict, List, Optional, Tuple
import requests
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
            "languages": ["English", "Chinese", "Spanish", "French", "Japanese", "Korean", 
                        "Arabic", "Russian", "Portuguese", "German", "Italian", "Hindi", "Bengali"],
            "strength": "high",
            "provider": "openai"
        },
        "gpt-3.5-turbo": {
            "languages": ["English", "Chinese", "Spanish", "French", "Japanese", "German", "Russian"],
            "strength": "medium",
            "provider": "openai"
        },
        # IMPORTANT FIX: TinyLlama definitely does NOT understand Chinese
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "languages": ["English"],  # Reduced language support to be more accurate
            "strength": "low",
            "provider": "huggingface",
            "description": "1.1B parameter model, very CPU-friendly"
        },
        "claude-3-opus-20240229": {
            "languages": ["English", "Chinese", "Spanish", "French", "Japanese", "German"],
            "strength": "high",
            "provider": "anthropic"
        },
        "claude-3-sonnet-20240229": {
            "languages": ["English", "Spanish", "French", "German"],
            "strength": "medium",
            "provider": "anthropic"
        },
        "claude-3-haiku-20240307": {
            "languages": ["English", "Spanish", "French"],
            "strength": "low",
            "provider": "anthropic"
        }
    }
    
    # Map actual models to their provider and API format
    MODEL_PROVIDERS = {
        "gpt-4o-mini": {"provider": "openai", "api_format": "openai"},
        "gpt-3.5-turbo": {"provider": "openai", "api_format": "openai"},
        "claude-3-opus-20240229": {"provider": "anthropic", "api_format": "anthropic"},
        "claude-3-sonnet-20240229": {"provider": "anthropic", "api_format": "anthropic"},
        "claude-3-haiku-20240307": {"provider": "anthropic", "api_format": "anthropic"},
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"provider": "huggingface", "api_format": "huggingface"},
        # Proxy versions
        "anthropic/claude-3-opus-20240229": {"provider": "openai", "api_format": "openai_proxy"},
        "anthropic/claude-3-sonnet-20240229": {"provider": "openai", "api_format": "openai_proxy"},
        "anthropic/claude-3-haiku-20240307": {"provider": "openai", "api_format": "openai_proxy"}
    }
    
    # Define language barrier pairs with explicit incompatibility
    LANGUAGE_BARRIER_PAIRS = [
        # Format: (model_with_strong_support, model_with_weaker_support, barrier_language)

        # gpt-4o-mini is generally stronger in lower-resource languages
        ("gpt-4o-mini", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Chinese"),
        ("gpt-4o-mini", "claude-3-sonnet-20240229", "Chinese"),
        ("gpt-4o-mini", "claude-3-haiku-20240307", "Japanese"),
        ("gpt-4o-mini", "claude-3-haiku-20240307", "Korean"),
    ]

    NAMED_PAIRS = {
        "gpt-tiny-chinese": 0,    # GPT-4o-mini and TinyLlama with Chinese barrier
        "gpt-claude-chinese": 1,  # GPT-4o-mini and Claude Sonnet with Chinese barrier
        "gpt-claude-japanese": 2,  # GPT-4o-mini and Claude Haiku with Japanese barrier
    }

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
    def language_barrier_pair(cls, pair_index_or_name: Any = 0) -> Tuple[str, str, str]:
        """
        Return a model pair with a true language barrier
        
        Args:
            pair_index_or_name: Index of the pair to use from LANGUAGE_BARRIER_PAIRS,
                               or a named pair from NAMED_PAIRS
            
        Returns:
            Tuple of (model1, model2, barrier_language)
            Where:
            - model1 understands the barrier_language
            - model2 does NOT understand the barrier_language
        """
        # Convert named pair to index if a string is provided
        if isinstance(pair_index_or_name, str) and pair_index_or_name in cls.NAMED_PAIRS:
            pair_index = cls.NAMED_PAIRS[pair_index_or_name]
        else:
            try:
                pair_index = int(pair_index_or_name)
            except (ValueError, TypeError):
                # Default to the first pair if invalid input
                pair_index = 0
                
        if pair_index >= len(cls.LANGUAGE_BARRIER_PAIRS):
            pair_index = 0
            
        pair = list(cls.LANGUAGE_BARRIER_PAIRS[pair_index])
            
        # Check if we're using the proxy
        if os.environ.get("USE_ASTRA_PROXY", "false").lower() == "true":
            # Convert Anthropic model names to proxy format if needed
            
            # Convert model1 if it's an Anthropic model
            if cls.MODEL_PROVIDERS.get(pair[0], {}).get("provider") == "anthropic":
                pair[0] = f"anthropic/{pair[0]}"
                
            # Convert model2 if it's an Anthropic model
            if cls.MODEL_PROVIDERS.get(pair[1], {}).get("provider") == "anthropic":
                pair[1] = f"anthropic/{pair[1]}"
        
        print(f"- Model 1: {pair[0]} (understands {pair[2]})")
        print(f"- Model 2: {pair[1]} (does NOT understand {pair[2]})")
        print(f"- Barrier language: {pair[2]}")
        
        # Verify that model1 understands the barrier language and model2 doesn't
        model1_understands = cls.can_model_understand_language(pair[0], pair[2])
        model2_understands = cls.can_model_understand_language(pair[1], pair[2])
        
        print(f"- Verification - Model 1 understands barrier: {model1_understands}")
        print(f"- Verification - Model 2 understands barrier: {model2_understands}")
        
        if not model1_understands:
            print(f"⚠️ WARNING: Model 1 ({pair[0]}) is supposed to understand {pair[2]} but doesn't!")
        
        if model2_understands:
            print(f"⚠️ WARNING: Model 2 ({pair[1]}) is NOT supposed to understand {pair[2]} but does!")
            
        return tuple(pair)
    
    @classmethod
    def can_model_understand_language(cls, model_id: str, language: str) -> bool:
        """
        Check if a model can understand a specific language
        
        Args:
            model_id: The model identifier
            language: The language to check
            
        Returns:
            True if the model supports the language, False otherwise
        """
        # Strip the provider prefix if using proxy format
        if "/" in model_id and not model_id.startswith("TinyLlama/"):
            model_id = model_id.split("/")[1]
            
        # Get languages supported by this model
        supported_languages = cls.MODEL_CAPABILITIES.get(model_id, {}).get("languages", [])
        
        # Debug log
        print(f"[DEBUG] Checking if {model_id} understands {language}")
        print(f"- Supported languages: {supported_languages}")
        print(f"- Result: {language in supported_languages}")
        
        return language in supported_languages
    
    @classmethod
    def translate_text(cls, text: str, source_language: str, target_language: str, model_id: str) -> str:
        """
        Translate text from source_language to target_language using the specified model
        
        Args:
            text: The text to translate
            source_language: The language of the input text
            target_language: The language to translate to
            model_id: The model to use for translation
            
        Returns:
            The translated text
        """
        print(f"\n[DEBUG] Translating text:")
        print(f"- From: {source_language} to {target_language}")
        print(f"- Using model: {model_id}")
        print(f"- Original text: {text[:50]}...")
        
        # Strip the provider prefix if using proxy format
        if "/" in model_id and not model_id.startswith("TinyLlama/"):
            model_id = model_id.split("/")[1]
            
        model_info = cls.MODEL_PROVIDERS.get(model_id, {})
        provider = model_info.get("provider", "openai")
        
        # IMPORTANT FIX: Always use a reliable model for translation
        # Don't trust TinyLlama to do translations
        if provider == "huggingface":
            print(f"- Switching from {model_id} to gpt-4o-mini for more reliable translation")
            model_id = "gpt-4o-mini"
            provider = "openai"
        
        if provider == "openai" or provider == "openai_proxy":
            result = cls._translate_with_openai(text, source_language, target_language, model_id)
        elif provider == "anthropic":
            result = cls._translate_with_anthropic(text, source_language, target_language, model_id)
        else:
            print(f"Unsupported provider: {provider}")
            return text
            
        print(f"- Translated text: {result[:50]}...")
        return result
    
    @classmethod
    def _translate_with_openai(cls, text: str, source_language: str, target_language: str, model_id: str) -> str:
        """Translate text using OpenAI models"""
        client = cls.get_openai_client()
        prompt = f"Translate the following {source_language} text to {target_language}. Maintain the original structure and format. Return only the translated text without explanations or metadata.\n\n{text}"
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Translation error with {model_id}: {e}")
            return text
    
    @classmethod
    def _translate_with_anthropic(cls, text: str, source_language: str, target_language: str, model_id: str) -> str:
        """Translate text using Anthropic models"""
        client = cls.get_anthropic_client()
        prompt = f"Translate the following {source_language} text to {target_language}. Maintain the original structure and format. Return only the translated text without explanations or metadata.\n\n{text}"
        
        try:
            response = client.messages.create(
                model=model_id,
                system="You are a professional translator.",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.content[0].text
        except Exception as e:
            print(f"Translation error with {model_id}: {e}")
            return text
            
    @classmethod
    def list_available_pairs(cls):
        """List all available language barrier pairs"""
        print("\n=== Available Language Barrier Pairs ===")
        
        # First list the named pairs for easy reference
        print("\nNamed Pairs (use with --pair parameter):")
        for name, index in cls.NAMED_PAIRS.items():
            model1, model2, language = cls.LANGUAGE_BARRIER_PAIRS[index]
            print(f"  '{name}': {model1} <-> {model2} ({language})")
        
        # Then list all pairs by index
        print("\nAll Pairs (use index with --pair parameter):")
        for idx, (model1, model2, language) in enumerate(cls.LANGUAGE_BARRIER_PAIRS):
            model1_understands = cls.can_model_understand_language(model1, language)
            model2_understands = cls.can_model_understand_language(model2, language)
            
            print(f"  {idx}: {model1} <-> {model2} ({language})")
            print(f"     - {model1} understands {language}: {model1_understands}")
            print(f"     - {model2} understands {language}: {model2_understands}")