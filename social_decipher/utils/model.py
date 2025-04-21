import os
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from openai import OpenAI


class ModelManager:
    """Manages different language models for language barrier experiments"""
    
    _openai_client = None
    _anthropic_client = None
    
    # Enhanced model capabilities with more precise language support
    MODEL_CAPABILITIES = {
        "gpt-4o": {
            "languages": ["English", "Chinese", "Spanish", "French", "Japanese", "Korean", 
                        "Arabic", "Russian", "Portuguese", "German", "Italian", "Hindi", "Bengali"],
            "strength": "high"
        },
        "gpt-3.5-turbo": {
            "languages": ["English", "Chinese", "Spanish", "French", "Japanese", "German", "Russian"],
            "strength": "medium"
        },
        "claude-3-opus-20240229": {
            "languages": ["English", "Chinese", "Spanish", "French", "Japanese", "German", "Korean", "Arabic", "Hindi", "Bengali"],
            "strength": "high"
        },
        "claude-3-sonnet-20240229": {
            "languages": ["English", "Chinese", "Spanish", "French", "Japanese", "German", "Korean", "Arabic", "Hindi", "Bengali"],
            "strength": "high"
        },
        "claude-3-haiku-20240307": {
            "languages": ["English", "Chinese", "Spanish", "French", "Japanese", "German", "Korean", "Arabic", "Hindi", "Bengali"],
            "strength": "medium"
        }
    }
    
    # Map actual models to their provider and API format
    MODEL_PROVIDERS = {
        "gpt-4o": {"provider": "openai", "api_format": "openai"},
        "gpt-3.5-turbo": {"provider": "openai", "api_format": "openai"},
        "claude-3-opus-20240229": {"provider": "anthropic", "api_format": "anthropic"},
        "claude-3-sonnet-20240229": {"provider": "anthropic", "api_format": "anthropic"},
        "claude-3-haiku-20240307": {"provider": "anthropic", "api_format": "anthropic"},
        # Proxy versions
        "anthropic/claude-3-opus-20240229": {"provider": "openai", "api_format": "openai_proxy"},
        "anthropic/claude-3-sonnet-20240229": {"provider": "openai", "api_format": "openai_proxy"},
        "anthropic/claude-3-haiku-20240307": {"provider": "openai", "api_format": "openai_proxy"}
    }
    
    # Define language barrier pairs with explicit incompatibility
    LANGUAGE_BARRIER_PAIRS = [
        # Format: (model_with_strong_support, model_with_weaker_support, barrier_language)

        # GPT-4o is generally stronger in lower-resource languages
        ("gpt-4o", "claude-3-haiku-20240307", "Hindi"),
        ("gpt-4o", "claude-3-haiku-20240307", "Arabic"),
        ("gpt-4o", "claude-3-haiku-20240307", "Korean"),

        # GPT-3.5-turbo has slightly stronger multilingual support than Haiku in benchmarks
        ("gpt-3.5-turbo", "claude-3-haiku-20240307", "Japanese"),

        # GPT-4o is stronger than Claude Sonnet in Bengali
        ("gpt-4o", "claude-3-sonnet-20240229", "Bengali"),

        # GPT-4o performs slightly better than Claude Sonnet in Korean
        ("gpt-4o", "claude-3-sonnet-20240229", "Korean")
    ]
    
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
    def language_barrier_pair(cls, pair_index: int = 0) -> Tuple[str, str, str]:
        """
        Return a model pair with a true language barrier
        
        Args:
            pair_index: Index of the pair to use from LANGUAGE_BARRIER_PAIRS
            
        Returns:
            Tuple of (model1, model2, barrier_language)
            Where:
            - model1 understands the barrier_language
            - model2 does NOT understand the barrier_language
        """
        if pair_index >= len(cls.LANGUAGE_BARRIER_PAIRS):
            pair_index = 0
            
        # Check if we're using the proxy
        if os.environ.get("USE_ASTRA_PROXY", "false").lower() == "true":
            # Convert Anthropic model names to proxy format if needed
            pair = list(cls.LANGUAGE_BARRIER_PAIRS[pair_index])
            
            # Convert model1 if it's an Anthropic model
            if cls.MODEL_PROVIDERS.get(pair[0], {}).get("provider") == "anthropic":
                pair[0] = f"anthropic/{pair[0]}"
                
            # Convert model2 if it's an Anthropic model
            if cls.MODEL_PROVIDERS.get(pair[1], {}).get("provider") == "anthropic":
                pair[1] = f"anthropic/{pair[1]}"
                
            return tuple(pair)
        else:
            # Using direct APIs - return the pair as defined
            return cls.LANGUAGE_BARRIER_PAIRS[pair_index]
    
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
        if "/" in model_id:
            model_id = model_id.split("/")[1]
            
        return language in cls.MODEL_CAPABILITIES.get(model_id, {}).get("languages", [])
    
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
        model_info = cls.MODEL_PROVIDERS.get(model_id, {"provider": "openai", "api_format": "openai"})
        
        if model_info["api_format"] == "openai" or model_info["api_format"] == "openai_proxy":
            # Use OpenAI API format (either direct or via proxy)
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
                
        elif model_info["api_format"] == "anthropic":
            # Use Anthropic's native API
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
        
        else:
            print(f"Unsupported API format: {model_info['api_format']}")
            return text
            
    @classmethod
    def relay_communication(cls, message: str, from_model: str, to_model: str, barrier_language: str) -> str:
        """
        Relay communication between two models through a language barrier
        
        Args:
            message: The original message
            from_model: The model sending the message
            to_model: The model receiving the message
            barrier_language: The language barrier
            
        Returns:
            The message processed through the language barrier
        """
        if from_model == to_model:
            return message  # No translation needed
            
        from_model_understands = cls.can_model_understand_language(from_model, barrier_language)
        to_model_understands = cls.can_model_understand_language(to_model, barrier_language)
    
        translator_model = from_model if from_model_understands else to_model
        
        if from_model_understands and not to_model_understands:
            return cls.translate_text(message, barrier_language, "English", translator_model)
            
        elif not from_model_understands and to_model_understands:
            return cls.translate_text(message, "English", barrier_language, translator_model)
        
        # If both models understand or neither understands, no translation needed
        return message