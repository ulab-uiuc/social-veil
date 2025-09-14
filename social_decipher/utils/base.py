import json
import os
import re
from typing import Any, Dict, Optional, Union

import anthropic
import requests
import time
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from mistralai import Mistral
from openai import OpenAI
from rich import print
from .local_model_manager import LocalModelManager
from .error_handler import api_calling_error_exponential_backoff

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/config.yaml"))
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

# Global token cap (response) with sensible default
_RESP_MAX_TOKENS = int((_config.get("models", {}) or {}).get("response_max_tokens", 160))

# Global clients for API access
openai_client = None
anthropic_client = None


def _get_default_template_for_model(model_id):
    """Auto-detect the appropriate template based on model name"""
    model_lower = model_id.lower()
    
    # Model-specific template mapping
    if "llama" in model_lower:
        if "3.1" in model_lower or "3-1" in model_lower:
            return "configs/llama3.1-8b.jinja"
        else:
            # For other Llama versions, use 3.1 template as it's backward compatible
            return "configs/llama3.1-8b.jinja"
    elif "qwen" in model_lower:
        if "2.5" in model_lower or "2-5" in model_lower:
            return "configs/qwen2.5-7b.jinja"
        else:
            # Default to Qwen 2.5 template for other Qwen versions
            return "configs/qwen2.5-7b.jinja"
    elif "mistral" in model_lower or "ministral" in model_lower:
        return "configs/mistral-8b.jinja"
    else:

        config_dir = os.path.dirname(CONFIG_PATH)
        project_root = os.path.dirname(config_dir)
        llama_template = os.path.join(project_root, "configs/llama3.1-8b.jinja")
        
        if os.path.exists(llama_template):
            return "configs/llama3.1-8b.jinja"
        else:
            return "configs/qwen2.5-7b.jinja"

def get_openai_client():
    """Get or create OpenAI client"""
    global openai_client
    if openai_client is None:
        openai_client = OpenAI()
    return openai_client


def get_anthropic_client():
    """Get or create Anthropic client"""
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    return anthropic_client


def direct_completion(
    agent=None, 
    message=None,
):
    model_id = agent.profile.model_id
    print(model_id)
 
    system_message = agent.instructions

    # Check if it's a local model (contains path or specific local model names)
    if "/" in model_id or "qwen" in model_id.lower() or "llama" in model_id.lower():
        return local_model_completion(model_id, system_message, message)
    elif "mistral" in model_id.lower() or "ministral" in model_id.lower():
        return mistral_completion(model_id, system_message, message)
    elif "claude" in model_id.lower():
        return anthropic_completion(model_id, system_message, message)
    else:
        return openai_completion(model_id, system_message, message)
    
@api_calling_error_exponential_backoff()
def openai_completion(model_id: str, system_message: str, message: str) -> Optional[str]:

    client = get_openai_client()
    try:    
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=_RESP_MAX_TOKENS,
        )
        content = response.choices[0].message.content
        if not (content.startswith("{") and content.endswith("}")):
            content = json.dumps({"action_type": "speak", "argument": content})

        return content
    except Exception as e:
        # If this is a rate limit error, re-raise so the decorator can wait and retry
        try:
            import openai as _openai_mod  # type: ignore
        except Exception:
            _openai_mod = None
        if _openai_mod and isinstance(e, _openai_mod.RateLimitError):
            raise
        print(f"[ERROR] OpenAI completion error: {e}")
        # Return a basic response to prevent the conversation from stopping
        return json.dumps(
            {
                "action_type": "speak",
                "argument": "I'm having trouble responding right now.",
            }
        )

@api_calling_error_exponential_backoff()
def anthropic_completion(model_id: str, system_message: str, message: str) -> Optional[str]:
    """
    Get completion from Anthropic API.
    
    Args:
        model_id: Anthropic model ID
        system_message: System message
        message: User message
        
    Returns:
        Completion from Anthropic API
    """
    client = get_anthropic_client()

    try:
        response = client.messages.create(
            model=model_id,
            system=system_message,
            messages=[{"role": "user", "content": message}],
            temperature=0.3,
            max_tokens=_RESP_MAX_TOKENS,
        )
        content = response.content[0].text

        if not (content.startswith("{") and content.endswith("}")):
            content = json.dumps({"action_type": "speak", "argument": content})

        return content
    except Exception as e:
        print(f"[ERROR] Anthropic completion error: {e}")
        return json.dumps(
            {
                "action_type": "speak",
                "argument": "I'm having trouble responding right now.",
            }
        )

def mistral_completion(model_id, system_message, message, max_retries=30):
    """Get completion from Mistral API with persistent retry logic for all error types."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set in environment variables")
    
    retry_count = 0
    max_wait_time = 120  # Maximum wait time in seconds
    
    while retry_count < max_retries:  # Add a high but finite retry limit for safety
        try:
            client = Mistral(api_key=api_key)
            
            user_prompt = f"Generate a response as a JSON object with 'action_type' and 'argument' fields. The message is: {message}"
                
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ]

            response = client.chat.complete(
                model=model_id,
                messages=messages,
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()

            if not (content.startswith("{") and content.endswith("}")):
                content = json.dumps({"action_type": "speak", "argument": content})

            return content

        except Exception as e:
            error_msg = str(e)
            retry_count += 1
            
            # Different backoff strategies based on error type
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                # Exponential backoff with jitter for rate limits
                wait_time = min(max_wait_time, (2 ** retry_count) + random.uniform(0, 1))
                print(f"[WARNING] Mistral rate limit exceeded. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}...")
            else:
                # Linear backoff for other errors (connection issues, server errors, etc.)
                wait_time = min(max_wait_time, 5 * retry_count + random.uniform(0, 2))
                print(f"[WARNING] Mistral error: {error_msg}. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}...")
            
            time.sleep(wait_time)
    
    # If we've exhausted all retries (extremely unlikely with max_retries=30)
    # We'll try one last time with a significantly longer wait
    print(f"[WARNING] Still encountering errors after {max_retries} retries. Waiting 5 minutes for final attempt...")
    time.sleep(300)  
    
    try:
        client = Mistral(api_key=api_key)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]

        response = client.chat.complete(
            model=model_id,
            messages=messages,
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()

        if not (content.startswith("{") and content.endswith("}")):
            content = json.dumps({"action_type": "speak", "argument": content})

        return content
    
    except Exception as final_e:
        # If even the final attempt fails, throw an exception to avoid silent failure
        raise RuntimeError(f"Mistral API repeatedly failed after exhausting all retries: {str(final_e)}")      


def local_model_completion(model_id, system_message, message):
    """Generate completion using local model via vLLM server (supports Qwen, Llama, etc.)."""
    print(f"🔧 Local model completion for: {model_id}")
    print(f"   User message: {message}")
    
    try:        
        # Create model manager directly
        # Read the port directly from the loaded config to ensure consistency.
        vllm_port = int((_config.get("models", {}) or {}).get("vllm_port", 8000))
        print(f"   Using vLLM server at port {vllm_port}")

        # Get the template path from config or auto-detect based on model
        template_path = _config.get("models", {}).get("chat_template")
        
        # Auto-detect template if not specified in config
        if not template_path:
            template_path = _get_default_template_for_model(model_id)
            print(f"   Auto-detected template for {model_id}: {template_path}")
        else:
            print(f"   Using configured template: {template_path}")
        
        # Convert relative path to absolute path if needed
        if not os.path.isabs(template_path):
            config_dir = os.path.dirname(CONFIG_PATH)
            project_root = os.path.dirname(config_dir)
            template_path = os.path.join(project_root, template_path)
        
        modal_name = _config.get("models", {}).get("served_model_name")
        
        if not modal_name:
            modal_name = model_id.split("/")[-1].lower()  # Extract model name from ID
     
        try:
            model_manager = LocalModelManager(
                model_path=model_id,  # Match server's GLOBAL_MODEL_B
                model_name=modal_name,  # Match server's served-model-name
                template_path=template_path,
                use_vllm=True,
                vllm_port=vllm_port
            )
        except Exception as e:
            print(f"   ❌ LocalModelManager creation failed: {e}")
            raise e
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]

        # Generate response
        print(f"🚀 Generating response via local model...")
        try:
            response = model_manager.generate(messages, max_new_tokens=_RESP_MAX_TOKENS)
        except Exception as e:
            print(f"   ❌ Generate call failed: {e}")
            print(f"   ❌ Generate error type: {type(e)}")
            return json.dumps({"action_type": "speak", "argument": "I'm having trouble responding right now."})

        # Sanitize noisy multi-block output for action mode
        sanitized = response
        if isinstance(sanitized, dict):
            return json.dumps(sanitized, ensure_ascii=False)
        # If it's still a string and not a clean JSON, wrap it
        s = sanitized.strip()
        if not (s.startswith("{") and s.endswith("}")):
            return json.dumps({"action_type": "speak", "argument": s}, ensure_ascii=False)
        return s
        
    except Exception as e:
        print(f"❌ [ERROR] Local model completion failed: {e}")
        print(f"   Make sure vLLM server is running with: ./scripts/start_vllm_server.sh")
        raise e


def error_response(error_message):
    """Generate an error response with the appropriate format"""
    return json.dumps(
        {
            "action_type": "speak",
            "argument": f"Error: {error_message}",
        }
    )