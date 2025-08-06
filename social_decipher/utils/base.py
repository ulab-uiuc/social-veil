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
from pypinyin import Style, lazy_pinyin
from rich import print
from huggingface_hub import InferenceClient
from .local_model_manager import LocalModelManager

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs/config.yaml"))
with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

# Global clients for API access
openai_client = None
anthropic_client = None

def _contains_chinese(text):
    """Check if text contains Chinese characters"""
    chinese_pattern = re.compile(
        r"[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf]"
    )
    return bool(chinese_pattern.search(text))


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
    elif "mistral" in model_lower:
        # Mistral models typically use Llama-style templates
        return "configs/llama3.1-8b.jinja"
    else:
        # Default fallback - check if Llama template exists, otherwise use Qwen
        import os
        config_dir = os.path.dirname(CONFIG_PATH)
        project_root = os.path.dirname(config_dir)
        llama_template = os.path.join(project_root, "configs/llama3.1-8b.jinja")
        
        if os.path.exists(llama_template):
            return "configs/llama3.1-8b.jinja"
        else:
            return "configs/qwen2.5-7b.jinja"


def chinese_to_pinyin(text: Any) -> Any:
    if isinstance(text, dict) and "argument" in text:
        argument = text["argument"]
        if _contains_chinese(argument):
            text["argument"] = _convert_text_to_pinyin(argument)
        return text
    else:
        if _contains_chinese(text):
            return _convert_text_to_pinyin(text)
        return text


def _convert_text_to_pinyin(text):
    try:
        if not isinstance(text, str):
            text = str(text)
        pinyin_list = lazy_pinyin(text, style=Style.NORMAL)
        return " ".join(pinyin_list)
    except Exception as e:
        print(f"[ERROR] Pinyin conversion failed: {e}")
        return text


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
    use_action=False
):
    model_id = agent.profile.model_id
    print(model_id)
 
    system_message = agent.instructions
    if hasattr(agent, 'encryption') and agent.encryption is not None:
        system_message = "IMPORTANT: Always respond in English only. Your response will be translated later if needed.\n\n" + system_message
    
    # Check if it's a local model (contains path or specific local model names)
    if "/" in model_id or "qwen" in model_id.lower() or "llama" in model_id.lower():
        return local_model_completion(model_id, system_message, message, use_action)
    elif "mistral" in model_id.lower() or "ministral" in model_id.lower():
        return mistral_completion(model_id, system_message, message, use_action)
    elif "claude" in model_id.lower():
        return anthropic_completion(model_id, system_message, message, use_action)
    else:
        return openai_completion(model_id, system_message, message, use_action)
    


def openai_completion(model_id, system_message, message, use_action=False):
    client = get_openai_client()
    try:
        if use_action:
            prompt = f"Generate a response as a JSON object with 'action_type' and 'argument' fields. The message is: {message}"
        else:
            prompt = message if isinstance(message, str) else json.dumps(message)

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        if use_action and not (content.startswith("{") and content.endswith("}")):
            content = json.dumps({"action_type": "speak", "argument": content})

        return content
    except Exception as e:
        print(f"[ERROR] OpenAI completion error: {e}")
        # Return a basic response to prevent the conversation from stopping
        if use_action:
            return json.dumps(
                {
                    "action_type": "speak",
                    "argument": "I'm having trouble responding right now.",
                }
            )
        return "I'm having trouble responding right now."

def anthropic_completion(model_id, system_message, message, use_action=False):
    """
    Get completion from Anthropic API.
    
    Args:
        model_id: Anthropic model ID
        system_message: System message
        message: User message
        use_action: Whether to use action-based communication
        
    Returns:
        Completion from Anthropic API
    """
    client = get_anthropic_client()

    try:
        if use_action:
            prompt = f"Generate a response as a JSON object with 'action_type' and 'argument' fields. The message is: {message}"
        else:
            prompt = message if isinstance(message, str) else json.dumps(message)

        response = client.messages.create(
            model=model_id,
            system=system_message,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.content[0].text

        if use_action and not (content.startswith("{") and content.endswith("}")):
            content = json.dumps({"action_type": "speak", "argument": content})

        return content
    except Exception as e:
        print(f"[ERROR] Anthropic completion error: {e}")
        if use_action:
            return json.dumps(
                {
                    "action_type": "speak",
                    "argument": "I'm having trouble responding right now.",
                }
            )
        return "I'm having trouble responding right now."

def mistral_completion(model_id, system_message, message, use_action=False, max_retries=30):
    """Get completion from Mistral API with persistent retry logic for all error types."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set in environment variables")
    
    retry_count = 0
    max_wait_time = 120  # Maximum wait time in seconds
    
    while retry_count < max_retries:  # Add a high but finite retry limit for safety
        try:
            client = Mistral(api_key=api_key)
            
            if use_action:
                user_prompt = f"Generate a response as a JSON object with 'action_type' and 'argument' fields. The message is: {message}"
            else:
                user_prompt = message
                
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

            if use_action and not (content.startswith("{") and content.endswith("}")):
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
    time.sleep(300)  # 5 minute wait
    
    # One final attempt
    try:
        client = Mistral(api_key=api_key)
        
        if use_action:
            user_prompt = f"Generate a response as a JSON object with 'action_type' and 'argument' fields. The message is: {message}"
        else:
            user_prompt = message

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

        if use_action and not (content.startswith("{") and content.endswith("}")):
            content = json.dumps({"action_type": "speak", "argument": content})

        return content
    except Exception as final_e:
        # If even the final attempt fails, throw an exception to avoid silent failure
        raise RuntimeError(f"Mistral API repeatedly failed after exhausting all retries: {str(final_e)}")      


def local_model_completion(model_id, system_message, message, use_action=False):
    """Generate completion using local model via vLLM server (supports Qwen, Llama, etc.)."""
    print(f"🔧 Local model completion for: {model_id}")
    print(f"   User message: {message}")
    
    try:        
        # Create model manager directly
        vllm_port = int(os.environ.get("VLLM_PORT", 8000))  # Default to 8000 if not set
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
            response = model_manager.generate(messages, max_new_tokens=512)
        except Exception as e:
            print(f"   ❌ Generate call failed: {e}")
            print(f"   ❌ Generate error type: {type(e)}")

        # Format response for action if needed
        if use_action and not (response.startswith("{") and response.endswith("}")):
            response = json.dumps({"action_type": "speak", "argument": response})
        
        return response
        
    except Exception as e:
        print(f"❌ [ERROR] Local model completion failed: {e}")
        print(f"   Make sure vLLM server is running with: ./scripts/start_vllm_server.sh")
        raise e



def error_response(use_action, error_message):
    """Generate an error response with the appropriate format"""
    if use_action:
        return json.dumps(
            {
                "action_type": "speak",
                "argument": f"Error: {error_message}",
            }
        )
    return f"Error: {error_message}"