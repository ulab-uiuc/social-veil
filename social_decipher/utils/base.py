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
    
    # Check for local model first
    if model_id.startswith("local:"):
        return local_model_completion(model_id, system_message, message, use_action)
    # Check for Qwen models and use local inference
    elif "qwen" in model_id.lower():
        return local_qwen_completion(model_id, system_message, message, use_action)
    # Call appropriate API based on model_id
    elif model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        return local_tinyllama_inference(system_message, message, use_action)
    elif "mistral" in model_id.lower() or "ministral" in model_id.lower():
        return mistral_completion(model_id, system_message, message, use_action)
    elif (
        "tinyllama" in model_id.lower()
        or "huggingface" in model_id.lower()
        or "phi" in model_id.lower()
    ):
        return huggingface_completion(model_id, system_message, message, use_action)
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

def local_tinyllama_inference(system_message, message, use_action=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import os
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # Optional: Check if model is already downloaded
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")
    if not os.path.exists(model_dir):
        print(f"[INFO] Model {model_id} not found locally. Downloading and deploying...")
    else:
        print(f"[INFO] Model {model_id} found locally. Using local deployment.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        prompt = f"{system_message}\n{message}" if system_message else message
        outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.7)
        content = outputs[0]["generated_text"]
        if use_action and not (content.startswith("{") and content.endswith("}")):
            content = json.dumps({"action_type": "speak", "argument": content})
        return content
    except Exception as e:
        print(f"[ERROR] Local TinyLlama inference failed: {e}")
        return error_response(use_action, f"Local TinyLlama inference failed: {str(e)}")


def huggingface_completion(model_id, system_message, message, use_action=False):
    hf_token = _config.get("HF_API_TOKEN")
    if not hf_token:
        error_msg = "HF_API_TOKEN not set in environment variables"
        print(f"[ERROR] {error_msg}")
        return error_response(use_action, error_msg)
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=model_id, token=hf_token)
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": message})
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        content = completion.choices[0].message.content
        if use_action and not (content.startswith("{") and content.endswith("}")):
            content = json.dumps({"action_type": "speak", "argument": content})
        return content
    except Exception as e:
        print(f"[ERROR] HuggingFace InferenceClient error: {e}")
        return error_response(use_action, str(e))

def local_qwen_completion(model_id, system_message, message, use_action=False):
    """Generate completion using local Qwen model."""
    print(f"🔧 Local Qwen completion for model: {model_id}")
    print(f"   System message: {system_message[:100]}{'...' if len(system_message) > 100 else ''}")
    print(f"   User message: {message[:100]}{'...' if len(message) > 100 else ''}")
    
    try:
        from .local_model_manager import local_model_registry
        
        # Try to get Qwen model from registry first
        qwen_model_names = ["qwen2.5-7b", "qwen", "qwen2.5", "qwen-7b"]
        model_manager = None
        
        for model_name in qwen_model_names:
            try:
                model_manager = local_model_registry.get_model(model_name)
                print(f"✅ Using registered Qwen model: {model_name}")
                break
            except KeyError:
                print(f"   Model '{model_name}' not found in registry")
                continue
        
        if model_manager is None:
            # If no registered Qwen model, create a default one
            print(f"⚠️ No registered Qwen model found. Creating default Qwen model for: {model_id}")
            # Get the absolute path to the template
            import os
            template_path = os.path.join(os.path.dirname(__file__), "../../configs/qwen2.5-7b.jinja")
            
            # Check what model is actually running on vLLM server
            try:
                import requests
                response = requests.get("http://localhost:8000/v1/models", timeout=5)
                if response.status_code == 200:
                    models = response.json()
                    if models.get("data") and len(models["data"]) > 0:
                        actual_model_name = models["data"][0]["id"]
                        print(f"🔍 Found vLLM server running model: {actual_model_name}")
                        model_path = actual_model_name
                    else:
                        print("⚠️ No models found on vLLM server, using default")
                        model_path = "Qwen/Qwen2.5-7B-Instruct"
                else:
                    print(f"⚠️ Could not check vLLM server models: {response.status_code}, using default")
                    model_path = "Qwen/Qwen2.5-7B-Instruct"
            except Exception as e:
                print(f"⚠️ Error checking vLLM server models: {e}, using default")
                model_path = "Qwen/Qwen2.5-7B-Instruct"
            
            model_manager = local_model_registry.register_model(
                name="qwen2.5-7b",
                model_path=model_path,
                template_path=template_path,
                use_vllm=True,
                vllm_port=8000,
            )
            print(f"✅ Default Qwen model registered")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]
        
        # Generate response
        print(f"🚀 Generating response via local Qwen model...")
        response = model_manager.generate(messages, max_new_tokens=512)
        print(f"✅ Local Qwen response: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Format response for action if needed
        if use_action and not (response.startswith("{") and response.endswith("}")):
            response = json.dumps({"action_type": "speak", "argument": response})
        
        return response
        
    except Exception as e:
        print(f"❌ [ERROR] Local Qwen completion failed: {e}")
        # Fallback to HuggingFace API if local model fails
        print(f"🔄 Falling back to HuggingFace API for {model_id}")
        return huggingface_completion(model_id, system_message, message, use_action)


def local_model_completion(model_id, system_message, message, use_action=False):
    """Generate completion using local model via LocalModelManager."""
    try:
        from .local_model_manager import local_model_registry
        
        # Extract model name from local:model_name format
        model_name = model_id.replace("local:", "")
        
        # Get the model from registry
        model_manager = local_model_registry.get_model(model_name)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]
        
        # Generate response
        response = model_manager.generate(messages, max_new_tokens=512)
        
        # Format response for action if needed
        if use_action and not (response.startswith("{") and response.endswith("}")):
            response = json.dumps({"action_type": "speak", "argument": response})
        
        return response
        
    except Exception as e:
        print(f"[ERROR] Local model completion failed: {e}")
        return error_response(use_action, f"Local model completion failed: {str(e)}")


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