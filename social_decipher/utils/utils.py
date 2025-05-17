import json
import os
import re
from typing import Any, Dict, Optional, Union

import anthropic
import requests
import time
import random
from mistralai import Mistral
from openai import OpenAI
from pypinyin import Style, lazy_pinyin
from rich import print

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
    model_id = agent.profile.profile.get("model_id")
    system_message = agent.instructions

    if model_id is None:
        raise ValueError("model_id must be provided either directly or via agent")
    
    if system_message is None:
        raise ValueError("system_message must be provided either directly or via agent")
    
    # Call appropriate API based on model_id
    if "claude" in model_id.lower():
        return anthropic_completion(model_id, system_message, message, use_action)
    elif "mistral" in model_id.lower():
        return mistral_completion(model_id, system_message, message, use_action)
    elif "tinyllama" in model_id.lower() or "huggingface" in model_id.lower():
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

def huggingface_completion(model_id, system_message, message, use_action=False):
    """
    Get completion from Hugging Face API with improved error handling and response parsing.
    """
    hf_token = os.environ.get("HF_API_TOKEN", "")
    if not hf_token:
        error_msg = "HF_API_TOKEN not set in environment variables"
        print(f"[ERROR] {error_msg}")
        return error_response(use_action, error_msg)

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    model_id_lower = model_id.lower()
    
    # Format prompt based on model type
    if "tinyllama" in model_id_lower:
        formatted_prompt = f"<|system|>\n{system_message}\n<|user|>\n{message}\n<|assistant|>"
    elif "mistral" in model_id_lower:
        formatted_prompt = f"<s>[INST] {system_message}\n\n{message} [/INST]"
    elif "phi" in model_id_lower:
        formatted_prompt = f"<|system|>\n{system_message}\n<|user|>\n{message}\n<|assistant|>"
    elif "gemma" in model_id_lower:
        formatted_prompt = f"<s>system\n{system_message}\n\nuser\n{message}\n\nmodel"
    elif "qwen" in model_id_lower:
        formatted_prompt = f"System: {system_message}\n\nUser: {message}\n\nAssistant:"
    else:
        formatted_prompt = f"{system_message}\n\nUser: {message}\n\nAssistant:"

    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.3,
            "top_p": 0.95,
            "do_sample": True,
        },
    }

    # Implement retry logic for HuggingFace as well
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        generated_text = result[0]["generated_text"]
                    else:
                        generated_text = str(result[0])
                else:
                    generated_text = str(result)
                
                # Extract content based on model type
                if "<|assistant|>" in generated_text:
                    content = generated_text.split("<|assistant|>")[-1].strip()
                elif "[/INST]" in generated_text and ("mistral" in model_id_lower):
                    content = generated_text.split("[/INST]")[-1].strip().split("</s>")[0].strip()
                elif "model" in generated_text and "gemma" in model_id_lower:
                    content = generated_text.split("model")[-1].strip()
                elif "<|im_start|>assistant" in generated_text and "qwen" in model_id_lower:
                    content = generated_text.split("<|im_start|>assistant")[-1].strip()
                    if "<|im_end|>" in content:
                        content = content.split("<|im_end|>")[0].strip()
                elif "Assistant:" in generated_text:
                    content = generated_text.split("Assistant:")[-1].strip()
                else:
                    # Fallback extraction - remove the prompt if possible
                    if formatted_prompt in generated_text:
                        content = generated_text.replace(formatted_prompt, "").strip()
                    else:
                        content = generated_text.strip()
                
                print(f"[DEBUG] Received response from Hugging Face API: {content[:100]}...")
                
                if use_action and not (content.startswith("{") and content.endswith("}")):
                    content = json.dumps({"action_type": "speak", "argument": content})
                
                return content
            else:
                # Handle rate limiting and other HTTP errors
                if response.status_code == 429:  # Too Many Requests
                    retry_count += 1
                    wait_time = min(60, (2 ** retry_count) + random.uniform(0, 1))
                    print(f"[WARNING] HuggingFace rate limit exceeded. Waiting {wait_time:.2f} seconds before retry {retry_count}...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Hugging Face API error ({response.status_code}): {response.text[:200]}"
                    print(f"[ERROR] {error_msg}")
                    return error_response(use_action, error_msg)
                    
        except Exception as e:
            error_msg = f"Hugging Face completion error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            
            # Retry on connection errors
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                retry_count += 1
                wait_time = min(60, (2 ** retry_count) + random.uniform(0, 1))
                print(f"[WARNING] HuggingFace connection error. Waiting {wait_time:.2f} seconds before retry {retry_count}...")
                time.sleep(wait_time)
                continue
            
            return error_response(use_action, error_msg)

    return error_response(use_action, "HuggingFace API failed after maximum retries")


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