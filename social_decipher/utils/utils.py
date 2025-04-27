import json
import os
import re

import anthropic
import requests
from mistralai import Mistral
from openai import OpenAI
from pypinyin import Style, lazy_pinyin

openai_client = None
anthropic_client = None


def _contains_chinese(text):
    chinese_pattern = re.compile(
        r"[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf]"
    )
    return bool(chinese_pattern.search(text))


def chinese_to_pinyin(text: str) -> str:
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
    global openai_client
    if openai_client is None:
        openai_client = OpenAI()
    return openai_client


def get_anthropic_client():
    global anthropic_client
    if anthropic_client is None:
        anthropic_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
    return anthropic_client


def direct_completion(agent, message, use_action=False):
    model_id = agent.profile.profile["model_id"]
    instructions = agent.instructions  # Use the agent's current instructions
    system_message = (
        instructions  # Just use the instructions directly as in agency-swarm
    )

    if "claude" in model_id.lower():
        return anthropic_completion(model_id, system_message, message, use_action)
    elif "mistral" in model_id.lower():
        response = mistral_completion(model_id, system_message, message, use_action)
        return response
    elif "tinyllama" in model_id.lower() or "huggingface" in model_id.lower():
        response = huggingface_completion(model_id, system_message, message, use_action)
        return response
    else:
        return openai_completion(model_id, system_message, message, use_action)


def openai_completion(model_id, system_message, message, use_action=False):
    """Get completion from OpenAI API"""
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
            # Format as action if needed
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


def mistral_completion(model_id, system_message, message, use_action=False):
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return error_response(
            use_action, "MISTRAL_API_KEY not set in environment variables"
        )

    try:
        client = Mistral(api_key=api_key)

        if use_action:
            user_prompt = f"Generate a response as a JSON object with 'action_type' and 'argument' fields. The message is: {message}"
        else:
            user_prompt = message

        # Format messages in OpenAI-compatible chat format
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
        error_msg = f"Mistral completion error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_response(use_action, error_msg)


def huggingface_completion(model_id, system_message, message, use_action=False):
    hf_token = os.environ.get("HF_API_TOKEN", "")
    if not hf_token:
        error_msg = "HF_API_TOKEN not set in environment variables"
        return error_response(use_action, error_msg)

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    model_id_lower = model_id.lower()
    try:
        if "tinyllama" in model_id.lower():
            formatted_prompt = (
                f"<|system|>\n{system_message}\n<|user|>\n{message}\n<|assistant|>"
            )
        elif "mistral" in model_id.lower():
            formatted_prompt = f"<s>[INST] {system_message}\n\n{message} [/INST]"
        elif "phi" in model_id.lower():
            formatted_prompt = (
                f"<|system|>\n{system_message}\n<|user|>\n{message}\n<|assistant|>"
            )
        elif "gemma" in model_id.lower():
            formatted_prompt = (
                f"<s>system\n{system_message}\n\nuser\n{message}\n\nmodel"
            )
        elif "qwen" in model_id.lower():
            formatted_prompt = (
                f"System: {system_message}\n\nUser: {message}\n\nAssistant:"
            )
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

        response = requests.post(api_url, headers=headers, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            if (
                isinstance(result, list)
                and len(result) > 0
                and "generated_text" in result[0]
            ):
                generated_text = result[0]["generated_text"]

                # Extraction logic
                if "<|assistant|>" in generated_text:
                    content = generated_text.split("<|assistant|>")[-1].strip()
                elif "[/INST]" in generated_text and ("mistral" in model_id_lower):
                    content = (
                        generated_text.split("[/INST]")[-1]
                        .strip()
                        .split("</s>")[0]
                        .strip()
                    )
                elif "model" in generated_text and "gemma" in model_id_lower:
                    content = generated_text.split("model")[-1].strip()
                elif (
                    "<|im_start|>assistant" in generated_text
                    and "qwen" in model_id_lower
                ):
                    content = generated_text.split("<|im_start|>assistant")[-1].strip()
                    if "<|im_end|>" in content:
                        content = content.split("<|im_end|>")[0].strip()
                elif "Assistant:" in generated_text:
                    content = generated_text.split("Assistant:")[-1].strip()
                else:
                    content = generated_text.replace(formatted_prompt, "").strip()
            else:
                content = str(result[0]) if result else "No output returned."

            print(
                f"[DEBUG] Received response from Hugging Face API: {content[:100]}..."
            )

            if use_action and not (content.startswith("{") and content.endswith("}")):
                content = json.dumps({"action_type": "speak", "argument": content})

            return content

        elif response.status_code == 503:
            error_msg = (
                "Hugging Face API is currently unavailable. Please try again later."
            )
            print(f"[ERROR] {error_msg}")
        else:
            error_msg = f"Hugging Face API error ({response.status_code}): {response.text[:200]}"
            print(f"[ERROR] {error_msg}")
            return error_response(use_action, error_msg)
    except Exception as e:
        error_msg = f"Hugging Face completion error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_response(use_action, error_msg)


def error_response(use_action, error_message):
    """Generate an error response with the appropriate format"""
    if use_action:
        return json.dumps(
            {
                "action_type": "speak",
                "argument": f"Error accessing model: {error_message}",
            }
        )
    return f"Error accessing model: {error_message}"


def custom_act(
    agent, message=None, initial: bool = False, use_action: bool = False
) -> str:
    assert (
        agent.agency is not None
    ), "Agent must be assigned to an agency before acting."

    if initial:
        prompt = "Now, generate your initial message to start the conversation, try to be concise"
        response = direct_completion(agent, prompt, use_action)
        print(f"**{agent.name} INITIAL RESPONSE: {response}")

        if use_action:
            response_json = json.loads(response)
            original_response = response_json

            if response_json["action_type"] == "speak":
                response_json["argument"] = (
                    agent.encryption(response_json["argument"])
                    if agent.encryption
                    else response_json["argument"]
                )
                response_json["argument"] = chinese_to_pinyin(response_json["argument"])
            encrypted_response = response_json

        else:
            original_response = response
            encrypted_response = (
                agent.encryption(response) if agent.encryption else response
            )
            encrypted_response = chinese_to_pinyin(encrypted_response)

        if agent.encryption is not None:
            print(f"**{agent.name} ENCRYPTED MESSAGE: {encrypted_response}")

        agent.log.append(
            {
                "initial": True,
                "response_raw": original_response,
                "response_encrypted": encrypted_response,
            }
        )
        return encrypted_response

    else:
        received = message

        if use_action and isinstance(message, dict) and "argument" in message:
            user_message = message["argument"]
        else:
            user_message = message

        response = direct_completion(agent, user_message, use_action)

        if use_action:
            response_json = json.loads(response)
            original_response = response_json

            if response_json["action_type"] == "speak":
                response_json["argument"] = (
                    agent.encryption(response_json["argument"])
                    if agent.encryption
                    else response_json["argument"]
                )
                response_json["argument"] = chinese_to_pinyin(response_json["argument"])
            encrypted_response = response_json

        else:
            original_response = response
            encrypted_response = (
                agent.encryption(response) if agent.encryption else response
            )
            encrypted_response = chinese_to_pinyin(encrypted_response)

        if agent.encryption is not None:
            print(f"[green]**{agent.name} ENCRYPTED RESPONSE: {encrypted_response}")

        agent.log.append(
            {
                "received_raw": received,
                "response_raw": original_response,
                "response_encrypted": encrypted_response,
            }
        )
        return encrypted_response


def predict_mcq_answer_direct(agent, transcript, mcqa, test_prompt, task_type):
    """
    Direct implementation of MCQ prediction that bypasses agency-swarm's get_completion
    Using the exact same prompt format as the original predict_mcq_answer method
    """
    assert task_type in {"goal", "reason"}, "task_type must be 'goal' or 'reason'"

    if len(transcript) > 4:
        short_transcript = transcript[-4:]
    else:
        short_transcript = transcript

    formatted_options = "\n".join([f"{k}: {v}" for k, v in mcqa["options"].items()])
    conversation_str = "\n".join(short_transcript)

    prompt = test_prompt[
        "MCQ_Goal_Prediction_Prompt"
        if task_type == "goal"
        else "MCQ_Reason_Prediction_Prompt"
    ].format(
        question=mcqa["question"],
        options=formatted_options,
        transcript=conversation_str,
    )

    # Use direct_completion instead of agency.get_completion
    response = direct_completion(agent, prompt, use_action=False).strip()

    selected = None
    confidence = 0.0
    try:
        for line in response.split("\n"):
            if line.lower().startswith("selected:"):
                selected = line.split(":")[1].strip().upper()
            elif line.lower().startswith("confidence:"):
                confidence = float(line.split(":")[1].strip())
    except Exception as e:
        print(f"Error parsing MCQ response from agent {agent.name}: {e}")

    return {
        "selected": selected if selected in mcqa["options"] else "Invalid",
        "confidence": max(0.0, min(confidence, 1.0)),  # clamp between 0-1
        "correct": selected == mcqa["correct_answer"],
    }
