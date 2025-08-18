import json
import os
import requests
import torch
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from jinja2 import Environment, FileSystemLoader
import time


class LocalModelManager:
    """Manager for local model inference with vLLM API and direct model loading support."""
    
    def __init__(
        self,
        model_path: str = "qwen2.5-7b-instruct",
        model_name: str = "qwen2.5-7b-instruct", 
        template_path: Optional[str] = None,
        use_vllm: bool = True,
        vllm_port: int = 8010,
        vllm_api_url: Optional[str] = None,
        use_quantization: bool = True,
        device_map: str = "auto",
        max_length: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.vllm_port = vllm_port
        self.vllm_api_url = vllm_api_url or f"http://localhost:{vllm_port}/v1"
        self.use_quantization = use_quantization
        self.device_map = device_map
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize template if provided
        self.template = None
        if template_path:
            self._setup_template(template_path)
        
        # Initialize direct model if not using vLLM
        self.model = None
        self.tokenizer = None
        if not use_vllm:
            self._setup_direct_model()
    
    def _setup_template(self, template_path: str):
        """Setup Jinja2 template for chat formatting."""
        template_dir = os.path.dirname(template_path)
        template_file = os.path.basename(template_path)
        
        if not template_dir:
            template_dir = "."
        
        env = Environment(loader=FileSystemLoader(template_dir))
        env.filters['tojson'] = lambda obj: json.dumps(obj)
        self.template = env.get_template(template_file)
       
    
    def _setup_direct_model(self):
        """Setup direct model loading for local inference."""
        print(f"Loading local model: {self.model_path}")
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization if enabled
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            quantization_config=quantization_config,
        )
        self.model.eval()
        print(f"Local model loaded successfully on device: {self.model.device}")
    
    def format_messages(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Format messages using template or default format."""
        if self.template:
            return self.template.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            # Default chat format
            formatted = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            if add_generation_prompt:
                formatted += "<|im_start|>assistant\n"
            
            return formatted
    
    def generate_via_vllm(self, messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
        """Simplified version for debugging timeout issues."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,  # Fixed lower temperature
            "max_tokens": min(max_new_tokens, 256),  # Cap at 256 tokens
            "stream": False,
            # Add stop sequences to avoid multi-turn/role echoing artifacts
            "stop": ["\nassistant\n", "\nassistant", "assistant\n", "assistant", "angstrom", "obutton"],
        }
        
        try:
            response = requests.post(
                f"{self.vllm_api_url}/chat/completions",
                json=payload,
                timeout=90,  # Single 90s timeout
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            # Debug: Print raw response to understand the issue
            print(f"🔍 Raw vLLM response status: {response.status_code}")
            print(f"🔍 Raw vLLM response text: {response.text[:500]}...")  # First 500 chars
            
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.Timeout:
            return "Error: Simple request timed out"
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"❌ Response content: {response.text[:1000]}")
            return f"Error: Invalid JSON response from vLLM server"
        except Exception as e:
            print(f"❌ Request error: {e}")
            if 'response' in locals():
                print(f"❌ Response status: {response.status_code}")
                print(f"❌ Response content: {response.text[:1000]}")
            return f"Error: {str(e)}"

    
    def generate_direct(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        """Generate response using direct model inference."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Direct model not initialized")
        
        prompt = self.format_messages(messages, add_generation_prompt=True)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        """Generate response using the configured method."""
        if self.use_vllm:
            return self.generate_via_vllm(messages, max_new_tokens)
        else:
            return self.generate_direct(messages, max_new_tokens)
    
    def chat_completion(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> Dict[str, Any]:
        """Generate chat completion in OpenAI-compatible format."""
        generated_text = self.generate(messages, max_new_tokens)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Could be calculated if needed
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
