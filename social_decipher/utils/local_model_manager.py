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
        model_path: str,
        model_name: str = "local-model",
        template_path: Optional[str] = None,
        use_vllm: bool = True,
        vllm_port: int = 8000,
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
    
    def generate_via_vllm(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        """Generate response using vLLM API."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_new_tokens,
            "stop": ["<|im_end|>", "\n\n"] if self.tokenizer else None,
        }
        
        print(f"🔧 vLLM API call to {self.vllm_api_url}/chat/completions")
        print(f"   Model: {self.model_name}")
        print(f"   Messages: {len(messages)} messages")
        
        try:
            response = requests.post(
                f"{self.vllm_api_url}/chat/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"].strip()
                print(f"✅ vLLM response: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
                return generated_text
            else:
                raise ValueError("No choices in vLLM response")
                
        except Exception as e:
            print(f"❌ vLLM API error: {e}")
            print(f"   Response status: {response.status_code if 'response' in locals() else 'N/A'}")
            print(f"   Response content: {response.text if 'response' in locals() else 'N/A'}")
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


class LocalModelRegistry:
    """Registry for managing multiple local models."""
    
    def __init__(self):
        self.models: Dict[str, LocalModelManager] = {}
    
    def register_model(
        self,
        name: str,
        model_path: str,
        template_path: Optional[str] = None,
        use_vllm: bool = True,
        vllm_port: Optional[int] = None,
        **kwargs
    ) -> LocalModelManager:
        """Register a new local model."""
        if vllm_port is None:
            vllm_port = 8000 + len(self.models)
        
        model_manager = LocalModelManager(
            model_path=model_path,
            model_name=name,
            template_path=template_path,
            use_vllm=use_vllm,
            vllm_port=vllm_port,
            **kwargs
        )
        
        self.models[name] = model_manager
        return model_manager
    
    def get_model(self, name: str) -> LocalModelManager:
        """Get a registered model by name."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self.models[name]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())


# Global registry instance
local_model_registry = LocalModelRegistry() 