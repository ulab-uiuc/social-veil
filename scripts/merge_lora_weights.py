import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model and save it.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--lora_checkpoint_path", type=str, required=True, help="Path to the LoRA checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model.")
    
    args = parser.parse_args()

    print(f"Loading base model from {args.base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {args.lora_checkpoint_path}...")
    model_to_merge = PeftModel.from_pretrained(base_model, args.lora_checkpoint_path)

    print("Merging the adapter into the base model...")
    merged_model = model_to_merge.merge_and_unload()

    print(f"Saving the merged model to {args.output_path}...")
    merged_model.save_pretrained(args.output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("✅ Merging complete.")

if __name__ == "__main__":
    main()