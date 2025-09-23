import os
from functools import partial

import torch
from jinja2 import Environment, FileSystemLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
try:
    # Available in PEFT for QLoRA preparation
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None

from social_decipher.training.data import SFTDataset

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def sft_collate_fn(batch, tokenizer):
    input_ids = pad_sequence(
        [x["input_ids"] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(
        [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
    )
    labels = pad_sequence(
        [x["labels"] for x in batch], batch_first=True, padding_value=-100
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SotopiaSFTTrainer(Trainer):
    def __init__(self, args, accelerator):
        self.accelerator = accelerator
        self.device = accelerator.device
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        is_distributed = world_size > 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        per_rank_device_map = {"": local_rank} if is_distributed else {"": 0}

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.model_max_length = args.max_length
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print(f"Using QLoRA (4bit) to load model: {args.model_name_or_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=quantization_config,
                device_map=per_rank_device_map,
            )
            if prepare_model_for_kbit_training is not None:
                base_model = prepare_model_for_kbit_training(base_model)
            if hasattr(base_model, "config"):
                base_model.config.use_cache = False
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=per_rank_device_map,
            )

        if hasattr(base_model, "gradient_checkpointing_enable"):
            try:
                base_model.gradient_checkpointing_enable()
            except TypeError:
                base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if args.use_lora:
            from peft import LoraConfig, get_peft_model
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules.split(","),
            )
            base_model = get_peft_model(base_model, peft_config)

        if args.lora_checkpoint_path:
            from peft import PeftModel
            print(f"Loading LoRA checkpoint from {args.lora_checkpoint_path}")
            model = PeftModel.from_pretrained(base_model, args.lora_checkpoint_path, is_trainable=True)
        else:
            model = base_model

        # Based on Sotopia-π, uses QLoRA for fine-tuning
        model_kwargs = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            "torch_dtype": torch.bfloat16,
        }

        print(f"Using QLoRA (4bit) to load model: {args.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map={"": accelerator.process_index},
            **model_kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup Jinja environment for templates
        # FIX: Use an absolute path to the configs directory to prevent pathing issues.
        # The trainer script is in social_decipher/training/, so we need to go up two directories.
        configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
        env = Environment(loader=FileSystemLoader(configs_dir))
        template = env.get_template(os.path.basename(args.template_path))
        self.template = template
        
        # Load and process dataset
        self.train_dataset = self._load_and_process_dataset(args.sft_data_path)

        hf_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",       # Use new argument name
            save_strategy="epoch",
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=1,
            report_to="none",
            bf16=True,
            optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_grad_norm=1.0,  # Gradient clipping for stability
            label_names=["labels"],
            remove_unused_columns=False,
            save_safetensors=True,
        )

        super().__init__(
            model=model,
            args=hf_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=partial(sft_collate_fn, tokenizer=tokenizer),
            tokenizer=tokenizer,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation to ensure logits are in float32 for stability, preventing NaN loss.
        """
        outputs = model(**inputs)
        logits = outputs.get("logits").to(torch.float32)
        labels = inputs.get("labels")
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def train(self, **kwargs):
        # Run the standard training loop. The trainer will save checkpoints at the end of each epoch.
        train_output = super().train(**kwargs)

        # After training, explicitly save the final model to a consistent "best-checkpoint" directory.
        # This ensures the main script can always find it, regardless of the final epoch number.
        if self.accelerator.is_main_process:
            final_checkpoint_dir = os.path.join(self.args.output_dir, "best-checkpoint")
            os.makedirs(final_checkpoint_dir, exist_ok=True)
            print(f"Saving final model to {final_checkpoint_dir}")
            self.model.save_pretrained(final_checkpoint_dir)
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                self.tokenizer.save_pretrained(final_checkpoint_dir)
        
        # Barrier to ensure all processes wait until the save is complete.
        self.accelerator.wait_for_everyone()
        
        return train_output