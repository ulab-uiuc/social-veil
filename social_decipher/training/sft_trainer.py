import os
from functools import partial

import torch
import wandb
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
        # 1️⃣ Initialize wandb on main process
        self.accelerator = accelerator
        self.device = accelerator.device
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        is_distributed = world_size > 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if self.accelerator.is_main_process:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))},
            )

        # 2️⃣ Load config + tokenizer
        config = AutoConfig.from_pretrained(args.model_name)
        config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.model_max_length = args.max_length
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
 

        # 3️⃣ Load model ONCE with memory-safe settings
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print(f"Using QLoRA (4bit) to load model: {args.model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=quantization_config,
                device_map={"": 0} if not is_distributed else None,
            )
            # Ensure model is ready for k-bit training (input grads, layer norms cast, etc.)
            if prepare_model_for_kbit_training is not None:
                base_model = prepare_model_for_kbit_training(base_model)
            # Ensure no cache during training
            if hasattr(base_model, "config"):
                base_model.config.use_cache = False
        else:
            # bf16; in distributed we let Accelerate handle device placement
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": 0} if not is_distributed else None,
            )

        if hasattr(base_model, "gradient_checkpointing_enable"):
            try:
                base_model.gradient_checkpointing_enable()
            except TypeError:
                base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Optional: wrap with LoRA (no second load!)
        if args.use_lora:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules.split(","),
            )
            base_model = get_peft_model(base_model, peft_config)

        model = base_model

        # 4️⃣ Prepare dataset + split
        env = Environment(loader=FileSystemLoader(os.path.dirname(args.template_path)))
        template = env.get_template(os.path.basename(args.template_path))
        full_ds = SFTDataset(args.sft_data_path, tokenizer, template, args.max_length)
        train_size = int(0.95 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, eval_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # 5️⃣ Build HF TrainingArguments
        hf_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_steps=args.evaluation_steps,
            save_steps=50,
            logging_dir="./logs",
            logging_steps=1,
            report_to=None,
            bf16=True,
            optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
            evaluation_strategy="steps",
            label_names=["labels"],
            remove_unused_columns=False,
            save_safetensors=True,
        )

        # 6️⃣ Call the Trainer constructor
        super().__init__(
            model=model,
            args=hf_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=partial(sft_collate_fn, tokenizer=tokenizer),
            tokenizer=tokenizer,
        )

    def train(self, **kwargs):
        # run the usual HF train loop
        super().train(**kwargs)
        # then save your LoRA adapter if needed
        self._save_lora()
        # optionally run final evaluation
        return self.evaluate()

    def _save_lora(self):
        if getattr(self.args, "use_lora", False):
            ckpt = os.path.join(self.args.output_dir, "best_lora_checkpoint")
            os.makedirs(ckpt, exist_ok=True)
            # HF/PEFT save
            self.model.save_pretrained(ckpt)
            print(f"LoRA checkpoint saved at {ckpt}")