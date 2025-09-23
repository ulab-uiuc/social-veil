import argparse
import os
import torch
from accelerate import Accelerator
import sys
from pathlib import Path

# Add project root to path to allow importing from social_decipher
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from social_decipher.training.sft_trainer import SotopiaSFTTrainer
from transformers import HfArgumentParser, TrainingArguments


def get_args():
    parser = HfArgumentParser(TrainingArguments)

    # Required
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the base model (HF hub id or local dir)",
    )
    parser.add_argument(
        "--sft_data_path",
        type=str,
        required=True,
        help="Path to the supervised fine-tuning (SFT) dataset (json/jsonl)",
    )

    # Optional
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to the LoRA checkpoint (if resuming LoRA/QLoRA training)",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="../configs/qwen-2.5-7b-instruct.jinja",
        help="Path to the chat template file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for training",
    )

    # Training control (custom)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--evaluation_steps", type=int, default=500)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )

    # Fine-tuning options
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA fine-tuning",
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to the lora checkpoint",
    )
    parser.add_argument(
        "--lora_checkpoint_path",
        type=str,
        default=None,
        help="Path to the lora checkpoint",
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Enable QLoRA fine-tuning",
    )

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    accelerator = Accelerator()
    args = get_args()
    trainer = SotopiaSFTTrainer(args, accelerator)
    trainer.train()
    trainer.save_model()
    trainer.save_state()