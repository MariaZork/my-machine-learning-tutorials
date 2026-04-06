"""
train_lora.py
─────────────
Fine-tune Qwen2.5-3B on the Légifrance dataset using LoRA + MPS.

Usage:
    python scripts/train_lora.py
"""

from pathlib import Path
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig 

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
    torch.set_float32_matmul_precision("high")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"Device: {DEVICE}  |  dtype: {DTYPE}")

MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR  = Path("outputs/lora-legifrance")
DATA_DIR    = Path("data")

LORA_CFG = dict(
    r             = 16,
    lora_alpha    = 32,
    lora_dropout  = 0.05,
    target_modules= [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias          = "none",
    task_type     = TaskType.CAUSAL_LM,
    use_rslora    = True,
)

SFT_ARGS = SFTConfig(
    output_dir                  = str(OUTPUT_DIR),
    max_length                  = 512,         # shorter sequence for speed on constrained mps memory
    num_train_epochs            = 3,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 32,           # fewer accumulations can speed up step throughput
    learning_rate               = 1e-4,
    warmup_ratio                = 0.10,
    lr_scheduler_type           = "cosine",
    fp16                        = True,        # MPS/CUDA prefers fp16 on modern HW
    bf16                        = False,
    dataloader_pin_memory       = True,
    dataloader_num_workers      = 2,
    optim                       = "adamw_torch",
    gradient_checkpointing      = True,
    logging_steps               = 20,
    eval_strategy               = "no",
    save_strategy               = "steps",
    save_steps                  = 200,
    save_total_limit            = 2,
    report_to                   = "none",
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if DEVICE == "cuda" else {"": DEVICE},
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # 3. LoRA
    model = get_peft_model(model, LoraConfig(**LORA_CFG))
    model.print_trainable_parameters()

    # 4. Dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train"     : str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "eval.jsonl"),
        },
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model             = model,
        args              = SFT_ARGS,
        train_dataset     = dataset["train"],
        eval_dataset      = dataset["validation"],
        processing_class  = tokenizer,
        formatting_func   = lambda x: x["text"],
    )

    print("Starting training on Mac (MPS)...")
    trainer.train()
    
    trainer.save_model(str(OUTPUT_DIR / "final"))
    print(f"\nTraining complete. Adapter saved to {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()