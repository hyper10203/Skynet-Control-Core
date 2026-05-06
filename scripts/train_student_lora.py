from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from core.config import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Laptop-friendly QLoRA trainer for the NeuroGolf student model.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "distillation" / "teacher_prompts.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "distillation" / "student_lora",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _require_training_packages(load_in_4bit: bool) -> None:
    required = ["torch", "transformers", "datasets", "peft", "trl", "accelerate"]
    if load_in_4bit:
        required.append("bitsandbytes")
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        joined = " ".join(missing)
        extra = (
            "\nIf you want a bigger 4-bit student locally, install Linux-friendly bitsandbytes support "
            "or run the Kaggle GPU path instead."
            if load_in_4bit
            else ""
        )
        raise SystemExit(
            "Missing training packages. Install them first with:\n"
            f".\\.venv\\Scripts\\python.exe -m pip install {joined}{extra}"
        )


def main() -> int:
    args = build_parser().parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if args.dry_run:
        print(
            {
                "model": args.model,
                "dataset": str(args.dataset),
                "out": str(args.out),
                "max_seq_length": args.max_seq_length,
                "epochs": args.epochs,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "grad_accum": args.grad_accum,
                "load_in_4bit": args.load_in_4bit,
                "note": "Dry run only. Use --load-in-4bit for larger QLoRA-capable models.",
            }
        )
        return 0

    _require_training_packages(bool(args.load_in_4bit))

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer
    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_seq_length = int(args.max_seq_length)
    if not tokenizer.model_max_length or tokenizer.model_max_length > max_seq_length:
        tokenizer.model_max_length = max_seq_length

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")

    def truncate_for_sft(text: str) -> str:
        truncate_budget = max(32, max_seq_length - 64)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=truncate_budget,
            add_special_tokens=False,
        )
        input_ids = tokenized.get("input_ids", [])
        if not input_ids:
            return text
        return tokenizer.decode(input_ids, skip_special_tokens=False)

    def formatting_func(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return truncate_for_sft(text)

    peft_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_r) * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_kwargs = {
        "output_dir": str(args.out),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": int(args.grad_accum),
        "num_train_epochs": float(args.epochs),
        "learning_rate": float(args.lr),
        "logging_steps": 5,
        "save_steps": 50,
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "fp16": torch.cuda.is_available(),
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "report_to": [],
    }
    sft_config_params = inspect.signature(SFTConfig.__init__).parameters if SFTConfig is not None else {}
    if SFTConfig is not None and "max_length" in sft_config_params:
        training_kwargs["max_length"] = max_seq_length
        training_args = SFTConfig(**training_kwargs)
    elif SFTConfig is not None and "max_seq_length" in sft_config_params:
        training_kwargs["max_seq_length"] = max_seq_length
        training_args = SFTConfig(**training_kwargs)
    else:
        training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "formatting_func": formatting_func,
        "peft_config": peft_config,
        "args": training_args,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = max_seq_length
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(args.out))
    print({"saved": str(args.out), "model": args.model, "load_in_4bit": bool(args.load_in_4bit)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
