from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from core.config import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Laptop-friendly QLoRA trainer for the NeuroGolf student model.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "outputs" / "distillation" / "teacher_prompts.jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "distillation" / "student_lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _require_training_packages() -> None:
    missing = []
    for package in ("torch", "transformers", "datasets", "peft", "trl", "accelerate"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        joined = " ".join(missing)
        raise SystemExit(
            "Missing training packages. Install them first with:\n"
            f".\\.venv\\Scripts\\python.exe -m pip install {joined}\n"
            "For 4-bit QLoRA on Windows, bitsandbytes may be unreliable; start with the 1.5B model."
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
                "note": "Dry run only. Install training packages before real training.",
            }
        )
        return 0

    _require_training_packages()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")

    def formatting_func(example):
        messages = example["messages"]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    peft_config = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_r) * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(args.grad_accum),
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.lr),
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=int(args.max_seq_length),
    )
    trainer.train()
    trainer.save_model(str(args.out))
    print({"saved": str(args.out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
