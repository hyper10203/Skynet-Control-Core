from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from core.config import PROJECT_ROOT


TRAIN_SCRIPT = r'''
from __future__ import annotations

import base64
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

DATASET_NAME = "teacher_prompts.jsonl"
EMBEDDED_DATASET_B64 = "__DATASET_B64__"
OUT_DIR = Path("/kaggle/working/student_lora")
MODEL_NAME = __MODEL_NAME__
MAX_SEQ_LENGTH = __MAX_SEQ_LENGTH__
LORA_R = __LORA_R__
GRAD_ACCUM = __GRAD_ACCUM__
NUM_EPOCHS = __NUM_EPOCHS__
LEARNING_RATE = __LEARNING_RATE__
LOAD_IN_4BIT = __LOAD_IN_4BIT__


def ensure_packages():
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision", "torchaudio"],
        check=False,
    )
    packages = [
        "torch==2.6.0",
        "transformers==4.51.3",
        "datasets",
        "peft",
        "trl==0.15.2",
        "accelerate",
        "bitsandbytes",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def find_dataset() -> Path:
    candidates = [
        Path(DATASET_NAME),
        Path("/kaggle/working") / DATASET_NAME,
        Path(__file__).resolve().parent / DATASET_NAME,
    ]
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            candidates.extend(root.rglob(DATASET_NAME))
    for path in candidates:
        if path.exists():
            return path
    if len(EMBEDDED_DATASET_B64) > 100:
        embedded_path = Path("/kaggle/working") / DATASET_NAME
        embedded_path.write_bytes(base64.b64decode(EMBEDDED_DATASET_B64.encode("ascii")))
        return embedded_path
    raise SystemExit(f"Dataset missing: {DATASET_NAME}; searched {[str(p) for p in candidates[:10]]}")


def main():
    ensure_packages()
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

    print(
        {
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "model": MODEL_NAME,
            "load_in_4bit": LOAD_IN_4BIT,
        }
    )
    dataset_path = find_dataset()
    print({"dataset": str(dataset_path)})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.model_max_length or tokenizer.model_max_length > MAX_SEQ_LENGTH:
        tokenizer.model_max_length = MAX_SEQ_LENGTH

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if LOAD_IN_4BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def truncate_for_sft(text: str) -> str:
        truncate_budget = max(32, MAX_SEQ_LENGTH - 64)
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
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return truncate_for_sft(text)

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    training_kwargs = {
        "output_dir": str(OUT_DIR),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": 5,
        "save_steps": 25,
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "fp16": torch.cuda.is_available(),
        "optim": "paged_adamw_8bit" if LOAD_IN_4BIT else "adamw_torch",
        "report_to": [],
    }
    sft_config_params = inspect.signature(SFTConfig.__init__).parameters if SFTConfig is not None else {}
    if SFTConfig is not None and "max_length" in sft_config_params:
        training_kwargs["max_length"] = MAX_SEQ_LENGTH
        training_args = SFTConfig(**training_kwargs)
    elif SFTConfig is not None and "max_seq_length" in sft_config_params:
        training_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
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
        trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(str(OUT_DIR))
    shutil.make_archive("/kaggle/working/student_lora", "zip", OUT_DIR)
    Path("/kaggle/working/distillation_result.json").write_text(
        json.dumps(
            {
                "base_model": MODEL_NAME,
                "records": len(dataset),
                "artifact": "/kaggle/working/student_lora.zip",
                "load_in_4bit": LOAD_IN_4BIT,
                "max_seq_length": MAX_SEQ_LENGTH,
                "lora_r": LORA_R,
                "epochs": NUM_EPOCHS,
                "gradient_accumulation_steps": GRAD_ACCUM,
                "train_metrics": train_result.metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a Kaggle GPU kernel for student LoRA training.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "distillation" / "teacher_prompts.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "kaggle" / "axiomgraph-distill-student",
    )
    parser.add_argument(
        "--id",
        default="subhampaulchoudhury/axiomgraph-neurogolf-student-distillation-v13",
    )
    parser.add_argument(
        "--title",
        default="AxiomGraph NeuroGolf Student Distillation v13",
    )
    parser.add_argument(
        "--dataset-source",
        default="subhampaulchoudhury/axiomgraph-neurogolf-distillation-data",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Base Hugging Face model for the Kaggle student run.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Disable the 4-bit QLoRA path and load the model in regular precision.",
    )
    parser.add_argument("--embed-dataset", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.dataset, args.out_dir / "teacher_prompts.jsonl")

    script_text = TRAIN_SCRIPT
    replacements = {
        "__MODEL_NAME__": json.dumps(str(args.model)),
        "__MAX_SEQ_LENGTH__": str(int(args.max_seq_length)),
        "__LORA_R__": str(int(args.lora_r)),
        "__GRAD_ACCUM__": str(int(args.grad_accum)),
        "__NUM_EPOCHS__": repr(float(args.epochs)),
        "__LEARNING_RATE__": repr(float(args.lr)),
        "__LOAD_IN_4BIT__": "False" if args.full_precision else "True",
    }
    for key, value in replacements.items():
        script_text = script_text.replace(key, value)

    if args.embed_dataset:
        dataset_b64 = base64.b64encode(args.dataset.read_bytes()).decode("ascii")
        script_text = script_text.replace("__DATASET_B64__", dataset_b64)
    else:
        script_text = script_text.replace("__DATASET_B64__", "")

    (args.out_dir / "kaggle_distill_train.py").write_text(
        script_text.strip() + "\n",
        encoding="utf-8",
    )
    metadata = {
        "id": args.id,
        "title": args.title,
        "code_file": "kaggle_distill_train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [args.dataset_source] if args.dataset_source else [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "kernel_dir": str(args.out_dir),
                "metadata": str(args.out_dir / "kernel-metadata.json"),
                "model": args.model,
                "load_in_4bit": not args.full_precision,
                "max_seq_length": int(args.max_seq_length),
                "lora_r": int(args.lora_r),
                "epochs": float(args.epochs),
                "grad_accum": int(args.grad_accum),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
