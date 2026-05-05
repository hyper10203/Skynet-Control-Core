from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from core.config import DEFAULT_DATA_DIR, PROJECT_ROOT
from core.distillation import load_distillation_plan


SYSTEM_PROMPT = (
    "You are a NeuroGolf teacher model. Infer the ARC transformation rule, "
    "then propose the smallest legal static ONNX graph template. Prefer symbolic "
    "geometry, color maps, slicing, padding, and lookup-style graphs over learned weights."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare NeuroGolf teacher/distillation JSONL.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "distillation" / "teacher_prompts.jsonl")
    parser.add_argument("--include-unlabeled", action="store_true", help="Write prompt-only records when no teacher signal exists.")
    parser.add_argument("--limit", type=int, default=0, help="0 means all tasks.")
    return parser


def _compact_examples(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": task.get("train", []),
        "test": task.get("test", []),
        "arc-gen": task.get("arc-gen", []),
    }


def _task_id(path: Path) -> str:
    return path.stem


def _teacher_signals_by_task() -> dict[str, list[dict[str, Any]]]:
    plan = load_distillation_plan()
    signals: dict[str, list[dict[str, Any]]] = {}
    for item in plan.get("teacher_signals", []):
        task_id = str(item.get("task_id", "")).strip()
        if not task_id:
            continue
        if task_id.isdigit():
            task_id = f"task{int(task_id):03d}"
        signals.setdefault(task_id, []).append(item)
    return signals


def _user_content(task_id: str, examples: dict[str, Any]) -> str:
    payload = {
        "task_id": task_id,
        "examples": examples,
        "required_output": {
            "rule_summary": "short symbolic rule",
            "graph_template": "smallest static ONNX graph idea",
            "cost_strategy": "how to minimize params, memory, and MACs",
            "validation_risks": "dynamic shape, banned op, or overfit risks",
        },
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _assistant_content(signal: dict[str, Any]) -> str:
    payload = {
        "rule_summary": signal.get("rule_summary", ""),
        "graph_template": signal.get("graph_hint", ""),
        "cost_strategy": "Prefer the smallest valid graph that solves all public examples; reject extra parameters and large intermediates.",
        "validation_risks": "Must pass static shape inference, banned-op checks, and local cost profiling before selection.",
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def main() -> int:
    args = build_parser().parse_args()
    task_paths = sorted(args.data_dir.glob("task*.json"))
    if args.limit > 0:
        task_paths = task_paths[: args.limit]
    signals = _teacher_signals_by_task()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    supervised = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for path in task_paths:
            task_id = _task_id(path)
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            examples = _compact_examples(task)
            task_signals = signals.get(task_id, [])
            if task_signals:
                for signal in task_signals:
                    record = {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": _user_content(task_id, examples)},
                            {"role": "assistant", "content": _assistant_content(signal)},
                        ],
                        "metadata": {"task_id": task_id, "source": signal.get("source", "operator")},
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    supervised += 1
            elif args.include_unlabeled:
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _user_content(task_id, examples)},
                    ],
                    "metadata": {"task_id": task_id, "source": "unlabeled_prompt"},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(json.dumps({"output": str(args.out), "records": written, "supervised_records": supervised}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
