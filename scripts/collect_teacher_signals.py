from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from core.config import DEFAULT_DATA_DIR, PROJECT_ROOT, REQUEST_TIMEOUT
from core.distillation import load_distillation_plan, record_teacher_signal, set_fine_tune_policy
from core.executor import ask_with_fallback


SYSTEM_PROMPT = """You are a NeuroGolf teacher model.

Goal: infer the ARC transformation and propose the smallest legal static ONNX student graph.

Rules:
- Return JSON only.
- Do not propose dynamic shapes, symbolic dimensions, banned ops, hidden constants, or metric quirks.
- Prefer tiny symbolic/static graphs: Identity, Slice, Pad, Transpose, Gather, OneHot, ArgMax, Add, Mul, simple color maps, or very small Conv only as fallback.
- The student graph is scored by params + memory + MACs, so remove every unnecessary op and parameter.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect NeuroGolf teacher signals from a local Ollama model.")
    parser.add_argument("--model", default="", help="Ollama model name. Defaults to distillation plan teacher or qwen3:8b.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=4)
    parser.add_argument("--max-test", type=int, default=1)
    parser.add_argument("--max-arc-gen", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _compact_examples(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "train": task.get("train", [])[: max(1, int(args.max_train))],
        "test": task.get("test", [])[: max(0, int(args.max_test))],
        "arc-gen": task.get("arc-gen", [])[: max(0, int(args.max_arc_gen))],
    }


def _prompt(task_id: str, examples: dict[str, Any]) -> str:
    payload = {
        "task_id": task_id,
        "examples": examples,
        "required_json_schema": {
            "rule_summary": "one concise sentence describing the transformation",
            "graph_hint": "smallest static ONNX graph template to try",
            "candidate_ops": ["short list of allowed ONNX op names"],
            "cost_notes": "how to minimize params, memory, and MACs",
            "risk_notes": "validation or overfit risks",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    markdown = _extract_markdown_sections(text)
    return markdown


def _extract_markdown_sections(text: str) -> dict[str, Any]:
    labels = {
        "rule_summary": ["rule summary", "rule"],
        "graph_hint": ["graph hint", "graph template"],
        "candidate_ops": ["candidate ops", "ops"],
        "cost_notes": ["cost notes", "cost"],
        "risk_notes": ["risk notes", "risks"],
    }
    result: dict[str, Any] = {}
    normalized = text.replace("\r\n", "\n")
    for key, names in labels.items():
        for name in names:
            pattern = re.compile(
                rf"(?:\*\*)?{re.escape(name)}\s*:?(?:\*\*)?\s*:?\s*(.*?)(?=\n\s*(?:\*\*)?(?:rule summary|rule|graph hint|graph template|candidate ops|ops|cost notes|cost|risk notes|risks)\s*:?(?:\*\*)?\s*:|\Z)",
                flags=re.IGNORECASE | re.DOTALL,
            )
            found = pattern.search(normalized)
            if found:
                value = found.group(1).strip(" \n`*-")
                if key == "candidate_ops":
                    result[key] = re.findall(r"[A-Za-z][A-Za-z0-9_]*", value)
                else:
                    result[key] = value
                break
    return result


def _existing_tasks() -> set[str]:
    plan = load_distillation_plan()
    seen = set()
    for signal in plan.get("teacher_signals", []):
        task_id = str(signal.get("task_id", "")).strip()
        if task_id:
            seen.add(task_id)
    return seen


def _ollama_generate(model: str, prompt: str, args: argparse.Namespace) -> str:
    return ask_with_fallback(
        model,
        prompt,
        system=SYSTEM_PROMPT,
        timeout=REQUEST_TIMEOUT,
        retries=1,
        options={
            "num_ctx": int(args.num_ctx),
            "num_predict": int(args.num_predict),
            "temperature": float(args.temperature),
        },
        fallback_models=("qwen2.5-coder:latest", "mistral:7b-text-fp16", "llama3:latest"),
    )


def main() -> int:
    args = build_parser().parse_args()
    plan = load_distillation_plan()
    model = args.model.strip() or str(plan.get("teacher_model") or "").strip() or "qwen3:8b"
    set_fine_tune_policy(
        enabled=bool((plan.get("fine_tune_policy") or {}).get("enabled", False)),
        dataset_path=str(args.data_dir),
        output_adapter_path=str((plan.get("fine_tune_policy") or {}).get("output_adapter_path", "")),
        teacher_model=model,
    )

    task_paths = sorted(args.data_dir.glob("task*.json"))
    if args.start > 1:
        task_paths = [path for path in task_paths if int(path.stem.replace("task", "")) >= args.start]
    if args.limit > 0:
        task_paths = task_paths[: args.limit]

    existing = set() if args.overwrite else _existing_tasks()
    written = 0
    failed: list[dict[str, str]] = []
    for path in task_paths:
        task_id = path.stem
        if task_id in existing:
            continue
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
            raw = _ollama_generate(model, _prompt(task_id, _compact_examples(task, args)), args)
            parsed = _extract_json(raw)
            rule_summary = str(parsed.get("rule_summary", "")).strip()
            graph_hint = str(parsed.get("graph_hint", "")).strip()
            if not rule_summary or not graph_hint:
                raise RuntimeError(f"teacher returned incomplete JSON: {raw[:500]}")
            candidate_ops = parsed.get("candidate_ops", [])
            cost_notes = str(parsed.get("cost_notes", "")).strip()
            risk_notes = str(parsed.get("risk_notes", "")).strip()
            enriched_hint = graph_hint
            if candidate_ops:
                enriched_hint += f" | ops={candidate_ops}"
            if cost_notes:
                enriched_hint += f" | cost={cost_notes}"
            if risk_notes:
                enriched_hint += f" | risks={risk_notes}"
            record_teacher_signal(task_id, rule_summary, enriched_hint, source=f"ollama:{model}")
            written += 1
            print(json.dumps({"task_id": task_id, "ok": True, "rule_summary": rule_summary}, ensure_ascii=False))
        except Exception as exc:
            failed.append({"task_id": task_id, "error": str(exc)})
            print(json.dumps({"task_id": task_id, "ok": False, "error": str(exc)}, ensure_ascii=False))
        time.sleep(max(0.0, float(args.sleep)))

    print(json.dumps({"model": model, "written": written, "failed": failed, "plan_path": str(PROJECT_ROOT / "memory" / "distillation_plan.json")}, indent=2, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
