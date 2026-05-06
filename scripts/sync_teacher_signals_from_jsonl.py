from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from core.config import PROJECT_ROOT
from core.distillation import load_distillation_plan, save_distillation_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Restore teacher signals into the distillation plan from a JSONL dataset.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "distillation" / "teacher_prompts.jsonl",
        help="Path to a supervised teacher-prompts JSONL file.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing teacher signals instead of merging by task/source/rule.",
    )
    return parser


def _normalize_task_id(task_id: str) -> str:
    task_id = str(task_id).strip()
    if task_id.isdigit():
        return f"task{int(task_id):03d}"
    return task_id


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _assistant_payload(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages", [])
    if not isinstance(messages, list):
        return {}
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip() != "assistant":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        if isinstance(content, dict):
            return content
    return {}


def main() -> int:
    args = build_parser().parse_args()
    if not args.jsonl.exists():
        raise SystemExit(f"JSONL not found: {args.jsonl}")

    plan = load_distillation_plan()
    existing = [] if args.replace else list(plan.get("teacher_signals", []))
    seen = {
        (
            _normalize_task_id(item.get("task_id", "")),
            str(item.get("source", "")).strip(),
            str(item.get("rule_summary", "")).strip(),
        )
        for item in existing
        if isinstance(item, dict)
    }

    restored = 0
    for record in _load_records(args.jsonl):
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        task_id = _normalize_task_id(metadata.get("task_id", ""))
        if not task_id:
            continue
        assistant = _assistant_payload(record)
        rule_summary = str(assistant.get("rule_summary", "")).strip()
        graph_hint = str(assistant.get("graph_template", "")).strip()
        source = str(metadata.get("source", "jsonl_import")).strip() or "jsonl_import"
        if not rule_summary or not graph_hint:
            continue
        key = (task_id, source, rule_summary)
        if key in seen:
            continue
        existing.append(
            {
                "created_at": str(record.get("created_at") or plan.get("updated_at") or ""),
                "task_id": task_id,
                "rule_summary": rule_summary,
                "graph_hint": graph_hint,
                "source": source,
            }
        )
        seen.add(key)
        restored += 1

    plan["teacher_signals"] = existing[-500:]
    save_distillation_plan(plan)
    print(
        json.dumps(
            {
                "jsonl": str(args.jsonl),
                "restored": restored,
                "teacher_signals": len(plan["teacher_signals"]),
                "replaced": bool(args.replace),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
