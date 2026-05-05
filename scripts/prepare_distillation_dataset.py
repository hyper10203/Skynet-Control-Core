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
    parser.add_argument("--max-train", type=int, default=3)
    parser.add_argument("--max-test", type=int, default=1)
    parser.add_argument("--max-arc-gen", type=int, default=1)
    parser.add_argument("--max-grid-cells", type=int, default=81)
    parser.add_argument("--preview-size", type=int, default=8)
    return parser


def _grid_summary(grid: Any, *, max_cells: int, preview_size: int) -> Any:
    if not isinstance(grid, list):
        return grid
    height = len(grid)
    width = max((len(row) for row in grid if isinstance(row, list)), default=0)
    cells = height * width
    if cells <= max_cells:
        return grid

    values: list[int] = []
    nonzero_points: list[tuple[int, int]] = []
    for row_index, row in enumerate(grid):
        if not isinstance(row, list):
            continue
        for col_index, value in enumerate(row):
            if isinstance(value, int):
                values.append(value)
                if value != 0:
                    nonzero_points.append((row_index, col_index))
    if nonzero_points:
        rows = [point[0] for point in nonzero_points]
        cols = [point[1] for point in nonzero_points]
        bbox = [min(rows), min(cols), max(rows), max(cols)]
    else:
        bbox = []
    return {
        "shape": [height, width],
        "palette": sorted(set(values)),
        "nonzero_count": len(nonzero_points),
        "nonzero_bbox": bbox,
        "preview": [row[:preview_size] if isinstance(row, list) else row for row in grid[:preview_size]],
        "summary_note": "large grid summarized to keep distillation records inside the training token budget",
    }


def _compact_pair(pair: Any, *, max_cells: int, preview_size: int, keep_output: bool = True) -> Any:
    if not isinstance(pair, dict):
        return pair
    compact = dict(pair)
    compact["input"] = _grid_summary(compact.get("input"), max_cells=max_cells, preview_size=preview_size)
    if keep_output and "output" in compact:
        compact["output"] = _grid_summary(compact.get("output"), max_cells=max_cells, preview_size=preview_size)
    elif not keep_output:
        compact.pop("output", None)
    return compact


def _compact_examples(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    max_cells = max(16, int(args.max_grid_cells))
    preview_size = max(4, int(args.preview_size))
    def compact_many(key: str, count: int, *, keep_output: bool = True) -> list[Any]:
        return [
            _compact_pair(item, max_cells=max_cells, preview_size=preview_size, keep_output=keep_output)
            for item in task.get(key, [])[: max(0, int(count))]
        ]
    return {
        "train": compact_many("train", max(1, int(args.max_train))),
        "test": compact_many("test", max(0, int(args.max_test)), keep_output=False),
        "arc-gen": compact_many("arc-gen", max(0, int(args.max_arc_gen))),
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
            examples = _compact_examples(task, args)
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
