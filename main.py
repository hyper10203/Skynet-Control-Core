from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.arc_loader import load_tasks
from core.config import DEFAULT_DATA_DIR
from core.solver_loop import solve_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local ARC agent system against task JSON files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Optional task limit. 0 means all tasks.")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--critique-rounds", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = load_tasks(args.data_dir)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    args.save_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using task directory: {Path(args.data_dir).resolve()}")

    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task['id']}")
        result = solve_task(task, critique_rounds=args.critique_rounds)
        output_path = args.save_dir / f"{task['id']}_solution.py"
        output_path.write_text(result["code"], encoding="utf-8")
        report_path = report_dir / f"{task['id']}_report.json"
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
