from __future__ import annotations

import json
from pathlib import Path

from core.config import MAX_ARC_GEN_EXAMPLES, MAX_TEST_EXAMPLES, MAX_TRAIN_EXAMPLES


def load_tasks(data_dir: str | Path) -> list[dict]:
    base = Path(data_dir)
    task_files = sorted(base.rglob("task*.json"))
    tasks: list[dict] = []
    for path in task_files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tasks.append(
            {
                "id": path.stem,
                "path": str(path),
                "payload": payload,
            }
        )
    return tasks


def serialize_task(task: dict) -> str:
    payload = {
        "id": task.get("id"),
        "train": task.get("payload", {}).get("train", [])[:MAX_TRAIN_EXAMPLES],
        "test": task.get("payload", {}).get("test", [])[:MAX_TEST_EXAMPLES],
        "arc-gen": task.get("payload", {}).get("arc-gen", [])[:MAX_ARC_GEN_EXAMPLES],
    }
    return json.dumps(payload, separators=(",", ":"))
