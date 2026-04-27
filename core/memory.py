from __future__ import annotations

import json
from datetime import datetime

from core.config import MEMORY_PATH

KEYWORD_VOCAB = [
    "rotate",
    "rotation",
    "flip",
    "mirror",
    "color",
    "count",
    "symmetry",
    "move",
    "translation",
    "diagonal",
    "flood fill",
    "fill",
    "crop",
    "repeat",
    "tile",
    "object",
    "bbox",
    "frame",
]


def load_memory() -> list[dict]:
    if not MEMORY_PATH.exists():
        return []
    with MEMORY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_memory(data: list[dict]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def extract_keywords(text: str) -> list[str]:
    lowered = text.lower()
    keywords: list[str] = []
    for word in KEYWORD_VOCAB:
        if word in lowered:
            keywords.append(word)
    return sorted(set(keywords))


def retrieve_patterns(task: dict, memory: list[dict], limit: int = 3) -> list[dict]:
    task_text = json.dumps(task, sort_keys=True).lower()
    task_keywords = set(extract_keywords(task_text))
    scored: list[tuple[int, dict]] = []
    for item in memory:
        overlap = sum(1 for keyword in item.get("keywords", []) if keyword in task_text)
        overlap += len(task_keywords.intersection(item.get("keywords", [])))
        if overlap > 0 and item.get("success", True):
            scored.append((overlap, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def replace_source_patterns(source: str, new_items: list[dict]) -> None:
    memory = [item for item in load_memory() if item.get("source") != source]
    memory.extend(new_items)
    save_memory(memory)


def store_pattern(task: dict, reasoning: str, success: bool = True, code: str | None = None, source: str = "solver_loop") -> None:
    memory = load_memory()
    memory.append(
        {
            "task": str(task)[:200],
            "reasoning": reasoning,
            "keywords": extract_keywords(reasoning),
            "success": success,
            "code_preview": (code or "")[:400],
            "source": source,
            "stored_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )
    save_memory(memory)
