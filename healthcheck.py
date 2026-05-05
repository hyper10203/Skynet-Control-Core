from __future__ import annotations

import requests

from core.config import MODEL_OPTIONS, MODEL_REGISTRY, OLLAMA_BASE_URL, REQUEST_TIMEOUT


def _needs_explicit_no_think(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("qwen3")


def _probe_model(model: str, options: dict) -> tuple[bool, str]:
    try:
        payload = {
            "model": model,
            "prompt": "Reply with exactly: pong",
            "system": "Return exactly one word: pong",
            "stream": False,
            "keep_alive": "2m",
            "options": {
                "num_ctx": max(2048, min(int(options.get("num_ctx", 4096)), 4096)),
                "num_predict": 8,
                "temperature": 0.0,
            },
        }
        if _needs_explicit_no_think(model):
            payload["think"] = False
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json=payload,
            timeout=min(REQUEST_TIMEOUT, 180),
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("response", "")).strip().lower()
        if not text:
            return False, "empty"
        return True, text
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=30)
    response.raise_for_status()
    tags = {item["name"] for item in response.json().get("models", [])}
    aliases = tags | {name.split(":", 1)[0] for name in tags}
    print("Ollama base URL:", OLLAMA_BASE_URL)
    failures = 0
    for role, model in MODEL_REGISTRY.items():
        installed = model in aliases
        if not installed:
            failures += 1
            print(f"{role}: {model} -> missing | options={MODEL_OPTIONS.get(role, {})}")
            continue
        ok, detail = _probe_model(model, MODEL_OPTIONS.get(role, {}))
        if ok:
            print(f"{role}: {model} -> ok | options={MODEL_OPTIONS.get(role, {})}")
        else:
            failures += 1
            print(f"{role}: {model} -> probe_failed ({detail}) | options={MODEL_OPTIONS.get(role, {})}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
