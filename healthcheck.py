from __future__ import annotations

import requests

from core.config import MODEL_OPTIONS, MODEL_REGISTRY, OLLAMA_BASE_URL


def main() -> int:
    response = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=30)
    response.raise_for_status()
    tags = {item["name"] for item in response.json().get("models", [])}
    aliases = tags | {name.split(":", 1)[0] for name in tags}
    print("Ollama base URL:", OLLAMA_BASE_URL)
    for role, model in MODEL_REGISTRY.items():
        status = "ok" if model in aliases else "missing"
        print(f"{role}: {model} -> {status} | options={MODEL_OPTIONS.get(role, {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
