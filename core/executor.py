from __future__ import annotations

from time import sleep

import requests

from core.config import OLLAMA_BASE_URL, REQUEST_TIMEOUT


OLLAMA_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"


def ask(
    model: str,
    prompt: str,
    system: str | None = None,
    timeout: int | None = None,
    retries: int = 2,
    options: dict | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False,
            }
            if options:
                payload["options"] = options
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout or REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload.get("response", "").strip()
            if not text:
                raise RuntimeError(f"Ollama returned an empty response for model '{model}'.")
            return text
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                sleep(1.0)
    raise RuntimeError(f"Failed to query Ollama model '{model}': {last_error}") from last_error
