from __future__ import annotations

from copy import deepcopy
import subprocess
from time import sleep

import requests

from core.config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OPENCLAUDE_BIN,
    OPENCLAUDE_EFFORT,
    OPENCLAUDE_PROVIDER,
    PROJECT_ROOT,
    REQUEST_TIMEOUT,
)


OLLAMA_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"


def _openclaude_command(model: str, prompt: str, system: str | None = None) -> list[str]:
    command = [
        OPENCLAUDE_BIN,
        "--provider",
        OPENCLAUDE_PROVIDER,
        "--model",
        str(model).strip(),
        "--bare",
        "--effort",
        OPENCLAUDE_EFFORT,
        "--print",
    ]
    if system:
        command.extend(["--system-prompt", str(system)])
    command.append(prompt)
    return command


def _needs_explicit_no_think(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("qwen3")


def _degraded_options(options: dict | None, attempt: int) -> dict | None:
    if not options:
        return None
    degraded = deepcopy(options)
    if attempt <= 0:
        return degraded
    if "num_ctx" in degraded:
        base_ctx = int(degraded["num_ctx"])
        degraded["num_ctx"] = max(4096, int(base_ctx * (0.75**attempt)))
    if "num_predict" in degraded:
        base_predict = int(degraded["num_predict"])
        degraded["num_predict"] = max(96, int(base_predict * (0.7**attempt)))
    if "temperature" in degraded:
        degraded["temperature"] = min(float(degraded["temperature"]), 0.2)
    return degraded


def _fallback_options(options: dict | None) -> dict | None:
    if not options:
        return None
    fallback = deepcopy(options)
    if "num_ctx" in fallback:
        fallback["num_ctx"] = max(4096, min(int(fallback["num_ctx"]), 8192))
    if "num_predict" in fallback:
        fallback["num_predict"] = max(128, min(int(fallback["num_predict"]), 320))
    if "temperature" in fallback:
        fallback["temperature"] = min(float(fallback["temperature"]), 0.2)
    return fallback


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
            if LLM_BACKEND == "openclaude":
                completed = subprocess.run(
                    _openclaude_command(model, prompt, system=system),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=PROJECT_ROOT,
                    timeout=timeout or REQUEST_TIMEOUT,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"OpenClaude returned {completed.returncode} for model '{model}': {str(completed.stderr or completed.stdout).strip()}"
                    )
                text = str(completed.stdout or "").strip()
                if not text:
                    raise RuntimeError(f"OpenClaude returned an empty response for model '{model}'.")
                return text

            payload = {
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
            if _needs_explicit_no_think(model):
                payload["think"] = False
            degraded_options = _degraded_options(options, attempt)
            if degraded_options:
                payload["options"] = degraded_options
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout or REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            text = str(payload.get("response", "")).strip()
            if not text and isinstance(payload.get("message"), dict):
                text = str(payload["message"].get("content", "")).strip()
            if not text:
                raise RuntimeError(f"Ollama returned an empty response for model '{model}'.")
            return text
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                sleep(1.0)
    raise RuntimeError(f"Failed to query Ollama model '{model}': {last_error}") from last_error


def ask_with_fallback(
    model: str,
    prompt: str,
    system: str | None = None,
    timeout: int | None = None,
    retries: int = 2,
    options: dict | None = None,
    fallback_models: list[str] | tuple[str, ...] | None = None,
) -> str:
    ordered_models: list[str] = [model]
    for candidate in fallback_models or ():
        clean = str(candidate or "").strip()
        if clean and clean not in ordered_models:
            ordered_models.append(clean)

    errors: list[str] = []
    for index, candidate_model in enumerate(ordered_models):
        try:
            candidate_options = options if index == 0 else _fallback_options(options)
            return ask(
                candidate_model,
                prompt,
                system=system,
                timeout=timeout,
                retries=retries,
                options=candidate_options,
            )
        except Exception as exc:
            errors.append(f"{candidate_model}: {exc}")
    raise RuntimeError("All model attempts failed: " + " | ".join(errors))
