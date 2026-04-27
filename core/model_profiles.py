from __future__ import annotations

MODEL_SOURCES = {
    "qwen3": "https://ollama.com/library/qwen3",
    "qwen3.6": "https://ollama.com/library/qwen3",
    "qwen3-coder": "https://ollama.com/library/qwen3-coder",
    "mistral-small": "https://ollama.com/library/mistral-small",
    "devstral": "https://ollama.com/library/devstral",
    "devstral-small-2": "https://ollama.com/library/devstral-small-2",
    "devstral-2": "https://ollama.com/library/devstral-2",
    "gemma3": "https://ollama.com/library/gemma3",
    "gemma4": "https://ollama.com/library/gemma3",
    "llama3.3": "https://ollama.com/library/llama3.3",
    "qwen2.5-coder": "https://ollama.com/library/qwen2.5-coder",
    "deepseek-r1": "https://ollama.com/library/deepseek-r1",
    "deepseek-coder-v2": "https://ollama.com/library/deepseek-coder-v2",
}


MODEL_PROFILES = {
    "current-safe": {
        "label": "Current Safe",
        "tier": "Installed / Low Risk",
        "summary": "Keeps the current lightweight stack and avoids aggressive swaps. Best if you want the daemon stable right now.",
        "roles": {
            "orchestrator": "qwen2.5",
            "operator_fast": "qwen2.5",
            "reasoning_primary": "qwen2.5",
            "reasoning_secondary": "mistral",
            "reasoning_tertiary": "llama3",
            "coder": "qwen2.5-coder",
            "critic": "llama3",
        },
        "contexts": {
            "orchestrator": 8192,
            "operator_fast": 6144,
            "reasoning_primary": 8192,
            "reasoning_secondary": 8192,
            "reasoning_tertiary": 8192,
            "coder": 8192,
            "critic": 6144,
        },
        "notes": [
            "Uses only models already installed on this machine.",
            "Primary reasoning is moved off DeepSeek-R1 because that model is unstable in the current runtime.",
        ],
        "recommended_pulls": [],
    },
    "balanced-2026": {
        "label": "Balanced 2026",
        "tier": "Single GPU / Strong Upgrade",
        "summary": "Best overall upgrade path if you want stronger planning, longer context, and a more agentic coding specialist without jumping to giant models.",
        "roles": {
            "orchestrator": "qwen3:8b",
            "operator_fast": "qwen3:8b",
            "reasoning_primary": "gemma3:12b",
            "reasoning_secondary": "mistral-small:24b",
            "reasoning_tertiary": "qwen3:14b",
            "coder": "devstral-small-2:24b",
            "critic": "qwen3:14b",
        },
        "contexts": {
            "orchestrator": 12288,
            "operator_fast": 8192,
            "reasoning_primary": 16384,
            "reasoning_secondary": 12288,
            "reasoning_tertiary": 12288,
            "coder": 16384,
            "critic": 8192,
        },
        "notes": [
            "Qwen3 is the latest Qwen generation and improves reasoning and tool use.",
            "Gemma3 12B is strong per GB and runs comfortably with a large context window.",
            "Mistral Small is a much better alternate reasoner than plain Mistral 7B.",
            "Devstral Small 2 is a stronger agent-style coding worker than the older local stack, with a 384K context window on Ollama.",
        ],
        "recommended_pulls": ["qwen3:8b", "gemma3:12b", "mistral-small:24b", "devstral-small-2:24b", "qwen3:14b"],
    },
    "agentic-max": {
        "label": "Agentic Max",
        "tier": "High-End Workstation",
        "summary": "For large RAM or unified-memory machines. Prioritizes stronger planning and critique quality over speed.",
        "roles": {
            "orchestrator": "qwen3:30b",
            "operator_fast": "qwen3:8b",
            "reasoning_primary": "qwen3:30b",
            "reasoning_secondary": "mistral-small:24b",
            "reasoning_tertiary": "deepseek-r1:32b",
            "coder": "devstral-2:123b",
            "critic": "llama3.3:70b",
        },
        "contexts": {
            "orchestrator": 20000,
            "operator_fast": 8192,
            "reasoning_primary": 24000,
            "reasoning_secondary": 16000,
            "reasoning_tertiary": 20000,
            "coder": 24000,
            "critic": 12000,
        },
        "notes": [
            "Best fit when throughput matters less than decision quality.",
            "Llama3.3 70B is powerful but very heavy, so use it only if you can actually host it.",
            "Devstral 2 is the strongest local agentic coding upgrade in this stack if your machine can host it.",
            "DeepSeek-R1 can still be tested as a specialist, but it stays off the default hot path until it proves stable in your runtime.",
        ],
        "recommended_pulls": ["qwen3:30b", "qwen3:8b", "mistral-small:24b", "devstral-2:123b", "llama3.3:70b", "deepseek-r1:32b"],
    },
    "coding-lab": {
        "label": "Coding Lab",
        "tier": "Focused SWE Agent Stack",
        "summary": "A coding-heavy setup for long repo work, debugging loops, and autonomous patch generation.",
        "roles": {
            "orchestrator": "qwen3:14b",
            "operator_fast": "qwen3:8b",
            "reasoning_primary": "qwen3:14b",
            "reasoning_secondary": "gemma3:12b",
            "reasoning_tertiary": "devstral:24b",
            "coder": "qwen3-coder:30b",
            "critic": "devstral:24b",
        },
        "contexts": {
            "orchestrator": 16384,
            "operator_fast": 8192,
            "reasoning_primary": 16384,
            "reasoning_secondary": 12288,
            "reasoning_tertiary": 12288,
            "coder": 24576,
            "critic": 12288,
        },
        "notes": [
            "Qwen3-Coder remains the best repo-scale coding specialist on Ollama for many code-generation tasks.",
            "Devstral 24B is a strong reviewer and tool-using critic when you want the coder and critic to disagree productively.",
            "This is the most practical upgrade if the main bottleneck is code synthesis instead of raw planning.",
        ],
        "recommended_pulls": ["qwen3:14b", "qwen3:8b", "gemma3:12b", "qwen3-coder:30b", "devstral:24b"],
    },
    "your-local-max": {
        "label": "Your Local Max",
        "tier": "Installed Heavy Stack",
        "summary": "Uses the stronger models you already have locally, with contexts tuned for quality without driving the autonomy loop into unnecessary slowdowns.",
        "roles": {
            "orchestrator": "qwen3.6:latest",
            "operator_fast": "qwen2.5",
            "reasoning_primary": "gemma4:31b",
            "reasoning_secondary": "mistral:7b-text-fp16",
            "reasoning_tertiary": "qwen3.6:latest",
            "coder": "deepseek-coder-v2:16b",
            "critic": "llama3.3:70b",
        },
        "contexts": {
            "orchestrator": 16384,
            "operator_fast": 6144,
            "reasoning_primary": 24576,
            "reasoning_secondary": 8192,
            "reasoning_tertiary": 12288,
            "coder": 16384,
            "critic": 12288,
        },
        "notes": [
            "Best practical profile from the models you listed as already installed.",
            "Qwen3.6 is the strongest front-door choice here for orchestration and tertiary synthesis.",
            "Gemma4 31B is a strong primary thinker and benefits from a larger context than the small models.",
            "Mistral 7B FP16 is fine as a fast alternate view, but it should not be the main planner.",
            "Llama3.3 70B is best reserved for critique unless your hardware is strong enough to run it in multiple roles comfortably.",
        ],
        "recommended_pulls": [],
    },
}


ROLE_LABELS = {
    "orchestrator": "Orchestrator",
    "operator_fast": "Fast Operator Link",
    "reasoning_primary": "Primary Reasoning",
    "reasoning_secondary": "Alternate Reasoning",
    "reasoning_tertiary": "Third Reasoner",
    "coder": "Coder",
    "critic": "Critic",
}


MODEL_CONTEXT_DEFAULTS = {
    "qwen2.5": 8192,
    "qwen2.5-coder": 8192,
    "qwen3": 16384,
    "qwen3.6": 16384,
    "qwen3-coder": 24576,
    "mistral": 8192,
    "mistral-small": 12288,
    "gemma3": 16384,
    "gemma4": 24576,
    "devstral": 12288,
    "devstral-small-2": 16384,
    "devstral-2": 24576,
    "llama3": 6144,
    "llama3.3": 12288,
    "deepseek-r1": 12288,
    "deepseek-coder-v2": 16384,
}


ROLE_CONTEXT_FALLBACKS = {
    "orchestrator": 8192,
    "operator_fast": 6144,
    "reasoning_primary": 12288,
    "reasoning_secondary": 8192,
    "reasoning_tertiary": 8192,
    "coder": 8192,
    "critic": 6144,
}


def normalize_model_name(model_name: str) -> str:
    return str(model_name or "").strip().split(":", 1)[0].lower()


def default_context_for_model(model_name: str, role: str | None = None) -> int:
    normalized = normalize_model_name(model_name)
    if normalized in MODEL_CONTEXT_DEFAULTS:
        return MODEL_CONTEXT_DEFAULTS[normalized]
    return ROLE_CONTEXT_FALLBACKS.get(str(role or "").strip(), 8192)


def env_updates_for_profile(profile_key: str) -> dict[str, str]:
    profile = MODEL_PROFILES[profile_key]
    roles = profile["roles"]
    contexts = profile["contexts"]
    return {
        "ARC_MODEL_ORCHESTRATOR": roles["orchestrator"],
        "ARC_MODEL_OPERATOR_FAST": roles["operator_fast"],
        "ARC_MODEL_REASONING_PRIMARY": roles["reasoning_primary"],
        "ARC_MODEL_REASONING_SECONDARY": roles["reasoning_secondary"],
        "ARC_MODEL_REASONING_TERTIARY": roles["reasoning_tertiary"],
        "ARC_MODEL_CODER": roles["coder"],
        "ARC_MODEL_CRITIC": roles["critic"],
        "ARC_CTX_ORCHESTRATOR": str(contexts["orchestrator"]),
        "ARC_CTX_OPERATOR_FAST": str(contexts["operator_fast"]),
        "ARC_CTX_REASONING_PRIMARY": str(contexts["reasoning_primary"]),
        "ARC_CTX_REASONING_SECONDARY": str(contexts["reasoning_secondary"]),
        "ARC_CTX_REASONING_TERTIARY": str(contexts["reasoning_tertiary"]),
        "ARC_CTX_CODER": str(contexts["coder"]),
        "ARC_CTX_CRITIC": str(contexts["critic"]),
    }
