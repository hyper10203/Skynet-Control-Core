"""Self-improvement system for the ARC agent.

This module enables the AI to analyze its own performance and propose
code improvements to achieve better efficiency and higher scores.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import (
    MODEL_REGISTRY,
    NEUROGOLF_AUTONOMY_REPORTS_DIR,
    PROJECT_ROOT,
)
from core.executor import ask_with_fallback
from core.prompts import SELF_IMPROVEMENT_SYSTEM_PROMPT

SELF_IMPROVEMENT_LOG_PATH = NEUROGOLF_AUTONOMY_REPORTS_DIR / "self_improvement.jsonl"
SELF_IMPROVEMENT_ENABLED_PATH = NEUROGOLF_AUTONOMY_REPORTS_DIR / "self_improvement_enabled"
MAX_SELF_IMPROVEMENT_CYCLES = 3
MIN_PERFORMANCE_THRESHOLD = 0.7  # Minimum success rate to trigger improvement


def _daemon_log(message: str) -> None:
    """Write to daemon log for debugging."""
    # Avoid circular import by writing directly
    from datetime import datetime, timezone
    log_path = NEUROGOLF_AUTONOMY_REPORTS_DIR / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] self_improvement:{message}\n")


def is_self_improvement_enabled() -> bool:
    """Check if self-improvement is enabled."""
    return SELF_IMPROVEMENT_ENABLED_PATH.exists()


def enable_self_improvement() -> None:
    """Enable the self-improvement system."""
    SELF_IMPROVEMENT_ENABLED_PATH.touch()
    _daemon_log("Self-improvement enabled")


def disable_self_improvement() -> None:
    """Disable the self-improvement system."""
    if SELF_IMPROVEMENT_ENABLED_PATH.exists():
        SELF_IMPROVEMENT_ENABLED_PATH.unlink()
    _daemon_log("Self-improvement disabled")


def load_improvement_history(limit: int = 50) -> list[dict]:
    """Load self-improvement history from log."""
    if not SELF_IMPROVEMENT_LOG_PATH.exists():
        return []
    lines = SELF_IMPROVEMENT_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
    records = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def log_improvement_attempt(
    analysis: str,
    proposed_changes: list[dict],
    validation_result: dict | None,
    applied: bool,
    reason: str | None = None,
) -> None:
    """Log an improvement attempt."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis": analysis,
        "proposed_changes": proposed_changes,
        "validation_result": validation_result,
        "applied": applied,
        "reason": reason,
    }
    SELF_IMPROVEMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SELF_IMPROVEMENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def analyze_recent_performance(reports: list[dict]) -> dict[str, Any]:
    """Analyze recent performance to identify improvement opportunities."""
    if not reports:
        return {"status": "no_data", "recommendation": "Need more reports to analyze"}

    total = len(reports)
    successes = sum(1 for r in reports if (r.get("submit_result") or {}).get("submitted"))
    failures = total - successes

    # Analyze score trends
    scores = []
    for r in reports:
        plan = r.get("plan", {})
        if plan.get("estimated_improvement"):
            try:
                scores.append(float(plan["estimated_improvement"]))
            except (ValueError, TypeError):
                pass

    avg_score_improvement = sum(scores) / len(scores) if scores else 0
    recent_trend = "improving" if len(scores) >= 3 and scores[-1] > scores[0] else "stable"

    # CRITICAL: Check for V3 validation failures (dynamic shapes, invalid V2 files)
    v3_validation_failures = []
    v2_file_failures = []
    for r in reports:
        submit_result = r.get("submit_result", {})
        reason = str(submit_result.get("reason", ""))
        if "V3_VALIDATION_FAILED" in reason:
            v3_validation_failures.append(r)
        elif "invalid_base_tasks" in str(submit_result.get("v3_validation", {})):
            v2_file_failures.append(r)

    # Identify patterns
    patterns = {
        "total_cycles": total,
        "success_rate": successes / total if total > 0 else 0,
        "avg_score_improvement": avg_score_improvement,
        "recent_trend": recent_trend,
        "failures": failures,
        "v3_validation_failures": len(v3_validation_failures),
        "v2_file_failures": len(v2_file_failures),
    }

    # Determine if improvement is needed
    # CRITICAL: V3 validation failures are high priority to fix
    needs_improvement = (
        patterns["success_rate"] < MIN_PERFORMANCE_THRESHOLD
        or patterns["recent_trend"] != "improving"
        or patterns["failures"] > 2
        or len(v3_validation_failures) > 0  # Any V3 validation failure needs fixing
        or len(v2_file_failures) > 0  # V2 files need to be rebuilt
    )

    # Build specific recommendation based on failure type
    recommendation = "Continue current strategy"
    if v3_validation_failures or v2_file_failures:
        recommendation = "CRITICAL: System is submitting V2-era invalid files. Need to implement V3-compliant file generation for dynamic-shape tasks."
    elif needs_improvement:
        recommendation = "Analyze codebase for optimization opportunities"

    return {
        "status": "needs_improvement" if needs_improvement else "performing_well",
        "patterns": patterns,
        "recommendation": recommendation,
    }


def get_core_modules_for_analysis() -> list[tuple[str, str]]:
    """Get core module source code for analysis."""
    modules = []
    core_dir = PROJECT_ROOT / "core"
    agent_dir = PROJECT_ROOT / "agents"

    key_files = [
        core_dir / "solver_loop.py",
        core_dir / "neurogolf_context.py",
        agent_dir / "coder.py",
        agent_dir / "reasoning.py",
        agent_dir / "orchestrator.py",
        PROJECT_ROOT / "autonomous_neurogolf.py",
    ]

    for path in key_files:
        if path.exists():
            try:
                source = path.read_text(encoding="utf-8")
                modules.append((str(path.relative_to(PROJECT_ROOT)), source))
            except Exception:
                pass

    return modules


def validate_python_code(code: str, filename: str = "<unknown>") -> dict:
    """Validate Python code for syntax errors."""
    try:
        ast.parse(code, filename=filename)
        return {"valid": True, "errors": []}
    except SyntaxError as e:
        return {
            "valid": False,
            "errors": [f"Syntax error at line {e.lineno}: {e.msg}"],
        }


def apply_code_change(file_path: Path, original: str, replacement: str, reason: str) -> dict:
    """Apply a code change to a file."""
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        content = file_path.read_text(encoding="utf-8")

        # Safety check: verify original exists exactly once
        if content.count(original) != 1:
            return {
                "success": False,
                "error": f"Original text appears {content.count(original)} times (expected 1)",
            }

        # Validate replacement code
        validation = validate_python_code(replacement, str(file_path))
        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Replacement code has syntax errors: {validation['errors']}",
            }

        # Create backup
        backup_path = file_path.with_suffix(f".py.backup.{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy2(file_path, backup_path)

        # Apply change
        new_content = content.replace(original, replacement, 1)
        file_path.write_text(new_content, encoding="utf-8")

        # Test import
        try:
            # Try to import the module to catch runtime errors
            spec = __import__("importlib.util").util.spec_from_file_location(
                "test_module", file_path
            )
            if spec and spec.loader:
                module = __import__("importlib.util").util.module_from_spec(spec)
                spec.loader.exec_module(module)
        except Exception as e:
            # Rollback on import error
            shutil.copy2(backup_path, file_path)
            return {
                "success": False,
                "error": f"Import test failed: {e}. Changes rolled back.",
            }

        return {
            "success": True,
            "backup_path": str(backup_path),
            "reason": reason,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_improvement_proposal(
    performance_analysis: dict,
    module_sources: list[tuple[str, str]],
) -> dict:
    """Ask the AI to propose code improvements."""
    modules_summary = "\n\n".join([
        f"=== {name} ===\n{source[:3000]}..." if len(source) > 3000 else f"=== {name} ===\n{source}"
        for name, source in module_sources[:3]  # Limit to first 3 modules
    ])

    prompt = f"""Analyze the following performance data and source code, then propose specific code improvements.

Performance Analysis:
- Status: {performance_analysis.get('status')}
- Patterns: {json.dumps(performance_analysis.get('patterns', {}), indent=2)}
- Recommendation: {performance_analysis.get('recommendation')}

Source Code (key modules):
{modules_summary}

Based on this analysis, propose concrete code improvements that would:
1. Increase success rate of submissions
2. Improve efficiency (faster processing, better parameter counting)
3. Better adapt to the April 28 metric changes (static shapes required)

Return JSON with this structure:
{{
    "analysis_summary": "Brief analysis of what needs improvement",
    "proposed_changes": [
        {{
            "file": "relative/path/to/file.py",
            "original": "exact code to replace (must match exactly)",
            "replacement": "new code to insert",
            "reason": "why this change improves performance"
        }}
    ],
    "confidence": 0.8,  // 0-1 confidence in these changes
    "expected_impact": "description of expected performance improvement"
}}

IMPORTANT:
- Original code must match exactly (including whitespace)
- Proposed changes must be syntactically valid Python
- Focus on high-impact, low-risk improvements
- Do not change core logic dramatically - optimize existing approaches
"""

    response = ask_with_fallback(
        model=MODEL_REGISTRY.get("orchestrator", "llama3.1"),
        prompt=prompt,
        system=SELF_IMPROVEMENT_SYSTEM_PROMPT,
    )

    # Extract JSON from response
    try:
        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "No valid JSON found in response", "raw_response": response}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw_response": response}


def run_self_improvement_cycle(reports: list[dict]) -> dict:
    """Run one self-improvement cycle."""
    if not is_self_improvement_enabled():
        return {"status": "disabled", "reason": "Self-improvement not enabled"}

    # Check recent improvement attempts to avoid loops
    history = load_improvement_history(limit=10)
    recent_attempts = [h for h in history if h.get("applied")]
    if len(recent_attempts) >= MAX_SELF_IMPROVEMENT_CYCLES:
        return {
            "status": "throttled",
            "reason": f"Max {MAX_SELF_IMPROVEMENT_CYCLES} improvements reached recently",
        }

    # Analyze performance
    analysis = analyze_recent_performance(reports)
    if analysis.get("status") != "needs_improvement":
        return {
            "status": "no_action",
            "reason": "Performance is acceptable, no improvements needed",
            "analysis": analysis,
        }

    # Get source code
    modules = get_core_modules_for_analysis()
    if not modules:
        return {"status": "error", "reason": "Could not load source modules"}

    # Generate improvement proposal
    proposal = generate_improvement_proposal(analysis, modules)
    if proposal.get("error"):
        log_improvement_attempt(
            analysis=str(analysis),
            proposed_changes=[],
            validation_result=None,
            applied=False,
            reason=f"Proposal generation failed: {proposal.get('error')}",
        )
        return {"status": "error", "reason": proposal.get("error")}

    # Validate and apply changes
    changes = proposal.get("proposed_changes", [])
    if not changes:
        return {
            "status": "no_changes",
            "reason": "No changes proposed",
            "analysis": analysis,
        }

    # Apply changes one by one
    applied_changes = []
    failed_changes = []

    for change in changes:
        file_path = PROJECT_ROOT / change["file"]
        result = apply_code_change(
            file_path,
            change["original"],
            change["replacement"],
            change.get("reason", "Self-improvement"),
        )

        if result["success"]:
            applied_changes.append({
                "file": change["file"],
                "reason": change.get("reason"),
                "backup": result.get("backup_path"),
            })
        else:
            failed_changes.append({
                "file": change["file"],
                "error": result.get("error"),
            })

    # Log the attempt
    log_improvement_attempt(
        analysis=proposal.get("analysis_summary", ""),
        proposed_changes=changes,
        validation_result={"applied": applied_changes, "failed": failed_changes},
        applied=len(applied_changes) > 0,
        reason=f"Applied {len(applied_changes)} changes, failed {len(failed_changes)}" if applied_changes else "No changes applied",
    )

    return {
        "status": "completed" if applied_changes else "failed",
        "applied_changes": applied_changes,
        "failed_changes": failed_changes,
        "analysis": analysis,
        "proposal": proposal,
    }


def get_improvement_status() -> dict:
    """Get current self-improvement status for UI."""
    history = load_improvement_history(limit=20)
    recent_applied = [h for h in history if h.get("applied")]

    return {
        "enabled": is_self_improvement_enabled(),
        "total_attempts": len(history),
        "recent_improvements": len(recent_applied),
        "max_cycles": MAX_SELF_IMPROVEMENT_CYCLES,
        "last_attempt": history[-1] if history else None,
    }
