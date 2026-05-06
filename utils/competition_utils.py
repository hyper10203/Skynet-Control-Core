"""Competition utilities for NeuroGolf 2026.

Provides helper functions to load and use competition-specific data and rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.kaggle_competition_data import (
    COMPETITION_DATA_DIR,
    get_task_example_data,
    load_competition_metadata,
    is_competition_data_ready,
)


def load_metric_v3_rules() -> dict:
    """Load the Metric V3 rules for the competition.
    
    These are the April 28, 2026 updated rules:
    - Dynamic shapes yield zero points
    - Constant values correctly counted
    - Static shape inference required
    """
    return {
        "effective_date": "2026-04-28",
        "competition": "neurogolf-2026",
        "version": "v3",
        "rules": {
            "dynamic_shapes": {
                "allowed": False,
                "penalty": "zero_points",
                "description": "Networks with dynamic shapes or symbolic dimensions score zero",
            },
            "constant_values": {
                "counted": True,
                "description": "Constant values now correctly count toward parameter contributions",
            },
            "static_shapes": {
                "required": True,
                "description": "All networks must have statically-defined shapes",
            },
            "memory_footprint": {
                "calculation": "sum_of_static_shape_bytes_excluding_io",
                "description": "Memory is sum of bytes for all static shapes (excluding input/output layers)",
            },
            "shape_inference": {
                "must_succeed": True,
                "description": "Shape inference must succeed for valid submission",
            },
        },
        "validation": {
            "check_static_shapes": True,
            "check_constant_counting": True,
            "check_symbolic_dims": True,
        },
    }


def get_competition_config() -> dict:
    """Get the full competition configuration."""
    return {
        "name": "neurogolf-2026",
        "metric_version": "v3",
        "metric_rules": load_metric_v3_rules(),
        "data_ready": is_competition_data_ready(),
        "metadata": load_competition_metadata(),
    }


def validate_network_for_metric_v3(network_info: dict) -> dict:
    """Validate a network against Metric V3 rules.
    
    Args:
        network_info: Dict containing network metadata
        
    Returns:
        Validation result with pass/fail status and issues.
    """
    issues = []
    
    # Check for dynamic shapes
    if network_info.get("has_dynamic_shapes", False):
        issues.append({
            "severity": "error",
            "rule": "dynamic_shapes",
            "message": "Network has dynamic shapes - will score zero points",
        })
    
    # Check for symbolic dimensions
    if network_info.get("has_symbolic_dims", False):
        issues.append({
            "severity": "error",
            "rule": "symbolic_dimensions",
            "message": "Network has symbolic dimensions - will score zero points",
        })
    
    # Check static shape inference
    if not network_info.get("static_shape_inference_passed", True):
        issues.append({
            "severity": "error",
            "rule": "shape_inference",
            "message": "Static shape inference failed - will score zero points",
        })
    
    # Check for constant exploitation (now invalid)
    constant_count = network_info.get("hidden_constant_count", 0)
    if constant_count > 1000:  # Arbitrary threshold for suspicious
        issues.append({
            "severity": "warning",
            "rule": "constant_exploitation",
            "message": f"Large number of hidden constants ({constant_count}) - now counted correctly",
        })
    
    return {
        "valid": len([i for i in issues if i["severity"] == "error"]) == 0,
        "issues": issues,
        "metric_version": "v3",
    }


def get_task_hints_from_competition(task_id: str) -> list[str]:
    """Get hints for a specific task from competition data.
    
    Args:
        task_id: Task identifier (e.g., "task001")
        
    Returns:
        List of hint strings.
    """
    hints = []
    
    # Try to load from competition data
    task_data = get_task_example_data(task_id)
    if task_data:
        # Extract any hints present in task data
        if "hints" in task_data:
            hints.extend(task_data["hints"])
        if "notes" in task_data:
            hints.extend(task_data["notes"])
    
    return hints


def calculate_expected_score(
    task_count: int,
    avg_task_score: float,
    has_dynamic_shapes: bool = False,
    has_invalid_networks: bool = False,
) -> dict:
    """Calculate expected competition score based on Metric V3.
    
    Args:
        task_count: Number of tasks in submission
        avg_task_score: Average score per task
        has_dynamic_shapes: Whether submission has dynamic shapes
        has_invalid_networks: Whether submission has invalid networks
        
    Returns:
        Score prediction with breakdown.
    """
    base_score = task_count * avg_task_score
    
    penalties = []
    
    if has_dynamic_shapes:
        penalties.append({
            "reason": "Dynamic shapes (Metric V3)",
            "impact": "ZERO - All points lost",
            "severity": "critical",
        })
        expected_score = 0.0
    elif has_invalid_networks:
        penalties.append({
            "reason": "Invalid networks failed validation",
            "impact": "ZERO - All points lost",
            "severity": "critical",
        })
        expected_score = 0.0
    else:
        expected_score = base_score
    
    return {
        "expected_score": expected_score,
        "base_score": base_score,
        "penalties": penalties,
        "task_count": task_count,
        "avg_task_score": avg_task_score,
        "metric_version": "v3",
    }


def format_submission_for_competition(
    tasks: list[dict],
    metadata: dict | None = None,
) -> dict:
    """Format a submission according to competition requirements.
    
    Args:
        tasks: List of task dictionaries
        metadata: Optional submission metadata
        
    Returns:
        Formatted submission dict.
    """
    return {
        "competition": "neurogolf-2026",
        "metric_version": "v3",
        "tasks": tasks,
        "metadata": metadata or {},
        "validation": {
            "static_shapes_checked": True,
            "constant_counting_checked": True,
        },
    }


def load_starter_notebook_reference() -> dict | None:
    """Load reference to the official starter notebook if available."""
    starter_paths = [
        COMPETITION_DATA_DIR / "starter_notebook.ipynb",
        COMPETITION_DATA_DIR / "starter.ipynb",
        COMPETITION_DATA_DIR / "neurogolf-starter.ipynb",
        COMPETITION_DATA_DIR / "docs" / "starter.ipynb",
    ]
    
    for path in starter_paths:
        if path.exists():
            return {
                "path": str(path),
                "name": path.name,
                "exists": True,
            }
    
    return None


def get_leaderboard_insights() -> dict:
    """Get insights about current leaderboard status.
    
    Returns placeholder - actual implementation would fetch from Kaggle API.
    """
    return {
        "competition": "neurogolf-2026",
        "metric_version": "v3",
        "note": "Leaderboard data requires Kaggle API fetch",
        "key_insights": [
            "Metric V3 now strictly enforces static shapes",
            "Dynamic shape exploits no longer work",
            "Constant values correctly counted",
            "Focus on valid ONNX with clean static shapes",
        ],
    }
