"""Local NeuroGolf submission validation.

This module mirrors the practical checks from the official NeuroGolf utility:
static ONNX graphs, no banned operators, per-file size limit, and optional
cost profiling through onnx_tool.
"""

from __future__ import annotations

import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any


EXPECTED_TASK_NAMES = {f"task{i:03d}.onnx" for i in range(1, 401)}
OPTIONAL_TASK_NAMES = {"task000.onnx"}
EXCLUDED_OP_TYPES = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION"}
FILESIZE_LIMIT_IN_BYTES = int(1.44 * 1024 * 1024)


def _static_dim_issue(name: str, index: int, dim: Any) -> str | None:
    try:
        if dim.HasField("dim_param") and dim.dim_param:
            return f"{name}[{index}] has symbolic dimension {dim.dim_param!r}"
        if not dim.HasField("dim_value"):
            return f"{name}[{index}] has unknown dimension"
        if int(dim.dim_value) <= 0:
            return f"{name}[{index}] has non-positive dimension {int(dim.dim_value)}"
    except Exception as exc:
        return f"{name}[{index}] dimension could not be inspected: {exc}"
    return None


def _shape_issues(value_info: Any) -> list[str]:
    issues: list[str] = []
    name = str(getattr(value_info, "name", "unnamed"))
    value_type = getattr(value_info, "type", None)
    if value_type is None:
        return issues
    try:
        if not value_type.HasField("tensor_type"):
            return issues
        tensor_type = value_type.tensor_type
        if not tensor_type.HasField("shape"):
            issues.append(f"{name} has no static shape")
            return issues
        for index, dim in enumerate(tensor_type.shape.dim):
            issue = _static_dim_issue(name, index, dim)
            if issue:
                issues.append(issue)
    except Exception as exc:
        issues.append(f"{name} shape could not be inspected: {exc}")
    return issues


def _profile_model(path: Path) -> dict[str, Any]:
    try:
        import onnx_tool  # type: ignore
    except Exception as exc:
        return {"available": False, "reason": f"onnx_tool unavailable: {exc}"}

    try:
        model = onnx_tool.loadmodel(str(path), {"verbose": False, "constant_folding": True})
        graph = model.graph
        graph.graph_reorder_nodes()
        graph.shape_infer(None)
        graph.profile()
        if not graph.valid_profile:
            return {"available": True, "valid": False, "reason": "onnx_tool profile invalid"}
        macs = int(sum(graph.macs))
        memory = int(graph.memory)
        params = int(graph.params)
        cost = max(1, macs + memory + params)
        return {
            "available": True,
            "valid": True,
            "macs": macs,
            "memory": memory,
            "params": params,
            "cost": cost,
            "points": max(1.0, 25.0 - math.log(cost)),
        }
    except Exception as exc:
        return {"available": True, "valid": False, "reason": str(exc)}


def validate_onnx_file(path: Path) -> dict[str, Any]:
    """Validate one ONNX file against NeuroGolf submission constraints."""
    result: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "valid": True,
        "errors": [],
        "warnings": [],
        "size_bytes": None,
        "op_count": None,
        "profile": None,
    }

    if not path.exists():
        result["valid"] = False
        result["errors"].append("file missing")
        return result

    size_bytes = path.stat().st_size
    result["size_bytes"] = size_bytes
    if size_bytes > FILESIZE_LIMIT_IN_BYTES:
        result["errors"].append(
            f"file size {size_bytes} exceeds {FILESIZE_LIMIT_IN_BYTES} byte NeuroGolf limit"
        )

    try:
        import onnx  # type: ignore
        from onnx import shape_inference  # type: ignore
    except Exception as exc:
        result["errors"].append(f"onnx package unavailable: {exc}")
        result["valid"] = False
        return result

    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model, full_check=True)
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        result["errors"].append(f"onnx check or static shape inference failed: {exc}")
        result["valid"] = False
        return result

    graph = inferred.graph
    op_types = [str(node.op_type).upper() for node in graph.node]
    result["op_count"] = len(op_types)
    banned = sorted(set(op_types) & EXCLUDED_OP_TYPES)
    if banned:
        result["errors"].append(f"banned ONNX operators present: {', '.join(banned)}")

    shape_items = list(graph.input) + list(graph.output) + list(graph.value_info)
    for item in shape_items:
        result["errors"].extend(_shape_issues(item))

    profile = _profile_model(path)
    result["profile"] = profile
    if profile.get("available") and not profile.get("valid", True):
        result["errors"].append(f"profile failed: {profile.get('reason')}")

    result["valid"] = not result["errors"]
    return result


def validate_submission_zip(zip_path: Path, *, require_all_tasks: bool = True) -> dict[str, Any]:
    """Validate a submission zip and return a compact machine-readable report."""
    report: dict[str, Any] = {
        "zip_path": str(zip_path),
        "valid": True,
        "errors": [],
        "warnings": [],
        "files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "missing_tasks": [],
        "unexpected_files": [],
        "invalid_task_reports": [],
        "score_estimate": None,
    }

    if not zip_path.exists():
        report["valid"] = False
        report["errors"].append(f"zip missing: {zip_path}")
        return report

    with tempfile.TemporaryDirectory(prefix="neurogolf_validate_") as tmpdir:
        tmp_root = Path(tmpdir)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                members = [member for member in archive.namelist() if not member.endswith("/")]
                task_members = []
                task_names = set()
                for member in members:
                    name = Path(member).name
                    if not name.lower().endswith(".onnx"):
                        report["unexpected_files"].append(member)
                        continue
                    if not name.startswith("task"):
                        report["unexpected_files"].append(member)
                        continue
                    task_members.append(member)
                    task_names.add(name)
                if require_all_tasks:
                    missing = sorted(EXPECTED_TASK_NAMES - task_names)
                    report["missing_tasks"] = missing
                    if missing:
                        report["errors"].append(f"missing {len(missing)} required task files")
                for name in sorted(task_names - EXPECTED_TASK_NAMES - OPTIONAL_TASK_NAMES):
                    report["unexpected_files"].append(name)

                total_points = 0.0
                profiled_files = 0
                for member in task_members:
                    extracted = archive.extract(member, tmp_root)
                    validation = validate_onnx_file(Path(extracted))
                    validation["archive_member"] = member
                    report["files"] += 1
                    if validation["valid"]:
                        report["valid_files"] += 1
                        profile = validation.get("profile") or {}
                        if profile.get("valid") and profile.get("points") is not None:
                            total_points += float(profile["points"])
                            profiled_files += 1
                    else:
                        report["invalid_files"] += 1
                        compact = {
                            "name": validation.get("name"),
                            "archive_member": member,
                            "errors": validation.get("errors", [])[:8],
                            "size_bytes": validation.get("size_bytes"),
                            "op_count": validation.get("op_count"),
                        }
                        report["invalid_task_reports"].append(compact)
        except zipfile.BadZipFile as exc:
            report["errors"].append(f"bad zip file: {exc}")

    if report["unexpected_files"]:
        report["warnings"].append(f"{len(report['unexpected_files'])} unexpected/non-task files in zip")
    if report["invalid_files"]:
        report["errors"].append(f"{report['invalid_files']} task files failed ONNX validation")
    if profiled_files:
        report["score_estimate"] = {
            "profiled_files": profiled_files,
            "total_points": total_points,
            "avg_points": total_points / profiled_files,
        }

    report["valid"] = not report["errors"]
    return report


def write_validation_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
