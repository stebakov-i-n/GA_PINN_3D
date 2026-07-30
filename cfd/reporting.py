from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import save_resolved_config


VALID_STATUSES = {
    "generated_not_executed",
    "failed",
    "mesh_failed",
    "mesh_validated",
    "executed_not_converged",
    "converged_validated",
}


def create_run_dir(output_root: str | Path, case: str, direction: str, config: dict[str, Any]) -> Path:
    root = Path(output_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{stamp}_{case}_{direction}_{config['resolved']['config_hash']}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}_{case}_{direction}_{config['resolved']['config_hash']}_{suffix}"
    for child in ("figures", "logs", "geometry", "openfoam_case", "solution"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def start_metadata(config: dict[str, Any], command: list[str], case: str, direction: str) -> dict[str, Any]:
    return {
        "case": case,
        "direction": direction,
        "command": command,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "physics": {
            "density_kg_m3": config["fluid"]["density_kg_m3"],
            "dynamic_viscosity_pa_s": config["fluid"]["dynamic_viscosity_pa_s"],
            "kinematic_viscosity_m2_s": config["resolved"]["kinematic_viscosity_m2_s"],
            "mean_inlet_velocity_m_s": config["flow"]["mean_inlet_velocity_m_s"],
            "outlet_gauge_pressure_pa": config["boundary_conditions"]["outlet_gauge_pressure_pa"],
        },
        "openfoam": config["openfoam"],
        "_start_time_monotonic": time.monotonic(),
    }


def finish_run(
    run_dir: str | Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    status: str,
    failure_reason: str | None = None,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Unsupported run status: {status}")
    run = Path(run_dir)
    runtime_s = time.monotonic() - metadata.pop("_start_time_monotonic", time.monotonic())
    metadata.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_s": runtime_s,
            "status": status,
            "failure_reason": failure_reason,
            "run_directory": str(run),
        }
    )
    save_resolved_config(config, run / "resolved_config.yaml")
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True), encoding="utf-8")
    (run / "report.md").write_text(render_report(metadata, metrics), encoding="utf-8")


def render_report(metadata: dict[str, Any], metrics: dict[str, Any]) -> str:
    status = metadata["status"]
    lines = [
        f"# CFD Run Report",
        "",
        f"Status: **{status}**",
        f"Case: `{metadata['case']}`",
        f"Direction: `{metadata['direction']}`",
        f"Command: `{' '.join(metadata['command'])}`",
        f"Runtime: {metadata['runtime_s']:.3f} s",
        "",
        "## Physics",
        "",
        f"- density_kg_m3: {metadata['physics']['density_kg_m3']}",
        f"- dynamic_viscosity_pa_s: {metadata['physics']['dynamic_viscosity_pa_s']}",
        f"- kinematic_viscosity_m2_s: {metadata['physics']['kinematic_viscosity_m2_s']}",
        f"- mean_inlet_velocity_m_s: {metadata['physics']['mean_inlet_velocity_m_s']}",
        f"- outlet_gauge_pressure_pa: {metadata['physics']['outlet_gauge_pressure_pa']}",
        "",
        "## Git",
        "",
        f"- commit: {metadata['git']['commit']}",
        f"- dirty_tree: {metadata['git']['dirty_tree']}",
    ]
    if metadata.get("failure_reason"):
        lines += ["", "## Failure", "", metadata["failure_reason"]]
    if "analytical" in metrics:
        lines += ["", "## Analytical Metrics", ""]
        lines += [f"- {k}: {v}" for k, v in metrics["analytical"].items()]
    if "geometry_validation" in metrics:
        gv = metrics["geometry_validation"]
        lines += ["", "## Geometry Validation", "", f"- status: {gv.get('status')}"]
        lines += [f"- warning: {w}" for w in gv.get("warnings", [])]
        lines += [f"- failure: {f}" for f in gv.get("failures", [])]
    figure_lines = []
    if "figures" in metrics:
        figure_lines += [f"- {name}: `{path}`" for name, path in metrics["figures"].items()]
    if "geometry_validation" in metrics:
        figure_lines += [f"- {name}: `{path}`" for name, path in metrics["geometry_validation"].get("figures", {}).items()]
    if "openfoam" in metrics:
        of = metrics["openfoam"]
        lines += ["", "## OpenFOAM", "", f"- status: {of.get('status')}", f"- reason: {of.get('reason', '')}"]
        lines += [f"- runner: {of.get('runner', 'not_run')}"]
        lines += [f"- pressure_convention: {of.get('pressure_convention', 'kinematic pressure; physical gauge pressure p_pa = rho * p')}"]
        if "patched_inlet_faces" in of:
            lines += [f"- patched_inlet_faces: {of['patched_inlet_faces']}"]
        if "latest_time" in of:
            lines += [f"- latest_time: {of['latest_time']}", f"- solution_cells: {of.get('solution_cells')}"]
        if "case_dir" in of:
            lines += [f"- case_dir: `{of['case_dir']}`"]
        if "surface_mapping" in of:
            lines += ["", "### Surface Mapping", ""]
            lines += [f"- {surface}: {patch}" for surface, patch in of["surface_mapping"].items()]
        if "location_in_mesh_m" in of:
            lines += ["", "### Vascular Meshing", ""]
            lines += [f"- location_in_mesh_m: {of['location_in_mesh_m']}"]
            lines += [f"- background_bounds_m: {of.get('background_bounds_m')}"]
        if "inlet_profile" in of:
            profile = of["inlet_profile"]
            lines += ["", "### Inlet Profile", ""]
            lines += [
                f"- profile: {profile.get('profile')}",
                f"- velocity_m_s: {profile.get('velocity_m_s')}",
                f"- confidence: {profile.get('confidence')}",
                f"- normal_alignment: {profile.get('normal_alignment')}",
            ]
        if "mesh_patches" in of:
            lines += ["", "### Mesh Patches", ""]
            for patch, info in of["mesh_patches"].items():
                lines += [f"- {patch}: type={info.get('type')}, nFaces={info.get('nFaces')}, startFace={info.get('startFace')}"]
        if "velocity_m_s" in of:
            lines += ["", "### Velocity [m/s]", ""]
            lines += [f"- {k}: {v}" for k, v in of["velocity_m_s"].items()]
        if "pressure_pa" in of:
            lines += ["", "### Physical Pressure [Pa]", ""]
            lines += [f"- {k}: {v}" for k, v in of["pressure_pa"].items()]
        if "residuals" in of:
            residuals = of["residuals"]
            lines += ["", "### Solver Diagnostics", ""]
            lines += [
                f"- completed: {residuals.get('completed')}",
                f"- final_time: {residuals.get('final_time')}",
                f"- final_initial_residuals: {residuals.get('final_initial_residuals')}",
                f"- final_continuity_sum_local: {residuals.get('final_continuity_sum_local')}",
                f"- final_continuity_global: {residuals.get('final_continuity_global')}",
                f"- final_continuity_cumulative: {residuals.get('final_continuity_cumulative')}",
            ]
        if "mass_balance" in of:
            mb = of["mass_balance"]
            lines += ["", "### Mass Balance", ""]
            lines += [
                f"- status: {mb.get('status')}",
                f"- inlet_flow_rate_m3_s: {mb.get('inlet_flow_rate_m3_s')}",
                f"- outlet_flow_rate_m3_s: {mb.get('outlet_flow_rate_m3_s')}",
                f"- net_flow_rate_m3_s: {mb.get('net_flow_rate_m3_s')}",
                f"- relative_imbalance: {mb.get('relative_imbalance')}",
            ]
            if mb.get("reason"):
                lines += [f"- reason: {mb.get('reason')}"]
        if "node_field_report" in of:
            node_report = of["node_field_report"]
            lines += ["", "### Cell-Centre Field Report", ""]
            lines += [
                f"- kind: {node_report.get('kind')}",
                f"- cell_count: {node_report.get('cell_count')}",
                f"- full_csv: `{node_report.get('full_csv')}`",
                f"- preview_csv: `{node_report.get('preview_csv')}`",
                f"- fields: {', '.join(node_report.get('fields', []))}",
                f"- PINN comparison: {node_report.get('pinn_comparison_key')}",
            ]
        if "figures" in of:
            figure_lines += [f"- {name}: `{path}`" for name, path in of["figures"].items()]
        if "pinn_comparison" in of:
            pc = of["pinn_comparison"]
            lines += ["", "## PINN Comparison", "", f"- status: {pc.get('status')}", f"- reason: {pc.get('reason', '')}"]
        if "stages" in of:
            lines += ["", "## Logs", ""]
            for stage, summary in of["stages"].items():
                lines += [f"- {stage}: returncode={summary.get('returncode')}, stdout=`{summary.get('stdout_log')}`, stderr=`{summary.get('stderr_log')}`"]
    if figure_lines:
        insert_at = len(lines)
        for idx, line in enumerate(lines):
            if line == "## PINN Comparison" or line == "## Logs":
                insert_at = idx - 1
                break
        lines[insert_at:insert_at] = ["", "## Figures", "", *figure_lines]
    return "\n".join(lines) + "\n"


def git_metadata() -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"]) or "unknown"
    status = _git(["status", "--short"]) or ""
    return {"commit": commit, "dirty_tree": bool(status.strip())}


def _git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
