from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import validate_bundle
from .openfoam import generate_vascular_openfoam_case, run_vascular_mesh_pipeline, run_vascular_solve_pipeline
from .reporting import create_run_dir, finish_run, start_metadata


def run_vessel(
    config: dict[str, Any],
    geometry: str,
    direction: str,
    output_root: str | Path,
    command: list[str],
    mode: str,
) -> Path:
    run_dir = create_run_dir(output_root, geometry.replace("\\", "_").replace("/", "_").replace(".stl", ""), direction, config)
    metadata = start_metadata(config, command, geometry.replace("\\", "/"), direction)
    metrics: dict[str, Any] = {}
    status = "generated_not_executed"
    failure: str | None = None
    try:
        validation = validate_bundle(config["geometry"]["dataset_root"], geometry, direction, config, run_dir / "figures")
        metrics["geometry_validation"] = validation
        if validation["failures"]:
            status = "failed"
            failure = "; ".join(validation["failures"])
        elif mode == "validate_only":
            status = "generated_not_executed"
        else:
            case_info = generate_vascular_openfoam_case(
                run_dir / "openfoam_case",
                config["geometry"]["dataset_root"],
                geometry,
                direction,
                config,
                validation,
            )
            metrics["openfoam"] = case_info
            status = case_info["status"]
            if mode in {"mesh_openfoam", "run_openfoam"}:
                mesh_result = run_vascular_mesh_pipeline(run_dir / "openfoam_case", run_dir / "logs", config)
                metrics["openfoam"].update(mesh_result)
                status = mesh_result["status"]
                failure = mesh_result.get("failure_reason")
                if mode == "run_openfoam" and status == "mesh_validated":
                    mesh_stages = dict(metrics["openfoam"].get("stages", {}))
                    solve_result = run_vascular_solve_pipeline(run_dir / "openfoam_case", run_dir / "logs", run_dir / "figures", config)
                    metrics["openfoam"].update(solve_result)
                    if "stages" in solve_result:
                        metrics["openfoam"]["stages"] = {**mesh_stages, **solve_result["stages"]}
                    status = solve_result["status"]
                    failure = solve_result.get("failure_reason")
    except Exception as exc:
        status = "failed"
        failure = str(exc)
        metrics["failure"] = failure
    finish_run(run_dir, config, metadata, metrics, status, failure)
    return run_dir
