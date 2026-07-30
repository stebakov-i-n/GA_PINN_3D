from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import bundle_paths, cap_properties, read_stl, write_ascii_stl


OPENFOAM_STAGES = ["blockMesh", "checkMesh", "simpleFoam"]
OPENFOAM_MESH_STAGES = [
    ("blockMesh", ["blockMesh"]),
    ("surfaceFeatureExtract", ["surfaceFeatureExtract"]),
    ("snappyHexMesh", ["snappyHexMesh", "-overwrite"]),
    ("checkMesh", ["checkMesh"]),
]
DOCKER_IMAGE = "opencfd/openfoam-run:2606"


def foam_header(class_name: str, object_name: str, location: str | None = None) -> str:
    location_line = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"{location_line}"
        f"    object      {object_name};\n"
        "}\n"
        "\n"
    )


def discover_openfoam(config: dict[str, Any]) -> dict[str, Any]:
    solver = config["openfoam"]["solver"]
    found = {cmd: shutil.which(cmd) for cmd in sorted(set(OPENFOAM_STAGES + [solver] + [stage[0] for stage in OPENFOAM_MESH_STAGES]))}
    required = sorted(set(OPENFOAM_STAGES + [solver]))
    missing = [cmd for cmd in required if found[cmd] is None]
    return {"available": not missing, "commands": found, "missing": missing, "required_version": config["openfoam"]["version"]}


def discover_docker() -> dict[str, Any]:
    docker = shutil.which("docker")
    return {"available": docker is not None, "command": docker, "image": DOCKER_IMAGE}


def generate_openfoam_case(case_dir: str | Path, config: dict[str, Any]) -> None:
    case = Path(case_dir)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant").mkdir(exist_ok=True)
    (case / "0").mkdir(exist_ok=True)
    nu = config["resolved"]["kinematic_viscosity_m2_s"]
    radius = config["cylinder"]["radius_m"]
    length = config["cylinder"]["length_m"]
    end_time = int(config["openfoam"]["end_time"])
    write_interval = int(config["openfoam"]["write_interval"])
    (case / "README.md").write_text(
        f"OpenCFD OpenFOAM {config['openfoam']['version']} cylinder case for {config['openfoam']['solver']}.\n"
        "The synthetic verification geometry is a circular pipe generated with arc edges.\n"
        "The inlet uses the analytical Hagen-Poiseuille profile U_z = 2 U_mean (1 - r^2 / R^2).\n"
        "Pressure is kinematic in the incompressible solver; exported physical pressure must use p_pa = rho * p.\n",
        encoding="utf-8",
    )
    (case / "constant" / "transportProperties").write_text(
        foam_header("dictionary", "transportProperties", "constant")
        + f"transportModel  Newtonian;\nnu [0 2 -1 0 0 0 0] {nu:.12g};\n",
        encoding="ascii",
    )
    (case / "constant" / "turbulenceProperties").write_text(
        foam_header("dictionary", "turbulenceProperties", "constant")
        + "simulationType laminar;\n",
        encoding="ascii",
    )
    (case / "system" / "controlDict").write_text(
        foam_header("dictionary", "controlDict", "system")
        + f"application simpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime {end_time};\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval {write_interval};\n",
        encoding="ascii",
    )
    (case / "system" / "fvSchemes").write_text(
        foam_header("dictionary", "fvSchemes", "system")
        + "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n",
        encoding="ascii",
    )
    (case / "system" / "fvSolution").write_text(
        foam_header("dictionary", "fvSolution", "system")
        + "solvers { p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.1; } U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; } }\nSIMPLE { nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; } }\nrelaxationFactors { fields { p 0.3; } equations { U 0.3; } }\n",
        encoding="ascii",
    )
    (case / "system" / "blockMeshDict").write_text(_block_mesh_dict(radius, length), encoding="ascii")
    (case / "0" / "U").write_text(_u_field(config), encoding="ascii")
    (case / "0" / "p").write_text(_p_field(), encoding="ascii")


def patch_cylinder_inlet_from_mesh(case_dir: str | Path, config: dict[str, Any]) -> int:
    case = Path(case_dir)
    radius = float(config["cylinder"]["radius_m"])
    mean_velocity = float(config["flow"]["mean_inlet_velocity_m_s"])
    centers = inlet_face_centers(case)
    values = []
    for x, y, _ in centers:
        radial2 = min(max((x * x + y * y) / (radius * radius), 0.0), 1.0)
        values.append((0.0, 0.0, 2.0 * mean_velocity * (1.0 - radial2)))
    (case / "0" / "U").write_text(_u_field(config, values), encoding="ascii")
    return len(values)


def run_openfoam_pipeline(case_dir: str | Path, logs_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    case = Path(case_dir)
    discovery = discover_openfoam(config)
    docker = discover_docker()
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    figures = logs.parent / "figures"
    result: dict[str, Any] = {
        "discovery": discovery,
        "docker": docker,
        "pressure_convention": "kinematic pressure; physical gauge pressure p_pa = rho * p",
    }
    if discovery["available"]:
        runner = "local"
    elif docker["available"]:
        runner = "docker"
    else:
        result.update({"status": "generated_not_executed", "reason": "OpenFOAM commands and Docker are unavailable."})
        return result
    result["runner"] = runner

    block = _run_stage("blockMesh", case, logs, discovery, runner)
    result.setdefault("stages", {})["blockMesh"] = _stage_summary("blockMesh", block)
    if block.returncode != 0:
        result.update({"status": "failed", "failure_reason": f"blockMesh exited with {block.returncode}", "failed_stage": "blockMesh"})
        return result

    result["patched_inlet_faces"] = patch_cylinder_inlet_from_mesh(case, config)

    for stage in ("checkMesh", "simpleFoam"):
        proc = _run_stage(stage, case, logs, discovery, runner)
        result.setdefault("stages", {})[stage] = _stage_summary(stage, proc)
        if proc.returncode != 0:
            result.update({"status": "failed", "failure_reason": f"{stage} exited with {proc.returncode}", "failed_stage": stage})
            return result

    analysis = analyze_openfoam_solution(case, figures, config, logs)
    result.update(analysis)
    continuity = analysis.get("residuals", {}).get("final_continuity_sum_local")
    completed = analysis.get("residuals", {}).get("completed", False)
    if completed and continuity is not None and abs(float(continuity)) < 1.0e-3:
        result.update({"status": "converged_validated", "reason": "simpleFoam completed and continuity residual passed the smoke threshold."})
    else:
        result.update({"status": "executed_not_converged", "reason": "simpleFoam completed, but convergence validation did not pass the smoke threshold."})
    return result


def _run_stage(stage: str, case_dir: Path, logs_dir: Path, discovery: dict[str, Any], runner: str) -> subprocess.CompletedProcess[str]:
    return _run_stage_command(stage, [stage], case_dir, logs_dir, discovery, runner)


def _run_stage_command(stage: str, command: list[str], case_dir: Path, logs_dir: Path, discovery: dict[str, Any], runner: str) -> subprocess.CompletedProcess[str]:
    if runner == "local":
        exe = discovery["commands"].get(command[0]) or command[0]
        proc = subprocess.run([exe, *command[1:]], cwd=case_dir, text=True, capture_output=True, check=False)
    else:
        repo = Path.cwd().resolve()
        case = case_dir.resolve()
        try:
            rel_case = case.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"Docker OpenFOAM cases must be inside the repository: {case}") from exc
        proc = subprocess.run(
            ["docker", "run", "--rm", "-i", "-v", f"{repo}:/work", DOCKER_IMAGE, "bash", "-lc", f"cd /work/{rel_case.as_posix()} && {' '.join(command)}"],
            text=True,
            capture_output=True,
            check=False,
        )
    (logs_dir / f"{stage}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (logs_dir / f"{stage}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return proc


def _stage_summary(stage: str, proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {"returncode": proc.returncode, "stdout_log": f"logs/{stage}.stdout.log", "stderr_log": f"logs/{stage}.stderr.log"}


def analyze_openfoam_solution(case_dir: str | Path, figures_dir: str | Path, config: dict[str, Any], logs_dir: str | Path) -> dict[str, Any]:
    case = Path(case_dir)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    latest = latest_time_dir(case)
    centers = cell_centers(case)
    velocity = read_vector_field(latest / "U")
    p_kinematic = read_scalar_field(latest / "p")
    speed = np.linalg.norm(velocity, axis=1)
    rho = float(config["fluid"]["density_kg_m3"])
    p_pa = p_kinematic * rho
    residuals = parse_simplefoam_log(Path(logs_dir) / "simpleFoam.stdout.log")
    node_fields = write_node_field_report(case, centers, velocity, speed, p_pa, p_kinematic)
    figs = {
        "residual_history": write_residual_figure(residuals, figures),
        "velocity_distribution": write_velocity_distribution_figure(centers, speed, figures),
        "velocity_vectors_3d": write_velocity_vectors_3d(centers, velocity, speed, figures),
        "pressure_distribution_3d": write_pressure_distribution_3d(centers, p_pa, figures),
        "velocity_profile_comparison": write_velocity_profile_comparison(centers, velocity[:, 2], config, figures),
        "pressure_profile": write_pressure_profile(centers, p_pa, figures),
    }
    return {
        "latest_time": latest.name,
        "solution_cells": int(len(centers)),
        "velocity_m_s": {
            "min": float(speed.min()),
            "mean": float(speed.mean()),
            "max": float(speed.max()),
        },
        "pressure_pa": {
            "min": float(p_pa.min()),
            "mean": float(p_pa.mean()),
            "max": float(p_pa.max()),
        },
        "residuals": residuals,
        "figures": figs,
        "node_field_report": node_fields,
        "pinn_comparison": {"status": "not_run", "reason": "No PINN checkpoint or prediction export was provided for this run."},
    }


def generate_vascular_openfoam_case(
    case_dir: str | Path,
    dataset_root: str | Path,
    geometry: str,
    direction: str,
    config: dict[str, Any],
    geometry_validation: dict[str, Any],
) -> dict[str, Any]:
    case = Path(case_dir)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (case / "0").mkdir(exist_ok=True)
    paths = bundle_paths(dataset_root, geometry, direction)
    scale = float(config["resolved"]["source_to_solver_scale"])
    meshes = {}
    for name, source in {"closed": paths["closed"], "wall": paths["wall"], "inlet": paths["inlet"], "outlet": paths["outlet"]}.items():
        mesh = read_stl(source)
        mesh.triangles *= scale
        meshes[name] = mesh
        write_ascii_stl(case / "constant" / "triSurface" / f"{name}.stl", mesh.triangles, name, scale_to_source=1.0)

    inlet_info = vascular_inlet_vector(meshes["inlet"], meshes["outlet"], config)
    bounds = _vascular_background_bounds(meshes["closed"].vertices, config)
    location = _vascular_location_in_mesh(meshes["inlet"], meshes["outlet"], meshes["closed"])
    cells = [int(v) for v in config["vascular"]["mesh"]["background_cells"]]
    refinement = [int(v) for v in config["vascular"]["mesh"]["surface_refinement_level"]]
    end_time = int(config["openfoam"]["end_time"])
    write_interval = int(config["openfoam"]["write_interval"])
    nu = config["resolved"]["kinematic_viscosity_m2_s"]

    (case / "README.md").write_text(
        f"OpenCFD OpenFOAM {config['openfoam']['version']} vascular case for {geometry} ({direction}).\n"
        "This is a generated meshing candidate, not a validated CFD reference until mesh quality and solver diagnostics pass.\n"
        "Surfaces are written in solver units (metres): wall.stl, inlet.stl, outlet.stl; closed.stl is metadata/debug only.\n"
        f"Inlet profile: {config['vascular']['inlet_profile']} with velocity {inlet_info['velocity_m_s']} m/s.\n",
        encoding="utf-8",
    )
    (case / "constant" / "transportProperties").write_text(
        foam_header("dictionary", "transportProperties", "constant")
        + f"transportModel  Newtonian;\nnu [0 2 -1 0 0 0 0] {nu:.12g};\n",
        encoding="ascii",
    )
    (case / "constant" / "turbulenceProperties").write_text(
        foam_header("dictionary", "turbulenceProperties", "constant") + "simulationType laminar;\n",
        encoding="ascii",
    )
    (case / "system" / "controlDict").write_text(
        foam_header("dictionary", "controlDict", "system")
        + f"application simpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime {end_time};\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval {write_interval};\n",
        encoding="ascii",
    )
    (case / "system" / "fvSchemes").write_text(
        foam_header("dictionary", "fvSchemes", "system")
        + "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n",
        encoding="ascii",
    )
    (case / "system" / "fvSolution").write_text(
        foam_header("dictionary", "fvSolution", "system")
        + "solvers { p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.1; } U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; } }\nSIMPLE { nNonOrthogonalCorrectors 2; residualControl { p 1e-4; U 1e-5; } }\nrelaxationFactors { fields { p 0.3; } equations { U 0.3; } }\n",
        encoding="ascii",
    )
    (case / "system" / "blockMeshDict").write_text(_vascular_block_mesh_dict(bounds, cells), encoding="ascii")
    (case / "system" / "surfaceFeatureExtractDict").write_text(_surface_feature_extract_dict(config), encoding="ascii")
    (case / "system" / "snappyHexMeshDict").write_text(_snappy_hex_mesh_dict(location, refinement), encoding="ascii")
    (case / "system" / "meshQualityDict").write_text(_mesh_quality_dict(), encoding="ascii")
    (case / "0" / "U").write_text(_vascular_u_field(inlet_info["velocity_m_s"], include_background=True), encoding="ascii")
    (case / "0" / "p").write_text(_vascular_p_field(include_background=True), encoding="ascii")

    return {
        "status": "generated_not_executed",
        "reason": "Vascular OpenFOAM case generated; meshing was not requested.",
        "case_dir": str(case),
        "surface_mapping": {"wall.stl": "wall", "inlet.stl": "inlet", "outlet.stl": "outlet", "closed.stl": "debug_only"},
        "background_bounds_m": {"min": bounds[0].tolist(), "max": bounds[1].tolist()},
        "location_in_mesh_m": location.tolist(),
        "inlet_profile": inlet_info,
        "geometry_status": geometry_validation.get("status"),
        "pressure_convention": "kinematic pressure; physical gauge pressure p_pa = rho * p",
    }


def run_vascular_mesh_pipeline(case_dir: str | Path, logs_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    case = Path(case_dir)
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    discovery = discover_openfoam(config)
    docker = discover_docker()
    mesh_required = [stage[0] for stage in OPENFOAM_MESH_STAGES]
    local_missing = [cmd for cmd in mesh_required if discovery["commands"].get(cmd) is None]
    result: dict[str, Any] = {
        "discovery": discovery,
        "docker": docker,
        "pressure_convention": "kinematic pressure; physical gauge pressure p_pa = rho * p",
    }
    if not local_missing:
        runner = "local"
    elif docker["available"]:
        runner = "docker"
    else:
        result.update({"status": "generated_not_executed", "reason": "OpenFOAM meshing commands and Docker are unavailable."})
        return result
    result["runner"] = runner
    for stage, command in OPENFOAM_MESH_STAGES:
        proc = _run_stage_command(stage, command, case, logs, discovery, runner)
        result.setdefault("stages", {})[stage] = _stage_summary(stage, proc)
        if proc.returncode != 0:
            result.update({"status": "mesh_failed", "failure_reason": f"{stage} exited with {proc.returncode}", "failed_stage": stage})
            return result
    result.update(
        {
            "status": "mesh_validated",
            "reason": "blockMesh, surfaceFeatureExtract, snappyHexMesh, and checkMesh completed successfully.",
            "mesh_patches": read_boundary_patches(case / "constant" / "polyMesh" / "boundary"),
        }
    )
    return result


def run_vascular_solve_pipeline(case_dir: str | Path, logs_dir: str | Path, figures_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    case = Path(case_dir)
    logs = Path(logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    discovery = discover_openfoam(config)
    docker = discover_docker()
    result: dict[str, Any] = {
        "discovery": discovery,
        "docker": docker,
        "pressure_convention": "kinematic pressure; physical gauge pressure p_pa = rho * p",
    }
    runner = "local" if discovery["commands"].get("simpleFoam") else "docker" if docker["available"] else None
    if runner is None:
        result.update({"status": "executed_not_converged", "failure_reason": "simpleFoam and Docker are unavailable.", "failed_stage": "simpleFoam"})
        return result
    result["runner"] = runner
    patches = validate_vascular_solution_patches(case)
    inlet_velocity = read_vascular_inlet_velocity(case / "0" / "U")
    rewrite_vascular_initial_fields(case, inlet_velocity)
    result["mesh_patches"] = patches
    proc = _run_stage_command("simpleFoam", ["simpleFoam"], case, logs, discovery, runner)
    result.setdefault("stages", {})["simpleFoam"] = _stage_summary("simpleFoam", proc)
    if proc.returncode != 0:
        result.update({"status": "executed_not_converged", "failure_reason": f"simpleFoam exited with {proc.returncode}", "failed_stage": "simpleFoam"})
        return result
    analysis = analyze_vascular_openfoam_solution(case, figures_dir, config, logs)
    result.update(analysis)
    if vascular_solution_converged(analysis, config):
        result.update({"status": "converged_validated", "reason": "Vascular simpleFoam completed and convergence smoke checks passed."})
    else:
        result.update({"status": "executed_not_converged", "reason": "Vascular simpleFoam completed, but convergence smoke checks did not pass."})
    return result


def validate_vascular_solution_patches(case_dir: str | Path) -> dict[str, Any]:
    patches = read_boundary_patches(Path(case_dir) / "constant" / "polyMesh" / "boundary")
    missing = [name for name in ("wall", "inlet", "outlet") if name not in patches]
    if missing:
        raise ValueError(f"Final mesh is missing required patches: {', '.join(missing)}")
    zero = [name for name in ("inlet", "outlet", "wall") if int(patches[name].get("nFaces") or 0) <= 0]
    if zero:
        raise ValueError(f"Final mesh has zero-face required patches: {', '.join(zero)}")
    return patches


def read_vascular_inlet_velocity(path: str | Path) -> list[float]:
    text = Path(path).read_text(encoding="ascii")
    match = re.search(r"inlet\s*\{.*?value\s+uniform\s+\(([^)]*)\)\s*;", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not read uniform inlet velocity from {path}")
    values = [float(value) for value in match.group(1).split()]
    if len(values) != 3:
        raise ValueError(f"Expected three inlet velocity components in {path}")
    return values


def rewrite_vascular_initial_fields(case_dir: str | Path, inlet_velocity: list[float]) -> None:
    case = Path(case_dir)
    validate_vascular_solution_patches(case)
    (case / "0" / "U").write_text(_vascular_u_field(inlet_velocity, include_background=False), encoding="ascii")
    (case / "0" / "p").write_text(_vascular_p_field(include_background=False), encoding="ascii")


def analyze_vascular_openfoam_solution(case_dir: str | Path, figures_dir: str | Path, config: dict[str, Any], logs_dir: str | Path) -> dict[str, Any]:
    case = Path(case_dir)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    latest = latest_time_dir(case)
    centers = cell_centers(case)
    velocity = read_vector_field(latest / "U")
    p_kinematic = read_scalar_field(latest / "p")
    speed = np.linalg.norm(velocity, axis=1)
    rho = float(config["fluid"]["density_kg_m3"])
    p_pa = p_kinematic * rho
    residuals = parse_simplefoam_log(Path(logs_dir) / "simpleFoam.stdout.log")
    node_fields = write_node_field_report(case, centers, velocity, speed, p_pa, p_kinematic)
    mass_balance = read_mass_balance(latest / "phi")
    figs = {
        "residual_history": write_residual_figure(residuals, figures),
        "velocity_vectors_3d": write_velocity_vectors_3d(centers, velocity, speed, figures),
        "pressure_distribution_3d": write_pressure_distribution_3d(centers, p_pa, figures),
    }
    return {
        "latest_time": latest.name,
        "solution_cells": int(len(centers)),
        "velocity_m_s": {"min": float(speed.min()), "mean": float(speed.mean()), "max": float(speed.max())},
        "pressure_pa": {"min": float(p_pa.min()), "mean": float(p_pa.mean()), "max": float(p_pa.max())},
        "residuals": residuals,
        "mass_balance": mass_balance,
        "figures": figs,
        "node_field_report": node_fields,
        "pinn_comparison": {"status": "not_run", "reason": "No PINN checkpoint or prediction export was provided for this run."},
    }


def vascular_solution_converged(analysis: dict[str, Any], config: dict[str, Any]) -> bool:
    residuals = analysis.get("residuals", {})
    if not residuals.get("completed"):
        return False
    continuity = residuals.get("final_continuity_sum_local")
    if continuity is None or abs(float(continuity)) > float(config["vascular"]["solver"]["max_final_continuity_local"]):
        return False
    final_initial = residuals.get("final_initial_residuals", {})
    velocity_limit = float(config["vascular"]["solver"]["max_final_velocity_residual"])
    pressure_limit = float(config["vascular"]["solver"]["max_final_pressure_residual"])
    velocity_ok = all(final_initial.get(field) is not None and abs(float(final_initial[field])) <= velocity_limit for field in ("Ux", "Uy", "Uz"))
    pressure_ok = final_initial.get("p") is not None and abs(float(final_initial["p"])) <= pressure_limit
    return velocity_ok and pressure_ok


def read_mass_balance(phi_path: str | Path) -> dict[str, Any]:
    path = Path(phi_path)
    if not path.exists():
        return {"status": "not_checked", "reason": f"phi field not found: {path}"}
    try:
        inlet = read_boundary_scalar_values(path, "inlet")
        outlet = read_boundary_scalar_values(path, "outlet")
    except ValueError as exc:
        return {"status": "not_checked", "reason": str(exc)}
    inlet_sum = float(np.sum(inlet))
    outlet_sum = float(np.sum(outlet))
    net = inlet_sum + outlet_sum
    scale = max(abs(inlet_sum), abs(outlet_sum), 1e-30)
    return {
        "status": "checked",
        "inlet_flow_rate_m3_s": inlet_sum,
        "outlet_flow_rate_m3_s": outlet_sum,
        "net_flow_rate_m3_s": net,
        "relative_imbalance": abs(net) / scale,
    }


def read_boundary_scalar_values(path: str | Path, patch: str) -> np.ndarray:
    text = Path(path).read_text(encoding="ascii")
    body = _boundary_patch_body(text, patch)
    uniform = re.search(r"value\s+uniform\s+([-+0-9.eE]+)\s*;", body)
    if uniform:
        return np.array([float(uniform.group(1))], dtype=float)
    match = re.search(r"value\s+nonuniform\s+List<scalar>\s+\d+\s*\((.*?)\)\s*;", body, flags=re.S)
    if not match:
        raise ValueError(f"Could not parse scalar boundary values for patch {patch} in {path}")
    return np.array([float(value) for value in match.group(1).split()], dtype=float)


def _boundary_patch_body(text: str, patch: str) -> str:
    start = re.search(rf"^\s*{re.escape(patch)}\s*\{{", text, flags=re.M)
    if not start:
        raise ValueError(f"Patch {patch!r} not found.")
    depth = 0
    body_start = start.end()
    for index in range(start.end() - 1, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:index]
    raise ValueError(f"Patch {patch!r} block is not closed.")


def vascular_inlet_vector(inlet_mesh, outlet_mesh, config: dict[str, Any]) -> dict[str, Any]:
    inlet = cap_properties(inlet_mesh)
    outlet = cap_properties(outlet_mesh)
    normal = np.array(inlet["normal"], dtype=float)
    inlet_center = np.array(inlet["center_m"], dtype=float)
    outlet_center = np.array(outlet["center_m"], dtype=float)
    interior_direction = outlet_center - inlet_center
    norm = np.linalg.norm(interior_direction)
    if norm <= 0 or not np.all(np.isfinite(normal)):
        raise ValueError("Could not determine vascular inlet normal direction.")
    interior_direction /= norm
    alignment = float(np.dot(normal, interior_direction))
    sign = 1.0 if alignment >= 0 else -1.0
    confidence_value = abs(alignment)
    if confidence_value < 0.25:
        raise ValueError(f"Low-confidence inlet normal orientation: |dot|={confidence_value:.3f}")
    velocity = sign * normal * float(config["flow"]["mean_inlet_velocity_m_s"])
    return {
        "profile": config["vascular"]["inlet_profile"],
        "cap_normal": normal.tolist(),
        "interior_direction": interior_direction.tolist(),
        "normal_alignment": alignment,
        "confidence": "high" if confidence_value >= 0.75 else "medium",
        "velocity_m_s": velocity.tolist(),
    }


def read_boundary_patches(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="ascii")
    patches: dict[str, Any] = {}
    for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*(.*?)^\s*\}", text, flags=re.M | re.S):
        name = match.group(1)
        body = match.group(2)
        type_match = re.search(r"type\s+([^;]+);", body)
        faces_match = re.search(r"nFaces\s+(\d+);", body)
        start_match = re.search(r"startFace\s+(\d+);", body)
        if faces_match:
            patches[name] = {
                "type": type_match.group(1).strip() if type_match else None,
                "nFaces": int(faces_match.group(1)),
                "startFace": int(start_match.group(1)) if start_match else None,
            }
    return patches


def latest_time_dir(case_dir: str | Path) -> Path:
    case = Path(case_dir)
    candidates = []
    for child in case.iterdir():
        if child.is_dir():
            try:
                candidates.append((float(child.name), child))
            except ValueError:
                continue
    if not candidates:
        raise ValueError(f"No OpenFOAM time directories found in {case}")
    return max(candidates, key=lambda item: item[0])[1]


def cell_centers(case_dir: str | Path) -> np.ndarray:
    poly_mesh = Path(case_dir) / "constant" / "polyMesh"
    points = np.array(_read_foam_points(poly_mesh / "points"), dtype=float)
    faces = _read_foam_faces(poly_mesh / "faces")
    owner = _read_foam_label_list(poly_mesh / "owner")
    neighbour_path = poly_mesh / "neighbour"
    neighbour = _read_foam_label_list(neighbour_path) if neighbour_path.exists() else []
    n_cells = max(owner + neighbour) + 1
    sums = np.zeros((n_cells, 3), dtype=float)
    counts = np.zeros(n_cells, dtype=float)
    face_centers = np.array([points[list(face)].mean(axis=0) for face in faces], dtype=float)
    for face_i, cell_i in enumerate(owner):
        sums[cell_i] += face_centers[face_i]
        counts[cell_i] += 1.0
    for face_i, cell_i in enumerate(neighbour):
        sums[cell_i] += face_centers[face_i]
        counts[cell_i] += 1.0
    if np.any(counts == 0):
        raise ValueError("Could not compute all OpenFOAM cell centres.")
    return sums / counts[:, None]


def _read_foam_label_list(path: str | Path) -> list[int]:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    start = _foam_list_start(lines)
    values: list[int] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped == ")":
            break
        if stripped and not stripped.startswith("//"):
            values.append(int(stripped))
    return values


def read_vector_field(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="ascii")
    uniform = re.search(r"internalField\s+uniform\s+\(([^)]*)\)\s*;", text)
    if uniform:
        values = [float(value) for value in uniform.group(1).split()]
        return np.array([values], dtype=float)
    body = _nonuniform_body(text, "vector")
    rows = []
    for match in re.finditer(r"\(([^)]*)\)", body):
        rows.append([float(value) for value in match.group(1).split()])
    return np.array(rows, dtype=float)


def read_scalar_field(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="ascii")
    uniform = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
    if uniform:
        return np.array([float(uniform.group(1))], dtype=float)
    body = _nonuniform_body(text, "scalar")
    return np.array([float(value) for value in body.split()], dtype=float)


def _nonuniform_body(text: str, field_type: str) -> str:
    match = re.search(rf"internalField\s+nonuniform\s+List<{field_type}>\s+\d+\s*\((.*?)\)\s*;\s*boundaryField", text, re.S)
    if not match:
        raise ValueError(f"Could not parse nonuniform OpenFOAM {field_type} internalField.")
    return match.group(1)


def parse_simplefoam_log(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    residuals: dict[str, list[float]] = {"Ux": [], "Uy": [], "Uz": [], "p": []}
    times = [float(value) for value in re.findall(r"^Time = ([0-9.eE+-]+)", text, flags=re.M)]
    for field, value in re.findall(r"Solving for (Ux|Uy|Uz|p), Initial residual = ([0-9.eE+-]+)", text):
        residuals[field].append(float(value))
    continuity = [
        tuple(float(value) for value in match)
        for match in re.findall(
            r"time step continuity errors : sum local = ([0-9.eE+-]+), global = ([0-9.eE+-]+), cumulative = ([0-9.eE+-]+)",
            text,
        )
    ]
    final = continuity[-1] if continuity else (None, None, None)
    final_initial_residuals = {field: values[-1] if values else None for field, values in residuals.items()}
    return {
        "completed": bool(re.search(r"^End\s*$", text, flags=re.M)),
        "time_steps": len(times),
        "final_time": times[-1] if times else None,
        "initial_residuals": residuals,
        "final_initial_residuals": final_initial_residuals,
        "final_continuity_sum_local": final[0],
        "final_continuity_global": final[1],
        "final_continuity_cumulative": final[2],
    }


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ga_pinn_cfd_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_residual_figure(residuals: dict[str, Any], figures_dir: str | Path) -> str:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    for field, values in residuals.get("initial_residuals", {}).items():
        if values:
            ax.semilogy(np.arange(1, len(values) + 1), values, label=field)
    ax.set_xlabel("linear solve")
    ax.set_ylabel("initial residual")
    ax.set_title("simpleFoam residual history")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    out = Path(figures_dir) / "openfoam_residual_history.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def write_velocity_distribution_figure(centers: np.ndarray, speed: np.ndarray, figures_dir: str | Path) -> str:
    plt = _pyplot()
    r_mm = np.linalg.norm(centers[:, :2], axis=1) * 1000.0
    z_mm = centers[:, 2] * 1000.0
    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(z_mm, r_mm, c=speed, s=5, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="|U| [m/s]")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("radius [mm]")
    ax.set_title("Velocity distribution in flow domain")
    out = Path(figures_dir) / "openfoam_velocity_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def write_velocity_vectors_3d(centers: np.ndarray, velocity: np.ndarray, speed: np.ndarray, figures_dir: str | Path, max_vectors: int = 1200) -> str:
    plt = _pyplot()
    sample = _even_sample_indices(len(centers), max_vectors)
    c = centers[sample] * 1000.0
    v = velocity[sample]
    s = speed[sample]
    vmax = float(np.max(np.linalg.norm(v, axis=1))) if len(v) else 1.0
    scale = 1.5 / max(vmax, 1e-12)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis((s - float(speed.min())) / max(float(speed.max() - speed.min()), 1e-12))
    ax.quiver(c[:, 0], c[:, 1], c[:, 2], v[:, 0] * scale, v[:, 1] * scale, v[:, 2] * scale, colors=colors, linewidth=0.45, arrow_length_ratio=0.25, normalize=False)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("3D velocity vectors")
    _set_equal_3d_axes(ax, c)
    mappable = plt.cm.ScalarMappable(cmap="viridis")
    mappable.set_array(speed)
    fig.colorbar(mappable, ax=ax, shrink=0.7, label="|U| [m/s]")
    out = Path(figures_dir) / "openfoam_velocity_vectors_3d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def write_pressure_distribution_3d(centers: np.ndarray, p_pa: np.ndarray, figures_dir: str | Path, max_points: int = 8000) -> str:
    plt = _pyplot()
    sample = _even_sample_indices(len(centers), max_points)
    c = centers[sample] * 1000.0
    p = p_pa[sample]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=p, s=4, cmap="plasma", alpha=0.85)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="p [Pa]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("3D pressure distribution")
    _set_equal_3d_axes(ax, c)
    out = Path(figures_dir) / "openfoam_pressure_distribution_3d.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def write_node_field_report(case_dir: str | Path, centers: np.ndarray, velocity: np.ndarray, speed: np.ndarray, p_pa: np.ndarray, p_kinematic: np.ndarray) -> dict[str, Any]:
    solution = Path(case_dir).parent / "solution"
    solution.mkdir(parents=True, exist_ok=True)
    fields_csv = solution / "cfd_cell_center_fields.csv"
    preview_csv = solution / "cfd_cell_center_fields_preview.csv"
    headers = ["cell_id", "x_m", "y_m", "z_m", "u_x_m_s", "u_y_m_s", "u_z_m_s", "speed_m_s", "p_pa", "p_kinematic_m2_s2"]
    _write_field_csv(fields_csv, headers, range(len(centers)), centers, velocity, speed, p_pa, p_kinematic)
    sample = _even_sample_indices(len(centers), 200)
    _write_field_csv(preview_csv, headers, sample, centers, velocity, speed, p_pa, p_kinematic)
    return {
        "kind": "OpenFOAM finite-volume cell centres",
        "cell_count": int(len(centers)),
        "fields": headers,
        "full_csv": str(fields_csv),
        "preview_csv": str(preview_csv),
        "pinn_comparison_key": "Evaluate PINN velocity and pressure at x_m,y_m,z_m rows, then compare u_x,u_y,u_z,speed,p_pa by cell_id.",
    }


def _write_field_csv(
    path: Path,
    headers: list[str],
    indices,
    centers: np.ndarray,
    velocity: np.ndarray,
    speed: np.ndarray,
    p_pa: np.ndarray,
    p_kinematic: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx in indices:
            writer.writerow(
                [
                    int(idx),
                    f"{centers[idx, 0]:.12g}",
                    f"{centers[idx, 1]:.12g}",
                    f"{centers[idx, 2]:.12g}",
                    f"{velocity[idx, 0]:.12g}",
                    f"{velocity[idx, 1]:.12g}",
                    f"{velocity[idx, 2]:.12g}",
                    f"{speed[idx]:.12g}",
                    f"{p_pa[idx]:.12g}",
                    f"{p_kinematic[idx]:.12g}",
                ]
            )


def _even_sample_indices(size: int, max_count: int) -> np.ndarray:
    if size <= max_count:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, max_count, dtype=int))


def _set_equal_3d_axes(ax, points_mm: np.ndarray) -> None:
    mins = points_mm.min(axis=0)
    maxs = points_mm.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-9)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def write_velocity_profile_comparison(centers: np.ndarray, uz: np.ndarray, config: dict[str, Any], figures_dir: str | Path) -> str:
    plt = _pyplot()
    radius = float(config["cylinder"]["radius_m"])
    mean_u = float(config["flow"]["mean_inlet_velocity_m_s"])
    length = float(config["cylinder"]["length_m"])
    z = centers[:, 2]
    slab = np.abs(z - 0.5 * length) < max(length / 80.0, 1e-9)
    if not np.any(slab):
        slab = np.ones(len(z), dtype=bool)
    r = np.linalg.norm(centers[slab, :2], axis=1)
    rr = np.linspace(0.0, radius, 100)
    analytical = 2.0 * mean_u * (1.0 - (rr / radius) ** 2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(r * 1000.0, uz[slab], s=8, alpha=0.45, label="OpenFOAM cells")
    ax.plot(rr * 1000.0, analytical, color="black", linewidth=1.5, label="analytical")
    ax.set_xlabel("radius [mm]")
    ax.set_ylabel("axial velocity [m/s]")
    ax.set_title("Mid-pipe velocity profile")
    ax.legend(loc="best")
    out = Path(figures_dir) / "openfoam_velocity_profile_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def write_pressure_profile(centers: np.ndarray, p_pa: np.ndarray, figures_dir: str | Path) -> str:
    plt = _pyplot()
    z = centers[:, 2]
    bins = np.linspace(float(z.min()), float(z.max()), 41)
    digitized = np.digitize(z, bins) - 1
    z_mid, p_mean = [], []
    for idx in range(len(bins) - 1):
        mask = digitized == idx
        if np.any(mask):
            z_mid.append(0.5 * (bins[idx] + bins[idx + 1]) * 1000.0)
            p_mean.append(float(p_pa[mask].mean()))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z_mid, p_mean)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("mean gauge pressure [Pa]")
    ax.set_title("OpenFOAM pressure profile")
    out = Path(figures_dir) / "openfoam_pressure_profile.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def _vascular_background_bounds(vertices: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    padding = float(config["vascular"]["mesh"]["background_padding_fraction"]) * span
    return mins - padding, maxs + padding


def _vascular_location_in_mesh(inlet_mesh, outlet_mesh, closed_mesh) -> np.ndarray:
    inlet_center = np.array(cap_properties(inlet_mesh)["center_m"], dtype=float)
    outlet_center = np.array(cap_properties(outlet_mesh)["center_m"], dtype=float)
    midpoint = 0.5 * (inlet_center + outlet_center)
    if np.all(np.isfinite(midpoint)):
        return midpoint
    return closed_mesh.vertices.mean(axis=0)


def _vascular_block_mesh_dict(bounds: tuple[np.ndarray, np.ndarray], cells: list[int]) -> str:
    lo, hi = bounds
    nx, ny, nz = cells
    return foam_header("dictionary", "blockMeshDict", "system") + f"""scale 1;
vertices
(
    ({lo[0]:.12g} {lo[1]:.12g} {lo[2]:.12g})
    ({hi[0]:.12g} {lo[1]:.12g} {lo[2]:.12g})
    ({hi[0]:.12g} {hi[1]:.12g} {lo[2]:.12g})
    ({lo[0]:.12g} {hi[1]:.12g} {lo[2]:.12g})
    ({lo[0]:.12g} {lo[1]:.12g} {hi[2]:.12g})
    ({hi[0]:.12g} {lo[1]:.12g} {hi[2]:.12g})
    ({hi[0]:.12g} {hi[1]:.12g} {hi[2]:.12g})
    ({lo[0]:.12g} {hi[1]:.12g} {hi[2]:.12g})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);
edges ();
boundary
(
    background
    {{
        type patch;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (1 2 6 5)
            (2 3 7 6)
            (3 0 4 7)
        );
    }}
);
mergePatchPairs ();
"""


def _surface_feature_extract_dict(config: dict[str, Any]) -> str:
    angle = float(config["vascular"]["mesh"]["feature_extract_included_angle_deg"])
    entries = []
    for surface in ("wall.stl", "inlet.stl", "outlet.stl"):
        entries.append(
            f"""{surface}
{{
    extractionMethod extractFromSurface;
    extractFromSurfaceCoeffs
    {{
        includedAngle {angle:.12g};
    }}
    writeObj no;
}}"""
        )
    return foam_header("dictionary", "surfaceFeatureExtractDict", "system") + "\n".join(entries) + "\n"


def _snappy_hex_mesh_dict(location: np.ndarray, refinement: list[int]) -> str:
    low, high = refinement
    return foam_header("dictionary", "snappyHexMeshDict", "system") + f"""castellatedMesh true;
snap true;
addLayers false;

geometry
{{
    wall.stl {{ type triSurfaceMesh; name wall; }}
    inlet.stl {{ type triSurfaceMesh; name inlet; }}
    outlet.stl {{ type triSurfaceMesh; name outlet; }}
}}

castellatedMeshControls
{{
    maxLocalCells 200000;
    maxGlobalCells 600000;
    minRefinementCells 0;
    nCellsBetweenLevels 2;
    features
    (
        {{ file "wall.eMesh"; level {high}; }}
        {{ file "inlet.eMesh"; level {high}; }}
        {{ file "outlet.eMesh"; level {high}; }}
    );
    refinementSurfaces
    {{
        wall {{ level ({low} {high}); patchInfo {{ type wall; }} }}
        inlet {{ level ({low} {high}); patchInfo {{ type patch; }} }}
        outlet {{ level ({low} {high}); patchInfo {{ type patch; }} }}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    locationInMesh ({location[0]:.12g} {location[1]:.12g} {location[2]:.12g});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap true;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{}}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    #include "meshQualityDict"
}}

debug 0;
mergeTolerance 1e-6;
"""


def _mesh_quality_dict() -> str:
    return foam_header("dictionary", "meshQualityDict", "system") + """maxNonOrtho 75;
maxBoundarySkewness 20;
maxInternalSkewness 4;
maxConcave 80;
minFlatness 0.5;
minVol 1e-18;
minTetQuality -1e30;
minArea -1;
minTwist 0.02;
minDeterminant 0.001;
minFaceWeight 0.02;
minVolRatio 0.01;
minTriangleTwist -1;
nSmoothScale 4;
errorReduction 0.75;
"""


def _vascular_u_field(inlet_velocity: list[float], include_background: bool = False) -> str:
    vx, vy, vz = inlet_velocity
    background = "    background { type zeroGradient; }\n" if include_background else ""
    return foam_header("volVectorField", "U", "0") + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform ({vx:.12g} {vy:.12g} {vz:.12g});
    }}
    outlet {{ type zeroGradient; }}
    wall {{ type noSlip; }}
{background.rstrip()}
}}
"""


def _vascular_p_field(include_background: bool = False) -> str:
    background = "    background { type zeroGradient; }\n" if include_background else ""
    return foam_header("volScalarField", "p", "0") + f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    wall {{ type zeroGradient; }}
{background.rstrip()}
}}
"""


def _block_mesh_dict(radius: float, length: float) -> str:
    r = radius
    q = radius / math.sqrt(2.0)
    return foam_header("dictionary", "blockMeshDict", "system") + f"""scale 1;
vertices
(
    ({r} 0 0) (0 {r} 0) (-{r} 0 0) (0 -{r} 0)
    ({r} 0 {length}) (0 {r} {length}) (-{r} 0 {length}) (0 -{r} {length})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 80) simpleGrading (1 1 1)
);
edges
(
    arc 0 1 ({q} {q} 0)
    arc 1 2 (-{q} {q} 0)
    arc 2 3 (-{q} -{q} 0)
    arc 3 0 ({q} -{q} 0)
    arc 4 5 ({q} {q} {length})
    arc 5 6 (-{q} {q} {length})
    arc 6 7 (-{q} -{q} {length})
    arc 7 4 ({q} -{q} {length})
);
boundary
(
    inlet {{ type patch; faces ((0 3 2 1)); }}
    outlet {{ type patch; faces ((4 5 6 7)); }}
    wall {{ type wall; faces ((0 1 5 4) (1 2 6 5) (2 3 7 6) (3 0 4 7)); }}
);
mergePatchPairs ();
"""


def inlet_face_centers(case_dir: str | Path) -> list[tuple[float, float, float]]:
    poly_mesh = Path(case_dir) / "constant" / "polyMesh"
    boundary = (poly_mesh / "boundary").read_text(encoding="ascii")
    match = re.search(r"inlet\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);", boundary, re.S)
    if not match:
        raise ValueError(f"Could not find inlet patch in {poly_mesh / 'boundary'}")
    n_faces, start_face = (int(match.group(1)), int(match.group(2)))
    points = _read_foam_points(poly_mesh / "points")
    faces = _read_foam_faces(poly_mesh / "faces")
    centers = []
    for face in faces[start_face : start_face + n_faces]:
        vertices = [points[i] for i in face]
        centers.append(tuple(sum(vertex[axis] for vertex in vertices) / len(vertices) for axis in range(3)))
    return centers


def _read_foam_points(path: str | Path) -> list[tuple[float, float, float]]:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    start = _foam_list_start(lines)
    points = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped == ")":
            break
        if not stripped or stripped.startswith("//"):
            continue
        values = stripped.strip("()").split()
        if len(values) == 3:
            points.append((float(values[0]), float(values[1]), float(values[2])))
    return points


def _read_foam_faces(path: str | Path) -> list[tuple[int, ...]]:
    text = Path(path).read_text(encoding="ascii")
    start = text.find("\n(")
    if start < 0:
        raise ValueError(f"OpenFOAM face list opening not found: {path}")
    tokens = re.findall(r"\d+|\(|\)", text[start:])
    index = 0
    if tokens[index] != "(":
        raise ValueError(f"OpenFOAM face list opening not found: {path}")
    index += 1
    faces = []
    while index < len(tokens) and tokens[index] != ")":
        n_vertices = int(tokens[index])
        index += 1
        if index >= len(tokens) or tokens[index] != "(":
            raise ValueError(f"Malformed OpenFOAM face list in {path}")
        index += 1
        face = tuple(int(value) for value in tokens[index : index + n_vertices])
        index += n_vertices
        if index >= len(tokens) or tokens[index] != ")":
            raise ValueError(f"Malformed OpenFOAM face list in {path}")
        index += 1
        faces.append(face)
    return faces


def _foam_list_start(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.strip() == "(":
            return i + 1
    raise ValueError("OpenFOAM list opening '(' not found.")


def _u_field(config: dict[str, Any], inlet_values: list[tuple[float, float, float]] | None = None) -> str:
    u = config["flow"]["mean_inlet_velocity_m_s"]
    radius = config["cylinder"]["radius_m"]
    if inlet_values is None:
        inlet_values = _parabolic_inlet_values(radius, u, cells_per_axis=20)
    return foam_header("volVectorField", "U", "0") + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 {u});
boundaryField
{{
    inlet
    {{
        type fixedValue;
        value {_nonuniform_vector_list(inlet_values)};
    }}
    outlet {{ type zeroGradient; }}
    wall {{ type noSlip; }}
}}
"""


def _p_field() -> str:
    return foam_header("volScalarField", "p", "0") + """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    wall { type zeroGradient; }
}
"""


def _parabolic_inlet_values(radius: float, mean_velocity: float, cells_per_axis: int) -> list[tuple[float, float, float]]:
    values = []
    # The inlet block face is a diamond inscribed in the circular arc boundary.
    # Values are ordered on the structured inlet patch and clipped to zero at the wall.
    for j in range(cells_per_axis):
        eta = -1.0 + (j + 0.5) * 2.0 / cells_per_axis
        for i in range(cells_per_axis):
            xi = -1.0 + (i + 0.5) * 2.0 / cells_per_axis
            x = radius * 0.5 * (xi - eta)
            y = radius * 0.5 * (xi + eta)
            radial2 = min(max((x * x + y * y) / (radius * radius), 0.0), 1.0)
            values.append((0.0, 0.0, 2.0 * mean_velocity * (1.0 - radial2)))
    return values


def _nonuniform_vector_list(values: list[tuple[float, float, float]]) -> str:
    body = "\n".join(f"            ({x:.12g} {y:.12g} {z:.12g})" for x, y, z in values)
    return f"nonuniform List<vector>\n        {len(values)}\n        (\n{body}\n        )"
