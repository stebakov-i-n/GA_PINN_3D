from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ga_pinn_cfd_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import validate_bundle, write_ascii_stl
from .openfoam import generate_openfoam_case, run_openfoam_pipeline
from .reporting import create_run_dir, finish_run, start_metadata


def velocity_profile(r: np.ndarray, radius_m: float, mean_velocity_m_s: float) -> np.ndarray:
    return 2.0 * mean_velocity_m_s * (1.0 - (r / radius_m) ** 2)


def pressure_gradient_pa_m(mu_pa_s: float, mean_velocity_m_s: float, radius_m: float) -> float:
    return -8.0 * mu_pa_s * mean_velocity_m_s / (radius_m**2)


def analytical_metrics(config: dict[str, Any]) -> dict[str, float]:
    radius = float(config["cylinder"]["radius_m"])
    u_mean = float(config["flow"]["mean_inlet_velocity_m_s"])
    mu = float(config["fluid"]["dynamic_viscosity_pa_s"])
    rho = float(config["fluid"]["density_kg_m3"])
    diameter = 2.0 * radius
    return {
        "radius_m": radius,
        "length_m": float(config["cylinder"]["length_m"]),
        "centerline_velocity_m_s": 2.0 * u_mean,
        "wall_velocity_m_s": 0.0,
        "mean_velocity_m_s": u_mean,
        "pressure_gradient_pa_m": pressure_gradient_pa_m(mu, u_mean, radius),
        "reynolds_number": rho * u_mean * diameter / mu,
    }


def generate_cylinder_bundle(output_dir: str | Path, config: dict[str, Any]) -> dict[str, str]:
    root = Path(output_dir) / "geometry" / "synthetic_cylinder"
    root.mkdir(parents=True, exist_ok=True)
    radius = float(config["cylinder"]["radius_m"])
    length = float(config["cylinder"]["length_m"])
    n = int(config["cylinder"]["circumferential_segments"])
    z0, z1 = 0.0, length
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring0 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, z0)])
    ring1 = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, z1)])
    c0 = np.array([0.0, 0.0, z0])
    c1 = np.array([0.0, 0.0, z1])
    wall, cap0, cap1 = [], [], []
    for i in range(n):
        j = (i + 1) % n
        wall.append([ring0[i], ring0[j], ring1[j]])
        wall.append([ring0[i], ring1[j], ring1[i]])
        cap0.append([c0, ring0[j], ring0[i]])
        cap1.append([c1, ring1[i], ring1[j]])
    closed = np.array(wall + cap0 + cap1)
    paths = {
        "closed": root / "1_1.stl",
        "cap_1": root / "1_1_1.stl",
        "cap_2": root / "1_1_2.stl",
        "wall": root / "1_1_3.stl",
    }
    write_ascii_stl(paths["closed"], closed, "cylinder_closed")
    write_ascii_stl(paths["cap_1"], np.array(cap0), "cylinder_cap_1")
    write_ascii_stl(paths["cap_2"], np.array(cap1), "cylinder_cap_2")
    write_ascii_stl(paths["wall"], np.array(wall), "cylinder_wall")
    return {k: str(v) for k, v in paths.items()}


def write_profile_figure(config: dict[str, Any], figures_dir: str | Path) -> str:
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    radius = float(config["cylinder"]["radius_m"])
    samples = int(config["cylinder"]["radial_samples"])
    r = np.linspace(0.0, radius, samples)
    u = velocity_profile(r, radius, float(config["flow"]["mean_inlet_velocity_m_s"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r * 1000.0, u)
    ax.set_xlabel("radius [mm]")
    ax.set_ylabel("axial velocity [m/s]")
    ax.set_title("Analytical Hagen-Poiseuille profile")
    out = figures / "analytical_velocity_profile.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def run_cylinder(config: dict[str, Any], output_root: str | Path, command: list[str], run_openfoam: bool = False) -> Path:
    run_dir = create_run_dir(output_root, "cylinder", "forward", config)
    metadata = start_metadata(config, command, "cylinder", "forward")
    metrics = {"analytical": analytical_metrics(config)}
    try:
        generate_cylinder_bundle(run_dir, config)
        metrics["geometry_validation"] = validate_bundle(run_dir / "geometry", "synthetic_cylinder/1_1.stl", "forward", config, run_dir / "figures")
        metrics["figures"] = {"analytical_velocity_profile": write_profile_figure(config, run_dir / "figures")}
        generate_openfoam_case(run_dir / "openfoam_case", config)
        solver_result = run_openfoam_pipeline(run_dir / "openfoam_case", run_dir / "logs", config) if run_openfoam else {"status": "generated_not_executed", "reason": "--run-openfoam was not requested."}
        metrics["openfoam"] = solver_result
        status = solver_result["status"]
        failure = solver_result.get("failure_reason")
    except Exception as exc:
        status = "failed"
        failure = str(exc)
        metrics["failure"] = failure
    finish_run(run_dir, config, metadata, metrics, status, failure)
    return run_dir
