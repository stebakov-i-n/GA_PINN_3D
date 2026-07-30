from __future__ import annotations

import hashlib
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ga_pinn_cfd_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TriMesh:
    triangles: np.ndarray

    @property
    def vertices(self) -> np.ndarray:
        return self.triangles.reshape(-1, 3)


def bundle_paths(dataset_root: str | Path, geometry: str, direction: str) -> dict[str, Path]:
    normalized = geometry.replace("\\", "/")
    if not normalized.endswith(".stl") or "/" not in normalized:
        raise ValueError("Geometry must look like <case_id>/<k>_<m>.stl")
    base = Path(dataset_root) / normalized
    stem = base.with_suffix("")
    inlet_suffix, outlet_suffix = ("_2", "_1") if direction == "forward" else ("_1", "_2")
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be 'forward' or 'reverse'")
    return {
        "closed": base,
        "cap_1": Path(str(stem) + "_1.stl"),
        "cap_2": Path(str(stem) + "_2.stl"),
        "wall": Path(str(stem) + "_3.stl"),
        "inlet": Path(str(stem) + inlet_suffix + ".stl"),
        "outlet": Path(str(stem) + outlet_suffix + ".stl"),
    }


def read_stl(path: str | Path) -> TriMesh:
    data = Path(path).read_bytes()
    triangles = _read_binary_stl(data)
    if triangles is None:
        triangles = _read_ascii_stl(data)
    if triangles.size == 0:
        raise ValueError(f"No triangles found in STL: {path}")
    return TriMesh(triangles.astype(float))


def validate_bundle(
    dataset_root: str | Path,
    geometry: str,
    direction: str,
    config: dict[str, Any],
    figures_dir: str | Path | None = None,
) -> dict[str, Any]:
    paths = bundle_paths(dataset_root, geometry, direction)
    scale = float(config["resolved"]["source_to_solver_scale"])
    rho = float(config["fluid"]["density_kg_m3"])
    mu = float(config["fluid"]["dynamic_viscosity_pa_s"])
    u_mean = float(config["flow"]["mean_inlet_velocity_m_s"])
    checks: dict[str, Any] = {}
    meshes: dict[str, TriMesh] = {}
    warnings: list[str] = []
    failures: list[str] = []

    for name, path in {k: v for k, v in paths.items() if k in {"closed", "cap_1", "cap_2", "wall"}}.items():
        record = {"path": str(path), "exists": path.exists(), "readable": False}
        if not path.exists():
            failures.append(f"Missing required geometry file: {path}")
            checks[name] = record
            continue
        try:
            mesh = read_stl(path)
            mesh.triangles *= scale
            meshes[name] = mesh
            record.update(_mesh_quality(mesh))
            record["readable"] = True
            record["sha256"] = sha256_file(path)
        except Exception as exc:
            failures.append(f"Could not read {path}: {exc}")
            record["error"] = str(exc)
        checks[name] = record

    cap_metrics = {}
    for cap_name in ("cap_1", "cap_2"):
        if cap_name in meshes:
            cap_metrics[cap_name] = cap_properties(meshes[cap_name])
    inlet_cap = "cap_2" if direction == "forward" else "cap_1"
    inlet = cap_metrics.get(inlet_cap, {})
    area = float(inlet.get("area_m2", 0.0) or 0.0)
    perimeter = float(inlet.get("perimeter_m", 0.0) or 0.0)
    hydraulic_diameter = 4.0 * area / perimeter if area > 0 and perimeter > 0 else math.nan
    reynolds = rho * u_mean * hydraulic_diameter / mu if math.isfinite(hydraulic_diameter) else math.nan
    if math.isfinite(reynolds) and reynolds > 2000:
        warnings.append(f"Inlet Reynolds number {reynolds:.1f} is outside a conservative laminar range.")

    watertight = _watertight(meshes.get("closed"))
    if watertight["status"] == "failed":
        warnings.append("Closed surface edge counts are not watertight.")

    orientation = _orientation_check(cap_metrics, direction)
    if orientation["confidence"] == "low":
        warnings.append("Inlet/outlet orientation could not be established robustly from cap normals alone.")

    result = {
        "geometry": geometry.replace("\\", "/"),
        "direction": direction,
        "units": {"source": config["geometry"]["source_length_unit"], "solver": "m", "scale": scale},
        "files": checks,
        "caps": cap_metrics,
        "inlet": {
            "cap": inlet_cap,
            "area_m2": area,
            "perimeter_m": perimeter,
            "hydraulic_diameter_m": hydraulic_diameter,
            "reynolds_number": reynolds,
        },
        "combined_surface": {
            "watertightness": watertight,
            "patch_gaps_intersections": {
                "status": "not_checked",
                "reason": "Robust patch gap/intersection checks require topology-aware surface processing beyond this phase.",
            },
        },
        "orientation": orientation,
        "warnings": warnings,
        "failures": failures,
        "status": "failed" if failures else "validated_with_warnings" if warnings else "validated",
    }
    if figures_dir and meshes:
        result["figures"] = {"geometry_preview": str(make_geometry_preview(meshes, figures_dir))}
    return result


def cap_properties(mesh: TriMesh) -> dict[str, Any]:
    triangles = mesh.triangles
    verts = mesh.vertices
    areas = triangle_areas(triangles)
    total_area = float(areas.sum())
    center = np.average(triangles.reshape(-1, 3), axis=0).tolist()
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal = normals.sum(axis=0)
    norm = np.linalg.norm(normal)
    normal_unit = (normal / norm).tolist() if norm > 0 else [math.nan, math.nan, math.nan]
    plane_normal = np.array(normal_unit, dtype=float)
    if np.all(np.isfinite(plane_normal)):
        distances = (verts - np.array(center)) @ plane_normal
        planarity = float(np.sqrt(np.mean(distances**2)))
    else:
        planarity = math.nan
    return {
        "center_m": center,
        "normal": normal_unit,
        "planarity_rms_m": planarity,
        "area_m2": total_area,
        "perimeter_m": boundary_perimeter(triangles),
    }


def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)


def boundary_perimeter(triangles: np.ndarray, decimals: int = 12) -> float:
    counts: dict[tuple[tuple[float, float, float], tuple[float, float, float]], int] = {}
    rounded = np.round(triangles, decimals=decimals)
    for tri in rounded:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((tuple(a.tolist()), tuple(b.tolist()))))
            counts[key] = counts.get(key, 0) + 1
    perimeter = 0.0
    for (a, b), count in counts.items():
        if count == 1:
            perimeter += float(np.linalg.norm(np.array(a) - np.array(b)))
    return perimeter


def write_ascii_stl(path: str | Path, triangles_m: np.ndarray, name: str = "mesh", scale_to_source: float = 1000.0) -> None:
    triangles = triangles_m * scale_to_source
    lines = [f"solid {name}"]
    for tri in triangles:
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm else normal
        lines.append(f"  facet normal {normal[0]:.12g} {normal[1]:.12g} {normal[2]:.12g}")
        lines.append("    outer loop")
        for vertex in tri:
            lines.append(f"      vertex {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def make_geometry_preview(meshes: dict[str, TriMesh], figures_dir: str | Path) -> Path:
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"wall": "#7c8da6", "cap_1": "#d95f02", "cap_2": "#1b9e77", "closed": "#bbbbbb"}
    for name in ("wall", "cap_1", "cap_2"):
        mesh = meshes.get(name)
        if mesh is None:
            continue
        tris = mesh.triangles
        sample = tris[:: max(1, len(tris) // 800)]
        ax.plot_trisurf(sample[:, :, 0].ravel(), sample[:, :, 1].ravel(), sample[:, :, 2].ravel(), triangles=np.arange(sample.size // 3).reshape(-1, 3), color=colors[name], alpha=0.55)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    out = figures / "geometry_preview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mesh_quality(mesh: TriMesh) -> dict[str, Any]:
    areas = triangle_areas(mesh.triangles)
    finite = bool(np.isfinite(mesh.triangles).all())
    nondegenerate = int(np.count_nonzero(areas > 0.0))
    return {
        "triangles": int(len(mesh.triangles)),
        "finite": finite,
        "nondegenerate_triangles": nondegenerate,
        "degenerate_triangles": int(len(areas) - nondegenerate),
    }


def _watertight(mesh: TriMesh | None) -> dict[str, Any]:
    if mesh is None:
        return {"status": "not_checked", "reason": "Closed surface was not readable."}
    counts: dict[Any, int] = {}
    rounded = np.round(mesh.triangles, decimals=12)
    for tri in rounded:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((tuple(a.tolist()), tuple(b.tolist()))))
            counts[key] = counts.get(key, 0) + 1
    bad = sum(1 for count in counts.values() if count != 2)
    return {"status": "passed" if bad == 0 else "failed", "non_manifold_or_boundary_edges": bad}


def _orientation_check(cap_metrics: dict[str, Any], direction: str) -> dict[str, Any]:
    if "cap_1" not in cap_metrics or "cap_2" not in cap_metrics:
        return {"status": "not_checked", "confidence": "low", "reason": "Both caps must be readable."}
    n1 = np.array(cap_metrics["cap_1"]["normal"], dtype=float)
    n2 = np.array(cap_metrics["cap_2"]["normal"], dtype=float)
    c1 = np.array(cap_metrics["cap_1"]["center_m"], dtype=float)
    c2 = np.array(cap_metrics["cap_2"]["center_m"], dtype=float)
    axis = c2 - c1
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0 or not np.all(np.isfinite(n1)) or not np.all(np.isfinite(n2)):
        return {"status": "not_checked", "confidence": "low", "reason": "Degenerate cap centres or normals."}
    axis /= axis_norm
    inlet_normal = n2 if direction == "forward" else n1
    alignment = float(abs(np.dot(inlet_normal, axis)))
    return {"status": "checked", "confidence": "medium" if alignment > 0.5 else "low", "axis_alignment_abs": alignment}


def _read_binary_stl(data: bytes) -> np.ndarray | None:
    if len(data) < 84:
        return None
    count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + count * 50
    if expected != len(data):
        return None
    tris = np.zeros((count, 3, 3), dtype=float)
    offset = 84
    for i in range(count):
        vals = struct.unpack("<12fH", data[offset : offset + 50])
        tris[i] = np.array(vals[3:12], dtype=float).reshape(3, 3)
        offset += 50
    return tris


def _read_ascii_stl(data: bytes) -> np.ndarray:
    vertices: list[list[float]] = []
    for raw in data.decode("utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vertices) % 3:
        raise ValueError("ASCII STL vertex count is not divisible by 3.")
    return np.array(vertices, dtype=float).reshape(-1, 3, 3)
