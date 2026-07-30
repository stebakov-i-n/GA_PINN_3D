from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


UNIT_SCALE_TO_M = {"m": 1.0, "mm": 1.0e-3}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping.")
    return resolve_config(data, base_dir=config_path.parent)


def resolve_config(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    cfg = dict(data)
    _require(cfg, "fluid.density_kg_m3", positive=True)
    _require(cfg, "fluid.dynamic_viscosity_pa_s", positive=True)
    _require(cfg, "flow.mean_inlet_velocity_m_s", positive=True)
    _expect(cfg, "flow.regime", "steady_incompressible_laminar")
    _expect(cfg, "flow.gravity", False)
    _expect(cfg, "boundary_conditions.wall", "no_slip")
    _require(cfg, "boundary_conditions.outlet_gauge_pressure_pa")
    _expect(cfg, "vascular.inlet_profile", "uniform_normal")
    cells = _get(cfg, "vascular.mesh.background_cells")
    if not isinstance(cells, list) or len(cells) != 3:
        raise ConfigError("vascular.mesh.background_cells must contain three positive integers.")
    for value in cells:
        if isinstance(value, bool):
            raise ConfigError("vascular.mesh.background_cells must contain three positive integers.")
        try:
            converted = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("vascular.mesh.background_cells must contain three positive integers.") from exc
        if converted != value or converted <= 0:
            raise ConfigError("vascular.mesh.background_cells must contain three positive integers.")
    _require(cfg, "vascular.mesh.background_padding_fraction", positive=True)
    levels = _get(cfg, "vascular.mesh.surface_refinement_level")
    if not isinstance(levels, list) or len(levels) != 2:
        raise ConfigError("vascular.mesh.surface_refinement_level must be [min, max] non-negative integers.")
    converted_levels = []
    for value in levels:
        if isinstance(value, bool):
            raise ConfigError("vascular.mesh.surface_refinement_level must be [min, max] non-negative integers.")
        try:
            converted = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError("vascular.mesh.surface_refinement_level must be [min, max] non-negative integers.") from exc
        if converted != value or converted < 0:
            raise ConfigError("vascular.mesh.surface_refinement_level must be [min, max] non-negative integers.")
        converted_levels.append(converted)
    if converted_levels[0] > converted_levels[1]:
        raise ConfigError("vascular.mesh.surface_refinement_level must be [min, max] non-negative integers.")
    _require(cfg, "vascular.mesh.feature_extract_included_angle_deg", positive=True)
    _expect(cfg, "vascular.mesh.location_in_mesh", "auto")
    _require(cfg, "vascular.solver.max_final_continuity_local", positive=True)
    _require(cfg, "vascular.solver.max_final_velocity_residual", positive=True)
    _require(cfg, "vascular.solver.max_final_pressure_residual", positive=True)
    _expect(cfg, "openfoam.distribution", "OpenCFD")
    _expect(cfg, "openfoam.version", "v2606")
    _expect(cfg, "openfoam.solver", "simpleFoam")
    _require_integer(cfg, "openfoam.end_time", positive=True)
    _require_integer(cfg, "openfoam.write_interval", positive=True)

    source_unit = _get(cfg, "geometry.source_length_unit")
    solver_unit = _get(cfg, "geometry.solver_length_unit")
    if source_unit not in UNIT_SCALE_TO_M:
        raise ConfigError(f"Unsupported geometry.source_length_unit: {source_unit!r}")
    if solver_unit != "m":
        raise ConfigError("Only solver_length_unit='m' is supported in this phase.")

    rho = float(_get(cfg, "fluid.density_kg_m3"))
    mu = float(_get(cfg, "fluid.dynamic_viscosity_pa_s"))
    cfg["resolved"] = {
        "kinematic_viscosity_m2_s": mu / rho,
        "source_to_solver_scale": UNIT_SCALE_TO_M[source_unit],
        "config_hash": stable_hash(cfg),
    }
    return cfg


def stable_hash(data: dict[str, Any]) -> str:
    dumped = yaml.safe_dump(data, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _get(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"Missing required config value: {dotted}")
        cur = cur[part]
    return cur


def _require(data: dict[str, Any], dotted: str, positive: bool = False) -> Any:
    value = _get(data, dotted)
    if positive and float(value) <= 0:
        raise ConfigError(f"Config value must be positive: {dotted}")
    return value


def _require_integer(data: dict[str, Any], dotted: str, positive: bool = False) -> int:
    value = _get(data, dotted)
    if isinstance(value, bool):
        raise ConfigError(f"Config value must be an integer: {dotted}")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Config value must be an integer: {dotted}") from exc
    if converted != value:
        raise ConfigError(f"Config value must be an integer: {dotted}")
    if positive and converted <= 0:
        raise ConfigError(f"Config value must be positive: {dotted}")
    return converted


def _expect(data: dict[str, Any], dotted: str, expected: Any) -> None:
    value = _get(data, dotted)
    if value != expected:
        raise ConfigError(f"Expected {dotted}={expected!r}, got {value!r}")
