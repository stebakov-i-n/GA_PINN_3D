import math

import pytest

from cfd.config import ConfigError, resolve_config


def _base():
    return {
        "fluid": {"density_kg_m3": 1050.0, "dynamic_viscosity_pa_s": 0.003},
        "flow": {"mean_inlet_velocity_m_s": 0.75, "regime": "steady_incompressible_laminar", "gravity": False},
        "geometry": {"source_length_unit": "mm", "solver_length_unit": "m", "dataset_root": "SimVascDataset", "split_manifest": "full_split.json", "selection_manifest": "geometry_selection.json"},
        "boundary_conditions": {"wall": "no_slip", "outlet_gauge_pressure_pa": 0.0, "vascular_inlet_profile": "unresolved"},
        "vascular": {"inlet_profile": "uniform_normal", "mesh": {"background_cells": [24, 24, 48], "background_padding_fraction": 0.25, "surface_refinement_level": [0, 1], "feature_extract_included_angle_deg": 150, "location_in_mesh": "auto"}, "solver": {"max_final_continuity_local": 1e-4, "max_final_velocity_residual": 1e-4, "max_final_pressure_residual": 1e-4}},
        "openfoam": {"distribution": "OpenCFD", "version": "v2606", "solver": "simpleFoam", "end_time": 500, "write_interval": 100},
    }


def test_config_resolves_viscosity_and_unit_scale():
    cfg = resolve_config(_base())
    assert cfg["resolved"]["source_to_solver_scale"] == 0.001
    assert math.isclose(cfg["resolved"]["kinematic_viscosity_m2_s"], 0.003 / 1050.0)


def test_config_rejects_inconsistent_solver_unit():
    data = _base()
    data["geometry"]["solver_length_unit"] = "mm"
    with pytest.raises(ConfigError, match="solver_length_unit"):
        resolve_config(data)


def test_config_rejects_non_positive_openfoam_iterations():
    data = _base()
    data["openfoam"]["end_time"] = 0
    with pytest.raises(ConfigError, match="openfoam.end_time"):
        resolve_config(data)
