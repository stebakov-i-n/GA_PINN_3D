import math

import numpy as np

from cfd.config import resolve_config
from cfd.cylinder import analytical_metrics, generate_cylinder_bundle, pressure_gradient_pa_m, velocity_profile
from cfd.geometry import validate_bundle


def _config():
    return resolve_config(
        {
            "fluid": {"density_kg_m3": 1050.0, "dynamic_viscosity_pa_s": 0.003},
            "flow": {"mean_inlet_velocity_m_s": 0.75, "regime": "steady_incompressible_laminar", "gravity": False},
            "geometry": {"source_length_unit": "mm", "solver_length_unit": "m", "dataset_root": "SimVascDataset", "split_manifest": "full_split.json", "selection_manifest": "geometry_selection.json"},
            "boundary_conditions": {"wall": "no_slip", "outlet_gauge_pressure_pa": 0.0, "vascular_inlet_profile": "unresolved"},
            "vascular": {"inlet_profile": "uniform_normal", "mesh": {"background_cells": [24, 24, 48], "background_padding_fraction": 0.25, "surface_refinement_level": [0, 1], "feature_extract_included_angle_deg": 150, "location_in_mesh": "auto"}, "solver": {"max_final_continuity_local": 1e-4, "max_final_velocity_residual": 1e-4, "max_final_pressure_residual": 1e-4}},
            "openfoam": {"distribution": "OpenCFD", "version": "v2606", "solver": "simpleFoam", "end_time": 500, "write_interval": 100},
            "cylinder": {"radius_m": 0.001, "length_m": 0.04, "radial_samples": 51, "circumferential_segments": 32, "axial_segments": 4},
        }
    )


def test_hagen_poiseuille_relations():
    cfg = _config()
    radius = cfg["cylinder"]["radius_m"]
    mean = cfg["flow"]["mean_inlet_velocity_m_s"]
    r = np.array([0.0, radius])
    u = velocity_profile(r, radius, mean)
    assert math.isclose(u[0], 2 * mean)
    assert math.isclose(u[1], 0.0, abs_tol=1e-12)
    rs = np.linspace(0, radius, 10001)
    us = velocity_profile(rs, radius, mean)
    area_mean = 2 * np.trapz(us * rs, rs) / radius**2
    assert math.isclose(area_mean, mean, rel_tol=1e-7)
    assert math.isclose(pressure_gradient_pa_m(0.003, mean, radius), -8 * 0.003 * mean / radius**2)
    assert analytical_metrics(cfg)["centerline_velocity_m_s"] == 2 * mean


def test_synthetic_cylinder_geometry_validation(tmp_path):
    cfg = _config()
    generate_cylinder_bundle(tmp_path, cfg)
    result = validate_bundle(tmp_path / "geometry", "synthetic_cylinder/1_1.stl", "forward", cfg, tmp_path / "figures")
    assert result["status"] in {"validated", "validated_with_warnings"}
    assert result["files"]["closed"]["finite"] is True
    assert result["combined_surface"]["watertightness"]["status"] == "passed"
    assert math.isclose(result["inlet"]["hydraulic_diameter_m"], 0.002, rel_tol=0.01)
    assert (tmp_path / "figures" / "geometry_preview.png").exists()
