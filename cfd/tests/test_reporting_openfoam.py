import json

from cfd.config import resolve_config
import numpy as np

from cfd.cylinder import generate_cylinder_bundle
from cfd.geometry import validate_bundle
from cfd.cli import build_parser
from cfd.openfoam import (
    discover_openfoam,
    generate_openfoam_case,
    generate_vascular_openfoam_case,
    parse_simplefoam_log,
    read_boundary_patches,
    read_boundary_scalar_values,
    read_mass_balance,
    read_scalar_field,
    read_vector_field,
    validate_vascular_solution_patches,
    write_node_field_report,
)
from cfd.reporting import create_run_dir, finish_run, start_metadata


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


def test_successful_report_generation(tmp_path):
    cfg = _config()
    run_dir = create_run_dir(tmp_path, "case", "forward", cfg)
    metadata = start_metadata(cfg, ["python", "-m", "cfd"], "case", "forward")
    finish_run(run_dir, cfg, metadata, {"analytical": {"reynolds_number": 525.0}}, "generated_not_executed")
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert "generated_not_executed" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_report_generation_after_controlled_failure(tmp_path):
    cfg = _config()
    run_dir = create_run_dir(tmp_path, "case", "forward", cfg)
    metadata = start_metadata(cfg, ["python", "-m", "cfd"], "case", "forward")
    finish_run(run_dir, cfg, metadata, {"failure": "controlled"}, "failed", "controlled")
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed"
    assert meta["failure_reason"] == "controlled"
    assert "controlled" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_openfoam_case_generation_and_discovery_skip(tmp_path):
    cfg = _config()
    case_dir = tmp_path / "case"
    generate_openfoam_case(case_dir, cfg)
    assert (case_dir / "system" / "blockMeshDict").exists()
    discovery = discover_openfoam(cfg)
    assert discovery["required_version"] == "v2606"
    assert isinstance(discovery["available"], bool)
    assert "simpleFoam" in discovery["commands"]


def test_openfoam_generated_files_have_standard_headers(tmp_path):
    cfg = _config()
    case_dir = tmp_path / "case"
    generate_openfoam_case(case_dir, cfg)

    expected = {
        "system/blockMeshDict": "blockMeshDict",
        "system/controlDict": "controlDict",
        "system/fvSchemes": "fvSchemes",
        "system/fvSolution": "fvSolution",
        "constant/transportProperties": "transportProperties",
        "constant/turbulenceProperties": "turbulenceProperties",
        "0/U": "U",
        "0/p": "p",
    }
    for rel_path, object_name in expected.items():
        text = (case_dir / rel_path).read_text(encoding="ascii")
        assert text.startswith("FoamFile\n{")
        assert f"object      {object_name};" in text


def test_control_dict_contains_required_time_controls(tmp_path):
    cfg = _config()
    case_dir = tmp_path / "case"
    generate_openfoam_case(case_dir, cfg)
    control_dict = (case_dir / "system" / "controlDict").read_text(encoding="ascii")

    assert "startTime 0;" in control_dict
    assert "endTime 500;" in control_dict
    assert "deltaT 1;" in control_dict
    assert "writeInterval 100;" in control_dict


def test_turbulence_properties_declares_laminar_flow(tmp_path):
    cfg = _config()
    case_dir = tmp_path / "case"
    generate_openfoam_case(case_dir, cfg)
    turbulence = (case_dir / "constant" / "turbulenceProperties").read_text(encoding="ascii")

    assert "simulationType laminar;" in turbulence


def test_fv_schemes_contains_simplefoam_laminar_divergence_terms(tmp_path):
    cfg = _config()
    case_dir = tmp_path / "case"
    generate_openfoam_case(case_dir, cfg)
    schemes = (case_dir / "system" / "fvSchemes").read_text(encoding="ascii")

    assert "div(phi,U) bounded Gauss linearUpwind grad(U);" in schemes
    assert "div((nuEff*dev2(T(grad(U))))) Gauss linear;" in schemes


def test_fv_solution_contains_required_pressure_smoother(tmp_path):
    cfg = _config()
    case_dir = tmp_path / "case"
    generate_openfoam_case(case_dir, cfg)
    solution = (case_dir / "system" / "fvSolution").read_text(encoding="ascii")

    assert "solver GAMG;" in solution
    assert "smoother GaussSeidel;" in solution
    assert "nNonOrthogonalCorrectors 2;" in solution
    assert "relaxationFactors" in solution
    assert "p 0.3;" in solution
    assert "U 0.3;" in solution


def test_openfoam_cylinder_mesh_uses_circular_arcs_not_square_corners(tmp_path):
    cfg = _config()
    case_dir = tmp_path / "case"
    generate_openfoam_case(case_dir, cfg)
    block_mesh = (case_dir / "system" / "blockMeshDict").read_text(encoding="ascii")
    radius = cfg["cylinder"]["radius_m"]
    corner = f"(-{radius} -{radius} 0)"

    assert block_mesh.count("arc ") == 8
    assert "scale 1;" in block_mesh
    assert "convertToMeters" not in block_mesh
    assert "blocks\n(" in block_mesh
    assert "((hex" not in block_mesh
    assert corner not in block_mesh
    assert f"({radius} 0 0)" in block_mesh
    assert f"(0 {radius} 0)" in block_mesh


def test_openfoam_inlet_uses_nonuniform_hagen_poiseuille_profile(tmp_path):
    cfg = _config()
    case_dir = tmp_path / "case"
    generate_openfoam_case(case_dir, cfg)
    u_field = (case_dir / "0" / "U").read_text(encoding="ascii")

    assert "type fixedValue;" in u_field
    assert "nonuniform List<vector>" in u_field
    assert "\n        400\n" in u_field
    assert "(0 0 1.49625)" in u_field
    assert "(0 0 0.14625)" in u_field
    assert "codedFixedValue" not in u_field
    assert "value uniform (0 0 0.75);" not in u_field


def test_simplefoam_log_parser_extracts_completion_and_continuity(tmp_path):
    log = tmp_path / "simpleFoam.stdout.log"
    log.write_text(
        """Time = 1
smoothSolver:  Solving for Ux, Initial residual = 1e-03, Final residual = 1e-07, No Iterations 2
smoothSolver:  Solving for Uy, Initial residual = 2e-03, Final residual = 1e-07, No Iterations 2
smoothSolver:  Solving for Uz, Initial residual = 3e-03, Final residual = 1e-07, No Iterations 2
GAMG:  Solving for p, Initial residual = 4e-02, Final residual = 1e-06, No Iterations 4
time step continuity errors : sum local = 5e-05, global = -6e-06, cumulative = 7e-04
End
""",
        encoding="utf-8",
    )
    parsed = parse_simplefoam_log(log)

    assert parsed["completed"] is True
    assert parsed["final_time"] == 1.0
    assert parsed["final_continuity_sum_local"] == 5e-05
    assert parsed["initial_residuals"]["Uz"] == [3e-03]
    assert parsed["initial_residuals"]["p"] == [4e-02]


def test_openfoam_internal_field_readers_parse_nonuniform_outputs(tmp_path):
    u = tmp_path / "U"
    p = tmp_path / "p"
    u.write_text(
        """FoamFile
{
    object U;
}
internalField nonuniform List<vector>
2
(
(0 0 1.25)
(0.1 0 0.5)
)
;
boundaryField
{
}
""",
        encoding="ascii",
    )
    p.write_text(
        """FoamFile
{
    object p;
}
internalField nonuniform List<scalar>
3
(
1
2.5
-3
)
;
boundaryField
{
}
""",
        encoding="ascii",
    )

    assert read_vector_field(u).tolist() == [[0.0, 0.0, 1.25], [0.1, 0.0, 0.5]]
    assert read_scalar_field(p).tolist() == [1.0, 2.5, -3.0]


def test_node_field_report_writes_pinn_comparison_table(tmp_path):
    case_dir = tmp_path / "run" / "openfoam_case"
    case_dir.mkdir(parents=True)
    centers = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.01]])
    velocity = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.5]])
    speed = np.linalg.norm(velocity, axis=1)
    p_pa = np.array([100.0, 50.0])
    p_kinematic = p_pa / 1050.0

    report = write_node_field_report(case_dir, centers, velocity, speed, p_pa, p_kinematic)
    full_csv = tmp_path / "run" / "solution" / "cfd_cell_center_fields.csv"
    preview_csv = tmp_path / "run" / "solution" / "cfd_cell_center_fields_preview.csv"

    assert report["cell_count"] == 2
    assert full_csv.exists()
    assert preview_csv.exists()
    text = full_csv.read_text(encoding="utf-8")
    assert "cell_id,x_m,y_m,z_m,u_x_m_s,u_y_m_s,u_z_m_s,speed_m_s,p_pa,p_kinematic_m2_s2" in text
    assert "Evaluate PINN velocity and pressure" in report["pinn_comparison_key"]


def test_vascular_openfoam_case_generation_uses_patch_surfaces_and_inward_inlet(tmp_path):
    cfg = _config()
    generate_cylinder_bundle(tmp_path, cfg)
    validation = validate_bundle(tmp_path / "geometry", "synthetic_cylinder/1_1.stl", "forward", cfg, tmp_path / "figures")
    case_dir = tmp_path / "run" / "openfoam_case"

    result = generate_vascular_openfoam_case(case_dir, tmp_path / "geometry", "synthetic_cylinder/1_1.stl", "forward", cfg, validation)

    assert result["surface_mapping"]["wall.stl"] == "wall"
    assert result["surface_mapping"]["closed.stl"] == "debug_only"
    assert (case_dir / "constant" / "triSurface" / "wall.stl").exists()
    assert (case_dir / "system" / "snappyHexMeshDict").exists()
    assert "wall.stl { type triSurfaceMesh; name wall; }" in (case_dir / "system" / "snappyHexMeshDict").read_text(encoding="ascii")
    assert "vertex 0.001" in (case_dir / "constant" / "triSurface" / "inlet.stl").read_text(encoding="ascii")
    u_field = (case_dir / "0" / "U").read_text(encoding="ascii")
    assert "inlet" in u_field
    assert "value uniform" in u_field
    assert result["inlet_profile"]["profile"] == "uniform_normal"
    assert result["inlet_profile"]["velocity_m_s"][2] < 0.0


def test_openfoam_boundary_patch_parser_reports_patch_face_counts(tmp_path):
    boundary = tmp_path / "boundary"
    boundary.write_text(
        """FoamFile
{
    object boundary;
}
3
(
    wall
    {
        type wall;
        nFaces 12;
        startFace 100;
    }
    inlet
    {
        type patch;
        nFaces 3;
        startFace 112;
    }
    outlet
    {
        type patch;
        nFaces 4;
        startFace 115;
    }
)
""",
        encoding="ascii",
    )

    patches = read_boundary_patches(boundary)

    assert patches["wall"]["type"] == "wall"
    assert patches["inlet"]["nFaces"] == 3
    assert patches["outlet"]["startFace"] == 115


def test_vessel_cli_accepts_guarded_run_openfoam_mode():
    args = build_parser().parse_args(
        [
            "vessel",
            "--config",
            "cfd/config/baseline.yaml",
            "--geometry",
            "0032_H_ABAO_AAA/1_10.stl",
            "--direction",
            "forward",
            "--output-root",
            "cfd_runs",
            "--run-openfoam",
        ]
    )

    assert args.run_openfoam is True


def test_vascular_solution_patch_validation_rejects_zero_face_inlet(tmp_path):
    poly = tmp_path / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    (poly / "boundary").write_text(
        """3
(
wall
{
    type wall;
    nFaces 12;
    startFace 0;
}
inlet
{
    type patch;
    nFaces 0;
    startFace 12;
}
outlet
{
    type patch;
    nFaces 4;
    startFace 12;
}
)
""",
        encoding="ascii",
    )

    try:
        validate_vascular_solution_patches(tmp_path)
    except ValueError as exc:
        assert "zero-face" in str(exc)
    else:
        raise AssertionError("zero-face inlet patch should fail validation")


def test_boundary_phi_parser_and_mass_balance(tmp_path):
    phi = tmp_path / "phi"
    phi.write_text(
        """FoamFile
{
    object phi;
}
boundaryField
{
    inlet
    {
        type calculated;
        value nonuniform List<scalar>
        2
        (
        -1.0e-6
        -2.0e-6
        )
        ;
    }
    outlet
    {
        type calculated;
        value nonuniform List<scalar>
        2
        (
        1.5e-6
        1.4e-6
        )
        ;
    }
}
""",
        encoding="ascii",
    )

    assert read_boundary_scalar_values(phi, "inlet").tolist() == [-1.0e-6, -2.0e-6]
    balance = read_mass_balance(phi)
    assert balance["status"] == "checked"
    assert balance["inlet_flow_rate_m3_s"] == -3.0e-6
    assert balance["outlet_flow_rate_m3_s"] == 2.9e-6
