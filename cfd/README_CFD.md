# CFD Simulation Runbook

Independent CFD tools for GA_PINN_3D. Run commands from the repository root.
The CFD package reads manifests and STL bundles, but does not import `main.py`
or `modules.py`.

## Purpose

The PINN project predicts steady incompressible Newtonian flow in 3D vessel
domains. This CFD subproject provides an independent OpenFOAM reference path
for checking those predictions.

Physics used by default:

- density: `1050 kg/m^3`
- dynamic viscosity: `0.003 Pa s`
- mean inlet speed: `0.75 m/s`
- steady, incompressible, laminar, no gravity
- wall: no-slip, `U = 0`
- outlet gauge pressure: `p = 0 Pa`
- dataset STL coordinates: millimetres
- OpenFOAM coordinates and exported fields: SI units

OpenFOAM incompressible pressure is kinematic. Reports and CSV files export
physical pressure as `p_pa = rho * p_kinematic`.

## Setup

```bash
source .venv/bin/activate
python3 -m pip install -r cfd/requirements.txt
docker pull opencfd/openfoam-run:2606
```

Docker Desktop must be running for OpenFOAM solves on macOS. The default tests
do not require Docker, OpenFOAM, CUDA, ClearML, or the medical dataset.

## Configure

Edit `cfd/config/baseline.yaml`.

```yaml
flow:
  mean_inlet_velocity_m_s: 0.75

openfoam:
  distribution: OpenCFD
  version: v2606
  solver: simpleFoam
  end_time: 500
  write_interval: 100

vascular:
  inlet_profile: uniform_normal
```

For `simpleFoam`, `end_time` is the maximum number of steady SIMPLE iterations.
The solver may stop earlier when OpenFOAM residual controls are satisfied.

## Test

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m pytest cfd/tests
```

## Pipe Reference

Generate a synthetic circular pipe case and report without OpenFOAM:

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m cfd cylinder \
  --config cfd/config/baseline.yaml \
  --output-root cfd_runs
```

Run the pipe OpenFOAM solve and automatic analysis:

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m cfd cylinder \
  --config cfd/config/baseline.yaml \
  --output-root cfd_runs \
  --run-openfoam
```

The pipe case uses the analytical Hagen-Poiseuille inlet profile and reports
numerical-vs-analytical diagnostics.

## Vessel Dataset Workflow

Dataset flow domains have this layout:

```text
SimVascDataset/<case_id>/<k>_<m>.stl
SimVascDataset/<case_id>/<k>_<m>_1.stl
SimVascDataset/<case_id>/<k>_<m>_2.stl
SimVascDataset/<case_id>/<k>_<m>_3.stl
```

Patch meaning:

- `<k>_<m>.stl`: closed vessel flow domain
- `<k>_<m>_3.stl`: vessel wall
- `<k>_<m>_1.stl`, `<k>_<m>_2.stl`: end caps
- `forward`: `_2` is inlet, `_1` is outlet
- `reverse`: `_1` is inlet, `_2` is outlet

List manifest-selected geometries:

```bash
python3 -m cfd list-geometries --config cfd/config/baseline.yaml
```

Choose the mode by the output you need:

| Need | Flag | Produces velocity/pressure solution? |
|---|---|---|
| Validate STL bundle and make geometry preview | `--validate-only` | No |
| Generate OpenFOAM case files only | `--generate-openfoam` | No |
| Generate mesh and run `checkMesh` only | `--mesh-openfoam` | No |
| Mesh, solve with `simpleFoam`, and postprocess | `--run-openfoam` | Yes |

Validate one vessel bundle:

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m cfd vessel \
  --config cfd/config/baseline.yaml \
  --geometry 0032_H_ABAO_AAA/1_10.stl \
  --direction forward \
  --output-root cfd_runs \
  --validate-only
```

Generate a vascular OpenFOAM case without running OpenFOAM:

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m cfd vessel \
  --config cfd/config/baseline.yaml \
  --geometry 0032_H_ABAO_AAA/1_10.stl \
  --direction forward \
  --output-root cfd_runs \
  --generate-openfoam
```

Mesh and run `checkMesh`, but do not solve:

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m cfd vessel \
  --config cfd/config/baseline.yaml \
  --geometry 0032_H_ABAO_AAA/1_10.stl \
  --direction forward \
  --output-root cfd_runs \
  --mesh-openfoam
```

Expected status: `mesh_validated`. This is a good pre-solve safety check, but
it does not create `solution/` or velocity/pressure figures.

Mesh, solve with `simpleFoam`, and postprocess velocity/pressure results:

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m cfd vessel \
  --config cfd/config/baseline.yaml \
  --geometry 0032_H_ABAO_AAA/1_10.stl \
  --direction forward \
  --output-root cfd_runs \
  --run-openfoam
```

`--run-openfoam` validates the final mesh patches before solving. If `wall`,
`inlet`, or `outlet` are missing or have zero faces, `simpleFoam` is not run.
This is the command that creates `solution/cfd_cell_center_fields.csv`,
`figures/openfoam_velocity_vectors_3d.png`, and
`figures/openfoam_pressure_distribution_3d.png` when the solve succeeds.

## Vessel Boundary Conditions

Current vessel inlet profile:

```text
vascular.inlet_profile = uniform_normal
```

This sets a uniform velocity vector with magnitude `0.75 m/s`, oriented along
the inlet cap normal into the vessel. The report records the selected vector and
orientation confidence.

This is a clear CFD boundary condition, but it is not proven equivalent to the
PINN inlet construction based on learned distance-like fields. Treat
PINN-equivalence as `not established` until a numerical comparison is run.

Boundary values:

- wall `U`: exact no-slip, `U = (0,0,0)`
- inlet `U`: fixed uniform normal velocity
- outlet `p`: exact fixed gauge pressure, `p_pa = 0`
- inlet `p`: zero-gradient, inferred from adjacent internal cells

Boundary checks in `report.md` distinguish exact patch boundary conditions from
near-boundary cell-centre samples. Wall-adjacent cells can have nonzero speed;
that is normal for finite-volume data and does not violate the no-slip wall
condition.

## Run Outputs

Each attempted run creates a unique directory under `cfd_runs/`:

- `report.md`: status, physics, geometry, OpenFOAM diagnostics, figures, logs
- `metadata.json`: command, timestamps, git state, runtime, failure reason
- `metrics.json`: machine-readable metrics
- `resolved_config.yaml`: exact resolved inputs
- `figures/`: generated diagnostic figures
- `logs/`: captured OpenFOAM stdout/stderr
- `openfoam_case/`: generated case and OpenFOAM time directories
- `solution/`: solved field exports when a solver run succeeds

Solved vessel outputs include:

- `solution/cfd_cell_center_fields.csv`
- `solution/cfd_cell_center_fields_preview.csv`
- `figures/openfoam_velocity_vectors_3d.png`
- `figures/openfoam_pressure_distribution_3d.png`
- `figures/openfoam_residual_history.png`

The full CSV contains one row per OpenFOAM finite-volume cell centre:

```text
cell_id,x_m,y_m,z_m,u_x_m_s,u_y_m_s,u_z_m_s,speed_m_s,p_pa,p_kinematic_m2_s2
```

These are cell-centre values, not boundary vertices. Use them for PINN
comparison by evaluating the PINN at the same `x_m,y_m,z_m` coordinates and
comparing `u_x,u_y,u_z,speed,p_pa` row by row.

## Status Meaning

- `generated_not_executed`: validation/case generation only
- `mesh_failed`: OpenFOAM meshing/checkMesh failed
- `mesh_validated`: meshing and `checkMesh` passed, no flow solution
- `executed_not_converged`: solver ran but convergence checks did not pass
- `converged_validated`: solver and smoke diagnostics passed
- `failed`: validation or pipeline error

## Current Limitations

- Vessel inlet is `uniform_normal`, not a proven PINN-equivalent profile.
- Vascular meshes use conservative default `snappyHexMesh` settings; refine per
  geometry before treating a run as high-fidelity.
- CFD-PINN comparison is not automated yet, but solved CSV files are already in
  the right coordinate format for it.
