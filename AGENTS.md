# GA_PINN_3D agent instructions

These instructions apply to the entire repository unless a more specific
`AGENTS.md` exists in a subdirectory.

## Project goal

The existing project trains one geometry-aware PINN for steady,
incompressible, Newtonian flow in multiple 3D vascular geometries. The new
`cfd/` subproject must provide an independent conventional CFD reference for
quantitative PINN validation.

CFD and PINN may share immutable inputs and physical assumptions, but they
must not share a numerical solution implementation.

## Read before changing anything

At the start of every task:

1. Read this file, `README.md`, and `CFD_task.md` completely.
2. Inspect `git status --short`, the current branch, and the relevant files.
3. Preserve all pre-existing user changes. Never discard or overwrite
   unrelated work.
4. State the planned files and validation commands before editing.

Read `main.py` and `modules.py` when needed to understand the current PINN,
but do not treat comments or variable names as proof of mathematical
equivalence. Check the actual computation.

## Repository boundaries

The CFD implementation belongs under `cfd/` and must be independently
executable.

It may read these existing inputs without modifying them:

- `full_split.json`;
- `geometry_selection.json`;
- the `SimVascDataset/` STL bundles;
- exported PINN checkpoints or predictions in later comparison work.

Do not import or execute `main.py` from the CFD package. Avoid a runtime
dependency on `modules.py`; the CFD package should have small, tested readers
for the shared file formats it needs.

Unless the user explicitly expands the scope, do not edit:

- `main.py`;
- `modules.py`;
- `select_geometries.py`;
- any notebook;
- `full_split.json`;
- `geometry_selection.json`;
- dataset files or cached point clouds.

Allowed root-level changes are limited to:

- `AGENTS.md`;
- a narrowly appended, clearly labelled CFD section in `.gitignore`;
- a short link in `README.md` only when requested.

Never perform repository-wide formatting or dependency cleanup as part of a
CFD task.

## Known PINN-side issues are separate work

The following suspected issues have been identified, but are out of scope for
the first CFD tasks:

- reverse-direction outlet geometry may load the inlet cap;
- the pressure distance factor may be disconnected from the coordinates used
  for autograd;
- validation residual selection may be inconsistent between training stages.

Record relevant observations as follow-up items. Do not silently fix them.
Any PINN-core correction requires a separate task, tests, and commit.

## Physical contract

Use SI units inside CFD calculations and exported fields:

- density: `1050 kg/m^3`;
- dynamic viscosity: `0.003 Pa s`;
- kinematic viscosity: compute `mu / rho`;
- target mean inlet speed: `0.75 m/s`;
- steady, incompressible, Newtonian, laminar flow;
- no gravity or other body force;
- no-slip vessel wall;
- outlet gauge pressure: `0 Pa`;
- source STL coordinates: millimetres;
- solver coordinates: metres.

Support both directions:

- forward: `_2.stl` is inlet and `_1.stl` is outlet;
- reverse: `_1.stl` is inlet and `_2.stl` is outlet.

Compute the actual inlet area, perimeter, hydraulic diameter, and Reynolds
number for every geometry. Flag questionable regimes; do not assume laminar
flow solely because the configured solver is laminar.

### Inlet-profile caveat

`CFD_task.md` calls the inlet profile parabolic, while the current PINN forms
it from the learned `phi_1` distance-like field. Distance to a wall is not, in
general, the same function as a Poiseuille profile.

- For the straight circular verification case, use the analytical
  Hagen-Poiseuille profile.
- For irregular vascular sections, keep the profile strategy explicit and
  configurable.
- Do not choose or claim a PINN-equivalent vascular inlet profile without
  user approval and a numerical equivalence test.

OpenFOAM commonly represents incompressible pressure as kinematic pressure.
When that convention is used, export physical gauge pressure in pascals as
`p_pa = rho * p_kinematic`, and record the conversion.

## Geometry and manifest rules

Treat shared manifests as read-only. `geometry_selection.json` currently uses
Windows-style backslashes in keys; normalize both `\` and `/` safely rather
than rewriting the manifest.

Geometry validation must report, at minimum:

- existence and readability of the closed surface, wall, and two caps;
- finite, non-degenerate triangles;
- source and converted units;
- cap area, perimeter, planarity, centres, and normals;
- patch gaps/intersections and combined-surface watertightness;
- direction/orientation checks and their confidence;
- geometry hashes;
- explicit failure reasons.

Do not silently repair, remesh, close, smooth, or reorient a medical geometry.
Validation and repair are different operations.

## OpenFOAM and execution safety

Pin and record the intended OpenFOAM distribution/version; the current target
is OpenCFD OpenFOAM v2606. Do not silently use an arbitrary installed version.

Do not install system packages, pull a large container image, launch a large
mesh, or run real vascular simulations unless the user explicitly requests
it. Case generation and CPU-only smoke tests are safe defaults.

Every external command must be invoked without `shell=True`, have captured
logs, a checked exit code, and a clear failure message.

## Outputs and reproducibility

Raw CFD outputs are heavy. Write them to a configurable output root, not to a
tracked source directory. A local fallback such as `cfd_runs/` is acceptable
only when it is ignored by Git.

Every attempted run, including a failed run, must create a unique run
directory containing:

- `resolved_config.yaml`;
- `metadata.json`;
- `metrics.json`;
- `report.md`;
- `figures/`;
- `logs/`;
- generated case/solution files when applicable.

Metadata must include the Git commit, dirty-tree flag, timestamps, geometry
and file hashes, direction, units, resolved physics, OpenFOAM version, command
line, status, runtime, and failure reason when applicable.

Reports must distinguish clearly between:

- generated but not executed;
- executed but not converged;
- converged and validated.

Never claim that OpenFOAM or the CFD pipeline works unless the corresponding
command was actually run successfully.

## Figures and metrics

Generate compact diagnostic figures automatically rather than relying on
manual notebook work. Depending on the stage, include:

- geometry/patch preview;
- analytical and numerical inlet/velocity profiles;
- residual or convergence history;
- pressure profile/drop;
- mass-balance diagnostics.

For later CFD-PINN comparison, evaluate the PINN directly at CFD cell centres
instead of interpolating CFD onto PINN points. That comparison is not part of
the initial preprocessing/cylinder task.

## Testing

New functionality must have deterministic, CPU-only tests that do not require
the medical dataset, ClearML, CUDA, Docker, or OpenFOAM.

At minimum, test:

- configuration validation and unit conversion;
- manifest parsing and cross-platform path normalization;
- analytical circular-pipe velocity and pressure-gradient relations;
- synthetic geometry validation;
- report generation, including a controlled failure.

If OpenFOAM is installed, an additional integration smoke test may run, but
the default test suite must skip it cleanly when unavailable.

Run the smallest relevant checks first, then the full CFD test suite. Report
exactly what was and was not executed.

## Git discipline

- Work on a feature branch, not `main`.
- Do not commit, push, open a PR, or modify remote resources unless explicitly
  asked.
- Never use `git add -A`; stage explicit paths if staging is requested.
- Keep generated data, meshes, solver time directories, checkpoints, and
  reports out of Git.
- Do not reorganize the existing `.gitignore`; append only narrow CFD rules.
- End each task with `git status --short`, a concise changed-file list, test
  results, and unresolved scientific or execution limitations.

