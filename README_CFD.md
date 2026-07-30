# CFD Reference Workflow

The independent CFD tools live under `cfd/`. They generate OpenFOAM reference
runs and reports without importing `main.py` or `modules.py`.

## Tune simpleFoam Iterations

Edit `cfd/config/baseline.yaml`:

```yaml
openfoam:
  distribution: OpenCFD
  version: v2606
  solver: simpleFoam
  end_time: 500
  write_interval: 100
```

For `simpleFoam`, `end_time` is the number of steady SIMPLE iterations. The
baseline is now 500. Try 1000 or 2000 only if the residual and profile error
still justify the longer run.

Then run:

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m cfd cylinder \
  --config cfd/config/baseline.yaml \
  --output-root cfd_runs \
  --run-openfoam
```

Each experiment writes a fresh directory in `cfd_runs/` with `report.md`,
`metrics.json`, `logs/`, `figures/`, and `solution/`.

## Outputs

Solved runs include:

- 3D velocity vector plot: `figures/openfoam_velocity_vectors_3d.png`
- 3D pressure distribution: `figures/openfoam_pressure_distribution_3d.png`
- full CFD cell-centre table: `solution/cfd_cell_center_fields.csv`
- sampled preview table: `solution/cfd_cell_center_fields_preview.csv`

The CSV table is the right substrate for PINN comparison: evaluate the PINN at
the same `x_m,y_m,z_m` cell-centre rows and compare `u_x,u_y,u_z,|u|,p_pa`.

## Tests

Default tests do not require OpenFOAM, Docker, CUDA, ClearML, or the medical
dataset:

```bash
MPLCONFIGDIR=/tmp/ga_pinn_cfd_matplotlib python3 -m pytest cfd/tests
```
