# Reproducible simulations for multiphase flow and transport in fractured geothermal systems

This directory contains the unified simulation and figure-generation workflow for reproducing the benchmark and numerical examples in the paper "**Oguntola et al. 2026: Mathematical Modeling of Salt Precipitation and Multi-Phase Flow in High Enthalpy Fractured Geothermal Systems.**"

The workflow supports the paper’s main computational results: verification of the PorePy implementation against the CSMP++ 1D salt benchmark, followed by 2D fractured-reservoir simulations that study halite precipitation and dissolution, permeability reduction, fracture and matrix clogging, and production performance in high-enthalpy geothermal systems.


The workflow has two stages:

1. **Simulation stage**  
   Runs a selected simulation case and writes standard PorePy visualization files to

   ```text
   visualization/<case_name>/
   ```

2. **Postprocessing and figure stage**  
   Uses the saved `.pvd` files, ParaView state files (`.pvsm`), CSV extraction scripts, and Matplotlib plotting scripts to reproduce the paper figures.

---

## Important dependency notes

### Example 1 and Example 2 are coupled for comparison figures

Examples 1 and 2 are paired cases. They use the same disconnected-fracture geometry, reference aperture, clogging exponents, but different injection rate:

| Case | Geometry | Injection rate ($\text{kgm}^{-3}\text{s}^{-1}$) |
|---|---|---:|
| `example1` | disconnected | `0.28` |
| `example2` | disconnected | `0.364` |

Several postprocessed figures depend on outputs from **both** simulations:

```text
figure14
figure15
```

In particular:

- `figure14` compares near-well halite saturation between Example 1 and Example 2.
- `figure15` compares production rate and energy production rate between Example 1 and Example 2.

Therefore, run both Example 1 and Example 2 before generating the comparison figures.

### Benchmark figure dependency

The benchmark figure requires:

```text
visualization/benchmark/benchmark.pvd
geothermal_flow/benchmark/reference_data/
```

The reference-data directory must contain the CSMP reference CSV files used for the benchmark comparison.

### ParaView dependency

Several figures are generated from saved ParaView state files. ParaView must be installed, and the `pvbatch` path in

```text
geothermal_flow/configs/figures.yaml
```

must match your local installation.

For example, on macOS:

```yaml
pvbatch: /Applications/ParaView-6.1.0.app/Contents/bin/pvbatch
```

If ParaView is installed elsewhere, update this path before running the figure workflow.

---

## Run simulations

Run commands from `src/porepy/examples`, or from a project root where the `geothermal_flow` module is importable.

### Benchmark

```bash
python -m geothermal_flow.simulation_driver \
  --config geothermal_flow/configs/benchmark.yaml
```

Expected output:

```text
visualization/benchmark/benchmark.pvd
```

### Example 1

```bash
python -m geothermal_flow.simulation_driver \
  --config geothermal_flow/configs/example1.yaml
```

Expected output:

```text
visualization/example1/example1.pvd
```

### Example 2

```bash
python -m geothermal_flow.simulation_driver \
  --config geothermal_flow/configs/example2.yaml
```

Expected output:

```text
visualization/example2/example2.pvd
```

### Example 3

```bash
python -m geothermal_flow.simulation_driver \
  --config geothermal_flow/configs/example3.yaml
```

Expected output:

```text
visualization/example3/example3.pvd
```

---

## Generate figures

All figure-generation instructions are stored in

```text
geothermal_flow/configs/figures.yaml
```

The figure workflow may perform one or more of the following operations depending on the figure:

- render a saved ParaView state file to PNG,
- render ParaView time-series panels,
- extract CSV data from a ParaView pipeline,
- extract time-series data from `.pvd` files,
- extract benchmark profiles from `.pvd` files,
- assemble panel figures,
- plot extracted CSV data with Matplotlib.

### Dry run

Before generating figures, check the commands that will be executed:

```bash
python -m geothermal_flow.make_figures \
  --config geothermal_flow/configs/figures.yaml \
  --dry-run
```

### Generate all figures

```bash
python -m geothermal_flow.make_figures \
  --config geothermal_flow/configs/figures.yaml
```

### Generate selected figures

```bash
python -m geothermal_flow.make_figures \
  --config geothermal_flow/configs/figures.yaml \
  --figures figure13
```

Multiple figures can be requested at once:

```bash
python -m geothermal_flow.make_figures \
  --config geothermal_flow/configs/figures.yaml \
  --figures figure13 figure14 figure15
```

---

## Recommended full reproduction order

A full reproduction of the simulation outputs and paper figures should follow this order:

```bash
python -m geothermal_flow.simulation_driver --config geothermal_flow/configs/benchmark.yaml
python -m geothermal_flow.simulation_driver --config geothermal_flow/configs/example1.yaml
python -m geothermal_flow.simulation_driver --config geothermal_flow/configs/example2.yaml
python -m geothermal_flow.simulation_driver --config geothermal_flow/configs/example3.yaml
```

Then generate all figures:

```bash
python -m geothermal_flow.make_figures \
  --config geothermal_flow/configs/figures.yaml
```

The final figures are written under:

```text
figures/
```

Intermediate extracted files are written under:

```text
csv/
output/
```

These intermediate files are generated automatically by the figure workflow when required.

---

## Configuration pattern

The simulation configuration files are:

```text
geothermal_flow/configs/defaults.yaml
geothermal_flow/configs/benchmark.yaml
geothermal_flow/configs/example1.yaml
geothermal_flow/configs/example2.yaml
geothermal_flow/configs/example3.yaml
```

`defaults.yaml` contains parameters common to the simulation cases.

Each case-specific YAML file contains the values that distinguish the case, including geometry, clogging exponent, reference aperture, injection settings, VTK thermodynamic tables, time stepping, and nonlinear-solver safeguards.

The figure configuration is separate:

```text
geothermal_flow/configs/figures.yaml
```

This file defines how each paper figure is generated from the simulation outputs.

---

## Case distinctions

| Case | Geometry | Clogging exponent `φ` | Reference aperture `a⁰` [m] | Injection multiplier | End time | PHZ table | Solver safeguards |
|---|---|---:|---:|---:|---:|---|---|
| `benchmark` | horizontal 1D benchmark | — | — | — | `2000 years` | `XHP_l2_original_salt_new.vtk` | no line search |
| `example1` | disconnected | `0.1` | `1e-3` | `1.0` | `74 days` | `XHP_l2_original_salt_new.vtk` | no line search |
| `example2` | disconnected | `0.1` | `1e-3` | `1.3` | `7 days` | `XHP_l2_original_salt_new.vtk` | line search |
| `example3` | connected | `2.0` | `1e-2` | `3.0` | `60 days` | `XHP_l2_original.vtk` | line search |

---

## Figures

The figure workflow currently covers:

| Figure | Source case(s) | Workflow |
|---|---|---|
| `figure6` | benchmark | Extract benchmark profile and compare with CSMP reference data |
| `figure8` | example1 | Render PHZ column states at selected times and assemble columns |
| `figure9` | example1 | Extract centerline CSV and plot with Matplotlib |
| `figure10` | example1 | Render phase saturation and permeability-ratio ParaView state |
| `figure11` | example1 | Render halite-saturation time panels and assemble with Matplotlib |
| `figure12` | example1 | Extract fracture-line CSVs and plot halite/aperture profiles |
| `figure13` | example2 | Extract fracture-line CSVs and plot halite/aperture profiles |
| `figure14` | example1 + example2 | Render near-well panels and assemble comparison |
| `figure15` | example1 + example2 | Extract production diagnostics and plot comparison |
| `figure16` | example3 | Render PHZ column states at selected times and assemble columns |
| `figure17` | example3 | Extract centerline CSV and plot with Matplotlib |
| `figure18` | example3 | Render phase-saturation ParaView state |

---

## Notes on generated data

The figure workflow writes intermediate CSV data to:

```text
csv/
```

and production-diagnostics caches to:

```text
output/
```

These files are generated from the simulation `.pvd` outputs and are safe to delete if the figures need to be regenerated.

Examples:

```text
csv/example1/plotover_line/
csv/example1/fracture_line/
csv/example2/fracture_line/
csv/example3/plotover_line/
csv/benchmark/

output/example1/production_diagnostics.csv
output/example2/production_diagnostics.csv
```

---

## Notes on ParaView state files

The ParaView-rendered figures depend on saved `.pvsm` state files under:

```text
geothermal_flow/paraview_states/
```

The state files store the visual setup, camera, layout, filters, scalar coloring, and colorbar placement. The figure-generation scripts replace the `.pvd` reader inside the saved state with the `.pvd` path defined in `figures.yaml`.

If a ParaView state file is renamed or moved, update the corresponding `state:` entry in `figures.yaml`.

---

## Troubleshooting

### `pvbatch: command not found`

Use the full ParaView path in `figures.yaml`, for example:

```yaml
pvbatch: /Applications/ParaView-6.1.0.app/Contents/bin/pvbatch
```

### ParaView OpenVKL warnings

You may see warnings similar to:

```text
[openvkl] INITIALIZATION ERROR
```

These warnings are usually not fatal for the current rendering workflow. Check whether the expected PNG or CSV was produced before treating them as a failure.

### Missing `.pvsm` state file

If a run fails with:

```text
State file not found
```

check the corresponding `state:` path in `figures.yaml` and verify the file exists under:

```text
geothermal_flow/paraview_states/
```

### Missing CSMP reference data

If `figure6` fails because a reference file is missing, check that the benchmark reference CSVs exist under:

```text
geothermal_flow/benchmark/reference_data/
```

The benchmark plotting script expects the CSMP reference files used for pressure, temperature, liquid saturation, and halite saturation.

### Figure 14 or Figure 15 fails

These figures depend on both Example 1 and Example 2 outputs. Make sure both simulations have been run before generating these comparison figures.

### Figure 15 production diagnostics missing

`figure15` automatically extracts production-cell time series and writes diagnostics caches under:

```text
output/example1/
output/example2/
```

If the production diagnostic CSVs are missing, rerun:

```bash
python -m geothermal_flow.make_figures \
  --config geothermal_flow/configs/figures.yaml \
  --figures figure16
```