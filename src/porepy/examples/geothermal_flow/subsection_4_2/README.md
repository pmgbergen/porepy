# Subsection 4.2 — N-phase gravity segregation through barriers (HU-BM family)

Independent 2-D finite-volume reproduction of **Bosma et al. (2022, CMAME 388:114288),
Example 6.3 / Fig. 5**: immiscible, incompressible, gravity-driven phase segregation in a
closed 100 × 100 m box crossed by seven near-impermeable barrier layers with openings
(K = 1000 mD, φ = 0.3, μ = 1e-3, barrier cells K × 1e-4).

The solver is written for an **arbitrary number of phases N**. At `N = 3` it reproduces the
Bosma three-phase case (water / oil / gas, ρ = 1500 / 1000 / 500) *exactly*; at `N = 4` the oil
is split into a mid-heavy and a mid-light phase (ρ = 1500 / 1167 / 833 / 500), demonstrating that
the simplicial construction extends unchanged to true multiphase flow. Phase densities are
`numpy.linspace(1500, 500, N)`; the initial state stacks them heaviest-on-top / lightest-on-bottom
(an unstable configuration that then overturns).

## The HU-BM family

The three hybrid-upwinding schemes form the **HU-BM** family — *Hybrid Upwinding with Background
Mobility*. They share the simplicial buoyancy (a sum over the C(N,2) phase pairs, each pair
carrying a **background mobility** that aggregates the remaining N−2 phases; void at N = 2) and
differ only in the buoyant-pair form:

| token   | label        | buoyancy discretization                                    | notes |
|---------|--------------|------------------------------------------------------------|-------|
| `hu`    | **HU-BM(ff)** | simplicial fractional-flow `f_a f_b λ_T`                    | the robust, monotone member |
| `hu-mp` | **HU-BM(mp)** | mobility-product `λ_a λ_b / λ_T` (classical Lee/Hamon `Uᴴᵁ`)| PPU-sharp; **reduces exactly to Lee (2015) at N = 2** |
| `hu-mw` | **HU-BM(mw)** | fractional-flow buoyancy, harmonic (mobility-weighted) `λ_T`| total mobility in the face transmissibility |
| `ppu`   | PPU          | per-phase potential upwind                                  | *not* an HU-BM member (baseline) |

The short tokens on the left stay the canonical CLI / filename keys; the `HU-BM(...)` labels
appear in the printed output and figure titles.

---

## Files

| file                 | purpose                                                             |
|----------------------|---------------------------------------------------------------------|
| `hamon_2d_solver.py` | the N-phase FV solver (`run(...)` + standalone CLI)                  |
| `run_reference.py`   | driver: runs every scheme, writes `.vtr` snapshots + stats          |
| `plot_reference.py`  | builds the figures from the driver output                           |
| `run_workflow.py`    | **one command** — runs the sims *and* builds the figures, N = 3 & 4 |

---

## Requirements

Activate the PorePy environment (needs `numpy`, `scipy`, `pyvista`, `matplotlib`; `seaborn` and
`petsc4py` are optional — `petsc4py` enables the CPR linear solver, otherwise use
`--linear-solver scipy`). All commands below assume that environment is active; `OMP_NUM_THREADS`
is pinned to 1 automatically.

---

## Quick start — reproduce everything (N = 3 and N = 4)

```bash
python run_workflow.py
```

This runs all four schemes through both reference steps and builds every figure, for **both**
`N = 3` and `N = 4`, in one command. The independent simulation runs are executed in parallel
across worker processes.

```bash
python run_workflow.py --nphase 3          # only N = 3
python run_workflow.py --quick             # coarse/short smoke test (fast, not the paper run)
python run_workflow.py --jobs 4            # cap the parallel worker processes
python run_workflow.py --skip-run          # (re)build figures only, from existing output
python run_workflow.py --linear-solver scipy   # spsolve instead of CPR (robust for parallel runs)
```

Outputs per `N` (`_nN`-suffixed for `N ≠ 3` so nothing clobbers the N = 3 reference):

```
vtr[_nN]/                 step-1 snapshots (100², snaps 0/78/571) + stats_<scheme>.txt
output_ref_<scheme>[_nN]/ step-2 snapshots (200², snaps 0/78)     + stats_<scheme>.txt
figures[_nN]/             saturation_maps_<scheme>, saturation_grid_<scheme>, gas_comparison_78d
```

---

## Step by step (what `run_workflow.py` orchestrates)

### N = 3 (Bosma Ex. 6.3 — the paper figure)

```bash
# 1. run all four schemes  (step 1: 100^2 to 571 days, snaps 0/78/571  -> ./vtr/
#                           step 2: 200^2 to 78 days,  snaps 0/78      -> ./output_ref_<scheme>/)
python run_reference.py

# 2. build the figures  ->  ./figures/
python plot_reference.py
```

Per scheme this produces:
- `saturation_maps_<scheme>.png/.pdf` — the Fig. 5-style **diverging** map (blue = heaviest,
  white = middle, red = lightest) at 0 / 78 / 571 days. At N = 3 the scalar is `s_gas − s_water`;
- `saturation_grid_<scheme>.png/.pdf` — each phase saturation separately (`vlag`, 0..1);

plus one comparison figure `gas_comparison_78d.png/.pdf` (lightest phase across all schemes, each
panel titled with that scheme's Newton-iteration and time-step-cut counts).

### N = 4 (oil split into mid-heavy + mid-light)

```bash
python run_reference.py --nphase 4         # -> ./vtr_n4/  and  ./output_ref_<scheme>_n4/
python plot_reference.py --nphase 4        # -> ./figures_n4/
```

For `N = 4`:
- the diverging map generalizes to the density-ranked composite `Σ_k c_k s_k`,
  `c = linspace(−1, +1, N)` (blue = heaviest `s_0`, red = lightest `s_{N-1}`); at `N = 3` this is
  identical to `s_gas − s_water`;
- the per-phase **grid** (`saturation_grid_<scheme>`) is the figure to read for the two interior
  phases the composite collapses;
- the comparison figure shows the **lightest** phase `s_{N-1}`.

Any other `N` works too, e.g. `--nphase 5` → `vtr_n5/`, `figures_n5/`.

---

## Useful options

```bash
python run_reference.py --dry-run              # print the planned runs, execute nothing
python run_reference.py --step 1               # only the 100^2/571-day runs (or --step 2)
python run_reference.py --scheme hu-mp         # one scheme: hu=HU-BM(ff) hu-mw=HU-BM(mw) hu-mp=HU-BM(mp) ppu
python run_reference.py --linear-solver scipy  # spsolve instead of the default CPR
python run_reference.py --nphase 4 --step 1 --scheme hu   # combine freely

python plot_reference.py --maps                # only the diverging maps (or --grid / --gas)
python plot_reference.py --nphase 4 --days 0 78   # override snapshot days
python plot_reference.py --cmap vlag           # colormap (default vlag) for all figures
```

## Single standalone run (no driver)

`hamon_2d_solver.py` has its own CLI (writes `.vtr` + `stats.txt` to `./vtr/` by default):

```bash
# HU-BM(mp) at N=4, coarse grid, to 78 days
python hamon_2d_solver.py --scheme hu-mp --nphase 4 --nx 100 --ny 100 \
       --t-end-days 78 --snap-days 0 78 --out ./vtr_n4

# then plot from that directory
python plot_reference.py --nphase 4 --vtr-dir ./vtr_n4
```

`run(scheme, ..., nphase=N)` is also importable directly:

```python
import hamon_2d_solver as H
grid, snaps, stats = H.run("hu-mp", nx=100, ny=100, nphase=4, t_end_days=78, snap_days=(0.0, 78.0))
print(stats.summary())          # -> [HU-BM(mp)] steps=...  total_it=...
```

---

## Notes

- **Mass conservation.** All four schemes conserve every phase's mass to machine precision
  (~1e-11 relative) for both `N = 3` and `N = 4` on the closed domain.
- **N = 3 is exact.** Setting `N = 3` reproduces the Bosma densities and initial condition
  bit-for-bit; the generalized figures reduce identically to the original three-phase plots.
- **HU-BM(mp) ⊃ Lee (2015).** At `N = 2` the background mobility is void, so HU-BM(mp) collapses
  exactly to the original two-phase hybrid-upwinding buoyancy `λ_a λ_b / (λ_a + λ_b)`.
- **Linear solver.** The solution is independent of the linear solver (CPR and scipy solve the
  same system to tolerance), so the figures are identical. Everything defaults to the fast **CPR**
  solver; scipy's direct solve is impractically slow at the full 100²/200² scale. `run_workflow.py`
  runs the workers in parallel and shuts the pool down gracefully, so PETSc/MPI finalizes cleanly.
- Wall time for a full sequential `run_reference.py` is a few minutes per scheme on one core;
  `run_workflow.py` parallelizes the independent runs across processes.
