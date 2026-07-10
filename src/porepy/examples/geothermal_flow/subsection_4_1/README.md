# Subsection 4.1 — Weis (2014) fig-5 verification figures

Scripts that build the paper's §4.1 figures. Two solvers:

- `weis_1d_solver.py` — fast, PorePy-independent 1-D finite-volume **reference** engine
  (imported by the figure scripts; no CLI).
- `porepy_1d_solver.py` — the **PorePy 2-D column** runner (produces the overlay data).

…and three figure scripts (`fig_weis_reference.py`, `fig_weis_profiles.py`,
`fig_weis_verification.py`). Outputs go to `figures/` (PDF + PNG); intermediate runs are
cached in `_cache/` and are **resumable** — delete the relevant cache files to force a recompute.

## Current configuration
| knob | value |
|---|---|
| spatial resolution | `N = 800` |
| nominal time step | `Δt = 0.25 yr` |
| Driesner table level | **3** (production); level **4** used *only* as the reference in `fig_weis_reference_b` |
| Newton stop criterion | absolute, per-equation residual `< 1e-5` (both solvers) |

## Environment
Use the `porepy` conda env and pin BLAS threads (avoids oversubscription — the multi-threaded
BLAS otherwise appears to hang on the heavy PorePy runs):

```bash
PY=~/miniconda/envs/porepy/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd src/porepy/examples/geothermal_flow/subsection_4_1
```

---

## Steps

### 1. `fig_weis_reference.py` → `figures/fig_weis_reference_{a,b}.pdf`
Convergence of the 1-D reference solver. **(a)** spatial refinement (`N = 100,200,400,800`);
**(b)** OBL table-level refinement (levels `0..3` vs the level-4 reference — shows level 3 is
already close enough that the finer table is not needed). 1-D solver only (no PorePy).

```bash
$PY fig_weis_reference.py
```
> Panel (b) loads the **level-4** table (~1.4 GB) once per scheme — the heavy part of this figure.

### 2. `fig_weis_profiles.py` → `figures/fig_weis_profiles_{a,b}.pdf`
Converged profiles of the three schemes (PPU, HU, HU-mw) against the digitized Weis-2014 curves
(thick pale band); **(a)** horizontal (200 yr), **(b)** vertical (1000 yr). 1-D solver only.
Also writes the `_cache/profiles_*` files that step 4 reuses as its 1-D reference.

```bash
$PY fig_weis_profiles.py
```

### 3. PorePy 2-D data → `_cache/porepy_{case}_{scheme}_N800_l3.pkl`  *(prerequisite for step 4)*
Runs the PorePy 2-D column for the selected cases and dumps each converged profile.
**Heavy** (vertical ≈ hours). Edit `main()` to choose cases — currently `vertical` × {HU, HU-mw};
set both orientations for all four `{horizontal, vertical} × {HU, HU-mw}`.

```bash
$PY porepy_1d_solver.py
```
> Also writes solver stats to `_cache/porepy_*_stats.{txt,pkl}`.

### 4. `fig_weis_verification.py` → `figures/fig_weis_verification_{horizontal,vertical}.pdf`
Overlays each PorePy 2-D scheme (thin dark line) on its 1-D reference (thick pale band).
**Cache-only / fast** — it renders whatever is already in `_cache/`: the `profiles_*` files from
step 2 and the `porepy_*` files from step 3. A scheme missing its PorePy pickle is drawn as the
1-D reference band alone.

```bash
$PY fig_weis_verification.py
```

---

## Typical order
1. `fig_weis_reference.py`     — independent
2. `fig_weis_profiles.py`      — independent; also builds the 1-D reference caches
3. `porepy_1d_solver.py`       — heavy; builds the PorePy caches
4. `fig_weis_verification.py`  — renders the overlays from steps 2 + 3
