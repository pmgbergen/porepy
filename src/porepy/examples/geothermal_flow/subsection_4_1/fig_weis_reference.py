"""Figure ``fig:weis_reference`` -- construction of the 1D references for the Weis (2014)
benchmark, vertical orientation with face-averaged (consistent) gravity densities.

(a) Spatial convergence of PPU / HU / HU-mw at a fixed, deliberately small time step, so the
    spatial error dominates: combined relative L2 error (T, s_liq, p) of each level against the
    scheme's own finest solution, log-log vs the cell size h, with a first-order guide; the
    lower sub-panel reports the average Newton iterations per level (the smooth mobility-weighted
    variant vs the switch at m_e = 0 of the upwinded weight).
(b) Parametric (OBL table) convergence: at a fixed spatial resolution, the error against the
    scheme's own finest table level, decreasing with the level until the spatial error dominates.

The runs (fine N x small dt x full time) are heavy; they are cached to pickles keyed by the
configuration -- delete them to recompute. Smoke-test cheaply by calling compute_spatial /
compute_obl with small N_levels and n_steps from a REPL.

    python fig_weis_reference.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weis_1d_solver as m  # noqa: E402
import plot_style as ps     # noqa: E402

# --- configuration (all heavy; cached) ----------------------------------------------------
CASE = "vertical"                      # references are built where gravity is active
GRAV_UPSTREAM = False                  # gravity density: False = averaged/consistent (Rem.gc),
#                                        True = upwinded (Weis). Tagged into the cache names below.
LAG_UPWIND = False                     # advective nonlinear weight, UNIFORM across schemes:
#                                        False = current iterate (fully implicit; PPU carries its
#                                        phase-potential switch, HU the m_e=0 switch, HU-mw the
#                                        within-Newton multipoint reassembly). True = old-state,
#                                        frozen once per step (HU-mw: one lambda*K assembly/step).
#                                        Tagged (cur/lag) into the cache names below.
N_LEVELS = [100, 200, 400, 800]        # spatial refinement ladder; finest is the reference
DT_FIXED = m.DT0 / 4.0                 # fixed, small time step so the spatial error dominates
OBL_LEVELS = [0, 1, 2, 3, 4]           # Driesner table refinement levels; finest (4) is the reference
# Panel (b) uses the SAME-N reference (error = level L vs level 4 at this same N), so the spatial
# error cancels and the pure OBL/table error is isolated at ANY N. Finer N only shrinks the
# residual near-front contamination; N=1600 is cleanest but its 18 runs cost ~4x those at N=400.
N_OBL = 800                            # spatial resolution for the OBL-convergence sweep (knob)
LEVEL_SPATIAL = m.TABLE_LEVEL          # OBL level used for the spatial-convergence panel
ERR_FIELDS = ("T", "s_liq", "p")       # fields entering the combined relative L2 error
OUT_DIR = os.path.join(m.HERE, "figures")


# --- error metric -------------------------------------------------------------------------
def combined_error(res, ref, npts=2000):
    """Combined relative L2 error of ``res`` vs ``ref`` over ERR_FIELDS, evaluated on a common
    dense grid (grid-independent): sqrt(mean_field (||f-f_ref|| / ||f_ref||)^2)."""
    xs, xr = res["y"], ref["y"]
    xe = np.linspace(max(xs[0], xr[0]), min(xs[-1], xr[-1]), npts)
    rel = []
    for fld in ERR_FIELDS:
        fe = np.interp(xe, xs, res[fld])
        fr = np.interp(xe, xr, ref[fld])
        rel.append(np.linalg.norm(fe - fr) / (np.linalg.norm(fr) + 1e-30))
    return float(np.sqrt(np.mean(np.square(rel))))


# --- runs (per-run cache in _cache/ -> resumable + observable) ----------------------------
CACHE_DIR = os.path.join(m.HERE, "_cache")


def _dens_tag(grav_upstream):
    return "up" if grav_upstream else "avg"


def _lag_tag(lag_upwind):
    return "lag" if lag_upwind else "cur"


def _run_path(kind, sk, N, level, dt, case, n_steps, grav_upstream, lag_upwind):
    ns = "" if n_steps is None else f"_ns{n_steps}"
    return os.path.join(
        CACHE_DIR,
        f"{kind}_{case}_{_dens_tag(grav_upstream)}_{_lag_tag(lag_upwind)}_{sk}"
        f"_N{N}_l{level}_dt{dt / m.YEAR:.5g}yr{ns}.pkl")


def _run(args):
    """Run one task at fixed small dt (adaptive off) with the chosen gravity-density and advective
    (lag/current) treatments. If the per-run cache exists it is loaded (skip); otherwise the run
    executes and its result is written to disk IMMEDIATELY -> the sweep is resumable and its
    progress is visible on disk. Returns (key, result, wall_seconds, was_cached)."""
    kind, key, sk, N, level, dt, case, n_steps, grav_upstream, lag_upwind = args
    path = _run_path(kind, sk, N, level, dt, case, n_steps, grav_upstream, lag_upwind)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return key, pickle.load(f), 0.0, True
    cfg = ps.SCHEMES[sk]
    t0 = time.time()
    res = m.run(scheme=cfg["scheme"], weighted_perm=cfg["weighted_perm"],
                grav_upstream=grav_upstream, N=N, case=case, level=level, dt=dt,
                adaptive=False, n_steps=n_steps, verbose=False, lag_upwind=lag_upwind)
    keep = {k: res[k] for k in ("y", "T", "p", "s_liq", "avg_it", "total_it", "n_steps",
                                "n_time_step_cuts")}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(keep, f)
    return key, keep, time.time() - t0, False


def _sweep(tasks, parallel, label):
    """Run tasks largest-N-first (load balance); each writes its own cache the instant it
    finishes and prints a live ``done i/n`` line. Returns {key: result}."""
    tasks = sorted(tasks, key=lambda t: -t[3])         # t[3] = N -> heaviest runs launch first
    n, out = len(tasks), {}

    def _report(i, key, res, wall, cached):
        out[key] = res
        print(f"[reference] {label} {i:2d}/{n}  {str(key):16s}  "
              f"{'cached' if cached else f'{wall:6.0f}s'}  "
              f"avg_it={res['avg_it']:.2f}  total_it={res['total_it']}", flush=True)

    if parallel and n > 1:
        import multiprocessing as mp
        nproc = min(n, max(1, (os.cpu_count() or 4) - 1))
        print(f"[reference] {label}: {n} runs on {nproc} procs "
              f"(per-run cache in _cache/)", flush=True)
        with mp.get_context("spawn").Pool(nproc) as pool:
            for i, r in enumerate(pool.imap_unordered(_run, tasks), 1):
                _report(i, *r)
    else:
        for i, t in enumerate(tasks, 1):
            _report(i, *_run(t))
    return out


def _load_if_covers(path, expected):
    """Load an aggregate pickle only if it contains EVERY expected key (else None). Guards against a
    stale aggregate silently returning a SUBSET when the requested level/N set has since grown (the
    aggregate name is not keyed by the set). On a partial hit it returns None so the caller recomputes
    -- cheap, since the per-run caches back every already-computed member."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        agg = pickle.load(f)
    missing = set(expected) - set(agg)
    if missing:
        print(f"[reference] aggregate {os.path.basename(path)} covers "
              f"{len(expected) - len(missing)}/{len(expected)} requested keys; "
              f"recomputing the remainder (per-run cached)", flush=True)
        return None
    print(f"[reference] loaded aggregate {os.path.basename(path)}", flush=True)
    return {k: agg[k] for k in expected}


def compute_spatial(N_levels=N_LEVELS, dt=DT_FIXED, level=LEVEL_SPATIAL, case=CASE,
                    grav_upstream=GRAV_UPSTREAM, lag_upwind=LAG_UPWIND, n_steps=None,
                    parallel=True, cache=True):
    """3 schemes x N_levels at fixed dt -> {(scheme_key, N): result}. Resumable: a killed sweep
    reloads the per-run caches in _cache/ and only runs what is missing. The gravity-density
    (avg/up) and advective (cur/lag) treatments are tagged into all cache names, so they coexist."""
    path = os.path.join(
        m.HERE,
        f"_cache_spatial_{case}_{_dens_tag(grav_upstream)}_{_lag_tag(lag_upwind)}"
        f"_l{level}_dt{dt / m.YEAR:.5g}yr.pkl")
    expected = [(sk, N) for sk in ps.SCHEMES for N in N_levels]
    if cache:
        hit = _load_if_covers(path, expected)
        if hit is not None:
            return hit
    m.prebuild_table_caches(level)
    tasks = [("spatial", (sk, N), sk, N, level, dt, case, n_steps, grav_upstream, lag_upwind)
             for sk in ps.SCHEMES for N in N_levels]
    out = _sweep(tasks, parallel, "spatial")
    if cache:
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def compute_obl(levels=OBL_LEVELS, N=N_OBL, dt=DT_FIXED, case=CASE,
                grav_upstream=GRAV_UPSTREAM, lag_upwind=LAG_UPWIND, n_steps=None,
                parallel=True, cache=True):
    """3 schemes x OBL levels at fixed N -> {(scheme_key, level): result}. Resumable; density
    (avg/up) and advective (cur/lag) treatments tagged into the cache names."""
    path = os.path.join(
        m.HERE, f"_cache_obl_{case}_{_dens_tag(grav_upstream)}_{_lag_tag(lag_upwind)}"
        f"_N{N}_dt{dt / m.YEAR:.5g}yr.pkl")
    expected = [(sk, lv) for sk in ps.SCHEMES for lv in levels]
    if cache:
        hit = _load_if_covers(path, expected)
        if hit is not None:
            return hit
    for lv in levels:
        m.prebuild_table_caches(lv)
    tasks = [("obl", (sk, lv), sk, N, lv, dt, case, n_steps, grav_upstream, lag_upwind)
             for sk in ps.SCHEMES for lv in levels]
    out = _sweep(tasks, parallel, "obl")
    if cache:
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


# --- figure -------------------------------------------------------------------------------
def _order_line(ax, h, e, order=1):
    """Straight O(h^order) reference line spanning the plotted cell-size range, anchored at the
    coarsest (largest-h) end of the data. Labelled O(h). Replaces the earlier slope triangle."""
    h = np.asarray(sorted(h))
    h0, e0 = h[-1], max(e)                    # anchor at the coarsest h, near the top of the data
    hh = np.array([h[0], h[-1]])
    ee = e0 * (hh / h0) ** order
    ax.plot(hh, ee, color="0.5", ls="--", lw=0.9, zorder=1)
    ax.text(np.sqrt(hh[0] * hh[1]), np.sqrt(ee[0] * ee[1]) * 1.15,
            r"$\mathcal{O}(h)$" if order == 1 else rf"$\mathcal{{O}}(h^{{{order}}})$",
            color="0.5", ha="center", va="bottom", fontsize=8)


FIG_W_HALF = 0.49 * ps.TEXTWIDTH_IN     # width of one subfigure (two share the text width)


def plot(spatial, obl, stem="fig_weis_reference"):
    """Render the two reference panels as SEPARATE figures (no sub-captions -- the LaTeX subfigure
    environment supplies '(a)'/'(b)'): ``{stem}_a`` = spatial refinement, ``{stem}_b`` = parametric
    (OBL) refinement. The Newton iteration statistics are reported in the table (tab:weis_newton)."""
    ps.apply_style()
    import matplotlib.pyplot as plt

    # (a) spatial convergence -> {stem}_a
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W_HALF, 2.9))
    N_all = sorted({N for (_, N) in spatial})
    N_ref = max(N_all)
    Ns = [N for N in N_all if N < N_ref]
    for sk, cfg in ps.SCHEMES.items():
        ref = spatial[(sk, N_ref)]
        h = [m.L_COLUMN / N for N in Ns]
        err = [combined_error(spatial[(sk, N)], ref) for N in Ns]
        ax.loglog(h, err, marker=cfg["marker"], color=cfg["color"], ms=4, label=cfg["label"])
    any_sk = next(iter(ps.SCHEMES))
    _order_line(ax, [m.L_COLUMN / N for N in Ns],
                [combined_error(spatial[(any_sk, N)], spatial[(any_sk, N_ref)]) for N in Ns])
    ax.set_xlabel(r"cell size $h\ [\mathrm{m}]$")
    ax.set_ylabel(r"relative $L^2$ error")
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    ps.bottom_legend(fig, handles, labels, ncol=3)
    ps.savefig(fig, f"{stem}_a", OUT_DIR)
    plt.close(fig)

    # (b) parametric OBL convergence -> {stem}_b. Log axis; error of each level against the finest
    # table (level 4), which is the reference (error 0) and thus not plotted -- the panel shows the
    # coarse-to-fine table convergence over levels 0..3. Its purpose: demonstrate that level 3 (used
    # by the production solvers) is already close enough to level 4 that the finer (and much larger:
    # ~1.4 GB) table is NOT needed.
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W_HALF, 2.9))
    L_all = sorted({lv for (_, lv) in obl})
    L_ref = max(L_all)
    Ls = [lv for lv in L_all if lv < L_ref]
    for sk, cfg in ps.SCHEMES.items():
        ref = obl[(sk, L_ref)]
        err = [combined_error(obl[(sk, lv)], ref) for lv in Ls]
        ax.semilogy(Ls, err, marker=cfg["marker"], color=cfg["color"], ms=4, label=cfg["label"])
    ax.set_xlabel(r"OBL table level")
    ax.set_ylabel(r"error vs finest level")
    ax.set_xticks(Ls)
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    ps.bottom_legend(fig, handles, labels, ncol=3)
    ps.savefig(fig, f"{stem}_b", OUT_DIR)
    plt.close(fig)


def main():
    plot(compute_spatial(), compute_obl())


if __name__ == "__main__":
    main()
