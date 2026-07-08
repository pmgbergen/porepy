"""Figure ``fig:weis_verification`` -- verification in the vertical orientation, where the
gravitational terms are active.

(a) Reference solutions isolating the density treatment: PPU and HU, each with the gravity-term
    densities UPWINDED (as in Weis 2014, dashed) and AVERAGED at the faces (Rem. gravity
    consistency, solid). The upwinded-density profiles reproduce the published curves; the
    averaged-density ones converge to a common, consistent front. The offset between the two
    families is thus the imprint of the density treatment, not of the upwind assignment.
(b) The PorePy solution (produced in the second refactoring step by the updated 2D script)
    superposed on the averaged-density references of the three schemes (PPU, HU, HU-mw).

Layout: 2 columns (a references | b verification) x 2 rows (T, s_liq).
Runs (5 total, vertical, fine N) are cached. The PorePy overlay is drawn if
``porepy_solution_vertical.pkl`` is present; otherwise the panel shows the references only.

    python fig_weis_verification.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weis_1d_solver as m  # noqa: E402
import plot_style as ps     # noqa: E402

# --- configuration ------------------------------------------------------------------------
CASE = "vertical"
N = 800
LEVEL = m.TABLE_LEVEL
LAG_UPWIND = False        # advective weight: False = current iterate (genuine schemes); True =
#                           old-state, frozen once per step. Tagged (cur/lag) into the cache names.
FIELDS = ("T", "s_liq")
# (scheme_key, density_key) runs. (a) needs PPU/HU x {averaged,upwinded}; (b) adds HU-mw averaged.
RUNS_A = [("ppu", "averaged"), ("ppu", "upwinded"), ("hu", "averaged"), ("hu", "upwinded")]
RUNS_B = [("ppu", "averaged"), ("hu", "averaged"), ("hu_mw", "averaged")]
ALL_RUNS = list(dict.fromkeys(RUNS_A + RUNS_B))     # unique, order-preserving
OUT_DIR = os.path.join(m.HERE, "figures")
PP_DATA = os.path.join(m.HERE, f"porepy_solution_{CASE}.pkl")   # produced in step 2


CACHE_DIR = os.path.join(m.HERE, "_cache")     # per-run caches (resumable + observable)


def _lag_tag(lag_upwind):
    return "lag" if lag_upwind else "cur"


def _run_path(sk, dk, N, level, case, n_steps, lag_upwind):
    dens = "up" if ps.DENSITY[dk]["grav_upstream"] else "avg"   # avg/up tag (density VARIES here)
    ns = "" if n_steps is None else f"_ns{n_steps}"
    return os.path.join(
        CACHE_DIR,
        f"verification_{case}_{dens}_{_lag_tag(lag_upwind)}_{sk}_N{N}_l{level}{ns}.pkl")


def _run(args):
    """One (scheme, density) run with the chosen advective (cur/lag) treatment. Per-run cached in
    _cache/ with the avg/up and cur/lag tags, resumable. Returns
    (key, result, wall_seconds, was_cached)."""
    sk, dk, N, level, case, n_steps, lag_upwind = args
    path = _run_path(sk, dk, N, level, case, n_steps, lag_upwind)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return (sk, dk), pickle.load(f), 0.0, True
    cfg, den = ps.SCHEMES[sk], ps.DENSITY[dk]
    t0 = time.time()
    res = m.run(scheme=cfg["scheme"], weighted_perm=cfg["weighted_perm"],
                grav_upstream=den["grav_upstream"], N=N, case=case, level=level,
                n_steps=n_steps, verbose=False, lag_upwind=lag_upwind)
    keep = {k: res[k] for k in ("y", "T", "p", "s_liq", "avg_it", "total_it")}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(keep, f)
    return (sk, dk), keep, time.time() - t0, False


def _sweep(tasks, parallel):
    """Per-run caching + live progress. Returns {key: result}."""
    n, out = len(tasks), {}

    def _report(i, key, res, wall, cached):
        out[key] = res
        print(f"[verification] {i:2d}/{n}  {str(key):24s}  "
              f"{'cached' if cached else f'{wall:6.0f}s'}  "
              f"avg_it={res['avg_it']:.2f}  total_it={res['total_it']}", flush=True)

    if parallel and n > 1:
        import multiprocessing as mp
        nproc = min(n, max(1, (os.cpu_count() or 4) - 1))
        print(f"[verification] {n} runs on {nproc} procs (per-run cache in _cache/)", flush=True)
        with mp.get_context("spawn").Pool(nproc) as pool:
            for i, r in enumerate(pool.imap_unordered(_run, tasks), 1):
                _report(i, *r)
    else:
        for i, t in enumerate(tasks, 1):
            _report(i, *_run(t))
    return out


def compute(N=N, level=LEVEL, case=CASE, lag_upwind=LAG_UPWIND, n_steps=None,
            parallel=True, cache=True):
    """Run the 5 (scheme, density) combinations for the vertical verification. Resumable per-run
    cache in _cache/ (avg/up and cur/lag tagged); aggregate cached."""
    path = os.path.join(
        m.HERE, f"_cache_verification_{case}_{_lag_tag(lag_upwind)}_N{N}_l{level}.pkl")
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            print(f"[verification] loaded aggregate {os.path.basename(path)}")
            return pickle.load(f)
    m.prebuild_table_caches(level)
    tasks = [(sk, dk, N, level, case, n_steps, lag_upwind) for sk, dk in ALL_RUNS]
    out = _sweep(tasks, parallel)
    if cache:
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def load_porepy(case=CASE):
    """PorePy 2D vertical-profile overlay for panel (b), if present (step 2). Returns a result
    dict with 'y'[m], 'T'[K], 's_liq' (like weis_1d_solver.run), or None."""
    if os.path.exists(PP_DATA):
        with open(PP_DATA, "rb") as f:
            return pickle.load(f)
    return None


def plot(out, stem="fig_weis_verification"):
    ps.apply_style()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(FIELDS), 2, figsize=(ps.TEXTWIDTH_IN, 4.4), sharex="col")
    titles = ("(a) reference solutions", "(b) verification")
    for i, field in enumerate(FIELDS):
        # --- (a) density-treatment references: scheme -> colour, density -> line style ---
        ax = axes[i][0]
        xr, yr = m.load_reference(CASE, field)
        ax.plot(xr, yr, **ps.REF_KW)
        for sk, dk in RUNS_A:
            cfg, den = ps.SCHEMES[sk], ps.DENSITY[dk]
            x, v = ps.to_plot_units(out[(sk, dk)], field)
            ax.plot(x, v, color=cfg["color"], ls=den["ls"], lw=1.2,
                    label=f"{cfg['label']}, {den['label']}")
        ax.set_ylabel(ps.FIELD_LABEL[field]); ax.set_xlim(0, 2)
        if i == 0:
            ax.set_title(titles[0]); ax.legend(loc="best", ncol=1, handlelength=1.8)

        # --- (b) averaged-density references + PorePy overlay ---
        ax = axes[i][1]
        ax.plot(xr, yr, **ps.REF_KW)
        for sk, dk in RUNS_B:
            cfg = ps.SCHEMES[sk]
            x, v = ps.to_plot_units(out[(sk, dk)], field)
            ax.plot(x, v, color=cfg["color"], ls="-", lw=1.2, label=cfg["label"])
        pp = load_porepy(CASE)
        if pp is not None:
            x, v = ps.to_plot_units(pp, field)
            ax.plot(x, v, color="0.15", ls="none", marker="x", ms=4, mew=0.8,
                    label="PorePy")
        elif i == 0:
            ax.text(0.63, 0.35, r"\textit{PorePy overlay}" "\n" r"\textit{pending (step 2)}"
                    if plt.rcParams["text.usetex"] else "PorePy overlay\npending (step 2)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=7, color="0.45")
        ax.set_xlim(0, 2)
        if i == 0:
            ax.set_title(titles[1]); ax.legend(loc="best", handlelength=1.8)
        if i == len(FIELDS) - 1:
            for j in (0, 1):
                axes[i][j].set_xlabel(ps.DIST_LABEL)
    fig.tight_layout()
    ps.savefig(fig, stem, OUT_DIR)


def main():
    plot(compute())


if __name__ == "__main__":
    main()
