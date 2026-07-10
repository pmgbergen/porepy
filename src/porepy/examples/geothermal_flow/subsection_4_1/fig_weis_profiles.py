"""Figure ``fig:weis_profiles`` -- converged profiles of the three schemes (PPU, HU, HU-mw)
against the published Weis (2014) fig-5 curves, in both orientations, with the consistent
face-averaged gravity densities (Rem. gravity_consistency).

Layout: 2 columns (a: horizontal, 200 yr | b: vertical, 1000 yr) x 3 rows (T, p, s_liq).
In the horizontal panel gravity is absent and the three schemes reproduce the published front
exactly; in the vertical panel they coincide with one another but sit at an offset from the
published curve (its origin -- the density treatment -- is isolated in fig_weis_verification).

The 6 heavy runs (3 schemes x 2 orientations at the full benchmark time) are cached; delete the
cache file to recompute. Runtime scales with N; raise N for the final, publication figure.

    python fig_weis_profiles.py            # compute (or load cache) + render
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

# --- configuration (raise N for the final figure; the runs are cached) --------------------
N = 800                              # spatial resolution of the "converged" profiles
LEVEL = m.TABLE_LEVEL                # OBL table refinement level
LAG_UPWIND = False                   # advective weight: False = current iterate (fully implicit,
#                                      genuine schemes); True = old-state, frozen once per step.
#                                      Tagged (cur/lag) into the cache names.
CASES = (("horizontal", r"(a) horizontal orientation, $200$ years"),
         ("vertical",   r"(b) vertical orientation, $1000$ years"))
FIELDS = ("T", "p", "s_liq")
EXTRA_N = 200          # 4th curve (vertical only): PPU with UPWINDED gravity densities at this N
OUT_DIR = os.path.join(m.HERE, "figures")


def _cache_path(N, level, lag_upwind):
    return os.path.join(m.HERE, f"_cache_profiles_{_lag_tag(lag_upwind)}_N{N}_l{level}.pkl")


CACHE_DIR = os.path.join(m.HERE, "_cache")     # per-run caches (resumable + observable)


def _lag_tag(lag_upwind):
    return "lag" if lag_upwind else "cur"


def _run_path(sk, case, N, level, n_steps, lag_upwind):
    ns = "" if n_steps is None else f"_ns{n_steps}"
    return os.path.join(CACHE_DIR,
                        f"profiles_{case}_avg_{_lag_tag(lag_upwind)}_{sk}_N{N}_l{level}{ns}.pkl")


def _run_one(args):
    """One (scheme, case) run at face-averaged densities (this figure is averaged-only, hence the
    fixed ``avg`` tag) with the chosen advective (cur/lag) treatment. Per-run cached in _cache/
    (resumable). Returns (key, result, wall_seconds, was_cached)."""
    sk, case, N, level, n_steps, lag_upwind = args
    path = _run_path(sk, case, N, level, n_steps, lag_upwind)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return (sk, case), pickle.load(f), 0.0, True
    cfg = ps.SCHEMES[sk]
    t0 = time.time()
    res = m.run(scheme=cfg["scheme"], weighted_perm=cfg["weighted_perm"], grav_upstream=False,
                N=N, case=case, level=level, n_steps=n_steps, verbose=False, lag_upwind=lag_upwind)
    keep = {k: res[k] for k in ("y", "T", "p", "s_liq", "avg_it", "total_it", "n_time_step_cuts")}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(keep, f)
    return (sk, case), keep, time.time() - t0, False


def _sweep(tasks, parallel):
    """Per-run caching + live progress. Returns {key: result}."""
    n, out = len(tasks), {}

    def _report(i, key, res, wall, cached):
        out[key] = res
        print(f"[profiles] {i:2d}/{n}  {str(key):22s}  "
              f"{'cached' if cached else f'{wall:6.0f}s'}  "
              f"avg_it={res['avg_it']:.2f}  total_it={res['total_it']}", flush=True)

    if parallel and n > 1:
        import multiprocessing as mp
        nproc = min(n, max(1, (os.cpu_count() or 4) - 1))
        print(f"[profiles] {n} runs on {nproc} procs (per-run cache in _cache/)", flush=True)
        with mp.get_context("spawn").Pool(nproc) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_one, tasks), 1):
                _report(i, *r)
    else:
        for i, t in enumerate(tasks, 1):
            _report(i, *_run_one(t))
    return out


def compute(N=N, level=LEVEL, lag_upwind=LAG_UPWIND, n_steps=None, parallel=True, cache=True):
    """Run 3 schemes x 2 orientations (averaged densities). Resumable per-run cache in _cache/;
    aggregate cached to a pickle keyed by (lag, N, level)."""
    path = _cache_path(N, level, lag_upwind)
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            print(f"[profiles] loaded aggregate {os.path.basename(path)}")
            return pickle.load(f)
    m.prebuild_table_caches(level)
    tasks = [(sk, case, N, level, n_steps, lag_upwind) for case, _ in CASES for sk in ps.SCHEMES]
    out = _sweep(tasks, parallel)
    if cache:
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def compute_extra(N=EXTRA_N, level=LEVEL, cache=True):
    """4th curve for fig_weis_profiles_b: PPU in the Weis (2014) setup -- UPWINDED gravity densities
    (Eq.25) AND lagged upwind directions (frozen once per time step) -- at N=EXTRA_N, vertical. This
    matches Weis's IMPES-style treatment: freezing the directions removes the well-balancedness
    instability that makes the fully-implicit current-iterate variant intractable, and the profile
    is expected to track the published curves closely. Per-run cached with an 'up_lag' tag."""
    path = os.path.join(CACHE_DIR, f"profiles_vertical_up_lag_ppu_N{N}_l{level}.pkl")
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            print(f"[profiles] loaded extra {os.path.basename(path)}")
            return pickle.load(f)
    m.prebuild_table_caches(level)
    res = m.run(scheme="ppu", weighted_perm=False, grav_upstream=True, N=N, case="vertical",
                level=level, lag_upwind=True, verbose=False)
    keep = {k: res[k] for k in ("y", "T", "p", "s_liq", "avg_it", "total_it", "n_time_step_cuts")}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(keep, f)
    return keep


FIG_W_HALF = 0.49 * ps.TEXTWIDTH_IN     # width of one subfigure (two share the text width)
_P_LS = (0, (4, 2))                     # pressure dashed, to read against solid temperature
_WEIS_T, _WEIS_P = "#8B0000", "#00008B"  # Weis reference / axis colours: dark red (T), dark blue (p)
_WEIS_T_LIGHT, _WEIS_P_LIGHT = "#F0A8A8", "#A6AEF0"  # Weis reference BAND: light red (T)/blue (p)
_REF_LW = 3.4                            # thick reference band underneath the scheme curves
_EXTRA_C = "#984EA3"                     # 4th curve (PPU, upwinded densities): purple


def _ref_line(case, field):
    """Digitized Weis (2014) reference sorted by distance, for drawing as a continuous band."""
    x, v = m.load_reference(case, field)
    o = np.argsort(x)
    return x[o], v[o]


def _plot_one(out, case, stem, extra=None):
    """One orientation -> one figure. Top panel merges temperature (left axis, solid) and pressure
    (right axis, dashed): the three schemes keep their own colours, the Weis reference is dark red
    (T) / dark blue (p), and each y-axis is coloured to match its Weis reference. The per-scheme
    iteration legend lives in the saturation panel below. ``extra`` (if given) adds a 4th curve --
    PPU with upwinded densities. No sub-caption (the LaTeX subfigure gives it)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax_tp, ax_s) = plt.subplots(2, 1, figsize=(FIG_W_HALF, 4.5), sharex=True,
                                      gridspec_kw=dict(height_ratios=[1.4, 1.0]))
    ax_p = ax_tp.twinx(); ax_p.grid(False)                 # right axis = pressure

    # Weis (2014) reference as a THICK LIGHT band underneath (experiment: was open markers) -- light
    # red (T, left), light blue (p, right), light grey (s_liq); the scheme curves ride on top.
    ax_tp.plot(*_ref_line(case, "T"), color=_WEIS_T_LIGHT, ls="-", lw=_REF_LW, zorder=1)
    ax_p.plot(*_ref_line(case, "p"), color=_WEIS_P_LIGHT, ls=_P_LS, lw=_REF_LW, zorder=1)
    ax_s.plot(*_ref_line(case, "s_liq"), color="0.78", ls="-", lw=_REF_LW, zorder=1)

    for sk, cfg in ps.SCHEMES.items():
        r = out[(sk, case)]
        ax_tp.plot(*ps.to_plot_units(r, "T"), color=cfg["color"], ls="-", lw=1.3, zorder=3)   # T
        ax_p.plot(*ps.to_plot_units(r, "p"), color=cfg["color"], ls=_P_LS, lw=1.1, zorder=3)  # p
        ax_s.plot(*ps.to_plot_units(r, "s_liq"), color=cfg["color"], lw=1.3, zorder=3,
                  label=fr"{cfg['label']} (${r['total_it']}$ it.)")

    if extra is not None:   # 4th curve (vertical only): PPU with upwinded gravity densities
        ax_tp.plot(*ps.to_plot_units(extra, "T"), color=_EXTRA_C, ls="-", lw=1.4)
        ax_p.plot(*ps.to_plot_units(extra, "p"), color=_EXTRA_C, ls=_P_LS, lw=1.2)
        ax_s.plot(*ps.to_plot_units(extra, "s_liq"), color=_EXTRA_C, lw=1.4,
                  label=r"PPU, upw.\ $\rho$")

    # colour each y-axis (label, ticks, spine) to match its quantity
    ax_tp.set_ylabel(ps.FIELD_LABEL["T"], color=_WEIS_T)
    ax_tp.tick_params(axis="y", which="both", right=False, colors=_WEIS_T)
    ax_p.set_ylabel(ps.FIELD_LABEL["p"], color=_WEIS_P)
    ax_p.tick_params(axis="y", colors=_WEIS_P)
    for a in (ax_tp, ax_p):
        a.spines["left"].set_color(_WEIS_T)
        a.spines["right"].set_color(_WEIS_P)
    ax_s.set_ylabel(ps.FIELD_LABEL["s_liq"]); ax_s.set_xlabel(ps.DIST_LABEL)
    ax_tp.set_xlim(0.0, 2.0)

    # T/p line-style key in BLACK -- it refers to the scheme curves (solid = T, dashed = p), which
    # keep their own colours -- shown in a rounded box in the clear top-right of the panel
    style_key = [Line2D([0], [0], color="black", ls="-", label=r"$T$ (left)"),
                 Line2D([0], [0], color="black", ls=_P_LS, label=r"$p$ (right)")]
    key = ax_tp.legend(handles=style_key, loc="upper right", handlelength=2.0, fontsize=8,
                       borderaxespad=0.5, borderpad=0.5, frameon=True, fancybox=True,
                       framealpha=1.0, edgecolor="0.6")
    key.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.4")

    # scheme + total-iteration legend (rounded box) below the figure, from the saturation panel
    handles, labels = ax_s.get_legend_handles_labels()
    fig.tight_layout()
    ps.bottom_legend(fig, handles, labels, ncol=2)
    ps.savefig(fig, stem, OUT_DIR)
    plt.close(fig)


def plot(out, extra_vertical=None, stem="fig_weis_profiles"):
    """Render each orientation as a SEPARATE figure (no sub-captions -- the LaTeX subfigure supplies
    '(a)'/'(b)'): ``{stem}_a`` = horizontal, ``{stem}_b`` = vertical. ``extra_vertical`` adds the
    upwinded-density PPU curve to the vertical figure only."""
    ps.apply_style()
    suffix = {"horizontal": "a", "vertical": "b"}
    for case, _sub in CASES:
        ex = extra_vertical if case == "vertical" else None
        _plot_one(out, case, f"{stem}_{suffix[case]}", extra=ex)


def main():
    plot(compute(), extra_vertical=compute_extra())


if __name__ == "__main__":
    main()
