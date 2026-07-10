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

import numpy as np

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
    keep = {k: res[k] for k in ("y", "T", "p", "s_liq", "avg_it", "total_it", "n_time_step_cuts")}
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


# ------------------------------------------------------------------------------------------- #
#  Horizontal verification: the PorePy 2D run vs. the corresponding Weis 1D-solver profile
# ------------------------------------------------------------------------------------------- #
def load_porepy_case(case, scheme, N=800, level=LEVEL):
    """Load the PorePy 2D overlay pickle ``porepy_{case}_{scheme}_N{N}_l{level}.pkl`` from _cache/.

    Returned normalized to the weis_1d_solver SI convention so ``plot_style.to_plot_units`` applies
    unchanged: PorePy's pressure primary variable is in the model's MPa units, so it is rescaled to
    Pa (x1e6). ``y``[m], ``T``[K], ``s_liq``[-] are already SI. Returns the dict, or None."""
    path = os.path.join(CACHE_DIR, f"porepy_{case}_{scheme}_N{N}_l{level}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        d = dict(pickle.load(f))
    d["p"] = d["p"] * 1.0e6                     # MPa (PorePy native) -> Pa (SI, as the 1D solver)
    return d


def load_weis_case(case, scheme, N=800, level=LEVEL):
    """Load the corresponding Weis 1D-solver profile (averaged density, current iterate) from the
    profiles cache ``profiles_{case}_avg_cur_{scheme}_N{N}_l{level}.pkl``. Returns the dict
    (``y``[m], ``T``[K], ``p``[Pa], ``s_liq``[-]) or None."""
    path = os.path.join(CACHE_DIR, f"profiles_{case}_avg_cur_{scheme}_N{N}_l{level}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


FIG_W_HALF = 0.49 * ps.TEXTWIDTH_IN     # width of one subfigure (two share the text width)
_P_LS = (0, (4, 2))                     # pressure dashed, to read against solid temperature
# Per-scheme colours (this figure): the 1D reference is a THICK, LIGHT band drawn underneath and the
# PorePy solution a THIN, DARK solid line on top -- so overlapping curves read as a dark line riding
# inside a pale band, and any divergence stands out immediately.
_HU_DARK, _HUMW_DARK = "#8B0000", "#00008B"     # PorePy (on top): dark red / dark blue
_HU_LIGHT, _HUMW_LIGHT = "#F0A8A8", "#A6AEF0"   # 1D reference (underneath): light red / light blue
_REF_LW, _PP_LW = 3.4, 1.3                      # reference thick / PorePy thin


def _it_suffix(result: dict) -> str:
    """``', N it.'`` from a result dict's total Newton-iteration count (``total_it``), else ``''``."""
    n = result.get("total_it") if isinstance(result, dict) else None
    return fr", ${int(n)}$ it." if n else ""


# Fixed output canvas [in] so the horizontal and vertical figures come out at EXACTLY the same size.
# (Their tight bboxes otherwise differ: the vertical legend is wider because its iteration counts have
# more digits.) Chosen a touch larger than the widest tight content so nothing is clipped.
_FIG_SIZE_IN = (4.05, 5.25)


def _savefig_fixed(fig, stem, out_dir, size_in=_FIG_SIZE_IN):
    """Save PDF+PNG with the tight content centred inside a FIXED ``size_in`` canvas, so figures
    saved by different calls are byte-for-byte the same dimensions."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    os.makedirs(out_dir, exist_ok=True)
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())          # tight content extent [in]
    w, h = size_in
    if tb.width > w or tb.height > h:                          # guard: would clip -> warn, don't lie
        print(f"[verification] WARNING: content {tb.width:.2f}x{tb.height:.2f} exceeds fixed "
              f"canvas {w}x{h} in for {stem}; increase _FIG_SIZE_IN")
    padx, pady = (w - tb.width) / 2.0, (h - tb.height) / 2.0
    bb = Bbox.from_extents(tb.x0 - padx, tb.y0 - pady, tb.x1 + padx, tb.y1 + pady)
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches=bb)
        print(f"wrote {p}")
    plt.close(fig)


def plot_verification(case="horizontal", schemes=("hu", "hu_mw"), stem=None):
    """fig_weis_profiles-style verification for one orientation -> one figure. Top panel merges
    temperature (left axis, solid) and pressure (right axis, dashed); the bottom panel is liquid
    saturation. The REFERENCES are the Weis 1D-solver solutions (lines, per-scheme colour) -- HU and
    HU-mw -- and the PorePy 2D solutions are overlaid as open markers (same colour) that should lie
    on them. No digitized Weis (2014) reference in this figure. Reads only cached pickles; a scheme
    with no PorePy cache is drawn as its 1D reference line alone."""
    if stem is None:
        stem = f"fig_weis_verification_{case}"
    ps.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = {}                                           # scheme -> (weis_dict|None, porepy_dict|None)
    for scheme in schemes:
        data[scheme] = (load_weis_case(case, scheme), load_porepy_case(case, scheme))
        if data[scheme][0] is None:
            print(f"[verification] no Weis 1D profile cache for {case}/{scheme}")
        if data[scheme][1] is None:
            print(f"[verification] no PorePy cache for {case}/{scheme}")

    fig, (ax_tp, ax_s) = plt.subplots(2, 1, figsize=(FIG_W_HALF, 4.5), sharex=True,
                                      gridspec_kw=dict(height_ratios=[1.4, 1.0]))
    ax_p = ax_tp.twinx(); ax_p.grid(False)              # right axis = pressure

    dark = {"hu": _HU_DARK, "hu_mw": _HUMW_DARK}        # PorePy: thin dark solid, on top
    light = {"hu": _HU_LIGHT, "hu_mw": _HUMW_LIGHT}     # 1D reference: thick light band, underneath
    for scheme in schemes:
        cfg = ps.SCHEMES[scheme]
        cd, cl = dark[scheme], light[scheme]
        weis, pp_ = data[scheme]
        if weis is not None:                            # 1D solver = THICK LIGHT reference band
            wlab = fr"{cfg['label']} (1D{_it_suffix(weis)})"
            ax_tp.plot(*ps.to_plot_units(weis, "T"), color=cl, ls="-", lw=_REF_LW, zorder=2)
            ax_p.plot(*ps.to_plot_units(weis, "p"), color=cl, ls=_P_LS, lw=_REF_LW, zorder=2)
            ax_s.plot(*ps.to_plot_units(weis, "s_liq"), color=cl, ls="-", lw=_REF_LW, zorder=2,
                      label=wlab)
        if pp_ is not None:                             # PorePy 2D = THIN DARK solid, on top
            plab = fr"{cfg['label']} (PorePy{_it_suffix(pp_)})"
            ax_tp.plot(*ps.to_plot_units(pp_, "T"), color=cd, ls="-", lw=_PP_LW, zorder=4)
            ax_p.plot(*ps.to_plot_units(pp_, "p"), color=cd, ls=_P_LS, lw=_PP_LW, zorder=4)
            ax_s.plot(*ps.to_plot_units(pp_, "s_liq"), color=cd, ls="-", lw=_PP_LW, zorder=4,
                      label=plab)

    # y-axes in default black (colour now encodes the scheme: HU dark red, HU-mw dark blue)
    ax_tp.set_ylabel(ps.FIELD_LABEL["T"])
    ax_tp.tick_params(axis="y", which="both", right=False)
    ax_p.set_ylabel(ps.FIELD_LABEL["p"])
    ax_s.set_ylabel(ps.FIELD_LABEL["s_liq"]); ax_s.set_xlabel(ps.DIST_LABEL)
    ax_tp.set_xlim(0.0, 2.0)

    # T/p line-style key in BLACK -- refers to the reference lines (solid = T, dashed = p)
    style_key = [Line2D([0], [0], color="black", ls="-", label=r"$T$ (left)"),
                 Line2D([0], [0], color="black", ls=_P_LS, label=r"$p$ (right)")]
    key = ax_tp.legend(handles=style_key, loc="upper right", handlelength=2.0, fontsize=8,
                       borderaxespad=0.5, borderpad=0.5, frameon=True, fancybox=True,
                       framealpha=1.0, edgecolor="0.6")
    key.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.4")

    # scheme legend (1D line vs PorePy markers) in a rounded box below, from the saturation panel
    handles, labels = ax_s.get_legend_handles_labels()
    fig.tight_layout()
    ps.bottom_legend(fig, handles, labels, ncol=2)
    _savefig_fixed(fig, stem, OUT_DIR)


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
    # PorePy 2D vs. the corresponding Weis 1D solver, per orientation (cache-only, fast). A scheme
    # missing its PorePy pickle is shown as the 1D reference line alone.
    for case in ("horizontal", "vertical"):
        plot_verification(case, schemes=("hu", "hu_mw"))


if __name__ == "__main__":
    main()
