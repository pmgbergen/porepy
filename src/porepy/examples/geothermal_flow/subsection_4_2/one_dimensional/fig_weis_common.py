"""Shared compute + reference helpers for the ``fig_weis_fig_{4,5,6}`` scripts.

Each figure runs the single weis brine engine (``weis_1d_solver.run_brine``) for PPU / HU / HU-mwp
and overlays the digitized Weis (2014) reference. The heavy N=800 runs are cached per (tag, scheme,
case) in ``_cache/`` (delete a pickle to recompute); a fresh compute uses a spawn ``Pool`` with the
BLAS threads pinned (below), matching ``run_workflow.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")        # pin BLAS threads BEFORE numpy: the spawn Pool
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # deadlocks with multi-threaded BLAS on a fresh
os.environ.setdefault("MKL_NUM_THREADS", "1")        # (uncached) compute -- see run_workflow.py.

import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weis_1d_solver as m  # noqa: E402
import plot_style as ps     # noqa: E402

CACHE_DIR = os.path.join(m.HERE, "_cache")
OUT_DIR = os.path.join(m.HERE, "figures")

# Reference band + pressure line styling, shared so the three figures read as one set.
P_LS = (0, (4, 2))                                   # pressure dashed against solid temperature
WEIS_T, WEIS_P = "#8B0000", "#00008B"                # reference / axis colours: dark red T, dark blue p
WEIS_S = "0.35"                                      # reference saturation: dark grey
# The Weis (2014) digitized reference is a thin continuous line with sparse open markers (so it reads
# as data over the model curves, not a heavy band).
REF_LW = 1.0
REF_MS = 3.6
REF_NMARK = 14                                       # ~this many markers along each reference curve


def _ref_plot(ax, ref, color, marker, ls, zorder=2):
    x, v = ref
    ax.plot(x, v, color=color, ls=ls, lw=REF_LW, marker=marker, mfc="none", mec=color, ms=REF_MS,
            mew=0.8, markevery=max(1, len(x) // REF_NMARK), zorder=zorder)


def ref_legend_handle():
    """Neutral legend proxy for the Weis reference (thin line + open marker)."""
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color="0.3", lw=REF_LW, marker="o", mfc="none", mec="0.3", ms=REF_MS,
                  label=r"Weis et al.\ (2014)")

_KEEP = ("y", "T", "p", "s_liq", "s_halite", "Xl", "total_it", "avg_it", "n_time_step_cuts")


def _run_path(tag, sk, case, N, level, pure_water=False, amr=False):
    # OBL identity tags the cache: purewater / hex-AMR / brine table at this level (l3, lgraded, ...).
    # The three are mutually exclusive, so distinct tags keep them off each other's cache path.
    obl = "pw" if pure_water else ("amr" if amr else f"l{level}")
    return os.path.join(CACHE_DIR, f"{tag}_{case}_{sk}_N{N}_{obl}.pkl")


def _run_one(args):
    """Run (or load) ONE scheme run, returning it under the caller's ``rkey``. The cache PATH is keyed
    by (tag, sk, case) -- ``tag`` must be unique per panel-group (Fig 4 encodes the pressure level,
    Fig 6 the column) so distinct panels sharing a run ``case`` do not collide on disk.
    ``args = (rkey, tag, sk, case, bc, N, level, grav_upstream, lag_upwind[, pure_water])`` -- the
    trailing ``pure_water`` is optional so pre-existing 9-field task tuples (Fig 4/5) still unpack."""
    rkey, tag, sk, case, bc, N, level, grav_upstream, lag_upwind = args[:9]
    pure_water = args[9] if len(args) > 9 else False
    amr_table = args[10] if len(args) > 10 else None       # optional AMR-OBL xph/xpt .vtu overrides
    amr_xpt = args[11] if len(args) > 11 else None
    path = _run_path(tag, sk, case, N, level, pure_water, amr=amr_table is not None)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return rkey, pickle.load(f), True
    cfg = ps.SCHEMES[sk]
    res = m.run_brine(scheme=cfg["scheme"], weighted_perm=cfg["weighted_perm"],
                      grav_upstream=grav_upstream, lag_upwind=lag_upwind, N=N, case=case,
                      level=level, pure_water=pure_water, amr_table=amr_table, amr_xpt=amr_xpt,
                      verbose=False, **bc)
    keep = {k: res[k] for k in _KEEP}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(keep, f)
    return rkey, keep, False


def run_tasks(label, tasks, parallel=True):
    """Run a list of ``_run_one`` task tuples, caching each. Returns ``{(scheme_key, case): result}``.
    A fresh run parallelises over a spawn Pool (BLAS threads pinned above); cached runs are instant.
    Tasks may carry per-task BC presets (e.g. Fig 4's per-pressure-level, per-orientation times)."""
    out = {}

    def _report(key, res, cached):
        out[key] = res
        print(f"[{label}] {str(key):26s} {'cached' if cached else 'done  '} "
              f"total_it={res['total_it']}", flush=True)

    if parallel and len(tasks) > 1:
        import multiprocessing as mp
        nproc = min(len(tasks), max(1, (os.cpu_count() or 4) - 1))
        print(f"[{label}] {len(tasks)} runs on {nproc} procs (per-run cache in _cache/)", flush=True)
        with mp.get_context("spawn").Pool(nproc) as pool:
            for key, res, cached in pool.imap_unordered(_run_one, tasks):
                _report(key, res, cached)
    else:
        for t in tasks:
            _report(*_run_one(t))
    return out


def sweep(tag, cases, bc, N, level, schemes=None, grav_upstream=False, lag_upwind=False,
          parallel=True, pure_water=False, amr_table=None, amr_xpt=None):
    """Run PPU/HU/HU-mwp over ``cases`` with a SHARED BC preset ``bc`` (Fig 5/6). Fig 4, whose BC and
    time vary per panel, builds its own tasks and calls :func:`run_tasks` directly. ``pure_water=True``
    (Fig-6 pure-water column) swaps in the fine z=0 tables for every run in the sweep. ``amr_table`` /
    ``amr_xpt`` (paths to the hex-AMR .vtu OBL tables) override the opensowat tables for every run."""
    if amr_table is None:
        m.prebuild_table_caches(level, pure_water=pure_water)   # opensowat/pw tensor-cache warm-up
    schemes = list(ps.SCHEMES) if schemes is None else list(schemes)
    tasks = [((sk, case), tag, sk, case, bc, N, level, grav_upstream, lag_upwind, pure_water,
              amr_table, amr_xpt)
             for case in cases for sk in schemes]
    return run_tasks(tag, tasks, parallel)


def ref_csv(name):
    """Digitized reference ``(distance_km, value)`` sorted by distance, or ``None`` if the CSV is
    absent (e.g. the Fig-6 salt reference not yet digitized -> caller draws a placeholder)."""
    path = os.path.join(m.REF_DIR, name)
    if not os.path.exists(path):
        return None
    x, v = m._load_ref_csv(name)
    o = np.argsort(x)
    return x[o], v[o]


def scheme_totals(panels):
    """Total Newton iterations per scheme, summed over the figure's panels. ``panels`` is a list of
    ``{scheme_key: result}`` dicts (``None`` panels skipped). Also prints the tally."""
    totals = {sk: 0 for sk in ps.SCHEMES}
    for d in panels:
        if d is None:
            continue
        for sk, r in d.items():
            totals[sk] = totals.get(sk, 0) + int(r.get("total_it", 0))
    print("[iterations] " + "  ".join(f"{sk}={totals[sk]}" for sk in ps.SCHEMES), flush=True)
    return totals


def parse_skip(spec):
    """Comma/space-separated solver names -> normalized skip set. Recognised names: the ps.SCHEMES keys
    (``ppu``, ``hu``, ``hu-mwp``), plus ``hu-porepy`` (the PorePy overlay) and ``ppu-weis`` (the Fig-5
    reference curve). Hyphen/underscore- and case-insensitive."""
    import re
    return {t.strip().lower().replace("_", "-") for t in re.split(r"[,\s]+", spec or "") if t.strip()}


def is_skipped(name, skip):
    """True if solver ``name`` is in the ``skip`` set (hyphen/underscore/case-insensitive)."""
    return name.lower().replace("_", "-") in skip


def active_schemes(skip):
    """ps.SCHEMES keys NOT in ``skip`` -- pass to :func:`sweep` (avoids running skipped schemes)."""
    return [sk for sk in ps.SCHEMES if not is_skipped(sk, skip)]


def scheme_handles(only=None):
    """One legend handle per scheme -- coloured, labelled with the scheme name (no counts; the counts
    live per panel via :func:`iteration_legend`, since they differ across simulation cases). ``only``
    (a list of scheme keys, e.g. :func:`active_schemes`) restricts the handles to those schemes."""
    from matplotlib.lines import Line2D
    keys = [sk for sk in ps.SCHEMES if only is None or sk in only]
    return [Line2D([0], [0], color=ps.SCHEMES[sk]["color"], lw=1.8, label=ps.SCHEMES[sk]["label"])
            for sk in keys]


def iteration_legend(ax, results, loc="lower right", fontsize=6.0, title="total it.", extra=None):
    """Small in-panel key with this CASE's Newton-iteration count per scheme: just the coloured line
    and the number (the scheme names live in the shared bottom legend). Counts differ by orientation
    / pressure level, so they belong to the panel. ``title`` doubles as a caption (Fig 4 puts its
    pressure level + time there). ``extra`` = optional [(colour, count), ...] appended (Fig 5's
    PPU-Weis curve)."""
    from matplotlib.lines import Line2D
    schemes = [sk for sk in ps.SCHEMES if sk in results]
    h = [Line2D([0], [0], color=ps.SCHEMES[sk]["color"], lw=2.2) for sk in schemes]
    lab = [fr"${int(results[sk]['total_it'])}$" for sk in schemes]
    for color, count in (extra or []):
        h.append(Line2D([0], [0], color=color, lw=2.2)); lab.append(fr"${int(count)}$")
    leg = ax.legend(h, lab, loc=loc, fontsize=fontsize, frameon=True, fancybox=True, framealpha=0.9,
                    edgecolor="0.7", borderpad=0.3, handlelength=0.9, handletextpad=0.4,
                    labelspacing=0.2, title=title, title_fontsize=fontsize)
    leg.get_frame().set_boxstyle("round,pad=0.18,rounding_size=0.22")
    ax.add_artist(leg)
    return leg


def draw_tp(ax_tp, ax_p, results, ref_T=None, ref_p=None, label_it=True):
    """Draw the temperature(left)+pressure(right, dashed) panel: the light reference band underneath,
    then the three scheme curves. ``results`` = {scheme_key: result}. Returns the (handles, labels)
    for a scheme legend (saturation-panel style: colour + iteration count)."""
    if ref_T is not None:
        _ref_plot(ax_tp, ref_T, WEIS_T, "o", "-")
    if ref_p is not None:
        _ref_plot(ax_p, ref_p, WEIS_P, "^", P_LS)
    handles, labels = [], []
    from matplotlib.lines import Line2D
    for sk in ps.SCHEMES:
        if sk not in results:
            continue
        cfg = ps.SCHEMES[sk]
        r = results[sk]
        ax_tp.plot(*ps.to_plot_units(r, "T"), color=cfg["color"], ls="-", lw=1.3, zorder=3)
        ax_p.plot(*ps.to_plot_units(r, "p"), color=cfg["color"], ls=P_LS, lw=1.1, zorder=3)
        handles.append(Line2D([0], [0], color=cfg["color"], lw=1.6))
        labels.append(fr"{cfg['label']} (${r['total_it']}$ it.)" if label_it else cfg["label"])
    return handles, labels


def draw_s(ax_s, results, ref_s=None, halite=False):
    """Draw the liquid-saturation panel (light reference band + three scheme curves). If ``halite``,
    a dashed halite-saturation twin is added when any scheme carries s_halite > 0."""
    if ref_s is not None:
        _ref_plot(ax_s, ref_s, WEIS_S, "s", "-")
    ax_h = None
    for sk in ps.SCHEMES:
        if sk not in results:
            continue
        cfg = ps.SCHEMES[sk]
        r = results[sk]
        ax_s.plot(*ps.to_plot_units(r, "s_liq"), color=cfg["color"], lw=1.3, zorder=3)
        if halite and np.max(np.abs(r.get("s_halite", 0.0))) > 1e-6:
            if ax_h is None:
                ax_h = ax_s.twinx(); ax_h.grid(False); ax_h.set_ylim(-0.03, 1.03)
            ax_h.plot(r["y"] / 1e3, r["s_halite"], color=cfg["color"], lw=1.1, ls=(0, (1, 1)), zorder=2)
    return ax_h
