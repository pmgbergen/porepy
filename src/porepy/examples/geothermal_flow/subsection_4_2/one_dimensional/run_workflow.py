#!/usr/bin/env python
"""One-command workflow for the Weis (2014) 1-D benchmark figures (subsection 4.2, one_dimensional).

Builds the four figures of the subsection from the two solver families in dependency order:

  [reference]     fig_weis_reference.py   -> figures/fig_weis_reference_{a,b}   (1-D convergence)
  [profiles]      fig_weis_profiles.py    -> figures/fig_weis_profiles_{a,b}    (1-D profiles;
                                             also writes the _cache/profiles_* the overlay reuses)
  [porepy 2D]     porepy_1d_solver.py     -> _cache/porepy_{case}_{scheme}_*    (heavy; hours)
  [verification]  fig_weis_verification.py-> figures/fig_weis_verification_{horizontal,vertical}
  [single-phase]  single_phase_porepy_1d_solver.py -> _cache/single_phase_*     (heavy)
                  fig_weis_single_phase.py-> figures/fig_4_single_phase

The 1-D reference engine (weis_1d_solver.py) is pure numpy and self-parallelizing; the heavy PorePy
2-D runs go through subprocesses (clean per-run PETSc state) and are resumable -- an existing cache
pickle is loaded and its run skipped, so re-running only computes what is missing.

Use the porepy conda env (its interpreter is reused for the subprocesses):

    PY=~/miniconda/envs/porepy/bin/python
    $PY run_workflow.py                 # full pipeline (resumable; fast if the caches exist)
    $PY run_workflow.py --quick         # coarse smoke, sandboxed to _quick/ (real cache untouched)
    $PY run_workflow.py --plot-only     # figures from existing _cache only (no 2-D solves)
    $PY run_workflow.py --skip-porepy   # no 2-D overlay runs (verification shows 1-D references)
    $PY run_workflow.py --porepy-schemes hu hu_mw   # add the HU-mw overlay (heavy new runs)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")        # pin BLAS threads before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # (the multi-threaded BLAS otherwise appears to
os.environ.setdefault("MKL_NUM_THREADS", "1")        #  hang the heavy runs -- see README)

import argparse
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                             # so the sibling modules import from anywhere

import fig_weis_reference as FR                       # noqa: E402  (1-D convergence figure)
import fig_weis_profiles as FP                        # noqa: E402  (1-D profiles + reference caches)
import fig_weis_verification as FV                    # noqa: E402  (2-D-on-1-D overlay, cache-only)
import fig_weis_single_phase as FS                    # noqa: E402  (single-phase figure, cache-only)

PY = sys.executable                                   # reuse the invoking (porepy-env) interpreter

PP_SCHEMES = {"hu": False, "hu_mw": True}             # overlay scheme -> weighted_perm for run_case
ORIENTATIONS = ["horizontal", "vertical"]


# --------------------------------------------------------------------------------------- #
#  Output redirection (quick mode only)
# --------------------------------------------------------------------------------------- #
def _sandbox_outputs(root):
    """Point every figure module's cache + output dir at a scratch ``root`` so a quick smoke run
    neither reads nor writes the real ``_cache/`` and ``figures/``."""
    cache, figs = os.path.join(root, "_cache"), os.path.join(root, "figures")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    for mod in (FR, FP, FV):
        mod.CACHE_DIR, mod.OUT_DIR = cache, figs
    FS.CACHE_DIR, FS.FIG_DIR = cache, figs
    return cache, figs


# --------------------------------------------------------------------------------------- #
#  Stages
# --------------------------------------------------------------------------------------- #
def stage_reference(quick, parallel):
    """1-D convergence: spatial refinement (a) and OBL table-level refinement (b)."""
    if quick:                                          # tiny N, coarse dt, short (horizontal) case,
        dt, case = FR.m.DT0, "horizontal"              # skip the ~1.4 GB level-4 table -- smoke only
        sp = FR.compute_spatial(N_levels=[50, 100], dt=dt, case=case, parallel=parallel)
        ob = FR.compute_obl(levels=[0, 1], N=100, dt=dt, case=case, parallel=parallel)
    else:
        sp = FR.compute_spatial(parallel=parallel)
        ob = FR.compute_obl(parallel=parallel)
    FR.plot(sp, ob)


def stage_profiles(quick, parallel):
    """1-D converged profiles of PPU/HU/HU-mw; also writes _cache/profiles_* for the overlay."""
    if quick:                                          # small N + capped integration (smoke, not physics)
        out = FP.compute(N=100, n_steps=40, parallel=parallel)
        extra = FP.compute_extra(N=50)
    else:
        out = FP.compute(parallel=parallel)
        extra = FP.compute_extra()
    FP.plot(out, extra_vertical=extra)


def stage_porepy(orientations, schemes, no_cache):
    """PorePy 2-D column per (orientation, scheme), each in its own process (clean PETSc state).
    Resumable: run_case loads an existing pickle and skips the run unless ``no_cache``."""
    for orient in orientations:
        for sk in schemes:
            code = (f"import porepy_1d_solver as P; "
                    f"P.run_case({orient!r}, {PP_SCHEMES[sk]}, cache={not no_cache})")
            t0 = time.time()
            print(f"  [run ] porepy {orient}/{sk}  (tail the run's own stdout below)", flush=True)
            rc = subprocess.run([PY, "-c", code], cwd=HERE).returncode
            print(f"  [{'ok  ' if rc == 0 else 'FAIL'}] porepy {orient}/{sk}"
                  f"  ({(time.time() - t0) / 60.0:.1f} min)", flush=True)


def stage_verification():
    """Overlay each PorePy 2-D scheme on its 1-D reference, per orientation (cache-only, fast).
    A scheme with no PorePy cache is drawn as its 1-D reference alone."""
    for case in ORIENTATIONS:
        FV.plot_verification(case, schemes=("hu", "hu_mw"))


def stage_single_phase_runs(no_cache):
    """Six single-phase heating fronts ({hP,mP,lP} x {horizontal,vertical}) via the solver's CLI."""
    cmd = [PY, "single_phase_porepy_1d_solver.py", "--case", "all", "--geometry", "all"]
    if no_cache:
        cmd.append("--no-cache")
    subprocess.run(cmd, cwd=HERE)


def stage_single_phase_fig():
    """Assemble the 3x2 single-phase figure from _cache/ (missing panels render a placeholder)."""
    FS.main()


def _stage(label, fn, *a):
    """Run one stage with a header, timing, and report-don't-abort error handling."""
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    try:
        fn(*a)
        ok = True
    except Exception as exc:                            # keep going -- one figure must not sink the run
        print(f"  [FAIL] {type(exc).__name__}: {exc}", flush=True)
        ok = False
    print(f"  ({(time.time() - t0) / 60.0:.1f} min)", flush=True)
    return ok


# --------------------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the subsection 4.2 one-dimensional (Weis 2014) figures end to end: "
                    "1-D reference + profiles, the heavy PorePy 2-D overlays, verification, and the "
                    "single-phase figure. Resumable via _cache/.")
    ap.add_argument("--quick", action="store_true",
                    help="coarse/short 1-D smoke, sandboxed to _quick/ (skips the heavy 2-D solves; "
                         "real _cache/ and figures/ are untouched)")
    ap.add_argument("--plot-only", action="store_true",
                    help="build every figure from existing _cache only; run no heavy 2-D solves")
    ap.add_argument("--skip-run", action="store_true", help="alias of --plot-only")
    ap.add_argument("--skip-porepy", action="store_true",
                    help="skip the PorePy 2-D overlay runs (verification then shows 1-D references)")
    ap.add_argument("--skip-single-phase", action="store_true",
                    help="skip the single-phase track (its 2-D runs and figure)")
    ap.add_argument("--single-phase-only", action="store_true",
                    help="run only the single-phase track (no multiphase reference/profiles/overlay)")
    ap.add_argument("--porepy-orientations", nargs="+", default=ORIENTATIONS,
                    choices=ORIENTATIONS, metavar="ORIENT",
                    help="orientations for the 2-D overlay runs (default: both)")
    ap.add_argument("--porepy-schemes", nargs="+", default=["hu"], choices=list(PP_SCHEMES),
                    metavar="SCHEME",
                    help="schemes for the 2-D overlay runs (default: hu; add hu_mw for the full "
                         "overlay -- heavy new runs)")
    ap.add_argument("--serial", action="store_true",
                    help="run the 1-D sweeps serially (default: parallel process pool)")
    ap.add_argument("--no-cache", action="store_true",
                    help="force the heavy 2-D solves to recompute even when a cache pickle exists")
    args = ap.parse_args(argv)

    plot_only = args.plot_only or args.skip_run
    # Quick mode forces serial: the sandbox redirection below patches the figure modules' CACHE_DIR
    # in THIS process, but a spawn worker pool re-imports them fresh and would revert to the real
    # _cache/. Serial keeps every run in-process, so the sandbox holds.
    parallel = not (args.serial or args.quick)
    do_multiphase = not args.single_phase_only
    do_single = args.single_phase_only or not args.skip_single_phase
    tag = "QUICK" if args.quick else "FULL"
    print(f"Weis-1D reproduction workflow -- {tag} config "
          f"(parallel={parallel}, plot_only={plot_only})", flush=True)

    if args.quick:
        cache, figs = _sandbox_outputs(os.path.join(HERE, "_quick"))
        print(f"  sandboxed to {os.path.relpath(os.path.dirname(cache), HERE)}/ "
              f"-- real _cache/ and figures/ untouched", flush=True)

    t_all = time.time()
    if do_multiphase:
        _stage("reference figure (1-D convergence)", stage_reference, args.quick, parallel)
        _stage("profiles figure (1-D)", stage_profiles, args.quick, parallel)
        if not (plot_only or args.skip_porepy or args.quick):
            _stage("PorePy 2-D overlay runs (heavy)", stage_porepy,
                   args.porepy_orientations, args.porepy_schemes, args.no_cache)
        else:
            print("\n(skipping PorePy 2-D overlay runs)", flush=True)
        _stage("verification figure (overlay)", stage_verification)

    if do_single:
        if not (plot_only or args.quick):
            _stage("single-phase 2-D runs (heavy)", stage_single_phase_runs, args.no_cache)
        else:
            print("\n(skipping single-phase 2-D runs)", flush=True)
        _stage("single-phase figure", stage_single_phase_fig)

    out = "_quick/figures" if args.quick else "figures"
    print(f"\n{'=' * 70}\n workflow complete in {(time.time() - t_all) / 60.0:.1f} min "
          f"({tag}) -- figures in {out}/\n{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
