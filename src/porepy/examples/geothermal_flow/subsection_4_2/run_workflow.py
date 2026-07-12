#!/usr/bin/env python
"""One-command reproduction workflow for subsection 4.2 (HU-BM gravity segregation through
barriers, Bosma et al. 2022 Ex. 6.3).

For every requested ``--nphase N`` it (1) RUNS all four schemes -- HU-BM(ff)=``hu`` /
HU-BM(mw)=``hu-mw`` / HU-BM(mp)=``hu-mp`` and PPU=``ppu`` -- through the two-step reference plan,
then (2) BUILDS every figure:

    step 1 : 100^2 to 571 days (snaps 0/78/571)   -> ./vtr[_nN]/  (+ stats)
    step 2 : 200^2 to  78 days (snaps 0/78)        -> ./output_ref_<scheme>[_nN]/  (+ stats)
    figures: maps + per-phase grid + lightest-phase comparison  -> ./figures[_nN]/

N=3 reproduces Bosma Fig. 5 exactly; N=4 splits the oil into a mid-heavy + mid-light phase. N != 3
output is written to ``_nN``-suffixed directories so the runs never clobber one another.

The individual simulation runs are independent, so they are executed in PARALLEL across worker
PROCESSES (each solver run is single-threaded numpy, ``OMP_NUM_THREADS=1``). The whole N=3 + N=4
sweep therefore finishes in a fraction of the sequential wall time. Figures are drawn after all
runs for a given N have completed.

Usage:
    python run_workflow.py                 # FULL reference, N=3 and N=4, all figures
    python run_workflow.py --nphase 3      # only N=3 (repeatable: --nphase 3 4 5)
    python run_workflow.py --quick         # coarse/short config -- fast end-to-end smoke test
    python run_workflow.py --jobs 4        # cap the number of parallel worker processes
    python run_workflow.py --skip-run      # (re)build figures only, from existing sim output
    python run_workflow.py --skip-plot     # run the simulations only, no figures
    python run_workflow.py --linear-solver scipy   # spsolve instead of the default CPR
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # single-threaded per process (set before numpy)

import argparse
import multiprocessing as mp
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                         # so the sibling modules import from anywhere

import hamon_2d_solver as H                        # noqa: E402
import plot_reference as PR                        # noqa: E402
import run_reference as RR                         # noqa: E402  (reuse its _suffix convention)

SCHEMES = ("hu", "ppu", "hu-mw", "hu-mp")

# Two run configurations. FULL == the paper reference (identical to run_reference.py); QUICK is a
# coarse, short smoke test that still exercises every code path and produces real figures.
FULL = {
    "step1": dict(nx=100, ny=100, t_end_days=None, snap_days=(0.0, 78.0, 571.0)),
    "step2": dict(nx=200, ny=200, t_end_days=78.0, snap_days=(0.0, 78.0)),
    "fig_days": (0, 78, 571), "gas_day": 78,
}
QUICK = {
    "step1": dict(nx=24, ny=24, t_end_days=6.0, snap_days=(0.0, 3.0, 6.0)),
    "step2": dict(nx=32, ny=32, t_end_days=6.0, snap_days=(0.0, 6.0)),
    "fig_days": (0, 3, 6), "gas_day": 6,
}


# --------------------------------------------------------------------------------------- #
#  Simulation task list + worker (module-level so it is picklable for multiprocessing)
# --------------------------------------------------------------------------------------- #
def _build_tasks(nphases, cfg, linear_solver, dir_lag):
    """One task per (N, step, scheme). Each is a self-contained dict for a worker process."""
    tasks = []
    for n in nphases:
        sfx = RR._suffix(n)
        for scheme in SCHEMES:                                          # step 1 -> vtr[_nN]/
            tasks.append(dict(
                N=n, step="step1", scheme=scheme,
                out_dir=os.path.join(HERE, f"vtr{sfx}"),
                kw=dict(cfg["step1"], nphase=n, linear_solver=linear_solver, dir_lag=dir_lag)))
        for scheme in SCHEMES:                                          # step 2 -> output_ref_*[_nN]/
            tasks.append(dict(
                N=n, step="step2", scheme=scheme,
                out_dir=os.path.join(HERE, f"output_ref_{scheme.replace('-', '_')}{sfx}"),
                kw=dict(cfg["step2"], nphase=n, linear_solver=linear_solver, dir_lag=dir_lag)))
    return tasks


def _run_one(task):
    """Run a single (N, step, scheme) simulation and write its .vtr + stats."""
    t0 = time.time()
    grid, snaps, stats = H.run(task["scheme"], verbose=False, **task["kw"])
    H.write_snapshots_vtr(task["out_dir"], task["scheme"], grid, snaps)
    H.write_stats(task["out_dir"], stats)
    return dict(N=task["N"], step=task["step"], scheme=task["scheme"],
                summary=stats.summary(), converged=stats.converged, wall=time.time() - t0)


# --------------------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------------------- #
def _run_simulations(tasks, jobs):
    print(f"\n=== running {len(tasks)} simulations on {jobs} worker process(es) ===", flush=True)
    t0 = time.time()
    results = []
    if jobs == 1:                                        # serial (easier to debug / low-core)
        for task in tasks:
            r = _run_one(task)
            results.append(r)
            _log_result(r)
    else:
        ctx = mp.get_context("spawn")                    # spawn: each worker re-imports cleanly
        pool = ctx.Pool(jobs)
        try:
            for r in pool.imap_unordered(_run_one, tasks):
                results.append(r)
                _log_result(r)
            pool.close()                                 # no more work; let workers drain
            pool.join()                                  # graceful exit -> no SIGTERM, so PETSc/MPI
        finally:                                         #   (CPR) finalizes cleanly (no MPI_Abort noise)
            pool.terminate()                             # safety net if the loop raised
    n_bad = sum(not r["converged"] for r in results)
    print(f"--- {len(results)} runs done in {(time.time() - t0) / 60.0:.1f} min "
          f"({n_bad} stalled) ---", flush=True)
    return results


def _log_result(r):
    flag = "" if r["converged"] else "  ** STALLED **"
    print(f"  [N={r['N']} {r['step']}] {H.scheme_label(r['scheme']):9s}  "
          f"{r['summary']}  [{r['wall']:.1f}s]{flag}", flush=True)


def _build_figures(nphases, cfg):
    for n in nphases:
        print(f"\n=== figures for N={n}  ->  figures{RR._suffix(n)}/ ===", flush=True)
        argv = ["--nphase", str(n),
                "--days", *[str(d) for d in cfg["fig_days"]],
                "--gas-day", str(cfg["gas_day"])]
        PR.main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the subsection 4.2 HU-BM reference simulations AND build all figures, "
                    "for N=3 (Bosma Ex. 6.3) and N=4 (oil split), in one command.")
    ap.add_argument("--nphase", type=int, nargs="+", default=[3, 4],
                    help="phase counts to run (default: 3 4). N != 3 uses _nN-suffixed dirs.")
    ap.add_argument("--quick", action="store_true",
                    help="coarse/short config for a fast end-to-end smoke test (not the paper run)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel worker processes (default: min(#tasks, cpu_count-2))")
    ap.add_argument("--linear-solver", default=None, choices=["cpr", "scipy"],
                    help="Newton linear solver for every run (default: cpr -- fast; scipy's direct "
                         "solve is impractically slow at full scale). The solution is "
                         "solver-independent, so the figures are identical either way.")
    ap.add_argument("--dir-lag", default="iteration", choices=["iteration", "step"],
                    help="upwind-direction lagging cadence (default: iteration)")
    ap.add_argument("--skip-run", action="store_true", help="skip the simulations (figures only)")
    ap.add_argument("--skip-plot", action="store_true", help="skip the figures (simulations only)")
    args = ap.parse_args(argv)

    cfg = QUICK if args.quick else FULL
    nphases = args.nphase
    tag = "QUICK" if args.quick else "FULL"

    n_tasks = len(nphases) * 2 * len(SCHEMES)                        # (steps) x (schemes) x (N)
    jobs = args.jobs or max(1, min(n_tasks, (os.cpu_count() or 2) - 2))
    # Default to the fast CPR solver (scipy's direct solve is impractically slow at the full 100^2/
    # 200^2 scale). The parallel pool exits gracefully (close/join), so PETSc/MPI finalizes cleanly.
    linear_solver = args.linear_solver or "cpr"
    print(f"HU-BM reproduction workflow -- {tag} config, N in {nphases}, "
          f"linear_solver={linear_solver} (jobs={jobs}), dir_lag={args.dir_lag}")

    t_all = time.time()
    if not args.skip_run:
        tasks = _build_tasks(nphases, cfg, linear_solver, args.dir_lag)
        results = _run_simulations(tasks, jobs)
        stalled = [r for r in results if not r["converged"]]
        if stalled:
            print("\nWARNING: some runs stalled (accepted a step at the dt floor):", flush=True)
            for r in stalled:
                print(f"  N={r['N']} {r['step']} {H.scheme_label(r['scheme'])}", flush=True)
    else:
        print("(--skip-run: using existing simulation output)")

    if not args.skip_plot:
        _build_figures(nphases, cfg)
    else:
        print("(--skip-plot: no figures generated)")

    print(f"\n{'=' * 70}\n workflow complete in {(time.time() - t_all) / 60.0:.1f} min  "
          f"(N in {nphases}, {tag})\n{'=' * 70}")
    for n in nphases:
        sfx = RR._suffix(n)
        print(f"  N={n}:  sims -> vtr{sfx}/ , output_ref_*{sfx}/     figures -> figures{sfx}/")


if __name__ == "__main__":
    main()
