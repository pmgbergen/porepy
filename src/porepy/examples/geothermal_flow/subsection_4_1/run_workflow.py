#!/usr/bin/env python
"""Subsection 4.2 workflow: all reference cases + figures in one command (Bosma Ex. 6.3).

Runs the completion checks first (reduction consistency + two-cell monotonicity/derivative sign,
both solvers; --skip-checks to opt out), then per N in --nphase (default 3 4; rho evenly spaced on
[500, 1500], N=3 = Bosma [1500, 1000, 500]):

  hamon (FV reference, parallel worker processes):
    step 1: schemes {hu, hu-mw, hu-mp, ppu} on 100^2 to 571 d, snaps {0, 78, 571} -> vtr[_nN]/
    step 2: same schemes on 200^2 to 78 d, snaps {0, 78}          -> output_ref_<scheme>[_nN]/
  porepy (CF model, subprocesses of porepy_2d_solver.py):
    scheme hu x {fixed-dim, --md}  -> visualization_barriers[_frac]_hu_N<n>/  (+ .log here)
  figures: plot_reference.py (hamon) + plot_porepy.py (saturation_maps_pp_hu, ... from the VTUs)
           -> figures/n<N>/

Usage:
    python run_workflow.py [--nphase 3 4] [--quick] [--jobs J] [--plot-only] [--skip-plot]
                           [--skip-porepy] [--linear-solver cpr|scipy] [--dir-lag iteration|step]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # single-threaded per process (set before numpy)

import argparse
import multiprocessing as mp
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                         # so the sibling modules import from anywhere

import completion_checks as CC                      # noqa: E402
import hamon_2d_solver as H                        # noqa: E402
import plot_porepy as PP                            # noqa: E402  (porepy-data figures from the VTUs)
import plot_reference as PR                        # noqa: E402  (hamon reference figures)
import run_reference as RR                         # noqa: E402  (reuse its _suffix convention)

SCHEMES = ("hu", "ppu", "hu-mw", "hu-mp")

# Two run configurations. FULL == the paper reference (identical to run_reference.py); QUICK is a
# coarse, short smoke test that still exercises every code path and produces real figures.
FULL = {
    "step1": dict(nx=100, ny=100, t_end_days=None, snap_days=H.SNAP_DAYS),   # = porepy sampling
    "step2": dict(nx=200, ny=200, t_end_days=78.0, snap_days=(0.0, 78.0)),
    "fig_days": (0, 78, 571), "gas_day": 78,
    "porepy_days": None,                          # None -> the solver's full default horizon
}
QUICK = {
    "step1": dict(nx=24, ny=24, t_end_days=6.0, snap_days=(0.0, 3.0, 6.0)),
    "step2": dict(nx=32, ny=32, t_end_days=6.0, snap_days=(0.0, 6.0)),
    "fig_days": (0, 3, 6), "gas_day": 6,
    "porepy_days": 6.0,
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
#  PorePy CF cases: porepy_2d_solver.py, scheme hu x {fixed-dim, --md}, run as subprocesses
#  (clean per-run PETSc/global state).  Output goes to the solver's own
#  visualization_barriers[_frac]_hu_N<n>/ ; stdout+stderr to porepy_hu_N<n>[_md].log here.
# --------------------------------------------------------------------------------------- #
def _build_porepy_tasks(nphases, cfg, cases=("fd", "md")):
    md_flags = [c == "md" for c in cases]
    tasks = []
    for n in nphases:
        for md in md_flags:
            cmd = [sys.executable, os.path.join(HERE, "porepy_2d_solver.py"),
                   "--nphase", str(n), "--scheme", "hu"] + (["--md"] if md else [])
            if cfg.get("porepy_days"):
                cmd += ["--days", str(cfg["porepy_days"])]
            tasks.append(dict(N=n, md=md, cmd=cmd,
                              log=os.path.join(HERE, f"porepy_hu_N{n}{'_md' if md else ''}.log")))
    return tasks


def _run_one_porepy(task):
    tag = f"porepy hu N={task['N']}{' --md' if task['md'] else ''}"
    print(f"  [started ] {tag}  (tail -f {os.path.basename(task['log'])} for live output)",
          flush=True)
    t0 = time.time()
    with open(task["log"], "w") as fh:
        rc = subprocess.run(task["cmd"], stdout=fh, stderr=subprocess.STDOUT, cwd=HERE).returncode
    return dict(N=task["N"], step="porepy" + ("-md" if task["md"] else ""), scheme="hu(porepy)",
                summary=f"exit={rc} (log: {os.path.basename(task['log'])})",
                converged=(rc == 0), wall=time.time() - t0)


# --------------------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------------------- #
def _run_simulations(tasks, jobs, worker=_run_one):
    print(f"\n=== running {len(tasks)} simulations on {jobs} worker process(es) ===", flush=True)
    t0 = time.time()
    results = []
    if jobs == 1:                                        # serial (easier to debug / low-core)
        for task in tasks:
            r = worker(task)
            results.append(r)
            _log_result(r)
    else:
        ctx = mp.get_context("spawn")                    # spawn: each worker re-imports cleanly
        pool = ctx.Pool(jobs)
        try:
            for r in pool.imap_unordered(worker, tasks):
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


def _run_checks(quick, skip_porepy):
    """Completion checks (reduction consistency + two-cell monotonicity/derivative sign) that cover
    the HU-BM theory, driven by both solvers. Reports failures but never aborts the workflow."""
    t0 = time.time()
    results = CC.run_all_checks(quick=quick, skip_porepy=skip_porepy)
    n_fail = sum(not r.get("passed", False) for r in results)
    print(f"--- {len(results)} checks in {time.time() - t0:.1f}s ({n_fail} failed) ---", flush=True)
    if n_fail:
        print("WARNING: completion checks reported failures (see above)", flush=True)
    return results


def _build_figures(nphases, cfg):
    days = [str(d) for d in cfg["fig_days"]]
    for n in nphases:
        print(f"\n=== hamon figures for N={n}  ->  figures/n{n}/ ===", flush=True)
        PR.main(["--nphase", str(n), "--days", *days, "--gas-day", str(cfg["gas_day"])])
    # PorePy-data figures (saturation_maps_pp_hu, saturation_grid_pp_hu, conservation_pp_hu, ...)
    # from the exported visualization_barriers*_hu_N<n>/ VTUs. Reads only -- no simulations -- so it
    # runs under --plot-only too; plot_porepy skips any N/case whose output dir is absent.
    print(f"\n=== porepy figures for N in {nphases}  ->  figures/n<N>/ ===", flush=True)
    try:
        PP.main(["--nphase", *[str(n) for n in nphases], "--days", *days])
    except Exception as exc:                            # a missing/partial VTU set must not abort
        print(f"  [FAIL] porepy figures: {type(exc).__name__}: {exc}", flush=True)


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
    ap.add_argument("--plot-only", action="store_true",
                    help="run NO simulations (neither hamon_2d_solver nor porepy_2d_solver); "
                         "build the figures from existing output only")
    ap.add_argument("--skip-run", action="store_true", help="alias of --plot-only")
    ap.add_argument("--skip-plot", action="store_true", help="skip the figures (simulations only)")
    ap.add_argument("--pdf", action="store_true",
                    help="also write a vector PDF next to each figure PNG (default: PNG only)")
    ap.add_argument("--skip-porepy", action="store_true",
                    help="skip the porepy_2d_solver cases (hamon reference only)")
    ap.add_argument("--skip-checks", action="store_true",
                    help="skip the completion checks (reduction consistency + two-cell "
                         "monotonicity) that run by default before the simulations")
    ap.add_argument("--checks-only", action="store_true",
                    help="run ONLY the completion checks (+ monotonicity figure); no simulations, "
                         "no reference figures")
    ap.add_argument("--porepy-only", action="store_true",
                    help="run ONLY the porepy_2d_solver cases (no hamon runs, no figures)")
    ap.add_argument("--porepy-cases", nargs="+", default=["fd", "md"],
                    choices=["fd", "md"], metavar="CASE",
                    help="which porepy cases to run: fd (fixed-dim), md, or both (default)")
    args = ap.parse_args(argv)

    CC.SAVE_PDF = args.pdf     # completion-check monotonicity.png gets a .pdf too when --pdf is set

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
    if args.checks_only:
        _run_checks(args.quick, args.skip_porepy)
        print(f"\n workflow (checks only) complete in {(time.time() - t_all) / 60.0:.1f} min")
        return

    if not args.skip_checks:                           # cheap (two-cell); run in every mode,
        _run_checks(args.quick, args.skip_porepy)      # incl. --plot-only, so the monotonicity
        #                                                figure + degenerate table always refresh

    if not (args.plot_only or args.skip_run):
        results = []
        if not args.porepy_only:
            tasks = _build_tasks(nphases, cfg, linear_solver, args.dir_lag)
            results = _run_simulations(tasks, jobs)
        if not args.skip_porepy:                          # porepy CF cases: hu x {fd, --md} per N
            ptasks = _build_porepy_tasks(nphases, cfg, cases=args.porepy_cases)
            results += _run_simulations(ptasks, min(len(ptasks), jobs),
                                        worker=_run_one_porepy)
        stalled = [r for r in results if not r["converged"]]
        if stalled:
            print("\nWARNING: some runs stalled or failed:", flush=True)
            for r in stalled:
                print(f"  N={r['N']} {r['step']} {H.scheme_label(r['scheme'])}", flush=True)
    else:
        print("(--plot-only: no simulations; using existing output)")

    if not (args.skip_plot or args.porepy_only):
        PR.SAVE_PDF = PP.SAVE_PDF = args.pdf     # PNG only unless --pdf (vector PDFs are the slow part)
        _build_figures(nphases, cfg)
    else:
        print("(no hamon figures generated)")

    print(f"\n{'=' * 70}\n workflow complete in {(time.time() - t_all) / 60.0:.1f} min  "
          f"(N in {nphases}, {tag})\n{'=' * 70}")
    for n in nphases:
        sfx = RR._suffix(n)
        print(f"  N={n}:  hamon -> vtr{sfx}/ , output_ref_*{sfx}/     "
              f"porepy -> visualization_barriers[_frac]_hu_N{n}/     figures -> figures/n{n}/")


if __name__ == "__main__":
    main()


# python run_workflow.py  94539.88s user 77494.89s system 894% cpu 5:20:35.36 total
# python run_workflow.py  141500.16s user 130354.66s system 1022% cpu 7:22:55.12 total

