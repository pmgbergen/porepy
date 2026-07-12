#!/usr/bin/env python
"""Reference runs for subsection 4.2 -- three-phase gravity segregation through barriers
(Bosma et al. 2022, Ex. 6.3), driving hamon_2d_solver.py.

    Step 1  full reference      : all three schemes at 100x100 to 571 days (snaps 0/78/571)
                                  -> ./vtr/
    Step 2  high-res early time : each scheme (ppu, hu, hu-mw) at 200x200 to 78 days
                                  (snaps 0/78) -> ./output_ref_<scheme>/

Each run writes the per-snapshot ``.vtr`` files and a ``stats_<scheme>.txt`` (Newton
iterations, dt-cuts, ...). Calls the solver's ``run()`` in-process (no subprocess), using the
default CPR iterative linear solver (FGMRES + pressure-AMG/saturation-ILU).

Usage:
    python run_reference.py                # run everything (needs the `porepy` env active)
    python run_reference.py --dry-run      # print the planned runs, execute nothing
    python run_reference.py --step 2       # run only step 2 (or --step 1)
    python run_reference.py --scheme hu-mp # run only one scheme (hu / ppu / hu-mw / hu-mp)

Wall time is a few minutes on one core with the default CPR iterative solver (the direct
factorization of the Lagrange bordered system was far slower).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # set before numpy is imported (via the solver)

import argparse
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                         # so `import hamon_2d_solver` works from anywhere
import hamon_2d_solver as H                       # noqa: E402


def _plan(step):
    """Return the list of (label, scheme, out_dir, run_kwargs) for the requested step(s)."""
    plan = []
    if step in ("1", "all"):                                        # Step 1: full reference
        for scheme in ("hu", "ppu", "hu-mw", "hu-mp"):              # (defaults -> 100^2, 571 d)
            plan.append(("step1", scheme, os.path.join(HERE, "vtr"), dict(nx=100, ny=100)))
    if step in ("2", "all"):                                        # Step 2: 200^2, early time
        for scheme in ("ppu", "hu", "hu-mw", "hu-mp"):
            out = os.path.join(HERE, f"output_ref_{scheme.replace('-', '_')}")
            plan.append(("step2", scheme, out,
                         dict(nx=200, ny=200, t_end_days=78, snap_days=(0.0, 78.0))))
    return plan


def _execute(label, scheme, out_dir, kw):
    print(f"\n--- {label}: scheme={scheme}  ->  {os.path.relpath(out_dir, HERE)}/", flush=True)
    t0 = time.time()
    grid, snaps, stats = H.run(scheme, **kw)
    H.write_snapshots_vtr(out_dir, scheme, grid, snaps)
    H.write_stats(out_dir, stats)
    print(f"  {stats.summary()}   [{time.time() - t0:.1f}s]", flush=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the subsection 4.2 reference simulations (Bosma 2022 Ex. 6.3).")
    ap.add_argument("--step", choices=["1", "2", "all"], default="all",
                    help="which step(s) to run (default: all)")
    ap.add_argument("--scheme", choices=["hu", "ppu", "hu-mw", "hu-mp", "all"], default="all",
                    help="run only this scheme (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned runs without executing them")
    ap.add_argument("--linear-solver", default="cpr", choices=["cpr", "scipy"],
                    help="Newton linear solver (default: cpr = FGMRES + CPR two-stage "
                         "preconditioner, iterative and fast at scale; 'scipy' = spsolve/SuperLU "
                         "on the Lagrange bordered system).")
    ap.add_argument("--dir-lag", default="iteration", choices=["iteration", "step"],
                    help="upwind-direction lagging: 'iteration' (default, refresh each Newton "
                         "iterate) or 'step' (freeze once per time step).")
    args = ap.parse_args(argv)

    plan = _plan(args.step)
    if args.scheme != "all":                         # keep only the requested scheme
        plan = [row for row in plan if row[1] == args.scheme]
    for _, _, _, kw in plan:                         # apply the chosen solver + lag to every run
        kw["linear_solver"] = args.linear_solver
        kw["dir_lag"] = args.dir_lag
    if args.dry_run:
        for label, scheme, out_dir, kw in plan:
            print(f"{label}: run(scheme={scheme!r}, out={os.path.relpath(out_dir, HERE)!r}, "
                  f"{', '.join(f'{k}={v!r}' for k, v in kw.items())})")
        return

    t_all = time.time()
    results = [(label, _execute(label, scheme, out_dir, kw))
               for (label, scheme, out_dir, kw) in plan]

    print("\n" + "=" * 70)
    print(f" All reference runs complete in {(time.time() - t_all) / 60.0:.1f} min")
    print("=" * 70)
    for label, st in results:
        print(f"  {label}  {st.summary()}")


if __name__ == "__main__":
    main()
