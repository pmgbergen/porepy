#!/usr/bin/env python
"""Reference runs for subsection 4.2 -- N-phase gravity segregation through barriers
(Bosma et al. 2022, Ex. 6.3 at --nphase 3), driving hamon_2d_solver.py.

    Step 1  full reference      : all four schemes at 100x100 to 571 days (snaps 0/78/571)
                                  -> ./vtr/
    Step 2  high-res early time : each scheme (ppu, hu, hu-mw, hu-mp) at 200x200 to 78 days
                                  (snaps 0/78) -> ./output_ref_<scheme>/

Schemes are the HU-BM (Hybrid Upwinding with Background Mobility) family: the tokens
hu / hu-mw / hu-mp are HU-BM(ff) / HU-BM(mw) / HU-BM(mp), plus ppu (phase-potential upwinding).
Tokens stay the CLI/filename keys; the HU-BM(...) labels appear in the printed output.

``--nphase 3`` (default) reproduces Bosma Fig. 5 exactly; ``--nphase 4`` splits the oil into a
mid-heavy + mid-light phase. Runs with N != 3 are written to suffixed dirs (``vtr_n4/``,
``output_ref_<scheme>_n4/``, ...) so they never clobber the N=3 reference, and ``plot_reference.py
--nphase 4`` reads them back.

Each run writes the per-snapshot ``.vtr`` files and a ``stats_<scheme>.txt`` (Newton
iterations, dt-cuts, ...). Calls the solver's ``run()`` in-process (no subprocess), using the
default CPR iterative linear solver (FGMRES + pressure-AMG/saturation-ILU).

Usage:
    python run_reference.py                # run everything (needs the `porepy` env active)
    python run_reference.py --dry-run      # print the planned runs, execute nothing
    python run_reference.py --step 2       # run only step 2 (or --step 1)
    python run_reference.py --scheme hu-mp # one scheme: hu=HU-BM(ff) hu-mw=HU-BM(mw) hu-mp=HU-BM(mp) ppu
    python run_reference.py --nphase 4     # 4-phase variant -> ./vtr_n4/, ./output_ref_*_n4/

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


def _suffix(nphase):
    """Path suffix so N != 3 runs never clobber the Bosma (N=3) reference dirs."""
    return "" if int(nphase) == 3 else f"_n{int(nphase)}"


def _plan(step, nphase=3):
    """Return the list of (label, scheme, out_dir, run_kwargs) for the requested step(s)."""
    sfx = _suffix(nphase)
    plan = []
    if step in ("1", "all"):                                        # Step 1: full reference
        for scheme in ("hu", "ppu", "hu-mw", "hu-mp"):              # (defaults -> 100^2, 571 d)
            plan.append(("step1", scheme, os.path.join(HERE, f"vtr{sfx}"), dict(nx=100, ny=100)))
    if step in ("2", "all"):                                        # Step 2: 200^2, early time
        for scheme in ("ppu", "hu", "hu-mw", "hu-mp"):
            out = os.path.join(HERE, f"output_ref_{scheme.replace('-', '_')}{sfx}")
            plan.append(("step2", scheme, out,
                         dict(nx=200, ny=200, t_end_days=78, snap_days=(0.0, 78.0))))
    return plan


def _execute(label, scheme, out_dir, kw):
    print(f"\n--- {label}: {H.scheme_label(scheme)} [{scheme}]  ->  "
          f"{os.path.relpath(out_dir, HERE)}/", flush=True)
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
                    help="run only this scheme -- hu=HU-BM(ff), hu-mw=HU-BM(mw), hu-mp=HU-BM(mp), "
                         "ppu=PPU (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned runs without executing them")
    ap.add_argument("--linear-solver", default="cpr", choices=["cpr", "scipy"],
                    help="Newton linear solver (default: cpr = FGMRES + CPR two-stage "
                         "preconditioner, iterative and fast at scale; 'scipy' = spsolve/SuperLU "
                         "on the Lagrange bordered system).")
    ap.add_argument("--dir-lag", default="iteration", choices=["iteration", "step"],
                    help="upwind-direction lagging: 'iteration' (default, refresh each Newton "
                         "iterate) or 'step' (freeze once per time step).")
    ap.add_argument("--nphase", type=int, default=3,
                    help="number of phases (default 3 = Bosma Ex. 6.3; 4 splits the oil into a "
                         "mid-heavy + mid-light phase). N != 3 writes to suffixed dirs "
                         "(vtr_n4/, output_ref_<scheme>_n4/, ...).")
    args = ap.parse_args(argv)

    plan = _plan(args.step, args.nphase)
    if args.scheme != "all":                         # keep only the requested scheme
        plan = [row for row in plan if row[1] == args.scheme]
    for _, _, _, kw in plan:                         # apply the chosen solver + lag to every run
        kw["linear_solver"] = args.linear_solver
        kw["dir_lag"] = args.dir_lag
        kw["nphase"] = args.nphase
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
