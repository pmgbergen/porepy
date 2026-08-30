#!/usr/bin/env python
"""Run a set of porepy_2d_solver.py parametrizations in parallel, with a live per-process
progress monitor and per-process + overall timing.

Each scenario runs as its own subprocess; stdout+stderr stream (unbuffered) to
run_scenarios_logs/<label>.log, and every scenario writes its VTUs to its own
visualization_<label>/ folder, so nothing collides.  The monitor redraws a status table
every --interval seconds showing, per process: status, elapsed time, VTU count, and the
latest solver output line.  On completion it prints each process's total time and the
overall wall time (and writes run_scenarios_logs/summary.txt).

Success is judged by a COMPLETION MARKER (visualization_<label>/run_complete.json) that the
solver writes only when the time loop reaches the final time -- NOT by the exit code, so a
run that finished but exited via a teardown signal (e.g. SIGPIPE -> rc -13) still counts as
complete (shown as "OK*").  A scenario whose marker already exists is CACHED: it is skipped
and never recomputed (use --force to recompute anyway).

Usage (run with the porepy env python):
    /Users/oduran/miniconda/envs/porepy/bin/python run_scenarios.py
    ... run_scenarios.py --jobs 4                 # concurrency (default 3, memory-safe)
    ... run_scenarios.py --force                  # ignore cache, recompute everything
    ... run_scenarios.py --threads-per-job 2      # BLAS/numba threads per child
    ... run_scenarios.py --interval 10            # status redraw period [s]

Labels match the solver's output-folder tags: the log <label>.log pairs with
visualization_<label>/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(HERE, "porepy_2d_solver.py")

sys.path.insert(0, HERE)
from case_naming import case_tag                          # noqa: E402  (label == solver output-folder tag)

# HARD-CODED globals: every scenario runs at 50 m resolution on the truncated domain.
_GLOBAL_ARGS = ["--cell-size", "50", "--truncated-domain"]

# Each scenario is defined by its distinguishing case_tag kwargs.  The LABEL and the CLI args are BOTH
# derived from it (+ the globals below), so the label can never desync from the solver's
# visualization_<tag>/ folder -- add/remove a flag in one place only.
_SCEN_SPECS: list[dict] = [
    dict(scheme="hu"),
    dict(scheme="hu", md=True),
    dict(scheme="hu", md=True, recombine=True),
    dict(scheme="hu", md=True, consistent=True),
    dict(scheme="hu", md=True, recombine=True, consistent=True),
    dict(scheme="hu", md=True, recombine=True, q_anomaly=9.0),
    dict(scheme="hu", md=True, recombine=True, consistent=True, q_anomaly=9.0),
    dict(scheme="hu", md=True, recombine=True, q_anomaly=9.0, z_init=0.032),
    dict(scheme="hu", md=True, recombine=True, consistent=True, q_anomaly=9.0, z_init=0.032),
]


def _spec_args(spec: dict) -> list[str]:
    """The distinguishing CLI flags for one scenario spec (the globals are appended separately)."""
    a = ["--scheme", spec["scheme"]]
    if spec.get("md"):
        a.append("--md")
    if spec.get("recombine"):
        a.append("--recombine")
    if spec.get("consistent"):
        a.append("--consistent")
    if spec.get("q_anomaly") is not None:
        a += ["--q-anomaly", f"{spec['q_anomaly']:g}"]
    if spec.get("z_init") is not None:
        a += ["--z-init", f"{spec['z_init']:g}"]
    return a


# (label == solver output-folder tag, extra CLI args for porepy_2d_solver.py)
SCENARIOS: list[tuple[str, list[str]]] = [
    (case_tag(cell_size=50.0, truncated_domain=True, **spec), _spec_args(spec) + _GLOBAL_ARGS)
    for spec in _SCEN_SPECS
]

_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _tail_line(path: str, maxbytes: int = 8192) -> str:
    """Last non-empty line of a log file (reads only the tail)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - maxbytes))
            text = f.read().decode("utf-8", "replace")
        for ln in reversed(text.splitlines()):
            if ln.strip():
                return ln.strip()
    except OSError:
        pass
    return ""


def _vtu_count(label: str) -> int:
    return len(glob.glob(os.path.join(HERE, f"visualization_{label}", "*.vtu")))


_MARKER = "run_complete.json"


def _marker_path(label: str) -> str:
    return os.path.join(HERE, f"visualization_{label}", _MARKER)


def _completed(label: str, since: float | None = None) -> bool:
    """True if the scenario's completion marker exists -- the solver writes it ONLY when the
    time loop reached the final time (not the trailing flux prints).  If ``since`` (a
    time.time() reference) is given, the marker must be at least that fresh, so a stale marker
    from an earlier run is not credited to this one."""
    p = _marker_path(label)
    if not os.path.isfile(p):
        return False
    if since is None:
        return True
    try:
        return os.path.getmtime(p) >= since - 2.0
    except OSError:
        return False


def _status_str(r: dict) -> str:
    s = r["status"]
    if s in ("cached", "run", "queued"):
        return {"cached": "cached", "run": "RUN", "queued": "queued"}[s]
    if r.get("completed"):                        # done
        return "OK" if r["rc"] == 0 else "OK*"    # OK* = completed, nonzero exit (teardown signal)
    return f"FAIL {r['rc']}"


def _render(records: list[dict], overall_start: float, tty: bool) -> None:
    now = time.monotonic()
    n_done = sum(1 for r in records if r["status"] == "done")
    n_run = sum(1 for r in records if r["status"] == "run")
    n_queue = sum(1 for r in records if r["status"] == "queued")
    n_cached = sum(1 for r in records if r["status"] == "cached")
    lines = [
        f"parallel scenarios  |  elapsed {_fmt(now - overall_start)}  |  "
        f"{n_done} done / {n_run} running / {n_queue} queued"
        + (f" / {n_cached} cached" if n_cached else ""),
        f"{'#':>2} {'scenario':28s} {'status':9s} {'time':>8s} {'vtu':>5s}  progress",
        "-" * 108,
    ]
    for r in records:
        st = _status_str(r)
        if r["status"] == "done":
            dur = _fmt(r["end"] - r["start"])
            if r.get("completed"):
                prog = "complete" if r["rc"] == 0 else f"complete (exit {r['rc']}, teardown)"
            else:
                prog = _tail_line(r["logpath"])[:64]
        elif r["status"] == "cached":
            dur, prog = "-", "cached (skipped)"
        elif r["status"] == "run":
            dur, prog = _fmt(now - r["start"]), _tail_line(r["logpath"])[:64]
        else:
            dur, prog = "-", ""
        vtu = _vtu_count(r["label"]) if r["status"] != "queued" else 0
        lines.append(f"{r['i'] + 1:>2} {r['label']:28s} {st:9s} {dur:>8s} {vtu:>5d}  {prog}")
    out = "\n".join(lines)
    if tty:
        sys.stdout.write("\033[H\033[J" + out + "\n")
    else:
        sys.stdout.write(datetime.now().strftime("[%H:%M:%S]\n") + out + "\n\n")
    sys.stdout.flush()


def _summary(records: list[dict], overall: float, logdir: str) -> None:
    lines = ["", "=" * 108, f"ALL DONE  |  overall wall time {_fmt(overall)}", "",
             f"{'#':>2} {'scenario':28s} {'status':9s} {'time':>10s}  command"]
    for r in records:
        st = _status_str(r)
        dur = _fmt(r["end"] - r["start"]) if (r.get("end") and r["status"] != "cached") else "-"
        cmd = "porepy_2d_solver.py " + " ".join(r["extra"])
        lines.append(f"{r['i'] + 1:>2} {r['label']:28s} {st:9s} {dur:>10s}  {cmd}")
    n_ok = sum(1 for r in records if r.get("completed"))
    n_fail = sum(1 for r in records if r["status"] == "done" and not r.get("completed"))
    n_cached = sum(1 for r in records if r["status"] == "cached")
    lines.append("")
    lines.append(f"{n_ok}/{len(records)} completed"
                 + (f" ({n_cached} cached)" if n_cached else "")
                 + (f", {n_fail} FAILED" if n_fail else ""))
    lines.append("OK* = reached final time but exited via a teardown signal (still complete).")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(logdir, "summary.txt"), "w") as f:
        f.write(text + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run porepy_2d_solver.py scenarios in parallel.")
    ap.add_argument("--jobs", type=int, default=3,
                    help="max concurrent processes (default: 3; each --md solver can use "
                         "several GB, so 3 keeps peak memory modest -- raise for more "
                         f"parallelism, up to {len(SCENARIOS)})")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter for the children (default: this one)")
    ap.add_argument("--logdir", default=os.path.join(HERE, "run_scenarios_logs"),
                    help="directory for per-scenario .log files and summary.txt")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="status-table redraw period in seconds (default 5)")
    ap.add_argument("--threads-per-job", default="auto",
                    help="BLAS/OpenMP/numba threads per child: an int, or 'auto' "
                         "(= cpus // jobs), or 0 to leave the environment untouched")
    ap.add_argument("--force", action="store_true",
                    help="recompute every scenario even if its completion marker already "
                         "exists (default: skip already-completed scenarios = cache)")
    args = ap.parse_args()

    if not os.path.isfile(SOLVER):
        sys.exit(f"solver not found: {SOLVER}")
    os.makedirs(args.logdir, exist_ok=True)
    jobs = max(1, min(args.jobs, len(SCENARIOS)))

    env = os.environ.copy()
    if str(args.threads_per_job) != "0":
        if args.threads_per_job == "auto":
            tpj = max(1, (os.cpu_count() or 4) // jobs)
        else:
            tpj = max(1, int(args.threads_per_job))
        for var in _THREAD_ENV_VARS:
            env[var] = str(tpj)

    records = [dict(i=i, label=label, extra=extra, status="queued", proc=None, lf=None,
                    logpath=os.path.join(args.logdir, f"{label}.log"),
                    start=None, start_wall=None, end=None, rc=None, completed=None)
               for i, (label, extra) in enumerate(SCENARIOS)]
    queue = list(records)
    running: list[dict] = []
    overall_start = time.monotonic()
    tty = sys.stdout.isatty()

    print(f"launching {len(records)} scenarios, up to {jobs} at a time "
          f"(threads/job={env.get('OMP_NUM_THREADS', 'default')}; "
          f"{'FORCE recompute' if args.force else 'completed runs are cached/skipped'}); "
          f"logs in {args.logdir}")
    try:
        while queue or running:
            while queue and len(running) < jobs:
                r = queue.pop(0)
                if not args.force and _completed(r["label"]):     # cache: already finished
                    r["status"] = "cached"
                    r["start"] = r["end"] = time.monotonic()
                    r["completed"] = True
                    r["rc"] = 0
                    continue
                r["lf"] = open(r["logpath"], "w")
                cmd = [args.python, "-u", SOLVER] + r["extra"]
                r["lf"].write("# CMD: " + " ".join(cmd) + "\n")
                r["lf"].flush()
                r["proc"] = subprocess.Popen(cmd, cwd=HERE, stdout=r["lf"],
                                             stderr=subprocess.STDOUT, env=env,
                                             start_new_session=True)
                r["start"] = time.monotonic()
                r["start_wall"] = time.time()
                r["status"] = "run"
                running.append(r)
            for r in list(running):
                rc = r["proc"].poll()
                if rc is not None:
                    r["end"] = time.monotonic()
                    r["rc"] = rc
                    r["completed"] = _completed(r["label"], since=r["start_wall"])
                    r["status"] = "done"
                    r["lf"].close()
                    running.remove(r)
            _render(records, overall_start, tty)
            if queue or running:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\ninterrupted — terminating running processes ...")
        for r in running:
            try:
                os.killpg(os.getpgid(r["proc"].pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        for r in running:
            try:
                r["proc"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(r["proc"].pid), signal.SIGKILL)
            r["end"] = time.monotonic()
            r["rc"] = r["proc"].poll()
            r["completed"] = _completed(r["label"], since=r["start_wall"])
            r["status"] = "done"
            r["lf"].close()

    overall = time.monotonic() - overall_start
    _summary(records, overall, args.logdir)
    return 1 if any(r["status"] == "done" and not r.get("completed") for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
