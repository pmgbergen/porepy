#!/usr/bin/env python
"""Minimal figure driver for subsection 4.1.

Runs ONLY the simulations the target figures need, skips anything already on disk, then
generates exactly those figures. Every sim runs as its own subprocess (clean NPHASE / PETSc
state) with its stdout streamed through a single progress parser.

Deliverables:
  figures/monotonicity.png
  figures/n{3,4}/conservation_comparison_{n}_phases.{png,pdf}
  figures/n{3,4}/comparison_saturation_maps_fixed_dim_{n}_phases.{png,pdf}
  figures/n{3,4}/comparison_saturation_maps_mixed_dim_{n}_phases.{png,pdf}
  figures/comparison/l2_difference_fd.{png,pdf}

Progress, three ways:
  * console  -- one banner per case + a throttled live line (which case, sim time, % done, elapsed)
  * PROGRESS.txt   -- `cat` it any time for the current case and how far along it is
  * progress/NN_*.log  -- full per-case solver output; `tail -f` for step-by-step detail

Usage:
  python run_figures.py            # run missing sims (skip cached), then plot
  python run_figures.py --plot-only  # skip all sims, just (re)generate the figures
  python run_figures.py --list       # print the plan (what would run) and exit
"""
import io
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hamon_2d_solver as H          # noqa: E402  (constants + CLI target)
import plot_porepy as PLOT           # noqa: E402  (figure functions)
import completion_checks as CC       # noqa: E402  (monotonicity.png)

PROG_DIR = os.path.join(HERE, "progress")
STATUS = os.path.join(HERE, "PROGRESS.txt")
PY = sys.executable

NPHASES = [3, 4]
STD_SCHEMES = ["ppu", "hu-mp"]           # CMP_SCHEMES for conservation/l2; ppu also feeds fixed_dim
FIG_DAYS = [0.0, 78.0, 571.0]
HORIZON = float(max(H.SNAP_DAYS))        # standalone run horizon [days], for the % readout


# --------------------------------------------------------------------------- paths / cache
def std_dir(n):     return os.path.join(HERE, "vtr" if n == 3 else f"vtr_n{n}")
def frac_dir(n):    return os.path.join(HERE, "vtr_frac" if n == 3 else f"vtr_n{n}_frac")
def pp_dir(n, md):  return os.path.join(HERE, f"visualization_barriers{'_frac' if md else ''}_hu_N{n}")


def _glob(d, pat):
    import glob
    return glob.glob(os.path.join(d, pat))


def std_cached(d, scheme):
    return bool(_glob(d, f"hamon_{scheme.replace('-', '_')}_*d.vtr"))


def pp_cached(d):
    return os.path.isdir(d) and bool(_glob(d, "*.pvd")) and bool(_glob(d, "*.vtu"))


# --------------------------------------------------------------------------- progress
_RE_HAMON_STEP = re.compile(r"t=\s*([\d.]+)\s*d\s+step\s+(\d+)")
_RE_HAMON_SNAP = re.compile(r"wrote .*_(\d+)d\.vtr")
_RE_PP_TIME = re.compile(r"time=([\d.eE+\-]+)\s+of\s+([\d.eE+\-]+)")
_RE_ALERT = re.compile(r"Traceback|Error|failed to factorize|not positive definite|"
                       r"Failed to solve|DT-CUT|CONVERGED")


def _elapsed(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class CaseProgress:
    """Streams one case's solver output: full copy to a log, a throttled console line, and
    PROGRESS.txt kept current with the latest simulated-time reading."""

    def __init__(self, idx, total, label, log_path):
        self.idx, self.total, self.label = idx, total, label
        self.log = open(log_path, "w")
        self.log_rel = os.path.relpath(log_path, HERE)
        self.t0 = time.time()
        self.last_emit = 0.0
        self.detail = "starting"
        self._emit(force=True)

    def _emit(self, force=False):
        now = time.time()
        if not force and now - self.last_emit < 20.0:
            return
        self.last_emit = now
        el = _elapsed(now - self.t0)
        print(f"  [{self.idx}/{self.total}] {self.label} | {self.detail} | elapsed {el}",
              flush=True)
        try:
            with open(STATUS, "w") as f:
                f.write(f"RUNNING  [{self.idx}/{self.total}]  {self.label}\n"
                        f"  {self.detail}\n"
                        f"  elapsed {el}   (started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.t0))})\n"
                        f"  live log: {self.log_rel}\n")
        except OSError:
            pass

    def feed(self, line):
        self.log.write(line + "\n")
        self.log.flush()
        m = _RE_HAMON_STEP.search(line) or _RE_HAMON_SNAP.search(line)
        if m:
            t = float(m.group(1))
            step = f"  step {m.group(2)}" if m.re is _RE_HAMON_STEP else ""
            self.detail = f"t={t:.0f}/{HORIZON:.0f} d ({100 * t / HORIZON:.0f}%){step}"
            self._emit()
            return
        m = _RE_PP_TIME.search(line)
        if m:
            td, tot = float(m.group(1)) / 86400.0, float(m.group(2)) / 86400.0
            self.detail = f"t={td:.0f}/{tot:.0f} d ({100 * td / max(tot, 1e-9):.0f}%)"
            self._emit()
            return
        if _RE_ALERT.search(line):
            self.detail = line.strip()[:90]
            self._emit(force=True)

    def close(self):
        self.log.close()


def banner(idx, total, label):
    bar = "=" * min(90, len(label) + 18)
    print(f"\n{bar}\n>>> [{idx}/{total}] {label}   (start {time.strftime('%H:%M:%S')})\n{bar}",
          flush=True)


# --------------------------------------------------------------------------- run one sim
def _stream(cmd, cp):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         cwd=HERE, text=True, bufsize=1)
    for line in p.stdout:
        cp.feed(line.rstrip("\n"))
    p.wait()
    return p.returncode == 0


def cmd_for(task):
    kind, n, scheme, md = task["kind"], task["n"], task["scheme"], task["md"]
    if kind == "hamon":
        c = [PY, os.path.join(HERE, "hamon_2d_solver.py"),
             "--scheme", scheme, "--nphase", str(n),
             "--out", frac_dir(n) if md else std_dir(n)]
        if md:
            c.append("--fractures")
        return c
    return [PY, os.path.join(HERE, "porepy_2d_solver.py"),
            "--nphase", str(n), "--scheme", "hu"] + (["--md"] if md else [])


# --------------------------------------------------------------------------- plan
# Phase "early" holds everything the 5 non-mixed-dim deliverables need (fast standalone +
# the PorePy HU solves); "late" holds only the slow equi-dim PPU that the 2 mixed-dim maps
# need. Early runs first so the fix-relevant PorePy data is secured and most figures land
# well before the ~10 h/each equi-dim jobs finish.
def build_plan():
    plan = []
    for n in NPHASES:                                   # standalone step1 (ppu, hu-mp) -- usually cached
        for scheme in STD_SCHEMES:
            if not std_cached(std_dir(n), scheme):
                plan.append(dict(kind="hamon", n=n, scheme=scheme, md=False, phase="early",
                                 label=f"standalone {scheme} step1  N={n}  ->  {os.path.basename(std_dir(n))}/"))
    for n in NPHASES:                                   # PorePy HU fd + md
        for md in (False, True):
            if not pp_cached(pp_dir(n, md)):
                plan.append(dict(kind="porepy", n=n, scheme="hu", md=md, phase="early",
                                 label=f"PorePy HU {'md' if md else 'fd'}  N={n}  ->  {os.path.basename(pp_dir(n, md))}/"))
    for n in NPHASES:                                   # equi-dim PPU (slow: 0.1 m band) -- mixed_dim only
        if not std_cached(frac_dir(n), "ppu"):
            plan.append(dict(kind="hamon", n=n, scheme="ppu", md=True, phase="late",
                             label=f"standalone equi-dim PPU  N={n}  (SLOW, 0.1 m band)  ->  {os.path.basename(frac_dir(n))}/"))
    return plan


# --------------------------------------------------------------------------- figures
def _try(desc, fn):
    """Generate one figure, guarded: skip (don't abort) if its inputs aren't on disk yet."""
    try:
        fn()
        print(f"  [done] {desc}", flush=True)
    except Exception as exc:                                          # noqa: BLE001
        print(f"  [skip] {desc}  ({type(exc).__name__}: {str(exc)[:70]})", flush=True)


def make_figures(mono=True):
    print("\n=== generating figures (only those whose sims are ready) ===", flush=True)
    _set_status("GENERATING FIGURES")
    if mono:
        _try("figures/monotonicity.png", lambda: CC.check_monotonicity_porepy(quick=False))
    for n in NPHASES:
        out = os.path.join(HERE, "figures", f"n{n}")
        os.makedirs(out, exist_ok=True)
        if pp_cached(pp_dir(n, False)):                              # needs PorePy HU fd (+ md curves)
            _try(f"figures/n{n}/conservation_comparison_{n}_phases",
                 lambda n=n, out=out: PLOT.conservation_comparison(n, out))
            _try(f"figures/n{n}/comparison_saturation_maps_fixed_dim_{n}_phases",
                 lambda n=n, out=out: PLOT.comparison_saturation_maps_fixed_dim(n, FIG_DAYS, out))
        if pp_cached(pp_dir(n, True)):                              # needs only PorePy HU md
            _try(f"figures/n{n}/enthalpy_temperature_mixed_dim_{n}_phases",
                 lambda n=n, out=out: PLOT.enthalpy_temperature_maps_mixed_dim(n, FIG_DAYS, out))
        if pp_cached(pp_dir(n, True)) and std_cached(frac_dir(n), "ppu"):   # needs equi-dim PPU + HU md
            _try(f"figures/n{n}/comparison_saturation_maps_mixed_dim_{n}_phases",
                 lambda n=n, out=out: PLOT.comparison_saturation_maps_mixed_dim(n, FIG_DAYS, out))
    if all(pp_cached(pp_dir(n, False)) for n in NPHASES):
        cmp = os.path.join(HERE, "figures", "comparison")
        os.makedirs(cmp, exist_ok=True)
        _try("figures/comparison/l2_difference_fd", lambda: PLOT.l2_difference(NPHASES, cmp))


def _set_status(text):
    try:
        with open(STATUS, "w") as f:
            f.write(text + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- main
def _run_tasks(tasks, start_idx, total, failures):
    for k, task in enumerate(tasks):
        i = start_idx + k
        banner(i, total, task["label"])
        tag = f"{i:02d}_{task['kind']}_N{task['n']}_{task['scheme'].replace('-', '_')}{'_md' if task['md'] else ''}"
        cp = CaseProgress(i, total, task["label"], os.path.join(PROG_DIR, tag + ".log"))
        t0 = time.time()
        try:
            ok = _stream(cmd_for(task), cp)
        except Exception as exc:                        # noqa: BLE001
            ok = False
            cp.feed(f"DRIVER EXCEPTION: {exc!r}")
        cp.close()
        print(f"<<< [{i}/{total}] {'CONVERGED' if ok else 'FAILED'}  in {_elapsed(time.time() - t0)}  "
              f"({task['label'].split('->')[0].strip()})", flush=True)
        if not ok:
            failures.append((i, task["label"], cp.log_rel))


def main():
    plot_only = "--plot-only" in sys.argv
    list_only = "--list" in sys.argv
    os.makedirs(PROG_DIR, exist_ok=True)

    plan = [] if plot_only else build_plan()
    header = ("--plot-only: skipping all sims" if plot_only
              else f"{len(plan)} simulation(s) to run (cached ones skipped)")
    print(f"\n=== run_figures: {header} ===", flush=True)
    for i, task in enumerate(plan, 1):
        print(f"   [{i}/{len(plan)}] ({task['phase']:5}) {task['label']}")
    if list_only:
        print("\n(--list: nothing run)", flush=True)
        return

    t_start = time.time()
    failures = []
    early = [t for t in plan if t["phase"] == "early"]
    late = [t for t in plan if t["phase"] == "late"]

    if early:
        _run_tasks(early, 1, len(plan), failures)
    make_figures()                                      # 5 non-mixed-dim + monotonicity if ready
    if late:
        print("\n=== remaining: the slow equi-dim PPU runs (mixed_dim figures) ===", flush=True)
        _run_tasks(late, len(early) + 1, len(plan), failures)
        make_figures()                                  # + the 2 mixed_dim figures

    if failures:
        print("\n!!! simulation failures -- some figures may be missing:", flush=True)
        for i, label, log_rel in failures:
            print(f"    [{i}] {label}   (see {log_rel})", flush=True)
    total = _elapsed(time.time() - t_start)
    _set_status(f"ALL DONE in {total}" + (f"  ({len(failures)} sim failure(s))" if failures else ""))
    print(f"\n=== all done in {total} ===", flush=True)


if __name__ == "__main__":
    main()
