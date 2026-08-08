"""Scalability of the phz OBL evaluation (VTKSampler, tensor backend).

Times VTKSampler evaluation vs the number of query points, sampled uniformly at random inside the
(z, h, p) OBL domain, and writes the plot as PNG + PDF next to this file.

The tensor backend is LAZY: ``sample_at`` only brackets the three axes; the 8-corner gather +
trilinear blend runs on field access. So the true OBL cost = ``sample_at`` + reading the value
fields -- both are timed here (the "full eval" curve is the one that matters).

Run:  python obl_scaling.py
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from porepy.examples.geothermal_flow.obl_sampler import VTKSampler

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.normpath(os.path.join(
    HERE, os.pardir, os.pardir,
    "model_configuration", "constitutive_description", "driesner_vtk_files"))
XPH = os.path.join(TABLE_DIR, "brine_graded_xph.vtr")

# OBL domain bounds from the tensor cache (z, h, p axes)
_cache = np.load(XPH + ".obltensor.npz")
BOUNDS = [(float(_cache["az"].min()), float(_cache["az"].max())),
          (float(_cache["a2"].min()), float(_cache["a2"].max())),
          (float(_cache["ap"].min()), float(_cache["ap"].max()))]

# point counts to sweep and the fit regime (amortised, before cache spill)
N_SWEEP = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
FIT_MIN_N = 10_000
# the solver's per-iteration cell counts, for context markers
SOLVER_NCELLS = [(800, "100 m"), (3_200, "50 m"), (12_800, "25 m")]


def _reps(n):
    return 7 if n <= 30_000 else (4 if n <= 300_000 else 3)


def main():
    sampler = VTKSampler(XPH)
    sampler.conversion_factors = (1.0, 1.0, 1.0)
    print("OBL domain (z,h,p):", [(round(a, 4), round(b, 4)) for a, b in BOUNDS])
    print("backend:", type(getattr(sampler, "_backend", None)).__name__)

    rng = np.random.default_rng(0)

    def rand_pts(n):
        return np.column_stack([rng.uniform(*BOUNDS[i], n) for i in range(3)])

    sampler.sample_at(rand_pts(2000))          # warm-up
    fields = [k for k in sampler.sampled_could.point_data.keys() if not k.startswith("grad_")]
    print(f"{len(fields)} value fields per evaluation:", fields)

    def t_bracket(pts):                        # sample_at only (axis bracketing, lazy)
        t0 = time.perf_counter(); sampler.sample_at(pts); return time.perf_counter() - t0

    def t_full(pts):                           # bracket + gather/blend of every field
        t0 = time.perf_counter()
        sampler.sample_at(pts)
        pd = sampler.sampled_could.point_data
        for f in fields:
            np.asarray(pd[f], float)
        return time.perf_counter() - t0

    N, t_b, t_f = [], [], []
    for n in N_SWEEP:
        pts = rand_pts(n)
        tb = float(np.median([t_bracket(pts) for _ in range(_reps(n))]))
        tf = float(np.median([t_full(pts) for _ in range(_reps(n))]))
        N.append(n); t_b.append(tb); t_f.append(tf)
        print(f"N={n:>9d}  bracket={tb*1e3:8.3f} ms   full={tf*1e3:9.3f} ms   "
              f"{n/tf/1e6:7.2f} Mpts/s   {tf/n*1e6:7.3f} us/pt")

    N = np.array(N, float); t_b = np.array(t_b); t_f = np.array(t_f)
    m = N >= FIT_MIN_N
    slope, intercept = np.polyfit(np.log(N[m]), np.log(t_f[m]), 1)
    thr = np.median(N[m] / t_f[m]) / 1e6
    print(f"\nfull-eval log-log slope (N>={FIT_MIN_N}): {slope:.3f}  (1.0 = linear O(N))")
    print(f"full-eval throughput: {thr:.1f} Mpts/s  ({1e3/thr:.4f} ns/point, all {len(fields)} fields)")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax[0].loglog(N, t_f * 1e3, "o-", color="#1f77b4", label=f"full eval ({len(fields)} fields)")
    ax[0].loglog(N, t_b * 1e3, "^-", color="#7f7f7f", alpha=.7, label="sample_at only (bracketing)")
    ax[0].loglog(N[m], np.exp(intercept) * N[m] ** slope * 1e3, "--", color="#d62728",
                 label=f"fit  t $\\propto$ N$^{{{slope:.2f}}}$")
    ax[0].set_xlabel("number of points N"); ax[0].set_ylabel("time [ms]")
    ax[0].set_title("phz OBL evaluation time vs N")
    ax[0].grid(True, which="both", alpha=.3); ax[0].legend()

    ax[1].semilogx(N, N / t_f / 1e6, "s-", color="#2ca02c")
    ax[1].set_xlabel("number of points N"); ax[1].set_ylabel("throughput [Mpoints/s]")
    ax[1].set_title("full-eval throughput (flat = linear scaling)")
    ax[1].grid(True, which="both", alpha=.3)
    for nc, lab in SOLVER_NCELLS:
        for a in ax:
            a.axvline(nc, color="grey", ls=":", lw=.8)
        ax[0].annotate(lab, (nc, ax[0].get_ylim()[0] * 1.5), rotation=90,
                       va="bottom", fontsize=8, color="grey")

    fig.suptitle("Scalability of the phz OBL (VTKSampler tensor backend) — random points in (z,h,p)",
                 y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"obl_scaling.{ext}")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print("saved:", out)


if __name__ == "__main__":
    main()
