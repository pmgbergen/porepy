"""Resolution sweep for the independent 1D solver -> 12 comparison images.

Runs N in {100,200,400,800,1600,3200} for BOTH cases (fig 5D vertical / fig 5B horizontal)
and BOTH buoyancy schemes (HU, PPU): 24 simulations executed in parallel across CPU cores,
then one HU-vs-PPU-vs-reference comparison image per (case, N) -> 12 PNGs in
visualization_1D_fig_5/  (fig5D_compare_N*.png and fig5B_compare_N*.png).
"""
import os
import sys
import time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geothermal_H2O_low_NaCl_content_1D_fig_5 as m   # noqa: E402

NS = [100, 200, 400, 800, 1600, 3200]
CASES = ["vertical", "horizontal"]
SCHEMES = ["hu", "ppu", "sppu"]
PANEL = {"vertical": "5D", "horizontal": "5B"}
OUT = os.path.join(m.HERE, "visualization_1D_fig_5")


def worker(task):
    scheme, N, case = task
    t0 = time.time()
    res = m.run(scheme=scheme, N=N, case=case, verbose=False)
    res["_wall"] = time.time() - t0
    return task, res


def main():
    os.makedirs(OUT, exist_ok=True)
    # build the table caches serially first so the workers all hit the .npz fast path
    m.Table(m.VTK_XPH, m._XPH_FIELDS, a_in=1e-6, b_in=1e-6)
    m.Table(m.VTK_XPT, {"H": 1e3}, a_in=1.0, b_in=1e-6)

    tasks = [(s, N, c) for c in CASES for N in NS for s in SCHEMES]
    tasks.sort(key=lambda t: -t[1])                       # largest N first -> load balance
    nproc = min(len(tasks), max(1, (os.cpu_count() or 4) - 1))
    print(f"[sweep] {len(tasks)} sims on {nproc} procs", flush=True)

    res = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(nproc) as pool:
        for task, r in pool.imap_unordered(worker, tasks):
            res[task] = r
            s, N, c = task
            print(f"[sweep] {c:10s} N={N:5d} {s:3s}: {r['_wall']:5.0f}s  "
                  f"avg_it={r['avg_it']:.2f}  total_it={r['total_it']}", flush=True)

    for c in CASES:
        for N in NS:
            pair = {s: res[(s, N, c)] for s in SCHEMES}
            path = os.path.join(OUT, f"fig{PANEL[c]}_compare_N{N}.png")
            m.plot_comparison(pair, path, case=c)
    print(f"[sweep] ALL DONE in {time.time() - t0:.0f}s -> 12 images in {OUT}", flush=True)


if __name__ == "__main__":
    main()
