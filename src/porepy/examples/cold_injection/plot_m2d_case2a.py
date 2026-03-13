"""Plotting script for cold_injection/ m2d case2a."""

import json
import pathlib
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import porepy as pp

timestamp = "13March2026_15-24-31_BUOY_False_AJUMP_False"
timestamp = "13March2026_20-33-40_BUOY_False_AJUMP_True"
folder = f"visualization/{timestamp}/"
file = f"{folder}solver_statistics.json"

DPI: int = 400
FIGSIZE: int = 10  # Base size
FIGUREPAD: float = 0.05
FONTSIZE: int = 22
MARKERSIZE: int = 18
LINEWIDTH: float = 3


path = pathlib.Path(file).resolve()
data: dict

if path.is_file():
    data = json.load(path.open("r"))
else:
    raise ValueError(f"Simulation data not found: {str(path)}")


globaldata = data["global"]
nc = int(sum(globaldata["num_cells"].values()))
ne = int(globaldata["num_entries"])
success = bool(globaldata["final_time_reached"])
total_num_iter = globaldata["total_num_iterations"]
total_num_time_steps = globaldata["total_num_time_steps"]

num_glob_iter = []
num_flash_iter = []
num_ls_iter = []

assembly_ct = []
linsolve_ct = []
flash_ct = []

time_in_days = []
dt_in_days = []
recomputations_per_timestep = []

rc: int = 0  # recomputation counter
for i in range(ne):
    locdata = data[str(i)]

    if locdata["simulation_status"] == "successful":
        num_glob_iter.append(int(locdata["num_iterations"]))
        num_flash_iter.append(int(locdata["flash_iterations"]))
        num_ls_iter.append(int(sum(locdata["armijo_iterations"])))
        assembly_ct.append(float(np.array(locdata["assembly_clocktime"]).sum()))
        linsolve_ct.append(float(np.array(locdata["linsolve_clocktime"]).sum()))
        flash_ct.append(float(np.array(locdata["flash_clocktime"]).sum()))
        time_in_days.append(float(locdata["time"] / pp.DAY))
        dt_in_days.append(float(locdata["dt"]))
        recomputations_per_timestep.append(rc)
        rc = 0
    else:
        rc += 1


ngi = np.array(num_glob_iter).astype(int)
nfi = np.array(num_flash_iter).astype(int)
nlsi = np.array(num_ls_iter).astype(int)
t = np.array(time_in_days)
dt = np.array(dt_in_days)
recomputations = np.array(recomputations_per_timestep).astype(int)

act = np.array(assembly_ct)
lsct = np.array(linsolve_ct)
fct = np.array(flash_ct)


# region Plotting number of iterations per time step

fig = plt.figure(figsize=(2 * FIGSIZE, FIGSIZE))
ax = fig.add_subplot(1, 1, 1)
imgs: list[Any] = []
imgsr: list[Any] = []

imgs += ax.plot(
    t,
    ngi,
    color="black",
    linestyle="solid",
    # marker="^",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="glob. Newton",
)
imgs += ax.plot(
    t,
    nlsi,
    color="black",
    linestyle="dotted",
    # marker="P",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="line search",
)

color = "salmon"
axr = ax.twinx()
ax.set_zorder(axr.get_zorder() + 1)
ax.patch.set_visible(False)
imgsr += axr.plot(
    t,
    nfi,
    color=color,
    linestyle="dashed",
    # marker="s",
    mfc="white",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="loc. flash",
)

rcid = recomputations > 0
if recomputations.size > 0:
    tid = t[rcid]
    ngiid = ngi[rcid] + 2
    rid = recomputations[rcid]
    n = 3e2
    N = 1e3
    m = rid.min()
    M = rid.max()
    a = (N - n) / max(M - m, 1)
    b = n - a * m
    sizes = a * rid + b

    mav = np.max([ngi.max(), nlsi.max()])
    ypos = float(1.1 * mav)
    imgs += [
        ax.scatter(
            tid,
            np.ones_like(tid).astype(int) * ypos,
            s=sizes,
            alpha=0.5,
            label="recomputations",
        )
    ]

    idx = rid == M

    ax.text(
        tid[idx][0],
        ypos,
        M,
        fontsize=FONTSIZE + 2,
        fontweight="heavy",
        horizontalalignment="center",
        verticalalignment="center",
    )

ax.xaxis.grid(visible=True, which="major", color="grey", alpha=0.3, linewidth=0.5)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Simulation Time [d]", fontsize=FONTSIZE + 2)
tmax = t.max()
ticks = ax.get_xticks()
ticks = np.concatenate((ticks[ticks < tmax - 10], np.array([tmax])))
ax.set_xticks(ticks)
ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator(base=10))
ax.set_ylabel("Global iterations", fontsize=FONTSIZE + 2)
ax.tick_params(axis="both", which="both", labelsize=FONTSIZE)
axr.set_ylabel("Local iterations", color=color, fontsize=FONTSIZE + 2)
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)

mav = np.max((nlsi.max(), ngi.max()))
miv = np.min((nlsi.min(), ngi.min()))
ticks = ax.get_yticks()
ticks = ticks[ticks > miv]
ticks = ticks[ticks < mav]
ticks = np.concatenate([ticks, np.array([miv, mav, int(np.ceil(ngi.mean()))])])
ax.set_yticks(ticks)

miv = nfi.min()
mav = nfi.max()
ticks = axr.get_yticks()
ticks = ticks[ticks > miv]
ticks = ticks[ticks < mav]
ticks = np.concatenate([ticks, np.array([miv, mav])])
axr.set_yticks(ticks)

ax.yaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)
axr.yaxis.grid(visible=True, which="both", color=color, alpha=0.3, linewidth=0.5)

ax.margins(0.05)
axr.margins(0.05)
axr.set_yscale("symlog")

ax.legend(
    [i.get_label() for i in imgs],
    fontsize=FONTSIZE,
    loc="upper left",
    bbox_to_anchor=(1.15, 1),
)
axr.legend(
    [i.get_label() for i in imgsr],
    fontsize=FONTSIZE,
    loc="upper left",
    bbox_to_anchor=(1.15, 0.6),
)
fig.tight_layout(pad=FIGUREPAD)
name = f"{folder}iterations_per_time_ph.png"
fig.savefig(
    name,
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)
# endregion
# region Plot time spent in algorithm.

fig = plt.figure(figsize=(2 * FIGSIZE, FIGSIZE))
ax = fig.add_subplot(1, 1, 1)
imgs = []

imgs += ax.plot(
    t,
    lsct,
    color="black",
    linestyle="solid",
    # marker="^",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="lin-solve",
)
imgs += ax.plot(
    t,
    act,
    color="black",
    linestyle="dotted",
    # marker="P",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="assembly",
)
imgs += ax.plot(
    t,
    fct,
    color="salmon",
    linestyle="solid",
    # marker="P",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="flash",
)

ax.xaxis.grid(visible=True, which="major", color="grey", alpha=0.3, linewidth=0.5)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Simulation Time [d]", fontsize=FONTSIZE + 2)
tmax = t.max()
ticks = ax.get_xticks()
ticks = np.concatenate((ticks[ticks < tmax - 10], np.array([tmax])))
ax.set_xticks(ticks)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylabel("Clock time [s]", fontsize=FONTSIZE + 2)
ax.tick_params(axis="both", which="both", labelsize=FONTSIZE)

ax.yaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)

ax.margins(0.05)

ax.legend(
    [i.get_label() for i in imgs],
    fontsize=FONTSIZE,
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
)
fig.tight_layout(pad=FIGUREPAD)
name = f"{folder}clocktimes.png"
fig.savefig(
    name,
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)
# endregion
