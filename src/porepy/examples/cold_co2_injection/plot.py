"""Scipts to obtain plots presented in article."""

from __future__ import annotations

import json
from typing import Literal, TypeAlias, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from porepy.examples.cold_co2_injection.run import (
    FLASH_TOLERANCES,
    FOLDER,
    LOCAL_STRIDES,
    MESH_SIZES,
    get_path,
)

EquilibriumCondition: TypeAlias = Literal["unified-p-T", "unified-p-h"]

PRINT_STATS_ph_pT_REFINEMENT: int = 2
"""Last refinement level for which both equilibrium conditions converged."""

DPI: int = 400
FIGUREWIDTH: int = 20
FIGUREHEIGHT: int = 0.4 * FIGUREWIDTH
FIGUREPAD: float = 0.05
FONTSIZE: int = 22
MARKERSIZE: int = 18
LINEWIDTH: float = 3
FIGUREPATH = FOLDER
NUM_REFINEMENTS: int = len(MESH_SIZES)

pT_colors = {0: "yellow", 1: "gold", 2: "orange", 3: "red", 4: "darkred"}

ph_colors = {
    0: "aquamarine",
    1: "aqua",
    2: "deepskyblue",
    3: "dodgerblue",
    4: "blue",
}


plt.rc("text", usetex=True)
plt.rcParams["font.size"] = FONTSIZE
# mypy: ignore-errors


class SimulationData(TypedDict, total=False):
    simulation_success: bool
    """Flag whether the simulation finished successfully or not."""
    equilibrium_condtion: EquilibriumCondition
    """Used equilibrium condition."""
    refinement_level: int
    """Refinement level of flash."""
    tol_flash_case: int
    """Tolerance case for equilibrium problem."""
    local_stride: int
    """Every n-th global iteration, the local solver is applied."""
    num_cells: int
    """Number of cells corresponding to h."""
    t: np.ndarray
    """1D array of times."""
    dt: np.ndarray
    """1D array of time step sizes."""
    recomputations: np.ndarray
    """Number of recomputations of dt at a time due to convergence failure."""
    num_global_iter: np.ndarray
    """Number of global iterations per time step."""
    num_flash_iter: np.ndarray
    """Number of cell-averaged flash iterations per time step
    (summed over global iterations)."""
    num_linesearch_iter: np.ndarray
    """Number of line search iterations per time step (summed over global
    iterations)."""

    clock_time_global_solver: tuple[float, float]
    """Clock time for for global solver, average and total."""
    clock_time_assembly: tuple[float, float]
    """Clock time for assembly, average and total."""
    clock_time_flash_solver: tuple[float, float]
    """Clock time for flash solver, average and total."""
    setup_time: float
    """"Clock time for simulation setup."""

    total_num_time_steps: int
    """Total number of time steps in the simulation."""
    total_num_global_iter: int
    """Total number of global iterations in the simulation."""
    total_num_flash_iter: int
    """Total number of flash iterations in the simulation."""


# region Loading simulation data


def load_data(
    condition: EquilibriumCondition,
    refinement: int,
    flash_tol_case: int = 2,
    flash_stride: int | None = 3,
    rel_perm: Literal["quadratic", "linear"] = "linear",
    num_months: int = 20,
) -> SimulationData:
    path = get_path(
        condition=condition,
        refinement=refinement,
        flash_tol_case=flash_tol_case,
        flash_stride=flash_stride,
        rel_perm=rel_perm,
        num_months=num_months,
        file_name=None,
    ).resolve()
    if not path.is_file():
        raise ValueError(
            "Simulation data not found for\n"
            f"equilibrium condition: {condition}\n"
            f"refinement level: {refinement}\n"
            f"flash tolerance case: {flash_tol_case}\n"
            f"at location: {str(path.resolve())}"
        )

    data = json.load(path.open("r"))
    return SimulationData(
        simulation_success=bool(data["simulation_success"]),
        refinement_level=int(data["refinement_level"]),
        equilibrium_condtion=str(data["equilibrium_condition"]),
        tol_flash_case=int(data["tol_flash_case"]),
        local_stride=int(data["local_stride"]),
        num_cells=int(data["num_cells"]),
        t=np.array(data["t"]).astype(float),
        dt=np.array(data["dt"]).astype(float),
        recomputations=np.array(data["recomputations"]).astype(int),
        num_global_iter=np.array(data["num_global_iter"]).astype(int),
        num_linesearch_iter=np.array(data["num_linesearch_iter"]).astype(int),
        num_flash_iter=np.array(data["num_flash_iter"]).astype(float),
        clock_time_global_solver=tuple(data["clock_time_global_solver"]),
        clock_time_assembly=tuple(data["clock_time_assembly"]),
        clock_time_flash_solver=tuple(data["clock_time_flash_solver"]),
        total_num_global_iter=int(data["total_num_global_iter"]),
        total_num_time_steps=int(data["total_num_time_steps"]),
        total_num_flash_iter=int(data["total_num_flash_iter"]),
        setup_time=float(data["setup_time"]),
    )


# endregion

# region Plotting total num iterations per flash tolerance and stride.

strides = np.array(LOCAL_STRIDES).astype(int)
ftols = np.array(list(FLASH_TOLERANCES.values())).astype(float)

DD: list[SimulationData] = [
    load_data(
        condition="unified-p-h",
        refinement=0,
        flash_tol_case=i,
        flash_stride=3,
        num_months=6,
    )
    for i in FLASH_TOLERANCES.keys()
]
DDS: list[SimulationData] = [
    load_data(
        condition="unified-p-h",
        refinement=0,
        flash_tol_case=7,
        flash_stride=i,
        num_months=6,
    )
    for i in [None] + LOCAL_STRIDES[1:]
]
tngi = np.array([d["total_num_global_iter"] for d in DD]).astype(int)
tnfi = np.array([d["total_num_flash_iter"] for d in DD]).astype(int)
tngis = np.array([d["total_num_global_iter"] for d in DDS]).astype(int)
tnfis = np.array([d["total_num_flash_iter"] for d in DDS]).astype(int)

success_tol = np.array([d["simulation_success"] for d in DD]).astype(bool)
markevery_tol = np.where(success_tol)[0].tolist()
success_strides = np.array([d["simulation_success"] for d in DDS]).astype(bool)
markevery_strides = np.where(success_strides)[0].tolist()

fig = plt.figure(figsize=(FIGUREWIDTH, 0.6 * FIGUREHEIGHT))
ax = fig.add_subplot(1, 2, 2)
imgs = []
imgsr = []

imgs += ax.plot(
    ftols,
    tngi,
    color="black",
    marker="^",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="global",
    markevery=markevery_tol,
)
if np.any(~success_tol):
    print("local tolerance failures", ftols[~success_tol])
    ax.plot(
        ftols[~success_tol],
        tngi[~success_tol],
        color="red",
        marker="X",
        markersize=MARKERSIZE + 3,
        linestyle="",
        label="tol failure",
    )

axr = ax.twinx()
ax.set_zorder(axr.get_zorder() + 1)
ax.patch.set_visible(False)
color = "salmon"
imgsr += axr.plot(
    ftols,
    tnfi,
    color=color,
    marker="s",
    mfc="white",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="local",
)

ax.set_title("Iteration stride = 3", fontsize=FONTSIZE + 2)
ax.set_xlabel("Local tolerance", fontsize=FONTSIZE + 2)
ax.set_xticks(ftols)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xscale("log")
ax.xaxis.grid(visible=True, which="major", color="grey", alpha=0.3, linewidth=0.5)
ax.tick_params(axis="both", which="both", labelcolor="black", labelsize=FONTSIZE)
ax.set_yscale("log")
ax.yaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)

mav = tngi.max()
miv = tngi.min()
ticks = ax.get_yticks()
ticks = ticks[ticks < mav - 200]
ticks = ticks[ticks > miv]
ticks = np.concatenate([ticks, np.array([miv, mav])]).astype(int)
ax.set_yticks(ticks)
ticks = ax.get_yticks(minor=True)
ticks = ticks[ticks < mav]
ticks = ticks[ticks > miv]
ax.set_yticks(ticks, minor=True)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())

axr.set_ylabel("Total local iterations", color=color, fontsize=FONTSIZE + 2)
axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# axr.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ticks = axr.get_yticks()
miv = tnfi.min()
mav = tnfi.max()
ticks = ticks[ticks < mav]
ticks = ticks[ticks > miv]
ticks = np.concatenate([ticks, np.array([miv, mav])]).astype(int)
axr.set_yticks(ticks)
axr.yaxis.grid(visible=True, which="both", color="salmon", alpha=0.3, linewidth=0.5)

ax.margins(0.10)
axr.margins(0.10)


ax = fig.add_subplot(1, 2, 1)
imgs = []
imgsr = []

imgs += ax.plot(
    strides,
    tngis,
    color="black",
    marker="^",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="global",
    markevery=markevery_strides,
)
if np.any(~success_strides):
    print("stride failures", strides[~success_strides])
    ax.plot(
        strides[~success_strides],
        tngis[~success_strides],
        color="red",
        marker="X",
        markersize=MARKERSIZE + 3,
        linestyle="",
        label="stride failure",
    )

axr = ax.twinx()
ax.set_zorder(axr.get_zorder() + 1)
ax.patch.set_visible(False)
color = "salmon"
imgsr += axr.plot(
    strides,
    tnfis,
    color=color,
    marker="s",
    mfc="white",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="local",
)

ax.set_title(f"Local solver tolerance = 1e-8", fontsize=FONTSIZE + 2)
ax.set_xlabel("Iteration stride", fontsize=FONTSIZE + 2)
ax.set_xticks(strides, [str(None)] + LOCAL_STRIDES[1:])
ax.set_ylabel("Total global iterations", color="black", fontsize=FONTSIZE + 2)
ax.tick_params(axis="both", which="both", labelcolor="black", labelsize=FONTSIZE)
ax.xaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)
ax.set_yscale("log")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)
ticks = ax.get_yticks()
ticks = np.concatenate([ticks, np.array([tngis.max(), tngis.min()]).astype(int)])
ax.set_yticks(ticks)

axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ticks = axr.get_yticks()
miv = tnfis.min()
mav = tnfis.max()
ticks = ticks[ticks < mav]
ticks = ticks[ticks > miv]
ticks = np.concatenate([ticks, np.array([miv, mav])]).astype(int)
axr.set_yticks(ticks)
axr.yaxis.grid(visible=True, which="both", color="salmon", alpha=0.3, linewidth=0.5)

ax.margins(0.1)
axr.margins(0.1)

fig.tight_layout(pad=FIGUREPAD)
name = f"{FIGUREPATH}total_iter_per_ftol_and_stride.png"
fig.savefig(
    name,
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)
print(f"\nSaved fig: {name}")

# endregion

# region Plotting total num iterations per grid refinement.

data: dict[EquilibriumCondition, dict[int, SimulationData]] = {
    "unified-p-T": dict(
        [
            (
                i,
                load_data(
                    condition="unified-p-T",
                    refinement=i,
                    flash_tol_case=2,
                    flash_stride=3,
                    rel_perm="linear",
                    num_months=20,
                ),
            )
            for i in range(len(MESH_SIZES))
        ]
    ),
    "unified-p-h": dict(
        [
            (
                i,
                load_data(
                    condition="unified-p-h",
                    refinement=i,
                    flash_tol_case=2,
                    flash_stride=3,
                    rel_perm="linear",
                    num_months=20,
                ),
            )
            for i in range(len(MESH_SIZES))
        ]
    ),
}

fig = plt.figure(figsize=(0.65 * FIGUREWIDTH, 0.7 * FIGUREHEIGHT))
ax = fig.add_subplot(1, 1, 1)
axr = ax.twinx()
ax.set_zorder(axr.get_zorder() + 1)
ax.patch.set_visible(False)
imgs = []
imgsr = []

ngi = []
nfi = []
nli = []
success = []
color = "salmon"
for i in MESH_SIZES.keys():
    if i in data["unified-p-T"]:
        D = data["unified-p-T"][i]
        assert i == D["refinement_level"]
        success.append(D["simulation_success"])
        ngi.append([MESH_SIZES[i], D["total_num_global_iter"]])
        nfi.append([MESH_SIZES[i], D["total_num_flash_iter"]])
        nli.append([MESH_SIZES[i], D["num_linesearch_iter"].sum()])
success = np.array(success).astype(bool)
markevery_tol = np.where(success)[0].tolist()
ngi = np.array(ngi).T
imgs += ax.plot(
    ngi[0][success],
    ngi[1][success],
    color=color,
    linestyle="solid",
    marker="^",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="pT-global",
    markevery=markevery_tol,
)
nfi = np.array(nfi).T
imgsr += axr.plot(
    nfi[0][success],
    nfi[1][success],
    color=color,
    linestyle="dashed",
    marker="s",
    mfc="white",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="pT-local",
)
nli = np.array(nli).T
imgs += ax.plot(
    nli[0][success],
    nli[1][success],
    color=color,
    linestyle="dotted",
    marker="P",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="pT-linesearch",
)
M = int(np.hstack((ngi[1], nli[1])).max())

if np.any(~success) and False:
    print("pT failures", ngi[0][~success])
    ax.plot(
        np.concatenate([ngi[0][~success], nli[0][~success]]),
        np.concatenate([ngi[1][~success], nli[1][~success]]),
        color="red",
        marker="X",
        markersize=MARKERSIZE + 3,
        linestyle="",
        label="pT-failure",
    )
    axr.plot(
        nfi[0][~success],
        nfi[1][~success],
        color="red",
        marker="X",
        markersize=MARKERSIZE + 3,
        linestyle="",
        label="pT-failure",
    )

ngi = []
nfi = []
nli = []
success = []
color = "slateblue"
for i in MESH_SIZES.keys():
    if i in data["unified-p-h"]:
        D = data["unified-p-h"][i]
        assert i == D["refinement_level"]
        success.append(D["simulation_success"])
        ngi.append([MESH_SIZES[i], D["total_num_global_iter"]])
        nfi.append([MESH_SIZES[i], D["total_num_flash_iter"]])
        nli.append([MESH_SIZES[i], D["num_linesearch_iter"].sum()])
success = np.array(success).astype(bool)
markevery_tol = np.where(success)[0].tolist()
ngi = np.array(ngi).T
imgs += ax.plot(
    ngi[0],
    ngi[1],
    color=color,
    linestyle="solid",
    marker="^",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="ph-global",
    markevery=markevery_tol,
)
nfi = np.array(nfi).T
imgsr += axr.plot(
    nfi[0],
    nfi[1],
    color=color,
    linestyle="dashed",
    marker="s",
    mfc="white",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="ph-local",
)
nli = np.array(nli).T
imgs += ax.plot(
    nli[0],
    nli[1],
    color=color,
    linestyle="dotted",
    marker="P",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="ph-linesearch",
)
m = int(np.hstack((ngi[1], nli[1])).max())
if m > M:
    M = m

if np.any(~success):
    print("ph failures", ngi[0][~success])
    ax.plot(
        np.concatenate([ngi[0][~success], nli[0][~success]]),
        np.concatenate([ngi[1][~success], nli[1][~success]]),
        color="red",
        marker="X",
        markersize=MARKERSIZE + 3,
        linestyle="",
        label="ph-failure",
    )
    axr.plot(
        nfi[0][~success],
        nfi[1][~success],
        color="red",
        marker="X",
        markersize=MARKERSIZE + 3,
        linestyle="",
        label="ph-failure",
    )

ax.set_xlabel("Mesh size [m]", fontsize=FONTSIZE + 2)
ax.set_xscale("log")
xticks = list(MESH_SIZES.values())[::-1]
ax.set_xticks(xticks, [str(i) for i in xticks])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.xaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)
ax.set_ylabel("Total global iterations", fontsize=FONTSIZE + 2)
ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
ax.yaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)
axr.get_yaxis().set_major_formatter(
    matplotlib.ticker.ScalarFormatter(useOffset=False, useMathText=True)
)
axr.set_ylabel("Total local iterations", fontsize=FONTSIZE + 2)
axr.tick_params(axis="y", which="both", labelcolor="salmon", labelsize=FONTSIZE)
axr.yaxis.grid(visible=True, which="both", color="salmon", alpha=0.3, linewidth=0.5)

ticks = ax.get_yticks()
ticks = ticks[ticks < M - 90]
ticks = np.concatenate([ticks, np.array([M])]).astype(int)
ax.set_yticks(ticks)

ax.margins(0.05)
axr.margins(0.05)

ax.legend(
    handles=imgs,
    fontsize=FONTSIZE,
    loc="upper left",
    bbox_to_anchor=(1.15, 1),
)
axr.legend(
    handles=imgsr,
    fontsize=FONTSIZE,
    loc="upper left",
    bbox_to_anchor=(1.15, 0.45),
)
fig.tight_layout(pad=FIGUREPAD)
name = f"{FIGUREPATH}total_iter_per_refinement.png"
fig.savefig(
    name,
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)
print(f"\nSaved fig: {name}")

# endregion

# region Plotting num iterations and re-computations per time step, and time progress
# for plot case

D = load_data(
    condition="unified-p-h",
    refinement=3,
    flash_tol_case=2,
    flash_stride=3,
    rel_perm="linear",
    num_months=20,
)

t = np.array(D["t"]) / (3600 * 24)
dt = np.array(D["dt"]) / (3600 * 24)
t_indices = np.arange(t.size).astype(int)
ngi = np.array(D["num_global_iter"]).astype(int)
nfi = np.array(D["num_flash_iter"]).astype(float)
nli = np.array(D["num_linesearch_iter"]).astype(int)

fig = plt.figure(figsize=(0.55 * FIGUREWIDTH, 0.6 * FIGUREHEIGHT))
ax = fig.add_subplot(1, 1, 1)
imgs = []
imgsr = []

imgs += ax.plot(
    t,
    ngi,
    color="black",
    linestyle="solid",
    marker="^",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="global",
)
imgs += ax.plot(
    t,
    nli,
    color="black",
    linestyle="dotted",
    marker="P",
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
    marker="s",
    mfc="white",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="local",
)

rcomps = np.array(D["recomputations"]).astype(int)
assert np.all(rcomps >= 0)
rcid = rcomps > 0
if np.any(rcid):
    tid = t[rcid]
    ngiid = ngi[rcid] + 2
    rid = rcomps[rcid]
    n = 3e2
    N = 1e3
    m = rid.min()
    M = rid.max()
    a = (N - n) / (M - m)
    b = n - a * m
    sizes = a * rid + b

    mav = np.max([ngi.max(), nli.max()])
    ypos = float(1.1 * mav)
    imgs += [
        ax.scatter(
            tid,
            np.ones_like(tid).astype(int) * ypos,
            s=sizes,
            alpha=0.5,
            label="halving",
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

ax.set_xscale("symlog", linthresh=1)
ax.xaxis.grid(visible=True, which="major", color="grey", alpha=0.3, linewidth=0.5)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Time [d]", fontsize=FONTSIZE + 2)
ticks = np.concatenate((ax.get_xticks(), np.array([t.max()])))
ax.set_xticks(ticks)
ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator(base=10))
ax.set_ylabel("Global iterations", fontsize=FONTSIZE + 2)
ax.tick_params(axis="both", which="both", labelsize=FONTSIZE)
axr.set_ylabel("Cell-averaged local iterations", color=color, fontsize=FONTSIZE + 2)
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)

mav = np.max((nli.max(), ngi.max()))
miv = np.min((nli.min(), ngi.min()))
ticks = ax.get_yticks()
ticks = ticks[ticks > miv]
ticks = ticks[ticks < mav]
ticks = np.concatenate([ticks, np.array([miv, mav])])
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
name = f"{FIGUREPATH}iterations_per_time_ph.png"
fig.savefig(
    name,
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)
print(f"\nSaved fig: {name}")

fig = plt.figure(figsize=(0.9 * FIGUREHEIGHT, 0.6 * FIGUREHEIGHT))
ax = fig.add_subplot(1, 1, 1)
imgs = []
imgsr = []

t = np.array(D["t"]) / (3600 * 24)


imgs += ax.plot(
    t_indices,
    t,
    color="black",
    marker="^",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label="t",
)

axr = ax.twinx()
ax.set_zorder(axr.get_zorder() + 1)
ax.patch.set_visible(False)
color = "salmon"
imgsr += axr.plot(
    t_indices,
    dt,
    color=color,
    marker="s",
    mfc="white",
    markersize=int(MARKERSIZE / 2),
    linewidth=LINEWIDTH,
    label=r"$\Delta$ t",
)

ax.set_xlabel("Time step index", fontsize=FONTSIZE + 2)
ax.set_ylabel("Time [d]", color="black", fontsize=FONTSIZE + 2)
ax.xaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)
ax.tick_params(axis="both", which="both", labelcolor="black", labelsize=FONTSIZE)
ax.set_yscale("symlog", linthresh=1)
mav = t_indices.max()
ticks = ax.get_xticks()
ticks = ticks[ticks < mav-3]
ticks = np.concatenate((ticks, np.array([mav]))).astype(int)
ax.set_xticks(ticks)
ticks = ax.get_yticks()
ticks = np.concatenate([ticks, np.array([20 * 30]).astype(int)])
ax.set_yticks(ticks)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.grid(visible=True, which="major", color="grey", alpha=0.3, linewidth=0.5)
axr.set_ylabel("Time step size [d]", color=color, fontsize=FONTSIZE + 2)
axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)
axr.set_yscale("symlog", linthresh=1)
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.yaxis.grid(visible=True, which="major", color=color, alpha=0.3, linewidth=0.5)
lim = axr.get_ylim()
mav = dt.max()
miv = dt.min()
ticks = axr.get_yticks()
ticks = ticks[ticks < mav]
ticks = ticks[ticks > miv]
ticks = np.concatenate([ticks, np.array([miv, mav])])
axr.set_yticks(ticks)
ticks = axr.get_yticks(minor=True)
ticks = ticks[ticks < mav]
ticks = ticks[ticks > miv]
axr.set_yticks(ticks, minor=True)

ax.margins(0.1)
axr.margins(0.1)

fig.tight_layout(pad=FIGUREPAD)
name = f"{FIGUREPATH}time_progress_ph.png"
fig.savefig(
    name,
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)
print(f"\nSaved fig: {name}")
# endregion

# region Printing table with clock times, at maximum refinement where both equilibrium
# conditions converge.


def format_times(vals: tuple[float, float]) -> str:
    """Format the clock time values."""
    return f"{vals[0]:.4f} ({vals[1]:.2f})"


headers = [""]

rows = {
    "Assembly time": [],
    "Linear solver time": [],
    "Flash solver time": [],
    "Number of time steps": [],
    "Number of global iterations": [],
    "Total number of local iterations": [],
}

for i, hmesh in MESH_SIZES.items():
    headers.append(f"ph({hmesh})")
    headers.append(f"pT({hmesh})")

    dph = data["unified-p-h"].get(i)
    dpt = data["unified-p-T"].get(i)

    # NOTE: The time step inforrmation also contains the start time of T=0, which we
    # must account for with -0 to get the actual computations

    ngipt = int(dpt["num_global_iter"].sum())
    ntpt = int(dpt["t"].size) - 1
    nfipt = int(dpt["num_flash_iter"].sum())
    ngiph = int(dph["num_global_iter"].sum())
    ntph = int(dph["t"].size) - 1
    nfiph = int(dph["num_flash_iter"].sum())

    if dph is not None:
        rows["Assembly time"].append(format_times(dph["clock_time_assembly"]))
        rows["Linear solver time"].append(format_times(dph["clock_time_global_solver"]))
        rows["Flash solver time"].append(format_times(dph["clock_time_flash_solver"]))
        rows["Number of time steps"].append(
            f"{ntph} ({dph['total_num_time_steps'] - 1 - ntph})"
        )
        rows["Number of global iterations"].append(
            f"{ngiph} ({dph['total_num_global_iter'] - ngiph})"
        )
        rows["Total number of local iterations"].append(
            f"{dph['total_num_flash_iter']}"
        )
    else:
        for k in rows:
            rows[k].append("-")
    if dpt is not None:
        rows["Assembly time"].append(format_times(dpt["clock_time_assembly"]))
        rows["Linear solver time"].append(format_times(dpt["clock_time_global_solver"]))
        rows["Flash solver time"].append(format_times(dpt["clock_time_flash_solver"]))
        rows["Number of time steps"].append(
            f"{ntpt} ({dpt['total_num_time_steps'] - 1 - ntpt})"
        )
        rows["Number of global iterations"].append(
            f"{ngipt} ({dpt['total_num_global_iter'] - ngipt})"
        )
        rows["Total number of local iterations"].append(
            f"{dpt['total_num_flash_iter']}"
        )
    else:
        for k in rows:
            rows[k].append("-")

table = tabulate(
    [[k] + v for k, v in rows.items()],
    headers=headers,
    tablefmt="orgtbl",
)

print(table)

# endregion
