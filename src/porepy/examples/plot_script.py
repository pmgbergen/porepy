"""Scipts to obtain plots presented in article."""

from __future__ import annotations

import json
import pathlib
from typing import Literal, TypeAlias, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

RefinementLevel: TypeAlias = Literal[0, 1, 2, 3, 4]
EquilibriumCondition: TypeAlias = Literal["p-T", "p-h"]


FIGUREPATH: str = f"{str(pathlib.Path(__file__).parent.resolve())}\\"
DPI: int = 400
FIGUREWIDTH: int = 20
FIGUREHEIGHT: int = 0.4 * FIGUREWIDTH
FIGUREPAD: float = 0.05
FONTSIZE: int = 24
MARKERSIZE: int = 20
LINEWIDTH: float = 3

USED_MESH_SIZES: dict[RefinementLevel, float] = {
    0: 4.0,
    1: 2.0,
    2: 1.0,
    3: 0.5,
    4: 0.25,
}
"""Dictionary containing the used mesh sizes for given abreviation."""

USED_FLASH_TOLERANCES: dict[int, float] = {
    0: 1e-1,
    1: 1e-2,
    2: 1e-3,
    3: 1e-5,
    4: 1e-8,
}
"""Dictionary containing the used flash tolerances for given case number."""

pT_colors = {0: "yellow", 1: "gold", 2: "orange", 3: "red", 4: "darkred"}

ph_colors = {
    0: "aquamarine",
    1: "aqua",
    2: "deepskyblue",
    3: "dodgerblue",
    4: "blue",
}


plt.rc("text", usetex=True)


class SimulationData(TypedDict, total=False):
    refinement_level: int
    """Refinement level of flash."""
    num_cells: int
    """Number of cells corresponding to h."""
    equilibrium_condtion: EquilibriumCondition
    """Used equilibrium condition."""
    tol_flash_case: int
    """Tolerance case for equilibrium problem."""
    t: np.ndarray
    """1D array of times."""
    dt: np.ndarray
    """1D array of time step sizes."""
    recomputations: list[int]
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
    refinement: RefinementLevel,
    tol_flash_case: int = 4,
) -> SimulationData | None:
    path = pathlib.Path(f"stats_{condition}_h{refinement}_ftol{tol_flash_case}.json")
    return SimulationData(
        refinement_level=refinement,
        tol_flash_case=tol_flash_case,
        num_cells=1,
        equilibrium_condtion=condition,
        tol_flash=1e-8,
        t=np.array([1.0, 2.0, 3.0]),
        dt=np.array([1.0, 1.0, 1.0]),
        recomputations=[0, 3, 1],
        num_global_iter=(
            np.array([10, 11, 12]) if condition == "p-h" else np.array([20, 21, 22])
        )
        + (refinement + 1),
        num_flash_iter=(
            np.array([5, 5, 5]) if condition == "p-h" else np.array([7, 7, 7])
        )
        + (refinement + 0.5),
        num_linesearch_iter=(
            np.array([10, 11, 12]) if condition == "p-h" else np.array([20, 21, 22])
        )
        + (refinement + 2),
        clock_time_global_solver=(1.0, 3.0),
        clock_time_assembly=(0.5, 1.5 / 60),
        clock_time_flash_solver=(0.2, 0.8 / 60),
        total_num_time_steps=3,
        total_num_global_iter=33 if condition == "p-h" else 63,
        total_num_flash_iter=5 if condition == "p-h" else 7,
        setup_time=1.0,
    )
    if not path.is_file():
        raise ValueError(
            "Simulation data not found for\n"
            f"equilibrium condition: {condition}\n"
            f"refinement level: {refinement}\n"
            f"flash tolerance case: {tol_flash_case}"
        )

    data = json.load(path)
    assert refinement == data["refinement_level"]
    assert condition == data["equilibrium_condition"]
    assert tol_flash_case == data["tol_flash_case"]
    sdata = SimulationData(
        refinement_level=data["refinement_level"],
        equilibrium_condtion=data["equilibrium_condition"],
        tol_flash_case=data["tol_flash_case"],
        num_cells=data["num_cells"],
        t=data["t"],
        dt=data["dt"],
        recomputations=data["recomputations"],
        num_global_iter=data["num_global_iter"],
        num_linesearch_iter=data["num_linesearch_iter"],
        num_flash_iter=data["num_flash_iter"],
        clock_time_global_solver=data["clock_time_global_solver"],
        clock_time_assembly=data["clock_time_assembly"],
        clock_time_flash_solver=data["clock_time_flash_solver"],
        total_num_global_iter=data["total_num_global_iter"],
        total_num_time_steps=data["total_num_time_steps"],
        total_num_flash_iter=data["total_num_flash_iter"],
        setup_time=data["setup_time"],
    )


data: dict[EquilibriumCondition, dict[RefinementLevel, SimulationData]] = {
    "p-T": dict(
        [(i, load_data("p-T", i)) for i in range(3) if load_data("p-T", i) is not None]
    ),
    "p-h": dict(
        [(i, load_data("p-h", i)) for i in range(5) if load_data("p-T", i) is not None]
    ),
}

# endregion

# region Plotting total num iterations per grid refinement.

fig = plt.figure(figsize=(FIGUREWIDTH, FIGUREHEIGHT))
ax = fig.add_subplot(1, 1, 1)
axr = ax.twinx()
imgs = []
imgsr = []

ngi = []
nfi = []
nli = []
color = "salmon"
for i in range(5):
    if i in data["p-T"]:
        D = data["p-T"][i]
        assert i == D["refinement_level"]
        ngi.append([USED_MESH_SIZES[i], D["num_global_iter"].sum()])
        nfi.append([USED_MESH_SIZES[i], D["num_flash_iter"].sum()])
        nli.append([USED_MESH_SIZES[i], D["num_linesearch_iter"].sum()])
ngi = np.array(ngi).T
imgs += ax.plot(
    ngi[0],
    ngi[1],
    color=color,
    linestyle="solid",
    marker="^",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="pT-global",
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
    label="pT-local",
)
nli = np.array(nli).T
imgs += ax.plot(
    nli[0],
    nli[1],
    color=color,
    linestyle="dotted",
    marker="X",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="pT-linesearch",
)

ngi = []
nfi = []
nli = []
color = "slateblue"
for i in range(5):
    if i in data["p-h"]:
        D = data["p-h"][i]
        assert i == D["refinement_level"]
        ngi.append([USED_MESH_SIZES[i], D["num_global_iter"].sum()])
        nfi.append([USED_MESH_SIZES[i], D["num_flash_iter"].sum()])
        nli.append([USED_MESH_SIZES[i], D["num_linesearch_iter"].sum()])
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
    marker="X",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="ph-linesearch",
)

ax.set_xlabel("Mesh size [m]", fontsize=FONTSIZE + 2)
ax.set_xscale("log")
ax.set_xticks(np.array(list(USED_MESH_SIZES.values())))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)
ax.set_ylabel("Total global iterations", fontsize=FONTSIZE + 2)
ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator(base=10))
ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
ax.tick_params(axis="both", which="minor", labelsize=FONTSIZE)
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.set_ylabel("Average total local iterations", fontsize=FONTSIZE + 2)
axr.tick_params(axis="y", which="both", labelsize=FONTSIZE)

ax.legend(
    [i.get_label() for i in imgs],
    fontsize=FONTSIZE,
    loc="upper right",
    bbox_to_anchor=(-0.1, 1),
)
axr.legend(
    [i.get_label() for i in imgsr],
    fontsize=FONTSIZE,
    loc="upper left",
    bbox_to_anchor=(1.1, 1),
)
fig.tight_layout(pad=FIGUREPAD)
fig.savefig(
    f"{FIGUREPATH}total_iter_per_refinement.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)

# endregion

# region Plotting num iterations, time step size and re-computations per time step.

D: SimulationData = data["p-h"][3]

t = np.array(D["t"])
ngi = np.array(D["num_global_iter"]).astype(int)
nfi = np.array(D["num_flash_iter"]).astype(float)
nli = np.array(D["num_linesearch_iter"]).astype(int)

fig = plt.figure(figsize=(FIGUREWIDTH, FIGUREHEIGHT))
ax = fig.add_subplot(1, 1, 1)
imgs = []
imgsr = []

color = "black"
imgs += ax.plot(
    t,
    ngi,
    color=color,
    linestyle="solid",
    marker="^",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="global",
)
imgs += ax.plot(
    t,
    nli,
    color=color,
    linestyle="dotted",
    marker="X",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="line search",
)

rcomps = np.array(D["recomputations"]).astype(int)
assert np.all(rcomps >= 0)
rcid = rcomps > 0
tid = t[rcid]
ngiid = ngi[rcid] + 2
rid = rcomps[rcid]
n = 3e2
N = 3e3
m = rid.min()
M = rid.max()
a = (N - n) / (M - m)
b = n - a * m
sizes = a * rid + b

imgs += [ax.scatter(tid, ngiid, s=sizes, alpha=0.5, label="re-computations")]

idx = rid == M

ax.text(
    tid[idx][0],
    ngiid[idx][0],
    M,
    fontsize=FONTSIZE + 2,
    fontweight="heavy",
    horizontalalignment="center",
    verticalalignment="center",
)

ax.set_xscale("log")
ax.set_xticks(np.array(list(USED_MESH_SIZES.values())))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)
ax.set_xlabel("Time [s]", fontsize=FONTSIZE + 2)
ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator())
ax.set_ylabel("Global iterations", fontsize=FONTSIZE + 2)
ax.tick_params(axis="both", which="both", labelsize=FONTSIZE)

color = "salmon"
axr = ax.twinx()
imgsr += axr.plot(
    t,
    nfi,
    color=color,
    linestyle="dashed",
    marker="s",
    mfc="white",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="local",
)
axr.set_ylabel("Average local iterations", color=color, fontsize=FONTSIZE + 2)
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)

ax.legend(
    [i.get_label() for i in imgs],
    fontsize=FONTSIZE,
    loc="upper right",
    bbox_to_anchor=(-0.1, 1),
)
axr.legend(
    [i.get_label() for i in imgsr],
    fontsize=FONTSIZE,
    loc="upper left",
    bbox_to_anchor=(1.1, 1),
)
fig.tight_layout(pad=FIGUREPAD)
fig.savefig(
    f"{FIGUREPATH}iterations_per_time_h{D['refinement_level']}_ftol{D['tol_flash_case']}.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)

# endregion

# region Plotting total num iterations per flash tolerance.

DD: list[SimulationData] = [
    load_data("p-h", 3, i) for i in USED_FLASH_TOLERANCES.keys()
]
ftols = np.array([USED_FLASH_TOLERANCES[d["tol_flash_case"]] for d in DD])
tngi = np.array([d["total_num_global_iter"] for d in DD]).astype(int)
tnfi = np.array([d["total_num_flash_iter"] for d in DD]).astype(int)

fig = plt.figure(figsize=(FIGUREWIDTH, FIGUREHEIGHT))
ax = fig.add_subplot(1, 1, 1)
imgs = []
imgsr = []

color = "black"
ax.set_xlabel("Local tolerance", fontsize=FONTSIZE + 2)
ax.set_ylabel("Total global iterations", color=color, fontsize=FONTSIZE + 2)
imgs += ax.plot(
    ftols,
    tngi,
    color=color,
    marker="^",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="global",
)
ax.tick_params(axis="both", which="both", labelcolor=color, labelsize=FONTSIZE)
ax.set_xticks(ftols)
ax.set_xscale("log")
# ax.set_yscale("log")
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator())
ax.yaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)

axr = ax.twinx()
color = "salmon"
axr.set_ylabel("Total local iterations", color=color, fontsize=FONTSIZE + 2)
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
ax.set_xticks(ftols)
axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)
# axr.set_yscale("log")
# axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.get_yaxis().set_major_locator(matplotlib.ticker.MultipleLocator())
axr.yaxis.grid(visible=True, which="both", color="orange", alpha=0.3, linewidth=0.5)

# ax.legend(
#     [i.get_label() for i in imgs],
#     fontsize=FONTSIZE,
#     loc="upper right",
#     bbox_to_anchor=(-0.1, 1),
# )
# axr.legend(
#     [i.get_label() for i in imgsr],
#     fontsize=FONTSIZE,
#     loc="upper left",
#     bbox_to_anchor=(1.1, 1),
# )
fig.tight_layout(pad=FIGUREPAD)
fig.savefig(
    f"{FIGUREPATH}total_iter_per_ftol_h{3}.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)

# endregion

# region Plotting time progress.

D: SimulationData = data["p-h"][3]
fig = plt.figure(figsize=(FIGUREWIDTH, FIGUREHEIGHT))
ax = fig.add_subplot(1, 1, 1)
imgs = []
imgsr = []

t = np.array(D["t"])
dt = np.array(D["dt"])
t_indices = np.arange(t.size)

color = "black"
ax.set_xlabel("Time step index", fontsize=FONTSIZE + 2)
ax.set_ylabel("time (s)", color=color, fontsize=FONTSIZE + 2)
imgs += ax.plot(
    t_indices,
    t,
    color=color,
    marker="^",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="t",
)
ax.tick_params(axis="both", which="both", labelcolor=color, labelsize=FONTSIZE)
ax.set_yscale("log")
ax.get_xaxis().set_major_locator(matplotlib.ticker.MultipleLocator(base=2))
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)

axr = ax.twinx()
color = "salmon"
axr.set_ylabel("time (s)", color=color, fontsize=FONTSIZE + 2)
imgsr += axr.plot(
    t_indices,
    dt,
    color=color,
    marker="s",
    mfc="white",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label=r"$\Delta$ t",
)
axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)
axr.set_yscale("log")
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.yaxis.grid(visible=True, which="both", color="orange", alpha=0.3, linewidth=0.5)

ax.legend(
    [i.get_label() for i in imgs],
    fontsize=FONTSIZE,
    loc="upper right",
    bbox_to_anchor=(-0.1, 1),
)
axr.legend(
    [i.get_label() for i in imgsr],
    fontsize=FONTSIZE,
    loc="upper left",
    bbox_to_anchor=(1.1, 1),
)
fig.tight_layout(pad=FIGUREPAD)
fig.savefig(
    f"{FIGUREPATH}time_progress_h{D['refinement_level']}_ftol{D['tol_flash_case']}.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
)

# endregion

# region Printing table with clock times, at maximum refinement where both equilibrium
# conditions converge.

Dph: SimulationData = data["p-h"][2]
DpT: SimulationData = data["p-T"][2]


def format_times(vals: tuple[float, float]) -> str:
    """Format the clock time values."""
    return f"{vals[0]:.6f} ({vals[1]:.2f})"


table = tabulate(
    [
        [
            "p-h",
            format_times(Dph["clock_time_assembly"]),
            format_times(Dph["clock_time_global_solver"]),
            format_times(Dph["clock_time_flash_solver"]),
            Dph["total_num_time_steps"],
            Dph["total_num_global_iter"],
            Dph["total_num_flash_iter"],
        ],
        [
            "p-T",
            format_times(DpT["clock_time_assembly"]),
            format_times(DpT["clock_time_global_solver"]),
            format_times(DpT["clock_time_flash_solver"]),
            DpT["total_num_time_steps"],
            DpT["total_num_global_iter"],
            DpT["total_num_flash_iter"],
        ],
    ],
    headers=[
        "Equilibrium condition",
        "Assembly time",
        "Linear solver time",
        "Flash solver time",
        "Number of time steps",
        "Number of global iterations",
        "Number of local iterations",
    ],
    tablefmt="orgtbl",
)
print(table)

# endregion
