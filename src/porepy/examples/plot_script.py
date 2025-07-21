"""Scipts to obtain plots presented in article."""

from __future__ import annotations

import json
import pathlib
from typing import Literal, TypeAlias, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

RefinementLevel: TypeAlias = Literal[0, 1, 2, 3, 4]
EquilibriumCondition: TypeAlias = Literal["p-T", "p-h"]


FIGUREPATH: str = f"{str(pathlib.Path(__file__).parent.resolve())}\\"
DPI: int = 400
FIGUREWIDTH: int = 10
FONTSIZE: int = 20
MARKERSIZE: int = 10

USED_MESH_SIZES: dict[RefinementLevel, float] = {
    0: 4.0,
    1: 2.0,
    2: 1.0,
    3: 5e-1,
    4: 2.5e-1,
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
    recomputations: dict[float, int]
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


# region Loading simulation data


def load_data(
    condition: EquilibriumCondition,
    refinement: RefinementLevel,
    tol_flash_case: int = 4,
) -> SimulationData | None:
    return SimulationData(
        refinement_level=refinement,
        tol_flash_case=tol_flash_case,
        num_cells=1,
        equilibrium_condtion=condition,
        tol_flash=1e-8,
        t=np.array([1.0, 2.0, 3.0]),
        dt=np.array([1.0, 1.0, 1.0]),
        recomputations={1.0: 1, 2.5: 2, 3.0: 3},
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

fig = plt.figure(figsize=(FIGUREWIDTH, 0.8 * FIGUREWIDTH))
ax = fig.add_subplot(1, 1, 1)
axr = ax.twinx()
imgs = []
imgsr = []
MARKERSIZE = 7

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
    label="pT-global",
)
nfi = np.array(nfi).T
imgsr += axr.plot(
    nfi[0],
    nfi[1],
    color=color,
    linestyle="dashed",
    marker="x",
    markersize=MARKERSIZE,
    label="pT-local",
)
nli = np.array(nli).T
imgs += ax.plot(
    nli[0],
    nli[1],
    color=color,
    linestyle="dotted",
    marker="1",
    markersize=MARKERSIZE,
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
    label="ph-global",
)
nfi = np.array(nfi).T
imgsr += axr.plot(
    nfi[0],
    nfi[1],
    color=color,
    linestyle="dashed",
    marker="x",
    markersize=MARKERSIZE,
    label="ph-local",
)
nli = np.array(nli).T
imgs += ax.plot(
    nli[0],
    nli[1],
    color=color,
    linestyle="dotted",
    marker="1",
    markersize=MARKERSIZE,
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

ax.legend([i.get_label() for i in imgs], fontsize=FONTSIZE, loc="upper left")
axr.legend([i.get_label() for i in imgsr], fontsize=FONTSIZE, loc="upper right")
fig.tight_layout()
fig.savefig(
    f"{FIGUREPATH}total_iter_per_refinement.png",
    format="png",
    dpi=DPI,
)

# endregion

# region Plotting num iterations, time step size and re-computations per time step.

D: SimulationData = data["p-h"][3]

t = np.array(D["t"])
ngi = np.array(D["num_global_iter"]).astype(int)
nfi = np.array(D["num_flash_iter"]).astype(float)
nli = np.array(D["num_linesearch_iter"]).astype(int)

fig = plt.figure(figsize=(FIGUREWIDTH, 0.8 * FIGUREWIDTH))
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
    label="global",
)
imgs += ax.plot(
    t,
    nli,
    color=color,
    linestyle="dotted",
    marker="1",
    markersize=MARKERSIZE,
    label="line search",
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
    marker="x",
    markersize=MARKERSIZE,
    label="local",
)
axr.set_ylabel("Average local iterations", color=color, fontsize=FONTSIZE + 2)
axr.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axr.tick_params(axis="y", which="both", labelcolor=color, labelsize=FONTSIZE)

ax.legend([i.get_label() for i in imgs], fontsize=FONTSIZE, loc="upper left")
axr.legend([i.get_label() for i in imgsr], fontsize=FONTSIZE, loc="upper right")
fig.tight_layout()
fig.savefig(
    f"{FIGUREPATH}iterations_per_time_h{D['refinement_level']}_ftol{D['tol_flash_case']}.png",
    format="png",
    dpi=DPI,
)

# endregion

# region Plotting total num iterations per flash tolerance.

D: SimulationData = data["p-h"][3]

# endregion

# region Plotting time progress.

D: SimulationData = data["p-h"][3]
fig = plt.figure(figsize=(FIGUREWIDTH, 0.8 * FIGUREWIDTH))
ax = fig.add_subplot(1, 1, 1)
imgs = []
imgsr = []

t = np.array(D["t"])
dt = np.array(D["dt"])
t_indices = np.arange(t.size)


color = "black"
ax.set_xlabel("Time step index")
ax.set_ylabel("t (s)", color=color)
imgs += ax.plot(
    t_indices, t, color=color, marker="^", markersize=MARKERSIZE, label="time step"
)
ax.tick_params(axis="y", labelcolor=color)
ax.set_yscale("log")
ax.get_xaxis().set_major_locator(matplotlib.ticker.MultipleLocator(base=10))
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.grid(visible=True, which="both", color="grey", alpha=0.3, linewidth=0.5)

axr = ax.twinx()
color = "salmon"
axr.set_ylabel("dt (s)", color=color)
imgsr += axr.plot(
    t_indices,
    dt,
    color=color,
    marker="1",
    markersize=MARKERSIZE,
    label="time step size",
)
axr.tick_params(axis="y", labelcolor=color)
axr.set_yscale("log")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.grid(visible=True, which="both", color="orange", alpha=0.3, linewidth=0.5)

ax.legend([i.get_label() for i in imgs], fontsize=FONTSIZE, loc="upper left")
axr.legend([i.get_label() for i in imgsr], fontsize=FONTSIZE, loc="upper right")
fig.tight_layout()
fig.savefig(
    f"{FIGUREPATH}time_progress_h{D['refinement_level']}_ftol{D['tol_flash_case']}.png",
    format="png",
    dpi=DPI,
)

# endregion

# region Printing table with clock times, at maximum refinement where both equilibrium
# conditions converge.

Dph: SimulationData = data["p-h"][2]
DpT: SimulationData = data["p-T"][2]

# endregion
