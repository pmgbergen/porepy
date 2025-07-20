"""Scipts to obtain plots presented in article."""

from __future__ import annotations

import json, pathlib, matplotlib

from typing import Literal, TypedDict, TypeAlias

import matplotlib.pyplot as plt
import numpy as np


RefinementLevel: TypeAlias = Literal[0, 1, 2, 3, 4]
EquilibriumCondition: TypeAlias = Literal["p-T", "p-h"]


FIGUREPATH: str = f"{str(pathlib.Path(__file__).parent.resolve())}\\"
DPI: int = 400
FIGUREWIDTH: int = 10
FONTSIZE: int = 20

USED_MESH_SIZES: dict[RefinementLevel, float] = {
    0: 4.0,
    1: 2.0,
    2: 1.0,
    3: 5e-1,
    4: 2.5e-1,
}
"""Dictionary containing the used mesh sizes for given abreviation."""

pT_colors = {0: "yellow", 1: "gold", 2: "orange", 3: "red", 4: "darkred"}

ph_colors = {
    0: "aquamarine",
    1: "aqua",
    2: "deepskyblue",
    3: "dodgerblue",
    4: "blue",
}


class SimulationData(TypedDict, total=False):
    h: float
    """Mesh size."""
    num_cells: int
    """Number of cells corresponding to h."""
    equilibrium_condtion: EquilibriumCondition
    """Used equilibrium condition."""
    tol_flash: float
    """Tolerance for equilibrium problem."""
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


# region Loading simulation data


def load_data(
    condition: EquilibriumCondition, refinement: RefinementLevel
) -> SimulationData | None:
    return SimulationData(
        h=USED_MESH_SIZES[refinement],
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
        + (refinement),
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
imgs = []
legend = []
h = np.array(USED_MESH_SIZES.values())

nig = []
nif = []
for i in range(5):
    if i in data["p-T"]:
        D = data["p-T"][i]
        nig.append([USED_MESH_SIZES[i], D["num_global_iter"].sum()])
        nif.append([USED_MESH_SIZES[i], D["num_flash_iter"].sum()])
nig = np.array(nig).T
img = ax.plot(nig[0], nig[1], color="salmon", linestyle="solid", marker='^')
imgs.append(img)
legend.append("pT-global")
nif = np.array(nif).T
img = ax.plot(nif[0], nif[1], color="salmon", linestyle="dashed", marker='x')
imgs.append(img)
legend.append("pT-flash")

nig = []
nif = []
for i in range(5):
    if i in data["p-h"]:
        D = data["p-h"][i]
        nig.append([USED_MESH_SIZES[i], D["num_global_iter"].sum()])
        nif.append([USED_MESH_SIZES[i], D["num_flash_iter"].sum()])
nig = np.array(nig).T
img = ax.plot(nig[0], nig[1], color="slateblue", linestyle="solid", marker='^')
imgs.append(img)
legend.append("ph-global")
nif = np.array(nif).T
img = ax.plot(nif[0], nif[1], color="slateblue", linestyle="dashed", marker='x')
imgs.append(img)
legend.append("ph-flash")

ax.set_xlabel("mesh size [m]", fontsize=FONTSIZE+2)
ax.set_xscale('log')
ax.set_xticks(np.array(list(USED_MESH_SIZES.values())))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.grid(visible=True, which='both', color='grey', alpha=0.3, linewidth=0.5)
ax.set_ylabel("total number of iterations", fontsize=FONTSIZE+2)
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE)
ax.legend(legend, fontsize=FONTSIZE)

fig.tight_layout()
fig.savefig(
    f"{FIGUREPATH}total_iter_per_refinement.png",
    format="png",
    dpi=DPI,
)

# endregion

# region Plotting num iterations, time step size and re-computations per time step.

D: SimulationData = data["p-h"][3]

# endregion

# region Plotting total num iterations per flash tolerance.

D: SimulationData = data["p-h"][3]

# endregion

# region Printing table with clock times, at maximum refinement where both equilibrium
# conditions converge.

Dph: SimulationData = data["p-h"][2]
DpT: SimulationData = data["p-T"][2]

# endregion
