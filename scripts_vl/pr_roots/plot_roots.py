"""Testing module for the Peng-Robinson EoS class."""
import pathlib
import sys
from typing import Any, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# adding script path to find relevant moduls
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from calculate_roots import (
    A_CRIT,
    B_CRIT,
    EPS,
    REGION_ENCODING,
    path,
    read_root_results,
)

RESULTFILE: str = "roots.csv"
FIGUREPATH: str = "figs"
FIGWIDTH: int = 15  # in inches, 1080 / 1920 ratio applied to height
DPI: int = 400


def res_compressibility(Z, A, B):
    """Returns the evaluation of the cubic compressibility polynomial p(A,B)[Z].

    If Z is a root, this should return zero.
    """
    return np.abs(
        Z * Z * Z
        + (B - 1) * Z * Z
        + (A - 2 * B - 3 * B * B) * Z
        + (B * B + B * B * B - A * B)
    )


def _plot_acbc_rectangle(axis: plt.Axes):

    # AcBc rectangle
    axis.plot([A_CRIT, A_CRIT], [0, B_CRIT], "--", color="red", linewidth=1)
    axis.plot([0, A_CRIT], [B_CRIT, B_CRIT], "--", color="red", linewidth=1)
    axis.plot([0, 0], [0, B_CRIT], "--", color="red", linewidth=1)
    axis.plot([0, A_CRIT], [0, 0], "--", color="red", linewidth=1)


def _plot_critical_line(axis: plt.Axes, A_mesh: np.ndarray):
    slope = B_CRIT / A_CRIT
    x_vals = np.sort(np.unique(A_mesh.flatten()))
    y_vals = 0.0 + slope * x_vals
    # critical line
    img_line = axis.plot(x_vals, y_vals, "-", color="red", linewidth=1)
    # critical point
    img_point = axis.plot(A_CRIT, B_CRIT, "*", markersize=6, color="red")
    _plot_acbc_rectangle(axis)
    return [img_point[0], img_line[0]], ["(Ac, Bc)", "Critical line"]


def _plot_B_violation(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    val_mesh: np.ndarray,
    val_name: str,
):
    violated = val_mesh <= B_mesh
    if np.any(violated):
        img = axis.plot(
            A_mesh[violated], B_mesh[violated], "v", markersize=0.1, color="red"
        )
        return [img[0]], [f"{val_name} < B"]
    else:
        return [], []


def _plot_negative(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    val_mesh: np.ndarray,
    val_name: str,
):
    neg = val_mesh < 0
    if np.any(neg):
        img = axis.plot(A_mesh[neg], B_mesh[neg], "*", markersize=3, color="red")
        return [img[0]], [f"{val_name} < 0"]
    else:
        return [], []


def _plot_positive(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    val_mesh: np.ndarray,
    val_name: str,
):
    pos = val_mesh > 0
    if np.any(pos):
        img = axis.plot(A_mesh[pos], B_mesh[pos], "*", markersize=1, color="green")
        return [img[0]], [f"{val_name} > 0"]
    else:
        return [], []


def _plot_iszero(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    val_mesh: np.ndarray,
    val_name: str,
):
    zero = np.isclose(val_mesh, 0.0, rtol=0, atol=EPS)
    if np.any(zero):
        img = axis.plot(A_mesh[zero], B_mesh[zero], "*", markersize=1.5, color="indigo")
        return [img[0]], [f"{val_name} = 0 (eps = {EPS})"]
    else:
        return [], []


def plot_root_regions(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    val_mesh: np.ndarray,
    val_name: str,
):
    """A discrete plot for plotting the root cases."""
    cmap = mpl.colors.ListedColormap(["yellow", "green", "blue", "indigo"])
    img_rr = axis.pcolormesh(A_mesh, B_mesh, val_mesh, cmap=cmap, vmin=0, vmax=3)
    imgs_c, legs_c = _plot_critical_line(axis, A_mesh)
    axis.legend(imgs_c, legs_c, loc="upper left")
    axis.set_title(val_name)
    axis.set_xlabel("A")
    axis.set_ylabel("B")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb_rr = fig.colorbar(img_rr, cax=cax, orientation="vertical")
    cb_rr.set_ticks(REGION_ENCODING)
    cb_rr.set_ticklabels(["triple-root", "1-real-root", "2-real-roots", "3-real-roots"])


def plot_extension_markers(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    liq_extended: np.ndarray,
    gas_extended: np.ndarray,
):
    """A discrete plot for plotting the root cases."""
    # empt mesh plot to scale the figure properly
    axis.pcolormesh(
        A_mesh, B_mesh, np.zeros(A_mesh.shape), cmap="Greys", vmin=0, vmax=1
    )
    img_l = axis.plot(
        A_mesh[liq_extended], B_mesh[liq_extended], ".", markersize=1, color="blue"
    )
    leg_l = "Liquid extended"
    img_g = axis.plot(
        A_mesh[gas_extended], B_mesh[gas_extended], ".", markersize=1, color="red"
    )
    leg_g = "Gas extended"
    imgs_c, legs_c = _plot_critical_line(axis, A_mesh)
    axis.legend(
        [img_l[0], img_g[0]] + imgs_c, [leg_l, leg_g] + legs_c, loc="upper left"
    )
    axis.set_title("Usage of extended roots")
    axis.set_xlabel("A")
    axis.set_ylabel("B")


def plot_values(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    val_mesh: np.ndarray,
    val_name: str,
    cmap: str = "Greys",
    norm: Optional[Any] = None,
    plot_B_violation: bool = False,
    plot_neg: bool = False,
    plot_pos: bool = False,
    plot_zero: bool = False,
):
    """Plot any values on given axis and mesh."""
    vmin, vmax = val_mesh.min(), val_mesh.max()

    if norm:
        img = axis.pcolormesh(
            A_mesh,
            B_mesh,
            val_mesh,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )
    else:
        img = axis.pcolormesh(
            A_mesh,
            B_mesh,
            val_mesh,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="nearest",
        )
    imgs_c, legs_c = _plot_critical_line(axis, A_mesh)
    if plot_neg:
        imgs_n, legs_n = _plot_negative(axis, A_mesh, B_mesh, val_mesh, val_name)
    else:
        imgs_n = []
        legs_n = []
    if plot_pos:
        imgs_p, legs_p = _plot_positive(axis, A_mesh, B_mesh, val_mesh, val_name)
    else:
        imgs_p = []
        legs_p = []
    if plot_zero:
        imgs_z, legs_z = _plot_iszero(axis, A_mesh, B_mesh, val_mesh, val_name)
    else:
        imgs_z = []
        legs_z = []
    if plot_B_violation:
        imgs_v, legs_v = _plot_B_violation(axis, A_mesh, B_mesh, val_mesh, val_name)
    else:
        imgs_v = []
        legs_v = []
    axis.legend(
        imgs_c + imgs_v + imgs_n + imgs_z + imgs_p,
        legs_c + legs_v + legs_n + legs_z + legs_p,
        loc="upper left",
    )
    axis.set_title(val_name)
    axis.set_xlabel("A")
    axis.set_ylabel("B")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(
        img,
        cax=cax,
        orientation="vertical",
    )
    cb.set_label(
        f"Min.: {'{:.6e}'.format(vmin)}"
        + f" ; Max.: {'{:.6e}'.format(vmax)}"
        + f"\nMean: {'{:.6e}'.format(np.mean(val_mesh))}"
    )


if __name__ == "__main__":
    fname = RESULTFILE[RESULTFILE.rfind("/") + 1 : RESULTFILE.rfind(".csv")]

    A, B, ab_map = read_root_results(RESULTFILE)

    A_mesh, B_mesh = np.meshgrid(A, B)

    n, m = A_mesh.shape
    nm = n * m

    REG = np.zeros(A_mesh.shape)

    GAS_RES = np.zeros(A_mesh.shape)
    LIQ_RES = np.zeros(A_mesh.shape)

    GAS_ROOT = np.zeros(A_mesh.shape)
    LIQ_ROOT = np.zeros(A_mesh.shape)

    GAS_EXTENDED = np.zeros(A_mesh.shape, dtype=bool)
    LIQ_EXTENDED = np.zeros(A_mesh.shape, dtype=bool)

    # region Calculating plot data
    print("Calculating Plot data: ...", end="", flush=True)
    counter: int = 1
    ignore = [np.infty, -np.infty]
    for i in range(n):
        for j in range(m):

            A_ = A_mesh[i, j]
            B_ = B_mesh[i, j]

            reg, Z_L, Z_G, is_extended = ab_map[(A_, B_)]

            # TODO delete this once the infinity issue is solved.
            if Z_L in ignore or Z_G in ignore:
                continue

            REG[i, j] = reg
            GAS_ROOT[i, j] = Z_G
            LIQ_ROOT[i, j] = Z_L

            # data in the 1-real-root region
            if reg == REGION_ENCODING[1]:
                if is_extended == 0:
                    GAS_EXTENDED[i, j] = True
                    LIQ_RES[i, j] = res_compressibility(Z_L, A_, B_)
                elif is_extended == 1:
                    LIQ_EXTENDED[i, j] = True
                    GAS_RES[i, j] = res_compressibility(Z_G, A_, B_)
            else:
                if not B_ > B_CRIT / A_CRIT * A_:  # exclude because of correction
                    LIQ_RES[i, j] = res_compressibility(Z_L, A_, B_)
                GAS_RES[i, j] = res_compressibility(Z_G, A_, B_)

            print(f"\rCalculating Plot data: {counter}/{nm}", end="", flush=True)
            counter += 1
    print("\nCalculating Plot data: Done", flush=True)

    NUM_TRIPPLEROOT_CASES = np.sum(REG == REGION_ENCODING[0])
    NUM_ONEROOT_CASES = np.sum(REG == REGION_ENCODING[1])
    NUM_DOUBLEROOT_CASES = np.sum(REG == REGION_ENCODING[2])
    NUM_THREEROOT_CASES = np.sum(REG == REGION_ENCODING[3])
    print(
        f"Cases with a double root: {NUM_DOUBLEROOT_CASES} ({NUM_DOUBLEROOT_CASES / nm * 100} %)"
    )
    print(
        f"Cases with a triple root: {NUM_TRIPPLEROOT_CASES} ({NUM_TRIPPLEROOT_CASES / nm * 100} %)"
    )
    print(f"Cases with 1 real root: {NUM_ONEROOT_CASES / nm * 100} %")
    print(f"Cases with 3 real roots: {NUM_THREEROOT_CASES / nm * 100} %")
    # endregion

    # # error scaling
    # MAX_ERR_LIQ = LIQ_RES.max()
    # MAX_ERR_GAS = GAS_RES.max()
    # LIQ_RES = LIQ_RES / MAX_ERR_LIQ
    # GAS_RES = GAS_RES / MAX_ERR_GAS
    # is_zero_l = LIQ_RES == 0
    # is_zero_g = GAS_RES == 0
    # LIQ_RES[np.logical_not(is_zero_l)] = np.log(LIQ_RES[np.logical_not(is_zero_l)])
    # GAS_RES[np.logical_not(is_zero_g)] = np.log(GAS_RES[np.logical_not(is_zero_g)])
    # LIQ_RES[is_zero_l] = 0
    # GAS_RES[is_zero_g] = 0

    # norm = mpl.colors.SymLogNorm(
    #     linthresh=1e-3, linscale=0.5, vmin=vmin, vmax=vmax
    # )
    # norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    # region Plot 1: Root cases
    print("Plotting: Root cases", end="", flush=True)
    fig = plt.figure(figsize=(FIGWIDTH, 1080 / 1920 * FIGWIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Overview")
    axis = fig.add_subplot(gs[0, 0])
    plot_root_regions(axis, A_mesh, B_mesh, REG, "Root cases")

    axis = fig.add_subplot(gs[0, 1])
    plot_extension_markers(axis, A_mesh, B_mesh, LIQ_EXTENDED, GAS_EXTENDED)

    fig.tight_layout()
    fig.savefig(
        f"{str(path())}/{FIGUREPATH}/1_roots__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 2: Residuals
    print("\rPlotting: Residuals", end="", flush=True)
    fig = plt.figure(figsize=(FIGWIDTH, 1080 / 1920 * FIGWIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Root residuals (non-extended)")
    axis = fig.add_subplot(gs[0, 0])
    plot_values(axis, A_mesh, B_mesh, LIQ_RES, "Liquid root residuals")

    axis = fig.add_subplot(gs[0, 1])
    plot_values(axis, A_mesh, B_mesh, GAS_RES, "Gas root residuals")

    fig.tight_layout()
    fig.savefig(
        f"{str(path())}/{FIGUREPATH}/2_residuals__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 3: root values
    print("\rPlotting: Root values", end="", flush=True)
    fig = plt.figure(figsize=(FIGWIDTH, 1080 / 1920 * FIGWIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Root values (with extension)")
    axis = fig.add_subplot(gs[0, 0])
    plot_values(
        axis, A_mesh, B_mesh, LIQ_ROOT, "Z_L", plot_B_violation=True, plot_zero=True
    )

    axis = fig.add_subplot(gs[0, 1])
    plot_values(
        axis, A_mesh, B_mesh, GAS_ROOT, "Z_G", plot_B_violation=True, plot_zero=True
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(path())}/{FIGUREPATH}/3_rootvals__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 4: B violations
    fig = plt.figure(figsize=(FIGWIDTH, 1080 / 1920 * FIGWIDTH))
    gs = fig.add_gridspec(1, 2)
    fig.suptitle(f"Violations of lower bound B")
    axis = fig.add_subplot(gs[0, 0])

    shift = np.min([LIQ_ROOT.min(), B_mesh.min()])
    if shift < 0:
        L_temp = LIQ_ROOT + np.abs(shift)
        B_temp = B_mesh + np.abs(shift)
        diff = np.abs(L_temp - B_temp)
        neg = L_temp < B_temp
        diff[neg] = diff[neg] * (-1)
    else:
        diff = np.abs(LIQ_ROOT - B_mesh)
        neg = LIQ_ROOT < B_mesh
        diff[neg] = diff[neg] * (-1)

    vmin = diff.min()
    vmax = diff.max()
    if vmin > 0:
        vcenter = vmin + EPS
    else:
        vcenter = EPS
    norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)

    plot_values(
        axis,
        A_mesh,
        B_mesh,
        diff,
        "diff(Z_L, B)",
        cmap="coolwarm",
        norm=norm,
        plot_B_violation=False,
        plot_neg=True,
        # plot_zero=True
    )

    axis = fig.add_subplot(gs[0, 1])

    shift = np.min([GAS_ROOT.min(), B_mesh.min()])
    if shift < 0:
        G_TEMP = GAS_ROOT + np.abs(shift)
        B_temp = B_mesh + np.abs(shift)
        diff = np.abs(G_TEMP - B_temp)
        neg = G_TEMP < B_temp
        diff[neg] = diff[neg] * (-1)
    else:
        diff = np.abs(GAS_ROOT - B_mesh)
        neg = GAS_ROOT < B_mesh
        diff[neg] = diff[neg] * (-1)

    vmin = diff.min()
    vmax = diff.max()
    if vmin > 0:
        vcenter = vmin + EPS
    else:
        vcenter = EPS
    norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    plot_values(
        axis,
        A_mesh,
        B_mesh,
        diff,
        "diff(Z_G, B)",
        cmap="coolwarm",
        # norm=norm,
        plot_B_violation=False,
        plot_neg=True,
        plot_zero=True,
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(path())}/{FIGUREPATH}/4_Bviolation__{fname}.png",
        format="png",
        dpi=DPI,
    )

    # endregion

    # region Plot 5: Root diff
    print("\rPlotting: Root differences", end="", flush=True)
    fig = plt.figure(figsize=(FIGWIDTH, 1080 / 1920 * FIGWIDTH))
    gs = fig.add_gridspec(1, 1)
    # fig.suptitle(f"Root differences")
    axis = fig.add_subplot(gs[0, 0])

    shift = np.min([GAS_ROOT.min(), LIQ_ROOT.min()])
    if shift < 0:
        G_temp = GAS_ROOT + np.abs(shift)
        L_temp = LIQ_ROOT + np.abs(shift)
        diff = np.abs(G_temp - L_temp)
        neg = G_temp < L_temp
        diff[neg] = diff[neg] * (-1)
    else:
        diff = np.abs(GAS_ROOT - LIQ_ROOT)
        neg = GAS_ROOT < LIQ_ROOT
        diff[neg] = diff[neg] * (-1)

    vmin = diff.min()
    vmax = diff.max()
    if vmin >= 0:
        vcenter = vmin + EPS
    else:
        vcenter = EPS
    norm = mpl.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    plot_values(
        axis,
        A_mesh,
        B_mesh,
        diff,
        "diff(Z_G, Z_L)",
        cmap="coolwarm",
        # norm=norm,
        plot_B_violation=False,
        plot_neg=True,
        plot_zero=True,
    )

    fig.tight_layout()
    fig.savefig(
        f"{str(path())}/{FIGUREPATH}/5_rootdiffs__{fname}.png",
        format="png",
        dpi=DPI,
    )
    # endregion

    # region Plot 6: Intermediate root
    # print("\rPlotting: Intermediate root", end='', flush=True)
    # fig = plt.figure(figsize=(FIGWIDTH, 1080 / 1920 * FIGWIDTH))
    # gs = fig.add_gridspec(1, 2)
    # fig.suptitle(f"Intermediate root")

    # axis = fig.add_subplot(gs[0, 0])
    # plot_values(
    #     axis, A_mesh, B_mesh, INT_ROOT, 'Z_i',
    #     plot_B_violation=False, plot_zero=False, plot_neg=True
    # )

    # axis = fig.add_subplot(gs[0, 1])
    # plot_values(axis, A_mesh, B_mesh, INT_RES, 'Z_i residuals')

    # fig.tight_layout()
    # fig.savefig(
    #     f"{str(path())}/{FIGUREPATH}/6_Z_i__{fname}.png",
    #     format="png",
    #     dpi=DPI,
    # )
    # endregion

    print("\nPlotting: done", flush=True)
