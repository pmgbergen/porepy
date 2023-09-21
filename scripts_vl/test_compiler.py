"""Test scripts for Peng-Robinson Compiler functionality"""

import sympy as sm
import numpy as np
import porepy as pp

import numba
import time
import inspect

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from porepy.composite.peng_robinson.eos import A_CRIT, B_CRIT

from porepy.composite.peng_robinson.compiler import (
    get_root_case_cv,
    get_root_case_c,
    one_root,
    three_root,
    three_root_intermediate,
    double_root,
    triple_root,
    characteristic_residual_c,
    red_coef0,
    red_coef1,
    discr,
    coef2,
)


def _plot_critical_line(axis: plt.Axes, A_mesh: np.ndarray):
    # slope = B_CRIT / A_CRIT
    # x_vals = np.sort(np.unique(A_mesh.flatten()))
    # y_vals = 0.0 + slope * x_vals
    # critical line
    img_line = axis.plot([0, A_CRIT], [0, B_CRIT], "-", color="red", linewidth=1)
    # critical point
    img_point = axis.plot(A_CRIT, B_CRIT, "*", markersize=6, color="red")
    return [img_point[0], img_line[0]], ["(Ac, Bc)", "Critical line"]


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
    cb_rr.set_ticks([0, 1, 2, 3])
    cb_rr.set_ticklabels(["t", "1", "2", "3"])


def plot_values(
    axis: plt.Axes,
    A_mesh: np.ndarray,
    B_mesh: np.ndarray,
    val_mesh: np.ndarray,
    val_name: str,
    cmap: str = "Greys",
    norm: float = None,
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
    axis.set_title(val_name)
    axis.set_xlabel("A")
    axis.set_ylabel("B")
    imgs_c, legs_c = _plot_critical_line(axis, A_mesh)
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


A = sm.Symbol("A")
B = sm.Symbol("B")

# t1 = - red_coef0(A, B) /2 + sm.sqrt(discr(red_coef0(A, B), red_coef1(A, B)))
# t2 = - red_coef0(A, B) /2 - sm.sqrt(discr(red_coef0(A, B), red_coef1(A, B)))

# print(t1)
# print(t2)

r3g = three_root(A, B, True)
r3g_c = numba.njit(sm.lambdify([A, B], r3g))
r3l = three_root(A, B, False)
r3l_c = numba.njit(sm.lambdify([A, B], r3l))
r3i = three_root_intermediate(A, B)
r3i_c = numba.njit(sm.lambdify([A, B], r3i))

r2g = double_root(A, B, True)
r2g_c = numba.njit(sm.lambdify([A, B], r2g, "math"))
r2l = double_root(A, B, False)
r2l_c = numba.njit(sm.lambdify([A, B], r2l, "math"))

r1 = one_root(A, B)
r1_c = numba.njit(
    sm.lambdify([A, B], r1, [{'cbrt': np.cbrt}, 'math']),
    # sm.lambdify([A, B], r1, "numpy"),
)

r0 = triple_root(A, B)
r0_c = numba.njit(sm.lambdify([A, B], r0))

# region Checks
# check_a = np.array(
#     [
#         0.33419361,
#         0.39345262,
#         0.37192364,
#         0.35763347,
#         0.4030023,
#         0.33790388,
#         0.39926982,
#         0.34914794,
#         0.37841537,
#         0.35080267,
#         0.37373099,
#         0.37443159,
#         0.40212645,
#         0.37465614,
#         0.34348945,
#         0.3416183,
#         0.4059814,
#         0.33613069,
#         0.34727329,
#         0.3360567,
#         0.39420234,
#         0.33835475,
#     ]
# )
# check_b = np.array(
#     [
#         0.00064153,
#         0.04090448,
#         0.02710387,
#         0.01746064,
#         0.04677902,
#         0.00339654,
#         0.04449987,
#         0.01152599,
#         0.03135192,
#         0.01269631,
#         0.0282944,
#         0.02875445,
#         0.04624608,
#         0.02890158,
#         0.00747517,
#         0.00611778,
#         0.04858332,
#         0.00208431,
#         0.0101926,
#         0.00202929,
#         0.04137089,
#         0.0037285,
#     ]
# )
# eos = pp.composite.peng_robinson.PengRobinsonEoS(True)
# for a, b in zip(check_a, check_b):
#     c = get_root_case_c(a, b, 1e-14)

#     if c == 1:
#         z = r1_c(a, b)
#         c2 = coef2_c(a, b)
#         q = red_coef0_c(a, b)
#         r = red_coef1_c(a, b)
#         d = discr_c(q, r)
#         t = np.sqrt(d) - q * 0.5
#         t = -np.sqrt(d) - q * .5 if np.abs(t) < 1e-14 else t
#         u = np.cbrt(t)
#         real = u - (1e10 * r) / ((1e10 * u) * 3)
#         z_ = real - c2 / 3
#         z_old = eos._Z(np.array([a]), np.array([b]))
#         print(check_if_root_c(z, a, b))
# endregion


@numba.njit
def test_residual(A_, B_, eps_):
    res = np.zeros_like(A_)
    cases = np.zeros_like(A_, dtype=np.int8)
    order_2 = np.zeros_like(A_, dtype=np.bool_)
    order_3 = np.zeros_like(A_, dtype=np.bool_)

    for i in numba.prange(len(A_)):
        # for i in range(len(A_)):
        a_ = A_[i]
        b_ = B_[i]
        c = get_root_case_c(a_, b_, eps_)
        cases[i] = c

        if c == 0:
            z_ = r0_c(a_, b_)
            r = characteristic_residual_c(z_, a_, b_)
            res[i] = r
        elif c == 1:
            z_ = r1_c(a_, b_)
            r = characteristic_residual_c(z_, a_, b_)
            res[i] = r
        elif c == 2:
            z1_ = r2g_c(a_, b_)
            z2_ = r2l_c(a_, b_)
            r1 = characteristic_residual_c(z1_, a_, b_)
            r2 = characteristic_residual_c(z2_, a_, b_)
            res[i] = (r1 + r2) / 2.0
            if z1_ >= z2_:
                order_2[i] = True
        elif c == 3:
            z1 = r3g_c(a_, b_)
            z2 = r3i_c(a_, b_)
            z3 = r3l_c(a_, b_)

            r1 = characteristic_residual_c(z1, a_, b_)
            r2 = characteristic_residual_c(z2, a_, b_)
            r3 = characteristic_residual_c(z3, a_, b_)

            res[i] = (r1 + r2 + r3) / 3.0

            if z1 >= z2 >= z3:
                order_3[i] = True

    return res, cases, order_2, order_3


n_tests = 1001
eps = 1e-16
tol = 1e-15

# region root cases and sensitivity plot
A = [0.0, 1.0]
B = [0.0, 0.16]
ref = 1000

A = np.linspace(A[0], A[1], ref, endpoint=True)
B = np.linspace(B[0], B[1], ref, endpoint=True)

Am, Bm = np.meshgrid(A, B)

res, cases, *_ = test_residual(Am.flatten(), Bm.flatten(), eps)

res = np.abs(res.reshape((ref, ref)))
cases = cases.reshape((ref, ref))

fig = plt.figure(figsize=(20, 10))
axis = fig.add_subplot(1, 2, 1)
axis.set_box_aspect(1)
plot_root_regions(axis, Am, Bm, cases, "Root cases")

axis = fig.add_subplot(1, 2, 2)
axis.set_box_aspect(1)
norm = mpl.colors.LogNorm(vmin=eps, vmax=res.max())
plot_values(axis, Am, Bm, res, "Residual", norm=norm)

# r1_case = cases == 1
# mr = np.ma.array(cases, mask=np.logical_not(r1_case))
# mr = np.ma.array(cases, mask=r1_case)
# hatch = axis.pcolor(
#     Am,
#     Bm,
#     mr,
#     hatch="//",
#     edgecolor="black",
#     cmap=mpl.colors.ListedColormap(["none"]),
#     facecolor="none",
#     vmin=0,
#     vmax=3,
#     shading="nearest",
#     lw=0,
#     zorder=2,
# )

fig.tight_layout()
fig.savefig(
    "root_res.png",
    format="png",
    dpi=300,
)
# endregion

# N = [1e2, 1e3, 1e4, 1e5, 1e6, 5e6, 1e7]
N = [1e2, 1e3, 1e4, 1e5]

N = [int(_) for _ in N]
AVG_T = []
MAX_T = []
AVG_num_violation = []
MAX_num_violation = []
AVG_max_err = []
MAX_max_err = []

check_a = []
check_b = []

start_total = time.time()
for k in range(len(N)):
    n = N[k]
    times = []
    max_errs = {}
    num_violations = {}
    print(f"\nStarting tests for n={n}", flush=True)

    for j in range(1, n_tests + 1):
        a = np.random.rand(n)
        b = np.random.rand(n)

        cases = get_root_case_cv(a, b, eps)

        print(f"\rStarting root computations (n={n}, test {j})...", flush=True, end="")
        start = time.time()
        res, cases_, order_2, order_3 = test_residual(a, b, eps)
        end = time.time()
        # print(f"\rComputations done. Required time: {end - start} s", end='')
        times.append(end - start)

        print("\rPerforming checks ...", flush=True, end="")
        if np.any(cases != cases_):
            print(f"Root case computation inconsistent: n={n} j={j}\n", flush=True)
        if not np.all(order_2[cases == 2] == True):
            print(f"Double root cases improperly ordered: n={n} j={j}\n", flush=True)
        if not np.all(order_3[cases == 3] == True):
            print(f"Triple root cases improperly ordered: n={n} j={j}\n", flush=True)
        if np.any(np.abs(res) > tol):
            # print("\rTolerance for root residual violated.", end='')
            is_infty = res == np.infty
            is_nan = res == np.nan
            if np.any(is_infty):
                print(f"\nInfty at", a[is_infty], b[is_infty])
                check_a.append(a[is_infty])
                check_b.append(b[is_infty])
            if np.any(is_nan):
                print(f"\nNan at", a[is_nan], b[is_nan])
                check_a.append(a[is_nan])
                check_b.append(b[is_nan])

            idx = np.abs(res) > 0.1
            if np.any(idx):
                check_a.append(a[idx])
                check_b.append(b[idx])
            max_errs.update({j: np.abs(res).max()})
            num_violations.update({j: len(res[np.abs(res) > tol])})

        # print(f"\rRandomized test {j} finished.", flush=True, end='')

    AVG_T.append(sum(times[1:]) / (len(times) - 1))
    MAX_T.append(max(times[1:]))
    if len(num_violations) > 0:
        AVG_num_violation.append(sum(num_violations.values()) / len(num_violations))
        MAX_num_violation.append(max(num_violations.values()))
    else:
        AVG_num_violation.append(0)
        MAX_num_violation.append(0)
    if len(max_errs) > 0:
        AVG_max_err.append(sum(max_errs.values()) / len(max_errs))
        MAX_max_err.append(max(max_errs.values()))
    else:
        AVG_max_err.append(0)
        MAX_max_err.append(0)
end_total = time.time()

print("\nPlotting ..", flush=True)
fig = plt.figure(figsize=(30, 10))
fig.suptitle(f"Tolerance: {tol} ; Total exec. time (s): {end_total - start_total}")
axis = fig.add_subplot(1, 3, 1)
axis.set_box_aspect(1)
img_a = axis.plot(N, AVG_T, "-*", color="black")
img_m = axis.plot(N, MAX_T, "--*", color="black")
axis.legend(img_a + img_m, ["Avg. calc. time (s)", "Max calc. time (s)"])
axis.set_xscale("log")
axis.set_yscale("log")

axis = fig.add_subplot(1, 3, 2)
axis.set_box_aspect(1)
img_a = axis.plot(N, AVG_num_violation, "-*", color="red")
img_m = axis.plot(N, MAX_num_violation, "--*", color="red")
axis.legend(img_a + img_m, ["Avg. num. violations", "Max num. violations"])
axis.set_xscale("log")
axis.set_yscale("log")
axis = fig.add_subplot(1, 3, 3)
axis.set_box_aspect(1)
img_a = axis.plot(N, AVG_max_err, "-*", color="black")
img_m = axis.plot(N, MAX_max_err, "--*", color="black")
axis.legend(img_a + img_m, ["Avg. residual", "Max residual"])
axis.set_xscale("log")
axis.set_yscale("log")

fig.tight_layout()
fig.savefig(
    "root_comp.png",
    format="png",
    dpi=300,
)
