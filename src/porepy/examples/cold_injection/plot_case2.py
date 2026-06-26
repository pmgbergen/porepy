"""Plot script for case 2.

Used to create plots for publication below.

"""

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from porepy.examples.cold_injection.run_case2a import (
    JUMP_TIME,
    T_BEFORE_JUMP,
    dt_init,
    dt_min,
)

# os.system(f"rm -rf {mpl.get_cachedir()}/tex.cache/*")

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 28

DPI = 400  # Figure resolution.
WIDTH = 10.0  # Figure width.
PAD = 0.05  # Figure pad to borders.
MS = 10  # Marker size.
LW = 3  # line width.
FOLDER = "visualization/"

# Fetching data stored in directories.
c2a_dir: str = "visualization/CI_CASE2A/"
c2b_dir: str = "visualization/CI_CASE2B/"

c2a_sims = [p.name for p in Path(c2a_dir).iterdir() if p.is_dir()]
c2b_sims = [p.name for p in Path(c2b_dir).iterdir() if p.is_dir()]

print("Found Case 2a dirs:\n" + "\n".join(c2a_sims))
print("Found Case 2b dirs:\n" + "\n".join(c2b_sims))


# region Read data
def fetch(folder: str, sim: str) -> dict:
    p = f"{folder}{sim}"
    path = Path(p + "/solver_statistics.json").resolve()
    if not path.is_file():
        raise ValueError(f"Statistics file not found for {p}")

    data = json.load(path.open("r"))
    data["_folder"] = folder
    data["_subfolder"] = sim
    return data


def collect_from_raw(data: dict) -> dict:

    folder = data["_folder"]

    if "CASE2A" in folder or "CASE2B" in folder:
        thermal = False
    else:
        thermal = True

    d = {}
    d["_folder"] = data["_folder"]
    d["_subfolder"] = data["_subfolder"]

    d["transient_times"] = np.array(data["global"]["transient_t"])
    d["transient_end_time"] = data["global"]["transient_end_time"]
    d["gas_disappears_time"] = data["global"]["gas_disappears_time"]
    d["p_transient_end_time"] = data["global"]["p_transient_end_time"]
    d["delta_p_l2_transient"] = np.array(data["global"]["delta_p_l2_transient"])
    d["delta_p_max_transient"] = np.array(data["global"]["delta_p_max_transient"])

    if thermal:
        d["delta_T_l2_transient"] = np.array(data["global"]["delta_T_l2_transient"])
        d["delta_T_max_transient"] = np.array(data["global"]["delta_T_max_transient"])
        d["T_transient_end_time"] = data["global"]["T_transient_end_time"]

    d["num_cells"] = sum([int(_) for _ in data["global"]["num_cells"].values()])
    d["success"] = (
        True if data["global"]["final_simulation_status"] == "successful" else False
    )
    d["total_global_iter"] = int(data["global"]["total_num_iterations"])
    d["wasted_global_iter"] = int(data["global"]["total_num_waisted_iterations"])
    d["total_time_steps"] = int(data["global"]["total_num_time_steps"])
    d["wasted_time_steps"] = int(data["global"]["total_num_failed_time_steps"])
    d["total_flash_iter"] = 0
    d["wasted_flash_iter"] = 0
    d["total_tr_iter"] = 0
    d["wasted_tr_iter"] = 0
    d["time_in_flash"] = 0.0
    d["time_in_linsolve"] = 0.0
    d["time_in_assembly"] = 0.0
    d["time_in_tr"] = 0.0

    n = int(data["global"]["num_entries"])
    times = [0.0]
    t_step_sizes = [dt_init]
    dt_halving_count = 0
    y_frac = [0.0]
    sat_frac = [0.0]

    times_of_halvings = {"t": [], "count": []}

    for k in range(n):
        sd = data[str(k)]
        success = True if sd["simulation_status"] == "successful" else False

        d["total_flash_iter"] += sum(sd["flash_iterations"])
        d["total_tr_iter"] += sum(sd["ntrdc_iterations"])

        d["time_in_flash"] += sum(sd["flash_clocktime"])
        d["time_in_linsolve"] += sum(sd["linsolve_clocktime"])
        d["time_in_assembly"] += sum(sd["assembly_clocktime"])
        d["time_in_tr"] += sum(sd["ntrdc_clocktime"])

        if success:
            y_frac.append(sd["gas_in_frac"])
            sat_frac.append(sd["sat_in_frac"])
            times.append(sd["time"])
            t_step_sizes.append(sd["dt"])

            if dt_halving_count > 0:
                times_of_halvings["t"].append(sd["time"])
                times_of_halvings["count"].append(dt_halving_count)
            dt_halving_count = 0
        else:
            d["wasted_flash_iter"] += sum(sd["flash_iterations"])
            d["wasted_tr_iter"] += sum(sd["ntrdc_iterations"])
            dt_halving_count += 1

    d["times"] = np.array(times)
    d["dts"] = np.array(t_step_sizes)
    d["y_frac"] = np.array(y_frac)
    d["sat_frac"] = np.array(sat_frac)
    d["times_of_halvings"] = (
        np.array(times_of_halvings["t"]),
        np.array(times_of_halvings["count"]),
    )

    return d


c2b_ajump = [
    fetch(c2b_dir, p) for p in c2b_sims if ("EPRIM_True" in p) and ("ICHOR_True" in p)
]
c2b_e = [fetch(c2b_dir, p) for p in c2b_sims if "EPRIM_False" in p][0]
c2b_npc = [fetch(c2b_dir, p) for p in c2b_sims if "ICHOR_False" in p][0]

c2a_ajump = [
    fetch(c2a_dir, p) for p in c2a_sims if ("EPRIM_False" in p) and ("ICHOR_True" in p)
][0]
c2a_npc = [
    fetch(c2a_dir, p) for p in c2a_sims if ("EPRIM_False" in p) and ("ICHOR_False" in p)
][0]

plot_data = {}


def get_a(data: dict) -> float:
    name: str = data["_subfolder"]
    i = name.find("_AJUMP_")
    j = name.find("_ICHOR_")
    return float(name[i + 7 : j])


# Case 2a data
plot_data["case2a"] = {}
plot_data["case2a"]["ajump"] = collect_from_raw(c2a_ajump)
plot_data["case2a"]["no_npc"] = collect_from_raw(c2a_npc)

# Case 2b data
plot_data["case2b"] = {}
plot_data["case2b"]["no_npc"] = collect_from_raw(c2b_npc)
# plot_data["case2b"]["ext_elim"] = collect_from_raw(c2b_e)
plot_data["case2b"]["ajump"] = {}

c2b_plot: dict[float, tuple[np.ndarray, np.ndarray]] = {}
for data in c2b_ajump:
    a = get_a(data)
    plot_data["case2b"]["ajump"][a] = collect_from_raw(data)

# Sort with aperture jump
plot_data["case2b"]["ajump"] = dict(sorted(plot_data["case2b"]["ajump"].items()))

# endregion

frac_tol = 1e-10  # Below this, gas is considered absent.
vT_cmap = plt.colormaps["cool"](
    np.linspace(0, 1, len(plot_data["case2b"]["ajump"]), endpoint=True)
)
uv_cmap = plt.colormaps["copper_r"](
    np.linspace(0, 1, len(plot_data["case2b"]["ajump"]), endpoint=True)
)

# region Plot gas content for vT and vu formulations (case2b and case2c)
fig = plt.figure(figsize=(WIDTH, 0.5 * WIDTH))
ax = fig.add_subplot(111)
imgs = []

x_max = []
x_vals = []
y_vals = []
labels = []
colors = []
is_empty = []
for i, ad in enumerate(plot_data["case2b"]["ajump"].items()):
    a, d = ad
    x = d["times"]
    y = d["sat_frac"]
    x_idx = x > (JUMP_TIME - dt_min / 10)
    idx = (y > frac_tol) & x_idx

    if np.any(idx):
        j = np.where(idx)[0].max()
        idx[j + 1] = True
    x = x[idx] - JUMP_TIME
    y = y[idx]
    if np.any(x):
        x_max.append(x.max())

    x_vals.append(x)
    y_vals.append(y)
    colors.append(vT_cmap[i])
    labels.append(f"vT(vT)-{a}")

xm = np.max(x_max)
x_factor = 1 / 60
for x, y, l, c in zip(x_vals, y_vals, labels, colors):
    if not np.any(y):
        x = np.linspace(0, xm, 10, endpoint=True)
        y = np.zeros_like(x)

    idx = x <= xm
    imgs += ax.plot(
        x[idx] * x_factor,
        y[idx],
        linestyle="dotted",
        linewidth=LW,
        # marker="*",
        # markersize=MS,
        color=c,
        label=l,
    )

ax.set_xlabel(r"$t - t_{\ast}$ [min]")
ax.set_ylabel(r"$s_{\text{frac}}$")
ax.set_xlim(0.0, np.max(x_max) * x_factor)

ax.grid(axis="y", alpha=0.5)

# ax.legend(handles=imgs, loc="upper right")
ax.legend(handles=imgs, loc="upper left", bbox_to_anchor=(1.05, 1))

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}sat_frac_over_tau_cases_bc.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Gas transient duration per aperture
fig = plt.figure(figsize=(0.5 * WIDTH, 0.5 * WIDTH))
ax = fig.add_subplot(111)
imgs = []

x = []
y = []

for a, d in plot_data["case2b"]["ajump"].items():
    x.append(a)
    y.append(d["gas_disappears_time"] - JUMP_TIME)

imgs += ax.plot(
    x,
    y,
    linestyle="solid",
    linewidth=LW,
    # marker="*",
    # markersize=MS,
    color="black",
    label=r"gas transient time",
)


ax.set_xlabel(r"$a(t_{\ast})/a(t_{-1})$")
ax.set_ylabel(r"$\tau_G$ [s]")

ax.set_xticks(x)
ax.grid(axis="x")
ax.grid(axis="y", which="major", alpha=0.5)

ax.set_yscale("log")

# ax.legend(handles=imgs, loc="upper left", bbox_to_anchor=(1.05, 1))
# axr.legend(handles=imgrs, loc="upper left", bbox_to_anchor=(1.05, 1))

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}vT_tau_G_per_a.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion

TAU_G_MAX = np.max(y)

# region Time steps and gas transients for isothermal cases.
fig = plt.figure(figsize=(WIDTH, 0.7 * WIDTH))
ax = fig.add_subplot(111)
axr = ax.twinx()
imgs = []
imgrs = []

T_MIN = T_BEFORE_JUMP - dt_min
T_MAX = JUMP_TIME + TAU_G_MAX + 3 * 3600

# pT no npc
x1 = plot_data["case2a"]["no_npc"]["times"]
idx1 = (x1 >= T_MIN) & (x1 <= T_MAX)
x1 = x1[idx1]
y1 = plot_data["case2a"]["no_npc"]["sat_frac"][idx1]
yr1 = plot_data["case2a"]["no_npc"]["dts"][idx1]
label1 = f"pT-{get_a(plot_data['case2a']['no_npc'])}"

# pT vT npc
x2 = plot_data["case2a"]["ajump"]["times"]
idx2 = (x2 >= T_MIN) & (x2 <= T_MAX)
x2 = x2[idx2]
y2 = plot_data["case2a"]["ajump"]["sat_frac"][idx2]
yr2 = plot_data["case2a"]["ajump"]["dts"][idx2]
label2 = f"pT(vT)-{get_a(plot_data['case2a']['ajump'])}"

# vT no npc
x3 = plot_data["case2a"]["no_npc"]["times"]
idx3 = (x3 >= T_MIN) & (x3 <= T_MAX)
x3 = x3[idx3]
y3 = plot_data["case2a"]["no_npc"]["sat_frac"][idx3]
yr3 = plot_data["case2a"]["no_npc"]["dts"][idx3]
label3 = f"vT-{get_a(plot_data['case2a']['no_npc'])}"

# vT vT npc
_a = int(np.max(list(plot_data["case2b"]["ajump"].keys())))
x4 = plot_data["case2b"]["ajump"][_a]["times"]
idx4 = (x4 >= T_MIN) & (x4 <= T_MAX)
x4 = x4[idx4]
y4 = plot_data["case2b"]["ajump"][_a]["sat_frac"][idx4]
yr4 = plot_data["case2b"]["ajump"][_a]["dts"][idx4]
label4 = f"vT(vT)-{_a}"

xs = [x1, x2, x3, x4]
ys = [y1, y2, y3, y4]
yrs = [yr1, yr2, yr3, yr4]
labels = [label1, label2, label3, label4]
colors = ["orange", "red", "purple", "black"]
markers = [8, 9, 10, ""]

Lneg = JUMP_TIME - T_BEFORE_JUMP
Lpos = T_MAX
frac = 0.05

# scale factor for the positive logarithmic part
A = (1 - frac) / np.log10(1 + Lpos / Lneg)


def xtrans(x):
    x = np.asarray(x, dtype=float)

    y = np.empty_like(x)

    neg = x <= 0
    y[neg] = frac * x[neg] / Lneg

    pos = ~neg
    y[pos] = A * np.log10(1 + x[pos] / Lneg)

    return y


for x, y, l, c, m in zip(xs, ys, labels, colors, markers):
    imgs += ax.plot(
        xtrans(x - JUMP_TIME),
        y,
        linestyle="solid",
        linewidth=LW,
        color=c,
        marker=m,
        markersize=1.5 * MS if "pT" in l else MS,
        label=l,
    )

for x, y, l, c, m in zip(xs, yrs, labels, colors, markers):
    imgrs += axr.plot(
        xtrans(x - JUMP_TIME),
        y,
        linestyle="dotted",
        linewidth=LW,
        marker=m,
        markersize=1.5 * MS if "pT" in l else MS,
        color=c,
        label=l + r"-$\Delta t$",
    )

ax.set_xlabel(r"$t - t_{\ast}$ [h]")
ax.set_ylabel(r"$s_{\text{frac}}$")
axr.set_ylabel(r"$\Delta t$")

axr.set_yscale("log")
axr.grid(axis="y", which="major", alpha=0.5)

axr.set_yticks(
    ticks=[10, 3600, 24 * 3600],
    labels=[r"$10$ s", r"$1$ h", r"$1$ d"],
)

ax.set_xticks(
    ticks=xtrans([T_BEFORE_JUMP - JUMP_TIME, 0, 3600, 3 * 3600]),
    labels=[-1, 0, 1, 3],
)
# ax.tick_params(axis='x', labelrotation=45)

# ax.legend(handles=imgs, loc="upper left", bbox_to_anchor=(1.25, 0.8))
ax.legend(handles=imgs, loc="lower right", bbox_to_anchor=(1, 0.1))
# ax.legend(handles=imgs, ncols = 4, loc="lower center", bbox_to_anchor=(0.5, 1))

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}dt_tau_G_isothermal.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Plot pressure drop per aperture jump for case 2b

fig = plt.figure(figsize=(0.5 * WIDTH, 0.5 * WIDTH))
ax = fig.add_subplot(111)
axr = ax.twinx()
imgs = []
imgrs = []

x = []
y = []
yr = []
color_right = "red"

for a, d in plot_data["case2b"]["ajump"].items():
    p_l2 = d["delta_p_l2_transient"][0]
    p_max = d["delta_p_max_transient"][0]
    x.append(a)
    y.append(p_l2)
    yr.append(p_max)

imgs += ax.plot(
    x,
    y,
    linestyle="solid",
    linewidth=LW,
    # marker="*",
    # markersize=MS,
    color="black",
    label=r"$vT(vT)-L_2$",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle="dotted",
    linewidth=LW,
    # marker="*",
    # markersize=MS,
    color=color_right,
    label=r"$vT(vT)-\infty$",
)


ax.set_xlabel(r"$a(t_{\ast})/a(t_{-1})$")
ax.set_ylabel(r"$\lVert p(t_{\ast}) - p(t_{-1})\rVert_{L^2(\Omega)}$")
axr.set_ylabel(r"$\lvert p(t_{\ast}) - p(t_{-1})\rvert_{\infty}$")

ax.set_xticks(x)
ax.grid(axis="x")
ax.grid(axis="y", which="major", alpha=0.5)

axr.tick_params(axis="y", colors=color_right)
axr.spines["right"].set_color(color_right)
axr.yaxis.label.set_color(color_right)
axr.grid(axis="y", which="major", color=color_right, alpha=0.5)

# axr.set_yscale('log')
# ax.set_yscale('log')

# ax.legend(handles=imgs, loc="upper left", bbox_to_anchor=(1.05, 1))
# axr.legend(handles=imgrs, loc="upper left", bbox_to_anchor=(1.05, 1))

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}p_norm_per_a_case_b.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region L2 difference of p after jump during transient.
fig = plt.figure(figsize=(WIDTH, 0.5 * WIDTH))
ax = fig.add_subplot(111)
imgs = []

x_max = []
x_vals = []
y_vals = []
labels = []
colors = []
is_empty = []
for i, ad in enumerate(plot_data["case2b"]["ajump"].items()):
    a, d = ad
    x = d["transient_times"]
    y = d["delta_p_l2_transient"]
    # x_idx = x > (JUMP_TIME - dt_min / 10)
    # idx = (y > frac_tol) & x_idx

    # if np.any(idx):
    #     j = np.where(idx)[0].max()
    #     idx[j + 1] = True
    # x = x[idx] - JUMP_TIME
    # y = y[idx]
    # if np.any(x):
    #     x_max.append(x.max())

    x_vals.append(x - JUMP_TIME)
    x_max.append(np.max(x_vals[-1]))
    y_vals.append(y)
    colors.append(vT_cmap[i])
    labels.append(f"vT(vT)-{a}")

xm = np.max(x_max)
x_factor = 1 / 3600
for x, y, l, c in zip(x_vals, y_vals, labels, colors):
    if not np.any(y):
        x = np.linspace(0, xm, 10, endpoint=True)
        y = np.zeros_like(x)

    idx = x <= xm
    imgs += ax.plot(
        x[idx] * x_factor,
        y[idx],
        linestyle="dotted",
        linewidth=LW,
        # marker="*",
        # markersize=MS,
        color=c,
        label=l,
    )

ax.set_xlabel(r"$t - t_{\ast}$ [h]")
ax.set_ylabel(r"$\lVert p(t \geq t_{\ast}) - p(t_{-1})\rVert_{L^2(\Omega)}$")
ax.set_xlim(0.0, np.max(x_max) * x_factor)

ax.grid(axis="y", alpha=0.5)
ax.set_ylim(1, ax.get_ylim()[1])

ax.set_yscale("log")

# ax.legend(handles=imgs, loc="upper right")
ax.legend(handles=imgs, loc="upper left", bbox_to_anchor=(1.05, 1))

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}p_l2_over_tau_case_b.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region plot pressure and temperature drop per aperture jump for case 2c

# endregion
