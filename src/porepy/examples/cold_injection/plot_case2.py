"""Plot script for case 2.

Used to create plots for publication below.

"""

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from porepy.examples.cold_injection.run_case2a import (
    JUMP_TIME,
    T_BEFORE_JUMP,
    dt_init,
    dt_min,
)

# os.system(f"rm -rf {mpl.get_cachedir()}/tex.cache/*")

# mypy: ignore-errors

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 28

DPI = 400  # Figure resolution.
WIDTH = 10.0  # Figure width.
PAD = 0.05  # Figure pad to borders.
MS = 15  # Marker size.
LW = 3  # line width.
FOLDER = "visualization/"

JUMP_MAX = 3.0

# Fetching data stored in directories.
c2a_dir: str = "visualization/CI_CASE2A/"
c2b_dir: str = "visualization/CI_CASE2B/"
c2c_dir: str = "visualization/CI_CASE2C/"
c2d_dir: str = "visualization/CI_CASE2D/"

c2a_sims = [p.name for p in Path(c2a_dir).iterdir() if p.is_dir()]
c2b_sims = [p.name for p in Path(c2b_dir).iterdir() if p.is_dir()]
c2c_sims = [p.name for p in Path(c2c_dir).iterdir() if p.is_dir()]
c2d_sims = [p.name for p in Path(c2d_dir).iterdir() if p.is_dir()]

print("Found Case 2a dirs:\n" + "\n".join(c2a_sims))
print("Found Case 2b dirs:\n" + "\n".join(c2b_sims))
print("Found Case 2c dirs:\n" + "\n".join(c2c_sims))
print("Found Case 2d dirs:\n" + "\n".join(c2d_sims))


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

    d["success"] = (
        True
        if (
            data["global"]["final_simulation_status"] == "successful"
            and bool(int(data["global"]["final_time_reached"]))
        )
        else False
    )

    d["num_cells"] = sum([int(_) for _ in data["global"]["num_cells"].values()])
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

    dg = data["global"]

    d["transient_times"] = np.zeros((0,))
    d["delta_p_l2_transient"] = np.zeros((0,))
    d["delta_p_max_transient"] = np.zeros((0,))
    d["delta_T_l2_transient"] = np.zeros((0,))
    d["delta_T_max_transient"] = np.zeros((0,))
    d["transient_end_time"] = np.nan
    d["gas_disappears_time"] = np.nan
    d["p_transient_end_time"] = np.nan
    d["T_transient_end_time"] = np.nan

    if "transient_t" in dg:
        d["transient_times"] = np.array(dg["transient_t"])
        d["delta_p_l2_transient"] = np.array(dg["delta_p_l2_transient"])
        d["delta_p_max_transient"] = np.array(dg["delta_p_max_transient"])
        if thermal:
            d["delta_T_l2_transient"] = np.array(dg["delta_T_l2_transient"])
            d["delta_T_max_transient"] = np.array(dg["delta_T_max_transient"])

    if "transient_end_time" in dg:
        d["transient_end_time"] = dg["transient_end_time"]
    if "gas_disappears_time" in dg:
        d["gas_disappears_time"] = dg["gas_disappears_time"]
    if "p_transient_end_time" in dg:
        d["p_transient_end_time"] = dg["p_transient_end_time"]
    if thermal and "T_transient_end_time" in dg:
        d["T_transient_end_time"] = dg["T_transient_end_time"]

    return d


def get_a(data: dict) -> float:
    name: str = data["_subfolder"]
    i = name.find("_AJUMP_")
    j = name.find("_NPC_")
    return float(name[i + 7 : j])


def fetch_all(
    d: str, sims: list[str], ajump: bool = True
) -> tuple[list[dict], dict, dict] | tuple[dict, dict]:
    c_npc = [fetch(d, p) for p in sims if "_NPC_True" in p and "_A_False" in p][0]
    c_no_npc = [fetch(d, p) for p in sims if "_NPC_False" in p and "_A_False" in p][0]

    if ajump:
        c_a = [fetch(d, p) for p in sims if "_NPC_True" in p and "_A_True" in p]
        return c_a, c_npc, c_no_npc
    else:
        return c_npc, c_no_npc


c2a_npc, c2a_no_npc = fetch_all(c2a_dir, c2a_sims, ajump=False)
c2b_ajump, c2b_npc, c2b_no_npc = fetch_all(c2b_dir, c2b_sims)
c2c_ajump, c2c_npc, c2c_no_npc = fetch_all(c2c_dir, c2c_sims)
c2d_ajump, c2d_npc, c2d_no_npc = fetch_all(c2d_dir, c2d_sims)

plot_data = {}

plot_data["case2a"] = {
    "npc": collect_from_raw(c2a_npc),
    "no_npc": collect_from_raw(c2a_no_npc),
}

plot_data["case2b"] = {
    "npc": collect_from_raw(c2b_npc),
    "no_npc": collect_from_raw(c2b_no_npc),
    "ajump": dict([(get_a(data), collect_from_raw(data)) for data in c2b_ajump]),
}

plot_data["case2c"] = {
    "npc": collect_from_raw(c2c_npc),
    "no_npc": collect_from_raw(c2c_no_npc),
    "ajump": dict([(get_a(data), collect_from_raw(data)) for data in c2c_ajump]),
}

plot_data["case2d"] = {
    "npc": collect_from_raw(c2d_npc),
    "no_npc": collect_from_raw(c2d_no_npc),
    "ajump": dict([(get_a(data), collect_from_raw(data)) for data in c2d_ajump]),
}

plot_data["case2b"]["ajump"] = dict(sorted(plot_data["case2b"]["ajump"].items()))
plot_data["case2c"]["ajump"] = dict(sorted(plot_data["case2c"]["ajump"].items()))
plot_data["case2d"]["ajump"] = dict(sorted(plot_data["case2d"]["ajump"].items()))

# Check success of simulations
print("Simulation success:")
print("\tCase 2a:")
print("\t\tNPC: ", plot_data["case2a"]["npc"]["success"])
print("\t\tNo NPC: ", plot_data["case2a"]["no_npc"]["success"])

print("\tCase 2b:")
print("\t\tNPC: ", plot_data["case2b"]["npc"]["success"])
print("\t\tNo NPC: ", plot_data["case2b"]["no_npc"]["success"])
for a, data in plot_data["case2b"]["ajump"].items():
    print(f"\t\tJump {a}: ", data["success"])

print("\tCase 2c:")
print("\t\tNPC: ", plot_data["case2c"]["npc"]["success"])
print("\t\tNo NPC: ", plot_data["case2c"]["no_npc"]["success"])
for a, data in plot_data["case2c"]["ajump"].items():
    print(f"\t\tJump {a}: ", data["success"])

print("\tCase 2d:")
print("\t\tNPC: ", plot_data["case2d"]["npc"]["success"])
print("\t\tNo NPC: ", plot_data["case2d"]["no_npc"]["success"])
for a, data in plot_data["case2d"]["ajump"].items():
    print(f"\t\tJump {a}: ", data["success"])

# endregion

frac_tol = 1e-10  # Below this, gas is considered absent.
vT_cmap = plt.colormaps["bone_r"](
    np.linspace(0.1, 1, len(plot_data["case2b"]["ajump"]), endpoint=True)
)
uv_cmap = plt.colormaps["autumn_r"](
    np.linspace(0.1, 1, len(plot_data["case2c"]["ajump"]), endpoint=True)
)
ph_cmap = plt.colormaps["cool_r"](
    np.linspace(0, 1, len(plot_data["case2d"]["ajump"]), endpoint=True)
)


def format_nums(nums):
    return "[" + ", ".join(f"{x:.2f}" for x in nums) + "]"


T_MIN = T_BEFORE_JUMP - dt_min
T_MAX = JUMP_TIME + 4 * 3600


def get_t_s_dt(data: dict) -> tuple[np.ndarray, np.ndarray]:
    x_ = data["times"]
    idx_ = (x_ >= T_MIN) & (x_ <= T_MAX)
    x_ = x_[idx_]
    y_ = data["sat_frac"][idx_]
    yr_ = data["dts"][idx_]
    return x_, y_, yr_


# region Time steps and gas transients for isothermal cases.
fig = plt.figure(figsize=(1.5 * WIDTH, 0.7 * WIDTH))
ax = fig.add_subplot(111)
axr = ax.twinx()
imgs = []
imgrs = []

xs = []
ys = []
yrs = []
markers = [4, 5, 6, 7]
markers_dt = ["1", "2", "3", "4"]
colors = ["gray", "black", "blue", "cyan"]
labels = [
    rf"i-vT-{JUMP_MAX}",
    rf"i-vT(vT)-{JUMP_MAX}",
    rf"i-pT-{JUMP_MAX}",
    rf"i-pT(vT)-{JUMP_MAX}",
]

xmax = []
xmin = []
for data in [
    plot_data["case2b"]["no_npc"],
    plot_data["case2b"]["npc"],
    plot_data["case2a"]["no_npc"],
    plot_data["case2a"]["npc"],
]:
    x_, y_, yr_ = get_t_s_dt(data)
    xs.append(x_)
    ys.append(y_)
    yrs.append(yr_)
    if x_.size > 2:
        xmax.append(x_.max())
        xmin.append(x_.min())

print("dT star:")
for x, y, c in zip(
    xs, yrs, ["Case 2b no npc", "Case 2b npc", "Case 2a no npc", "Case 2a npc"]
):
    idx = x <= JUMP_TIME + 1
    if np.any(idx):
        print(f"\t{c}: {y[idx].min():.2f}")
    else:
        print(f"\t{c}: {y.min():.2f}")

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


lo = 0
for x, y, l, c, m in zip(xs, yrs, labels, colors, markers_dt):
    kwargs = dict(
        linestyle=(lo, (1, 4)),
        linewidth=LW,
    )

    if x.size < 2:
        x_ = np.linspace(min(xmin), max(xmax), 10)
        y_ = np.ones_like(x_) * y[0]
    else:
        x_ = x
        y_ = y
        kwargs["marker"] = m
        kwargs["markersize"] = 2 * MS
        kwargs["markevery"] = 0.1

    imgrs += axr.plot(
        xtrans(x_ - JUMP_TIME),
        y_,
        color=c,
        label=l,
        **kwargs,
    )
    lo += 1

lo = 0
for x, y, l, c, m in zip(xs, ys, labels, colors, markers):
    kwargs = dict(
        linestyle=(lo, (5, 10)),
        linewidth=LW,
    )

    if x.size < 2:
        x_ = np.linspace(min(xmin), max(xmax), 10)
        y_ = np.ones_like(x_) * y[0]
    else:
        x_ = x
        y_ = y
        kwargs["marker"] = m
        kwargs["markersize"] = MS
        kwargs["markevery"] = 0.1

    imgs += ax.plot(
        xtrans(x_ - JUMP_TIME),
        y_,
        color=c,
        label=l,
        **kwargs,
    )
    lo += 2

ax.set_xlabel(r"$t - t_{\ast}$ [h]")
ax.set_ylabel(r"$s_{\text{frac}}$")
axr.set_ylabel(r"$\Delta t$")

axr.set_yscale("log")
axr.grid(axis="y", which="major", alpha=0.5)

axr.set_yticks(
    ticks=[10, 60, 3600, 24 * 3600],
    labels=[r"$10$ s", r"$1$ min", r"$1$ h", r"$1$ d"],
)
ax.set_xticks(
    ticks=xtrans([T_BEFORE_JUMP - JUMP_TIME, 0, 3600, 3 * 3600]),
    labels=[-1, 0, 1, 3],
)

ax.legend(
    handles=imgs,
    loc="upper left",
    bbox_to_anchor=(1.2, 1),
    title=r"$s_{\text{frac}}$",
)
axr.legend(
    handles=imgrs,
    loc="upper left",
    bbox_to_anchor=(1.2, 0.4),
    title=r"$\Delta t$",
)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}dt_tau_g_case_ab.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Time steps and gas transients for thermal cases.
fig = plt.figure(figsize=(1.5 * WIDTH, 0.7 * WIDTH))
ax = fig.add_subplot(111)
axr = ax.twinx()
imgs = []
imgrs = []

xs = []
ys = []
yrs = []
markers = [4, 5, 6, 7]
markers_dt = ["1", "2", "3", "4"]
colors = ["gray", "black", "blue", "cyan"]
labels = [
    rf"vT-{JUMP_MAX}",
    rf"vT(uv)-{JUMP_MAX}",
    rf"ph-{JUMP_MAX}",
    rf"ph(uv)-{JUMP_MAX}",
]

xmax = []
xmin = []
for data in [
    plot_data["case2c"]["no_npc"],
    plot_data["case2c"]["npc"],
    plot_data["case2d"]["no_npc"],
    plot_data["case2d"]["npc"],
]:
    x_, y_, yr_ = get_t_s_dt(data)
    xs.append(x_)
    ys.append(y_)
    yrs.append(yr_)
    if x_.size > 2:
        xmax.append(x_.max())
        xmin.append(x_.min())

print("dT star:")
for x, y, c in zip(
    xs, yrs, ["Case 2c no npc", "Case 2c npc", "Case 2d no npc", "Case 2d npc"]
):
    idx = x <= JUMP_TIME + 1
    if np.any(idx):
        print(f"\t{c}: {y[idx].min():.2f}")
    else:
        print(f"\t{c}: {y.min():.2f}")

lo = 0
for x, y, l, c, m in zip(xs, yrs, labels, colors, markers_dt):
    kwargs = dict(
        linestyle=(lo, (1, 4)),
        linewidth=LW,
    )

    if x.size < 2:
        x_ = np.linspace(min(xmin), max(xmax), 10)
        y_ = np.ones_like(x_) * y[0]
    else:
        x_ = x
        y_ = y
        kwargs["marker"] = m
        kwargs["markersize"] = 2 * MS
        kwargs["markevery"] = 0.1

    imgrs += axr.plot(
        xtrans(x_ - JUMP_TIME),
        y_,
        color=c,
        label=l,
        **kwargs,
    )
    lo += 1

lo = 0
for x, y, l, c, m in zip(xs, ys, labels, colors, markers):
    kwargs = dict(
        linestyle=(lo, (5, 10)),
        linewidth=LW,
    )

    if x.size < 2:
        x_ = np.linspace(min(xmin), max(xmax), 10)
        y_ = np.ones_like(x_) * y[0]
    else:
        x_ = x
        y_ = y
        kwargs["marker"] = m
        kwargs["markersize"] = MS
        kwargs["markevery"] = 0.1

    imgs += ax.plot(
        xtrans(x_ - JUMP_TIME),
        y_,
        color=c,
        label=l,
        **kwargs,
    )
    lo += 2

ax.set_xlabel(r"$t - t_{\ast}$ [h]")
ax.set_ylabel(r"$s_{\text{frac}}$")
axr.set_ylabel(r"$\Delta t$")

axr.set_yscale("log")
axr.grid(axis="y", which="major", alpha=0.5)

axr.set_yticks(
    ticks=[10, 60, 3600, 24 * 3600],
    labels=[r"$10$ s", r"$1$ min", r"$1$ h", r"$1$ d"],
)
ax.set_xticks(
    ticks=xtrans([T_BEFORE_JUMP - JUMP_TIME, 0, 3600, 3 * 3600]),
    labels=[-1, 0, 1, 3],
)

ax.legend(
    handles=imgs,
    loc="upper left",
    bbox_to_anchor=(1.2, 1),
    title=r"$s_{\text{frac}}$",
)
axr.legend(
    handles=imgrs,
    loc="upper left",
    bbox_to_anchor=(1.2, 0.4),
    title=r"$\Delta t$",
)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}dt_tau_g_case_cd.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Transient duration per aperture for cases bcd
fig = plt.figure(figsize=(1.1 * WIDTH, 0.5 * WIDTH))
ax = fig.add_subplot(111)
axr = ax.twinx()
imgs = []
imgrs = []

x = []
y = []
yr = []

for a, d in plot_data["case2b"]["ajump"].items():
    x.append(a)
    y.append((d["p_transient_end_time"] - JUMP_TIME) / 3600)
    yr.append((d["gas_disappears_time"] - JUMP_TIME) / 60)

imgs += ax.plot(
    x,
    y,
    linestyle="dashed",
    linewidth=LW,
    marker="P",
    markersize=MS,
    color="black",
    label=r"i-vT(vT)",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle="dotted",
    linewidth=LW,
    marker="+",
    markersize=MS,
    color="black",
    label=r"i-vT(vT)",
)

print(f"Transient durations per jump: {format_nums(x)}")
print("\tCase i-vT(vT):")
print(f"\t\ttau_p: {format_nums(y)}")
print(f"\t\ttau_g: {format_nums(yr)}")

x = []
y = []
yr = []

for a, d in plot_data["case2c"]["ajump"].items():
    x.append(a)
    y.append((d["p_transient_end_time"] - JUMP_TIME) / 3600)
    yr.append((d["gas_disappears_time"] - JUMP_TIME) / 60)

imgs += ax.plot(
    x,
    y,
    linestyle="dashed",
    linewidth=LW,
    marker="X",
    markersize=MS,
    color="red",
    label=r"vT(uv)",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle="dotted",
    linewidth=LW,
    marker="x",
    markersize=MS,
    color="red",
    label=r"vT(uv)",
)

print("\tCase vT(uv):")
print(f"\t\ttau_p: {format_nums(y)}")
print(f"\t\ttau_g: {format_nums(yr)}")

x = []
y = []
yr = []

for a, d in plot_data["case2d"]["ajump"].items():
    x.append(a)
    y.append((d["p_transient_end_time"] - JUMP_TIME) / 3600)
    yr.append((d["gas_disappears_time"] - JUMP_TIME) / 60)

imgs += ax.plot(
    x,
    y,
    linestyle=(0, (5, 10)),
    linewidth=LW,
    marker=".",
    markersize=MS,
    color="purple",
    label=r"ph(uv)",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle=(2, (1, 5)),
    linewidth=LW,
    marker="o",
    markerfacecolor="none",
    markeredgecolor="purple",
    markersize=MS,
    color="purple",
    label=r"ph(uv)",
)

print("\tCase ph(uv):")
print(f"\t\ttau_p: {format_nums(y)}")
print(f"\t\ttau_g: {format_nums(yr)}")

ax.set_xlabel(r"$a(t_{\ast})/a(t_{-1})$")
ax.set_ylabel(r"$\tau_p$")
axr.set_ylabel(r"$\tau_g$")

ax.set_xticks(x)
ax.grid(axis="x")
ax.grid(axis="y", which="major", alpha=0.5)
axr.grid(axis="y", which="major", alpha=0.5, linestyle="dashed")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: rf"${y:.2f}$"))
axr.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: rf"${y:.1f}$"))

ax.legend(
    handles=imgs,
    loc="upper left",
    bbox_to_anchor=(1.25, 1.0),
    title=r"$\tau_p$ [h]",
    fontsize=24,
)
axr.legend(
    handles=imgrs,
    loc="upper left",
    bbox_to_anchor=(1.25, 0.4),
    title=r"$\tau_g$ [min]",
    fontsize=24,
)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}tau_per_a_case_bcd.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Pressure drop per aperture jump for cases bcd

fig = plt.figure(figsize=(1.1 * WIDTH, 0.5 * WIDTH))
ax = fig.add_subplot(111)
axr = ax.twinx()
imgs = []
imgrs = []

x = []
y = []
yr = []

for a, d in plot_data["case2b"]["ajump"].items():
    p_l2 = d["delta_p_l2_transient"][0]
    p_max = d["delta_p_max_transient"][0]
    x.append(a)
    y.append(p_l2)
    yr.append(p_max)

imgs += ax.plot(
    x,
    y,
    linestyle="dashed",
    linewidth=LW,
    marker="P",
    markersize=MS,
    color="black",
    label=r"i-vT(vT)",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle="dotted",
    linewidth=LW,
    marker="+",
    markersize=MS,
    color="black",
    label=r"i-vT(vT)",
)

print(f"Pressure drop per jump: {format_nums(x)}")
print("\tCase i-vT(vT):")
print(f"\t\tL2: {format_nums(y)}")
print(f"\t\tMax: {format_nums(yr)}")

x = []
y = []
yr = []

for a, d in plot_data["case2c"]["ajump"].items():
    p_l2 = d["delta_p_l2_transient"][0]
    p_max = d["delta_p_max_transient"][0]
    x.append(a)
    y.append(p_l2)
    yr.append(p_max)

imgs += ax.plot(
    x,
    y,
    linestyle="dashed",
    linewidth=LW,
    marker="X",
    markersize=MS,
    color="red",
    label=r"vT(uv)",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle="dotted",
    linewidth=LW,
    marker="x",
    markersize=MS,
    color="red",
    label=r"vT(uv)",
)

print("\tCase vT(uv):")
print(f"\t\tL2: {format_nums(y)}")
print(f"\t\tMax: {format_nums(yr)}")

x = []
y = []
yr = []

for a, d in plot_data["case2d"]["ajump"].items():
    p_l2 = d["delta_p_l2_transient"][0]
    p_max = d["delta_p_max_transient"][0]
    x.append(a)
    y.append(p_l2)
    yr.append(p_max)

imgs += ax.plot(
    x,
    y,
    linestyle=(0, (5, 10)),
    linewidth=LW,
    marker=".",
    markersize=MS,
    color="purple",
    label=r"ph(uv)",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle=(2, (1, 5)),
    linewidth=LW,
    marker="o",
    markerfacecolor="none",
    markeredgecolor="purple",
    markersize=MS,
    color="purple",
    label=r"ph(uv)",
)

print("\tCase ph(uv):")
print(f"\t\tL2: {format_nums(y)}")
print(f"\t\tMax: {format_nums(yr)}")

ax.set_xlabel(r"$a(t_{\ast})/a(t_{-1})$")
ax.set_ylabel(r"$\lVert p(t_{\ast}) - p(t_{-1})\rVert_{L^2(\Omega)}$")
axr.set_ylabel(r"$\lvert p(t_{\ast}) - p(t_{-1})\rvert_{\infty}$")

ax.set_xticks(x)
ax.grid(axis="x")
ax.grid(axis="y", which="major", alpha=0.5)
axr.grid(axis="y", which="major", alpha=0.5, linestyle="dashed")

ax.legend(
    handles=imgs,
    loc="upper left",
    bbox_to_anchor=(1.25, 1.0),
    title=r"$\lVert\cdot\rVert_{L^2(\Omega)}$",
    fontsize=24,
)
axr.legend(
    handles=imgrs,
    loc="upper left",
    bbox_to_anchor=(1.25, 0.4),
    title=r"$\lvert\cdot\rvert_{\infty}$ [MPa]",
    fontsize=24,
)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}p_drop_per_a_case_bcd.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Temperature drop per aperture jump for cases cd

fig = plt.figure(figsize=(0.7 * WIDTH, 0.5 * WIDTH))
ax = fig.add_subplot(111)
axr = ax.twinx()
imgs = []
imgrs = []

x = []
y = []
yr = []

for a, d in plot_data["case2c"]["ajump"].items():
    p_l2 = d["delta_T_l2_transient"][0]
    p_max = d["delta_T_max_transient"][0]
    x.append(a)
    y.append(p_l2)
    yr.append(p_max)

imgs += ax.plot(
    x,
    y,
    linestyle="dashed",
    linewidth=LW,
    marker="X",
    markersize=MS,
    color="red",
    label=r"vT(uv)",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle="dotted",
    linewidth=LW,
    marker="x",
    markersize=MS,
    color="red",
    label=r"vT(uv)",
)

print(f"Temperature drops per jump: {format_nums(x)}")
print("\tCase vT(uv):")
print(f"\t\tL2: {format_nums(y)}")
print(f"\t\tMax: {format_nums(yr)}")

x = []
y = []
yr = []

for a, d in plot_data["case2d"]["ajump"].items():
    p_l2 = d["delta_T_l2_transient"][0]
    p_max = d["delta_T_max_transient"][0]
    x.append(a)
    y.append(p_l2)
    yr.append(p_max)

imgs += ax.plot(
    x,
    y,
    linestyle=(0, (5, 10)),
    linewidth=LW,
    marker=".",
    # markerfacecolor="none",
    # markeredgecolor="purple",
    markersize=MS,
    color="purple",
    label=r"ph(uv)",
)

imgrs += axr.plot(
    x,
    yr,
    linestyle=(2, (1, 5)),
    linewidth=LW,
    marker="o",
    markerfacecolor="none",
    markeredgecolor="purple",
    markersize=MS,
    color="purple",
    label=r"ph(uv)",
)
print("\tCase ph(uv):")
print(f"\t\tL2: {format_nums(y)}")
print(f"\t\tMax: {format_nums(yr)}")

ax.set_xlabel(r"$a(t_{\ast})/a(t_{-1})$")
ax.set_ylabel(r"$\lVert T(t_{\ast}) - T(t_{-1})\rVert_{L^2(\Omega)}$")
axr.set_ylabel(r"$\lvert T(t_{\ast}) - T(t_{-1})\rvert_{\infty}$")

ax.set_xticks(x)
ax.grid(axis="x")
ax.grid(axis="y", which="major", alpha=0.5)
axr.grid(axis="y", which="major", alpha=0.5, linestyle="dashed")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))
axr.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.1f}"))

ax.legend(
    handles=imgs,
    loc="upper left",
    bbox_to_anchor=(-0.02, 1.02),
    title=r"$\lVert\cdot\rVert_{L^2(\Omega)}$",
    fontsize=24,
)
axr.legend(
    handles=imgrs,
    loc="lower right",
    bbox_to_anchor=(1.02, -0.02),
    title=r"$\lvert\cdot\rvert_{\infty}$ [K]",
    fontsize=24,
)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}T_drop_per_a_case_cd.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Gas content during transient for cases bcd
fig = plt.figure(figsize=(1.5 * WIDTH, 0.7 * WIDTH))
ax = fig.add_subplot(111)
imgs = []

x_max = []
x_vals = []
y_vals = []
labels = []
colors = []

print("Gas content after jump:")
print("\tCase i-vT(vT):")
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
    labels.append(rf"i-vT(vT)-{a}")
    if y.size > 0:
        print(f"\t\ta={a}: {y[0] * 100:.2f}%")
    else:
        print(f"\t\ta={a}: 0.00%")

print("\tCase vT(uv):")
for i, ad in enumerate(plot_data["case2c"]["ajump"].items()):
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
    colors.append(uv_cmap[i])
    labels.append(rf"vT(uv)-{a}")
    if y.size > 0:
        print(f"\t\ta={a}: {y[0] * 100:.2f}%")
    else:
        print(f"\t\ta={a}: 0.00%")

print("\tCase ph(uv):")
for i, ad in enumerate(plot_data["case2d"]["ajump"].items()):
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
    colors.append(ph_cmap[i])
    labels.append(rf"ph(uv)-{a}")
    if y.size > 0:
        print(f"\t\ta={a}: {y[0] * 100:.2f}%")
    else:
        print(f"\t\ta={a}: 0.00%")

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
        linestyle=(0, (5, 8)) if "ph" in l else "solid",
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
ax.legend(
    handles=imgs,
    loc="upper left",
    bbox_to_anchor=(0.3, 1.0),
    ncols=3,
    fontsize=24,
)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}sat_frac_over_tau_case_bcd.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region p-L2 during transient for cases bcd.
fig = plt.figure(figsize=(1.5 * WIDTH, 0.7 * WIDTH))
ax = fig.add_subplot(111)
imgs = []

x_max = []
x_vals = []
y_vals = []
labels = []
colors = []

for i, ad in enumerate(plot_data["case2b"]["ajump"].items()):
    a, d = ad
    x = d["transient_times"]
    y = d["delta_p_l2_transient"]

    x_vals.append(x - JUMP_TIME)
    x_max.append(np.max(x_vals[-1]))
    y_vals.append(y)
    colors.append(vT_cmap[i])
    labels.append(rf"i-vT(vT)-{a}")

for i, ad in enumerate(plot_data["case2c"]["ajump"].items()):
    a, d = ad
    x = d["transient_times"]
    y = d["delta_p_l2_transient"]

    x_vals.append(x - JUMP_TIME)
    x_max.append(np.max(x_vals[-1]))
    y_vals.append(y)
    colors.append(uv_cmap[i])
    labels.append(rf"vT(uv)-{a}")

for i, ad in enumerate(plot_data["case2d"]["ajump"].items()):
    a, d = ad
    x = d["transient_times"]
    y = d["delta_p_l2_transient"]

    x_vals.append(x - JUMP_TIME)
    x_max.append(np.max(x_vals[-1]))
    y_vals.append(y)
    colors.append(ph_cmap[i])
    labels.append(rf"ph(uv)-{a}")

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
        linestyle=(0, (5, 8)) if "ph" in l else "solid",
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

leg_1 = ax.legend(
    handles=imgs[:5],
    loc="upper left",
    bbox_to_anchor=(0.76, 1.02),
    ncols=1,
    fontsize=22,
)
ax.add_artist(leg_1)
leg_2 = ax.legend(
    handles=imgs[5:],
    loc="upper left",
    bbox_to_anchor=(0.99, 1.02),
    ncols=1,
    fontsize=22,
)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}p_l2_over_tau_case_bcd.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Plot clock times in bar chart
fig = plt.figure(figsize=(1.5 * WIDTH, 0.7 * WIDTH))
ax = fig.add_subplot(111)
imgs = []

# restructure data
assembly = np.array(
    [
        [d["time_in_assembly"] for d in plot_data["case2b"]["ajump"].values()],
        [d["time_in_assembly"] for d in plot_data["case2c"]["ajump"].values()],
        [d["time_in_assembly"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
linsolve = np.array(
    [
        [d["time_in_linsolve"] for d in plot_data["case2b"]["ajump"].values()],
        [d["time_in_linsolve"] for d in plot_data["case2c"]["ajump"].values()],
        [d["time_in_linsolve"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
flash = np.array(
    [
        [d["time_in_flash"] for d in plot_data["case2b"]["ajump"].values()],
        [d["time_in_flash"] for d in plot_data["case2c"]["ajump"].values()],
        [d["time_in_flash"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
trustreg = np.array(
    [
        [d["time_in_tr"] for d in plot_data["case2b"]["ajump"].values()],
        [d["time_in_tr"] for d in plot_data["case2c"]["ajump"].values()],
        [d["time_in_tr"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
REF: float = assembly[0, 0] + linsolve[0, 0] + flash[0, 0] + trustreg[0, 0]

assembly /= REF
linsolve /= REF
flash /= REF
trustreg /= REF

n_models, n_params = assembly.shape

group_width = 0.8
bar_width = group_width / n_params
bar_spacing = 0.9

centers = np.arange(n_models)
models = ["i-vT(vT)", "vT(uv)", "ph(uv)"]
colors = {
    "assembly": "tab:blue",
    "linsolve": "tab:green",
    "flash": "tab:orange",
    "tr": "tab:purple",
}

for j in range(n_params):
    x = centers - group_width / 2 + (j + 0.5) * bar_width

    ax.bar(
        x,
        assembly[:, j],
        width=bar_width * bar_spacing,
        label="Assembly" if j == 0 else None,
        color=colors["assembly"],
    )

    ax.bar(
        x,
        linsolve[:, j],
        width=bar_width * bar_spacing,
        bottom=assembly[:, j],
        label="Linear solve" if j == 0 else None,
        color=colors["linsolve"],
    )

    ax.bar(
        x,
        flash[:, j],
        width=bar_width * bar_spacing,
        bottom=assembly[:, j] + linsolve[:, j],
        label="Flash" if j == 0 else None,
        color=colors["flash"],
    )

    ax.bar(
        x,
        trustreg[:, j],
        width=bar_width * bar_spacing,
        bottom=assembly[:, j] + linsolve[:, j] + flash[:, j],
        label="Trust region" if j == 0 else None,
        color=colors["tr"],
    )
bar_positions = []
bar_labels = []
params = list(plot_data["case2b"]["ajump"].keys())

for i in range(n_models):
    for j in range(n_params):
        x = centers[i] - group_width / 2 + (j + 0.5) * bar_width
        bar_positions.append(x)
        bar_labels.append(f"{params[j]}")

ax.set_xticks(ticks=bar_positions, labels=bar_labels)
for x, model in zip(centers, models):
    ax.text(
        x,
        -0.08,
        model,
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
    )

ax.set_ylabel("Normalized wall-clock time")
ax.axhline(1.0, color="black", ls="--", lw=LW)
ax.text(
    centers[0] - group_width / 2 + 0.5 * bar_width,
    1.1,
    f"{REF:.2f} s",
    va="bottom",
    ha="center",
)
y_ticks = ax.get_yticks().astype(int).tolist()
if 1 not in y_ticks:
    y_ticks.append(1)
ax.set_yticks(np.sort(np.array(y_ticks)))

ax.legend(loc="upper center", ncol=4)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}clocktimes.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
# region Wasted iterations
fig = plt.figure(figsize=(1.5 * WIDTH, 0.7 * WIDTH))
ax = fig.add_subplot(111)
axr = ax.twinx()
imgs = []

# restructure data
total_ts = np.array(
    [
        [d["total_time_steps"] for d in plot_data["case2b"]["ajump"].values()],
        [d["total_time_steps"] for d in plot_data["case2c"]["ajump"].values()],
        [d["total_time_steps"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
total_gi = np.array(
    [
        [d["total_global_iter"] for d in plot_data["case2b"]["ajump"].values()],
        [d["total_global_iter"] for d in plot_data["case2c"]["ajump"].values()],
        [d["total_global_iter"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
total_fi = np.array(
    [
        [d["total_flash_iter"] for d in plot_data["case2b"]["ajump"].values()],
        [d["total_flash_iter"] for d in plot_data["case2c"]["ajump"].values()],
        [d["total_flash_iter"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
total_tri = np.array(
    [
        [d["total_tr_iter"] for d in plot_data["case2b"]["ajump"].values()],
        [d["total_tr_iter"] for d in plot_data["case2c"]["ajump"].values()],
        [d["total_tr_iter"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)

wasted_ts = np.array(
    [
        [d["wasted_time_steps"] for d in plot_data["case2b"]["ajump"].values()],
        [d["wasted_time_steps"] for d in plot_data["case2c"]["ajump"].values()],
        [d["wasted_time_steps"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
wasted_gi = np.array(
    [
        [d["wasted_global_iter"] for d in plot_data["case2b"]["ajump"].values()],
        [d["wasted_global_iter"] for d in plot_data["case2c"]["ajump"].values()],
        [d["wasted_global_iter"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
wasted_fi = np.array(
    [
        [d["wasted_flash_iter"] for d in plot_data["case2b"]["ajump"].values()],
        [d["wasted_flash_iter"] for d in plot_data["case2c"]["ajump"].values()],
        [d["wasted_flash_iter"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)
wasted_tri = np.array(
    [
        [d["wasted_tr_iter"] for d in plot_data["case2b"]["ajump"].values()],
        [d["wasted_tr_iter"] for d in plot_data["case2c"]["ajump"].values()],
        [d["wasted_tr_iter"] for d in plot_data["case2d"]["ajump"].values()],
    ]
)

eff_ts = (1 - wasted_ts / total_ts) * 100
eff_gi = (1 - wasted_gi / total_gi) * 100
eff_fi = (1 - wasted_fi / total_fi) * 100
eff_tri = (1 - wasted_tri / total_tri) * 100

accepted_ts = total_ts - wasted_ts
accepted_gi = total_gi - wasted_gi
accepted_fi = total_fi - wasted_fi
accepted_tri = total_tri - wasted_tri

cost_ratio_fi = (wasted_fi / wasted_gi) / (accepted_fi / accepted_gi)
cost_ratio_tri = (wasted_tri / wasted_gi) / (accepted_tri / accepted_gi)
# cost_ratio_fi[np.isnan(cost_ratio_fi)] = 0
# cost_ratio_tri[np.isnan(cost_ratio_tri)] = 0

n_models, n_params = total_ts.shape

group_width = 1.0
bar_width = group_width / n_params
bar_spacing = 1.0

centers = np.arange(n_models)
models = ["i-vT(vT)", "vT(uv)", "ph(uv)"]
colors = {
    "ts": "tab:blue",
    "gi": "tab:green",
    "fi": "tab:orange",
    "tri": "tab:purple",
}
param_points = np.arange(n_params)

for j, c in enumerate(centers):
    x = c - group_width / 2 + (param_points + 0.5) * bar_width

    ax.plot(
        x,
        eff_ts[j],
        linewidth=LW,
        linestyle="solid",
        color=colors["ts"],
        label="Time stepping" if j == 0 else None,
    )
    ax.plot(
        x,
        eff_gi[j],
        linewidth=LW,
        linestyle="solid",
        color=colors["gi"],
        label="Nonlinear" if j == 0 else None,
    )
    ax.plot(
        x,
        eff_fi[j],
        linewidth=LW,
        linestyle="solid",
        color=colors["fi"],
        label="Flash" if j == 0 else None,
    )
    ax.plot(
        x,
        eff_tri[j],
        linewidth=LW,
        linestyle="solid",
        color=colors["tri"],
        label="Trust region" if j == 0 else None,
    )

    axr.plot(
        x,
        cost_ratio_fi[j],
        linewidth=LW,
        linestyle="dashed",
        color=colors["fi"],
        label="Flash" if j == 0 else None,
    )
    axr.plot(
        x,
        cost_ratio_tri[j],
        linewidth=LW,
        linestyle="dashed",
        color=colors["tri"],
        label="Trust region" if j == 0 else None,
    )

bar_positions = []
bar_labels = []
params = list(plot_data["case2b"]["ajump"].keys())

for i in range(n_models):
    for j in range(n_params):
        x = centers[i] - group_width / 2 + (j + 0.5) * bar_width
        bar_positions.append(x)
        bar_labels.append(f"{params[j]}")

ax.set_xticks(ticks=bar_positions, labels=bar_labels)
for x, model in zip(centers, models):
    ax.text(
        x,
        -0.08,
        model,
        ha="center",
        va="top",
        transform=ax.get_xaxis_transform(),
        # fontsize=plt.rcParams['font.size'],
    )

ax.set_ylabel(r"\% usefull iterations")
axr.set_ylabel(r"work rejection / work acceptance")

axr.grid(axis="y", ls="dashed", alpha=0.5, color="grey")
ax.grid(axis="x", ls="solid", color="grey", alpha=0.5)

ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1), title="Contributing iterations")
axr.legend(
    loc="upper left",
    bbox_to_anchor=(1.1, 0.3),
    title="Work ratio",
)

fig.tight_layout(pad=PAD)
fig.savefig(
    f"{FOLDER}iterations.png",
    format="png",
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.01,
)
# endregion
