from __future__ import annotations
from typing import Sequence
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

import argparse
import numpy as np
import pandas as pd

import os

os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler


def compute_production_rate(
    p_cell: np.ndarray,
    z_NaCl: np.ndarray,
    h_cell: np.ndarray,
    s_halite: np.ndarray,
    s_vapour: np.ndarray,
    p_BHP: float,
    vtk_sampler: VTKSampler,
    K0: float = 1.0e-15,
    phi0: float = 0.1,
    r_e: float = 0.2,
    r_w: float = 0.1,
    h: float = 1.0,
    skin: float = 0.0,
) -> float:
    """
    Peaceman production rate: q = -λ · WI · (p_cell - p_BHP)

    Parameters
    ----------
    p_cell : Cell pressure at the well block [Pa]
    s_halite : Halite saturation at the well block [-]
    s_vapour : Vapour saturation at the well block [-]
    p_BHP : Bottom-hole pressure [Pa]
    K0 : Reference (intrinsic) permeability [m^2]
    phi0 : Reference porosity [-]
    r_e : Peaceman equivalent radius [m]
    r_w : Wellbore radius [m]
    h : Well cell thickness [m]
    vtk_sampler : VTKSampler instance (ptz space)
    z_NaCl : Overall NaCl mass fraction at the well cell [-]
    h_cell : Enthalpy at the well cell [K]
    skin : Skin factor [-]

    Returns
    -------
    q_prod : Production mass rate [kg/s], negative = outflow
    """

    # ---------------------------------------------------------
    # 1. Halite-corrected permeability: K = K0 * (1 - s_h)^2
    #    (Kozeny-Carman, Eq. 17 in paper)
    # ---------------------------------------------------------
    K = K0 * (1.0 - s_halite) ** 2.0

    # ---------------------------------------------------------
    # 2. Well index (Peaceman, Eq. 42)
    #    WI = 2π h K / (ln(r_e / r_w) + s)
    # ---------------------------------------------------------
    WI = 2.0 * np.pi * h * K / (np.log(r_e / r_w) + skin)

    # ---------------------------------------------------------
    # 3. Total mass mobility: λ = Σ_γ (ρ^γ · kr_γ / μ^γ)
    #    over mobile phases only (halite excluded)
    #    Sample phase properties from VTK table
    # ---------------------------------------------------------
    par_point = np.array((z_NaCl, h_cell, p_cell)).T
    vtk_sampler.sample_at(par_point)
    data = vtk_sampler.sampled_cloud.point_data

    # Phase saturations
    # s_hal = data["S_h"]
    # s_vap = data["S_v"]
    s_hal = s_halite
    s_vap = s_vapour
    s_liq = 1 - (s_hal + s_vap)

    # Phase densities
    rho_liq = np.asarray(data["Rho_l"])
    rho_vap = np.asarray(data["Rho_v"])

    # Phase viscosities
    mu_liq = np.asarray(data["mu_l"])
    mu_vap = np.asarray(data["mu_v"])

    # Relative permeabilities (Corey-type with halite correction, Eq. 39)
    R_liq = 0.3
    R_vap = 0.0
    mobile_pore = 1.0  # - s_halite
    s_liq_res = R_liq * mobile_pore
    s_vap_res = R_vap * mobile_pore
    denom = mobile_pore - s_liq_res - s_vap_res

    kr_liq = np.where(denom > 0, np.maximum((s_liq - s_liq_res) / denom, 0.0), 0.0)
    kr_vap = np.where(denom > 0, np.maximum((s_vap - s_vap_res) / denom, 0.0), 0.0)

    # Total mass mobility
    # Total mass mobility: λ = Σ_γ (ρ^γ · kr_γ / μ^γ)
    lambda_total = np.zeros_like(p_cell)
    mask_liq = (mu_liq > 0) & (kr_liq >= 0)
    mask_vap = (mu_vap > 0) & (kr_vap >= 0)

    # Apply Corey exponents consistently with the model
    kr_liq_corey = kr_liq**1.5
    kr_vap_corey = kr_vap  # linear

    lambda_total = np.where(mask_liq, rho_liq * kr_liq_corey / mu_liq, 0.0) + np.where(
        mask_vap, rho_vap * kr_vap_corey / mu_vap, 0.0
    )

    # ---------------------------------------------------------
    # 4. Production rate (Eq. 41)
    #    q = -λ · WI · (p - p_BHP)
    # ---------------------------------------------------------
    q_prod = -lambda_total * WI * (p_cell - p_BHP)

    return q_prod


"""Plot production diagnostics and Figure 16 comparison."""

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}\usepackage{bm}"


def style_axes(ax: plt.Axes) -> None:
    """Apply consistent styling to one Matplotlib axis."""
    ax.tick_params(axis="both", which="major", width=2.0, length=6, labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def load_timeseries_csv(
    csv_path: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load extracted well/cell time-series data from CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Time-series CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "time_days" not in df.columns:
        raise ValueError(f"CSV must contain a 'time_days' column: {csv_path}")

    timesteps = df["time_days"].to_numpy(dtype=float)
    results = {
        col: df[col].to_numpy(dtype=float) for col in df.columns if col != "time_days"
    }

    return timesteps, results


def compute_production_diagnostics(
    timesteps: Sequence[float],
    results: dict[str, np.ndarray],
    *,
    vtk_file: str | Path,
    p_bhp: float = 7.0e6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute production mass rate and energy rate from extracted time series."""
    required = {"pressure", "s_halite", "s_gas", "z_NaCl", "enthalpy"}
    missing = required.difference(results)
    if missing:
        raise ValueError(
            f"Missing required variables for production diagnostics: {sorted(missing)}"
        )

    vtk_file = Path(vtk_file)
    if not vtk_file.exists():
        raise FileNotFoundError(f"VTK thermodynamic table not found: {vtk_file}")

    sampler = VTKSampler(vtk_file)
    sampler.conversion_factors = (1.0, 1.0e-3, 1.0e-5)

    pressure = np.asarray(results["pressure"], dtype=float)
    s_halite = np.asarray(results["s_halite"], dtype=float)
    s_vapour = np.asarray(results["s_gas"], dtype=float)
    z_nacl = np.asarray(results["z_NaCl"], dtype=float)
    enthalpy = np.asarray(results["enthalpy"], dtype=float)

    q_prod = compute_production_rate(
        p_cell=pressure,
        z_NaCl=z_nacl,
        h_cell=enthalpy,
        s_halite=s_halite,
        s_vapour=s_vapour,
        p_BHP=p_bhp,
        vtk_sampler=sampler,
    )

    energy_prod = -q_prod * enthalpy
    return np.asarray(timesteps, dtype=float), q_prod, energy_prod


def save_production_diagnostics_cache(
    output_csv: str | Path,
    timesteps: Sequence[float],
    q_prod: Sequence[float],
    energy_prod: Sequence[float],
) -> None:
    """Save computed production diagnostics to a reusable CSV cache."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "time_days": np.asarray(timesteps, dtype=float),
            "q_prod": np.asarray(q_prod, dtype=float),
            "energy_prod": np.asarray(energy_prod, dtype=float),
        }
    )
    df.to_csv(output_csv, index=False)
    print(f"Saved production diagnostics cache: {output_csv}")


def load_production_diagnostics_cache(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached production diagnostics from CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Production diagnostics cache not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"time_days", "q_prod", "energy_prod"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    return (
        df["time_days"].to_numpy(dtype=float),
        df["q_prod"].to_numpy(dtype=float),
        df["energy_prod"].to_numpy(dtype=float),
    )


def plot_production_diagnostics_single(
    timesteps: Sequence[float],
    q_prod: Sequence[float],
    energy_prod: Sequence[float],
    output_path: str | Path,
    *,
    show: bool = False,
) -> None:
    """Plot production mass rate and energy rate for one simulation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(timesteps, dtype=float)
    q = np.asarray(q_prod, dtype=float)
    e = np.asarray(energy_prod, dtype=float)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        gridspec_kw={"wspace": 0.3},
    )
    ax_q, ax_e = axes

    ax_q.plot(t, -q, color="tab:blue", linewidth=2.5)
    ax_q.set_xlabel(r"\textbf{Time [days]}", fontsize=19)
    ax_q.set_ylabel(r"$\bm{q_{\mathrm{prod}}}$ \textbf{[kg/s]}", fontsize=19)
    style_axes(ax_q)

    ax_e.plot(t, e, color="tab:red", linewidth=2.5)
    ax_e.set_xlabel(r"\textbf{Time [days]}", fontsize=19)
    ax_e.set_ylabel(r"$\bm{E_{\mathrm{prod}}}$ \textbf{[W]}", fontsize=19)
    style_axes(ax_e)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved single-case production diagnostics figure: {output_path}")


def plot_production_diagnostics_comparison(
    timesteps_phi_01: Sequence[float],
    q_prod_phi_01: Sequence[float],
    energy_prod_phi_01: Sequence[float],
    timesteps_phi_10: Sequence[float],
    q_prod_phi_10: Sequence[float],
    energy_prod_phi_10: Sequence[float],
    output_path: str | Path,
    *,
    t_min: float = 2.0,
    t_max: float = 7.0,
    q_ylim: tuple[float, float] = (0.0, 0.008),
    energy_ylim: tuple[float, float] = (0.0, 7000.0),
    show: bool = False,
) -> None:
    """Plot Figure 16 production mass-rate and energy-rate comparison."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_01 = np.asarray(timesteps_phi_01, dtype=float)
    q_01 = np.asarray(q_prod_phi_01, dtype=float)
    e_01 = np.asarray(energy_prod_phi_01, dtype=float)

    t_10 = np.asarray(timesteps_phi_10, dtype=float)
    q_10 = np.asarray(q_prod_phi_10, dtype=float)
    e_10 = np.asarray(energy_prod_phi_10, dtype=float)

    mask_01 = t_01 >= t_min
    mask_10 = t_10 >= t_min
    label_q_inj_0_28 = r"$q_{\mathrm{inj}} = 0.28~\mathrm{kg}\,\mathrm{m}^{-3}\,\mathrm{s}^{-1}$"
    label_q_inj_0_364 = r"$q_{\mathrm{inj}} = 0.364~\mathrm{kg}\,\mathrm{m}^{-3}\,\mathrm{s}^{-1}$"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        gridspec_kw={"wspace": 0.3},
    )
    ax_q, ax_e = axes

    ax_q.plot(
        t_01[mask_01],
        -q_01[mask_01],
        label=label_q_inj_0_28,
        color="tab:blue",
        linewidth=2.5,
    )
    ax_q.plot(
        t_10[mask_10],
        -q_10[mask_10],
        label=label_q_inj_0_364,
        color="tab:orange",
        linewidth=2.5,
        linestyle="--",
    )
    ax_q.set_xlabel(r"\textbf{Time [days]}", fontsize=19)
    ax_q.set_ylabel(r"$\bm{q_{\mathrm{prod}}}$ \textbf{[kg/s]}", fontsize=19)
    ax_q.set_xlim(t_min, t_max)
    ax_q.set_ylim(*q_ylim)
    ax_q.legend(fontsize=14, loc="upper right")
    ax_q.grid(True, alpha=0.3)
    style_axes(ax_q)

    ax_e.plot(
        t_01[mask_01],
        e_01[mask_01],
        label=label_q_inj_0_28,
        color="tab:blue",
        linewidth=2.5,
    )
    ax_e.plot(
        t_10[mask_10],
        e_10[mask_10],
        label=label_q_inj_0_364,
        color="tab:orange",
        linewidth=2.5,
        linestyle="--",
    )
    ax_e.set_xlabel(r"\textbf{Time [days]}", fontsize=19)
    ax_e.set_ylabel(r"$\bm{E_{\mathrm{prod}}}$ \textbf{[W]}", fontsize=19)
    ax_e.set_xlim(t_min, t_max)
    ax_e.set_ylim(*energy_ylim)
    ax_e.legend(fontsize=14, loc="upper right")
    ax_e.grid(True, alpha=0.3)
    style_axes(ax_e)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved production comparison figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot production diagnostics from cached production CSV files."
    )
    parser.add_argument("--phi-0-1", required=True, help="CSV cache for phi = 0.1.")
    parser.add_argument("--phi-1-0", required=True, help="CSV cache for phi = 1.0.")
    parser.add_argument("--out", required=True, help="Output comparison figure.")
    parser.add_argument("--t-min", type=float, default=2.0)
    parser.add_argument("--t-max", type=float, default=7.0)
    parser.add_argument("--q-ymin", type=float, default=0.0)
    parser.add_argument("--q-ymax", type=float, default=0.008)
    parser.add_argument("--energy-ymin", type=float, default=0.0)
    parser.add_argument("--energy-ymax", type=float, default=7000.0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the command-line comparison plotter."""
    args = parse_args()

    t_01, q_01, e_01 = load_production_diagnostics_cache(args.phi_0_1)
    t_10, q_10, e_10 = load_production_diagnostics_cache(args.phi_1_0)

    plot_production_diagnostics_comparison(
        timesteps_phi_01=t_01,
        q_prod_phi_01=q_01,
        energy_prod_phi_01=e_01,
        timesteps_phi_10=t_10,
        q_prod_phi_10=q_10,
        energy_prod_phi_10=e_10,
        output_path=args.out,
        t_min=args.t_min,
        t_max=args.t_max,
        q_ylim=(args.q_ymin, args.q_ymax),
        energy_ylim=(args.energy_ymin, args.energy_ymax),
        show=args.show,
    )


if __name__ == "__main__":
    main()
