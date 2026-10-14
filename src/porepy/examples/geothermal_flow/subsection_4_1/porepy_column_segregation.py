#!/usr/bin/env python
"""PorePy reproduction of the first row of ``column_segregation_kx10.pdf``.

A 1-D vertical column, density-inverted: heavy alpha on top, light gamma on the
bottom, a passive intermediate phase beta in a central band of height ``LY/3``.
Alpha sinks and gamma rises in a counter-current; the question is which stream
entrains beta. Three runs sweep the intermediate density across the alpha-gamma pair,

    near-heavy   rho_beta = 1250   -> m_beta^{ag} = 1    (joins heavy)
    exact        rho_beta = 1000   -> m_beta^{ag} = 1/2  (on the mean-density separator)
    near-light   rho_beta =  750   -> m_beta^{ag} = 0    (joins light)

and BOTH HU constructions are run so they can be compared at the exact mean: ``hu`` (the
mobility-product G = lambda_g lambda_d / lambda) and ``hu-mw`` (the fractional-flow
G = f_g f_d lambda). The hamon reference compares HU against PPU; here the PPU track is
replaced by the second HU construction.

This reuses the subsection_4_1 barriers solver (``porepy_2d_solver``) unchanged
and only overrides geometry (1 x ny barrier-free column), the IC band layout
(equal thirds instead of 10/80/10) and the middle-phase density. The barriers
model keeps working.

Run:  python porepy_column_segregation.py          # kx10, both schemes, 3 cases
      python porepy_column_segregation.py --ny 100 --t-end-days 30
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import porepy as pp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import porepy_2d_solver as base  # noqa: E402  (barriers solver primitives, reused verbatim)

RHO_A, RHO_G = 1500.0, 500.0  # heavy (alpha) / light (gamma) [kg/m^3]
CASES = {  # label -> rho_beta
    "near_heavy": 0.75 * RHO_A + 0.25 * RHO_G,  # 1250
    "exact": 0.50 * RHO_A + 0.50 * RHO_G,  # 1000
    "near_light": 0.25 * RHO_A + 0.75 * RHO_G,  # 750
}
CASE_TITLE = {
    "near_heavy": r"near-heavy  $\rho_\beta=1250$",
    "exact": r"exact mean  $\rho_\beta=1000$",
    "near_light": r"near-light  $\rho_\beta=750$",
}
SCHEMES = ("hu", "hu-mw")  # mobility-product vs fractional-flow HU
DISPLAY = {"hu": "HU (mp)", "hu-mw": "HU (ff)"}  # figure labels

LY = 100.0  # column height [m]
WIDTH = 1.0  # column width [m] (nx=1; cancels in 1-D)


def midpoint_weight(rho_beta: float) -> float:
    """Midpoint background weight ``m_beta`` of the (alpha, gamma) pair (Def. 3.3):
    ``1/2 (1 + sgn[(rho_a - rho_g)(2 rho_b - rho_a - rho_g)])`` in {0, 1/2, 1}."""
    s = np.sign((RHO_A - RHO_G) * (2.0 * rho_beta - RHO_A - RHO_G))
    return 0.5 * (1.0 + s)


# --------------------------------------------------------------------------- #
#  Overrides: swap geometry + IC band layout, capture s_beta(y, t) per step.
#  First in the MRO so they win over BarriersBoundingBox2D / IC_NphaseSegregation.
# --------------------------------------------------------------------------- #
class _ColumnOverrides(pp.PorePyModel):
    _height = LY
    _width = WIDTH

    # ---- geometry: barrier-free 1 x ny column ----
    def set_domain(self) -> None:
        H = self.units.convert_units(self._height, "m")
        W = self.units.convert_units(self._width, "m")
        self._domain = pp.Domain({"xmax": W, "ymax": H})

    def grid_type(self):
        return "cartesian"

    def meshing_arguments(self) -> dict:
        H = self.units.convert_units(self._height, "m")
        W = self.units.convert_units(self._width, "m")
        ny = int(self.params.get("ny", 200))
        # cell_size sets dy (ny cells); cell_size_x spans the full width -> nx = 1.
        return {"cell_size": H / ny, "cell_size_x": W}

    def set_fractures(self) -> None:
        self._fractures = []

    def barrier_cell_mask(self, sd: pp.Grid) -> np.ndarray:
        return np.zeros(sd.num_cells, dtype=bool)

    # ---- IC: three equal bands (alpha top, beta middle, gamma bottom third) ----
    def _band_bounds(self) -> list:
        H = self.units.convert_units(self._height, "m")
        return list(np.linspace(H, 0.0, base.NPHASE + 1))

    # ---- capture s_beta(y, t) in memory (nx=1 -> a true 1-D profile) ----
    def initial_condition(self) -> None:
        super().initial_condition()
        self._seg_t = []
        self._seg_prof = []
        self._seg_y = None
        self._record_profile()

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        self._record_profile()

    def _record_profile(self) -> None:
        es = self.equation_system
        sd = self.mdg.subdomains(dim=self.nd)[0]
        order = np.argsort(sd.cell_centers[1])  # bottom -> top
        if self._seg_y is None:
            self._seg_y = sd.cell_centers[1][order]
        beta = list(self.fluid.phases)[1]  # phase 1 = intermediate (beta)
        s_beta = np.asarray(beta.saturation([sd]).value(es), dtype=float)[order]
        self._seg_t.append(float(self.time_manager.time) / base.day)
        self._seg_prof.append(s_beta)


class ColumnFlowModel(_ColumnOverrides, base.FlowModel):
    """``hu`` (mobility-product) on the column."""


class ColumnFractionalFlowModel(_ColumnOverrides, base.FractionalFlowModel):
    """``hu-mw`` (fractional-flow) on the column."""


def column_model_class(params: dict):
    return (
        ColumnFractionalFlowModel
        if params.get("fractional_flow", False)
        else ColumnFlowModel
    )


# --------------------------------------------------------------------------- #
#  One case: build the column model, run it, return the captured s_beta(y, t).
# --------------------------------------------------------------------------- #
def run_case(
    rho_beta: float,
    scheme: str,
    *,
    ny: int = 200,
    perm_factor: float = 10.0,
    t_end_days: float = 60.0,
    dt_init_days: float = 0.05,
    dt_max_days: float = 0.25,
    out_root: str = "visualization_column",
) -> dict:
    """Advance one column to ``t_end_days`` with the real PorePy solver. Returns
    ``dict(t[days], y[m] ascending, prof[(ntime, ny)], com[m], m_beta)``."""
    snaps = (0.0, t_end_days)  # VTU only at the ends; map is per-step
    k_rock = base.k_rock * perm_factor
    solid = pp.SolidConstants(
        permeability=k_rock,
        porosity=base.porosity,
        thermal_conductivity=2.0 * base.to_Mega,
        density=2500.0,
        specific_heat_capacity=base.C_P,
        residual_aperture=base.FRACTURE_APERTURE,
    )
    tag = f"{scheme}_{int(round(rho_beta))}"
    params = base.build_params(
        3,
        scheme,
        t_end_days=t_end_days,
        dt_init_days=dt_init_days,
        dt_max_days=dt_max_days,
        snap_days=snaps,
        fractures=False,
        ny=ny,
        material_constants={"solid": solid},
        lagrange_linear_solver="scipy",
        folder_name=os.path.join(out_root, f"col_{tag}"),
        file_name=f"col_{tag}",
    )
    # Override the evenly-spaced middle density with this case's rho_beta (read
    # at model-build and per-residual time by the EOS and the z->s inversion).
    base.RHO = np.array([RHO_A, float(rho_beta), RHO_G])
    base.RHO_BANDS = base.RHO.copy()

    model = column_model_class(params)(params)
    n_steps = max(1, round(t_end_days / dt_max_days))
    drift_tol = 10.0 ** (-(base.DRIFT_ORDER - 1)) / (2 * n_steps)
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=1.0e-4, metric=pp.EquationBasedLebesgueMetric(model)
            ),
            "null_drift": base.NullSpaceDriftCriterion(model, tol=drift_tol),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=13),
        },
    }
    runner = pp.ModelRunner(
        model,
        solver_params,
        nonlinear_solver=base.geothermal_nonlinear_solver(solver_params),
    )
    runner.run()

    t = np.asarray(model._seg_t)
    y = np.asarray(model._seg_y)
    prof = np.asarray(model._seg_prof)  # (ntime, ny), bottom -> top
    w = np.clip(prof, 0.0, None)
    com = (w * y[None, :]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-30)
    m_beta = midpoint_weight(rho_beta)
    # Persist the per-step space-time profile so the figure (and a hamon-difference row)
    # can be rebuilt without re-running: only t=0/t=end VTU are otherwise kept.
    np.savez(os.path.join(out_root, f"{tag}.npz"), t=t, y=y, prof=prof, com=com,
             m_beta=m_beta)
    print(
        f"  {scheme:5s} rho_b={rho_beta:6.0f}  m_beta={m_beta:.2f}  "
        f"com: {com[0]:.2f} -> {com[-1]:.2f} m  (d={com[-1] - com[0]:+.2f} m)  "
        f"steps={len(t) - 1}",
        flush=True,
    )
    return dict(t=t, y=y, prof=prof, com=com, m_beta=m_beta)


# --------------------------------------------------------------------------- #
#  Figure: first row of column_segregation_kx10.pdf (beta saturation maps + CoM tracks).
# --------------------------------------------------------------------------- #
def make_figure(
    results: dict,
    out_dir: str,
    suffix: str = "",
    perm_factor: float = 10.0,
    t_end_days: float = 60.0,
) -> None:
    """One row of three panels (one per density case). Background: the ``hu``
    beta-saturation space-time map. Overlaid: the beta centre-of-mass track for
    ``hu`` (solid) and ``hu-mw`` (dashed). Mirrors the top row of the hamon figure
    with the PPU track replaced by ``hu-mw``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        cmap = sns.color_palette("vlag", as_cmap=True)
    except Exception:  # pragma: no cover
        cmap = plt.get_cmap("coolwarm")

    labels = ["near_heavy", "exact", "near_light"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True)
    s_im = None
    for k, lab in enumerate(labels):
        r_hu = results[lab]["hu"]
        r_ff = results[lab]["hu-mw"]
        ax = axes[k]
        s_im = ax.pcolormesh(
            r_hu["t"],
            r_hu["y"],
            r_hu["prof"].T,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            shading="gouraud",
            rasterized=True,
        )
        ax.plot(r_hu["t"], r_hu["com"], color="k", lw=1.6, label=DISPLAY["hu"])
        ax.plot(
            r_ff["t"],
            r_ff["com"],
            color="#15b01a",
            lw=1.5,
            ls="--",
            label=DISPLAY["hu-mw"],
        )
        ax.set_title(
            f"{CASE_TITLE[lab]}\n$m_\\beta^{{\\alpha\\gamma}}={r_hu['m_beta']:.1f}$",
            fontsize=12,
        )
        ax.set_xlim(0, t_end_days)
        ax.set_ylim(0, LY)
        ax.set_xlabel("time [days]")
        if k == 0:
            ax.set_ylabel("height $y$ [m]")
            ax.legend(
                loc="lower right",
                fontsize=8,
                framealpha=0.85,
                title=r"$\bar{y}_\beta=\dfrac{\int_\Omega s_\beta\,y\,\mathrm{d}V}"
                r"{\int_\Omega s_\beta\,\mathrm{d}V}$",
            )
        else:
            ax.set_yticklabels([])

    cb = fig.colorbar(
        s_im, ax=list(axes), location="right", fraction=0.04, pad=0.02, shrink=0.9
    )
    cb.set_label(r"$\beta$ saturation $s_\beta$ [-]  (HU, mobility-product)")

    k_md = perm_factor * 1000.0
    fig.suptitle(
        f"PorePy column segregation:  $\\rho_\\alpha={RHO_A:.0f}$, "
        f"$\\rho_\\gamma={RHO_G:.0f}$ kg m$^{{-3}}$   "
        f"($K={k_md:g}$ mD, $t_f={t_end_days:.0f}$ d)",
        fontsize=13,
    )
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"porepy_column_segregation{suffix}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("wrote", os.path.relpath(path, HERE))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ny", type=int, default=200, help="column cells (vertical); default 200"
    )
    ap.add_argument(
        "--perm-factor",
        type=float,
        default=10.0,
        help="rock permeability factor (default 10 = kx10)",
    )
    ap.add_argument(
        "--t-end-days", type=float, default=60.0, help="run horizon [days] (default 60)"
    )
    ap.add_argument(
        "--dt-init-days", type=float, default=0.05, help="initial adaptive step [days]"
    )
    ap.add_argument(
        "--dt-max-days", type=float, default=0.25, help="max adaptive step [days]"
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES),
        choices=list(CASES),
        help="density cases to run",
    )
    ap.add_argument(
        "--schemes",
        nargs="+",
        default=list(SCHEMES),
        choices=list(SCHEMES),
        help="HU constructions to run",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(HERE, "passive_phase_example"),
        help="figure output folder (default passive_phase_example/)",
    )
    args = ap.parse_args()

    suffix = "" if args.perm_factor == 1.0 else f"_kx{args.perm_factor:g}"
    print(
        f"=== PorePy column segregation  (perm x{args.perm_factor:g}, "
        f"ny={args.ny}, t_end={args.t_end_days:g} d) ==="
    )
    results: dict = {}
    for lab in args.cases:
        results[lab] = {}
        for scheme in args.schemes:
            results[lab][scheme] = run_case(
                CASES[lab],
                scheme,
                ny=args.ny,
                perm_factor=args.perm_factor,
                t_end_days=args.t_end_days,
                dt_init_days=args.dt_init_days,
                dt_max_days=args.dt_max_days,
            )
    if set(args.schemes) == set(SCHEMES) and set(args.cases) == set(CASES):
        make_figure(
            results,
            args.out,
            suffix=suffix,
            perm_factor=args.perm_factor,
            t_end_days=args.t_end_days,
        )


if __name__ == "__main__":
    main()
