"""Immiscible three-phase gravity segregation through impermeable barriers.

Reproduces Example 6.3 / Fig. 5 of Bosma, Hamon, Mallison & Tchelepi, "Smooth implicit
hybrid upwinding for compositional multiphase flow in porous media", CMAME 388 (2022)
114288: a 100 m x 100 m closed vertical box in which a heavy fluid (initially top 10%),
a light fluid (bottom 10%) and an intermediate fluid (the rest) segregate by gravity
through a field of horizontal impermeable barriers with openings.

Structure follows ``tp_tc_gravitational_segregation.py`` (inline constant-property EOS,
``FlowModelBase``, ``pp.ModelRunner``).  The 3-phase / 3-component IMMISCIBLE machinery
follows ``tests/functional/setups/buoyancy_flow_model.py`` (one component per phase,
immiscibility via partial-fraction chi = 1/0, temperature eliminated to 0).  Geometry
(box + barriers), the closed boundary, and the segregation IC come from the market modules.

Run this on your end (it is NOT auto-run). Items that may need tuning for your PorePy
version are flagged with NOTE.
"""
from __future__ import annotations

import os

# Numba JIT: keeping this disabled forces every numba-compiled discretization/AD kernel to
# run as interpreted Python (10-50x slower assembly). Leave JIT ENABLED for real runs; set
# PP_DISABLE_JIT=1 only when debugging numba kernels.
if os.environ.get("PP_DISABLE_JIT", "0") == "1":
    os.environ["NUMBA_DISABLE_JIT"] = "1"

from typing import Callable, Optional, Sequence, cast  # noqa: E402

import numpy as np  # noqa: E402
import porepy as pp  # noqa: E402
from porepy.models.abstract_equations import LocalElimination  # noqa: E402

# Absolute imports (like geothermal_H2O_low_NaCl_content_fig_5.py) so the market modules'
# internal ``from ...vtk_sampler import VTKSampler`` resolves. Requires porepy importable.
from porepy.examples.geothermal_flow.flow_model_base import FlowModelBase  # noqa: E402
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E402,E501
    GeometryBarriers2D,
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E402
    BC_three_phase_closed,
)
from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E402
    IC_three_phase_segregation,
)

# --------------------------------------------------------------------------------------- #
#  Fluid + rock constants (WA-HU Ex. 6.3).  Mega-scaled units: p[MPa], mu[MPa.s], rho[kg/m3].
# --------------------------------------------------------------------------------------- #
to_Mega = 1.0e-6

# phase densities [kg/m^3]: water = heavy, oil = intermediate, gas = light
rho_w, rho_o, rho_g = 1500.0, 1000.0, 500.0
# all phases mu = 0.001 kg/(m.s) = 1 cP  (scaled by to_Mega -> MPa.s in the model)
mu_w = mu_o = mu_g = 1.0e-3
# specific enthalpies [MJ/kg]: isothermal immiscible -> arbitrary but distinct
h_w, h_o, h_g = 1.0, 1.5, 2.0

milli_darcy = 9.869233e-16          # 1 mD in m^2
k_rock = 100.0 * milli_darcy          # homogeneous rock permeability (k = 1 mD)
porosity = 0.3
BARRIER_K_FACTOR = 1.0e-8           # barrier cells get k * this (effectively impermeable)


# --------------------------------------------------------------------------------------- #
#  Constant-property EOS, one per phase (mirrors buoyancy_flow_model.py BaseEOS)
# --------------------------------------------------------------------------------------- #
class BaseEOS(pp.compositional.EquationOfState):
    """Constant per-phase rho / mu / h / kappa; zero derivatives."""

    _rho = rho_w
    _mu = mu_w
    _h = h_w

    def kappa(self, *deps):
        nc = len(deps[0])
        return 2.0 * np.ones(nc) * to_Mega, np.zeros((len(deps), nc))

    def rho_func(self, *deps):
        nc = len(deps[0])
        return self._rho * np.ones(nc), np.zeros((len(deps), nc))

    def mu_func(self, *deps):
        nc = len(deps[0])
        return self._mu * np.ones(nc) * to_Mega, np.zeros((len(deps), nc))

    def h(self, *deps):
        nc = len(deps[0])
        return self._h * np.ones(nc), np.zeros((len(deps), nc))

    def compute_phase_properties(self, phase_state, *thermo, params=None):
        nc = len(thermo[0])
        rho, drho = self.rho_func(*thermo)
        h, dh = self.h(*thermo)
        mu, dmu = self.mu_func(*thermo)
        kappa, dkappa = self.kappa(*thermo)
        # NOTE: phis/dphis shapes copied verbatim from the working buoyancy_flow_model
        # test (placeholders for immiscible EOS; not used by the segregation physics).
        return pp.compositional.PhaseProperties(
            state=phase_state, rho=rho, drho=drho, h=h, dh=dh, mu=mu, dmu=dmu,
            kappa=kappa, dkappa=dkappa, phis=np.empty((2, nc)), dphis=np.empty((2, 3, nc)),
        )


class WaterEOS(BaseEOS):
    _rho, _mu, _h = rho_w, mu_w, h_w


class OilEOS(BaseEOS):
    _rho, _mu, _h = rho_o, mu_o, h_o


class GasEOS(BaseEOS):
    _rho, _mu, _h = rho_g, mu_g, h_g


# --------------------------------------------------------------------------------------- #
#  Local-elimination closures (mirror buoyancy_flow_model.py): dependent saturations from
#  the overall fractions z, immiscibility chi = 1/0, and temperature == 0.
#  deps = (pressure, enthalpy, z_C5H12, z_CH4).  Each returns (values, zero-derivatives).
# --------------------------------------------------------------------------------------- #
def _sat_denominator(z_oil, z_gas):
    return (
        -((-1.0 + z_oil + z_gas) * rho_g * rho_o)
        + z_oil * rho_g * rho_w
        + z_gas * rho_o * rho_w
    )


def oil_saturation_func(*deps):
    z_oil, z_gas = deps[2], deps[3]
    nc = len(z_oil)
    vals = np.clip((z_oil * rho_g * rho_w) / _sat_denominator(z_oil, z_gas), 1.0e-16, 1.0)
    return vals, np.zeros((len(deps), nc))


def gas_saturation_func(*deps):
    z_oil, z_gas = deps[2], deps[3]
    nc = len(z_oil)
    vals = np.clip((z_gas * rho_o * rho_w) / _sat_denominator(z_oil, z_gas), 1.0e-16, 1.0)
    return vals, np.zeros((len(deps), nc))


def _chi(active: bool):
    def f(*deps):
        nc = len(deps[0])
        vals = (np.ones(nc) if active else np.zeros(nc))
        return np.clip(vals, 1.0e-16, 1.0), np.zeros((len(deps), nc))
    return f


# immiscibility map: C5H12 lives only in oil, CH4 only in gas (H2O reference -> closure)
chi_functions_map = {
    "C5H12_water": _chi(False), "C5H12_oil": _chi(True), "C5H12_gas": _chi(False),
    "CH4_water": _chi(False), "CH4_oil": _chi(False), "CH4_gas": _chi(True),
}

saturation_functions_map = {"oil": oil_saturation_func, "gas": gas_saturation_func}


def temperature_func(*deps):
    nc = len(deps[0])
    return np.zeros(nc), np.zeros((len(deps), nc))   # isothermal: T == 0 (decouples energy)


# --------------------------------------------------------------------------------------- #
#  Fluid mixture (3 components, 3 immiscible phases) and secondary-equation eliminations
# --------------------------------------------------------------------------------------- #
class FluidMixture3N(pp.PorePyModel):
    def get_components(self) -> Sequence[pp.FluidComponent]:
        # H2O = reference (dependent z); C5H12, CH4 are the independent overall fractions.
        return [
            pp.FluidComponent(name="H2O"),
            pp.FluidComponent(name="C5H12"),
            pp.FluidComponent(name="CH4"),
        ]

    def get_phase_configuration(self, components):
        # first phase (water) is the reference phase (dependent saturation).
        return [
            (pp.compositional.PhysicalState.liquid, "water", WaterEOS(components)),
            (pp.compositional.PhysicalState.liquid, "oil", OilEOS(components)),
            (pp.compositional.PhysicalState.gas, "gas", GasEOS(components)),
        ]

    def dependencies_of_phase_properties(self, phase):
        z = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z


class SecondaryEquations3N(LocalElimination):
    dependencies_of_phase_properties: Callable
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()
        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix))
        sd_and_bnd = subdomains + [matrix_boundary]
        rphase = self.fluid.reference_phase

        # eliminate the non-reference (oil, gas) saturations as functions of z
        for phase in self.fluid.phases:
            if phase == rphase:
                continue
            self.eliminate_locally(
                phase.saturation,
                self.dependencies_of_phase_properties(phase),
                saturation_functions_map[phase.name],
                sd_and_bnd,
            )
        # eliminate the independent partial fractions (immiscibility chi = 1/0)
        for phase in self.fluid.phases:
            for comp in phase:
                if self.has_independent_partial_fraction(comp, phase):
                    self.eliminate_locally(
                        phase.partial_fraction_of[comp],
                        self.dependencies_of_phase_properties(phase),
                        chi_functions_map[comp.name + "_" + phase.name],
                        sd_and_bnd,
                    )
        # eliminate temperature -> 0 (isothermal)
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),
            temperature_func,
            sd_and_bnd,
        )


# --------------------------------------------------------------------------------------- #
#  The model
# --------------------------------------------------------------------------------------- #
class FlowModel(
    GeometryBarriers2D,
    FluidMixture3N,
    IC_three_phase_segregation,
    BC_three_phase_closed,
    SecondaryEquations3N,
    FlowModelBase,
):
    def __init__(self, params):
        super().__init__(params)

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains) ** 2          # quadratic kr (paper Ex. 6.3)

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.TpfaAd:
        return pp.ad.TpfaAd(self.darcy_keyword, subdomains)

    def fourier_flux_discretization(self, subdomains: Sequence[pp.Grid]) -> pp.ad.TpfaAd:
        return pp.ad.TpfaAd(self.fourier_keyword, list(subdomains))

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Homogeneous rock permeability with impermeable barrier cells.

        Mirrors ConstantPermeability.permeability but uses a cell-wise array: barrier
        cells (from the geometry's ``barrier_cell_mask``) get k * BARRIER_K_FACTOR.
        """
        size = sum(sd.num_cells for sd in subdomains)
        vals = np.full(size, self.solid.permeability)
        offset = 0
        for sd in subdomains:
            mask = self.barrier_cell_mask(sd)          # provided by GeometryBarriers2D
            cell_vals = vals[offset:offset + sd.num_cells]
            cell_vals[mask] = self.solid.permeability * BARRIER_K_FACTOR
            offset += sd.num_cells
        permeability = pp.wrap_as_dense_ad_array(vals, size, name="permeability")
        return self.isotropic_second_order_tensor(subdomains, permeability)


# --------------------------------------------------------------------------------------- #
#  Run configuration (module level), mirroring tp_tc_gravitational_segregation.py
# --------------------------------------------------------------------------------------- #
day = 86400.0
# Fig. 5 snapshots are at 0, 78 and 571 days. constant_dt steps and exports at the schedule.
# time_manager = pp.TimeManager(
#     schedule=[0.0, 78.0 * day, 571.0 * day],
#     dt_init=1.0 * day,                                 # NOTE: tune dt for convergence/cost
#     constant_dt=True,
#     iter_max=50,
#     print_info=True,
# )

dt = 0.25 * day
tf = 571.0 * day
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,                                 # NOTE: tune dt for convergence/cost
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

# Export configuration: number of time steps between consecutive VTK/PVD exports.
export_every_n_steps = 8

# Build times_to_export as multiples of dt. Include t=0 and final time tf.
times = list(np.arange(0.0, tf, dt * export_every_n_steps))
times.append(tf)
times_to_export = times

solid_constants = pp.SolidConstants(
    permeability=k_rock,                               # 1 mD
    porosity=porosity,                                 # 0.3
    thermal_conductivity=2.0 * to_Mega,                # unused (isothermal)
    density=2500.0,
    specific_heat_capacity=1000.0 * to_Mega,
)

params = {
    "fractional_flow": False,
    "mass_mobility_weighted_permeability": False,
    "enable_buoyancy_effects": True,
    # buoyancy scheme: "hybrid" (HU), "phase_potential" (PPU), or your simplicial-PPU
    "buoyancy_upwinding": "hybrid",
    "lag_buoyancy_direction": False,
    "material_constants": {"solid": solid_constants},
    "time_manager": time_manager,
    "times_to_export": times_to_export,
    "grid_type": "cartesian",
    "prepare_simulation": False,
    "folder_name": "visualization_three_phase_barriers",
    "file_name": "three_phase_barriers",
    "max_iterations": 100,
    # AD backend: "reference" (PorePy's parser, default) or "sparsa" (external sparsa
    # engine via the adapter -- bit-exact, ~5x faster assembly). Requires `sparsa`
    # importable in the active environment (pip install -e on the sparsa repo).
    "ad_backend": "sparsa",
}

if __name__ == "__main__":
    model = FlowModel(params)
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.ResidualBasedAbsoluteCriterion(
                tol=1.0e-3, metric=pp.EquationBasedLebesgueMetric(model)
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=50),
        },
    }
    pp.ModelRunner(model, solver_params).run()
    print("cells:", sum(sd.num_cells for sd in model.mdg.subdomains()),
          " dofs:", model.equation_system.num_dofs())
