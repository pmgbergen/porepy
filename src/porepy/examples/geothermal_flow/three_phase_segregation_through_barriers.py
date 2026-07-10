"""Immiscible three-phase gravity segregation through impermeable barriers.

Reproduces Example 6.3 / Fig. 5 of Bosma, Hamon, Mallison & Tchelepi, "Smooth implicit
hybrid upwinding for compositional multiphase flow in porous media", CMAME 388 (2022)
114288: a 100 m x 100 m closed vertical box (100 x 100 cells of 1 m) in which a heavy fluid
(water, rho=1500, initially top 10%), a light fluid (gas, rho=500, bottom 10%) and an
intermediate fluid (oil, rho=1000, the rest) segregate by gravity through SEVEN horizontal
impermeable barrier layers with openings.

The barriers sit at figure depth-rows 16, 23, 38, 45, 58, 74, 82 (of 100; row 0 = top,
gravity downward), with 5, 2, 5, 2, 4, 6, 3 open-ended segments respectively (bottom -> top:
3, 6, 4, 2, 5, 2, 5). They are re-extracted at pixel resolution from Fig. 5(a) and defined in
``model_configuration/geometry_description/geometry_market.py`` (``_BARRIER_LAYERS_FIG``,
consumed by ``GeometryBarriers2D.barrier_cell_mask``); the extraction / verification lives in
``benchmark_figures_data/wahu_fig5_digitized/fig5_barriers_and_saturations.py``.

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

from typing import Callable, Optional, Sequence, cast  # noqa: E402

import numpy as np  # noqa: E402
import porepy as pp  # noqa: E402
from porepy.models.abstract_equations import LocalElimination  # noqa: E402

# Absolute imports (like geothermal_H2O_low_NaCl_content_fig_5.py) so the market modules'
# internal ``from ...obl_sampler import VTKSampler`` resolves. Requires porepy importable.
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import FlowModelBase  # noqa: E402,E501
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
#h_w, h_o, h_g = 1.0, 1.5, 2.0
h_w, h_o, h_g = 1.0, 1.0, 1.0

milli_darcy = 9.869233e-16          # 1 mD in m^2
k_rock = 1000.0 * milli_darcy          # homogeneous rock permeability (k = 1 mD)
porosity = 0.3
BARRIER_K_FACTOR = 1.0e-4           # barrier cells get k * this (effectively impermeable)


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


def _clip_to_simplex(z_oil, z_gas):
    """Project the overall fractions onto the valid simplex ``z_oil, z_gas >= 0`` and
    ``z_oil + z_gas <= 1`` (so ``z_water = 1 - z_oil - z_gas >= 0``).

    With ``z`` on the simplex the three derived phase saturations are guaranteed to form a
    valid partition -- each in ``[0, 1]`` and summing to 1, *including* the by-unity water
    saturation ``s_water = 1 - s_oil - s_gas``.  Clipping ``s_oil`` / ``s_gas``
    independently does NOT guarantee this: when a Newton iterate strays off the simplex
    the two clipped saturations can sum to more than 1, making ``s_water`` negative.
    Clipping the evaluation here prevents the incorrect (negative) saturations.
    """
    z_oil = np.clip(z_oil, 0.0, 1.0)
    z_gas = np.clip(z_gas, 0.0, 1.0)
    total = z_oil + z_gas
    scale = np.where(total > 1.0, 1.0 / np.maximum(total, 1.0e-30), 1.0)
    return z_oil * scale, z_gas * scale


def oil_saturation_func(*deps):
    z_oil, z_gas = _clip_to_simplex(deps[2], deps[3])
    nc = len(z_oil)
    vals = np.clip((z_oil * rho_g * rho_w) / _sat_denominator(z_oil, z_gas), 0.0, 1.0)
    return vals, np.zeros((len(deps), nc))


def gas_saturation_func(*deps):
    z_oil, z_gas = _clip_to_simplex(deps[2], deps[3])
    nc = len(z_oil)
    vals = np.clip((z_gas * rho_o * rho_w) / _sat_denominator(z_oil, z_gas), 0.0, 1.0)
    return vals, np.zeros((len(deps), nc))


def _chi(active: bool):
    def f(*deps):
        nc = len(deps[0])
        vals = (np.ones(nc) if active else np.zeros(nc))
        # Clip to [0, 1] (not [eps, 1]) so the by-unity reference partial fraction
        # x_ref = 1 - sum(others) also stays non-negative.
        return np.clip(vals, 0.0, 1.0), np.zeros((len(deps), nc))
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
    """Secondary-quantity closures for the 3-phase / 3-component model.

    Two representations are supported per quantity, selected by
    ``params["substitute_as_function"]`` (a subset of ``{"saturation",
    "partial_fraction"}``; default empty):

    * **Eliminated variable** (classic): the quantity is an independent variable with a
      local equation ``var - surrogate(p,h,z) = 0`` (:meth:`eliminate_locally`).
    * **Substituted function**: the quantity is the ``SurrogateFactory`` itself, dropped
      straight into the equations (:meth:`substitute_locally`). This removes its DOFs and
      its elimination equations -> smaller, faster system. It is Jacobian-equivalent here
      because the closures return zero derivatives (lagged), so the elimination already
      drops the z-coupling.

    Temperature is intentionally NOT substitutable: the energy equation's Fourier flux
    ``K_e grad(T)`` is discretized with MPFA; turning ``T`` into ``T(p,h,z)`` would expand
    ``grad(T)`` into ``grad(p)/grad(h)/grad(z)`` terms the MPFA stencil cannot carry. So
    temperature is always eliminated as a variable.
    """

    dependencies_of_phase_properties: Callable
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    # ----------------------------------------------------------------------------------
    #  Substitution configuration + surrogate registry
    # ----------------------------------------------------------------------------------
    def _substitute_as_function(self) -> set[str]:
        return set(self.params.get("substitute_as_function", []))

    def _substitute_saturation(self) -> bool:
        return "saturation" in self._substitute_as_function()

    def _substitute_partial_fraction(self) -> bool:
        return "partial_fraction" in self._substitute_as_function()

    def _substitutions(self) -> list:
        reg = getattr(self, "_substitution_registry", None)
        if reg is None:
            reg = []
            self._substitution_registry = reg
        return reg

    def substitute_locally(
        self, name: str, dependencies, func
    ) -> pp.ad.SurrogateFactory:
        """Build (once) a ``SurrogateFactory`` for ``name`` over ``dependencies`` via
        ``func`` and register it for value/derivative refreshes. The returned object is
        itself the accessor callable: on subdomains it evaluates to the function, on
        boundary grids to a time-dependent dense array of its values.

        Unlike :meth:`eliminate_locally`, this creates **no** secondary variable and
        **no** elimination equation.
        """
        for expr, *_ in self._substitutions():
            if expr.name == name:  # accessor may be requested more than once
                return expr
        sec_expr = pp.ad.SurrogateFactory(
            name=name, mdg=self.mdg, dependencies=dependencies, dof_info={"cells": 1}
        )
        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        boundary = cast(pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix))
        boundaries = [boundary] if boundary is not None else []
        self._substitutions().append(
            (sec_expr, func, list(self.mdg.subdomains()), boundaries)
        )
        return sec_expr

    def _refresh_substitutions(self, on_boundaries: bool) -> None:
        for sec_expr, func, subdomains, boundaries in self._substitutions():
            grids = boundaries if on_boundaries else subdomains
            for grid in grids:
                X = [
                    self.equation_system.evaluate(d([grid]))
                    for d in sec_expr._dependencies
                ]
                vals, diffs = func(*X)
                sec_expr.set_values_on_grid(vals, grid)
                if not on_boundaries:  # no derivatives are stored on the boundary
                    sec_expr.set_derivatives_on_grid(diffs, grid)

    # ----------------------------------------------------------------------------------
    #  Accessor + DOF-gate overrides (route substituted quantities to their surrogate and
    #  suppress their independent variables)
    # ----------------------------------------------------------------------------------
    def has_independent_saturation(self, phase: pp.Phase) -> bool:
        if self._substitute_saturation() and super().has_independent_saturation(phase):
            return False  # represented as a function -> no DOF
        return super().has_independent_saturation(phase)

    def saturation(self, phase: pp.Phase):
        if self._substitute_saturation() and super().has_independent_saturation(phase):
            return self.substitute_locally(
                f"saturation_{phase.name}",
                self.dependencies_of_phase_properties(phase),
                saturation_functions_map[phase.name],
            )
        return super().saturation(phase)

    def has_independent_partial_fraction(
        self, component: pp.Component, phase: pp.Phase
    ) -> bool:
        if self._substitute_partial_fraction() and super().has_independent_partial_fraction(
            component, phase
        ):
            return False  # represented as a function -> no DOF
        return super().has_independent_partial_fraction(component, phase)

    def partial_fraction(self, component: pp.Component, phase: pp.Phase):
        if self._substitute_partial_fraction() and super().has_independent_partial_fraction(
            component, phase
        ):
            return self.substitute_locally(
                f"partial_fraction_{component.name}_{phase.name}",
                self.dependencies_of_phase_properties(phase),
                chi_functions_map[component.name + "_" + phase.name],
            )
        return super().partial_fraction(component, phase)

    # ----------------------------------------------------------------------------------
    #  Per-iteration / per-time-step / initial refresh of the substituted surrogates
    #  (mirrors LocalElimination's update of eliminated surrogates).
    # ----------------------------------------------------------------------------------
    def update_derived_quantities(self) -> None:
        super().update_derived_quantities()
        self._refresh_substitutions(on_boundaries=False)

    def update_all_boundary_conditions(self) -> None:
        super().update_all_boundary_conditions()
        self._refresh_substitutions(on_boundaries=True)

    def initial_condition(self) -> None:
        super().initial_condition()
        # Seed the substituted surrogates so the t=0 state/export is consistent.
        self._refresh_substitutions(on_boundaries=False)

    def initialize_previous_iterate_and_time_step_values(self) -> None:
        # Substituted secondary quantities appear in accumulation terms, so (like phase
        # density/enthalpy) they need previous-iterate AND previous-time-step values.
        # Copy the current iterate-0 value into all iterate and time-step indices.
        super().initialize_previous_iterate_and_time_step_values()
        ni = self.iterate_indices.size
        nt = self.time_step_indices.size
        self._refresh_substitutions(on_boundaries=False)  # ensure iterate-0 is current
        for sec_expr, _, subdomains, _ in self._substitutions():
            for sd in subdomains:
                for _ in self.iterate_indices:
                    vals = sec_expr.get_values_on_grid(sd, iterate_index=0)
                    sec_expr.progress_iterate_values_on_grid(vals, sd, depth=ni)
                for _ in self.time_step_indices:
                    sec_expr.progress_values_in_time([sd], depth=nt)

    def after_nonlinear_convergence(self) -> None:
        # Shift the converged iterate value into the time-step history, mirroring the
        # framework's progression of phase-property surrogates (density/enthalpy).
        super().after_nonlinear_convergence()
        nt = self.time_step_indices.size
        for sec_expr, _, subdomains, _ in self._substitutions():
            sec_expr.progress_values_in_time(subdomains, depth=nt)

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()
        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix))
        sd_and_bnd = subdomains + [matrix_boundary]
        rphase = self.fluid.reference_phase

        # Non-reference (oil, gas) saturations: eliminate as variables, unless substituted
        # as functions (then has_independent_saturation is False and the accessor already
        # returns the surrogate -> nothing to eliminate here).
        for phase in self.fluid.phases:
            if not self.has_independent_saturation(phase):
                continue
            self.eliminate_locally(
                phase.saturation,
                self.dependencies_of_phase_properties(phase),
                saturation_functions_map[phase.name],
                sd_and_bnd,
            )
        # Independent partial fractions (immiscibility chi = 1/0): same logic -- the loop
        # gates on has_independent_partial_fraction, which is False when substituted.
        for phase in self.fluid.phases:
            for comp in phase:
                if self.has_independent_partial_fraction(comp, phase):
                    self.eliminate_locally(
                        phase.partial_fraction_of[comp],
                        self.dependencies_of_phase_properties(phase),
                        chi_functions_map[comp.name + "_" + phase.name],
                        sd_and_bnd,
                    )
        # Temperature -> 0 (isothermal): ALWAYS eliminated as a variable (see class docs:
        # grad(T) under MPFA must not become grad(p)/grad(h)/grad(z)).
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

    def data_to_export(self):
        """Export the standard variables plus the quantities that were substituted as
        functions (so they remain visible even though they are no longer variables):

        * all phase saturations ``s_<phase>`` (water/oil/gas),
        * any partial fractions that were substituted (``partial_fraction_*``),
        * temperature in Celsius ``T_C``.
        """
        data = super().data_to_export()
        es = self.equation_system
        sds = self.mdg.subdomains()
        offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        already = {v.name for v in es.variables}  # auto-exported variables (avoid dupes)

        def add_field(name, operator):
            if name in already:  # already exported as a variable
                return
            vals = np.asarray(operator.value(es), dtype=float)
            for i, sd in enumerate(sds):
                data.append((sd, name, vals[offsets[i]:offsets[i + 1]]))

        # All phase saturations (functions when substituted -> not auto-exported; the
        # reference-phase saturation is by-unity and never a variable).
        for phase in self.fluid.phases:
            add_field(f"s_{phase.name}", phase.saturation(sds))

        # Partial fractions substituted as functions (no longer auto-exported), named with
        # the x_<comp>_<phase> convention.
        for sec_expr, *_ in getattr(self, "_substitution_registry", []):
            if sec_expr.name.startswith("partial_fraction_"):
                add_field(sec_expr.name.replace("partial_fraction_", "x_"), sec_expr(sds))

        # Temperature in Celsius (temperature is kept as a variable).
        add_field("T_C", self.temperature(sds) - pp.ad.Scalar(273.15))
        return data

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

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Hydrostatic initial pressure, consistent with the top-boundary Dirichlet pressure
        AND with the initial phase layering (top 10% water, middle 80% oil, bottom 10% gas).

        In mechanical equilibrium the pressure gradient balances gravity,
        ``dp/d(depth) = rho_column * g``. Integrating downward from the top boundary
        (``p = p_top`` at height ``y = H``), with the piecewise-constant column density set by
        the initial phase distribution, gives

            ``p(y) = p_top + g * INT_y^H rho(y') dy'``.

        This uses the SAME gravity as :meth:`gravity_field`
        (``convert_units(GRAVITY_ACCELERATION) * to_Mega``) and the UNSCALED phase densities,
        so at t = 0 the Darcy potential ``grad p - rho_m g`` vanishes within each layer -- no
        spurious pressure-driven flow; only the density contrast across the layer interfaces
        drives the buoyant segregation. ``p_top = _p_ref`` equals the closed BC's top Dirichlet
        value, so the IC and the boundary agree at ``y = H``.
        """
        y = sd.cell_centers[1]                                       # height (0 bottom, H top)
        H = self.units.convert_units(self._height, "m")             # box height (= 100 m)
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2") * to_Mega
        y_wo = 0.9 * H                                               # water / oil interface
        y_og = 0.1 * H                                              # oil / gas interface
        # column density integrated from height y up to the top boundary H
        column = (
            rho_w * (H - np.maximum(y, y_wo))                       # water band [0.9H, H]
            + rho_o * np.maximum(0.0, y_wo - np.maximum(y, y_og))   # oil band  [0.1H, 0.9H]
            + rho_g * np.maximum(0.0, y_og - y)                     # gas band  [0, 0.1H]
        )
        return self._p_ref + g * column


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

dt = 0.03125 * day
tf = 78.0 * day
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,                                 # NOTE: tune dt for convergence/cost
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

# time_manager = pp.TimeManager(
#     schedule=[0.0, tf],
#     dt_init=dt,
#     constant_dt=False,
#     dt_min_max=(0.01 * dt, 1.0 * dt),
#     iter_relax_factors=(0.5, 2.0),
#     iter_optimal_range=(3, 8),
#     recomp_factor=0.3,
#     print_info=True,
# )

# Export configuration: number of time steps between consecutive VTK/PVD exports.
export_every_n_steps = 32

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
    # "substitute_as_function": ["saturation", "partial_fraction"],
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
    # Step control method options:
    # - "LS": Line Search (backtracking with Armijo condition)
    # - "TR": Trust Region with CFL-aware dynamic radius adjustment
    # - "TR-LS": Trust Region + Line Search refinement
    # - "None": Plain Newton (no step control)
    "step_control_method": "None",
    "step_control_alpha_min": 1.0e-5,  # Minimum acceptable step length
    "activate_step_control_after_iter": 2,  # Activate after this many iterations
    # AD backend: "reference" (PorePy's parser, default) or "sparsa" (external sparsa
    # engine via the adapter -- bit-exact, ~5x faster assembly). Requires `sparsa`
    # importable in the active environment (pip install -e on the sparsa repo).
    "ad_backend": "native",
    "use_petsc": True,  # Set to True to use PETSc with MUMPS solver
    "petsc_preconditioner": "cpr",
    # Options: 'bjacobi', 'asm', 'jacobi', 'lump_colsum', 'amg_hypre', 'ilu0', 'lu', 'cpr'
}

def report_system_size(model) -> int:
    """Print the system size and a table of registered variables: their dof count, role,
    and whether they may be represented as a function (``substitute_locally``).

    The primary PDE unknowns (pressure, enthalpy, overall fractions z) and temperature
    (its ``K_e grad(T)`` Fourier flux is MPFA-discretized) must stay variables; the
    secondary saturations and partial fractions are substitutable. Quantities already
    substituted as functions are listed separately.

    Returns the number of cells (for convenience).
    """
    es = model.equation_system
    ncells = sum(sd.num_cells for sd in model.mdg.subdomains())
    ndof = es.num_dofs()
    var_dofs: dict[str, int] = {}
    for v in es.variables:
        var_dofs[v.name] = var_dofs.get(v.name, 0) + int(es.dofs_of([v]).size)
    upc = ndof // ncells if ncells and ndof % ncells == 0 else round(ndof / max(ncells, 1), 2)

    def role(name: str) -> tuple[str, str]:
        if name in ("pressure", "enthalpy") or name.startswith("z_"):
            return "primary unknown", "no"
        if name == "temperature":
            return "secondary (keep: MPFA grad(T))", "no"
        if name.startswith("s_"):
            return "secondary saturation", "yes"
        if name.startswith("x_"):
            return "secondary partial fraction", "yes"
        return "other", "?"

    substituted = list(getattr(model, "_substitution_registry", []) or [])

    # AD expression-graph size: number of UNIQUE operators across all equations (the DAG
    # the assembly walks; the parser caches by node, so this ~ the per-assemble work) and
    # how many of them are function/surrogate ("evaluate") nodes.
    seen, n_eval = set(), 0
    stack = list(model.equation_system.equations.values())
    while stack:
        op = stack.pop()
        if id(op) in seen:
            continue
        seen.add(id(op))
        if getattr(op, "operation", None) == pp.ad.operators.Operations.evaluate:
            n_eval += 1
        stack.extend(op.children)
    n_nodes = len(seen)

    print("=" * 74)
    print(f" System size:  ncells = {ncells}   ndof = {ndof}   unknowns/cell = {upc}")
    print(f" AD graph:     {n_nodes} operators ({n_eval} function/surrogate nodes)")
    print(f" substitute_as_function = {model.params.get('substitute_as_function', [])}")
    print("=" * 74)
    print(f" {'variable':26s} {'ndof':>8}  {'role':32s} substitutable")
    print("-" * 74)
    for name in sorted(var_dofs):
        r, sub = role(name)
        print(f" {name:26s} {var_dofs[name]:>8}  {r:32s} {sub}")
    if substituted:
        print("-" * 74)
        print(f" substituted as functions ({len(substituted)}):")
        for sec_expr, *_ in substituted:
            print(f"   {sec_expr.name}")
    print("=" * 74)
    return ncells


if __name__ == "__main__":
    model = FlowModel(params)
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.ResidualBasedAbsoluteCriterion(
                tol=1.0e-5, metric=pp.EquationBasedLebesgueMetric(model)
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=30),
        },
    }
    # Construct the runner first (this prepares the simulation), so the system size can
    # be reported before the (long) time loop starts.
    # Construct the runner first (this prepares the simulation), then report the system
    # size + the registered variables (role and whether each is substitutable) before the
    # (long) time loop starts.
    runner = pp.ModelRunner(model, solver_params)
    ncells = report_system_size(model)
    ndof = model.equation_system.num_dofs()

    runner.run()
    print("cells:", ncells, " dofs:", ndof)
