"""Immiscible N-phase gravity segregation through impermeable barriers (PorePy CF model).

PorePy compositional-flow reference for subsection 4.2 -- the model to overlay against the
independent ``hamon_2d_solver.py`` in this folder. Reproduces Example 6.3 / Fig. 5 of Bosma,
Hamon, Mallison & Tchelepi, "Smooth implicit hybrid upwinding for compositional multiphase flow
in porous media", CMAME 388 (2022) 114288: a 100 m x 100 m closed vertical box (100 x 100 cells
of 1 m) in which a heavy fluid (water, rho=1500, initially top 10%), a light fluid (gas,
rho=500, bottom 10%) and the intermediate fluid(s) segregate by gravity through SEVEN horizontal
impermeable barrier layers with openings.

Parametrized by the phase count via ``build_params(nphase, scheme)`` / ``configure_phase_system``:
N=3 reproduces Bosma exactly; N=4 splits the oil into a mid-heavy + mid-light phase (densities
evenly spaced 1500..500). The immiscible one-component-per-phase machinery and the analytic
z -> s inversion  s_i = (z_i / rho_i) / sum_j (z_j / rho_j)  generalize to any N. The buoyancy
scheme ``"hu"`` = HU-BM(mp): the mobility-product buoyant term (classical Lee/Hamon U^HU),
obtained with ``fractional_flow=False`` + ``buoyancy_upwinding="hybrid"``.

    Run:  python porepy_2d_solver.py [--nphase N] [--scheme hu]   (NOT auto-run; heavy).

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

import logging
import os
import time

from typing import Callable, Optional, Sequence, cast  # noqa: E402

import numpy as np  # noqa: E402
import porepy as pp  # noqa: E402

logger = logging.getLogger(__name__)
from porepy.models.abstract_equations import LocalElimination  # noqa: E402

# Absolute imports (like geothermal_H2O_low_NaCl_content_fig_5.py) so the market modules'
# internal ``from ...obl_sampler import VTKSampler`` resolves. Requires porepy importable.
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import FlowModelBase  # noqa: E402,E501
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E402,E501
    GeometryBarriers2D,
    _BARRIER_LAYERS_FIG,
)
from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E402
    BC_three_phase_closed,
)
# The initial condition is defined inline (IC_NphaseSegregation) so it generalizes to any N;
# the 3-phase market class ``IC_three_phase_segregation`` is its N=3 special case.

# --------------------------------------------------------------------------------------- #
#  Fluid + rock constants (WA-HU Ex. 6.3).  Mega-scaled units: p[MPa], mu[MPa.s], rho[kg/m3].
# --------------------------------------------------------------------------------------- #
to_Mega = 1.0e-6

# --------------------------------------------------------------------------------------- #
#  Phase system (N phases; reconfigured by configure_phase_system / build_params(nphase=...))
# --------------------------------------------------------------------------------------- #
# Densities are evenly spaced heaviest -> lightest, so N=3 reproduces Bosma [1500, 1000, 500]
# EXACTLY and N=4 splits the intermediate (oil) into a MID-HEAVY + MID-LIGHT phase
# [1500, 1167, 833, 500].  The immiscible one-component-per-phase machinery and the analytic
# z -> s inversion  s_i = (z_i / rho_i) / sum_j (z_j / rho_j)  generalize to any N.
MU = 1.0e-3                     # phase viscosity [kg/(m.s)] = 1 cP (all phases; scaled to_Mega)
# Per-phase CONSTANT specific enthalpies [MJ/kg], evenly spaced in [H_MIN, H_MAX].  DENSER phase
# -> LOWER enthalpy, so the per-phase array ``H_PHASE`` is ASCENDING while ``RHO`` is DESCENDING
# (phase 0 densest -> H_MIN, phase N-1 lightest -> H_MAX).  This makes the initial enthalpy a
# function of the saturation layering + phase densities (see ic_values_enthalpy) and, via the
# caloric closure T = h/C_P, the initial temperature too (denser phase -> colder).
H_MIN, H_MAX = 1.0, 3.0         # phase specific-enthalpy bounds [MJ/kg]
C_P = 0.0035                       # caloric coupling  h = C_P * T  ->  T = h / C_P. Un-pins T (it is
#                                 no longer set to 0): the conduction K_e grad(T) = (K_e/C_P) grad(h)
#                                 makes temperature a genuine elliptic unknown, but that block is
#                                 regularized by the energy accumulation d(U)/dh, so it stays
#                                 solvable -- only the incompressible pressure block is constrained.


def _phase_names(n: int) -> list[str]:
    if n == 3:
        return ["water", "oil", "gas"]        # Bosma names (N=3 back-compat)
    return ["water"] + [f"oil{k}" for k in range(1, n - 1)] + ["gas"]


def _component_names(n: int) -> list[str]:
    if n == 3:
        return ["H2O", "C5H12", "CH4"]        # Bosma names (N=3 back-compat)
    return ["H2O"] + [f"C{k}" for k in range(1, n)]


NPHASE = 3
RHO = np.linspace(1500.0, 500.0, NPHASE)      # [kg/m^3], phase 0 heaviest .. N-1 lightest
H_PHASE = np.linspace(H_MIN, H_MAX, NPHASE)   # [MJ/kg] per phase, ASCENDING (phase 0 densest -> H_MIN)
PHASE_NAMES = _phase_names(NPHASE)
COMPONENT_NAMES = _component_names(NPHASE)
# N=3 back-compat scalar density/viscosity/enthalpy aliases (used by WaterEOS/OilEOS/GasEOS).
rho_w, rho_o, rho_g = 1500.0, 1000.0, 500.0
mu_w = mu_o = mu_g = MU
h_w, h_o, h_g = float(H_PHASE[0]), float(H_PHASE[1]), float(H_PHASE[2])   # 0.5, 1.0, 1.5

milli_darcy = 9.869233e-16          # 1 mD in m^2
k_rock = 1000.0 * milli_darcy          # homogeneous rock permeability (k = 1 mD)
porosity = 0.3
BARRIER_K_FACTOR = 1.0e-4           # barrier cells get k * this (effectively impermeable)

# Optional mixed-dimensional CONDUCTIVE fractures (params["fractures"]=True; --md).  The
# equi-dimensional barriers are UNCHANGED; the fractures are 1D lines embedded in the 2D matrix.
FRACTURE_K_FACTOR = 1.0           # fracture (dim<nd) permeability = k * this (1000x matrix)
FRACTURE_APERTURE = 1.0e-6          # fracture aperture [m] -> specific volume of the 1D fracture


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


def make_eos(components, rho: float, mu: float = MU, h: float = 1.0) -> BaseEOS:
    """Constant-property EOS instance for one phase with the given density and specific enthalpy
    (N-phase path). ``h`` defaults to the midpoint; callers pass the per-phase ``H_PHASE[i]``."""
    class _PhaseEOS(BaseEOS):
        _rho, _mu, _h = rho, mu, h
    return _PhaseEOS(components)


# --------------------------------------------------------------------------------------- #
#  Local-elimination closures (mirror buoyancy_flow_model.py): dependent saturations from
#  the overall fractions z, immiscibility chi = 1/0, and temperature == 0.
#  deps = (pressure, enthalpy, z_C5H12, z_CH4).  Each returns (values, zero-derivatives).
# --------------------------------------------------------------------------------------- #
def _clip_to_simplex(z_indep: list) -> list:
    """Project the independent overall fractions ``[z_1, ..., z_{N-1}]`` onto the valid simplex
    (each ``>= 0``, ``sum <= 1`` so the by-unity reference ``z_0 = 1 - sum >= 0``). Clipping each
    ``z_i`` independently would let a stray Newton iterate push the derived saturations off the
    partition; projecting here keeps every derived saturation in ``[0, 1]`` and summing to 1."""
    z = [np.clip(zk, 0.0, 1.0) for zk in z_indep]
    total = sum(z) if z else 0.0
    scale = np.where(total > 1.0, 1.0 / np.maximum(total, 1.0e-30), 1.0)
    return [zk * scale for zk in z]


def _saturations_from_z(z_indep: list) -> list:
    """N-phase immiscible ``z -> s`` inversion (one component per phase):
    ``s_i = (z_i / rho_i) / sum_j (z_j / rho_j)``, with the reference fraction
    ``z_0 = 1 - sum(z_1..z_{N-1})``. Reproduces the hardcoded 3-phase formula exactly."""
    z = _clip_to_simplex(list(z_indep))
    z0 = np.clip(1.0 - (sum(z) if z else 0.0), 0.0, 1.0)
    z_all = [z0] + z
    weighted = [z_all[j] / RHO[j] for j in range(NPHASE)]
    denom = sum(weighted)
    denom = np.where(denom > 0.0, denom, 1.0)
    return [np.clip(weighted[i] / denom, 0.0, 1.0) for i in range(NPHASE)]


def _make_saturation_func(i: int):
    """Closure: phase ``i``'s saturation from ``deps = (p, h, z_1, ..., z_{N-1})``."""
    def f(*deps):
        s = _saturations_from_z(list(deps[2:2 + NPHASE - 1]))
        nc = len(deps[0])
        return s[i], np.zeros((len(deps), nc))
    return f


def _chi(active: bool):
    def f(*deps):
        nc = len(deps[0])
        vals = (np.ones(nc) if active else np.zeros(nc))
        # Clip to [0, 1] (not [eps, 1]) so the by-unity reference partial fraction
        # x_ref = 1 - sum(others) also stays non-negative.
        return np.clip(vals, 0.0, 1.0), np.zeros((len(deps), nc))
    return f


# Immiscibility + z->s maps, rebuilt for the active N by ``configure_phase_system``. Component i
# (>= 1) lives only in phase i; phase 0 (water) is the reference (by-unity saturation).
saturation_functions_map: dict = {}
chi_functions_map: dict = {}


def configure_phase_system(nphase: int) -> None:
    """Configure the module for ``nphase`` phases: evenly-spaced densities, phase/component
    names, and the (name-keyed) saturation + immiscibility maps consumed by
    :class:`SecondaryEquations3N`. N=3 reproduces the Bosma names/maps exactly; N=4 splits the
    oil into a mid-heavy + mid-light phase."""
    global NPHASE, RHO, H_PHASE, PHASE_NAMES, COMPONENT_NAMES
    global saturation_functions_map, chi_functions_map
    NPHASE = int(nphase)
    RHO = np.linspace(1500.0, 500.0, NPHASE)
    H_PHASE = np.linspace(H_MIN, H_MAX, NPHASE)     # ascending: denser phase -> lower enthalpy
    PHASE_NAMES = _phase_names(NPHASE)
    COMPONENT_NAMES = _component_names(NPHASE)
    saturation_functions_map = {
        PHASE_NAMES[i]: _make_saturation_func(i) for i in range(1, NPHASE)
    }
    chi_functions_map = {
        f"{COMPONENT_NAMES[i]}_{PHASE_NAMES[j]}": _chi(i == j)
        for i in range(1, NPHASE) for j in range(NPHASE)
    }


configure_phase_system(NPHASE)   # N=3 default (Bosma): {"oil","gas"} + C5H12/CH4 chi map


def temperature_func(*deps):
    """Caloric closure ``T = h / C_P`` (``deps = p, h, z_1..z_{N-1}``). NOT pinned to a constant:
    the non-zero ``dT/dh = 1/C_P`` couples temperature to enthalpy in the Jacobian, so the energy
    balance + Fourier conduction ``K_e grad(T)`` make T a genuine elliptic unknown. That block is
    regularized by the energy accumulation (``d(U)/dh``), so it needs no constraint -- only the
    incompressible pressure block does (see :class:`_LagrangeConstrainedSolve`)."""
    nc = len(deps[0])
    d = np.zeros((len(deps), nc))
    d[1, :] = 1.0 / C_P                              # dT/dh = 1/C_P (h is deps[1])
    return deps[1] / C_P, d


# --------------------------------------------------------------------------------------- #
#  Fluid mixture (3 components, 3 immiscible phases) and secondary-equation eliminations
# --------------------------------------------------------------------------------------- #
class FluidMixture3N(pp.PorePyModel):
    def get_components(self) -> Sequence[pp.FluidComponent]:
        # COMPONENT_NAMES[0] (H2O) = reference (dependent z); the rest are the independent
        # overall fractions. Rebuilt for the active N by configure_phase_system.
        return [pp.FluidComponent(name=n) for n in COMPONENT_NAMES]

    def get_phase_configuration(self, components):
        # first phase (water, PHASE_NAMES[0]) is the reference phase (dependent saturation);
        # the lightest phase is the gas state, the rest liquid; densities from RHO.
        cfg = []
        for i in range(NPHASE):
            state = (pp.compositional.PhysicalState.gas if i == NPHASE - 1
                     else pp.compositional.PhysicalState.liquid)
            cfg.append((state, PHASE_NAMES[i],
                        make_eos(components, float(RHO[i]), h=float(H_PHASE[i]))))
        return cfg

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
#  N-phase initial condition (generalizes IC_three_phase_segregation to any N)
# --------------------------------------------------------------------------------------- #
class IC_NphaseSegregation(pp.PorePyModel):
    """N-phase gravity-segregation initial condition.

    Heaviest phase (phase 0) fills the top 10 % band, lightest (phase N-1) the bottom 10 %, and
    the N-2 interior phases split the middle 80 % equally. Immiscible: component ``i >= 1`` fills
    band ``i`` entirely, so the reference component (H2O) fills the top band by unity. At N=3
    this is exactly the Bosma layering (water top, oil middle, gas bottom).
    """

    _height = 100.0          # m, vertical (y) extent of the box
    _p_ref = 10.0            # reference pressure [MPa] (closed incompressible -> arbitrary)

    def _band_bounds(self) -> list:
        """Descending y-boundaries ``[H, 0.9H, ..., 0.1H, 0]`` (length NPHASE+1); band ``k``
        spans ``(b[k+1], b[k]]`` (band 0 = top 10 % heaviest, band N-1 = bottom 10 % lightest).

        Mirrors ``hamon_2d_solver._band_boundaries`` EXACTLY: the loop already ends at the last
        interior boundary ``0.1H`` (k = N-2 gives ``0.9H - 0.8H = 0.1H``), so only the bottom
        boundary ``0.0`` is appended.  (A previous ``+= [0.1H, 0.0]`` double-counted ``0.1H``,
        which collapsed the lightest phase's band to empty and left the bottom 10 % uncovered ->
        filled by the reference component; gas was absent from the IC.)"""
        H = self.units.convert_units(self._height, "m")
        if NPHASE == 2:
            return [H, 0.5 * H, 0.0]
        b = [H, 0.9 * H]
        w = 0.8 * H / (NPHASE - 2)
        b += [0.9 * H - k * w for k in range(1, NPHASE - 1)]
        b.append(0.0)
        return b

    def _band_masks(self, sd: pp.Grid) -> list:
        y = sd.cell_centers[1]
        b = self._band_bounds()
        return [(y <= b[k]) & (y > b[k + 1]) for k in range(NPHASE)]

    # NOTE: the initial PRESSURE is hydrostatic (FlowModel.ic_values_pressure, consistent with
    # the layering so grad(p) - rho g = 0 within each band); no constant-pressure IC here.

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        """Initial mixture specific enthalpy from the layered saturations, phase densities and
        per-phase enthalpies -- the mass-weighted mean

            h = sum_k (rho_k s_k h_k) / sum_k (rho_k s_k).

        With the layered IC (one phase per band, s_k = 1 in band k) this is exactly h_k = H_PHASE[k]
        inside band k, so the initial enthalpy is a step profile in y (denser top phase -> lower h).
        The caloric closure T = h/C_P then sets the initial temperature (denser phase -> colder)."""
        masks = self._band_masks(sd)                    # s_k = 1 in band k (partition of the domain)
        num = np.zeros(sd.num_cells)
        den = np.zeros(sd.num_cells)
        for k in range(NPHASE):
            num[masks[k]] += RHO[k] * H_PHASE[k]        # rho_k s_k h_k
            den[masks[k]] += RHO[k]                      # rho_k s_k
        return num / den

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        masks = self._band_masks(sd)
        comps = self.fluid.components
        z = np.zeros(sd.num_cells)
        for i in range(1, NPHASE):            # non-reference component i -> band i
            if component == comps[i]:
                z[masks[i]] = 1.0
        return z                              # reference (H2O) = 1 - sum (top band)

    def ic_values_saturation(self, sd: pp.Grid) -> list:
        masks = self._band_masks(sd)
        return [np.where(masks[i], 1.0, 0.0) for i in range(1, NPHASE)]   # s_1..s_{N-1}

    def initial_condition(self) -> None:
        super().initial_condition()
        phases = list(self.fluid.phases)
        for sd in self.mdg.subdomains():
            s_list = self.ic_values_saturation(sd)      # [s_1, ..., s_{N-1}]
            for i in range(1, NPHASE):
                ph = phases[i]
                # Seed only quantities that are still independent variables (a substituted
                # saturation is computed from z -> no variable to seed).
                if self.has_independent_saturation(ph):
                    self.equation_system.set_variable_values(
                        s_list[i - 1], [ph.saturation([sd])], 0, 0)


# --------------------------------------------------------------------------------------- #
#  All-Neumann (closed) BCs for BOTH the Darcy (pressure) and Fourier (temperature) fluxes,
#  so the singular pressure block is fixed by the solver's null-mean constraint.
# --------------------------------------------------------------------------------------- #
class BC_all_neumann(BC_three_phase_closed):
    """No-flow (all-Neumann) BCs on every boundary for the Darcy AND Fourier fluxes -- removes the
    pressure and temperature Dirichlet data. Under all-Neumann the pressure block (incompressible,
    no accumulation) is singular in its constant mode; the enthalpy/temperature block, though also
    elliptic (Fourier conduction), is regularized by the energy accumulation and stays solvable. So
    a single null-mean constraint, ``Sum(p) = 0``, makes the whole system solvable (see the solver)."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd)          # no Dirichlet facets -> all Neumann (no-flow)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd)          # all Neumann (no conductive flux across bdry)


# --------------------------------------------------------------------------------------- #
#  Null-mean-constrained linear solve (localized here). Assemble the NORMAL (pressure-singular)
#  system, then add one zero-mean constraint -- Sum(dp)=0 at the pressure DOF indices -- to fix
#  the constant-pressure nullspace, and solve. CPR/PETSc mirrors hamon_2d_solver.
# --------------------------------------------------------------------------------------- #
class _LagrangeConstrainedSolve(pp.PorePyModel):
    """Override :meth:`solve_linear_system` for the all-Neumann system.

    Under all-Neumann the pressure block (incompressible, no accumulation) is elliptic and singular
    in its constant mode; the enthalpy/temperature block, though elliptic too, is regularized by the
    energy accumulation and stays solvable (verified: the measured nullspace of ``A`` is 1-D and
    lives entirely in pressure). This solve borders the assembled ``A`` with one zero-mean
    constraint row/col per variable in ``params['null_mean_variables']`` (default ``['pressure']``)
    -- an indicator of that variable's DOFs from ``equation_system.dofs_of`` -- so ``Sum(dp)=0``
    removes the constant-pressure direction, then solves. Set ``params['report_nullspace']`` to have
    the FIRST call report the numerically-measured nullspace dimension of ``A``.
    """

    def _null_mean_dof_indices(self) -> list:
        """Global DOF indices for each null-mean-constrained (elliptic) variable in
        ``params['null_mean_variables']`` (default ``['pressure']``), taken on the EQUIDIMENSIONAL
        MATRIX subdomain only.

        The closed (all-Neumann) domain has a SINGLE floating pressure level: the whole matrix +
        fracture network shifts together by one constant (the 1-D kernel of the Jacobian).  The
        FRACTURE pressures are NOT independently singular -- each is pinned to the matrix through the
        Robin-type interface (mortar) Darcy coupling.  So a single ``Sum(dp_matrix)=0`` on the matrix
        pressure fixes that one constant (the matrix indicator has a non-zero component along the
        kernel), and the fracture pressures follow from the coupling; there is no reason to include
        them in the constraint.  In the fixed-dimensional case (matrix is the only subdomain) this is
        exactly the original single constraint."""
        es = self.equation_system
        matrix = self.mdg.subdomains(dim=self.nd)          # equidimensional matrix only
        names = self.params.get("null_mean_variables", ["pressure"])
        return [np.asarray(es.dofs_of([es.md_variable(nm, matrix)]), dtype=int) for nm in names]

    def solve_linear_system(self) -> np.ndarray:
        A, b = self.linear_system
        A = A.tocsr()
        n = A.shape[0]
        null_mean_dofs = self._null_mean_dof_indices()

        if self.params.get("report_nullspace", False) and not getattr(
            self, "_nullspace_reported", False
        ):
            self._nullspace_reported = True
            es = self.equation_system
            sds = self.mdg.subdomains()
            blocks = {nm: np.asarray(es.dofs_of([es.md_variable(nm, sds)]), dtype=int)
                      for nm in ("pressure", "enthalpy", "temperature")}
            _, sv, Vt = np.linalg.svd(A.toarray())              # coarse-grid diagnostic only
            nz = int((sv < 1.0e-8 * sv[0]).sum())
            v = Vt[-1]                                          # smallest right singular vector
            vn = np.linalg.norm(v)
            supp = {nm: np.linalg.norm(v[idx]) / vn for nm, idx in blocks.items()}
            print(f"  [null-mean-diag] ndof={n}  nullspace_dim(A)={nz}  "
                  f"n_constraints={len(null_mean_dofs)}",
                  flush=True)
            print(f"    smallest 5 sv/max = {np.array2string(sv[-5:]/sv[0], precision=2)}",
                  flush=True)
            print(f"    nullvector norm-fraction per block: "
                  f"pressure={supp['pressure']:.3f} enthalpy={supp['enthalpy']:.3f} "
                  f"temperature={supp['temperature']:.3f}", flush=True)

        b = np.asarray(b, dtype=float)
        if self.params.get("lagrange_linear_solver", "cpr") == "cpr":
            try:
                # _schur_cpr_solve validates its own (projected) accuracy and raises on failure.
                return self._schur_cpr_solve(A, b)
            except Exception as exc:
                if not getattr(self, "_cpr_fallback_warned", False):
                    self._cpr_fallback_warned = True
                    print(f"  [warn] Schur-reduced CPR unavailable ({exc!r}); falling back "
                          f"to the SciPy bordered solve for this run", flush=True)
        return self._scipy_bordered_solve(A, b, null_mean_dofs)

    @staticmethod
    def _equation_for_variable(varname: str, eq_names: list) -> Optional[str]:
        """The equation that determines ``varname`` (PorePy names them independently of variables):
        pressure<->mass_balance, enthalpy<->energy_balance, z_<c><->component_mass_balance_<c>, each
        interface (mortar) flux <-> its ``<var>_equation`` (present once fractures are added), and
        every locally-eliminated variable <-> its ``elimination_of_<var>_on_grids_...`` equation."""
        if varname == "pressure":
            return "mass_balance_equation"
        if varname == "enthalpy":
            return "energy_balance_equation"
        if varname.startswith("z_"):
            return "component_mass_balance_equation_" + varname[2:]
        if varname.startswith("interface_"):          # interface_darcy_flux -> ..._equation, etc.
            return varname + "_equation"
        cands = [e for e in eq_names if e.startswith(f"elimination_of_{varname}_on_grids")]
        return cands[0] if cands else None

    # The ELLIPTIC variables -> the AMG block: pressure (Darcy Laplacian) and enthalpy (once the
    # local T = h/C_P closure is Schur-eliminated, the conduction K_e grad(T) = (K_e/C_P) grad(h)
    # is a genuine Laplacian in h). Everything else -> the ILU block. NOTE: this is the correct AMG
    # set for the SCHUR-REDUCED (p, h, z) system; on the raw un-reduced Jacobian the h-Laplacian is
    # still buried in the temperature columns, so GAMG cannot see it and the CPR falls back.
    _ELLIPTIC_VARS = ("pressure", "enthalpy")

    def _primary_secondary_indices(self, n: int):
        """Partition the assembled DOFs into PRIMARY (p, h, z -- balance equations) and SECONDARY
        (T, s, x -- local elimination equations), each aligned equation-row <-> variable-column
        (PorePy does not align them). Primary is ordered ELLIPTIC-first ({p, h}), so the Schur
        complement's leading ``n_elliptic`` block is the {p, h} elliptic system for AMG and ``z`` --
        after it -- goes to ILU. Returns (primary_cols, primary_rows, secondary_cols, secondary_rows,
        n_pressure, n_elliptic)."""
        es = self.equation_system
        aei = es.assembled_equation_indices
        eq_names = list(aei.keys())
        vars_by_name: dict = {}
        for v in es.variables:                          # atomic Variable objects, one per grid
            vars_by_name.setdefault(v.name, []).append(v)
        var_names = sorted(vars_by_name)

        def eq_of(v):
            return self._equation_for_variable(v, eq_names)

        def is_secondary(v):
            eq = eq_of(v)
            return eq is not None and eq.startswith("elimination_of_")

        # ELLIPTIC (-> AMG): pressure + enthalpy, on ALL subdomains (matrix + fractures).  Interface
        # mortar-flux variables (interface_darcy/fourier/enthalpy_flux) are NON-elliptic, NON-secondary
        # -> primary, so they land in the reduced system's ILU (z) tail with the transported z.
        elliptic = [v for v in self._ELLIPTIC_VARS if v in var_names]
        primary_vars = elliptic + [
            v for v in var_names if v not in elliptic and not is_secondary(v)]
        secondary_vars = [v for v in var_names if is_secondary(v)]

        def cols(vs):
            # Gather each variable's global DOFs by NAME across ALL grids -- subdomains AND interfaces.
            # (md_variable(name, subdomains) would miss the interface mortar variables that fractures
            # add; pressure/enthalpy/z/secondaries automatically span matrix + fracture subdomains.)
            return [np.asarray(es.dofs_of(vars_by_name[v]), dtype=int) for v in vs]

        def rows(vs):
            return [np.asarray(aei[eq_of(v)], dtype=int) for v in vs]

        p_cols, p_rows = cols(primary_vars), rows(primary_vars)
        s_cols, s_rows = cols(secondary_vars), rows(secondary_vars)
        primary_cols = np.concatenate(p_cols)
        primary_rows = np.concatenate(p_rows)
        secondary_cols = np.concatenate(s_cols) if s_cols else np.zeros(0, dtype=int)
        secondary_rows = np.concatenate(s_rows) if s_rows else np.zeros(0, dtype=int)
        if not (np.array_equal(np.sort(np.concatenate([primary_cols, secondary_cols])), np.arange(n))
                and np.array_equal(
                    np.sort(np.concatenate([primary_rows, secondary_rows])), np.arange(n))):
            raise RuntimeError("Schur partition: primary+secondary indices do not partition [0,n)")
        n_pressure = len(p_cols[0])
        n_elliptic = sum(len(p_cols[i]) for i in range(len(elliptic)))
        return (primary_cols, primary_rows, secondary_cols, secondary_rows, n_pressure, n_elliptic)

    def _schur_cpr_solve(self, A, b) -> np.ndarray:
        """Pure-algebraic Schur reduction of the LOCAL secondary closures (T, s, x), then CPR on the
        reduced (p, h, z) primary system, then back-substitution. The secondary block ``A_ss`` is
        local (block-diagonal per cell) so its factorization is cheap and the Schur complement stays
        sparse. Eliminating T substitutes ``K_e grad(T) -> (K_e/C_P) grad(h)``, so the reduced energy
        equation is a genuine h-Laplacian and ``{p, h} -> GAMG`` becomes effective (the reason the
        raw un-reduced CPR could not work)."""
        import scipy.sparse as sps
        from scipy.sparse.linalg import splu, spsolve

        t0 = time.perf_counter()
        n = A.shape[0]
        pc, pr, sc, sr, n_p, n_ell = self._primary_secondary_indices(n)
        A = A.tocsc()
        App = A[pr][:, pc].tocsr()
        Aps = A[pr][:, sc].tocsr()
        Asp = A[sr][:, pc].tocsc()
        Ass = A[sr][:, sc].tocsc()
        bp, bs = b[pr], b[sr]
        t_blocks = time.perf_counter()

        # The secondary block A_ss is the Jacobian of the LOCAL closures ``var - func(primary) = 0``:
        # its diagonal is the identity and it has no secondary<->secondary coupling, so A_ss == I (the
        # eliminations depend only on primary DOFs).  Detect that and skip the factorization entirely;
        # fall back to a sparse LU only if some closure is non-trivial.
        Ass = Ass.tocsr()
        is_identity = (Ass.shape[0] == Ass.nnz
                       and np.allclose(Ass.data, 1.0)
                       and np.array_equal(Ass.indices, np.arange(Ass.shape[0])))
        if is_identity:
            Ainv_Asp = Asp                                     # A_ss^{-1} = I
            Ainv_bs = bs
            lu = None
        else:
            lu = splu(Ass.tocsc())                             # local block-diagonal -> still cheap
            Ainv_Asp = spsolve(Ass.tocsc(), Asp)
            if not sps.issparse(Ainv_Asp):
                Ainv_Asp = sps.csc_matrix(Ainv_Asp)
            Ainv_bs = lu.solve(bs)
        t_fact = time.perf_counter()

        S = (App - Aps @ Ainv_Asp).tocsr()                     # Schur complement (p, h, z)
        g = bp - Aps @ Ainv_bs                                 # reduced RHS
        t_schur = time.perf_counter()

        xp, cpr_its = self._cpr_petsc_solve(S, g, n_p, n_ell)  # CPR on the reduced primary system
        t_cpr = time.perf_counter()

        # Honest accuracy gate.  During Newton the reduced system is genuinely INCONSISTENT: v = [1_p;
        # 0] is BOTH the right nullvector (constant pressure) AND the left nullvector (summing the mass
        # rows = boundary flux = 0), so v^T g = the not-yet-converged total-mass imbalance is nonzero.
        # The null-mean solution -- exactly like SciPy's bordered solve -- therefore leaves an
        # IRREDUCIBLE raw residual |v^T g| along v; only the residual with that conservation direction
        # PROJECTED OUT measures solver error.  A correct CPR drives it to ~1e-11 (matches the bordered
        # solution to ~1e-12); a GAMG that ignores the singular {p, h} mode stalls near ~1e-4.
        r = S @ xp - g
        vp = np.zeros(len(xp)); vp[:n_p] = 1.0
        r_proj = r - (vp @ r) / (vp @ vp) * vp
        rel = np.linalg.norm(r_proj) / max(np.linalg.norm(g), 1.0e-30)
        if rel > 1.0e-8:
            raise RuntimeError(f"CPR projected residual too large for Newton (rel={rel:.1e})")

        rhs_s = bs - Asp @ xp
        xs = rhs_s if lu is None else lu.solve(rhs_s)          # back-substitute the secondaries
        t_end = time.perf_counter()

        logger.info(
            "Schur-CPR linear solve: %.3fs total | blocks %.3fs, A_ss %s %.3fs, "
            "Schur %.3fs, CPR %.3fs (%d KSP its, proj_res %.1e), back-sub %.3fs "
            "[reduced %d = {p,h} %d + z %d, secondary %d]",
            t_end - t0, t_blocks - t0, "==I" if is_identity else "LU", t_fact - t_blocks,
            t_schur - t_fact, t_cpr - t_schur, cpr_its, rel, t_end - t_cpr,
            S.shape[0], n_ell, S.shape[0] - n_ell, len(sr),
        )

        x = np.empty(n, dtype=float)
        x[pc] = xp
        x[sc] = xs
        return x

    @staticmethod
    def _scipy_bordered_solve(A, b, null_mean_dofs) -> np.ndarray:
        """SciPy direct solve of the bordered saddle-point ``[[A, C^T],[C, 0]] [dx; lam] = [b; 0]``,
        one null-mean constraint row (indicator of the block DOFs) per entry of ``null_mean_dofs``.
        Returns ``dx``."""
        from scipy.sparse import csr_matrix, bmat
        from scipy.sparse.linalg import spsolve

        n = A.shape[0]
        nd = len(null_mean_dofs)
        C = np.zeros((nd, n))
        for k, dofs in enumerate(null_mean_dofs):
            C[k, dofs] = 1.0                                     # Sum(d<var>) = 0 null-mean row
        Cs = csr_matrix(C)
        M = bmat([[A, Cs.T], [Cs, None]], format="csr")         # bordered saddle-point
        rhs = np.concatenate([b, np.zeros(nd)])
        return np.asarray(spsolve(M, rhs)[:n])                  # drop the multipliers

    @staticmethod
    def _cpr_petsc_solve(S, g, n_p, n_ell, rtol=1.0e-10, maxit=300) -> np.ndarray:
        """PETSc FGMRES + CPR THREE-field split on the (elliptic-first ordered) reduced primary system
        ``S x = g``: pressure ``p`` [0, n_p) -> GAMG (+ constant null space -- it is singular in the
        constant mode), enthalpy ``h`` [n_p, n_ell) -> GAMG (elliptic h-Laplacian, regular), and the
        transported ``z`` [n_ell, n) -> ILU(0).  Splitting p and h into SEPARATE scalar GAMG fields is
        ~8x cheaper than one combined ``{p, h}`` block (25 vs ~200 FGMRES iterations at 1e4 cells):
        each is a clean scalar elliptic operator GAMG coarsens well, whereas the interleaved 2-field
        block defeats its coarse space.  The constant-pressure null space (first ``n_p`` DOFs) is
        attached to the global operator so no dense border is formed; that makes FGMRES return the
        null-space-orthogonal solution, i.e. ``sum(x_p) = 0`` -- exactly the null-mean gauge of the
        SciPy bordered solve.  Returns ``(x, n_iterations)``."""
        from petsc4py import PETSc

        S = S.tocsr()
        S.sort_indices()
        n = S.shape[0]
        mat = PETSc.Mat().createAIJ(
            size=(n, n),
            csr=(S.indptr.astype(PETSc.IntType), S.indices.astype(PETSc.IntType),
                 np.ascontiguousarray(S.data, dtype=PETSc.ScalarType)),
            comm=PETSc.COMM_SELF,
        )
        mat.assemble()

        def const_p_nullspace(m, upto):
            """One-vector NullSpace [1..1 on [0, upto), 0 else], normalized -- the constant-pressure
            singular mode of ``m``."""
            w = m.createVecRight()
            a = w.getArray()
            a[:] = 0.0
            a[:upto] = 1.0
            w.assemble()
            w.normalize()
            return PETSc.NullSpace().create(constant=False, vectors=[w], comm=PETSc.COMM_SELF)

        nsp = const_p_nullspace(mat, n_p)                       # global constant-pressure null space
        mat.setNullSpace(nsp)

        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(mat)
        ksp.setType("fgmres")                                   # flexible: CPR is a nonlinear PC
        ksp.setTolerances(rtol=rtol, atol=1.0e-50, max_it=maxit)

        pc = ksp.getPC()
        pc.setType("fieldsplit")
        is_p = PETSc.IS().createGeneral(
            np.arange(0, n_p, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        is_h = PETSc.IS().createGeneral(
            np.arange(n_p, n_ell, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        is_z = PETSc.IS().createGeneral(
            np.arange(n_ell, n, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        pc.setFieldSplitIS(("p", is_p), ("h", is_h), ("z", is_z))
        pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
        pc.setUp()
        kp, kh, kz = pc.getFieldSplitSubKSP()
        # pressure: elliptic Darcy operator, singular in the constant mode (S_pp 1 = 0) -> GAMG must be
        # told, else its coarse space is singular there and FGMRES meets rtol on a solution that still
        # carries an O(1e-4) constant-pressure error.
        kp.setType("preonly")
        kp.getPC().setType("gamg")
        App_block = kp.getOperators()[0]
        p_nsp = const_p_nullspace(App_block, n_p)              # whole block is pressure -> constant
        App_block.setNullSpace(p_nsp)
        App_block.setNearNullSpace(p_nsp)                      # GAMG coarse-space interpolation
        # enthalpy: after the T = h/C_P elimination the conduction is a genuine h-Laplacian, regular
        # (energy accumulation d(U)/dh regularizes it) -> its own scalar GAMG.
        kh.setType("preonly")
        kh.getPC().setType("gamg")
        # transported overall fractions z: hyperbolic -> ILU(0).
        kz.setType("preonly")
        kz.getPC().setType("ilu")

        xv = mat.createVecRight()
        bv = mat.createVecLeft()
        bv.setArray(np.ascontiguousarray(g, dtype=PETSc.ScalarType))
        # v is BOTH the right and left nullvector, so removing its component projects g onto range(S) --
        # the consistent RHS whose solution equals SciPy's bordered null-mean solution (sum(dp) = 0).
        nsp.remove(bv)
        ksp.solve(bv, xv)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f"PETSc CPR KSP diverged (reason {ksp.getConvergedReason()}, "
                               f"its={ksp.getIterationNumber()})")
        return xv.getArray().copy(), ksp.getIterationNumber()


# --------------------------------------------------------------------------------------- #
#  Interior CONDUCTIVE fractures (endpoints in reference metres on the 100-cell grid, so they lie
#  on cell faces and stay conforming under any cell_size = 1/k refinement -- nx a multiple of 100).
#  5 VERTICAL fractures cross the horizontal barriers PERPENDICULARLY; 5 HORIZONTAL fractures sit in
#  barrier-free bands (they never touch a barrier).  None reach the domain boundary and none intersect
#  each other, so the mixed-dimensional grid is exactly 2D matrix + ten 1D fractures (NO 0D points).
#  Validated against _BARRIER_LAYERS_FIG (see frac placement check).  Each entry is (x0, y0, x1, y1).
# --------------------------------------------------------------------------------------- #
# The horizontal fractures stay INSIDE the middle (oil) phase band (10 < y < 90) so no interface
# straddles an initial phase boundary -- H1/H5 are at y=88/12, NOT the band edges y=90/10, which
# otherwise put an oil fracture next to water (top) / a gas fracture next to oil (bottom) and made the
# advective enthalpy/component coupling blow up on the adjacent matrix cells at the first Newton step.
_FRACTURES_REF = [
    (20.0, 56.0, 20.0, 74.0),   # V1  crosses barrier at y~61.5
    (48.0, 44.0, 48.0, 68.0),   # V2  crosses barriers at y~54.5 and y~61.5
    (72.0, 35.0, 72.0, 50.0),   # V3  crosses barrier at y~41.5
    (30.0, 20.0, 30.0, 38.0),   # V4  crosses barrier at y~25.5
    (85.0, 12.0, 85.0, 30.0),   # V5  crosses barriers at y~17.5 and y~25.5
    (8.0, 88.0, 44.0, 88.0),    # H1  barrier-free, inside the oil band (below the y=90 water/oil edge)
    (56.0, 70.0, 92.0, 70.0),   # H2  barrier-free band
    (8.0, 48.0, 44.0, 48.0),    # H3  barrier-free band
    (56.0, 33.0, 92.0, 33.0),   # H4  barrier-free band
    (28.0, 12.0, 72.0, 12.0),   # H5  barrier-free, inside the oil band (above the y=10 oil/gas edge)
]


# --------------------------------------------------------------------------------------- #
#  Barriers by physical BOUNDING BOXES (refinement-independent), mirroring hamon_2d_solver
# --------------------------------------------------------------------------------------- #
class BarriersBoundingBox2D(GeometryBarriers2D):
    """``GeometryBarriers2D`` with a REFINEMENT-INDEPENDENT barrier mask.

    The base class places each barrier on a single index/figure ROW and index-scales the
    columns, so under mesh refinement the barriers thin out (a 1 m layer is 1 cell = 0.5 m thick
    at 200x200) and openings shift by half a cell.  Here every digitized segment is instead a
    FIXED PHYSICAL BOUNDING BOX derived from the 100-cell reference layout, and a cell is a
    barrier iff its CENTRE lies in any box -- exactly ``hamon_2d_solver.barrier_mask``.  So a 1 m
    layer stays 1 m thick at any resolution (2 cells at 200x200) and every opening keeps its exact
    x-span; at 100x100 this reproduces the base class mask cell-for-cell.
    """

    def _barrier_boxes(self) -> list:
        """Physical boxes ``(x_lo, x_hi, y_lo, y_hi)`` of every barrier segment, from the 100-cell
        digitized ``_BARRIER_LAYERS_FIG`` (figure row 0 = top, gravity down).  Figure row ``fig_r``
        is the physical band ``y in [L - (fig_r+1) dref, L - fig_r dref]``; inclusive columns
        ``[a, b]`` span ``x in [a dref, (b+1) dref]`` (``dref = L/100`` = one reference cell)."""
        length = self.units.convert_units(self._length, "m")     # square domain (both = _length)
        dref = length / 100.0                                    # one 100-cell reference cell
        boxes = []
        for fig_r, segments in _BARRIER_LAYERS_FIG.items():
            y_lo = length - (fig_r + 1) * dref
            y_hi = length - fig_r * dref
            for a, b in segments:
                boxes.append((a * dref, (b + 1) * dref, y_lo, y_hi))
        return boxes

    def barrier_cell_mask(self, sd: pp.Grid) -> np.ndarray:
        xc = sd.cell_centers[0]
        yc = sd.cell_centers[1]                                  # 0 = bottom, increases UPWARD
        mask = np.zeros(sd.num_cells, dtype=bool)
        for x_lo, x_hi, y_lo, y_hi in self._barrier_boxes():
            mask |= (xc >= x_lo) & (xc <= x_hi) & (yc >= y_lo) & (yc <= y_hi)
        return mask

    def meshing_arguments(self) -> dict:
        """Honour ``params['cell_size']`` (metres) so the mesh can be refined; default 1 m (100x100).
        Use ``cell_size = 1/k`` (nx a multiple of 100) to keep the barriers AND the integer-metre
        fracture endpoints exactly on cell faces."""
        cell = self.params.get("cell_size", self._cell)
        return {"cell_size": self.units.convert_units(cell, "m")}

    def set_fractures(self) -> None:
        """Ten conductive 1D fractures (only when ``params['fractures']``), conforming to the base
        mesh.  Reference-metre endpoints (``_FRACTURES_REF``) are scaled by the metre unit exactly
        as the domain/cell size, so they land on cell faces at any ``cell_size = 1/k`` resolution.
        The equi-dimensional barriers are untouched; these add the mixed-dimensional structure."""
        if not self.params.get("fractures", False):
            self._fractures = []
            return
        scale = self.units.convert_units(1.0, "m")               # 1 reference metre in solver units
        self._fractures = [
            pp.LineFracture(np.array([[x0, x1], [y0, y1]], dtype=float) * scale)
            for (x0, y0, x1, y1) in _FRACTURES_REF
        ]


# --------------------------------------------------------------------------------------- #
#  The model
# --------------------------------------------------------------------------------------- #
class FlowModel(
    _LagrangeConstrainedSolve,
    BarriersBoundingBox2D,
    FluidMixture3N,
    IC_NphaseSegregation,
    BC_all_neumann,
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

    def _rock_permeability_values(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cell-wise ROCK permeability (no mobility weighting), branching on dimension:
          * 2D matrix (``sd.dim == self.nd``): k, with barrier cells (``barrier_cell_mask``) set to
            k * BARRIER_K_FACTOR.
          * lower-dim fractures (``sd.dim < self.nd``): k * FRACTURE_K_FACTOR (fully conductive); the
            barrier mask is NOT applied (a fracture cell whose centre falls in a barrier box must stay
            conductive).
        Shared by :meth:`permeability` (subdomain tensor) and :meth:`normal_permeability` (interface),
        so both use the SAME rock permeability -- the total-mass (HU-BM(mp)) formulation applies the
        fluid mobility separately via the upwind mobility terms, never baked into the permeability."""
        size = sum(sd.num_cells for sd in subdomains)
        vals = np.full(size, self.solid.permeability)
        offset = 0
        for sd in subdomains:
            cell_vals = vals[offset:offset + sd.num_cells]
            if sd.dim == self.nd:                       # 2D matrix: barrier-masked rock
                cell_vals[self.barrier_cell_mask(sd)] = self.solid.permeability * BARRIER_K_FACTOR
            else:                                       # 1D/0D fractures: fully conductive
                cell_vals[:] = self.solid.permeability * FRACTURE_K_FACTOR
            offset += sd.num_cells
        return pp.wrap_as_dense_ad_array(vals, size, name="permeability")

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Homogeneous rock permeability with impermeable barrier cells, plus (when fractures are
        enabled) fully-conductive 1D fractures.  See :meth:`_rock_permeability_values`."""
        return self.isotropic_second_order_tensor(
            subdomains, self._rock_permeability_values(subdomains))

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Interface (normal) permeability = the lower-dimensional subdomain's ROCK permeability,
        projected to the mortar -- WITHOUT the total-mass-mobility weighting.

        The base ``MassWeightedPermeability.normal_permeability`` unconditionally returns
        ``total_mass_mobility * k`` (the fractional-flow diffusive tensor).  But this model runs the
        NON-fractional HU-BM(mp) formulation, whose subdomain :meth:`permeability` is rock-only and
        which applies the mobility separately (the mp buoyancy multiplies by
        ``lambda_gamma lambda_delta / lambda_T``, and the interface mobility is upwinded).  Keeping
        the mobility in ``normal_permeability`` therefore counts it TWICE on the matrix-fracture
        interface, making the interface buoyancy flux ``total_mass_mobility`` (~1e13) times too large
        and driving the adjacent matrix cells to zero saturation (``1/total_mobility`` -> NaN).  Using
        the rock permeability here matches the subdomain and removes the double weighting."""
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        kn = projection.secondary_to_mortar_avg() @ self._rock_permeability_values(subdomains)
        kn.set_name("normal_permeability")
        return kn

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
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2") * to_Mega
        b = self._band_bounds()                                     # descending [H, ..., 0]
        # column density integrated from height y up to the top boundary H, over the N bands
        column = np.zeros_like(np.asarray(y, dtype=float))
        for k in range(NPHASE):                                     # band k spans (b[k+1], b[k]]
            column = column + RHO[k] * np.maximum(0.0, b[k] - np.maximum(y, b[k + 1]))
        return self._p_ref + g * column


# --------------------------------------------------------------------------------------- #
#  Run configuration (module level), mirroring tp_tc_gravitational_segregation.py
# --------------------------------------------------------------------------------------- #
day = 86400.0

# --------------------------------------------------------------------------------------- #
#  Time stepping -- mirrors hamon_2d_solver.run(): backward Euler, nominal 1-day step,
#  reject-and-halve a non-converged step down to dt0/64 and grow it back toward dt0, and --
#  crucially -- hit the Fig-5 snapshot instants (0, 78, 571 days) EXACTLY.  The snapshot days
#  are placed in the TimeManager SCHEDULE: PorePy clips any step that would overshoot a
#  scheduled time (dt := t_sched - t; see time_step_control.py), so ``time_manager.time`` lands
#  on 78 d and 571 d to machine precision, and ``times_to_export`` (matched with ``np.isclose``
#  against ``time_manager.time`` in data_saving_model_mixin) then writes the VTU there exactly.
#  This is the robust cure for "adaptive dt exports at the wrong instants": the must-hit times
#  are SCHEDULED, not left to the adaptive cadence.
# --------------------------------------------------------------------------------------- #
T_END_DAYS = 78.0                        # hamon T_END
SNAP_DAYS = (0.0, 78.0)                   # hamon SNAP_DAYS -- the Fig-5 saturation-map instants
DT_DAYS = 1.0                             # nominal step [days] -- the constant-dt march value
DT_INIT_DAYS = 0.01                      # INITIAL adaptive step [days] -- start small on the stiff,
#                                           fully density-inverted IC (denser fluid over lighter)
DT_MAX_DAYS = 1.0                         # MAXIMUM (cap) adaptive step [days] -- never exceeded;
#                                           the floor is DT_MAX_DAYS/64


def make_time_manager(t_end_days: float = T_END_DAYS, dt_days: float = DT_DAYS,
                      snap_days: Sequence[float] = SNAP_DAYS,
                      constant_dt: bool = False,
                      dt_init_days: float = DT_INIT_DAYS,
                      dt_max_days: float = DT_MAX_DAYS) -> pp.TimeManager:
    """A FRESH TimeManager per run (it is stateful -> must never be shared between models).

    The snapshot instants + the horizon go into the schedule so they are hit exactly.
    ``constant_dt=True`` is a pure ``dt_days``-day march (``dt_days`` must divide every schedule
    interval) -- fully deterministic, but a non-converged step aborts the run (no cut).
    ``constant_dt=False`` (default) is the hamon analog: the step STARTS at ``dt_init_days``, may
    GROW to the CAP ``dt_max_days`` when Newton is easy, HALVES on a non-converged step down to
    ``dt_max_days/64``, then grows back -- and still hits the snapshots.

    Time-step controls (all in DAYS; module defaults ``DT_*_DAYS``):
      * ``dt_days``      -- the CONSTANT-dt march value (only used when ``constant_dt=True``).
      * ``dt_init_days`` -- INITIAL adaptive step (default ``DT_INIT_DAYS``).  Small on a stiff IC.
      * ``dt_max_days``  -- MAXIMUM adaptive step / cap (default ``DT_MAX_DAYS``); never exceeded.
    The adaptive floor is ``dt_max_days/64`` (but never above ``dt_init_days``, so it stays valid).
    """
    sched = sorted({0.0, *(d * day for d in snap_days if d <= t_end_days + 1e-9),
                    t_end_days * day})
    if constant_dt:                                  # pure constant march at dt_days
        return pp.TimeManager(schedule=sched, dt_init=dt_days * day, constant_dt=True,
                              iter_max=20, print_info=True)
    dt_init = dt_init_days * day
    dt_max = dt_max_days * day
    dt_min = min(dt_max / 64.0, dt_init)             # floor; keep dt_init within [dt_min, dt_max]
    dt_init = min(max(dt_init, dt_min), dt_max)      # clamp the initial step into [dt_min, dt_max]
    return pp.TimeManager(
        schedule=sched, dt_init=dt_init, constant_dt=False,
        dt_min_max=(dt_min, dt_max),           # floor dt_max/64, cap dt_max (never overshoot it)
        iter_optimal_range=(4, 10),            # grow dt when Newton is easy, shrink when it is hard
        iter_relax_factors=(0.5, 2.0),         # halve on a cut / double on grow-back (hamon *0.5, *2)
        recomp_factor=0.5, recomp_max=8,       # reject-and-halve, up to 8 consecutive cuts
        iter_max=20, print_info=True,          # hamon Newton cap = 20
    )


def make_times_to_export(snap_days: Sequence[float] = SNAP_DAYS) -> list:
    """VTU/PVD written exactly at each snapshot instant (matched to ``time_manager.time``)."""
    return [d * day for d in snap_days]

solid_constants = pp.SolidConstants(
    permeability=k_rock,                               # 1 mD
    porosity=porosity,                                 # 0.3
    thermal_conductivity=2.0 * to_Mega,                # unused (isothermal)
    density=2500.0,
    specific_heat_capacity=1000.0 * to_Mega,
    residual_aperture=FRACTURE_APERTURE,               # 1D fracture aperture (no effect without fractures)
)

# HU-BM scheme -> model parametrization. "hu" = HU-BM(mp): the mobility-product buoyant term
# (classical Lee/Hamon U^HU), reached via ``fractional_flow=False`` (the total-mass formulation,
# whose FluidBuoyancy non-fractional branch is the mobility-product form) + hybrid upwinding.
_SCHEME_CONFIG = {
    "hu": dict(fractional_flow=False, mass_mobility_weighted_permeability=False,
               buoyancy_upwinding="hybrid"),
}


def build_params(nphase: int = 3, scheme: str = "hu", *, t_end_days: float = T_END_DAYS,
                 dt_days: float = DT_DAYS, dt_init_days: float = DT_INIT_DAYS,
                 dt_max_days: float = DT_MAX_DAYS, snap_days: Sequence[float] = SNAP_DAYS,
                 constant_dt: bool = False, fractures: bool = False,
                 ad_backend: Optional[str] = None, **overrides) -> dict:
    """Assemble run parameters for ``nphase`` phases and the named HU-BM ``scheme``.

    Configures the module's phase system (so mixture / closures / IC build for ``nphase``) and
    returns the params dict. ``scheme="hu"`` maps to HU-BM(mp). The time stepping mirrors
    ``hamon_2d_solver`` (backward-Euler to ``t_end_days``, snapshots at ``snap_days`` hit exactly
    via the schedule; ``constant_dt`` toggles pure-constant vs the reject-and-halve adaptive
    analog).  Time-step size (days): ``dt_days`` is the nominal default for both the INITIAL step
    and the CAP; pass ``dt_init_days`` and/or ``dt_max_days`` to control them independently (start
    small on a stiff IC and let the step grow up to ``dt_max_days``).  Extra keyword ``overrides``
    are merged last (e.g. a different ``time_manager`` for a short test run, or ``use_petsc=False``).
    """
    if scheme not in _SCHEME_CONFIG:
        raise ValueError(f"unknown scheme {scheme!r}; options: {list(_SCHEME_CONFIG)}")
    configure_phase_system(nphase)
    frac_tag = "_frac" if fractures else ""
    if ad_backend is None:
        # The sparsa backend does not yet handle multi-subdomain (mixed-dimensional) variable
        # slices, so the fractured runs use the native PorePy parser; single-domain runs keep sparsa.
        ad_backend = "native" if fractures else "sparsa"
    params = dict(
        enable_buoyancy_effects=True,
        lag_buoyancy_direction=False,
        material_constants={"solid": solid_constants},
        fractures=fractures,                            # -> geometry.set_fractures (10 conductive 1D lines)
        time_manager=make_time_manager(t_end_days, dt_days, snap_days, constant_dt,
                                       dt_init_days=dt_init_days, dt_max_days=dt_max_days),
        times_to_export=make_times_to_export(snap_days),
        grid_type="cartesian",
        prepare_simulation=False,
        folder_name=f"visualization_barriers{frac_tag}_{scheme}_N{nphase}",
        file_name=f"barriers{frac_tag}_{scheme}_N{nphase}",
        # Step control: "LS" (line search) / "TR" (trust region) / "TR-LS" / "None" (plain Newton)
        step_control_method="None",
        step_control_alpha_min=1.0e-5,
        activate_step_control_after_iter=2,
        # AD backend: "native" (PorePy parser) or "sparsa" (external, ~5x faster; needs sparsa).
        ad_backend=ad_backend,
    )
    params.update(_SCHEME_CONFIG[scheme])
    params.update(overrides)
    return params


# Back-compat module-level default params (N=3, "hu" = HU-BM(mp)).
params = build_params(3, "hu")

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
    import argparse

    ap = argparse.ArgumentParser(
        description="PorePy N-phase gravity segregation through barriers (Bosma Ex. 6.3 at "
                    "--nphase 3). scheme 'hu' = HU-BM(mp).")
    ap.add_argument("--nphase", type=int, default=3,
                    help="number of phases (default 3; 4 splits oil into mid-heavy + mid-light)")
    ap.add_argument("--scheme", default="hu", choices=list(_SCHEME_CONFIG),
                    help="HU-BM scheme (default 'hu' = HU-BM(mp))")
    ap.add_argument("--days", type=float, default=T_END_DAYS,
                    help=f"run horizon in days (default {T_END_DAYS:g} = hamon T_END)")
    ap.add_argument("--dt-days", type=float, default=DT_DAYS,
                    help=f"CONSTANT-dt march step in days (only with --constant-dt; default "
                         f"{DT_DAYS:g})")
    ap.add_argument("--dt-init-days", type=float, default=DT_INIT_DAYS,
                    help=f"INITIAL adaptive step in days (default {DT_INIT_DAYS:g}); small on a "
                         f"stiff IC")
    ap.add_argument("--dt-max-days", type=float, default=DT_MAX_DAYS,
                    help=f"MAXIMUM (cap) adaptive step in days (default {DT_MAX_DAYS:g}); never "
                         f"exceeded, floor is dt-max/64")
    ap.add_argument("--constant-dt", action="store_true",
                    help="pure constant march at --dt-days (else reject-and-halve adaptive)")
    ap.add_argument("--md", action="store_true",
                    help="mixed-dimensional: add 10 conductive fractures (k*%g), conforming to the "
                         "mesh: 5 vertical crossing the barriers, 5 horizontal in barrier-free bands"
                         % FRACTURE_K_FACTOR)
    ap.add_argument("--linear-solver", default="cpr", choices=["cpr", "scipy"],
                    help="null-mean linear solver: 'cpr' (PETSc Schur+CPR) or 'scipy' (direct "
                         "bordered solve -- isolates/bypasses the PETSc CPR)")
    args = ap.parse_args()

    snaps = tuple(d for d in SNAP_DAYS if d <= args.days + 1e-9)
    model = FlowModel(build_params(
        args.nphase, args.scheme, t_end_days=args.days, dt_days=args.dt_days,
        dt_init_days=args.dt_init_days, dt_max_days=args.dt_max_days,
        snap_days=snaps, constant_dt=args.constant_dt, fractures=args.md,
        lagrange_linear_solver=args.linear_solver))
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.ResidualBasedAbsoluteCriterion(
                tol=1.0e-4, metric=pp.EquationBasedLebesgueMetric(model)   # hamon atol=1e-4
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=20),      # hamon Newton cap = 20
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
