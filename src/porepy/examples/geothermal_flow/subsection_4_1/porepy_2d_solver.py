"""N-phase immiscible gravity segregation through barriers (PorePy CF, subsection 4.2).

Setup (Bosma et al. 2022, Ex. 6.3 / Fig. 5): closed 100 m x 100 m box, 100 x 100 cells, seven
horizontal impermeable barrier layers with openings.  Initial layering: water (rho = 1500, top
10%), gas (rho = 500, bottom 10%), intermediate phase(s) between; all segregate under gravity.
One component per phase; the overall fractions z_i are the transported primaries and the
saturations follow exactly from  s_i = (z_i / rho_i) / sum_j (z_j / rho_j), with analytic
Jacobian.  Densities and viscosities are constant (rho_i, mu = 1 cP); hydrostatic initial p.

This script demonstrates three things:

1. N-phase functionality.  --nphase 3 reproduces Bosma Fig. 5; --nphase 4 splits the middle
   phase in two (densities evenly spaced in [500, 1500]).  No part of the formulation is
   specific to the phase count.

2. Background mobility.  The buoyant flux of a phase pair (i, j),
   F_ij = k (rho_j - rho_i) g lambda_i lambda_j / lambda_T, is upwinded pairwise while the
   remaining N-2 phases enter only through an aggregated background mobility -- so the scheme
   needs no density sorting or phase ordering at any point.

3. Conservation.  The box is closed (all-Neumann): total and per-phase masses are exact
   invariants, so any drift measures the discrete conservation of the buoyancy terms.
   run_statistics.{txt,json} and the exported fields (saturations, delta_p = p - p_ic) make
   the check direct.

Schemes (--scheme): hu = HU-BM(mp), the pair form above with fractional_flow=False;
hu-mw = mobility-weighted variant (total_mass_mobility * k in the Darcy tensor,
fractional-flow template, fractional_flow=True).

--md adds ten conductive fractures (5 vertical, 5 horizontal, intersecting; thin fault-zone
parameterization via FRACTURE_K_FACTOR and FRACTURE_APERTURE) as a mixed-dimensional
2D/1D/0D grid.

Gauge and solvers: all-Neumann leaves p defined up to a constant, fixed by the bordered
constraint Sum(dp) = 0 (Lagrange multiplier) inside the linear solver; CPR = FGMRES on the
row-equilibrated bordered system with a {p, lambda} / {h, z} multiplicative split (direct MUMPS
fallback).  Newton: tol 1e-3 (per-equation Lebesgue norm), cap 11 iterations.  dt starts at
DT_INIT_DAYS, doubles on easy steps up to DT_MAX_DAYS, halves on failure (floor = cap/64).

Run on your end (not auto-run):
    python porepy_2d_solver.py --nphase 3 --scheme hu [--md]
    python porepy_2d_solver.py --nphase 4 --scheme hu-mw --md --cpr-rtol 1e-8
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
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import (  # noqa: E402,E501
    FlowModelBase,              # total-mass formulation      -> CompositionalFlowTemplate
    FractionalFlowModelBase,    # fractional-flow formulation -> CompositionalFractionalFlowTemplate
    geothermal_nonlinear_solver,  # NewtonSolver that dispatches to model.solve_linear_system
)
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
# Densities evenly spaced heaviest -> lightest: N=3 gives Bosma's [1500, 1000, 500],
# N=4 gives [1500, 1167, 833, 500]; the z -> s inversion generalizes to any N.
MU = 1.0e-3                     # phase viscosity [kg/(m.s)] = 1 cP (all phases; scaled to_Mega)
# Constant per-phase specific enthalpies [MJ/kg], evenly spaced in [H_MIN, H_MAX]; denser phase
# -> lower enthalpy (and via T = h/C_P, colder).
H_MIN, H_MAX = 1.0, 3.0         # phase specific-enthalpy bounds [MJ/kg]
C_P = 0.0035                    # caloric closure h = C_P * T (T elliptic but regularized by
#                                 the energy accumulation; only pressure needs the gauge).


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

# Optional --md fractures: 1D fault-zone conduits in the 2D matrix (barriers unchanged).
# Transmissivity contrast K_FACTOR * a / dx = 100 makes them preferential conduits; point CFL
# scales as K_FACTOR / a.
FRACTURE_K_FACTOR = 1.0e+3          # fracture (dim<nd) permeability = k * this
FRACTURE_APERTURE = 1.0e-1          # fracture aperture [m] -> specific volume of 1D cells (a) and 0D points (a^2)


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
    """Closure: phase ``i``'s saturation AND its exact Jacobian from
    ``deps = (p, h, z_1, ..., z_{N-1})``.

    ``s_i = w_i / D`` with ``w_j = z_all[j] / rho_j``, ``z_all = [1 - sum(z), z_1, ...]`` and
    ``D = sum_j w_j``.  Values and derivatives are evaluated from the RAW iterate -- no clipping
    inside the residual -- so value and Jacobian stay consistent; physical bounds are enforced by
    projecting the ITERATE onto the z-simplex after each Newton update
    (:meth:`_FlowModelBody.after_nonlinear_iteration`).  The previous version returned clipped
    values with IDENTICALLY ZERO derivatives, leaving the Jacobian blind to the z->s coupling:
    survivable fixed-dimensionally (a lagged-saturation Picard), fatal on the mixed-dimensional
    grid whose near-volume-less fracture cells are corrected purely through the Jacobian.
    """
    def f(*deps):
        nc = len(deps[0])
        z = [np.asarray(zk, dtype=float) for zk in deps[2:2 + NPHASE - 1]]
        z0 = 1.0 - (sum(z) if z else 0.0)
        z_all = [z0] + z
        w = [z_all[j] / RHO[j] for j in range(NPHASE)]
        D = sum(w)
        s = w[i] / D
        diffs = np.zeros((len(deps), nc))
        for k in range(1, NPHASE):                     # z_k lives at dependency index 2 + (k-1)
            if i == 0:
                dw_i = -1.0 / RHO[0]                   # w_0 = (1 - sum z)/rho_0
            else:
                dw_i = 1.0 / RHO[i] if k == i else 0.0
            dD = 1.0 / RHO[k] - 1.0 / RHO[0]
            diffs[2 + k - 1] = (dw_i * D - w[i] * dD) / (D * D)
        return s, diffs
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


def configure_phase_system(nphase: int, equal_middle: bool = False,
                           permute_middle: bool = False,
                           linear_kr_middle: bool = False) -> None:
    """Configure the module for ``nphase`` phases: evenly-spaced densities, phase/component
    names, and the (name-keyed) saturation + immiscibility maps consumed by
    :class:`SecondaryEquations3N`. N=3 reproduces the Bosma names/maps exactly; N=4 splits the
    oil into a mid-heavy + mid-light phase."""
    global NPHASE, RHO, RHO_BANDS, PHASE_PERM, MIDDLE_LINEAR
    global H_PHASE, PHASE_NAMES, COMPONENT_NAMES
    global saturation_functions_map, chi_functions_map
    NPHASE = int(nphase)
    if equal_middle:
        # N=4 degeneracy check: both middle densities equal the N=3 oil value, so
        # the 4-phase run must reproduce the 3-phase solution (ordering-free
        # evidence). Phase enthalpies stay distinct -- passive markers only.
        if NPHASE != 4:
            raise ValueError("equal_middle is defined for nphase=4 only")
        RHO_BANDS = np.array([1500.0, 1000.0, 1000.0, 500.0])
    else:
        RHO_BANDS = np.linspace(1500.0, 500.0, NPHASE)
    if permute_middle:
        # pure LABEL swap of the two middle phases (bands + densities + enthalpy
        # markers): the same physical problem, so the solution must be identical
        # up to the swap.
        if NPHASE != 4:
            raise ValueError("permute_middle is defined for nphase=4 only")
        PHASE_PERM = np.array([0, 2, 1, 3])
    else:
        PHASE_PERM = np.arange(NPHASE)
    RHO = RHO_BANDS[PHASE_PERM]                     # phase k has the density of ITS band
    MIDDLE_LINEAR = (bool(linear_kr_middle)
                     & (PHASE_PERM >= 1) & (PHASE_PERM <= NPHASE - 2))
    H_PHASE = np.linspace(H_MIN, H_MAX, NPHASE)[PHASE_PERM]   # per-phase marker follows the label
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

    # Accessor + DOF-gate overrides for substituted quantities.
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

    # Refresh of the substituted surrogates (initial / per-step / per-iteration).
    def update_derived_quantities(self) -> None:
        super().update_derived_quantities()
        # NOTE on the ~2 its/step vs the hamon reference's ~1.1 (same criteria): the second
        # iteration is the price of the compositional formulation, not a solver defect.  The
        # elimination rows demand the eliminated saturations ON the s(z) manifold, and after
        # one Newton step they carry the linearization error of the nonlinear inversion
        # (~|ds|; the linear T = h/C_P elimination lands at machine zero).  Overwriting
        # s := s(z) between iterations removes that residual but shifts cell masses off the
        # mass-exact linear path, tripping the total-mass drift budget instead (measured:
        # ~3 its/step via the stagnation escape).  hamon's (p, s) formulation has the
        # constraint by construction and pays neither cost.
        self._refresh_substitutions(on_boundaries=False)

    def update_all_boundary_conditions(self) -> None:
        super().update_all_boundary_conditions()
        self._refresh_substitutions(on_boundaries=True)

    def initial_condition(self) -> None:
        super().initial_condition()
        # Seed the substituted surrogates so the t=0 state/export is consistent.
        self._refresh_substitutions(on_boundaries=False)

    def prepare_simulation(self) -> None:
        super().prepare_simulation()
        # Sync the temperature TIME-STEP store with its (caloric-consistent) iterate value
        # T = h/C_P, which the elimination computes late in prepare.  The rock accumulation
        # parses c_p (T - T_ref) at the PREVIOUS time step; left at the reference value,
        # step 1 is charged with the full fictitious rock energy and the closed box pays by
        # collapsing (h, T) until the books balance.
        T0 = self.equation_system.get_variable_values(["temperature"], iterate_index=0)
        for ti in self.time_step_indices:
            self.equation_system.set_variable_values(
                T0, ["temperature"], time_step_index=int(ti))

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
        masks = self._band_masks(sd)                    # phase k = 1 in band PHASE_PERM[k]
        num = np.zeros(sd.num_cells)
        den = np.zeros(sd.num_cells)
        for k in range(NPHASE):
            m = masks[PHASE_PERM[k]]
            num[m] += RHO[k] * H_PHASE[k]               # rho_k s_k h_k
            den[m] += RHO[k]                             # rho_k s_k
        return num / den

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        masks = self._band_masks(sd)
        comps = self.fluid.components
        z = np.zeros(sd.num_cells)
        for i in range(1, NPHASE):            # non-reference component i -> band PHASE_PERM[i]
            if component == comps[i]:
                z[masks[PHASE_PERM[i]]] = 1.0
        return z                              # reference (H2O) = 1 - sum (top band)

    def ic_values_saturation(self, sd: pp.Grid) -> list:
        masks = self._band_masks(sd)
        return [np.where(masks[PHASE_PERM[i]], 1.0, 0.0)
                for i in range(1, NPHASE)]                                # s_1..s_{N-1}

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


# CPR tolerances (defaults; override per run with --cpr-rtol / --cpr-maxit / --cpr-accuracy-tol,
# or programmatically via the params of the same names).
CPR_RTOL = 1.0e-7          # FGMRES relative tolerance (PETSc preconditioned norm, scaled system)
CPR_MAXIT = 300             # FGMRES iteration budget (after a stall the probe uses half of it)
CPR_ACCURACY_TOL = 1.0e-5   # acceptance gate on the TRUE projected relative residual; tripping it
                            # triggers the direct MUMPS retry (Newton tol is 1e-3 -- keep it well below)

# Conservation target order for the total-mass drift over the whole run; the per-step budget is
# tol = 10^-(order-1) / (2 n_steps) (as in tests/functional/test_buoyancy_flow.py). --drift-order.
DRIFT_ORDER = 4


class NullSpaceDriftCriterion(pp.solvers.ConvergenceCriterion):
    """Converge the dt-scaled total-mass drift of the residual.

    The summed mass residual is a null-space component the linear solve cannot reduce,
    and it is a rate: scaled by ``dt`` it is what the conservation plots accumulate.
    Once Newton stagnates the drift is frozen, so the criterion stops objecting.
    (Copy of tests/functional/setups/buoyancy_flow_model.NullSpaceDriftCriterion.)
    """

    #: Consecutive checks with relative drift change below this are considered frozen.
    _stagnation_rtol: float = 1.0e-3
    _stagnation_checks: int = 3

    def __init__(self, model: pp.PorePyModel, tol: float) -> None:
        self._model = model
        self.tol = tol
        self._history: list[float] = []
        self._total_volume: float | None = None

    def reset(self) -> None:
        self._history = []

    def check(
        self, residual: np.ndarray, **kwargs
    ) -> tuple[pp.solvers.ConvergenceStatus, float]:
        model = self._model
        rows = model.equation_system.assembled_equation_indices["mass_balance_equation"]
        if self._total_volume is None:
            # The geometry is fixed, so the normalization volume is computed once.
            self._total_volume = sum(
                np.sum(
                    model.equation_system.evaluate(
                        model.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
                    )
                )
                for sd in model.mdg.subdomains()
            )
        total_volume = self._total_volume
        drift = float(
            abs(np.sum(np.asarray(residual, dtype=float)[rows]))
            * model.time_manager.dt
            / total_volume
        )
        self._history.append(drift)
        if drift <= self.tol:
            return pp.solvers.ConvergenceStatus.CONVERGED, drift
        # Stagnation escape: the drift has frozen (quadratic basin) and cannot improve.
        recent = self._history[-self._stagnation_checks :]
        if len(recent) == self._stagnation_checks and all(
            abs(a - b) <= self._stagnation_rtol * max(abs(b), 1e-300)
            for a, b in zip(recent[:-1], recent[1:])
        ):
            return pp.solvers.ConvergenceStatus.CONVERGED, drift
        return pp.solvers.ConvergenceStatus.CONTINUE_ITERATING, drift


# --------------------------------------------------------------------------------------- #
#  Null-mean-constrained linear solve: bordered Sum(dp) = 0 fixes the constant-pressure gauge.
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
            # No SciPy fallback: _cpr_petsc_solve retries with direct MUMPS internally, so a
            # failure here is terminal.  --linear-solver scipy still selects the direct path.
            return self._schur_cpr_solve(A, b)
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

    # Elliptic variables: pressure and, after the T = h/C_P elimination, enthalpy.  Valid for
    # the Schur-reduced system only.
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

        # Primary ordering [p, h, <z + interface fluxes>, interface_darcy_flux last]: the
        # trailing darcy block is diagonal in the reduced system, so _schur_cpr_solve folds it
        # into the pressure block cheaply.
        elliptic = [v for v in self._ELLIPTIC_VARS if v in var_names]
        darcy = [v for v in var_names if v == "interface_darcy_flux"]     # eliminated LAST (if present)
        middle = [v for v in var_names
                  if v not in elliptic and v not in darcy and not is_secondary(v)]
        primary_vars = elliptic + middle + darcy
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
        n_darcy = sum(len(p_cols[len(elliptic) + len(middle) + i]) for i in range(len(darcy)))
        # Positions of the MATRIX pressure DOFs within the pressure block [0, n_pressure): the CPR puts
        # the Newton mass-imbalance on these rows only (the Sum(dp_matrix)=0 constraint).
        matrix_p = np.asarray(
            es.dofs_of([es.md_variable("pressure", [self.mdg.subdomains(dim=self.nd)[0]])]), dtype=int)
        matrix_p_pos = np.nonzero(np.isin(p_cols[0], matrix_p))[0]
        return (primary_cols, primary_rows, secondary_cols, secondary_rows,
                n_pressure, n_elliptic, n_darcy, matrix_p_pos)

    def _schur_cpr_solve(self, A, b) -> np.ndarray:
        """Schur-eliminate the local secondary closures (T, s, x; ``A_ss`` block-diagonal per cell)
        and the interface fluxes, solve the reduced (p, h, z) system with the bordered CPR
        (:meth:`_cpr_petsc_solve`), back-substitute, and gate the result on the true projected
        residual (with a direct MUMPS retry before failing)."""
        import scipy.sparse as sps
        from scipy.sparse.linalg import splu, spsolve

        t0 = time.perf_counter()
        n = A.shape[0]
        pc, pr, sc, sr, n_p, n_ell, n_darcy, matrix_p_pos = self._primary_secondary_indices(n)
        A = A.tocsc()
        App = A[pr][:, pc].tocsr()
        Aps = A[pr][:, sc].tocsr()
        Asp = A[sr][:, pc].tocsc()
        Ass = A[sr][:, sc].tocsc()
        bp, bs = b[pr], b[sr]
        t_blocks = time.perf_counter()

        # A_ss = Jacobian of the local closures var - func(primary): identity when the closures
        # depend only on primaries; detect and skip the factorization, sparse LU otherwise.
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

        S = (App - Aps @ Ainv_Asp).tocsr()                     # Schur complement (p, h, z, interface)
        g = bp - Aps @ Ainv_bs                                 # reduced RHS
        t_schur = time.perf_counter()

        # Second exact Schur: eliminate the trailing interface_darcy_flux block (self-block = I),
        # folding the matrix<->fracture coupling into pressure.  Fixed-dim: n_darcy == 0 -> no-op.
        m = S.shape[0] - n_darcy
        if n_darcy:
            Skl = S[:m, m:]                                    # kept rows x lambda cols
            Slk = S[m:, :m]                                    # lambda rows x kept cols
            dll = S[m:, m:].diagonal()                         # lambda self-block is diagonal (== I)
            Sc = (S[:m, :m] - Skl @ sps.diags(1.0 / dll) @ Slk).tocsr()
            gk, gl = g[:m], g[m:]
            gc = gk - Skl @ (gl / dll)
        else:
            Sc, gc = S, g

        # Tolerances: run-configurable (params / CLI), defaulting to the module constants.
        cpr_rtol = float(self.params.get("cpr_rtol", CPR_RTOL))
        cpr_maxit_full = int(self.params.get("cpr_maxit", CPR_MAXIT))
        acc_tol = float(self.params.get("cpr_accuracy_tol", CPR_ACCURACY_TOL))
        # After a stall, probe with half the budget (the MUMPS fallback does the real work); a
        # converging probe clears the flag.  The probe must exceed the healthy 60-100 its band.
        cpr_maxit = (max(1, cpr_maxit_full // 2)
                     if getattr(self, "_cpr_stalled", False) else cpr_maxit_full)
        xk, cpr_its = self._cpr_petsc_solve(
            Sc, gc, n_p, n_ell, matrix_p_pos, rtol=cpr_rtol, maxit=cpr_maxit)
        self._cpr_stalled = cpr_its >= cpr_maxit
        if n_darcy:
            xp = np.concatenate([xk, (gl - Slk @ xk) / dll])    # back-substitute interface_darcy_flux
        else:
            xp = xk
        t_cpr = time.perf_counter()

        # Accuracy gate on the PROJECTED residual: the transient mass imbalance sits along the
        # constant-p direction and is irreducible; only its projection measures solver error.
        vp = np.zeros(len(xp)); vp[matrix_p_pos] = 1.0     # irreducible imbalance now sits on MATRIX rows

        def _proj_rel(x_):
            r_ = S @ x_ - g
            r_ = r_ - (vp @ r_) / (vp @ vp) * vp
            return np.linalg.norm(r_) / max(np.linalg.norm(g), 1.0e-30)

        # Gate default 1e-6: three decades below the Newton tol.  A tighter gate rejects healthy
        # solves and freezes Newton via zero increments (dt ping-pong); genuine failures >= 1e-4.
        rel = _proj_rel(xp)
        if rel > acc_tol:
            # One accurate retry before giving up: direct MUMPS on the same bordered system.
            logger.warning("CPR gate tripped (rel=%.1e); retrying with direct MUMPS LU", rel)
            xk, _ = self._cpr_petsc_solve(Sc, gc, n_p, n_ell, matrix_p_pos, direct=True)
            xp = np.concatenate([xk, (gl - Slk @ xk) / dll]) if n_darcy else xk
            rel = _proj_rel(xp)
            if rel > acc_tol:
                raise RuntimeError(
                    f"CPR projected residual too large for Newton even after the direct "
                    f"MUMPS retry (rel={rel:.1e} > {acc_tol:.1e})")

        rhs_s = bs - Asp @ xp
        xs = rhs_s if lu is None else lu.solve(rhs_s)          # back-substitute the secondaries
        t_end = time.perf_counter()

        logger.info(
            "Schur-CPR linear solve: %.3fs total | blocks %.3fs, A_ss %s %.3fs, "
            "Schur %.3fs, CPR %.3fs (%d KSP its, proj_res %.1e), back-sub %.3fs "
            "[reduced %d = {p,lam} %d + {h,z} %d, secondary %d]",
            t_end - t0, t_blocks - t0, "==I" if is_identity else "LU", t_fact - t_blocks,
            t_schur - t_fact, t_cpr - t_schur, cpr_its, rel, t_end - t_cpr,
            S.shape[0], n_p + 1, S.shape[0] - n_p, len(sr),
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
    def _cpr_petsc_solve(S, g, n_p, n_ell, matrix_p_pos, rtol=1.0e-10, maxit=300,
                         direct=False) -> np.ndarray:
        """PETSc FGMRES + CPR on the BORDERED null-mean system (the gauge as an explicit constraint).

        The reduced system is singular in the constant-pressure mode, and its LEFT null vector is
        NOT the constant-pressure indicator ``v`` (measured ``||S^T v|| = O(1)``): summing the mass
        rows leaves the advective/buoyant couplings, so the old trick of subtracting the mean of the
        matrix-pressure rows -- which enforces ``g`` orthogonal to ``v`` -- perturbs ``g`` ALONG the true left
        null direction whenever the transient mass drift makes ``sum(g_p)`` nonzero.  That made the
        Krylov problem inconsistent by exactly the drift (~1e-4 relative), and any minimum-residual
        method stalled at that floor once buoyancy started moving mass (KSP reason -3 at 300 its),
        while the first iterations of a step (zero drift) still converged.

        The fix mirrors the SciPy fallback EXACTLY: solve the bordered saddle-point system

            [[S, C^T], [C, 0]] [x; lam] = [g; 0],      C = indicator of the matrix pressure dofs,

        with FGMRES on the augmented matrix and a THREE-field multiplicative split: {p, lam} -> the
        bordered pressure block (nonsingular BECAUSE of the gauge row/column) via direct LU (MUMPS:
        it pivots through the zero lam-diagonal), h -> GAMG (regular h-Laplacian), z -> ILU(0).
        Consistent by construction -- no RHS shifts, no declared null spaces -- and measured FASTER
        than the old path even where that one worked (4 vs 7 its at the IC, 12 vs 39 in-step).
        Validated on captured failing systems: 13 its to 1e-10 where production stalled at 300.
        Returns ``(x, n_iterations)``."""
        import scipy.sparse as sps
        from petsc4py import PETSc

        S = S.tocsr()
        n = S.shape[0]
        C = sps.csr_matrix(
            (np.ones(len(matrix_p_pos)), (np.zeros(len(matrix_p_pos), int), matrix_p_pos)),
            shape=(1, n))
        corner = sps.coo_matrix(([0.0], ([0], [0])), shape=(1, 1))   # explicit lam diagonal (PETSc LU)
        Mb = sps.bmat([[S, C.T], [C, corner]], format="csr")
        nb = n + 1
        gb = np.concatenate([np.asarray(g, dtype=float), [0.0]])
        # Row-equilibrate the bordered system: fracture rows are orders smaller than matrix rows
        # and FGMRES minimizes the unscaled norm (without this even exact-block splits stall).
        # The solution is unchanged; ~no-op for fixed-dim.
        row_max = np.asarray(np.abs(Mb).max(axis=1).todense()).ravel()
        row_max[row_max == 0.0] = 1.0
        Mb = (sps.diags(1.0 / row_max) @ Mb).tocsr()
        Mb.sort_indices()
        gb = gb / row_max

        mat = PETSc.Mat().createAIJ(
            size=(nb, nb),
            csr=(Mb.indptr.astype(PETSc.IntType), Mb.indices.astype(PETSc.IntType),
                 np.ascontiguousarray(Mb.data, dtype=PETSc.ScalarType)),
            comm=PETSc.COMM_SELF,
        )
        mat.assemble()

        if direct:
            # Straight to the direct MUMPS LU on the bordered system (skip FGMRES and the
            # fieldsplit setup entirely).  Used by the caller's accuracy-gate retry.
            xv = mat.createVecRight()
            bv = mat.createVecLeft()
            bv.setArray(np.ascontiguousarray(gb, dtype=PETSc.ScalarType))
            ksp_lu = PETSc.KSP().create(PETSc.COMM_SELF)
            ksp_lu.setOperators(mat)
            ksp_lu.setType("preonly")
            ksp_lu.getPC().setType("lu")
            ksp_lu.getPC().setFactorSolverType("mumps")
            ksp_lu.solve(bv, xv)
            if ksp_lu.getConvergedReason() < 0:
                raise RuntimeError(
                    f"direct MUMPS bordered solve failed (reason {ksp_lu.getConvergedReason()})")
            return xv.getArray()[:n].copy(), 0

        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(mat)
        ksp.setType("fgmres")
        ksp.setGMRESRestart(maxit)                     # full (un-restarted) FGMRES
        ksp.setTolerances(rtol=rtol, atol=1.0e-50, max_it=maxit)

        pc = ksp.getPC()
        pc.setType("fieldsplit")
        is_pl = PETSc.IS().createGeneral(
            np.concatenate([np.arange(n_p), [n]]).astype(PETSc.IntType), comm=PETSc.COMM_SELF)
        is_hz = PETSc.IS().createGeneral(
            np.arange(n_p, n, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        pc.setFieldSplitIS(("pl", is_pl), ("hz", is_hz))
        pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
        pc.setUp()
        kpl, khz = pc.getFieldSplitSubKSP()
        # bordered pressure block {p, lam}: elliptic Darcy + gauge row/column -> nonsingular; direct
        # LU via MUMPS (pivots through the zero lam diagonal; n_p+1 is small).
        kpl.setType("preonly")
        kpl.getPC().setType("lu")
        kpl.getPC().setFactorSolverType("mumps")
        # {h, z} as one exact block: at large dt the advective h<->z coupling makes the split
        # h | z marginal (chaotic convergence); the combined MUMPS block is robust and cheap.
        khz.setType("preonly")
        khz.getPC().setType("lu")
        khz.getPC().setFactorSolverType("mumps")

        xv = mat.createVecRight()
        bv = mat.createVecLeft()
        bv.setArray(np.ascontiguousarray(gb, dtype=PETSc.ScalarType))
        ksp.solve(bv, xv)
        if ksp.getConvergedReason() < 0:
            # FGMRES stalled: fall back to a direct MUMPS LU on the same (reduced) bordered
            # system.
            logger.warning("CPR FGMRES stalled (%d its); retrying with direct MUMPS LU",
                           ksp.getIterationNumber())
            ksp_lu = PETSc.KSP().create(PETSc.COMM_SELF)
            ksp_lu.setOperators(mat)
            ksp_lu.setType("preonly")
            ksp_lu.getPC().setType("lu")
            ksp_lu.getPC().setFactorSolverType("mumps")
            ksp_lu.solve(bv, xv)
            if ksp_lu.getConvergedReason() < 0:
                raise RuntimeError(
                    f"bordered CPR diverged (reason {ksp.getConvergedReason()}) AND the direct "
                    f"MUMPS fallback failed (reason {ksp_lu.getConvergedReason()})")
            return xv.getArray()[:n].copy(), ksp.getIterationNumber()
        return xv.getArray()[:n].copy(), ksp.getIterationNumber()


# --------------------------------------------------------------------------------------- #
#  Fractures, (x0, y0, x1, y1) in reference metres on the 100-cell grid (endpoints on cell
#  faces -> conforming under refinement).  Verticals cross barriers; horizontals sit in
#  barrier-free bands.  All stay inside the oil band 10 < y < 90 (touching a phase-band edge
#  destabilizes the first Newton step) and none reach the boundary.
# --------------------------------------------------------------------------------------- #
_FRACTURES_REF = [
    # Verticals elongated to intersect the horizontals (0D points at the crossings).
    (20.0, 47.0, 20.0, 89.0),   # V1  crosses H3 (y=48) and H1 (y=88); barriers en route
    (48.0, 11.0, 48.0, 68.0),   # V2  crosses H5 (y=12); barriers at y~54.5, 61.5
    (72.0, 15.0, 72.0, 80.0),   # V3  crosses H4 (y=33), H2 (y=70); no longer reaches H5 (y=12)
    (30.0, 11.0, 30.0, 49.0),   # V4  crosses H5 (y=12) and H3 (y=48); barrier at y~25.5
    (85.0, 12.0, 85.0, 71.0),   # V5  crosses H4 (y=33) and H2 (y=70); barriers at y~17.5, 25.5
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
class _FlowModelBody(
    _LagrangeConstrainedSolve,
    BarriersBoundingBox2D,
    FluidMixture3N,
    IC_NphaseSegregation,
    BC_all_neumann,
    SecondaryEquations3N,
):
    """Everything of the 2D barrier model EXCEPT the compositional-flow template.

    The template is left to the concrete classes below (:class:`FlowModel` /
    :class:`FractionalFlowModel`) because ``hu`` and ``hu-mw`` need DIFFERENT templates.  Mixed in
    ahead of the template exactly as before, so the MRO -- and hence every ``super()`` call in this
    body -- is unchanged."""

    def __init__(self, params):
        super().__init__(params)

    def update_derived_quantities(self) -> None:
        """Project the iterate's overall fractions onto the z-simplex BEFORE every flash.

        The eliminated saturation functions are clip-free with exact Jacobians (see
        :func:`_make_saturation_func`); positivity of the derived saturations/mobilities is
        enforced here instead (Appleyard-style chop on the ITERATE).  Implemented as a
        pre-step of ``update_derived_quantities`` -- NOT as an ``after_nonlinear_iteration``
        override -- so the base chain (which also refreshes the buoyancy upwind direction and
        rediscretizes in ``flow_model_base``) stays fully intact.
        """
        self._project_overall_fractions_to_simplex()
        super().update_derived_quantities()

    def _project_overall_fractions_to_simplex(self) -> None:
        """Project the independent overall fractions jointly onto the simplex
        (each ``z_k >= 0`` and ``sum_k z_k <= 1``, so the by-unity reference stays valid)."""
        es = self.equation_system
        sds = self.mdg.subdomains()
        names = sorted({v.name for v in es.variables if v.name.startswith("z_")})
        if not names:
            return
        zvars = [es.md_variable(nm, sds) for nm in names]
        vals = [es.get_variable_values([zv], iterate_index=0) for zv in zvars]
        projected = _clip_to_simplex(vals)
        for zv, old, new in zip(zvars, vals, projected):
            if np.any(new != old):
                es.set_variable_values(np.asarray(new, dtype=float), [zv], iterate_index=0)

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

        # Pressure CHANGE from the initial hydrostatic state, Delta p = p - p_ic.  The initial
        # field is recomputed from ic_values_pressure (analytic, per subdomain), so no state needs
        # to be stored and the field is exact at t = 0 (identically zero).
        p_now = np.asarray(self.pressure(sds).value(es), dtype=float)
        for i, sd in enumerate(sds):
            p_ic = np.asarray(self.ic_values_pressure(sd), dtype=float)
            data.append((sd, "delta_p", p_now[offsets[i]:offsets[i + 1]] - p_ic))
        return data

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        # quadratic kr (paper Ex. 6.3); with --linear-kr-middle the INTERIOR
        # phases use linear kr = s (additive middle mobilities -> exact N=4eq
        # degeneracy to N=3)
        k = PHASE_NAMES.index(phase.name)
        if MIDDLE_LINEAR[k]:
            return phase.saturation(domains)
        return phase.saturation(domains) ** 2

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
        """Rock permeability (barrier-masked matrix + conductive fractures), scheme-branched.

        hu (total-mass): ROCK-only -- the mobility is applied separately by the upwind terms.
        hu-mw (fractional flow): ``total_mass_mobility * k`` -- the FF formulation carries the
        mobility inside the Darcy tensor; without it the fractional fluxes are off by the total
        mobility (1e3-1e5) and Newton explodes immediately.  Mirrors the 3D solver's split."""
        vals = self._rock_permeability_values(subdomains)
        if pp.compositional_flow.is_fractional_flow(self):
            vals = self.total_mass_mobility(subdomains) * vals
        return self.isotropic_second_order_tensor(subdomains, vals)


    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Hydrostatic initial pressure in the solver's null-mean gauge, consistent with the
        initial phase layering (top 10% water, middle 80% oil, bottom 10% gas).

        In mechanical equilibrium the pressure gradient balances gravity,
        ``dp/d(depth) = rho_column * g``; integrating downward from the top with the
        piecewise-constant column density of the initial phase bands gives

            ``p(y) = p_top + g * INT_y^H rho(y') dy'``.

        This uses the SAME gravity as :meth:`gravity_field`
        (``convert_units(GRAVITY_ACCELERATION) * to_Mega``) and the UNSCALED phase densities,
        so at t = 0 the Darcy potential ``grad p - rho_m g`` vanishes within each layer -- no
        spurious pressure-driven flow; only the density contrast across the layer interfaces
        drives the buoyant segregation.  The global mean is removed
        (:meth:`_ic_pressure_mean`): the closed all-Neumann problem fixes p only up to a
        constant, and the solver's ``Sum(dp) = 0`` constraint freezes the mean at its initial
        value -- a null-mean IC keeps the whole trajectory in that gauge.
        """
        return self._hydrostatic_pressure(sd.cell_centers[1]) - self._ic_pressure_mean()

    def _hydrostatic_pressure(self, y) -> np.ndarray:
        """The raw hydrostatic profile ``p_ref + g INT rho`` at heights ``y``."""
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2") * to_Mega
        b = self._band_bounds()                                     # descending [H, ..., 0]
        # column density integrated from height y up to the top boundary H, over the N bands
        column = np.zeros_like(np.asarray(y, dtype=float))
        for k in range(NPHASE):                                     # band k spans (b[k+1], b[k]]
            column = column + RHO_BANDS[k] * np.maximum(0.0, b[k] - np.maximum(y, b[k + 1]))
        return self._p_ref + g * column

    def _ic_pressure_mean(self) -> float:
        """Unweighted mean of the hydrostatic IC over ALL pressure cells (cached).

        The linear solver constrains every increment to ``Sum(dp) = 0``, so the mean of p is
        frozen at its initial value; removing it here puts the whole trajectory in the same
        null-mean gauge as the solver's constraint (no gauge offset from the IC).
        """
        if not hasattr(self, "_ic_p_mean"):
            vals = np.concatenate([self._hydrostatic_pressure(sd.cell_centers[1])
                                   for sd in self.mdg.subdomains()])
            self._ic_p_mean = float(vals.mean())
        return self._ic_p_mean


class FlowModel(_FlowModelBody, FlowModelBase):
    """``--scheme hu`` -- HU-BM(mp), the TOTAL-MASS formulation.

    ``fractional_flow=False`` selects the mobility-product branch of ``FluidBuoyancy``, whose primary
    equations are those of ``CompositionalFlowTemplate`` (via ``FlowModelBase``)."""


class FractionalFlowModel(_FlowModelBody, FractionalFlowModelBase):
    """``--scheme hu-mw`` -- the MOBILITY-WEIGHTED variant.

    ``fractional_flow=True`` selects the fractional-flow branch of ``FluidBuoyancy``, which is only
    consistent with the fractional-flow primary equations of
    ``CompositionalFractionalFlowTemplate`` (via ``FractionalFlowModelBase``) -- hence a distinct
    class rather than a parameter flip.  Identical physics/geometry otherwise (same
    :class:`_FlowModelBody`)."""


# --------------------------------------------------------------------------------------- #
#  Run configuration (module level), mirroring tp_tc_gravitational_segregation.py
# --------------------------------------------------------------------------------------- #
day = 86400.0

# --------------------------------------------------------------------------------------- #
#  Time stepping: backward Euler, reject-and-halve to a floor of cap/64, growth to the cap.
#  Snapshot instants go into the TimeManager schedule (steps are clipped to land on them
#  exactly); times_to_export writes the VTU precisely there.
# --------------------------------------------------------------------------------------- #
T_END_DAYS = 600.0                        # hamon T_END
SNAP_DAYS = (0.0, 1.0, 2.0,3.0,4.0, 5.0, 6.0, 7.0, 8.0,9.0, 10.0, 25.0, 50.0, 75.0, 78.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0, 425.0, 450.0, 475.0, 500.0, 525.0, 550.0, 571.0, 575.0, 600.0)                # hamon SNAP_DAYS -- the Fig-5 saturation-map instants
DT_DAYS = 0.25                             # nominal step [days] -- the constant-dt march value
DT_INIT_DAYS = 0.005                     # INITIAL adaptive step [days] -- start small on the stiff,
#                                           fully density-inverted IC (denser fluid over lighter)
DT_MAX_DAYS = 0.5                         # MAXIMUM (cap) adaptive step [days] -- never exceeded;
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
        iter_optimal_range=(3, 8),            # grow dt when Newton is easy, shrink when it is hard
        iter_relax_factors=(0.5, 2.0),         # halve on a cut / double on grow-back (hamon *0.5, *2)
        recomp_factor=0.5, recomp_max=8,       # reject-and-halve, up to 8 consecutive cuts
        iter_max=13, print_info=True,          # matches the solver's max_iterations
    )


def make_times_to_export(snap_days: Sequence[float] = SNAP_DAYS) -> list:
    """VTU/PVD written exactly at each snapshot instant (matched to ``time_manager.time``)."""
    return [d * day for d in snap_days]

solid_constants = pp.SolidConstants(
    permeability=k_rock,                               # 1 mD
    porosity=porosity,                                 # 0.3
    thermal_conductivity=2.0 * to_Mega,                # unused (isothermal)
    density=2500.0,
    specific_heat_capacity= C_P,
    residual_aperture=FRACTURE_APERTURE,               # 1D fracture aperture (no effect without fractures)
)

# Scheme map: "hu" = HU-BM(mp) (fractional_flow=False + hybrid upwinding); "hu-mw" = the
# mobility-weighted variant (fractional_flow=True, fractional-flow template; the model class is
# selected by flow_model_class).
_SCHEME_CONFIG = {
    "hu":    dict(fractional_flow=False, buoyancy_upwinding="hybrid"),
    "hu-mw": dict(fractional_flow=True,  buoyancy_upwinding="hybrid"),
}


def flow_model_class(params: dict):
    """The model class matching ``params['fractional_flow']``.

    The two CF templates define DIFFERENT primary equations, so the flag cannot just be passed as a
    parameter -- it selects the base: ``fractional_flow=True`` (hu-mw) needs
    ``FractionalFlowModelBase`` (``CompositionalFractionalFlowTemplate``), while ``False`` (hu) uses
    ``FlowModelBase`` (``CompositionalFlowTemplate``).  Both carry the identical
    :class:`_FlowModelBody` mixin stack, so only the template differs."""
    return FractionalFlowModel if params.get("fractional_flow", False) else FlowModel


def build_params(nphase: int = 3, scheme: str = "hu", *, equal_middle: bool = False,
                 permute_middle: bool = False, linear_kr_middle: bool = False,
                 t_end_days: float = T_END_DAYS,
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
    configure_phase_system(nphase, equal_middle, permute_middle, linear_kr_middle)
    frac_tag = "_frac" if fractures else ""
    n_tag = (f"N{nphase}" + ("eq" if equal_middle else "")
             + ("perm" if permute_middle else "")
             + ("lk" if linear_kr_middle else ""))
    if ad_backend is None:
        # sparsa lowers mixed-dimensional variables (concat of per-grid sub-variables); default
        # for both fd and md, bit-exact vs native.  Pass ad_backend="native" to override.
        ad_backend = "native"
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
        folder_name=f"visualization_barriers{frac_tag}_{scheme}_{n_tag}",
        file_name=f"barriers{frac_tag}_{scheme}_{n_tag}",
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
    ap.add_argument("--equal-middle", action="store_true",
                    help="(with --nphase 4) set BOTH middle densities to the N=3 oil value "
                         "1000, so the 4-phase run degenerates to the 3-phase solution -- "
                         "the ordering-free check. Output tag becomes N4eq.")
    ap.add_argument("--permute-middle", action="store_true",
                    help="(with --nphase 4) swap the LABELS of the two middle phases (bands, "
                         "densities, enthalpy markers) -- a pure relabeling, so the solution "
                         "must be identical up to the swap. Tag gains 'perm'.")
    ap.add_argument("--linear-kr-middle", action="store_true",
                    help="linear kr = s for the INTERIOR (middle) phases only; middle "
                         "mobilities become additive so the --equal-middle N=4 run "
                         "degenerates to N=3 EXACTLY. Tag gains 'lk'.")
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
    ap.add_argument("--cpr-rtol", type=float, default=CPR_RTOL, metavar="TOL",
                    help=f"FGMRES relative tolerance, PETSc preconditioned norm "
                         f"(default {CPR_RTOL:.0e})")
    ap.add_argument("--cpr-maxit", type=int, default=CPR_MAXIT, metavar="N",
                    help=f"FGMRES iteration budget; after a stall the probe uses N/2 "
                         f"(default {CPR_MAXIT})")
    ap.add_argument("--cpr-accuracy-tol", type=float, default=CPR_ACCURACY_TOL, metavar="TOL",
                    help=f"acceptance gate on the TRUE projected relative residual; tripping it "
                         f"triggers the direct MUMPS retry (default {CPR_ACCURACY_TOL:.0e}; "
                         f"keep well below the Newton tol 1e-3)")
    ap.add_argument("--drift-order", type=int, default=DRIFT_ORDER, metavar="K",
                    help=f"conservation target order K for the total-mass drift: per-step budget "
                         f"10^-(K-1) / (2 n_steps) (default {DRIFT_ORDER})")
    args = ap.parse_args()

    snaps = tuple(d for d in SNAP_DAYS if d <= args.days + 1e-9)
    params = build_params(
        args.nphase, args.scheme, equal_middle=args.equal_middle,
        permute_middle=args.permute_middle, linear_kr_middle=args.linear_kr_middle,
        t_end_days=args.days, dt_days=args.dt_days,
        dt_init_days=args.dt_init_days, dt_max_days=args.dt_max_days,
        snap_days=snaps, constant_dt=args.constant_dt, fractures=args.md,
        lagrange_linear_solver=args.linear_solver,
        cpr_rtol=args.cpr_rtol, cpr_maxit=args.cpr_maxit,
        cpr_accuracy_tol=args.cpr_accuracy_tol)
    # hu -> CompositionalFlowTemplate; hu-mw -> CompositionalFractionalFlowTemplate.
    model = flow_model_class(params)(params)
    # Per-step total-mass-drift budget: the target order split over the planned number of
    # steps with a factor-2 margin (adaptive cuts only shrink dt, and the drift is dt-scaled).
    n_steps = max(1, round(args.days / (args.dt_days if args.constant_dt else args.dt_max_days)))
    drift_tol = 10.0 ** (-(args.drift_order - 1)) / (2 * n_steps)
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=1.0e-4, metric=pp.EquationBasedLebesgueMetric(model)   # hamon atol=1e-4
            ),
            "null_drift": NullSpaceDriftCriterion(model, tol=drift_tol),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=13),
        },
    }
    # Construct the runner first (prepares the simulation) so the system size and variable
    # roles are reported before the long time loop.
    runner = pp.ModelRunner(model, solver_params,
                            nonlinear_solver=geothermal_nonlinear_solver(solver_params))
    ncells = report_system_size(model)
    ndof = model.equation_system.num_dofs()

    runner.run()
    print("cells:", ncells, " dofs:", ndof)
