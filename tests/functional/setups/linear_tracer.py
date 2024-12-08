r"""Model setups for testing the CF framework.

Two models are implemented, a single-phase, 2-component model and a 3-phase, 2-component
model.
The domain is a pipe modelled as a 2D cartesian grid with a single line of cells, i.e.
actually a 1D-problem blown up to a 2D-problem for simplicity.

**Single-phase, 2-components:**

Classical tracer setup, with an incompressible fluid and a higher tracer amount entering
the domain. All fluid properties are chosen to be equal 1.

Due to assumptions on the fluid properties, the pressure equation is reduced to

.. math::

    \nabla \cdot (K \nabla p) = 0

Giving boundary conditions ``p_i`` and ``p_o`` on inlet and outlet respectively, the
analytical solution is simply ``p(x, t) = p_i + (p_o - p_i)x``, a linear gradient from
inlet to outlet. The velocity of the flow is roughly ``c = |K(p_o - p_i)|``, assuming a
isotropic absolute permeability and not taking into account what the flux discretization
does.

The advection equation for the tracer is then simply

.. math::

    \frac{\partial}{\partial t} z - c \frac{\partial}{\partial x} z = 0

With some inflow of ``z_i``, the analytical solution is a moving front raising the
initial ``z_0`` to ``z_i``. The front moves with the velocity ``c`` from inlet to
outlet.

The CFL condition is given by ``c dt / dx <=1`` in this simple case.

When comparing the numerical solution to the analytical one, numerical diffusion must
be taken into account, which decays roughly at a rate ``dx**2``, ``dx`` being the mesh
size.

**3-phase, 2-component:**

This is an exaggerated tracer model with two additional phases, testing the full CF
machinery.

The modelled is closed with a local pseudo-equilibrium. Mass is distributed equally
accross all 3 phases (saturations = phase fractions = 1/3), and K-value equations are
used with K-values 1 (mass of components distributes equally accross phases).
With these extensions, the analytical solution should be the same as for the previous
model. With local mass conservation, partial fractions should be equal to overall
fractions.

To avoid the system going singular, no phase vanishes and no component reaches ever a
overall fraction of 1.

To test isothermal conditions, the two-variable energy balance is included, with
trivial IC and BC (T=0) and a simple heuristic law for phase enthalpies
(``h(T) = a * (T-T_0)``). A closure for the dangling enthalpy variable is provided by
simply introducing an equation `` h - h_mix = 0``, where the fluid mixture enthalpy
``h_mix`` uses the heuristic law.

This model is absolutely useless. Only use it if you want to showcase bad practices in
CF modelling.
It exists only to test the solution strategy for CF with surrogate operators as phase
properties, and the framework of local equations and eliminations.

"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence, cast

import numpy as np

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.viz.data_saving_model_mixin import VerificationDataSaving


@dataclass
class LinearTracerSaveData:
    """Data collection for a solution of the linear time step at some time."""

    approx_z_tracer: np.ndarray
    """Approximate solution for tracer overall fraction in space at time ``t``."""
    approx_p: np.ndarray
    """Approximate solution for pressure in space at time ``t``."""

    exact_z_tracer: np.ndarray
    """Exact solution for tracer overall fraction in space at time ``t``."""
    exact_p: np.ndarray
    """Exact solution for pressure in space at time ``t``."""

    error_z_tracer: pp.number
    """L2-error of approximate solution for tracer overall fraction."""
    error_p: pp.number
    """L2-error of approximate solution for pressure."""

    t: pp.number
    """The time ``t`` at which this snapshot is taken."""
    dt: pp.number
    """Time step size used to get to time ``t``."""

    num_iter: int
    """Number of iterations required to converge for this time step.

    Should in theory not exceed 1 after the first time step when the pressure profile
    is established. But due to upwinding and some numerical issues, it is mostly 2.

    """


class LinearTracerExactSolution1D:
    """Class implementing the analytical solution of the pressure profile and the
    tracer propagation on a single grid in space and time."""

    z_tracer_initial: float = 0.2
    """Initial tracer fraction in the domain."""
    z_tracer_inlet: float = 0.8
    """Amount of tracer flowing in."""

    p_inlet: float = 2.
    """Pressure at the inlet."""
    p_outlet: float = 1.
    """Pressure at the outlet."""

    def __init__(self, tracer_model: TracerFlowSetup_1p) -> None:

        self._mu = tracer_model.fluid.reference_component.viscosity
        self._rho = tracer_model.fluid.reference_component.density
        self._perm = tracer_model.solid.permeability
        self._L = tracer_model.units.convert_units(tracer_model.pipe_length, 'm')
        

    def pressure(self, sd: pp.Grid) -> np.ndarray:
        """Linear pressure profile depending on inlet and outlet pressure, and the
        x-coordinates of the grid's cell centers."""
        return self.p_inlet + (self.p_outlet - self.p_inlet) / self._L * sd.cell_centers[0]
    
    def tracer_fraction(self, sd: pp.Grid, t: float) -> np.ndarray:
        """The tracer fraction in the cell centers depending on time t."""
        _, nx = sd.cell_centers.shape
        z_0 = np.ones(nx) * self.z_tracer_initial

        front_x = self.front_position(sd, t)

        z_inflow = np.ones(nx) * (self.z_tracer_inlet - self.z_tracer_initial)
        # super-positioning final solution and initial solution.
        z_inflow[sd.cell_centers[0] > front_x] = 0.

        return z_0 + z_inflow
    
    def diffused_tracer_fraction(self, sd: pp.Grid, t: float) -> np.ndarray:
        """Returns a tracer fraction assuming the numerical scheme is diffusive due
        to Upwinding and backward Euler.
        
        
        """
    
    def darcy_flux(self, sd: pp.Grid) -> np.ndarray:
        """Returns the Darcy flux on all faces, including inlet and outlet."""
        p = self.pressure(sd)
        dp = np.ediff1d(np.flip(p))

        # assuming cartesian grid with square cells.
        dx = np.ones(p.shape) * np.ediff1d(sd.cell_centers[0])[0]

        T = self._perm / dx  # transmissibility

        flux_internal = [
            2 / (1/T[i] + 1 / T[i + 1]) * dp_
            for i, dp_ in enumerate(dp)
        ]

        flux_inlet = (self.p_inlet - p[0]) / (dx[0] / 2) * self._perm
        flux_outlet = (p[-1] - self.p_outlet) / (dx[-1] / 2) * self._perm

        flux = np.array([flux_inlet] + flux_internal + [flux_outlet])
        return flux / self._mu

    def flow_velocity(self, sd: pp.Grid) -> float:
        """Returns the flow velocity per face assuming incompressible flow and a
        2-cell stencil to calculate the flux discretization."""
        # Due to incompressibility, constant BC and contant properties, we assume the
        # velocity to be constant as well.
        return float(np.mean(self.darcy_flux(sd)))

    def front_position(self, sd: pp.Grid, t: float) -> float:
        """Returns the position of the front along the x-axis at a given time t."""
        return self.flow_velocity(sd) * t
    
    def cfl(self, sd: pp.Grid, dt: float) -> float:
        """Returns the CFL number ``v * dt / dx`` assuming a uniform dx"""
        # assumes uniform cartesian grid
        return self.flow_velocity(sd) * dt / np.ediff1d(sd.cell_centers[0])[0]
    
    def dt_from_cfl(self, sd: pp.Grid, eps: float = 1e-8) -> float:
        """Returns the maximal time step size which does not violate the CFL condition
        ``v * dt / dx <=1``, minus a threshhold ``eps`` to avoid numerical issues."""
        return np.ediff1d(sd.cell_centers[0])[0] / self.flow_velocity(sd) - eps


class LinearTracerDataSaving(VerificationDataSaving, pp.PorePyModel):
    """Mixin class to safe data relevant for tests."""

    exact_sol: LinearTracerExactSolution1D

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def collect_data(self) -> LinearTracerSaveData:
        sds = self.mdg.subdomains()

        t = self.time_manager.time
        dt = self.time_manager.dt
        num_iter = self.nonlinear_solver_statistics.num_iteration

        _, tracer = self.fluid.components
        approx_p = self.pressure(sds).value(self.equation_system)
        exact_p = self.exact_sol.pressure(sds[0])
        approx_z_tracer = tracer.fraction(sds).value(self.equation_system)
        exact_z_tracer = self.exact_sol.tracer_fraction(sds[0], t)

        return LinearTracerSaveData(
            approx_z_tracer=approx_z_tracer,
            approx_p=approx_p,
            exact_z_tracer=exact_z_tracer,
            exact_p=exact_p,
            error_z_tracer=ConvergenceAnalysis.l2_error(
                sds[0], exact_z_tracer, approx_z_tracer, is_scalar=True, is_cc=True
            ),
            error_p=ConvergenceAnalysis.l2_error(
                sds[0], exact_p, approx_p, is_scalar=True, is_cc=True,
            ),
            t=t,
            dt=dt,
            num_iter=num_iter,
        )


class SimplePipe2D(pp.PorePyModel):
    """Simple 2D channel with a length ~10 m and aspect ratio of 1:10.

    A cartesian grid is used and the width and is chosen such that it is
    a single line of cells mimiking a 1D problem.
     _ _ _ _ _
    |_|_|_|_|_|

    The cells are all unit squares with side length ``10/ nx``, where ``nx`` is given
    in the model parameters.

    """

    pipe_length: float = 10.
    """Pipe length of domain in meters."""

    @property
    def _nx(self) -> int:
        """Returns the target number of cells ``nx`` which can be given as a model
        parameter."""
        return int(self.params.get("num_cells", 10))

    @property
    def _dx(self) -> float:
        """Returns the cell size of the unit square cells using target length of 10 m
        and given number of cells (converted to simulation units)."""
        return self.units.convert_units(10 / self._nx, "m")

    def grid_type(self) -> Literal["cartesian"]:
        return "cartesian"

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            bounding_box={
                "xmin": 0.0,
                "xmax": self.units.convert_units(self.pipe_length, "m"),
                "ymin": 0.0,
                "ymax": self._dx,
            }
        )

    def meshing_arguments(self) -> dict:
        return {"cell_size": self._dx}


class TracerFluid_1p:
    """Incompressible 2-component, 1-phase fluid with unit properties (everything is 1)."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        # This fluid will be used for the heuristic thermodynamic properties of the
        # single phase
        component_1 = pp.FluidComponent(
            name="fluid",
            compressibility=0,
            density=1,
            thermal_conductivity=1,
            thermal_expansion=1,
            specific_heat_capacity=1,
            viscosity=1,
        )
        component_2 = pp.FluidComponent(name="tracer")
        return [component_1, component_2]


class TracerIC_1p:
    """Mixes in the initial pressure and overall fraction values."""

    exact_sol: LinearTracerExactSolution1D

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Setting initial pressure equal to pressure on outflow boundary."""
        # Initial and outlet pressure are the same.
        return np.ones(sd.num_cells) * self.exact_sol.p_outlet

    def initial_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        """Setting initial tracer overall fraction to zero."""
        assert component.name == "tracer"
        # No tracer in the domain at the beginning.
        return np.ones(sd.num_cells) * self.exact_sol.z_tracer_initial


class TracerBC_1p(pp.PorePyModel):
    """Mixes in the BC for pressure and the boundary type definition, and the BC for
    overall fractions."""

    exact_sol: LinearTracerExactSolution1D

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Flagging the inlet and outlet faces as Dirichlet boundary, where pressure
        is given."""
        dirichlet_faces = np.zeros(sd.num_faces, dtype=bool)
        sides = self.domain_boundary_sides(sd)
        dirichlet_faces[sides.east | sides.west] = True

        return pp.BoundaryCondition(sd, dirichlet_faces, "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Corresponding to Darcy flux."""
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Defines some non-trivial values on inlet and outlet faces of the matrix."""

        p = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        p[sides.west] = self.exact_sol.p_inlet
        p[sides.east] = self.exact_sol.p_outlet

        return p

    def bc_values_overall_fraction(
        self, component: pp.Component, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Defines some non-trivial inflow of the tracer component on the inlet."""

        assert component.name == "tracer"
        z = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        z[sides.west] = self.exact_sol.z_tracer_inlet

        return z


class TracerFlowSetup_1p(
    SimplePipe2D,
    TracerFluid_1p,
    pp.compositional.CompositionalVariables,
    pp.compositional_flow.ComponentMassBalanceEquations,
    # Mixin IC/BC for CF and do not put them as bases for TracerIC/BC. The latter will
    # be reused for the other model, which has the former in the base CF setup.
    TracerBC_1p,
    pp.compositional_flow.BoundaryConditionsCF,
    TracerIC_1p,
    pp.compositional_flow.InitialConditionsCF,
    LinearTracerDataSaving,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """Tracer setup with 2 components and 1 phase."""

    exact_sol: LinearTracerExactSolution1D
    results: list[LinearTracerSaveData]

    def set_materials(self):
        """Setting the exact solution after the material has been set."""
        super().set_materials()
        self.exact_sol = LinearTracerExactSolution1D(self)
        self.results: list[LinearTracerSaveData] = []


class TrivialEoS(pp.compositional.EquationOfState):
    """Trivial EoS returning 1 for every property and zero derivatives."""

    def compute_phase_properties(self, phase_state, *thermodynamic_input):
        # number of derivatives and number of values per derivative
        nd = len(thermodynamic_input)
        nx = len(thermodynamic_input[0])

        # zero derivative
        d = np.zeros((nd, nx))
        # trivial values
        v = np.ones(nx)
        return pp.compositional.PhaseProperties(
            h=v.copy(),
            rho=v.copy(),
            state=phase_state,
            mu=v.copy(),
            kappa=v.copy(),
            dh=d.copy(),
            drho=d.copy(),
            dmu=d.copy(),
            dkappa=d.copy(),
        )


class TracerFluid_3p(TracerFluid_1p):
    """2-component, 3-phase tracer fluid with 3 unitary phases (all properties are 1)."""

    def get_phase_configuration(
        self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        """Returns configs for 3 phases with same dummy EoS."""
        eos = TrivialEoS(components)
        state = pp.compositional.PhysicalState.liquid
        return [(eos, state, "1"), (eos, state, "2"), (eos, state, "3")]


def saturation_function(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The saturation functions are of form ``s(x) = 1/3``, i.e. no derivatives."""
    return (np.ones(x.shape) / 3, np.zeros(x.shape))


class ModelClosure_3p(pp.LocalElimination):
    """Closing the 3-phase tracer flow with local equations.

    Provides also consistent initial and BC values for saturations, phase fractions and
    partial fractions.

    """

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def set_equations(self):
        super().set_equations()
        self._set_enthalpy_closure()
        self._set_saturation_closure()
        self._set_phase_fraction_closure()
        self._set_partial_fraction_closure()

    def _set_saturation_closure(self) -> None:
        """Closes the system by introducing s_i - 1/3 = 0 for the two independent
        phases."""
        for phase in self.fluid.phases:
            if phase != self.fluid.reference_phase:
                self.eliminate_locally(
                    phase.saturation,
                    # choose any scalar variable to inform the framework about the DOFs
                    [self.enthalpy],
                    saturation_function,
                    # Eliminate the saturation on subdomains and boundaries, and let the
                    # framework handle the IC/BC using the function.
                    self.mdg.subdomains() + self.mdg.boundaries(),
                )

    def _set_enthalpy_closure(self) -> None:
        """Relates the two energy variables h and T with each other by using the
        heuristic law."""

        sds = self.mdg.subdomains()
        op = self.enthalpy(sds) - self.fluid.specific_enthalpy(sds)
        op.set_name("enthalpy_closure")
        self.equation_system.set_equation(op, sds, {"cells": 1})

    def _set_phase_fraction_closure(self) -> None:
        """Relates the phase fractions to the saturation variables y=s."""
        sds = self.mdg.subdomains()
        for phase in self.fluid.phases:
            if phase != self.fluid.reference_phase:
                op = phase.fraction(sds) - phase.saturation(sds)
                op.set_name(f"phase_fraction_relation_{phase.name}")
                self.equation_system.set_equation(op, sds, {"cells": 1})

    def _set_partial_fraction_closure(self) -> None:
        """Closes the system by introducing local mass balance equations for the
        tracer.

        There are 3 remaining dangling variables, the partial tracer fractions.
        (other partial fraction is eliminated by unity by default).

        The first equation is the local mass conservation for the tracer.
        The second and third equation are pseudo-equilibrium equations, relating the
        partial fractions with each other using a K-value of 1 (same amount in every
        phase)

        """
        sds = self.mdg.subdomains()
        _, tracer = self.fluid.components
        rphase = self.fluid.reference_phase
        # local mass conservation for the tracer
        op = tracer.fraction(sds) - pp.ad.sum_operator_list(
            [
                phase.fraction(sds) * phase.partial_fraction_of[tracer](sds)
                for phase in self.fluid.phases
            ]
        )
        op.set_name(f"local_mass_conservation_{tracer.name}")
        self.equation_system.set_equation(op, sds, {"cells": 1})

        indpendent_phases = [p for p in self.fluid.phases if p != rphase]
        assert len(indpendent_phases) == 2

        # K-value equations for tracer, with K-value 1, distributing the tracer
        # equally accross all phases.
        op = rphase.partial_fraction_of[tracer](sds) - indpendent_phases[
            0
        ].partial_fraction_of[tracer](sds)
        op.set_name(f"K_value_{tracer.name}_{indpendent_phases[0].name}")
        self.equation_system.set_equation(op, sds, {"cells": 1})

        op = rphase.partial_fraction_of[tracer](sds) - indpendent_phases[
            1
        ].partial_fraction_of[tracer](sds)
        op.set_name(f"K_value_{tracer.name}_{indpendent_phases[1].name}")
        self.equation_system.set_equation(op, sds, {"cells": 1})

    def update_all_boundary_conditions(self):
        """Partial fractions for the tracer must be provided on the boundary.
        Set equal to z."""
        super().update_all_boundary_conditions()

        _, tracer = self.fluid.components

        f = partial(TracerBC_1p.bc_values_overall_fraction, self, tracer)
        sds = self.mdg.subdomains()

        for phase in self.fluid.phases:
            x = phase.partial_fraction_of[tracer](sds)
            self.update_boundary_condition(x.name, f)

    def initial_condition(self):
        """Set initial values for partial fractions equal to overall fractions and
        initial phase fractions equal to phase saturations."""
        super().initial_condition()
        sds = self.mdg.subdomains()

        _, tracer = self.fluid.components

        z = tracer.fraction(sds).value(self.equation_system)
        for phase in self.fluid.phases:

            x = phase.partial_fraction_of[tracer](sds)
            self.equation_system.set_variable_values(z, [x], iterate_index=0)

            if phase != self.fluid.reference_phase:
                s = phase.saturation(sds).value(self.equation_system)
                self.equation_system.set_variable_values(
                    s, [phase.fraction(sds)], iterate_index=0
                )


class TracerFlowSetup_3p(
    # putting this constitutive law above the fluid to have enthalpy as a function,
    # and not a surrogate operator,
    # pp.constitutive_laws.FluidEnthalpyFromTemperature,
    TracerFluid_3p,
    TracerIC_1p,
    TracerBC_1p,
    ModelClosure_3p,
    LinearTracerDataSaving,
    pp.compositional_flow.ModelSetupCF,
):
    """Exaggerated model for tracer flow, which includes 2 additional phases.
    Solution should be the same as the 1-p case, with partial fractions being equal to
    overall fractions, and saturations and phase fractions equal to 1/3.

    It also includes the two-variable energy equation, with enthalpy and temperature.
    Using the heuristic law and trivial BC, the result should be under isothermal
    conditions, with h=T=0.

    """

    exact_sol: LinearTracerExactSolution1D
    results: list[LinearTracerSaveData]

    def set_materials(self):
        """Setting the exact solution after the material has been set."""
        super().set_materials()
        self.exact_sol = LinearTracerExactSolution1D(self)
        self.results: list[LinearTracerSaveData] = []

    # NOTE Due to how the constitutive law for EnthalpyFromTemperature is implemented,
    # we have an MRO issue here. We cannot put FluidEnthalpyFromTemperature above
    # because it is part of EnthalpyFromTemperature which is used for heuristic laws
    # for the solid in ModelSetupCF. This will be resolved at some point in the future
    # when the solid is generalized (and respectively the constitutive laws for it).

    def fluid_specific_heat_capacity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """
        Parameters:
            subdomains: List of subdomains. Not used, but included for consistency with
                other implementations.

        Returns:
            Operator representing the fluid specific heat capacity  [J/kg/K]. The value
            is picked from the constants of the reference component.

        """
        return pp.ad.Scalar(
            self.fluid.reference_component.specific_heat_capacity,
            "fluid_specific_heat_capacity",
        )

    def specific_enthalpy_of_phase(
        self, phase: pp.Phase
    ) -> pp.ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a linear specific enthalpy for the fluid's phase.

        .. math::

            h = c \\left(T - T_0\\right)

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def h(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            c = self.fluid_specific_heat_capacity(cast(list[pp.Grid], domains))
            enthalpy = c * self.perturbation_from_reference(
                "temperature", cast(list[pp.Grid], domains)
            )
            enthalpy.set_name("fluid_enthalpy")
            return enthalpy

        return h



# material_constants = {
#     "solid": pp.SolidConstants(porosity=1., permeability=1., residual_aperture=1),
# }
# time_manager = pp.TimeManager([0, 10, 30, 80, 100], 5, True)
# model_params = {
#     "material_constants": material_constants,
#     "time_manager": time_manager,
#     'num_cells': 10,
#     'prepare_simulation': False,
# }
# model = TracerFlowSetup_1p(model_params)
# model.prepare_simulation()
# pp.run_time_dependent_model(model, model_params)
# pp.plot_grid(
#     model.mdg,
#     "pressure",
#     figsize=(10, 8),
#     linewidth=0.2,
#     title="Pressure distribution",
#     plot_2d=True,
# )
# pp.plot_grid(
#     model.mdg,
#     "z_tracer",
#     figsize=(10, 8),
#     linewidth=0.2,
#     title="Tracer distribution after 2 hours",
#     plot_2d=True,
# )

# dt = 10.
# sol = LinearTracerExactSolution1D(model)
# sd = model.mdg.subdomains()[0]
# print(sol.pressure(sd))
# print(sol.flow_velocity(sd))
# print(sol.tracer_fraction(sd, dt))
# print(sol.cfl(sd, dt) <= 1.)
# print(sol.dt_from_cfl(sd))