"""Example implementing a multi-phase multi component flow of H2O-NaCl using
the Driesner correlations as constitutive laws.

This model uses pressure, specific fluid mixture enthalpy and NaCl overall fraction as
primary variables.

No equilibrium calculations included.

Ergo, the user must close the model to provide expressions for saturations, partial
fractions and temperature, depending on primary variables.

Note:
    With some additional work, it is straight forward to implement a model without
    h as the primary variable, but T.

    What needs to change is:

    1. Overwrite
       porepy.models.compositional_flow.VariablesCF
       mixin s.t. it does not create a h variable.
    2. Modify accumulation term in
       porepy.models.compositional_flow.TotalEnergyBalanceEquation_h
       to use T, not h.
    3. H20_NaCl_brine.dependencies_of_phase_properties: Use T instead of h.

"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

import porepy as pp
import porepy.composite as ppc
from porepy.applications.md_grids.domains import nd_cube_domain

# CompositionalFlow has data savings mixin, composite variables mixin,
# Solution strategy eliminating local equations with Schur complement and no flash.
# It also hase the ConstitutiveLaws for CF, which use the FluidMixture.
# For changing constitutive laws, import ConstitutiveLawsCF and overwrite mixins
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
    SecondaryEquationsMixin,
)


def dummy_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dummy function for computing the values and derivatives of a thermodynamic
    property.

    Correlations are designed to depend on p, h, z_H2O and z_CO2,
    i.e. ``thermodynamic_properties`` will be a 4-tuple of ``(nc,)`` arrays
    """
    nc = len(thermodynamic_dependencies[0])
    vals = np.ones(nc)
    # row-wise storage of derivatives, (4, nc) array
    diffs = np.ones((len(thermodynamic_dependencies), nc))

    return vals, diffs


class DriesnerCorrelations(ppc.AbstractEoS):
    """Class implementing the calculation of thermodynamic properties.

    Note:
        By thermodynamic properties, this framework refers to the below
        indicated quantities, and **not** quantitie which are variables.

        Fractions (partial and saturations) and other intensive quantities like
        temperature need a separate treatment because they are always modelled as
        variables, whereas properties are always dependent expressions.

    """

    def compute_phase_state(
        self, phase_type: int, *thermodynamic_input: np.ndarray
    ) -> ppc.PhaseState:
        """Function will be called to compute the values for a phase.
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas).
        ``thermodynamic_dependencies`` are as defined by the user.
        """

        p, h, z_NaCl = thermodynamic_input
        n = len(p)  # same for all input (number of cells)

        # specific volume of phase
        v, dv = dummy_func(p, h, z_NaCl)  # (n,), (3, n) array
        # NOTE specific molar density rho not required since always computed as
        # reciprocal of v by PhaseState class

        # specific enthalpy of phase
        h, dh = dummy_func(p, h, z_NaCl)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = dummy_func(p, h, z_NaCl)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = dummy_func(p, h, z_NaCl)  # (n,), (3, n) array

        # Fugacity coefficients
        # not required for this formulation, since no equilibrium equations
        # just show-casing it here
        phis = np.empty((2, n))  # (2, n) array  (2 components)
        dphis = np.empty(
            (2, 3, n)
        )  # (2, 3, n)  array (2 components, 3 dependencies, n cells)

        return ppc.PhaseState(
            phasetype=phase_type,
            v=v,
            dv=dv,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=phis,
            dphis=dphis,
        )


class H20_NaCl_brine(ppc.FluidMixtureMixin):
    """Mixture mixin creating the brine mixture with two components."""

    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.compositional_flow.VariablesEnergyBalance`."""

    def get_components(self) -> Sequence[ppc.Component]:
        """Setting H20 as first component in Sequence makes it the reference component.
        z_H20 will be eliminated."""
        species = ppc.load_species(["H2O", "NaCl"])
        components = [ppc.Component.from_species(s) for s in species]
        return components

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.AbstractEoS, int, str]]:
        """Define all phases the model should consider, here 1 liquid 1 gas.
        Use custom Correlation class."""
        eos = DriesnerCorrelations(components)
        # Configure model with 1 liquid phase (phase type is 0)
        # and 1 gas phase (phase type is 1)
        # The Mixture always choses the first liquid-like phase it finds as reference
        # phase. Ergo, the saturation of the liquid phase is not a variable, but given
        # by 1 - s_gas
        return [(eos, 0, "liq"), (eos, 1, "gas")]

    def dependencies_of_phase_properties(
        self, phase: ppc.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]:
        """Overwrite parent method which gives by default a dependency on
        p, T, z_CO2.
        """
        # This will give the independent overall fractions
        z_NaCl = [
            comp.fraction
            for comp in self.fluid_mixture.components
            if comp != self.fluid_mixture.reference_component
        ]
        return [self.pressure, self.enthalpy] + z_NaCl

    def set_components_in_phases(
        self, components: Sequence[ppc.Component], phases: Sequence[ppc.Phase]
    ) -> None:
        """By default, the unified assumption is applied: all components are present
        in all phases."""
        super().set_components_in_phases(components, phases)


class ModelGeometry:
    def set_domain(self) -> None:
        size = self.solid.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1_points = self.solid.convert_units(
            np.array([[0.2, 1.8], [0.2, 1.8]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.25, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class BoundaryConditions(BoundaryConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""


class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""


class SecondaryEquations(SecondaryEquationsMixin):
    """Mixin to provide expressions for dangling variables.

    The CF framework has the following quantities always as independent variables:

    - independent phase saturations
    - partial fractions (independent since no equilibrium)
    - temperature (needs to be expressed through primary variables in this model, since
      no p-h equilibrium)

    """

    dependencies_of_phase_properties: Sequence[
        Callable[[pp.GridLikeSequence], pp.ad.Operator]
    ]
    """Defined in the Brine mixture mixin."""

    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    def set_equations(self) -> None:
        subdomains = self.mdg.subdomains()

        ### Providing constitutive law for gas saturation based on correlation
        rphase = self.fluid_mixture.reference_phase  # liquid phase
        # gas phase is independent
        independent_phases = [p for p in self.fluid_mixture.phases if p != rphase]

        for phase in independent_phases:
            self.eliminate_by_constitutive_law(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdoains
                dummy_func,  # numerical function implementing correlation
                subdomains,  # all subdomains on which to eliminate s_gas
                # dofs = {'cells': 1},  # default value
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid_mixture.phases:
            for comp in phase:
                self.eliminate_by_constitutive_law(
                    phase.partial_fraction_of[comp],
                    self.dependencies_of_phase_properties(phase),
                    dummy_func,
                    subdomains,
                )

        ### Provide constitutive law for temperature
        self.eliminate_by_constitutive_law(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            dummy_func,
            subdomains,
        )


class ModelEquations(
    PrimaryEquationsCF,
    SecondaryEquations,
):
    """Collecting primary flow and transport equations, and secondary equations
    which provide substitutions for independent saturations and partial fractions.
    """

    def set_equations(self):
        """Call to the equation. Parent classes don't use super(). User must provide
        proper order resultion.

        I don't know why, but the other models are doing it this way was well.
        Maybe it has something to do with the sparsity pattern.

        """
        # Flow and transport in MD setting
        PrimaryEquationsCF.set_equations(self)
        # local elimination of dangling secondary variables
        SecondaryEquations.set_equations(self)


class DriesnerBrineFlowModel(
    ModelGeometry,
    H20_NaCl_brine,
    InitialConditions,
    BoundaryConditions,
    ModelEquations,
    CFModelMixin,
):
    """Model assembly. For more details see class CompositionalFlow."""


time_manager = pp.TimeManager(
    schedule=[0, 0.3, 0.6],
    dt_init=1e-4,
    dt_min_max=[1e-4, 0.1],
    constant_dt=False,
    iter_max=50,
    print_info=True,
)

# Model setup:
# eliminate reference phase fractions  and reference component.
params = {
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
}
model = DriesnerBrineFlowModel(params)
pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)
