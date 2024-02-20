"""A module containing equilibrium formulations for fluid mixtures using PorePy's AD
framework.

"""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np

import porepy as pp

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS as symbols
from .base import AbstractEoS, Component, Compound, Mixture, Phase
from .chem_species import ChemicalSpecies
from .composite_utils import DomainProperty, safe_sum
from .flash import Flash
from .states import FluidState, PhaseState

__all__ = [
    "evaluate_homogenous_constraint",
    "MixtureMixin",
    "EquilibriumEquationsMixin",
    "FlashMixin",
    "SecondaryExpressionsMixin",
]


def evaluate_homogenous_constraint(
    phi: Any, phi_i: list[Any], weights: Optional[list[Any]] = None
) -> Any:
    """Method to evaluate the equality between a quantity ``phi`` and its
    sub-quantities ``phi_i``.

    A safe sum function is used, avoiding an allocation of zero as the first
    summand.

    This method can be used with any first-order homogenous quantity, i.e.
    quantities which are a sum of phase-related quantities weighed with some
    fraction.

    Examples include mass, enthalpy and any other energy of the thermodynamic model.

    Parameters:
        phi: Any homogenous quantity
        y_j: Fractions of how ``phi`` is split into sub-quantities
        phi_i: Sub-quantities of ``phi``, if entity ``i`` where saturated.
        weights: ``default=None``

            If given it must be of equal length as ``phi_i``.

    Returns:
        A (weighed) equality of form :math:`\\phi - \\sum_i \\phi_i w_i`.

        If no weights are given, :math:`w_i` are assumed 1.

    """
    if weights:
        return phi - safe_sum(phi_i)
    else:
        return phi - safe_sum([phi_ * w for phi_, w in zip(phi_i, weights)])


class MixtureMixin:
    """Mixin class for modelling a mixture.

    Provides means to create domain-dependent operators representing fractional unknowns
    and phase properties.

    Introduces the fluid mixture into the model, as well as variables relevant for the
    equilibrium formulation and multi-phase flow (saturations).

    The methods ``get_*`` for various fractions can be overwritten to introduce
    constitutive laws instead of unknowns.

    """

    fluid_mixture: Mixture
    """The fluid mixture set by this class during :meth:`create_mixture`."""

    phases_have_fractions: bool
    """**IMPORTANT**

    A flag indicates whether (extended) fractions of components in phases should
    be created or not. This must be True, if
    :attr:`EquilibriumEquationsMixinEquiequilibrium_type` is not None.

    Important:
        The user **must** inherit this class and set a value explictly.

    If True and ``equilibrium_type == None``, the user must provide secondary
    expressions for the fractional variables to close the system.

    Note:
        Molar fractions of phases (:attr:`~porepy.composite.base.Phase.fraction`) are
        always created, as well as saturations
        (:attr:`~porepy.composite.base.Phase.saturation`).

        This is due to the definition of a mixture properties as a sum of partial
        phase properties weighed with respective fractions.

        Secondary equations, relating molar fractions and saturations via densities, are
        always included in the compositional flow equation.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by: class:`MixtureMixin`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    eliminate_reference_phase: bool
    """Provided by
    :class:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow`."""
    eliminate_reference_component: bool
    """Provided by
    :class:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow`."""

    _feed_fraction_variables: list[str]
    """Created during :meth:`create_fractional_variables`."""
    _solute_fraction_variables: list[str]
    """Created during :meth:`create_fractional_variables`"""
    _saturation_variables: list[str]
    """Created during :meth:`create_fractional_variables`"""
    _phase_fraction_variables: list[str]
    """Created during :meth:`create_fractional_variables`. Contains the names of both
    phase molar fractions and fractions of components in phases."""
    _compositional_fraction_variables: list[str]
    """Created during meth:`create_fractional_variables`. Is empty if
    :attr:`phases_have_fractions` is False"""

    def create_mixture(self) -> None:
        """Mixed-in method to create a mixture.

        It calls :meth:`get_components`, :meth:`get_phase_configuration` and
        :meth:`set_components_in_phases` in that order and creates the instance
        :attr:`fluid_mixture`

        Raises:
            AssertionError: If no component or no phase was modelled.
        """

        components = self.get_components()
        assert len(components) > 0, "No components modelled."
        phase_configurations = self.get_phase_configuration()
        assert len(phase_configurations) > 0, "No phases configured."

        phases: list[Phase] = list()
        for config in phase_configurations:
            eos, type_, name = config
            phases.append(Phase(eos, type_, name))

        self.set_components_in_phases(components, phases)

        self.fluid_mixture = Mixture(components, phases)

    def get_components(self) -> Sequence[Component]:
        """Method to return a list of modelled components.

        Raises:
            NotImplementedError: If not overwritten to return a list of components.

        """
        raise NotImplementedError("No components defined in mixture mixin.")

    def get_phase_configuration(
        self, components: Sequence[Component]
    ) -> Sequence[tuple[AbstractEoS, int, str]]:
        """Method to return a configuration of modelled phases.

        Must return the instance of used EoS, the phase type (integer) and a name
        for each phase.

        This reflects the required input to instantiate a phase
        (see :class:`~porepy.composite.base.Phase`)

        Parameters:
            components: The list of components modelled by :meth:`get_components`.

                Note:
                    The reason why this is passed as an argument is to avoid
                    constructing multiple, possibly expensive EoS compiler instances.

                    The user can use only a single EoS instance for all phases f.e.

        Raises:
            NotImplementedError: If not overwritten to return a list of phases.

        """
        raise NotImplementedError("No phases configured in mixture mixin.")

    def set_components_in_phases(
        self, components: Sequence[Component], phases: Sequence[Phase]
    ) -> None:
        """Method to implement a strategy for which components are added to which phase.

        By default, the unified assumption is applied: All phases contain all
        components.

        Overwrite to do otherwise.

        Important:
            This defines how many unknowns are introduced into the system (fractions
            of a component in a phase).

        Parameters:
            components: The list of components modelled by :meth:`get_components`.
            phases: The list of phases modelled by :meth:`get_phases`.

        """
        for phase in phases:
            phase.components = components

    def fractional_state_from_vector(
        self,
        subdomains: Sequence[pp.Grid],
        state: Optional[np.ndarray] = None,
    ) -> FluidState:
        """Uses the AD framework to create a fluid state from currently stored values of
        fractions.

        Convenience function to get the values for fractions in iterative procedures.

        Evaluates:

        1. Overall fractions per component
        2. Molar fractions per phase
        3. Volumetric fractions per phase (saturations)
        4. Extended fractions per phase per component

        Parameters:
            state: ``default=None``

                Argument for the evaluation methods of the AD framework.
                Can be used to assemble a fluid state from an alternative global vector
                of unknowns.

        Returns:
            A partially filled fluid state data structure containing the above
            fractional values.

        """

        z = np.array(
            [
                component.fraction(subdomains).value(self.equation_system, state)
                for component in self.fluid_mixture.components
            ]
        )

        y = np.array(
            [
                phase.fraction(subdomains).value(self.equation_system, state)
                for phase in self.fluid_mixture.phases
            ]
        )

        sat = np.array(
            [
                phase.saturation(subdomains).value(self.equation_system, state)
                for phase in self.fluid_mixture.phases
            ]
        )

        x = [
            np.array(
                [
                    phase.fraction_of[component](subdomains).value(
                        self.equation_system, state
                    )
                    for component in self.fluid_mixture.components
                ]
            )
            for phase in self.fluid_mixture.phases
        ]

        return FluidState(
            z=z,
            y=y,
            sat=sat,
            phases=[PhaseState(x=x_) for x_ in x],
        )

    def create_fractional_variables(self) -> None:
        """After creating and setting :attr:`fluid_mixture`, this method introduces
        the fractional quantities into the whole system.

        1. :meth:`get_fraction_of_component` is called to assign
           :attr:`~porepy.composite.base.Component.fraction` to components.
        2. :meth:`get_fraction_of_solute_in_compound` is called to assign
           :attr:`~porepy.composite.base.Compound.solute_fraction_of` for each solute in
           a compound.
        3. :meth:`get_saturation_of_phase` is called to assign
           :attr:`~porepy.composite.base.Phase.saturation` to phases.
        4. :meth:`get_fraction_of_phase` is called to assign
           :attr:`~porepy.composite.base.Phase.fraction` to phases.
        5. :meth:`get_fraction_of_component_in_phase` is called to assign
           :attr:`~porepy.composite.base.Phase.fraction_of` for each phase and component
           in that phase.

        Finally, it creates assignes normalized fractions of components in phases
        (see :attr:`~porepy.composite.base.Phase.normalized_fraction_of`).

        The last step, as well as step 5 are skipped if
        :attr:`phases_have_fractions` is False.

        """

        rcomp = self.fluid_mixture.reference_component
        rphase = self.fluid_mixture.reference_phase

        # Creating name containers
        self._feed_fraction_variables = list()
        self._solute_fraction_variables = list()
        self._phase_fraction_variables = list()
        self._saturation_variables = list()
        self._compositional_fraction_variables = list()

        # The creation of variables seems repetative (it is), but it is done this way
        # to preserve a certain order (component-wise, phase-wise and familiy-wise for
        # each family of fractions)

        # Creation of feed fractions
        for component in self.fluid_mixture.components:
            if component == rcomp:  # will be called last
                continue
            fraction, name = self.get_fraction_of_component(component)
            if isinstance(name, str):
                self._feed_fraction_variables.append(name)
            component.fraction = fraction

        # reference feed fraction
        fraction, name = self.get_fraction_of_component(
            self.fluid_mixture.reference_component
        )
        if isinstance(name, str):
            self._feed_fraction_variables.append(name)
        self.fluid_mixture.reference_component.fraction = fraction

        # Creation of solute fractions
        for comp in self.fluid_mixture.components:
            if isinstance(comp, Compound):
                comp.solute_fraction_of = dict()
                for solute in comp.solutes:
                    solute_fraction, name = self.get_fraction_of_solute_in_compound(
                        solute, comp
                    )
                    if isinstance(name, str):
                        self._solute_fraction_variables.append(name)
                    comp.solute_fraction_of.update({solute: solute_fraction})

        # Creation of saturation variables
        for phase in self.fluid_mixture.phases:
            if phase == rphase:  # will be called last
                continue
            saturation, name = self.get_saturation_of_phase(phase)
            if isinstance(name, str):
                self._saturation_variables.append(name)
            phase.saturation = saturation

        # reference phase saturation
        saturation, name = self.get_saturation_of_phase(
            self.fluid_mixture.reference_phase
        )
        if isinstance(name, str):
            self._saturation_variables.append(name)
        self.fluid_mixture.reference_phase.saturation = saturation

        # Creation of phase molar fractions
        for phase in self.fluid_mixture.phases:
            if phase == rphase:  # will be called last
                continue
            fraction, name = self.get_fraction_of_phase(phase)
            if isinstance(name, str):
                self._phase_fraction_variables.append(name)
            phase.fraction = fraction

        # reference phase fraction
        fraction, name = self.get_fraction_of_phase(self.fluid_mixture.reference_phase)
        if isinstance(name, str):
            self._phase_fraction_variables.append(name)
        self.fluid_mixture.reference_phase.fraction = fraction

        # Creation of fractions of components in phases
        if self.phases_have_fractions:
            for phase in self.fluid_mixture.phases:
                phase.fraction_of = dict()
                phase.normalized_fraction_of = dict()

                # creating extended fractions
                for comp in phase.components:
                    fraction_of, name = self.get_fraction_of_component_in_phase(
                        comp, phase
                    )
                    if isinstance(name, str):
                        self._compositional_fraction_variables.append(name)
                    phase.fraction_of.update({comp: fraction_of})

                # creating normalized fractions
                for comp in phase.components:
                    phase.normalized_fraction_of.update(
                        {
                            comp: self.get_normalized_fraction_of_component_in_phase(
                                comp, phase
                            )[0]
                        }
                    )

    def _create_fractional_variable(
        self,
        name: str,
        subdomains: Sequence[pp.Grid],
    ) -> None:
        """Helper method to create individual variables."""
        self.equation_system.create_variables(
            name=name, subdomains=subdomains, tags={"si_units": "-"}
        )

    def get_fraction_of_component(
        self,
        component: Component,
    ) -> tuple[Callable[[Sequence[pp.Grid]], pp.ad.Operator], str | None]:
        """Getter method to create a callable representing the overall fraction of a
        component on a list of subdomains.

        The base method creates independent variables for all components, except for the
        reference component (eliminated by unity).
        The returned callable returns the respective operator.
        If only 1 component is available, it the Callable returns a scalar.

        This method introduces flexibility for the user to implemnt constitutive laws.

        Note:
            This method is called during :meth:`create_fractional_variables`.
            It is called last for the reference component.

            I.e. The user can access fractions of the other components.

        Parameters:
            component: A component modelled in :meth:`get_components`.

        Returns:
            A 2-tuple containing a callable which returns the feed fraction for a given
            set of domains, and a string representing the name of the fraction if it
            was introduced as an independent variable.
            Return a None instead of the string if the operator is not an independent
            variable.

        """
        assert hasattr(self, "fluid_mixture"), "Mixture not set."
        ncomp = self.fluid_mixture.num_components
        rcomp = self.fluid_mixture.reference_component
        subdomains = self.mdg.subdomains()

        name: str | None = None
        fraction: Callable[[Sequence[pp.Grid]], pp.ad.Operator]

        if ncomp == 1:  # If only 1 component, the fraction is always 1

            def fraction(subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single-feed-fraction")

        else:
            # if reference component, eliminate by unity
            if component == rcomp and self.eliminate_reference_component:

                def fraction(domains: Sequence[pp.Grid]) -> pp.ad.Operator:
                    z_R = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
                        [
                            comp.fraction(domains)
                            for comp in self.fluid_mixture.components
                            if comp != rcomp
                        ]
                    )
                    z_R.set_name("reference-feed-fraction-by-unity")
                    return z_R

            else:  # create an independent variable
                name = f"{symbols['component_fraction']}_{component.name}"
                self._create_fractional_variable(name, subdomains)

                def fraction(domains: Sequence[pp.Grid]) -> pp.ad.Operator:
                    return self.equation_system.md_variable(name, domains)

        return fraction, name

    def get_fraction_of_solute_in_compound(
        self, solute: ChemicalSpecies, compound: Compound
    ) -> tuple[Callable[[Sequence[pp.Grid]], pp.ad.Operator], str | None]:
        """Method is called for every compound created and every solute in that compound

        The base method creates solute fractions as independend variables
        (transportable), after asserting the solute is indeed in that compound.

        """
        assert (
            solute in compound.solutes
        ), f"Solute {solute.name} not in compound {compound.name}"
        subdomains = self.mdg.subdomains()

        name = f"{symbols['solute_fraction']}_{solute.name}_{compound.name}"
        self._create_fractional_variable(name, subdomains)

        def solute_fraction(domains: Sequence[pp.Grid]) -> pp.ad.Operator:
            return self.equation_system.md_variable(name, domains)

        return solute_fraction, name

    def get_saturation_of_phase(
        self, phase: Phase
    ) -> tuple[Callable[[Sequence[pp.Grid]], pp.ad.Operator], str | None]:
        """Same as for :meth:`get_feed_fraction_of_component` but for phase saturations.

        The base method creates independent variables for all phases, except for the
        reference phase (eliminated by unity).
        The returned callable returns the respective operator.
        If only 1 phase is available, callable returns a scalar 1.

        This method will be called last for the reference phase.

        """
        assert hasattr(self, "fluid_mixture"), "Mixture not set."
        nphase = self.fluid_mixture.num_phases
        rphase = self.fluid_mixture.reference_phase
        subdomains = self.mdg.subdomains()

        name: str | None = None
        saturation: Callable[[Sequence[pp.Grid]], pp.ad.Operator]

        if nphase == 1:  # If only 1 component, the fraction is always 1

            def saturation(subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single-phase-saturation")

        else:
            # if reference component, eliminate by unity
            if phase == rphase and self.eliminate_reference_phase:

                def saturation(domains: Sequence[pp.Grid]) -> pp.ad.Operator:
                    s_R = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
                        [
                            phase.saturation(domains)
                            for phase in self.fluid_mixture.phases
                            if phase != rphase
                        ]
                    )
                    s_R.set_name("reference-phase-saturation-by-unity")
                    return s_R

            else:  # create an independent variable
                name = f"{symbols['phase_saturation']}_{phase.name}"
                self._create_fractional_variable(name, subdomains)

                def saturation(domains: Sequence[pp.Grid]) -> pp.ad.Operator:
                    return self.equation_system.md_variable(name, domains)

        return saturation, name

    def get_fraction_of_phase(
        self, phase: Phase
    ) -> tuple[Callable[[Sequence[pp.Grid]], pp.ad.Operator], str | None]:
        """Analogous to :meth:`get_saturation_of_phase` but for phase molar fractions."""
        assert hasattr(self, "fluid_mixture"), "Mixture not set."
        nphase = self.fluid_mixture.num_phases
        rphase = self.fluid_mixture.reference_phase
        subdomains = self.mdg.subdomains()

        name: str | None = None
        fraction: Callable[[Sequence[pp.Grid]], pp.ad.Operator]

        if nphase == 1:  # If only 1 component, the fraction is always 1

            def fraction(subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single-phase-fraction")

        else:
            # if reference component, eliminate by unity
            if phase == rphase and self.eliminate_reference_phase:

                def fraction(domains: Sequence[pp.Grid]) -> pp.ad.Operator:
                    y_R = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
                        [
                            phase.fraction(domains)
                            for phase in self.fluid_mixture.phases
                            if phase != rphase
                        ]
                    )
                    y_R.set_name("reference-phase-fraction-by-unity")
                    return y_R

            else:  # create an independent variable
                name = f"{symbols['phase_fraction']}_{phase.name}"
                self._create_fractional_variable(name, subdomains)

                def fraction(domains: Sequence[pp.Grid]) -> pp.ad.Operator:
                    return self.equation_system.md_variable(name, domains)

        return fraction, name

    def get_fraction_of_component_in_phase(
        self, component: Component, phase: Phase
    ) -> tuple[Callable[[Sequence[pp.Grid]], pp.ad.Operator], str | None]:
        """The base method creates an independent variable for any phase and component
        combination, after asserting the component is modelled in that phase."""
        assert (
            component in phase.components
        ), f"Component {component.name} not in phase {phase.name}"
        subdomains = self.mdg.subdomains()

        name = f"{symbols['phase_composition']}_{component.name}_{phase.name}"
        self._create_fractional_variable(name, subdomains)

        def fraction_of(domains: Sequence[pp.Grid]) -> pp.ad.Operator:
            return self.equation_system.md_variable(name, domains)

        return fraction_of, name

    def get_normalized_fraction_of_component_in_phase(
        self, component: Component, phase: Phase
    ) -> tuple[Callable[[Sequence[pp.Grid]], pp.ad.Operator], str | None]:
        """The returned callable constructs an operator which normalizes the fraction of
        a component in a phase, based on what was set in
        :attr:`~porepy.composite.base.Phase.fraction_of`"""
        assert (
            component in phase.components
        ), f"Component {component.name} not in phase {phase.name}"

        name = f"{symbols['phase_composition']}_{component.name}_{phase.name}"

        def normalized_fration_of(subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
            xn = phase.fraction_of[component](subdomains) / pp.ad.sum_operator_list(
                [phase.fraction_of[comp_k](subdomains) for comp_k in phase.components]
            )
            xn.set_name(f"normalized_{name}")
            return xn

        # This is naturally not a variable, but dependent expression
        return normalized_fration_of, None

    def create_phase_properties(self) -> None:
        """A method to create various thermodynamic properties of phases in AD form.

        Will be called by the solution strategy after all variables have been created.

        This base method introduces the properties as
        :class:`~porepy.composite.composite_utils.DomainProperty` instances,
        where each phase property depends on pressure, temperature and normalized
        fractions of components in respective phase.

        It introduces the property on all subdomains their their boundary grids.

        """
        assert hasattr(self, "fluid_mixture"), "Mixture not set."

        subdomains = self.mdg.subdomains()
        boundaries = self.mdg.boundaries()

        for phase in self.fluid_mixture.phases:
            x_j = tuple(phase.fraction_of.values())
            phase.density = DomainProperty(
                f"phase-density-{phase.name}",
                subdomains,
                boundaries,
                self.pressure,
                self.temperature,
                *x_j,
            )
            phase.volume = DomainProperty(
                f"phase-volume-{phase.name}",
                subdomains,
                boundaries,
                self.pressure,
                self.temperature,
                *x_j,
            )
            phase.enthalpy = DomainProperty(
                f"phase-enthalpy-{phase.name}",
                subdomains,
                boundaries,
                self.pressure,
                self.temperature,
                *x_j,
            )
            phase.viscosity = DomainProperty(
                f"phase-viscosity-{phase.name}",
                subdomains,
                boundaries,
                self.pressure,
                self.temperature,
                *x_j,
            )
            phase.conductivity = DomainProperty(
                f"phase-conductivity-{phase.name}",
                subdomains,
                boundaries,
                self.pressure,
                self.temperature,
                *x_j,
            )
            phase.fugacity_of = dict()
            for comp in phase.components:
                phase.fugacity_of[comp] = DomainProperty(
                    f"fugacity-of-{comp.name}-in-{phase.name}",
                    subdomains,
                    boundaries,
                    self.pressure,
                    self.temperature,
                    *x_j,
                )

    def create_mixture_properties(self) -> None:
        """A method to create various thermodynamic properties of the fluid mixture in
        AD form.

        Will be called by the solution strategy after all variables and all phase
        properties have been created.

        It calls:

        - :meth:`get_mixture_density` and assigns it to
          :attr:`~porepy.composite.base.Mixture.density`
        - :meth:`get_mixture_volume` and assigns it to
          :attr:`~porepy.composite.base.Mixture.volume`
        - :meth:`get_mixture_enthalpy` and assigns it to
          :attr:`~porepy.composite.base.Mixture.enthalpy`

        """

        self.fluid_mixture.density = self.get_mixture_density()
        self.fluid_mixture.volume = self.get_mixture_volume()
        self.fluid_mixture.enthalpy = self.get_mixture_enthalpy()

    def get_mixture_density(
        self,
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Implements the mixture density function which returns the sum of
        phase densities weighed with phase saturations.

        Can be overwritten to implement constitutive laws.

        """

        def mixture_density(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            op = pp.ad.sum_operator_list(
                [
                    phase.saturation(domains) * phase.density(domains)
                    for phase in self.fluid_mixture.phases
                ],
                "fluid-mixture-density",
            )
            return op

        return mixture_density

    def get_mixture_volume(
        self,
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Implements the mixture volume function which returns the
        reciprocal of whatever :meth:`get_mixture_volume` returns.

        Can be overwritten to implement constitutive laws.

        """

        def mixture_volume(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            op = self.get_mixture_density()(domains) ** pp.ad.Scalar(-1.0)
            op.set_name("fluid-mixture-volume")
            return op

        return mixture_volume

    def get_mixture_enthalpy(
        self,
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Implements the mixture density function which returns the sum of
        phase enthalpies weighed with phase fractions.

        Can be overwritten to implement constitutive laws.

        """

        def mixture_enthalpy(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            op = pp.ad.sum_operator_list(
                [
                    phase.fraction(domains) * phase.enthalpy(domains)
                    for phase in self.fluid_mixture.phases
                ],
                "fluid-mixture-enthalpy",
            )
            return op

        return mixture_enthalpy


class SecondaryExpressionsMixin:
    """Base class for introducing secondary expressions into the compositional flow
    formulation.

    By default, expressions relating phase molar fractions and saturations via densities
    are always included (see :meth:`density_relation_for_phase`).

    Examples:
        If no equilibrium is defined, the user needs to implement expressions for
        phase densities, phase enthalpies or compositional fractions if they are
        introduced.

        If the user wants a fractional flow formulation (f.e. mobilities are unknowns),
        the user has to introduce laws for them as algebraic equations (cell-wise).

    """

    fluid_mixture: Mixture
    """Provided by: class:`MixtureMixin`."""

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    normalize_state_constraints: bool
    """Provided by
    :class:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow`."""
    eliminate_reference_phase: bool
    """Provided by
    :class:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow`."""

    _density_relations_equation_names: list[str]
    """Created during :meth:`set_density_relations_for_phases`."""

    def get_secondary_equation_names(self) -> list[str]:
        """Returns a list of secondary equations introduced by this mixin.

        The base method returns the names of density relations, since they are included
        by default.

        Important:
            Override this method and append names of additionally included, secondary
            equations in :meth:`set_secondary_equations`.

        """
        return [name for name in self._density_relations_equation_names]

    def set_secondary_equations(self) -> None:
        """Override this method to set secondary expressions in equation form

        .. math::

            f(x) = 0

        by setting the left-hand side as an equation in the Ad framework.

        All equations should be scalar, single, cell-wise equations on each subdomains.

        """

    def set_density_relations_for_phases(self) -> None:
        """Introduced the mass relations for phases into the AD system.

        All equations are scalar, single, cell-wise equations on each subdomains.

        This method is separated, because it has another meaning when coupling the
        equilibrium problem with flow and transport.

        In multiphase flow in porous media, saturations must always be provided.
        Hense even if there are no isochoric specifications in the flash, the model
        necessarily introduced the saturations as unknowns.

        The mass relations per phase close the system, by relating molar phase fractions
        to saturations. Hence rendering the system solvable.

        Important:
            If there is only 1 phase, this method does nothing, since in that case
            the molar fraction and saturation of a phase is always 1.

            If the user wants it nevertheless for some reason,
            :meth:`density_relation_for_phase` must be called explicitly.

            That equation is not tracked by this class in this case.
            Hence :meth:`get_secondary_equation_names` will give an empty list.

        """
        rphase = self.fluid_mixture.reference_phase
        subdomains = self.mdg.subdomains()
        self._density_relations_equation_names = list()
        if self.fluid_mixture.num_phases > 1:
            for phase in self.fluid_mixture.phases:
                if phase == rphase and self.eliminate_reference_phase:
                    continue
                equ = self.density_relation_for_phase(phase, subdomains)
                self._density_relations_equation_names.append(equ.name)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})

    def density_relation_for_phase(
        self, phase: Phase, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs a local mass relation based on a relation between mixture
        density, saturated phase density and phase fractions.

        For a phase :math:`j` it holds:

        .. math::

            y_j \\rho - s_j \\rho_j = 0~,~
            y_j - s_j \\dfrac{\\rho_j}{rho} = 0

        with the mixture density :math:`\\rho = \\sum_k s_k \\rho_k`, assuming
        :math:`\\rho_k` is the density of a phase when saturated.

        - :math:`y` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`s` : Phase :attr:`~porepy.composite.base.Phase.saturation`

        The relation based on phase fractions is required for isochoric
        equilibrium specificiations, which have both saturations and molar fractions as
        unknowns.

        Parameters:
            phase: A phase for which the constraint should be assembled.
            subdomains: A list of subdomains on which this relation is defined.

        Returns:
            The left-hand side of above equations.

            If normalization of state constraints is set in the solution strategy,
            it returns the normalized form.

        """
        if self.normalize_state_constraints:
            equ = phase.fraction(subdomains) - phase.saturation(
                subdomains
            ) * phase.density(subdomains) / self.fluid_mixture.density(subdomains)
        else:
            equ = phase.fraction(subdomains) * self.fluid_mixture.density(
                subdomains
            ) - phase.saturation(subdomains) * phase.density(subdomains)
        equ.set_name(f"density-relation-{phase.name}")
        return equ


class EquilibriumEquationsMixin:
    """Basic class introducing the fluid phase equilibrium equations into the model."""

    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """**IMPORTANT**

    A string denoting the two state functions which are assumed constant at
    equilibrium.

    Important:
        The user **must** inherit this class and set a value explictly.

    If set to ``None``, the framework assumes there are no equilibrium calculations,
    hence there are **no equilibrium equations** and **no equilibriation of the fluid**.

    Also, **no fractions of components in phases** are created as unknowns
    (see :attr:`~porepy.composite.base.Phase.fraction_of`).

    The user must in this case provide secondary equations which provide expressions for
    the unknowns in the equilibrium:

    Examples:
        If no equilibrium is defined, there are dangling variables which need a
        definition or a constitutive law.

        These include:

        1. Temperature
           :meth:`~porepy.models.energy_balance.VariablesEnergyBalance.temperature`
        2. Phase saturations :attr:`~porepy.composite.base.Phase.saturation`
        3. Optionally molar fractions of components in phases
           :attr:`~porepy.composite.base.Phase.fraction_of` if
           :attr:`phases_have_fractions` is True.

        Note that secondary expressions relating molar phase fractions and saturations
        via densities are always included.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    fluid_mixture: Mixture
    """Provided by :class:`MixtureMixin`."""

    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by
    :class:`~porepy.models.compositional_balance.VariablesCompositionalFlow`."""
    volume: Callable[[list[pp.Grid]], pp.ad.Operator]
    """TODO Not covered so far."""

    phases_have_fractions: bool
    """Provided by :class:`MixtureMixin`."""
    eliminate_reference_component: bool
    """Provided by
    :class:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow`."""
    eliminate_reference_phase: bool
    """Provided by
    :class:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow`."""
    use_semismooth_complementarity: bool
    """Provided by
    :class:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow`."""
    normalize_state_constraints: bool
    """Provided by
    :class:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow`."""

    _equilibrium_equation_names: list[str]
    """Created during :meth:`set_equations`."""

    def get_equilibrium_equation_names(self) -> list[str]:
        """Get a list of equation names introduced into the AD framework by this class
        depending on :attr:`equilibrium_type`.

        Returns an empty list if ``equilibrium_type==None``.

        """
        if self.equilibrium_type is None:
            return list()
        else:
            return [name for name in self._equilibrium_equation_names]

    def set_equations(self) -> None:
        """Introduced the local equilibrium equations into the AD framework.

        All equations are scalar, single, cell-wise equations on each subdomains.

        1. Mass constraints: Except for reference component if eliminiated
          If only one component is present, the mass constraint is set nevertheless.
        2. Isofugacity constraints for all components between an independent phase and
          the reference phase
        3. Complementary conditions per phase.
        4. State constraints for more complex flash types than p-T

        Note:
            The equilibrium calculations make no sense if only one phase is modelled.
            This method will raise an error in this case.

        Important:
            This set-up does not cover all cases. If for example only 1 component is
            present, and the reference phase fraction is eliminated, the equilibrium
            system is over-determined.

            This singular case must be dealt with separately.

        """
        subdomains = self.mdg.subdomains()
        self._equilibrium_equation_names = list()
        ncomp = self.fluid_mixture.num_components
        nphase = self.fluid_mixture.num_phases

        assert (
            self.phases_have_fractions
        ), "Cannot set equilibrium equations: Compositional fractions not created"

        if nphase == 1:
            raise ValueError(
                f"Equilibrium system set-up is meaningless for only 1 modelled phase."
            )
        ## starting with equations common to all equilibrium definitions
        # local mass constraint per component
        for component in self.fluid_mixture.components:
            # skip for reference component if eliminated
            if (
                component == self.fluid_mixture.reference_component
                and self.eliminate_reference_component
                and ncomp > 1
            ):
                continue
            equ = self.mass_constraint_for_component(component, subdomains)
            self._equilibrium_equation_names.append(equ.name)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # isofugacity constraints
        # NOTE This need more elaboration for complex mixtures where components are not
        # in all phases
        rphase = self.fluid_mixture.reference_phase
        for phase in self.fluid_mixture.phases:
            if phase == rphase:
                continue
            for component in self.fluid_mixture.components:
                if not (
                    component in phase.components and component in rphase.components
                ):
                    continue
                equ = self.isofugacity_constraint_for_component_in_phase(
                    component, phase, subdomains
                )
                self._equilibrium_equation_names.append(equ.name)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # complementarity conditions
        for phase in self.fluid_mixture.phases:
            equ = self.complementarity_condition_for_phase(phase, subdomains)
            self._equilibrium_equation_names.append(equ.name)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # No more equations for p-T based flash
        if self.equilibrium_type == "p-T":
            pass
        # 1 more equation for p-h based flash (T unknown)
        elif self.equilibrium_type == "p-h":
            # here T is another unknown, but h is fixed. Introduce 1 more equations
            equ = self.mixture_enthalpy_constraint(subdomains)
            self._equilibrium_equation_names.append(equ.name)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})
        # 2 + num_phase - 1 more equations for v-h flash (p, T, s_j unknown)
        elif self.equilibrium_type == "v-h":
            equ = self.mixture_enthalpy_constraint(subdomains)
            self._equilibrium_equation_names.append(equ.name)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})
            equ = self.mixture_volume_constraint(subdomains)
            self._equilibrium_equation_names.append(equ.name)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})

    def mass_constraint_for_component(
        self, component: Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the local mass constraint for a component :math:`i`.

        .. math::

            z_i - \\sum_j x_{ij} y_j = 0.

        - :math:`z` : Component :attr:`~porepy.composite.base.Component.fraction`
        - :math:`y` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`x` : Phase :attr:`~porepy.composite.base.Phase.fraction_of` component

        The above sum is performed over all phases the component is present in.

        Parameter:
            component: The component represented by the overall fraction :math:`z_i`.
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            An operator representing the left-hand side of above equation.

        """
        # get all phases the component is present in
        phases = [
            phase
            for phase in self.fluid_mixture.phases
            if component in phase.components
        ]

        equ: pp.ad.Operator = evaluate_homogenous_constraint(
            component.fraction(subdomains),
            [phase.fraction_of[component](subdomains) for phase in phases],
            [phase.fraction(subdomains) for phase in phases],
        )  # type:ignore
        equ.set_name(f"mass-constraint-{component.name}")
        return equ

    def complementarity_condition_for_phase(
        self, phase: Phase, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the complementarity condition for a given phase.

        .. math::

            y_j (1 - \\sum_i x_{ij}) = 0~,~
            \\min \\{y_j, (1 - \\sum_i x_{ij}) \\} = 0.

        - :math:`y` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`x` : Phase :attr:`~porepy.composite.base.Phase.fraction_of` component

        The sum is performed over all components modelled in that phase
        (see :attr:`~porepy.composite.base.Phase.components`).

        Parameters:
            phase: The phase for which the condition is assembled.
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equation. If the semi-smooth form is
            requested by the solution strategy, then the :math:`\\min\\{\\}` operator is
            used.

        """

        unity: pp.ad.Operator = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
            [phase.fraction_of[comp](subdomains) for comp in phase.components]
        )

        if self.use_semismooth_complementarity:
            equ = pp.ad.SemiSmoothMin(phase.fraction(subdomains), unity)
            equ.set_name(f"semismooth-complementary-condition-{phase.name}")
        else:
            equ = phase.fraction(subdomains) * unity
            equ.set_name(f"complementary-condition-{phase.name}")
        return equ

    def isofugacity_constraint_for_component_in_phase(
        self, component: Component, phase: Phase, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Construct the local isofugacity constraint for a component between a given
        phase and the reference phase.

        .. math::

            x_{ij} \\varphi_{ij} - x_{iR} \\varphi_{iR} = 0.

        - :math:`x_{ij}` : :attr:`~porepy.composite.base.Phase.fraction_of` component
        - :math:`\\varphi_{ij}` : Phase :attr:`~porepy.composite.base.Phase.fugacity_of`
          component

        Parameters:
            component: A component characterized by the relative fractions in above
                equation.
            phase: The phase denoted by index :math:`j` in above equation.
            subdomains: A list of subdomains on which to define the equation.

        Raises:
            ValueError: If ``phase`` is the reference phase.
            AssertionError: If the component is not present in both reference and passed
                phase.

        Returns:
            The left-hand side of above equation.

        """
        if phase == self.fluid_mixture.reference_phase:
            raise ValueError(
                "Cannot construct isofugacity constraint between reference phase and "
                + "itself."
            )
        assert (
            component in phase.components
        ), "Passed component not modelled in passed phase."
        assert (
            component in self.fluid_mixture.reference_phase.components
        ), "Passed component not modelled in reference phase."

        equ = phase.fraction_of[component](subdomains) * phase.fugacity_of[component](
            subdomains
        ) - self.fluid_mixture.reference_phase.fraction_of[component](
            subdomains
        ) * self.fluid_mixture.reference_phase.fugacity_of[
            component
        ](
            subdomains
        )
        equ.set_name(
            f"isofugacity-constraint-"
            + f"{component.name}-{phase.name}-{self.fluid_mixture.reference_phase.name}"
        )
        return equ

    def mixture_enthalpy_constraint(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the enthalpy constraint for the mixture enthalpy and the
        transported enthalpy variable.

        .. math::

            \\sum_j y_j h_j  - h = 0~,~
            (\\sum_j y_j h_j) / h - 1= 0~

        - :math:`y_j`: Phase :attr:`~porepy.composite.base.Phase.fraction`.
        - :math:`h_j`: Phase :attr:`~porepy.composite.base.Phase.enthalpy`.
        - :math:`h`: The transported enthalpy :attr:`enthalpy`.

        The first term represents the mixture enthalpy based on the thermodynamic state.
        The second term represents the target enthalpy in the equilibrium problem.
        The target enthalpy is a transportable quantity in flow and transport.

        Parameters:
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equations. If the normalization of state
            constraints is required by the solution strategy, the second form is
            returned.

        """
        if self.normalize_state_constraints:
            equ = self.fluid_mixture.enthalpy(subdomains) / self.enthalpy(
                subdomains
            ) - pp.ad.Scalar(1.0)
        else:
            equ = self.fluid_mixture.enthalpy(subdomains) - self.enthalpy(subdomains)
        equ.set_name("mixture-enthalpy-constraint")
        return equ

    def mixture_volume_constraint(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the volume constraint using the reciprocal of the mixture density.

        .. math::

            \\dfrac{1}{\\sum_j s_j \\rho_j} - v = 0~,~
            v \\left(\\sum_j s_j \\rho_j\\right) - 1 = 0.

        - :math:`s_j` : Phase :attr:`~porepy.composite.base.Phase.saturation`
        - :math:`\\rho_j` : Phase :attr:`~porepy.composite.base.Phase.density`

        Parameters:
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equations. If the normalization of state
            constraints is required by the solution strategy, the second form is
            returned.

        """
        if self.normalize_state_constraints:
            equ = self.volume(subdomains) * self.fluid_mixture.density(
                subdomains
            ) - pp.ad.Scalar(1.0)
        else:
            equ = self.volume(subdomains) - self.fluid_mixture.density(
                subdomains
            ) ** pp.ad.Scalar(-1.0)
        equ.set_name("mixture-volume-constraint")
        return equ


class FlashMixin:
    """Mixin class to introduce the flash procedure into the solution strategy.

    Main ideas of the FlashMixin:

    1. Instantiation of Flash object and make it available for other mixins.
    2. Convenience methods to equilibriate the fluid.
    3. Abstraction to enable customization.

    """

    flash: Flash
    """A flasher object able to compute the fluid phase equilibrium for a mixture
    defined in the mixture mixin.

    This object should be created here during :meth:`set_up_flasher`.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    fluid_mixture: Mixture
    """Provided by :class:`MixtureMixin`."""

    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by :class:`EquilibriumEquationsMixin`."""

    fractional_state_from_vector: Callable[
        [Sequence[pp.Grid], Optional[np.ndarray]], FluidState
    ]
    """Provided by :class:`MixtureMixin`."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by
    :class:`~porepy.models.compositional_balance.VariablesCompositionalFlow`."""
    volume: Callable[[list[pp.Grid]], pp.ad.Operator]
    """TODO Not covered so far."""

    def set_up_flasher(self) -> None:
        """Method to introduce the flash class, if an equilibrium is defined.

        This method is called by the solution strategy after the model is set up.

        """
        if self.equilibrium_type is not None:
            raise NotImplementedError(
                f"No flash set-up implemented for {self.equilibrium_type} equilibrium."
            )

    def equilibriate_fluid(
        self, subdomains: Sequence[pp.Grid], state: Optional[np.ndarray] = None
    ) -> tuple[FluidState, np.ndarray]:
        """Convenience method to assemble the state of the fluid based on a global
        vector and to equilibriate it using the flasher.

        This method is called in
        :meth:`~porepy.models.compositional_balance.SolutionStrategyCompositionalFlow.
        before_nonlinear_iteration` to use the flash as a predictor during nonlinear
        iterations.

        Parameters:
            state: ``default=None``

                Global state vector to be passed to the Ad framework when evaluating the
                current state (fractions, pressure, temperature, enthalpy,..)

        Returns:
            The equilibriated state of the fluid and an indicator where the flash was
            successful (or not).

            For more information on the `success`-indicators, see respective flash
            object.

        """
        assert (
            self.equilibrium_type is not None
        ), "Cannot equilibriate fluid with no equilibrium type defined."
        # Extracting the current, iterative state to use as initial guess for the flash
        fluid_state = self.fractional_state_from_vector(subdomains, state)

        if self.equilibrium_type == "p-T":
            p = self.pressure(subdomains).value(self.equation_system, state)
            T = self.temperature(subdomains).value(self.equation_system, state)
            result_state, succes, _ = self.flash.flash(
                z=fluid_state.z,
                p=p,
                T=T,
                initial_state=fluid_state,
            )
        elif self.equilibrium_type == "p-h":
            p = self.pressure(subdomains).value(self.equation_system, state)
            h = self.enthalpy(subdomains).value(self.equation_system, state)
            # initial guess for T from iterate
            fluid_state.T = self.temperature(subdomains).value(
                self.equation_system, state
            )
            result_state, succes, _ = self.flash.flash(
                z=fluid_state.z,
                p=p,
                h=h,
                initial_state=fluid_state,
            )
        elif self.equilibrium_type == "v-h":
            v = self.volume(subdomains).value(self.equation_system, state)
            h = self.enthalpy(subdomains).value(self.equation_system, state)
            # initial guess for T, p from iterate, saturations already contained
            fluid_state.T = self.temperature(subdomains).value(
                self.equation_system, state
            )
            fluid_state.p = self.pressure(subdomains).value(self.equation_system, state)
            result_state, succes, _ = self.flash.flash(
                z=fluid_state.z,
                v=v,
                h=h,
                initial_state=fluid_state,
            )
        else:
            raise NotImplementedError(
                f"Equilibriation not implemented for {self.equilibrium_type} flash."
            )

        return result_state, succes

    def postprocess_failures(
        self, fluid_state: FluidState, success: np.ndarray
    ) -> FluidState:
        """A method called after :meth:`equilibriate_fluid` to post-process failures if
        any.

        Parameters:
            fluid_state: Fluid state returned from :meth:`equilibriate_fluid`.
            success: Success flags returned along the fluid state.

        Returns:
            A final fluid state, with treatment of values where the flash did not
            succeed.

        """
        # nothing to do if everything successful
        if np.all(success == 0):
            return fluid_state
        else:
            NotImplementedError("No flash postprocessing implemented.")
