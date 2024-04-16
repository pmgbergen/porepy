"""A module containing equilibrium formulations for fluid mixtures using PorePy's AD
framework.

"""
from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence

import numpy as np

import porepy as pp

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS as symbols
from .base import AbstractEoS, Component, Compound, Mixture, Phase
from .chem_species import ChemicalSpecies
from .composite_utils import CompositeModellingError, SecondaryExpression
from .states import FluidState, PhaseState

__all__ = [
    "CompositeVariables",
    "FluidMixtureMixin",
]


class CompositeVariables(pp.VariableMixin):
    """Mixin class for models with mixtures which defines the respective fractional
    unknowns.

    Fractional variables are relevant for the equilibrium formulation, as well as
    for compositional flow.

    Various methods can be overwritten to introduce constitutive laws instead of
    unknowns.

    Important:
        For compositional flow without a local equilibrium formulation, the flow and
        transport formulationd does not require molar phase fractions or extended
        fractions of components in phases.
        Phases have only saturations as phase related variables, and instead of extended
        fractions, the (physical) partial fractions are independent variables.

        When setting up models, keep in mind that you need to close the flow and
        transport model with local equations, eliminating saturations and partial
        fractions by some function depending on the primary flow and transport variables
        (pressure, temperature, overall fractions or pressure, enthalpy, and overall
        fractions).

    """

    fluid_mixture: Mixture
    """Provided by :class:`FluidMixtureMixin`."""

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Provided by :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.
    """

    eliminate_reference_phase: bool
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""
    eliminate_reference_component: bool
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""
    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by
    :class:`~porepy.models.composite_balance.SolutionStrategyCF`."""

    @property
    def overall_fraction_variables(self) -> list[str]:
        """Names of independent feed fraction variables created by the mixture mixin."""
        names = list()
        if hasattr(self, "fluid_mixture"):
            for comp in self.fluid_mixture.components:
                if not (
                    comp == self.fluid_mixture.reference_component
                    and self.eliminate_reference_component
                ):
                    names.append(self._overall_fraction_variable(comp))
        return names

    @property
    def solute_fraction_variables(self) -> list[str]:
        """Names of solute fraction variables created by the mixture mixin."""
        names = list()
        if hasattr(self, "fluid_mixture"):
            for comp in self.fluid_mixture.components:
                if isinstance(comp, Compound):
                    for solute in comp.solutes:
                        names.append(self._solute_fraction_variable(solute, comp))
        return names

    @property
    def phase_fraction_variables(self) -> list[str]:
        """Names of independent phase fraction variables created by the mixture mixin."""
        names = list()
        if hasattr(self, "fluid_mixture"):
            for phase in self.fluid_mixture.phases:
                if not (
                    phase == self.fluid_mixture.reference_phase
                    and self.eliminate_reference_phase
                ):
                    names.append(self._phase_fraction_variable(phase))
        return names

    @property
    def saturation_variables(self) -> list[str]:
        """Names of phase saturation variables created by the mixture mixin."""
        names = list()
        if hasattr(self, "fluid_mixture"):
            for phase in self.fluid_mixture.phases:
                if not (
                    phase == self.fluid_mixture.reference_phase
                    and self.eliminate_reference_phase
                ):
                    names.append(self._saturation_variable(phase))
        return names

    @property
    def relative_fraction_variables(self) -> list[str]:
        """Names of fraction variables denoting the fraction of a component in a phase.

        If a local equilibrium is defined, this denotes the extended fractions,
        otherwise it denotes the partial fractions.

        """
        names = list()
        if hasattr(self, "fluid_mixture"):
            for phase in self.fluid_mixture.phases:
                for comp in phase:
                    names.append(self._relative_fraction_variable(comp, phase))
        return names

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
        2. Fractions per phase
        3. Volumetric fractions per phase (saturations)
        4. Fractions per phase per component
           (extended if equilibrium defined, else partial)

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
                    if self.equilibrium_type is not None
                    else phase.partial_fraction_of[component](subdomains).value(
                        self.equation_system, state
                    )
                    for component in phase
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

    def _create_fractional_variable(
        self,
        name: str,
        subdomains: Sequence[pp.Grid],
    ) -> None:
        """Helper method to create individual variables."""
        self.equation_system.create_variables(
            name=name, subdomains=subdomains, tags={"si_units": "-"}
        )

    def create_variables(self) -> None:
        """Creates the sets of required variables for a fluid mixture.

        1. :meth:`overall_fraction` is called to assign
           :attr:`~porepy.composite.base.Component.fraction` to components.
        2. :meth:`solute_fraction` is called to assign
           :attr:`~porepy.composite.base.Compound.solute_fraction_of` for each solute in
           a compound.
        3. :meth:`saturation` is called to assign
           :attr:`~porepy.composite.base.Phase.saturation` to phases.

        If a local :attr:`equilibrium_type` is defined, it introduces additionally

        4. :attr:`~porepy.composite.base.Phase.fraction` to phases by calling
           :meth:`phase_fraction'
        5. :attr:`~porepy.composite.base.Phase.fraction_of` for each phase and component
           by calling :meth:`extended_fraction`

        If a local equilibrium is defined :meth:`partial_fractions` returns dependent
        operators by normalizing extended fractions. Otherwise partial fractions are
        also introduced as as independent operators by calling
        :attr:`~porepy.composite.base.Phase.partial_fraction_of` for each phase and
        component in that phase, and assigning it to :meth:`partial_fraction`.

        """
        if not hasattr(self, "fluid_mixture"):
            raise CompositeModellingError(
                "Cannot create fluid mixture variables before defining a fluid mixture."
            )

        rcomp = self.fluid_mixture.reference_component
        rphase = self.fluid_mixture.reference_phase
        subdomains = self.mdg.subdomains()

        # NOTE: The creation of variables seems repetative (it is), but it is done this
        # way to preserve a certain order (component-wise, phase-wise and familiy-wise
        # for each family of fractions)

        ## Creation of feed fractions
        for component in self.fluid_mixture.components:
            if component != rcomp:  # will be called last
                name = self._overall_fraction_variable(component)
                self._create_fractional_variable(name, subdomains)
                component.fraction = self.overall_fraction(component)
        # reference feed fraction
        rcomp.fraction = self.overall_fraction(rcomp)

        ## Creation of solute fractions
        for comp in self.fluid_mixture.components:
            if isinstance(comp, Compound):
                comp.solute_fraction_of = dict()
                for solute in comp.solutes:
                    name = self._solute_fraction_variable(solute, comp)
                    self._create_fractional_variable(name, subdomains)
                    comp.solute_fraction_of.update(
                        {solute: self.solute_fraction(solute, comp)}
                    )

        # Creation of saturation variables
        for phase in self.fluid_mixture.phases:
            if phase != rphase:  # will be called last
                name = self._saturation_variable(phase)
                self._create_fractional_variable(name, subdomains)
                phase.saturation = self.saturation(phase)
        # reference phase saturation
        rphase.saturation = self.saturation(rphase)

        # Creation of molar phase fractions, extended fractions of components in phases
        # and partial fractions as dependent operators (by normalization)

        # some kind of relative fraction exists always, independent of equilibrium type
        for phase in self.fluid_mixture.phases:
            for comp in phase:
                self._create_fractional_variable(
                    self._relative_fraction_variable(comp, phase), subdomains
                )

        # If the equilibrium is defined, molar phase fractions and extended fractions
        # exist. The extended fractions are the independent compositional fractions
        if self.equilibrium_type is not None:
            for phase in self.fluid_mixture.phases:
                # assigning molar fractions to independent phases
                if phase != rphase:
                    name = self._phase_fraction_variable(phase)
                    self._create_fractional_variable(name, subdomains)
                    phase.fraction = self.phase_fraction(phase)

                # creating extended fractions
                phase.fraction_of = dict()
                for comp in phase:
                    phase.fraction_of.update(
                        {comp: self.extended_fraction(comp, phase)}
                    )

            # reference phase fraction
            rphase.fraction = self.phase_fraction(rphase)

        # Partial fractions exist independent of the equilibrium type
        # But if there is no equilibrium type, they are the independent variables
        # Otherwise they are created by normalization of extended variables
        for phase in self.fluid_mixture.phases:
            phase.partial_fraction_of = dict()
            for comp in phase:
                phase.partial_fraction_of.update(
                    {comp: self.partial_fraction(comp, phase)}
                )

    def _overall_fraction_variable(self, component: Component) -> str:
        """Returns the name of the feed fraction variable assigned to ``component``."""
        return f"{symbols['overall_fraction']}_{component.name}"

    def overall_fraction(
        self,
        component: Component,
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Getter method to create a callable representing the overall fraction of a
        component on a list of subdomains or boundaries.

        The base method creates independent variables for all components, except for the
        reference component (eliminated by unity).
        The returned callable returns the respective operator.
        If only 1 component is available, it the Callable returns a scalar.

        Note:
            This method is called during :meth:`create_variables`.
            It is called last for the reference component.
            I.e. The user can access fractions of the other components.

        Parameters:
            component: A component in the fluid mixture.

        Returns:
            A callable which returns the feed fraction for a given set of domains.

        """
        ncomp = self.fluid_mixture.num_components
        rcomp = self.fluid_mixture.reference_component

        fraction: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

        if ncomp == 1:  # If only 1 component, the fraction is always 1

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single-feed-fraction")

        else:
            # if reference component, eliminate by unity
            if component == rcomp and self.eliminate_reference_component:

                def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
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

                def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    name = self._overall_fraction_variable(component)
                    if len(domains) > 0 and all(
                        [isinstance(g, pp.BoundaryGrid) for g in domains]
                    ):
                        return self.create_boundary_operator(
                            name=name, domains=domains  # type: ignore[call-arg]
                        )
                    # Check that the domains are grids.
                    if not all([isinstance(g, pp.Grid) for g in domains]):
                        raise ValueError(
                            """Argument domains a mixture of subdomain and boundaries"""
                        )
                    return self.equation_system.md_variable(name, domains)

        return fraction

    def _solute_fraction_variable(
        self, solute: ChemicalSpecies, compound: Compound
    ) -> str:
        """Returns the name of the solute fraction variable assigned to solute in a
        compound."""
        return f"{symbols['solute_fraction']}_{solute.name}_{compound.name}"

    def solute_fraction(
        self, solute: ChemicalSpecies, compound: Compound
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Method is called for every compound created and every solute in that
        compound.

        The base method creates solute fractions as independend variables
        (transportable), after asserting the solute is indeed in that compound.

        """
        assert (
            solute in compound.solutes
        ), f"Solute {solute.name} not in compound {compound.name}"

        def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            name = self._solute_fraction_variable(solute, compound)
            if len(domains) > 0 and all(
                [isinstance(g, pp.BoundaryGrid) for g in domains]
            ):
                return self.create_boundary_operator(
                    name=name, domains=domains  # type: ignore[call-arg]
                )
            # Check that the domains are grids.
            if not all([isinstance(g, pp.Grid) for g in domains]):
                raise ValueError(
                    """Argument domains a mixture of subdomain and boundaries."""
                )
            return self.equation_system.md_variable(name, domains)

        return fraction

    def _saturation_variable(self, phase: Phase) -> str:
        """Returns the name of the saturation variable assigned to ``phase``."""
        return f"{symbols['phase_saturation']}_{phase.name}"

    def saturation(
        self, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Same as for :meth:`overall_fraction` but for phase saturations.

        The base method creates independent variables for all phases, except for the
        reference phase (eliminated by unity).
        The returned callable returns the respective operator.
        If only 1 phase is available, callable returns a scalar 1.

        This method will be called last for the reference phase.

        """
        assert hasattr(self, "fluid_mixture"), "Mixture not set."
        nphase = self.fluid_mixture.num_phases
        rphase = self.fluid_mixture.reference_phase

        saturation: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

        if nphase == 1:  # If only 1 component, the fraction is always 1

            def saturation(subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single-phase-saturation")

        else:
            # if reference component, eliminate by unity
            if phase == rphase and self.eliminate_reference_phase:

                def saturation(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
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

                def saturation(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    name = self._saturation_variable(phase)
                    if len(domains) > 0 and all(
                        [isinstance(g, pp.BoundaryGrid) for g in domains]
                    ):
                        return self.create_boundary_operator(
                            name=name, domains=domains  # type: ignore[call-arg]
                        )
                    # Check that the domains are grids.
                    if not all([isinstance(g, pp.Grid) for g in domains]):
                        raise ValueError(
                            """Argument domains a mixture of subdomain and boundaries"""
                        )
                    return self.equation_system.md_variable(name, domains)

        return saturation

    def _phase_fraction_variable(self, phase: Phase) -> str:
        """Returns the name of the molar phase fraction variable assigned to ``phase``."""
        return f"{symbols['phase_fraction']}_{phase.name}"

    def phase_fraction(
        self, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Analogous to :meth:`saturation` but for phase molar fractions."""
        assert hasattr(self, "fluid_mixture"), "Mixture not set."
        nphase = self.fluid_mixture.num_phases
        rphase = self.fluid_mixture.reference_phase

        fraction: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

        if nphase == 1:  # If only 1 component, the fraction is always 1

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single-phase-fraction")

        else:
            # if reference component, eliminate by unity
            if phase == rphase and self.eliminate_reference_phase:

                def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
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

                def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    name = self._phase_fraction_variable(phase)
                    if len(domains) > 0 and all(
                        [isinstance(g, pp.BoundaryGrid) for g in domains]
                    ):
                        return self.create_boundary_operator(
                            name=name, domains=domains  # type: ignore[call-arg]
                        )
                    # Check that the domains are grids.
                    if not all([isinstance(g, pp.Grid) for g in domains]):
                        raise ValueError(
                            """Argument domains a mixture of subdomain and boundaries"""
                        )
                    return self.equation_system.md_variable(name, domains)

        return fraction

    def _relative_fraction_variable(self, component: Component, phase: Phase) -> str:
        """Returns the name of the (extended or partial) fraction variable of
        ``component`` in ``phase``."""
        return f"{symbols['phase_composition']}_{component.name}_{phase.name}"

    def extended_fraction(
        self, component: Component, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """The base method creates an independent variable for any phase and component
        combination, after asserting the component is modelled in that phase.

        Note:
            Compared to partial fractions, extended fractions are always independent
            even in the case of only 1 component in a phase.
            This is because they are not necessarily 1 at equilibrium.

        """
        assert (
            component in phase
        ), f"Component {component.name} not in phase {phase.name}"

        # Extended fractions are always independent, even in the single-component case,
        # because they are not necessarily 1.
        def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            name = self._relative_fraction_variable(component, phase)
            if len(domains) > 0 and all(
                [isinstance(g, pp.BoundaryGrid) for g in domains]
            ):
                return self.create_boundary_operator(
                    name=name, domains=domains  # type: ignore[call-arg]
                )
            # Check that the domains are grids.
            if not all([isinstance(g, pp.Grid) for g in domains]):
                raise ValueError(
                    """Argument domains a mixture of subdomain and boundaries."""
                )
            return self.equation_system.md_variable(name, domains)

        return fraction

    def partial_fraction(
        self, component: Component, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Returns Ad representations of (physical) partial fractions.

        If the mixture has extended fractions (unified flash), partial fractions
        are obtained by normalizing the :meth:`extended_fraction` per phase.

        If the mixture has no extended fractions, the partial fractions are independent
        operators.

        if the phase has only 1 component, the single partial fraction is automatically
        constant 1.

        Note:
            If the partial fractions are independent operators, this method
            uses internally :meth:`extended_fractions` because the code to create
            an independent variable for a fraction of a component in a phase is
            identical.

            The same variable name is used.

        """

        if self.equilibrium_type is not None:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                xn = self.extended_fraction(component, phase)(
                    domains
                ) / pp.ad.sum_operator_list(
                    [self.extended_fraction(comp_k, phase)(domains) for comp_k in phase]
                )
                xn.set_name(
                    "normalized_"
                    + f"{self._relative_fraction_variable(component, phase)}"
                )
                return xn

        else:
            # Physical fraction are constant 1, if only 1 component
            # Otherwise we use the code from extended fractions to create independent
            # variables
            if len(phase.components) == 1:

                def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    return pp.ad.Scalar(
                        1.0, f"single-partial-fraction_{component.name}_{phase.name}"
                    )

            else:
                fraction = self.extended_fraction(component, phase)

        return fraction


class FluidMixtureMixin:
    """Mixin class for modelling a mixture.

    Introduces the fluid mixture into the model.

    Provides means to create domain-dependent callables representing thermodynamic
    properties which appear in various equations.

    Various methods methods returning callable properties can be overwritten for
    customization.

    The callable properties are assigned to the instances of components, phases and
    the fluid mixture for convenience
    (see :meth:`assign_thermodynamic_properties_to_mixture`).

    This reduces the code complexity of the CF framework significantly.

    This base class is designed to accomodate the most general formulation of the
    compositional flow paired with local equilibrium equations.

    Properties of phases
    (:class:`~porepy.composite.composite_utils.SecondaryExpression`)
    are such that the solution strategy can populate values depending on whether flash
    calculations are performed or secondary expressions evaluated.

    Before modifying anything here, try accomodating the update in the solution
    strategy (see :meth:`~porepy.models.compositional_flow.
    SolutionStrategyCF.update_thermodynamic_properties`).

    Note:
        Secondary expressions are given a time step and iterate depth, depending on
        the solution strategy.

        Expressions which do not appear in the time derivative, such as viscosity,
        conductivity and fugacity coefficients, never have a time step depth, and only
        the iterate values are stored.

    """

    fluid_mixture: Mixture
    """The fluid mixture set by this class during :meth:`create_mixture`."""

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    eliminate_reference_phase: bool
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""
    eliminate_reference_component: bool
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""
    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by
    :class:`~porepy.models.composite_balance.SolutionStrategyCF`."""

    time_step_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`"""
    iterate_indices: np.ndarray
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`"""

    def create_mixture(self) -> None:
        """Mixed-in method to create a mixture.

        It calls :meth:`get_components`, :meth:`get_phase_configuration` and
        :meth:`set_components_in_phases` in that order and creates the instance
        :attr:`fluid_mixture`

        Raises:
            CompositeModellingError: If no component or no phase was modelled.

        """

        components = self.get_components()
        # TODO other models can be abstracted to use this class, which implement
        # simple constitutive laws without usage of chemistry -> Code unification
        # for incompressible single-phase flow for example, no need to define phases or
        # components
        assert len(components) > 0, "Need at least one component."
        phase_configurations = self.get_phase_configuration(components)
        assert len(phase_configurations) > 0, "Need at least one phase."

        phases: list[Phase] = list()
        for config in phase_configurations:
            eos, type_, name = config
            phases.append(Phase(eos, type_, name))

        self.set_components_in_phases(components, phases)

        self.fluid_mixture = Mixture(components, phases)

    def get_components(self) -> Sequence[Component]:
        """Method to return a list of modelled components."""
        raise CompositeModellingError("Call to mixin method. No components defined.")

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

        Returns:
            A sequence of 3-tuples containing

            1. An instance of an EoS (or some other class providing means to compute
               phase properties)
            2. The phase type (integer denoting the phsycial state like liquid or gas)
            3. A name for the phase.

            Each tuple will be used to create a phase in the fluid mixture.
            For more information on the required return values see
            :class:`~porepy.composite.base.Phase`.

        """
        raise CompositeModellingError("Call to mixin method. No phases configured.")

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

    def assign_thermodynamic_properties_to_mixture(self) -> None:
        """A method to create various thermodynamic properties of phases in AD form.

        After that, it assignes properties to the ``fluid_mixture`` based on phase
        properties.

        Will be called by the solution strategy after all variables have been created.

        Phases get the following properties assigned:

        - :meth:`phase_density` to :attr:`~porepy.composite.base.Phase.density`
        - :meth:`phase_volume` to :attr:`~porepy.composite.base.Phase.volume`
        - :meth:`phase_enthalpy` to :attr:`~porepy.composite.base.Phase.enthalpy`
        - :meth:`phase_viscosity` to :attr:`~porepy.composite.base.Phase.viscosity`
        - :meth:`phase_conductivity` to
          :attr:`~porepy.composite.base.Phase.conductivity`
        - :meth:`fugacity_coefficient` to
          :attr:`~porepy.composite.base.Phase.fugacity_of`
          for each component in respective phase.
          This is only done for mixtures with a defined equilibrium type

        The :attr:`fluid_mixture` gets following properties assigned:

        - :meth:`fluid_density` to :attr:`~porepy.composite.base.Mixture.density`
        - :meth:`fluid_volume` to :attr:`~porepy.composite.base.Mixture.volume`
        - :meth:`fluid_enthalpy` to :attr:`~porepy.composite.base.Mixture.enthalpy`

        Customization is possible in respective methods by inheritance.

        """
        assert hasattr(self, "fluid_mixture"), "Mixture not set."

        for phase in self.fluid_mixture.phases:
            phase.density = self.phase_density(phase)
            phase.volume = self.phase_volume(phase)
            phase.enthalpy = self.phase_enthalpy(phase)
            phase.viscosity = self.phase_viscosity(phase)
            phase.conductivity = self.phase_conductivity(phase)
            phase.fugacity_of = dict()
            for comp in phase:
                phase.fugacity_of[comp] = self.fugacity_coefficient(comp, phase)

        self.fluid_mixture.density = self.fluid_density
        self.fluid_mixture.volume = self.fluid_volume
        self.fluid_mixture.enthalpy = self.fluid_enthalpy

    def fluid_density(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Implements the mixture density function which returns the sum of phase
        densities weighed with phase saturations."""

        op = pp.ad.sum_operator_list(
            [
                phase.saturation(domains) * phase.density(domains)
                for phase in self.fluid_mixture.phases
            ],
            "fluid-mixture-density",
        )
        return op

    def fluid_enthalpy(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Implements the mixture enthalpy function which returns the sum of
        phase enthalpies weighed with phase fractions.

        Raises:
            CompositeModellingError: If :attr:`equilibrium_type` is None, and hence no
                molar phase fractions are introduced by this class.
                The consistent definition of the specific molar enthalpy of a fluid
                mixture always depends on molar phase fractions.

        """

        if self.equilibrium_type is None:
            raise CompositeModellingError(
                "Attempting to define the (specific) fluid mixture enthalpy as sum of"
                + " phase enthalpies weighed with molar phase fractions, even though"
                + " no equilibrium conditions defined. Per default, molar phase"
                + " fractions are only created when an equilibrium formulation is"
                + " used in the model."
            )

        op = pp.ad.sum_operator_list(
            [
                phase.fraction(domains) * phase.enthalpy(domains)
                for phase in self.fluid_mixture.phases
            ],
            "fluid-mixture-enthalpy",
        )
        return op

    def fluid_volume(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Implements the mixture volume function which returns the reciprocal of
        whatever :meth:`get_mixture_volume` returns."""
        op = self.fluid_density(domains) ** pp.ad.Scalar(-1)
        op.set_name("fluid-mixture-volume")
        return op

    def dependencies_of_phase_properties(
        self, phase: Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]:
        """Method to define the signature of phase properties, which are secondary
        expressions.

        In the case of a local equilibrium formulation, the properties
        depend on pressure, temperature and extended fractions.

        Otherwise they depend on pressure, temperature and independent overall
        fractions.

        Note:
            Strictly speaking, the properties should depend on partial fractions
            in the equilibrium formulation.

            But since they are dependent operators and
            :class:`~porepy.composite.composite_utils.SecondaryExpression`
            requires independent variables as dependencies, this little `hack` is
            performed here and the derivatives w.r.t. the fractions are expanded
            in the solution strategy of the CF framework, to account for the partial
            fractions being normalizations of extended fractions.

        """
        dependencies = [self.pressure, self.temperature]
        if self.equilibrium_type is not None:
            dependencies += [phase.fraction_of[component] for component in phase]
        else:
            if self.eliminate_reference_component:
                independent_overall_fractions = [
                    comp.fraction
                    for comp in self.fluid_mixture.components
                    if comp != self.fluid_mixture.reference_component
                ]
            else:
                independent_overall_fractions = [
                    comp.fraction for comp in self.fluid_mixture.components
                ]

            dependencies += independent_overall_fractions
        return dependencies

    def phase_density(
        self, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """This base method returns the specific (molar) density of a ``phase`` as a
        :class:`~porepy.composite.composite_utils.SecondaryExpression` on all subdomains
        and boundaries.

        It is an object which is populated by the solution strategy, depending on how
        the equilibrium conditions and constitutive laws are implemented.

        The phase density (like all thermodynamic properties) is a property depending on
        pressure, temperature and some fractions.

        """
        return SecondaryExpression(
            f"phase_density_{phase.name}",
            self.mdg,
            self.dependencies_of_phase_properties(phase),
            time_step_depth=len(self.time_step_indices),
            iterate_depth=len(self.iterate_indices),
        )

    def phase_volume(
        self, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Implements the volume as the reciprocal of whatever is assigned to a phase
        density."""
        return SecondaryExpression(
            f"phase_volume_{phase.name}",
            self.mdg,
            self.dependencies_of_phase_properties(phase),
            time_step_depth=len(self.time_step_indices),
            iterate_depth=len(self.iterate_indices),
        )

    def phase_enthalpy(
        self, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Analogous to :meth:`phase_density`, creating a new expression for the
        specific (molar) enthalpy of a ``phase``."""
        return SecondaryExpression(
            f"phase_enthalpy_{phase.name}",
            self.mdg,
            self.dependencies_of_phase_properties(phase),
            time_step_depth=len(self.time_step_indices),
            iterate_depth=len(self.iterate_indices),
        )

    def phase_viscosity(
        self, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Analogous to :meth:`phase_density`, creating a new expression for the
        dynamic (molar) viscosity of a ``phase``.

        Note:
            The viscosity has no time step depth, because it does not appear in the
            accumulation term.

        """
        return SecondaryExpression(
            f"phase_viscosity_{phase.name}",
            self.mdg,
            self.dependencies_of_phase_properties(phase),
            time_step_depth=0,
            iterate_depth=len(self.iterate_indices),
        )

    def phase_conductivity(
        self, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Analogous to :meth:`phase_density`, creating a new expression for the
        thermal conductivity of a ``phase``.

        Note:
            The conductivity has no time step depth, because it does not appear in the
            accumulation term.

        """
        return SecondaryExpression(
            f"phase_conductivity_{phase.name}",
            self.mdg,
            self.dependencies_of_phase_properties(phase),
            time_step_depth=0,
            iterate_depth=len(self.iterate_indices),
        )

    def fugacity_coefficient(
        self, component: Component, phase: Phase
    ) -> Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]:
        """Analogous to :meth:`phase_density`, creating a new expression for the
        fugacity coefficient of a ``component`` in a ``phase``.

        Note:
            The fugacity coefficients have no time step depth, because they are used
            only locally.

            Note also, that fugacity coefficient appear only in the local equilibrium
            equation or other chemistry-related models, but not in flow and transport.

        """
        return SecondaryExpression(
            f"fugacity_of_{component.name}_in_{phase.name}",
            self.mdg,
            self.dependencies_of_phase_properties(phase),
            time_step_depth=0,
            iterate_depth=len(self.iterate_indices),
        )
