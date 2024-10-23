"""A module containing mixins for defining fluid mixtures and relatd variables in a
PorePy model.

While the mixins operate on some base assumptions, they are highly customizable by
inheritance and the user is encouraged to read up on them.

Important:
    The framework does not support the variable switiching approach.

    Variables are persistent and the user must be familiar with the DOFs implemented
    in the class :class:`_MixtureDOFHandler` (whose methods are not supposed to be
    overwritten).

    Once the :meth:`~porepy.compositional.base.Fluid.reference_phase_index` and
    :meth:`~porepy.compositional.base.Fluid.reference_component_index` of a the
    fluid, :meth:`~porepy.compositional.base.Phase.reference_component_index` for each
    phase are set, the mixins create the mixture and associated variables and it is not
    possible to change fraction and saturation variables in the course of a simulation.

"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS as symbols
from ._core import PhysicalState
from .base import AbstractEoS, Component, Compound, Fluid, Phase
from .states import FluidProperties, PhaseProperties
from .utils import CompositionalModellingError

__all__ = [
    "CompositionalVariables",
    "FluidMixin",
]

DomainFunctionType = pp.DomainFunctionType
ExtendedDomainFunctionType = pp.ExtendedDomainFunctionType


def _no_property_function(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
    """ "Helper function to define missing phase property functions."""
    raise NotImplementedError("Missing mixed-in constitutive laws.")


def _get_surrogate_factory_as_property(
    name: str,
    mdg: pp.MixedDimensionalGrid,
    dependencies: Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]],
) -> pp.ad.SurrogateFactory:
    """Helper function to get a surrogate factory as a phase property, by providing
    the name, the mD grid and the list of dependencies (like pressure, temperature)."""
    return pp.ad.SurrogateFactory(
        name,
        mdg,
        dependencies,
    )


class _MixtureDOFHandler:
    """A class to help resolve the independent fractional variables of an arbitrary
    mixture, and respectivly the DOFs.

    .. rubric:: Assumptions and unity constraints

    1. Reference phase and component can be eliminated.

       - Phase fraction, saturation and component overall fractions of the reference
         instance can be expressed by unity of fractions.

    2. Components do not have to be in all phases. E.g., solid phase models usually
       contain only 1 mineral/component, or ions in liquid phase do not have to
       evaporate into a gas phase.
    3. Partial fractions of components in phases have to fulfill the unity constraint.
       A phase's reference component and its partial fraction are eliminated as DOFs.
    4. While the third point holds especially for partial (physical) fractions, it does
       not hold necessarily for extended fraction in the unified setting. Vanished
       phases with only 1 component still have extended fractions (weak unity).
    5. Tracer fractions of active tracers in compounds are always unknown (transport).

    .. rubric:: Resolution

    The logic of whether a fraction is an independent variable or not is implemented in
    various ``has_independent_*`` methods:

    1. Independent overall :attr:`~porepy.compositional.base.Component.fraction` of
       components and massic/molar :attr:`~porepy.compositional.base.Phase.fraction` of
       phases are determined by :meth:`has_independent_fraction`.
    2. Independent :attr:`~porepy.compositional.base.Phase.saturation` variables are
       determined by :meth:`has_independent_saturation`.
    3. Independent :attr:`~porepy.compositional.base.Compound.tracer_fraction_of` are
       determined by :meth:`has_independent_tracer_fraction`.
    4. Independent :attr:`~porepy.compositional.base.Phase.partial_fraction_of`
       are determined by :meth:`has_independent_partial_fraction` for all components
       and phases in the mixture, independent of whether an equilibrium is defined or
       not.
    5. Independent :attr:`~porepy.compositional.base.Phase.extended_fraction_of`
       are determined by :meth:`has_independent_extended_fraction` for all components
       and phases in the mixture, independent of whether an equilibrium is defined or
       not.

    Notes:
        The logic is guided by the

        - :meth:`~porepy.compositional.base.Fluid.num_components` in the
          :attr:`fluid`
        - :meth:`~porepy.compositional.base.Fluid.num_phases` in the
          :attr:`fluid`
        - :meth:`~porepy.compositional.base.Fluid.reference_component` and
          :meth:`~porepy.compositional.base.Fluid.reference_phase` in the
          :attr:`fluid`
        - :meth:`~porepy.compositional.base.Phase.reference_component` in each phase
          in the :attr:`fluid`

        It is also guided by the flags ``'eliminate_reference_component'`` and
        ``'eliminate_reference_phase'``, which can be set in the model's :attr:`params`.

        Finally, the ``'equilibrium_type'`` set in the model's :attr:`params` is used
        to determine the indepency of partial and extended fractions of components
        in phases.

    """

    fluid: Fluid
    """See :class:`FluidMixin`."""

    equation_system: pp.ad.EquationSystem
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    params: dict
    """See the solutions strategy mixin."""

    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """See :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`."""

    # Logic methods determining existence of DOFs

    @property
    def _is_ref_phase_eliminated(self) -> bool:
        """Helper property to access the model parameters and check if the
        reference phase is eliminated. Default value is True."""
        return bool(self.params.get("eliminate_reference_phase", True))

    @property
    def _is_ref_comp_eliminated(self) -> bool:
        """Helper property to access the model parameters and check if the
        reference component is eliminated. Default value is True."""
        return bool(self.params.get("eliminate_reference_component", True))

    @property
    def _has_equilibrium(self) -> bool:
        """Helper method to access the model parameters and check if the
        equilibrium type is defined. Defaults to False."""

        equilibrium_type = self.params.get("equilibrium_type", None)

        if equilibrium_type is None:
            return False
        else:
            return True

    @property
    def _has_unified_equilibrium(self) -> bool:
        """Helper method to access the model parameters and check if the
        equilibrium type is defined and if it is unified."""
        if self._has_equilibrium:
            # NOTE _has_equilibrium already checks that the value is not none.
            if "unified" in str(self.params["equilibrium_type"]):
                return True
            else:
                return False
        else:
            return False

    def has_independent_fraction(self, instance: Phase | Component) -> bool:
        """Checks whether the ``instance`` has an independent variable for the
        fraction of total mass associated with it (
        :attr:`~porepy.compositional.base.Component.fraction` of a component,
        :attr:`~porepy.compositional.base.Phase.fraction` of a phase
        )

        Works the same for both components and phases.

        Parameters:
            instance: A phase or a component in the :attr:`fluid`.

        Raises:
            ValueError: If the ``instance`` is not in the fluid mixture.
            TypeError: If ``instance`` is neither a phase nor a component.

        Returns:
            False, if there is only 1 instance (of phases or component) in the fluid
            mixture, or it is the reference instance and it was eliminated.
            Otherwise it returns True.

        """
        instances: list[Phase | Component]
        if isinstance(instance, Phase):
            instances = list(self.fluid.phases)
            if instance not in instances:
                raise ValueError(f"Phase {instance} not in fluid mixture.")
            idx = instances.index(instance)
            ref_idx = self.fluid.reference_phase_index
            num_instances = self.fluid.num_phases
            eliminated = self._is_ref_phase_eliminated
        elif isinstance(instance, Component):
            instances = list(self.fluid.components)
            if instance not in instances:
                raise ValueError(f"Component {instance} not in fluid mixture.")
            idx = instances.index(instance)
            ref_idx = self.fluid.reference_component_index
            num_instances = self.fluid.num_components
            eliminated = self._is_ref_comp_eliminated
        else:
            raise TypeError(
                f"Unknown type {type(instance)}. Expecting phase or component."
            )

        if (idx == ref_idx and eliminated) or num_instances == 1:
            return False
        else:
            return True

    def has_independent_saturation(self, phase: Phase) -> bool:
        """Checks if the :attr:`~porepy.compositional.base.Phase.saturation` of the
        ``phase`` is an independent variable.

        Note:
            The logic is analogous to :meth:`has_independent_fraction` for phases.

        Paremters:
            phase: A phase in the :attr:`fluid`.

        Returns:
            False, if there is only 1 phase, or the phase is the reference phase and
            it was eliminated.
            Otherwise it returns True.

        """
        return self.has_independent_fraction(phase)

    def has_independent_tracer_fraction(
        self, tracer: pp.FluidConstants, compound: Compound
    ) -> bool:
        """Checks if the :attr:`~porepy.compositional.base.Compound.tracer_fraction_of`
        a ``tracer`` in the ``compound`` is an independent variable.

        Paramters:
            tracer: An active tracer in one of the compounds in the
                :attr:`fluid`
            compound: A component/compound in the :attr:`fluid`'s
                :attr:`~porepy.compositional.base.Fluid.components`.

        Raises:
            ValueError: If the ``compound`` is not in the :attr:`fluid`.

        Returns:
            True, if the ``tracer`` is in the compound``, False otherwise.

        """
        if compound not in list(self.fluid.components):
            raise ValueError(f"Compound {compound} not in fluid mixture.")
        if tracer in compound.active_tracers:
            return True
        else:
            return False

    def has_independent_partial_fraction(
        self, component: Component, phase: Phase
    ) -> bool:
        """Checks whether the
        :attr:`~porepy.compositional.base.Phase.partial_fraction_of` a ``component`` in
        a ``phase`` is an independent variable or not.

        If the ``'equilibrium_type'`` in :attr:`params` indicates that any *unified*
        equilibrium is included in the model, i.e.
        ``'unified' in params['equilibrium_type'] == True``, partial fractions are
        always dependent, since they are obtained by normalization of extended
        fractions.

        If there is no equilibrium or a non-unified equilibrium, partial fractions can
        be independent variables.

        If there is only 1 phase, the partial fractions are replaced by the overall
        fractions and hence not independent. Since all components must be at least in
        one phase, the system is closed.

        If there is only 1 component in that phase (or 1 component in the whole mixture)
        the partial fraction is always 1, hence not an independent variable.

        If there are multiple components in the phase, the partial fraction is an
        independent variable. Except if it is the fraction of the phase's reference
        component. Then it is assumed to be eliminated by unity of fractions, hence
        a dependent quantity.

        Note:
            The flag ``'eliminate_reference_component'`` is only relevant for the
            overall :attr:`~porepy.compositional.base.Component.fraction` of a
            component, and hence for the number of transport equations.
            The :attr:`~porepy.compositional.base.Phase.partial_fraction_of` a the
            reference component of a phase is always eliminated by unity, since
            it is a local quantity. There are no benefits or alternative formulations of
            making it a genuine variable. It would only increase the system size
            unnecessarily.

        Paramters:
            component: Any component in the :attr:`fluid`.
            phase: Any phase in the :attr:`fluid`.

        Raises:
            ValueError: If the ``phase`` or the ``component are not in the
                :attr:`fluid`.

        Returns:
            True, if the partial fraction is an independent variable according to above
            logic. False otherwise.

        """
        if phase not in self.fluid.phases:
            raise ValueError(f"Phase {phase} not in fluid mixture.")
        if component not in self.fluid.components:
            raise ValueError(f"Component {component} not in fluid mixture.")

        if self._has_unified_equilibrium:
            return False
        else:
            if component not in phase:
                return False
            # By logic, component is now in phase.
            # NOTE The FluidMixin does not allow for any component not being in
            # any phase.
            # If there is only 1 phase, the partial fractions are replaced by the
            # overall component fraction.
            # If there is only 1 component in that phase, the partial fraction is 1.
            # Both cases lead to the partial fraction not being an independent variable.
            if self.fluid.num_phases == 1 or phase.num_components == 1:
                return False

            # Now, the component can either be the reference component of the phase
            # or not. If it is the reference component, its partial fraction is always
            # eliminated by unity. Otherwise it has an independent partial fraction
            # NOTE if there is only 1 component and multiple phases, it is automatically
            # the reference component of each phase.
            if component == phase.reference_component:
                return False
            else:
                return True

    def has_independent_extended_fraction(
        self, component: Component, phase: Phase
    ) -> bool:
        """Checks whether the
        :attr:`~porepy.compositional.base.Phase.extended_fraction_of` a ``component`` in
        a ``phase`` is an independent variable or not.

        Extended fractions are only used if there is a unified equilibrium defintion,
        i.e. ``'unified' in params['equilibrium_type'] == True``.

        In that case, if the ``component`` is modelled in the ``phase``, the respective
        extended fraction is always an independent variable.
        Since for vanished phases the extended fractions do not necessarily fulfill the
        unity constraint, no extended fraction can be eliminated.

        Paramters:
            component: Any component in the :attr:`fluid`.
            phase: Any phase in the :attr:`fluid`.

        Raises:
            ValueError: If the ``phase`` or the ``component are not in the
                :attr:`fluid`.
            CompositionalModellingError: If the ``component`` is not in the phase.
                The unified setting expects all components to be modelled in all phases.

        Returns:
            True, if the ``'equilibrium_type'`` in :attr:`params` contains
            ``'unified'``. False otherwise.

        """
        if phase not in self.fluid.phases:
            raise ValueError(f"Phase {phase} not in fluid mixture.")
        if component not in self.fluid.components:
            raise ValueError(f"Component {component} not in fluid mixture.")

        if self._has_unified_equilibrium:
            if component not in phase:
                raise CompositionalModellingError(
                    f"Component {component} not in phase {phase}."
                    + " Models with unified equilibrium require all components to"
                    + " be modelled in all phases."
                )
            return True
        else:
            return False

    # Utility methods for DOFs and variables

    def _overall_fraction_variable(self, component: Component) -> str:
        """Returns the name of the fraction variable assigned to ``component``."""
        return f"{symbols['overall_fraction']}_{component.name}"

    def _saturation_variable(self, phase: Phase) -> str:
        """Returns the name of the saturation variable assigned to ``phase``."""
        return f"{symbols['phase_saturation']}_{phase.name}"

    def _tracer_fraction_variable(
        self, tracer: pp.FluidConstants, compound: Compound
    ) -> str:
        """Returns the name of the tracer fraction variable assigned to tracer in a
        compound."""
        return f"{symbols['tracer_fraction']}_{tracer.name}_{compound.name}"

    def _phase_fraction_variable(self, phase: Phase) -> str:
        """Returns the name of the phase fraction variable assigned to ``phase``."""
        return f"{symbols['phase_fraction']}_{phase.name}"

    def _partial_fraction_variable(self, component: Component, phase: Phase) -> str:
        """Returns the name of the (extended or partial) fraction variable of
        ``component`` in ``phase``.

        Note:
            For simplicity we use the same name for the extended fractions, because in
            the case they are used, partial fractions are always dependent operators.

        """
        return f"{symbols['phase_composition']}_{component.name}_{phase.name}"

    def _fraction_factory(self, name: str) -> DomainFunctionType:
        """Factory method to create a callable representing any independent fraction
        with given ``name`` on subdomain or boundary grids."""

        # If the factory is called the first time for a specific variable name,
        # create the variable.
        if name not in set([var.name for var in self.equation_system.variables]):
            self.equation_system.create_variables(
                name=name,
                subdomains=self.equation_system.mdg.subdomains(),
                tags={"si_units": "-"},
            )

        def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            if len(domains) > 0 and all(
                [isinstance(g, pp.BoundaryGrid) for g in domains]
            ):
                return self.create_boundary_operator(
                    name=name, domains=domains  # type: ignore[call-arg]
                )
            # Check that the domains are grids.
            if not all([isinstance(g, pp.Grid) for g in domains]):
                raise ValueError(
                    """Argument 'domains' a mixture of subdomain and boundaries."""
                )
            domains = cast(list[pp.Grid], domains)
            return self.equation_system.md_variable(name, domains)

        return fraction

    @property
    def overall_fraction_variables(self) -> list[str]:
        """Names of independent overall
        :attr:`~porepy.compositional.base.Component.fraction` variables created for this
        model."""
        names: list[str] = []
        for comp in self.fluid.components:
            if self.has_independent_fraction(comp):
                names.append(self._overall_fraction_variable(comp))
        return names

    @property
    def tracer_fraction_variables(self) -> list[str]:
        """Names of independent
        :attr:`~porepy.compositional.base.Compound.tracer_fraction_of` -variables
        created for this model."""
        names: list[str] = []
        compounds = [
            comp for comp in self.fluid.components if isinstance(comp, Compound)
        ]
        for comp in compounds:
            for tracer in comp.active_tracers:
                if self.has_independent_tracer_fraction(tracer, comp):
                    names.append(self._tracer_fraction_variable(tracer, comp))
        return names

    @property
    def phase_fraction_variables(self) -> list[str]:
        """Names of independent phase :attr:`~porepy.compositional.base.Phase.fraction`
        variables created for this model."""
        names: list[str] = []
        if self._has_equilibrium:
            for phase in self.fluid.phases:
                if self.has_independent_fraction(phase):
                    names.append(self._phase_fraction_variable(phase))
        return names

    @property
    def saturation_variables(self) -> list[str]:
        """Names of independent phase
        :attr:`~porepy.compositional.base.Phase.saturation` variables created for this
        model."""
        names: list[str] = []
        for phase in self.fluid.phases:
            if self.has_independent_saturation(phase):
                names.append(self._saturation_variable(phase))
        return names

    @property
    def fraction_in_phase_variables(self) -> list[str]:
        """Names of either partial fractions or extended fraction variables created
        for this model.

        Note that only 1 type of fractions is created, depending on the equilibrium
        settings.

        See Also:
            :attr:`~porepy.compositional.base.Phase.extended_fraction_of`
            :attr:`~porepy.compositional.base.Phase.partial_fraction_of`

        """
        names: list[str] = []
        for phase in self.fluid.phases:
            for comp in phase:
                append = False
                if self._has_unified_equilibrium:
                    if self.has_independent_extended_fraction(comp, phase):
                        append = True
                elif self.has_independent_partial_fraction(comp, phase):
                    append = True
                if append:
                    names.append(self._partial_fraction_variable(comp, phase))

        return names


class CompositionalVariables(pp.VariableMixin, _MixtureDOFHandler):
    """Mixin class for models with mixtures which defines the respective fractional
    unknowns.

    Fractional variables are relevant for the equilibrium formulation, as well as
    for compositional flow.

    Various methods can be overwritten to introduce constitutive laws instead of
    unknowns.

    Important:
        For compositional flow without a local equilibrium formulation, the flow and
        transport formulation does not require phase fractions or extended
        fractions of components in phases.
        Phases have only saturations as phase related variables, and instead of extended
        fractions, the (physical) partial fractions are independent variables.

        When setting up models, keep in mind that you need to close the flow and
        transport model with local equations, eliminating saturations and partial
        fractions by some function depending on the primary flow and transport variables
        (pressure, temperature, overall fractions or pressure, enthalpy, and overall
        fractions).

    """

    def fractional_state_from_vector(
        self,
        subdomains: Sequence[pp.Grid],
        state: Optional[np.ndarray] = None,
    ) -> FluidProperties:
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

                See :meth:`~porepy.numerics.ad.operators.Operator.value`.

        Returns:
            A partially filled fluid state data structure containing the above
            fractional values.

        """

        z = np.array(
            [
                component.fraction(subdomains).value(self.equation_system, state)
                for component in self.fluid.components
            ]
        )

        y = np.array(
            [
                phase.fraction(subdomains).value(self.equation_system, state)
                for phase in self.fluid.phases
            ]
        )

        sat = np.array(
            [
                phase.saturation(subdomains).value(self.equation_system, state)
                for phase in self.fluid.phases
            ]
        )

        x = [
            np.array(
                [
                    (
                        phase.extended_fraction_of[component](subdomains).value(
                            self.equation_system, state
                        )
                        if self._has_unified_equilibrium
                        else phase.partial_fraction_of[component](subdomains).value(
                            self.equation_system, state
                        )
                    )
                    for component in phase
                ]
            )
            for phase in self.fluid.phases
        ]

        return FluidProperties(
            z=z,
            y=y,
            sat=sat,
            phases=[PhaseProperties(x=x_) for x_ in x],
        )

    def create_variables(self) -> None:
        """Creates the sets of required variables for a fluid mixture.

        1. :meth:`overall_fraction` is called to assign
           :attr:`~porepy.compositional.base.Component.fraction` to components.
        2. :meth:`tracer_fraction` is called to assign
           :attr:`~porepy.compositional.base.Compound.tracer_fraction_of` for each
           tracer in a compound.
        3. :meth:`saturation` is called to assign
           :attr:`~porepy.compositional.base.Phase.saturation` to phases.
        4. :attr:`~porepy.compositional.base.Phase.fraction` to phases by calling
           :meth:`phase_fraction'
        5. :attr:`~porepy.compositional.base.Phase.extended_fraction_of` for each phase
           and component by calling :meth:`extended_fraction`
        6. :attr:`~porepy.compositional.base.Phase.partial_fraction_of` for each phase
           and component by calling :meth:`partial_fraction`

        Note however, that dependent on the mixture and model configuration, the
        objects and callables created here are both independent variables and dependent
        expressions (regular AD operators).

        """
        if not hasattr(self, "fluid"):
            raise CompositionalModellingError(
                "Cannot create fluid mixture variables before defining a fluid mixture."
            )

        # NOTE: The creation of variables seems repetative (it is), but it is done this
        # way to preserve a certain order (component-wise, phase-wise and familiy-wise
        # for each family of fractions)

        # Creation of feed fractions
        for component in self.fluid.components:
            component.fraction = self.overall_fraction(component)

        # Creation of tracer fractions for compounds
        for component in self.fluid.components:
            if isinstance(component, Compound):
                component.tracer_fraction_of = {}
                for tracer in component.active_tracers:
                    component.tracer_fraction_of[tracer] = self.tracer_fraction(
                        tracer, component
                    )

        # NOTE all variables associated with transport of mass are now created.
        # Below variables are of local nature.

        # Creation of saturation variables
        for phase in self.fluid.phases:
            phase.saturation = self.saturation(phase)

        # Creation of phase fraction variables
        for phase in self.fluid.phases:
            phase.fraction = self.phase_fraction(phase)

        # Creation of extended fractions
        for phase in self.fluid.phases:
            phase.extended_fraction_of = dict()
            # NOTE iterate over components in phase, not all components to avoid
            # conflicts with non-unified set-ups.
            # The check of whether all phases have all components must be done elsewhere
            for comp in phase:
                phase.extended_fraction_of[comp] = self.extended_fraction(comp, phase)

        # Creation of partial fractions
        for phase in self.fluid.phases:
            phase.partial_fraction_of = dict()
            for comp in phase:
                phase.partial_fraction_of[comp] = self.partial_fraction(comp, phase)

    def overall_fraction(
        self,
        component: Component,
    ) -> DomainFunctionType:
        """Getter method to create a callable representing the overall fraction of a
        component on a list of subdomains or boundaries.

        Cases where the overall fraction is not an independent variable:

        1. If there is only 1 component, the fraction is constant 1.
        2. If the reference component fraction was eliminated, is is a dependent
           operator.

        Parameters:
            component: A component in the fluid mixture.

        Returns:
            A callable which returns the feed fraction for a given set of domains.

        """

        fraction: DomainFunctionType

        # If only 1 component, the fraction is always 1
        if self.fluid.num_components == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single_feed_fraction")

        # NOTE if the reference component fraction is independent, below elif-clause
        # will be executed, instead of the next one
        elif self.has_independent_fraction(component):
            fraction = self._fraction_factory(
                self._overall_fraction_variable(component)
            )
        elif component == self.fluid.reference_component:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                z_R = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
                    [
                        comp.fraction(domains)
                        for comp in self.fluid.components
                        if comp != self.fluid.reference_component
                    ]
                )
                z_R.set_name("reference_feed_fraction_by_unity")
                return z_R

        # Should never happen
        else:
            raise NotImplementedError("Missing logic for overall fractions.")

        return fraction

    def tracer_fraction(
        self, tracer: pp.FluidConstants, compound: Compound
    ) -> DomainFunctionType:
        """Getter method to create a callable representing the tracer fraction of an
        active tracer in a compound, on a list of subdomains or boundaries.

        The base method creates tracer fractions as an independend variables
        (transportable), after asserting the tracer is indeed in that compound.

        Parameters:
            tracer: An active tracer in the fluid mixture.
            compound: A compound in the fluid mixture.

        Returns:
            A callable which returns the tracer fraction for a given set of domains.

        """
        assert (
            tracer in compound.active_tracers
        ), f"Solute {tracer.name} not in compound {compound.name}"

        if self.has_independent_tracer_fraction(tracer, compound):
            fraction = self._fraction_factory(
                self._tracer_fraction_variable(tracer, compound)
            )
        else:
            raise NotImplementedError("Missing logic for tracer fraction.")

        return fraction

    def saturation(self, phase: Phase) -> DomainFunctionType:
        """Analogous to :meth:`overall_fraction` but for phase saturations.

        Cases where the saturation is not an independent variable:

        1. If there is only 1 phase, the saturation is constant 1.
        2. If the reference phase was eliminated, is is a dependent operator.

        Parameters:
            phase: A phase in the fluid mixture.

        Returns:
            A callable which returns the saturation for a given set of domains.

        """

        saturation: DomainFunctionType

        # If only 1 phase, the saturation is always 1
        if self.fluid.num_phases == 1:

            def saturation(subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single_phase_saturation")

        # NOTE if the reference phase is independent, below elif-clause will be
        # executed, instead of the next one.
        elif self.has_independent_saturation(phase):
            saturation = self._fraction_factory(self._saturation_variable(phase))
        # if reference component, eliminate by unity
        elif phase == self.fluid.reference_phase:

            def saturation(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                s_R = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
                    [
                        phase.saturation(domains)
                        for phase in self.fluid.phases
                        if phase != self.fluid.reference_phase
                    ]
                )
                s_R.set_name("reference_phase_saturation_by_unity")
                return s_R

        # Should never happen
        else:
            raise NotImplementedError("Missing logic for saturations.")

        return saturation

    def phase_fraction(self, phase: Phase) -> DomainFunctionType:
        """Analogous to :meth:`saturation` but for phase molar fractions."""

        fraction: DomainFunctionType
        # Code is completely analogous to method saturation, except that we raise a
        # modelling error if no equilibrium is defined. phase fractions can completely
        # be omitted in that case
        if self.fluid.num_phases == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single_phase_fraction")

        elif not self._has_equilibrium:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                raise CompositionalModellingError(
                    "Phase fractions are not defined in model without equilibrium."
                    + " A re-formulation using saturations is required."
                )

        elif self.has_independent_fraction(phase):
            fraction = self._fraction_factory(self._phase_fraction_variable(phase))
        elif phase == self.fluid.reference_phase:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                y_R = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
                    [
                        phase.fraction(domains)
                        for phase in self.fluid.phases
                        if phase != self.fluid.reference_phase
                    ]
                )
                y_R.set_name("reference_phase_fraction_by_unity")
                return y_R

        else:
            raise NotImplementedError("Missing logic for phase fractions.")

        return fraction

    def extended_fraction(
        self, component: Component, phase: Phase
    ) -> DomainFunctionType:
        """Getter method to create a callable representing the extended fraction of a
        component in a phase, on a list of subdomains or boundaries.

        Cases where the extended fractions are not independent variables:

        1. If no unified equilibrium is defined, accessing the extended fractions will
           raise an :class:`~porepy.compositional.utils.CompositionalModellingError`.
        2. If there is only 1 phase **and** 1 component, the extended fraction is always
           1 since there the 1 phase cannot vanish.

        Note:
            Even if a phase has only 1 component in it, if it vanishes the extended
            fraction is not necessarily 1, hence an unknown.

        Parameters:
            component: A componend in the fluid mixture.
            phase: A phase in the fluid mixture.

        Returns:
            A callable which returns the extended fraction for a given set of domains.

        """
        assert (
            component in phase
        ), f"Component {component.name} not in phase {phase.name}"

        fraction: DomainFunctionType

        # Add this for completeness reasons, s.t. the phase has the respective
        # attribute. But raise an error if the user tries to access the fraction.

        # NOTE Extended fractions are in general always unknowns, even in
        # 1 component, multiphase case (they are some value below 1 if a phase vanishes)
        # Only in the case with 1 component and 1 phase, the extendeded fraction is
        # also a scalar 1, since the 1 modelled phase cannot vanish.
        if self.fluid.num_components == self.fluid.num_phases == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(
                    1.0, "single_component_single_phase_extended_fraction"
                )

        # If no unified equilibrium, calling the extended fractions will raise an error
        elif not self._has_unified_equilibrium:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                raise CompositionalModellingError(
                    "Attempting to access extended fractions in set-up where unified"
                    + " equilibrium was not defined."
                )

        elif self.has_independent_extended_fraction(component, phase):
            fraction = self._fraction_factory(
                self._partial_fraction_variable(component, phase)
            )
        else:
            raise NotImplementedError("Missing logic for extended fractions.")

        return fraction

    def partial_fraction(
        self, component: Component, phase: Phase
    ) -> DomainFunctionType:
        """Getter method to create a callable representing the partial fraction of a
        component in a phase, on a list of subdomains or boundaries.

        Cases where the partial fractions are not independent variables:

        1. If there is only 1 component in the phase, the partial fraction is always 1.
        2. If there is only 1 phase, the partial fractions are equal to the overall
           fraction, hence dependent.
        3. If a *unified* equilibrium is defined, partial fractions are always dependent
           operators, obtained by normalization of extended fractions.
        4. If it is the fraction of a phase's reference component, it is eliminated by
           unity.

        Parameters:
            component: A componend in the fluid mixture.
            phase: A phase in the fluid mixture.

        Returns:
            A callable which returns the extended fraction for a given set of domains.

        """
        assert (
            component in phase
        ), f"Component {component.name} not in phase {phase.name}"

        fraction: DomainFunctionType

        # Case only 1 component in phase: partial fraction is always 1
        if phase.num_components == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(
                    1.0, f"single_partial_fraction_{component.name}_{phase.name}"
                )

        # Case only 1 phase in entire mixture, partial fractions are equal to overall
        # fractions
        elif self.fluid.num_phases == 1:
            # Mypy complains that above the argument of fraction is explicitly
            # stated as 'domains', while extended_fraction returns no information
            # on how the argument is called.
            # But both are (pp.SubdomainOrBoundaries) -> pp.ad.Operator
            fraction = component.fraction  # type:ignore[assignment]
        # Case reference component: Partial fraction of a phase's reference component
        # is always expressed by unity of partial fractions in that phase
        elif component == phase.reference_component:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:

                x_r = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
                    [
                        phase.partial_fraction_of[comp](domains)
                        for comp in phase
                        if comp != phase.reference_component
                    ]
                )
                x_r.set_name(
                    f"reference_partial_fraction_by_unity_in_phase_{phase.name}"
                )
                return x_r

        # Case of unified equilibrium: Partial fractions are obtained by normalization
        elif self._has_unified_equilibrium:
            # NOTE the fraction of the phase's reference component is covered above
            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                xn = phase.extended_fraction_of[component](
                    domains
                ) / pp.ad.sum_operator_list(
                    [phase.extended_fraction_of[comp](domains) for comp in phase]
                )
                xn.set_name(
                    "normalized_"
                    + f"{self._partial_fraction_variable(component, phase)}"
                )
                return xn

        # The general case of multi-component, multi-phase mixtures with or without
        # non-unified equilibrium.
        # Partial fractions are independent, except for the reference component in that
        # phase, which is eliminated by unity above
        elif self.has_independent_partial_fraction(component, phase):
            fraction = self._fraction_factory(
                self._partial_fraction_variable(component, phase)
            )
        else:
            raise NotImplementedError("Missing logic for partial fractions.")

        return fraction


class FluidMixin:
    """Mixin class for introducing a general fluid (mixture) into a PorePy model and
    providing it as an attribute :attr:`fluid`.

    Fluid properties are by definition expressed through respective phase properties,
    which can be overriden here as part of the constitutive modelling.

    The following methods provide expressions for them:

    - :meth:`density_of_phase`
    - :meth:`specific_volume_of_phase`
    - :meth:`specific_enthalpy_of_phase`
    - :meth:`viscosity_of_phase`
    - :meth:`thermal_conductivity_of_phase`
    - :meth:`fugacity_coefficient`

    There is no need to modify the :attr:`fluid` itself.

    Important:
        Solution strategies must follow a certain order during the set up
        (`prepare_simulation'):

        1. The fluid must be created (phases and components)
        2. The variables must be created (depending on present phases and components)
        3. The fluid properties are assigned (since they in general depend on variables)

    The base class provides phase properties as general
    :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory`.
    To use heuristic laws, override above mentioned methods via mixins.

    The base class also provides a default set-up in form of a 1-phase, 1-component
    fluid, based on fluid constants found in ``params``.
    To modify the default set-up, provide overrides for

    - :meth:`get_components`
    - :meth:`get_phase_configuration`

    To define the dependency of phase properties in terms of variables, see

    - :meth:`dependencies_of_phase_properties`

    For a more complex set-up involving phases with *different* components, see

    - :meth:`set_components_in_phases`

    """

    fluid: Fluid
    """The fluid mixture set by this class during :meth:`create_fluid`."""

    mdg: pp.MixedDimensionalGrid
    """See :class:`~porepy.models.geometry.ModelGeometry`."""
    params: dict
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""

    pressure: DomainFunctionType
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`."""
    temperature: DomainFunctionType
    """See :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    _has_unified_equilibrium: bool
    """See :class:`_MixtureDOFHandler`."""
    _has_equilibrium: bool
    """See :class:`_MixtureDOFHandler`."""

    has_independent_partial_fraction: Callable[[Component, Phase], bool]
    """See :class:`_MixtureDOFHandler`."""
    has_independent_extended_fraction: Callable[[Component, Phase], bool]
    """See :class:`_MixtureDOFHandler`."""
    has_independent_fraction: Callable[[Component], bool]
    """See :class:`_MixtureDOFHandler`."""

    def create_fluid(self) -> None:
        """Set-up method to create a fluid mixture.

        It calls

        1. :meth:`get_components`,
        2. :meth:`get_phase_configuration` and
        3. :meth:`set_components_in_phases`,

        and creates the instance :attr:`fluid`.

        """

        components = self.get_components()
        phase_configurations = self.get_phase_configuration(components)

        phases: list[Phase] = []
        for config in phase_configurations:
            eos, type_, name = config
            phases.append(Phase(eos, type_, name))

        self.set_components_in_phases(components, phases)

        self.fluid = Fluid(components, phases)

    def get_components(self) -> list[Component]:
        """Method to return a list of modelled components.

        The default implementation takes the user-provided or default fluid constants found in
        the model ``params`` and returns a single component.

        Override this method via mixin to provide a more complex component context for the
        :attr:`fluid`.

        """
        # Should be available after SolutionStrategy.set_materials()
        fluid_constants: pp.FluidConstants = self.params["_default_fluid_constants"]
        default_component = Component.from_fluid_constants(fluid_constants)
        return [default_component]

    def get_phase_configuration(
        self, components: Sequence[Component]
    ) -> Sequence[tuple[AbstractEoS, PhysicalState, str]]:
        """Method to return a configuration of modelled phases.

        The default implementation returns a liquid-like phase with an abstract EoS instance
        (to be used in the standard set-up with heuristic fluid properties).

        Parameters:
            components: The list of components modelled by :meth:`get_components`.

                Note:
                    The reason why this is passed as an argument is to avoid
                    constructing multiple, possibly expensive EoS instances.
                    The user can use only a single EoS instance for all phases f.e.

        Returns:
            A sequence of 3-tuples containing

            1. An instance of an EoS.
            2. The phase state.
            3. A name for the phase.

            Each tuple will be used to create a phase in the fluid mixture.
            For more information on the required return values see
            :class:`~porepy.compositional.base.Phase`.

        """
        return [(AbstractEoS(components), PhysicalState.liquid, "liquid")]

    def set_components_in_phases(
        self, components: Sequence[Component], phases: Sequence[Phase]
    ) -> None:
        """Method to implement a strategy for which components are added to which phase.

        By default, the unified assumption is applied: All phases contain all components.

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

    def assign_thermodynamic_properties_to_phases(self) -> None:
        """A method to create various thermodynamic properties of phases in AD form.

        Will be called by the solution strategy after all variables have been created.

        Phases get the following properties assigned:

        - :meth:`density_of_phase` to
          :attr:`~porepy.compositional.base.Phase.density`
        - :meth:`specific_volume_of_phase` to
          :attr:`~porepy.compositional.base.Phase.specific_volume`
        - :meth:`specific_enthalpy_of_phase` to
          :attr:`~porepy.compositional.base.Phase.specific_enthalpy`
        - :meth:`viscosity_of_phase` to :attr:`~porepy.compositional.base.Phase.viscosity`
        - :meth:`thermal_conductivity_of_phase` to
          :attr:`~porepy.compositional.base.Phase.thermal_conductivity`
        - :meth:`fugacity_coefficient` to
          :attr:`~porepy.compositional.base.Phase.fugacity_coefficient_of`
          for each component in respective phase.
          This is only done for mixtures with a defined equilibrium type

        Customization is possible in respective methods by mixing-in.

        """
        assert hasattr(self, "fluid"), "Mixture not set."

        for phase in self.fluid.phases:
            phase.density = self.density_of_phase(phase)
            phase.specific_volume = self.specific_volume_of_phase(phase)
            phase.specific_enthalpy = self.specific_enthalpy_of_phase(phase)
            phase.viscosity = self.viscosity_of_phase(phase)
            phase.thermal_conductivity = self.thermal_conductivity_of_phase(phase)
            phase.fugacity_coefficient_of = {}
            for comp in phase:
                phase.fugacity_coefficient_of[comp] = self.fugacity_coefficient(
                    comp, phase
                )

    def dependencies_of_phase_properties(
        self, phase: Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        """Method to define the signature of phase properties, which are dependent
        quantities.

        In the case of a unified equilibrium formulation, the properties
        depend on pressure, temperature and extended fractions.

        In the case of a non-unified equilibrium formulation, the properties are
        dependent on pressure, temperature and partial fractions in that phase.

        Otherwise they are depend on pressure, temperature and **independent** overall
        fractions.

        Important:
            This method must be overwritten in every flow problem which does not rely
            on heuristics. The base method returns an empty list for reasons of compatibility
            with pure mechanics models. Below is an example to implement the logic explained
            above.

        Example:

            .. code-block:: python

                class MyMixin:

                    def dependencies_of_phase_properties(
                        self, phase: Phase
                    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:

                        dependencies = [self.pressure]
                        if self._has_unified_equilibrium:

                            dependencies +=  [self.temperature] + [
                                phase.extended_fraction_of[component]
                                for component in phase
                                if self.has_independent_extended_fraction(component, phase)
                            ]

                        elif self._has_equilibrium:

                            dependencies += [self.temperature] + [
                                phase.partial_fraction_of[component]
                                for component in phase
                                if self.has_independent_partial_fraction(component, phase)
                            ]

                        else:

                            dependencies += [
                                component.fraction
                                for component in self.fluid.components
                                if self.has_independent_fraction(component)
                            ]

                        return dependencies

        """
        return []

    def density_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """This base method returns the density of a ``phase`` as a
        :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory`,
        if :meth:`dependencies_of_phase_properties`. Otherwise it returns an empty function,
        raising an error when called (missing mixin of constitutive laws).

        The phase density (like all thermodynamic properties) is a dependent quantity.

        """
        name = f"phase_{phase.name}_density"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def specific_volume_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """The specific volume of the phase is returned as a function calling the
        the phase density and taking the reciprocal of it."""

        def volume(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            op = phase.density(domains)
            op = op ** pp.ad.Scalar(-1.0)
            op.set_name(f"phase_{phase.name}_specific_volume")
            return op

        return volume

    def specific_enthalpy_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`, but for
        :attr:`~porepy.compositional.base.Phase.specific_enthalpy`."""
        name = f"phase_{phase.name}_specific_enthalpy"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def viscosity_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`,  but for
        :attr:`~porepy.compositional.base.Phase.viscosity`."""
        name = f"phase_{phase.name}_viscosity"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def thermal_conductivity_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`, but for
        :attr:`~porepy.compositional.base.Phase.thermal_conductivity`."""
        name = f"phase_{phase.name}_conductivity"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def fugacity_coefficient(
        self, component: Component, phase: Phase
    ) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`, but for
        :attr:`~porepy.compositional.base.Phase.fugacity_coefficient_of`.

        Note:
            Fugacity coefficient appear only in the local equilibrium
            equation or other chemistry-related models, but not in flow and transport.

        """
        name = f"fugacity_coefficient_{component.name}_in_{phase.name}"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function
