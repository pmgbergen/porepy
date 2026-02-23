"""A module containing mixins for defining fluid mixtures and related variables in a
PorePy model.

While the mixins operate on some base assumptions, they are highly customizable by
inheritance and the user is encouraged to read up on them.

Important:
    The framework does not support the variable switching approach.

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

from typing import List, Tuple, Callable, Sequence, cast, Optional
import numpy as np
import porepy as pp


from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS as symbols
from ._core import PhysicalState
from .base import (
    Component,
    ComponentLike,
    Compound,
    EquationOfState,
    Fluid,
    Phase,
    Solid,
    Element,
    Reaction,
)
from .utils import CompositionalModellingError
from chempy.chemistry import Species
from chempy.util.periodic import symbols as chemical_symbols
import re


__all__ = [
    "get_local_equilibrium_condition",
    "has_unified_equilibrium",
    "CompositionalVariables",
    "FluidMixin",
    "ActivityModels",
    "ReactionRatesKineticArrhenius",
    "ReactionRatesKineticFirstOrder",
    "ModifiedSourceAsWells",
]

DomainFunctionType = pp.DomainFunctionType
ExtendedDomainFunctionType = pp.ExtendedDomainFunctionType


def _no_property_function(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
    """Helper function to define missing phase property functions."""
    raise NotImplementedError("Missing mixed-in constitutive laws.")


def _get_surrogate_factory_as_property(
    name: str,
    mdg: pp.MixedDimensionalGrid,
    dependencies: Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]],
) -> pp.ad.SurrogateFactory:
    """Helper function to get a surrogate factory as a phase property, by providing
    the name, the md-grid and the list of dependencies (like pressure, temperature)."""
    return pp.ad.SurrogateFactory(
        name,
        mdg,
        dependencies,
    )


# TODO move below two inquires once flash and CFLE are merged.
def get_local_equilibrium_condition(model: pp.PorePyModel) -> str | None:
    """
    Parameters:
        model: A PorePy model.

    Returns:
        The local equilibrium condition stored in
        ``model.params['equilibrium_condition']`. Defaults to None.

        Expected equilibrium conditions are any combination of to state functions fixed
        at equilibrium, e.g. ``'p-T'``, ``'p-h'``.

        Additional qualifiers also also allowed, e.g. ``'unified-p-h'``.

    """
    et = model.params.get("equilibrium_condition", None)
    if et is not None:
        return str(et)
    else:
        return None


def has_unified_equilibrium(model: pp.PorePyModel) -> bool:
    """
    Parameters:
        model: A PorePy model.

    Returns:
        True, if the keyword ``'unified'`` is in
        ``model.params['equilibrium_condition']``, if given at all. Defaults to False.

    """
    if "unified" in str(get_local_equilibrium_condition(model)).lower():
        return True
    else:
        return False


class _MixtureDOFHandler(pp.PorePyModel):
    """A class to help resolve the independent fractional variables of an arbitrary
    mixture, and respectively the DOFs.

    For fluid with more than 1 phase or more than 1 component, the system automatically
    includes additional unknowns. In the case of more than 1 phase, saturations become
    unknowns. In the case of more than 1 component, various massic/molar fractions
    become unknowns.

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

        Finally, the ``'equilibrium_condition'`` set in the model's :attr:`params` is
        used to determine the independency of partial and extended fractions of
        components in phases.

    """

    # Logic methods determining existence of DOFs.

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
            eliminated = self._is_reference_phase_eliminated()
        elif isinstance(instance, Component):
            instances = list(self.fluid.components)
            if instance not in instances:
                raise ValueError(f"Component {instance} not in fluid mixture.")
            idx = instances.index(instance)
            ref_idx = self.fluid.reference_component_index
            num_instances = self.fluid.num_components
            eliminated = self._is_reference_component_eliminated()
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
        if phase.state == PhysicalState.solid:
            return False
        else:
            return self.has_independent_fraction(phase)

    def has_independent_tracer_fraction(
        self, tracer: Component, compound: Compound
    ) -> bool:
        """Checks if the :attr:`~porepy.compositional.base.Compound.tracer_fraction_of`
        a ``tracer`` in the ``compound`` is an independent variable.

        Paramters:
            tracer: An active tracer in one of the compounds in the
                :attr:`fluid`.
            compound: A component/compound in the :attr:`fluid`'s
                :attr:`~porepy.compositional.base.Fluid.components`.

        Raises:
            ValueError: If the ``compound`` is not in the :attr:`fluid`.

        Returns:
            True, if the ``tracer`` is in the compound``, False otherwise.

        """
        if compound not in self.fluid.components:
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

        If the model contains a unified local equilibrium, partial fractions are always
        dependent, since they are obtained by normalization of extended fractions.

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

        See also:

            :func:`get_local_equilibrium_condition`

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

        if has_unified_equilibrium(self):
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

        Extended fractions are only used if there is a unified equilibrium defintion.

        In that case, if the ``component`` is modelled in the ``phase``, the respective
        extended fraction is always an independent variable.
        Since for vanished phases the extended fractions do not necessarily fulfill the
        unity constraint, no extended fraction can be eliminated.

        See also:

            :func:`get_local_equilibrium_condition`

        Paramters:
            component: Any component in the :attr:`fluid`.
            phase: Any phase in the :attr:`fluid`.

        Raises:
            ValueError: If the ``phase`` or the ``component are not in the
                :attr:`fluid`.
            CompositionalModellingError: If the ``component`` is not in the phase.
                The unified setting expects all components to be modelled in all phases.

        Returns:
            True, the model has a unified local equilibrium defined. False otherwise.

        """
        if phase not in self.fluid.phases:
            raise ValueError(f"Phase {phase} not in fluid mixture.")
        if component not in self.fluid.components:
            raise ValueError(f"Component {component} not in fluid mixture.")

        if has_unified_equilibrium(self):
            if component not in phase:
                raise CompositionalModellingError(
                    f"Component {component} not in phase {phase}."
                    + " Models with unified equilibrium require all components to"
                    + " be modelled in all phases."
                )
            return True
        else:
            return False

    def has_independent_fluid_fraction(self, instance: Element) -> bool:
        """Checks whether the ``instance`` has an independent variable for the
        fraction of moles associated with it (
        :attr:`~porepy.compositional.base.Element.fluid_fraction` of an element
        )


        Parameters:
            instance: An element in the :attr:`fluid`.

        Raises:
            ValueError: If the ``instance`` is not in the fluid mixture.
            TypeError: If ``instance`` is not an element.

        Returns:
            False, if there is only 1 instance (of elements) in the fluid
            mixture, or it is the reference instance and it was eliminated.
            Otherwise it returns True.

        """
        instances: list[Element]
        if isinstance(instance, Element):
            instances = list(self.fluid.elements)
            if instance not in instances:
                raise ValueError(f"Element {instance} not in fluid mixture.")
            idx = instances.index(instance)
            ref_idx = 0
            num_instances = len(instances)
            eliminated = True
        else:
            raise TypeError(f"Unknown type {type(instance)}. Expecting element.")

        if (idx == ref_idx and eliminated) or num_instances == 1:
            self.fluid.reference_element = instance
            return False
        else:
            return True

    # Utility methods for DOFs and variables

    def _overall_fraction_variable(self, component: Component) -> str:
        """Returns the name of the fraction variable assigned to ``component``."""
        return f"{symbols['overall_fraction']}_{component.name}"

    def _mineral_saturation_variable(self, component: Component) -> str:
        """Returns the name of the mineral_saturation variable assigned to ``component``."""
        return f"{symbols['mineral_saturation']}_{component.name}"

    def _saturation_variable(self, phase: Phase) -> str:
        """Returns the name of the saturation variable assigned to ``phase``."""
        return f"{symbols['phase_saturation']}_{phase.name}"

    def _tracer_fraction_variable(self, tracer: Component, compound: Compound) -> str:
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

    def _element_fluid_fraction_variable(self, element: Element) -> str:
        """Returns the name of the fraction variable assigned to ``component``."""
        return f"{symbols['element_fluid_fraction']}_{element.name}"

    def _element_chemical_potential_variable(self, element: Element) -> str:
        return f"{symbols['element_chemical_potential']}_{element.name}"

    def _equilibrium_stability_index_variable(self, component: Component) -> str:
        return f"{symbols['equilibrium_stability_index']}_{component.name}"

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
                    name=name, domains=cast(Sequence[pp.BoundaryGrid], domains)
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
        if get_local_equilibrium_condition(self) is not None:
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
                if has_unified_equilibrium(self):
                    if self.has_independent_extended_fraction(comp, phase):
                        append = True
                elif self.has_independent_partial_fraction(comp, phase):
                    append = True
                if append:
                    names.append(self._partial_fraction_variable(comp, phase))

        return names

    @property
    def element_fluid_fraction_variables(self) -> list[str]:
        """Names of independent element fluid fractions
        :attr:`~porepy.compositional.base.element.fluid_fraction` variables created for this
        model."""
        names: list[str] = []
        for ele in self.fluid.elements:
            if self.has_independent_fluid_fraction(ele):
                names.append(self._element_fluid_fraction_variable(ele))
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
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]

    def create_variables(self) -> None:
        """Creates the sets of required variables for a fluid mixture.

        The following actions are taken:
        1. :meth:`overall_fraction` is called to assign
           :attr:`~porepy.compositional.base.Component.fraction` to components.
        2. :meth:`tracer_fraction` is called to assign
           :attr:`~porepy.compositional.base.Compound.tracer_fraction_of` for each
           tracer in a compound.
        3. :meth:`saturation` is called to assign
           :attr:`~porepy.compositional.base.Phase.saturation` to phases.
        4. :meth:`phase_fraction' is called to assign
           :attr:`~porepy.compositional.base.Phase.fraction` to phases.
        5. :meth:`extended_fraction` is called to assing
           :attr:`~porepy.compositional.base.Phase.extended_fraction_of` for each phase
           and component.
        6. :meth:`partial_fraction` is called to assign
           :attr:`~porepy.compositional.base.Phase.partial_fraction_of` for each phase
           and component.

        Note however, that dependening on the mixture and model configuration, the
        objects and callables created here can be independent variables or dependent
        expressions (regular AD operators).

        """
        super().create_variables()

        if not hasattr(self, "fluid"):
            raise CompositionalModellingError(
                "Cannot create fluid mixture variables before defining a fluid mixture."
            )

        # NOTE: The creation of variables seems repetitive (it is), but it is done this
        # way to preserve a certain order (component-wise, phase-wise and family-wise
        # for each family of fractions).

        # Creation of feed fractions.
        for component in self.fluid.components:
            component.fraction = self.overall_fraction(component)
            component.equilibrium_stability_index = self.equilibrium_stability_index(
                component
            )
            component.mineral_saturation = self.mineral_saturation(component)
            component.reactive_source = self.reactive_source(component)

        # Creation of tracer fractions for compounds.
        for component in self.fluid.components:
            if isinstance(component, Compound):
                component.tracer_fraction_of = {}
                for tracer in component.active_tracers:
                    component.tracer_fraction_of[tracer] = self.tracer_fraction(
                        tracer, component
                    )

        # NOTE: All variables associated with transport of mass are now created.
        # Below variables are of local nature.

        # Creation of saturation variables.
        for phase in self.fluid.phases:
            phase.saturation = self.saturation(phase)

        # Creation of phase fraction variables.
        for phase in self.fluid.phases:
            phase.fraction = self.phase_fraction(phase)

        # Creation of extended fractions.
        for phase in self.fluid.phases:
            phase.extended_fraction_of = {}
            # NOTE: Iterate over components in phase, not all components to avoid
            # conflicts with non-unified set-ups. The check of whether all phases have
            # all components must be done elsewhere.
            for comp in phase:
                phase.extended_fraction_of[comp] = self.extended_fraction(comp, phase)

        # Creation of partial fractions.
        for phase in self.fluid.phases:
            phase.partial_fraction_of = {}
            for comp in phase:
                phase.partial_fraction_of[comp] = self.partial_fraction(comp, phase)

        # Creation of activities.
        for phase in self.fluid.phases:
            phase.activity_of = {}
            for comp in phase:
                phase.activity_of[comp] = self.activity_model(comp, phase)

        # Creation of element fluid fractions.
        for element in self.fluid.elements:
            element.fluid_fraction = self.element_fluid_fraction(element)
            element.element_chemical_potential = self.element_chemical_potential(
                element
            )

        self.fluid.element_density_ratio = self.element_density_ratio

        if self.reactions:
            self.set_kinetic_reaction_rates(self.reactions)


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

        # If only 1 component, the fraction is always 1.
        if self.fluid.num_components == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single_feed_fraction")
        elif component in self.fluid.solid_components:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                z = (
                    component.mineral_saturation(domains)
                    * pp.ad.Scalar(self.solid.total_porosity / component.molar_volume)
                    / self.total_molar_concentration(domains)
                )
                z.set_name(f"mineral_overall_fraction_{component.name}")
                return z
        # NOTE: If the reference component fraction is independent, below elif-clause
        # will be executed, instead of the next one.
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

        # Should never happen.
        else:
            raise NotImplementedError("Missing logic for overall fractions.")

        return fraction

    def tracer_fraction(
        self, tracer: pp.FluidComponent, compound: Compound
    ) -> DomainFunctionType:
        """Getter method to create a callable representing the tracer fraction of an
        active tracer in a compound, on a list of subdomains or boundaries.

        The base method creates tracer fractions as an independent variables
        (transportable), after asserting the tracer is indeed in that compound.

        Parameters:
            tracer: An active tracer in the fluid mixture.
            compound: A compound in the fluid mixture.

        Returns:
            A callable which returns the tracer fraction for a given set of domains.

        """
        assert tracer in compound.active_tracers, (
            f"Solute {tracer.name} not in compound {compound.name}"
        )

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

        # If only 1 phase, the saturation is always 1.
        if self.fluid.num_phases == 1:

            def saturation(subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single_phase_saturation")

        # NOTE: If the reference phase is independent, below elif-clause will be
        # executed.
        elif self.has_independent_saturation(phase):
            saturation = self._fraction_factory(self._saturation_variable(phase))
        # If reference component, eliminate by unity.
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
        elif phase.state == PhysicalState.solid:

            def saturation(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(0.0, "solid_phase_saturation")

        # Should never happen
        else:
            raise NotImplementedError("Missing logic for saturations.")

        return saturation

    def phase_fraction(self, phase: Phase) -> DomainFunctionType:
        """Analogous to :meth:`saturation` but for phase molar fractions."""

        fraction: DomainFunctionType
        # Code is completely analogous to method saturation, except that we raise a
        # modelling error if no equilibrium is defined. Phase fractions can completely
        # be omitted in that case.
        if self.fluid.num_phases == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single_phase_fraction")

        elif phase.state == PhysicalState.solid:
            # here set the fraction of solid phase to 0 because it is related to saturation

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(0.0, "solid_phase_fraction")
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
        elif get_local_equilibrium_condition(self) is None:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                raise CompositionalModellingError(
                    "Phase fractions are not defined in model without equilibrium."
                    + " A re-formulation using saturations is required."
                )
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
            component: A component in the fluid mixture.
            phase: A phase in the fluid mixture.

        Returns:
            A callable which returns the extended fraction for a given set of domains.

        """
        assert component in phase, (
            f"Component {component.name} not in phase {phase.name}"
        )

        fraction: DomainFunctionType

        # Add this for completeness reasons, s.t. the phase has the respective
        # attribute. But raise an error if the user tries to access the fraction.

        # NOTE Extended fractions are in general always unknowns, even in
        # 1 component, multiphase case (they are some value below 1 if a phase vanishes)
        # Only in the case with 1 component and 1 phase, the extended fraction is
        # also a scalar 1, since the 1 modelled phase cannot vanish.
        if self.fluid.num_components == self.fluid.num_phases == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(
                    1.0, "single_component_single_phase_extended_fraction"
                )

        # If no unified equilibrium, calling the extended fractions will raise an error.
        elif not has_unified_equilibrium(self):

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
            component: A component in the fluid mixture.
            phase: A phase in the fluid mixture.

        Returns:
            A callable which returns the extended fraction for a given set of domains.

        """
        assert component in phase, (
            f"Component {component.name} not in phase {phase.name}"
        )

        fraction: DomainFunctionType

        # Case only 1 component in phase: partial fraction is always 1
        if phase.num_components == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(
                    1.0, f"single_partial_fraction_{component.name}_{phase.name}"
                )

        # Case only 1 phase in entire mixture, partial fractions are equal to overall
        # fractions.
        elif self.fluid.num_phases == 1:
            # Mypy complains that above the argument of fraction is explicitly
            # stated as 'domains', while extended_fraction returns no information
            # on how the argument is called.
            # But both are (pp.SubdomainOrBoundaries) -> pp.ad.Operator.
            fraction = component.fraction  # type:ignore[assignment]
        # Case reference component: Partial fraction of a phase's reference component
        # is always expressed by unity of partial fractions in that phase.
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

        # Case of unified equilibrium: Partial fractions are obtained by normalization.
        elif has_unified_equilibrium(self):
            # NOTE the fraction of the phase's reference component is covered above.
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

    def element_fluid_fraction(
        self,
        element: Element,
    ) -> DomainFunctionType:
        """Getter method to create a callable representing the overall fraction of an
        element on a list of subdomains or boundaries.

        Cases where the element fluid fraction is not an independent variable:

        1. If there is only 1 element, the fraction is constant 1.
        2. If the reference element fluid fraction was eliminated, it is a dependent
           operator.

        Parameters:
            element: An element in the fluid mixture.

        Returns:
            A callable which returns the element fluid fraction for a given set of domains.

        """

        fraction: DomainFunctionType

        # If only 1 component, the fraction is always 1.
        if self.fluid.num_elements == 1:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(1.0, "single_element_fluid_fraction")

        # NOTE: If the reference component fraction is independent, below elif-clause
        # will be executed, instead of the next one.
        elif self.has_independent_fluid_fraction(element):
            # we enable chemical equilibrium only for testing purposes
            # self.fluid.enable_chemical_equilibrium = True
            if self.fluid.enable_chemical_equilibrium:
                fraction = self._fraction_factory(
                    self._element_fluid_fraction_variable(element)
                )
            else:

                def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    return pp.ad.Scalar(0.0, "could_not_care_less")
        elif element == self.fluid.reference_element:

            def fraction(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                e_R = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
                    [
                        element.fluid_fraction(domains)
                        for element in self.fluid.elements
                        if element != self.fluid.reference_element
                    ]
                )
                e_R.set_name("reference_element_fluid_fraction_by_unity")
                return e_R

        # Should never happen.
        else:
            raise NotImplementedError("Missing logic for element fluid fractions.")

        return fraction

    def element_density_ratio(
        self, subdomains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:  # TODO
        """
        Evaluate the ratio of total element density to total species molar density
        over a given set of subdomains.

        The formula is:
            ratio = ∑_{e=1}^{E} ∑_{ξ=1}^{C} W[e, ξ] * z_ξ(domains)

        Args:
            subdomains: The subdomains (e.g. grid cells) to evaluate on.

        Returns:
            An AD-compatible Operator evaluated over the subdomains.
        """
        if not (hasattr(self, "fluid") and hasattr(self.fluid, "fluid_formula_matrix")):
            raise AttributeError("Fluid or fluid_formula_matrix not initialized.")

        W = self.fluid.fluid_formula_matrix  # shape (E, C)
        species_names = self.fluid.fluid_species_names
        components = self.fluid.components

        # Map species name -> AD function
        z_funcs = {comp.name: comp.fraction for comp in components}

        # Evaluate z_ξ(subdomains) to get a list of Operators
        try:
            z_ops = [z_funcs[name](subdomains) for name in species_names]  # shape (C,)
        except KeyError as e:
            raise KeyError(f"Species name '{e.args[0]}' not found in fluid components.")

        # Compute ∑_{e} ∑_{ξ} W[e, ξ] * z_ξ
        total_op = pp.ad.sum_operator_list(
            [
                pp.ad.sum_operator_list(
                    [pp.ad.Scalar(w) * z for w, z in zip(W_row, z_ops)]
                )
                for W_row in W
            ]
        )

        # scaling_factor = self.fluid_molar_fraction(subdomains) ** pp.ad.Scalar(-1.0)
        # total_op = total_op * scaling_factor
        total_op.set_name("element_density_ratio")
        return total_op

    def element_chemical_potential(
        self,
        element: Element,
    ) -> DomainFunctionType:
        """Getter method to create a callable representing the chemical potential of an
        element on a list of subdomains or boundaries.

        Parameters:
            element: An element in the fluid mixture.

        Returns:
            A callable which returns the element chemical potential for a given set of domains.

        """

        yy: DomainFunctionType
        # disable this for now
        if self.fluid.enable_chemical_equilibrium and 1 == 0:
            yy = self._fraction_factory(
                self._element_chemical_potential_variable(element)
            )
        else:

            def yy(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(0.0, "could_not_care_less")

        return yy

    def equilibrium_stability_index(
        self,
        component: Component,
    ) -> DomainFunctionType:
        """Getter method to create a callable representing the equilibrium_stability_index of a
        component on a list of subdomains or boundaries.

        Parameters:
            component: A component in the fluid mixture.

        Returns:
            A callable which returns the equilibrium_stability_index for a given set of domains.

        """

        zz: DomainFunctionType
        # disable this for now
        if self.fluid.enable_chemical_equilibrium and 1 == 0:
            zz = self._fraction_factory(
                self._equilibrium_stability_index_variable(component)
            )
        else:

            def zz(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                return pp.ad.Scalar(0.0, "could_not_care_less")

        return zz

    def fluid_molar_fraction(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Molar fraction of the fluid.
        Parameters:
            domains: A sequence of grids.

        """

        if len(self.fluid.solid_components) > 0:
            op = (
                self.porosity(domains)
                * self.fluid.density(domains)
                / self.total_molar_concentration(domains)
            )

        else:
            op = pp.ad.Scalar(1)
            op.set_name("fluid_molar_fraction")

        return op

    def total_molar_concentration(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Total molar concentration of the fluid."""

        if len(self.fluid.solid_components) > 0:
            solid_molar_concentration = pp.ad.sum_operator_list(
                [
                    comp.mineral_saturation(domains)
                    * pp.ad.Scalar(self.solid.total_porosity)
                    / pp.ad.Scalar(comp.molar_volume)
                    for comp in self.fluid.solid_components
                ]
            )

            op = (
                self.porosity(domains) * self.fluid.molar_density(domains)
            ) + solid_molar_concentration
        else:
            op = self.porosity(domains) * self.fluid.molar_density(domains)
        return op

    def mineral_saturation(
        self,
        component: Component,
    ) -> DomainFunctionType:
        """
        def mineral_saturation(
            domains: pp.SubdomainsOrBoundaries,
        ) -> pp.ad.Operator:
            s = (
                self.total_molar_concentration(domains)
                * component.fraction(domains)
                * pp.ad.Scalar(component.molar_volume)
                / pp.ad.Scalar(self.solid.total_porosity)
            )
            s.set_name(f"mineral_saturation_of_{component.name}")
            return s

        return mineral_saturation
        """
        if component in self.fluid.solid_components:
            fraction = self._fraction_factory(
                self._mineral_saturation_variable(component)
            )
            return fraction
        else:

            def mineral_saturation(
                domains: pp.SubdomainsOrBoundaries,
            ) -> pp.ad.Operator:
                return pp.ad.Scalar(0.0, f"{component.name} is not a mineral")

            return mineral_saturation

    def reactive_solid_density(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Density of the solid phase."""
        if len(self.fluid.solid_components) == 0:
            raise CompositionalModellingError(
                "No solid components in fluid, cannot compute solid density."
            )

        rho_s = pp.ad.sum_operator_list(
            [
                comp.mineral_saturation(domains)
                * pp.ad.Scalar(self.solid.total_porosity)
                / pp.ad.Scalar(comp.molar_volume)
                / (pp.ad.Scalar(self.solid.total_porosity) - self.porosity(domains))
                for comp in self.fluid.solid_components
            ]
        )
        rho_s.set_name("reactive_solid_density")
        return rho_s

    def reactive_source(self, component: pp.Component) -> DomainFunctionType:
        """Source term in a component's mass balance equation due to reactions.

        Parameters:
            component: A component in the :attr:`fluid`.
            subdomains: A list of subdomains in the :attr:`mdg`.
        Returns:
            The reactive source term for the given component.
        """
        # --- Validation

        op: DomainFunctionType

        def op(subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            if not hasattr(self.fluid, "stoichiometric_matrix"):
                raise ValueError("stoichiometric_matrix is missing.")
            if not hasattr(self, "species_names") or not self.species_names:
                raise ValueError("self.species_names must be a non-empty list.")
            if not hasattr(self, "reactions"):
                raise ValueError("self.reactions is missing.")
            S = self.fluid.stoichiometric_matrix
            reactions = self.reactions
            n_rxn, n_sp = S.shape
            if n_sp != len(self.species_names):
                raise ValueError(
                    "Column count of stoichiometric matrix must equal len(self.species_names)."
                )
            if len(reactions) != n_rxn:
                raise ValueError(
                    "Number of reactions must match the number of rows in the stoichiometric matrix."
                )

            species_names = self.species_names
            reaction_formulas = self.reaction_formulas

            # Map species name -> AD function
            r_funcs = {
                reaction.formula: reaction.reaction_rate for reaction in reactions
            }

            # Evaluate z_ξ(subdomains) to get a list of Operators
            try:
                z_ops = [
                    r_funcs[formula](subdomains) for formula in reaction_formulas
                ]  # shape (C,)
            except KeyError as e:
                raise KeyError(f"Reaction '{e.args[0]}' not found.")

            species_index = species_names.index(component.name)

            total_op = pp.ad.sum_operator_list(
                [pp.ad.Scalar(S[r, species_index]) * z for r, z in enumerate(z_ops)]
            )

            return total_op

        return op

    def molar_bulk_concentration(
        self, comp: pp.Component, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Total molar concentration of the fluid."""

        op = self.total_molar_concentration(domains) * comp.fraction(domains)
        op.set_name(f"molar_bulk_concentration_of_{comp.name}")
        return op



class FluidMixin(pp.PorePyModel):
    """Mixin class for introducing a general fluid (mixture) into a PorePy model and
    providing it as an attribute :attr:`fluid`.

    Fluid properties are by definition expressed through respective phase properties,
    which can be overridden here as part of the constitutive modelling.

    The following methods are factories to provide functions for phase properties:

    - :meth:`density_of_phase`
    - :meth:`specific_volume_of_phase`
    - :meth:`specific_enthalpy_of_phase`
    - :meth:`viscosity_of_phase`
    - :meth:`thermal_conductivity_of_phase`
    - :meth:`fugacity_coefficient`
    - :meth:`chemical_potential`

    There is no need to modify the :attr:`fluid` itself.

    Important:
        Solution strategies must follow a certain order during the set up
        (`prepare_simulation'):

        1. The fluid must be created (phases and components).
        2. The variables must be created (depending on present phases and components).
        3. The fluid properties are assigned (since they in general depend on
            variables).

    The base class provides phase properties as general
    :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory`.
    To use heuristic laws in form of AD-compatible functions, override above mentioned
    methods via mixins.

    The base class also provides a default set-up in form of a 1-phase, 1-component
    fluid, based on fluid component found in ``params``.
    To modify the default set-up, provide overrides for

    - :meth:`get_components`
    - :meth:`get_phase_configuration`

    To define the dependency of phase properties in terms of variables, see

    - :meth:`dependencies_of_phase_properties`

    For a more complex set-up involving phases with *different* components, see

    - :meth:`set_components_in_phases`

    """

    def create_fluid(self) -> None:
        """Set-up method to create a fluid mixture.

        It calls

        1. :meth:`get_components`,
        2. :meth:`get_phase_configuration` and
        3. :meth:`set_components_in_phases`,

        and creates the instance :attr:`fluid`.

        """

        # Need annotations which represent the default implementation using
        # FluidComponent.
        phases: list[Phase[pp.FluidComponent]] = []
        components: list[pp.FluidComponent] = [c for c in self.get_components()]

        for config in self.get_phase_configuration(components):
            # Configuration of phase with EoS.
            if len(config) == 3:
                phase_state, name, eos = config
                assert isinstance(eos, EquationOfState), (
                    f"Expecting an instance of `EquationOfState`, got {type(eos)}."
                )
            # Configuration of phase without EoS.
            elif len(config) == 2:
                phase_state, name = config
                eos = None

            assert phase_state in PhysicalState, (
                f"Expecting a valid `PhysicalState`, got {phase_state}."
            )
            assert isinstance(name, str), (
                f"Expecting a string as name for phase, got {type(name)}."
            )

            phases.append(Phase(phase_state, name, eos=eos))

        self.set_components_in_phases(components, phases)

        self.fluid = Fluid(components, phases)

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Method to return a list of modelled components.

        The default implementation takes the user-provided or default fluid component
        found in the model ``params`` and returns a single component.

        Override this method via mixin to provide a more complex component context for
        the :attr:`fluid`.

        """
        # Should be available after SolutionStrategy.set_materials()
        # Getting the user-passed or default fluid component to create the default fluid
        # component.
        fluid_constants = self.params["material_constants"]["fluid"]
        # All materials are assumed to derive from Component.
        assert isinstance(fluid_constants, Component), (
            "model.params['material_constants']['fluid'] must be of type "
            + f"{Component}"
        )
        # Need to cast into FluidComponent, because of the assert statement above.
        return [cast(pp.FluidComponent, fluid_constants)]

    def get_phase_configuration(
        self, components: Sequence[ComponentLike]
    ) -> Sequence[
        tuple[PhysicalState, str] | tuple[PhysicalState, str, EquationOfState]
    ]:
        """Method to return a configuration of modelled phases.

        The default implementation returns a liquid-like phase named ``'liquid'``
        (to be used in the standard set-up with heuristic fluid properties implemented
        for 1-phase fluids).

        Parameters:
            components: The list of components modelled by :meth:`get_components`.

                Note:
                    The reason why this is passed as an argument is to avoid
                    constructing multiple, possibly expensive EoS instances.
                    The user can use only a single EoS instance for all phases f.e.

        Returns:
            A sequence of 2-tuples or 3-tuples containing

            1. The phase state.
            2. A name for the phase.
            3. (optional) An instance of an EoS.

            Each tuple will be used to create a phase in the fluid mixture.
            For more information on the required return values see
            :class:`~porepy.compositional.base.Phase`.

            Phase configurations which do not return an EoS are assumed to use
            heuristics.

        """
        return [(PhysicalState.liquid, "liquid")]

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
        - :meth:`viscosity_of_phase` to
          :attr:`~porepy.compositional.base.Phase.viscosity`
        - :meth:`thermal_conductivity_of_phase` to
          :attr:`~porepy.compositional.base.Phase.thermal_conductivity`
        - :meth:`fugacity_coefficient` to
          :attr:`~porepy.compositional.base.Phase.fugacity_coefficient_of`
          for each component in respective phase.
          This is only done for mixtures with a defined equilibrium condition.

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
            phase.chemical_potential_of = {}
            phase.molar_density=self.molar_density_of_phase(phase)

            for comp in phase:
                phase.fugacity_coefficient_of[comp] = self.fugacity_coefficient(
                    comp, phase
                )
                phase.chemical_potential_of[comp] = self.chemical_potential(comp, phase)

    def molar_density_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
       #if the eos is provided for massic density, we can compute the molar density
        molar_density: ExtendedDomainFunctionType

        def molar_density(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            mc=self.params["material_constants"]
            mode=mc.get("molar_density_mode","fully_coupled")
            mean_molar_mass=pp.ad.sum_operator_list(
            [
                pp.ad.Scalar(comp.molar_mass) * phase.partial_fraction_of[comp](domains)
                for comp in phase.components
            ]
            )
            if mode=="provided":
                #return phase.density(domains)
                rho_ref = pp.ad.Scalar(self.fluid.reference_component.molar_density)

                rho_ = rho_ref * self.pressure_exponential(cast(list[pp.Grid], domains))
                rho_.set_name("fluid_molar_density_from_pressure")
                return rho_


            elif mode=="fully_coupled":
                return phase.density(domains) / mean_molar_mass

        return molar_density


    def dependencies_of_phase_properties(
        self, phase: Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        """Method to define the signature of phase properties, which are dependent
        quantities.

        In the case of a unified equilibrium formulation, the properties depend on
        pressure, temperature and extended fractions.

        In the case of a non-unified equilibrium formulation, the properties are
        dependent on pressure, temperature and partial fractions in that phase.

        Otherwise they are depend on pressure, temperature and **independent** overall
        fractions.

        Important:
            This method returns an empty list of dependencies, for reasons of
            compatibility with pure mechanics models. The method must be overwritten in
            every flow problem which relies on e.g.,
            :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory` for
            wrapping externally computed values as Ad operators. Specifically, the
            SurrogateFactory framework requires information about the input arguments of
            the externally computed property values (primary variables and their indices
            in the system's Jacobian). See below for an example to overwrite this
            method.

        Important:
            The default return value (empty list), leads to phase property function
            factories

            - :attr:`density_of_phase`
            - :attr:`specific_enthalpy_of_phase`
            - :attr:`viscosity_of_phase`
            - :attr:`thermal_conductivity_of_phase`
            - :attr:`fugacity_coefficient`
            - :attr:`chemical_potential`


            returning functions which raise an error when called, alerting the user that
            some some mixin is missing to provide respective functions as the
            implementation of some heuristic law.

        Example:

            Let's assume a general mixture with multiple components and phases, which
            uses the :class:`CompositionalVariables`, and an equilibrium is defined.
            I.e., the fluid and phase properties are not characterized by some mixed-in,
            heuristic law, but by a local phase equilibrium system or some
            interpolation. The values of properties like density are then dependent on
            some variables. This information is used to evaluate those variables and
            provide the information to flash calculations or look-up, and subsequently
            to populate the Jacobians of the property.

            The above mentioned logic can then be translated into code the following
            way:

            .. code-block:: python

                class MyMixin:

                    def dependencies_of_phase_properties(
                        self, phase: Phase
                    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:

                        dependencies = [self.pressure]
                        if 'unified' in pp.get_local_equilibrium_condition(
                            self
                        ).lower():

                            dependencies +=  [self.temperature] + [
                                phase.extended_fraction_of[component]
                                for component in phase
                                if self.has_independent_extended_fraction(
                                    component, phase
                                )
                            ]

                        elif pp.get_local_equilibrium_condition(self):

                            dependencies += [self.temperature] + [
                                phase.partial_fraction_of[component]
                                for component in phase
                                if self.has_independent_partial_fraction(
                                    component, phase
                                )
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
        if :meth:`dependencies_of_phase_properties` has a non-empty return value.

        Otherwise it returns an empty function, raising an error when called
        (missing mixin of constitutive laws).

        The phase density (like all thermodynamic properties) is a dependent quantity.

        Parameters:
            phase: A phase in the :attr:`fluid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """
        name = f"phase_{phase.name}_density"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def specific_volume_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """The specific volume of the phase is returned as a function calling the
        the phase density and taking the reciprocal of it.

        Parameters:
            phase: A phase in the :attr:`fluid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """

        def volume(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            op = phase.density(domains)
            op = op ** pp.ad.Scalar(-1.0)
            op.set_name(f"phase_{phase.name}_specific_volume")
            return op

        return volume

    def specific_enthalpy_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`, but for
        :attr:`~porepy.compositional.base.Phase.specific_enthalpy` of a ``phase``.

        Parameters:
            phase: A phase in the :attr:`fluid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """
        name = f"phase_{phase.name}_specific_enthalpy"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def viscosity_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`,  but for
        :attr:`~porepy.compositional.base.Phase.viscosity` of a ``phase``.

        Parameters:
            phase: A phase in the :attr:`fluid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """
        name = f"phase_{phase.name}_viscosity"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def thermal_conductivity_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`, but for
        :attr:`~porepy.compositional.base.Phase.thermal_conductivity` of a ``phase``.

        Parameters:
            phase: A phase in the :attr:`fluid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """
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
        :attr:`~porepy.compositional.base.Phase.fugacity_coefficient_of` of a ``phase``.

        Note:
            Fugacity coefficient appear only in the local equilibrium
            equation or other chemistry-related models, but not in flow and transport.

        Parameters:
            phase: A phase in the :attr:`fluid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """
        name = f"fugacity_coefficient_{component.name}_in_{phase.name}"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def chemical_potential(
        self, component: Component, phase: Phase
    ) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`, but for
        :attr:`~porepy.compositional.base.Phase.chemical_potential_of` of a ``phase``.

        Note:
            Chemical potential appear only in the local equilibrium
            equation or other chemistry-related models, but not in flow and transport.

        Parameters:
            phase: A phase in the :attr:`fluid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """
        name = f"chemical_potential_{component.name}_in_{phase.name}"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function


class SolidMixin(pp.PorePyModel):
    """Mixin class for introducing a general solid (mixture) into a PorePy model and
    providing it as an attribute :attr:`solid`.

    Solid properties are by definition expressed through respective phase properties,
    which can be overridden here as part of the constitutive modelling.

    The following methods are factories to provide functions for phase properties:

    - :meth:`density_of_phase`
    - :meth:`specific_volume_of_phase`
    - :meth:`thermal_conductivity_of_phase`

    There is no need to modify the :attr:`solid` itself.

    Important:
        Solution strategies must follow a certain order during the set up
        (`prepare_simulation'):

        1. The solid must be created (phases and components).
        2. The variables must be created (depending on present phases and components).
        3. The solid properties are assigned (since they in general depend on
            variables).

    The base class provides phase properties as general
    :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory`.
    To use heuristic laws in form of AD-compatible functions, override above mentioned
    methods via mixins.

    The base class also provides a default set-up in form of a 1-phase, 1-component
    solid, based on solid component found in ``params``.
    To modify the default set-up, provide overrides for

    - :meth:`get_solid_components`
    - :meth:`get_solid_phase_configuration`

    To define the dependency of phase properties in terms of variables, see

    - :meth:`dependencies_of_phase_properties`

    For a more complex set-up involving phases with *different* components, see

    - :meth:`set_components_in_phases`

    """

    def create_solid(self) -> None:
        phases: list[Phase[pp.SolidComponent]] = []
        components: list[pp.SolidComponent] = [c for c in self.get_solid_components()]

        for config in self.get_solid_phase_configuration(components):
            # Configuration of phase with EoS.
            if len(config) == 3:
                phase_state, name, eos = config
                assert isinstance(eos, EquationOfState), (
                    f"Expecting an instance of `EquationOfState`, got {type(eos)}."
                )
            # Configuration of phase without EoS.
            elif len(config) == 2:
                phase_state, name = config
                eos = None

            assert phase_state in PhysicalState, (
                f"Expecting a valid `PhysicalState`, got {phase_state}."
            )
            assert isinstance(name, str), (
                f"Expecting a string as name for phase, got {type(name)}."
            )

            phases.append(Phase(phase_state, name, eos=eos))

        self.set_components_in_solid_phases(components, phases)
        self.solid = Solid(components, phases)
        if hasattr(self, "fluid"):
            self.fluid.solid_components = components

    def get_solid_components(self) -> Sequence[pp.SolidComponent]:
        """Method to return a list of modelled components.

        The default implementation takes the user-provided or default solid component
        found in the model ``params`` and returns a single component.

        Override this method via mixin to provide a more complex component context for
        the :attr:`solid`.

        """
        # Should be available after SolutionStrategy.set_materials()
        # Getting the user-passed or default solid component to create the default solid
        # component.
        solid_constants = self.params["material_constants"]["solid"]
        # All materials are assumed to derive from Component.
        assert isinstance(solid_constants, Component), (
            "model.params['material_constants']['solid'] must be of type "
            + f"{Component}"
        )
        # Need to cast into SolidComponent, because of the assert statement above.
        return [cast(pp.SolidComponent, solid_constants)]

    def get_solid_phase_configuration(
        self, components: Sequence[ComponentLike]
    ) -> Sequence[
        tuple[PhysicalState, str] | tuple[PhysicalState, str, EquationOfState]
    ]:
        return [(PhysicalState.solid, "solid")]

    def set_components_in_solid_phases(
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

    def dependencies_of_phase_properties(
        self, phase: Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        return []

    def density_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """This base method returns the density of a ``phase`` as a
        :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory`,
        if :meth:`dependencies_of_phase_properties` has a non-empty return value.

        Otherwise it returns an empty function, raising an error when called
        (missing mixin of constitutive laws).

        The phase density (like all thermodynamic properties) is a dependent quantity.

        Parameters:
            phase: A phase in the :attr:`solid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """
        name = f"phase_{phase.name}_density"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function

    def specific_volume_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """The specific volume of the phase is returned as a function calling the
        the phase density and taking the reciprocal of it.

        Parameters:
            phase: A phase in the :attr:`solid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """

        def volume(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            op = phase.density(domains)
            op = op ** pp.ad.Scalar(-1.0)
            op.set_name(f"phase_{phase.name}_specific_volume")
            return op

        return volume

    def thermal_conductivity_of_phase(self, phase: Phase) -> ExtendedDomainFunctionType:
        """Analogous to :meth:`density_of_phase`, but for
        :attr:`~porepy.compositional.base.Phase.thermal_conductivity` of a ``phase``.

        Parameters:
            phase: A phase in the :attr:`solid`.

        Returns:
            A callable taking some domains and returning an AD operator representing
            this thermodynamic property.

        """
        name = f"phase_{phase.name}_conductivity"
        dependencies = self.dependencies_of_phase_properties(phase)
        if dependencies:
            return _get_surrogate_factory_as_property(name, self.mdg, dependencies)
        else:
            return _no_property_function


class ChemicalSystem(FluidMixin):
    def __init__(self, params: Optional[dict] = None):
        """Initialize the ChemicalSystem with empty fluid and solid."""
        super().__init__(params)
        self.species_in_phase = {}
        self.species_names = []
        self.element_names = []
        self.formula_matrix = None
        self.element_objects = []
        self.print_formula_matrix()
        reactions = self.get_reactions()
        self.set_reactions(reactions)

    def get_all_components_by_phase(self):
        """Return a dictionary of all components grouped by phase."""
        system_info = {}
        if not hasattr(self, "fluid"):
            self.create_fluid()
        species_in_phase = {}

        # Handle fluid phases
        for phase in self.fluid.phases:
            system_info[phase.name] = [comp.name for comp in phase.components]
            for comp in phase.components:
                species_in_phase[comp.name] = phase.name

        return system_info, species_in_phase

    def describe(self):
        """Prints a structured summary of the system."""
        print("Chemical System Overview:")
        print("--------------------------")
        system, _ = self.get_all_components_by_phase()
        for phase, components in system.items():
            print(f"Phase: {phase}")
            print("  Components:", ", ".join(components))

    def choose_independent_element_rows(
        self,
        A: np.ndarray,
        elements: list[str],
        *,
        max_rank: int,
        prefer_drop: tuple[str, ...] = ("Z",),
    ) -> list[str]:
        """
        Choose a 'nice' set of linearly independent rows from A (elements x species).

        Strategy:
        - Never pick rows that don't increase rank.
        - Prefer rows that cover more species (more nonzeros) and have larger variation.
        - De-prioritize rows in prefer_drop (e.g., charge 'Z') unless needed.

        Returns:
        list of chosen element names (subset of `elements`) with rank == target_rank.
        """
        assert A.shape[0] == len(elements)

        full_rank = np.linalg.matrix_rank(A)
        target_rank = min(full_rank, max_rank)

        # Build a priority score for each row: higher is better
        nnz = np.count_nonzero(A, axis=1).astype(float)
        var = np.var(A.astype(float), axis=1)
        l2 = np.linalg.norm(A.astype(float), axis=1)

        # Base score: coverage + variability + magnitude
        score = 10.0 * nnz + 1.0 * var + 0.1 * l2

        # Penalize rows we prefer to drop (e.g., Z)
        for i, el in enumerate(elements):
            if el in prefer_drop:
                score[i] -= 1e6

        # Sort candidate rows by descending score
        order = np.argsort(-score)

        chosen_idx: list[int] = []
        current_rank = 0
        for i in order:
            trial = chosen_idx + [int(i)]
            trial_rank = np.linalg.matrix_rank(A[trial, :])
            if trial_rank > current_rank:
                chosen_idx.append(int(i))
                current_rank = trial_rank
                if current_rank == target_rank:
                    break

        # If we still didn't reach target rank, allow Z etc. as fallback
        if current_rank < target_rank:
            for i in range(len(elements)):
                if i in chosen_idx:
                    continue
                trial = chosen_idx + [i]
                trial_rank = np.linalg.matrix_rank(A[trial, :])
                if trial_rank > current_rank:
                    chosen_idx.append(i)
                    current_rank = trial_rank
                    if current_rank == target_rank:
                        break

        if current_rank < target_rank:
            raise ValueError(
                f"Could not reach target rank {target_rank}. Got {current_rank}."
            )

        return [elements[i] for i in chosen_idx]




    def print_formula_matrix(self):
        """Prints the formula matrix of the system."""
        _, species_in_phase = self.get_all_components_by_phase()
        self.species_in_phase = species_in_phase  # Save for later

        # Parse each unique component name into a Species object
        species_objs = {}
        for name, comp in species_in_phase.items():
            try:
                species_objs[name] = Species.from_formula(name)
            except Exception as e:
                print(f"Skipping {name}: {e}")

        species_names = list(species_objs.keys())
        self.species_names = species_names

        # Collect all unique element symbols
        element_set = {
            chemical_symbols[atomic_number - 1] if atomic_number != 0 else "Z"
            for s in species_objs.values()
            for atomic_number in s.composition
        }
        # Create a mapping from element symbol to atomic number
        symbol_to_atomic = {s: i + 1 for i, s in enumerate(chemical_symbols)}
        symbol_to_atomic["Z"] = 0  # Special case for charge

        # Sort elements by atomic number (ascending), Z first
        elements = sorted(
            element_set, key=lambda el: symbol_to_atomic.get(el, float("inf"))
        )

        # Build initial matrix
        def build_matrix(element_list):
            matrix = []
            for elem in element_list:
                row = []
                for sp in species_names:
                    comp = species_objs[sp].composition
                    count = 0
                    for atomic_number, num in comp.items():
                        if atomic_number == 0 and elem == "Z":
                            count = num
                        elif (
                            atomic_number != 0
                            and chemical_symbols[atomic_number - 1] == elem
                        ):
                            count = num
                    row.append(count)
                matrix.append(row)
            return np.array(matrix)

        matrix = build_matrix(elements)

        working_elements = self.choose_independent_element_rows(
            matrix,
            elements,
            max_rank=len(species_names),
            prefer_drop=("Z",),   # drop charge unless necessary
        )
        # Final full-rank matrix
        matrix = build_matrix(working_elements)
        elements = working_elements
        self.element_names = working_elements
        self.formula_matrix = matrix

        # Print final matrix
        print("\nFinal Formula Matrix (full-rank):")
        header = ["Element"] + species_names
        print("\t".join(header))
        for i, elem in enumerate(elements):
            row = [elem] + [str(matrix[i, j]) for j in range(len(species_names))]
            print("\t".join(row))

        self.element_objects = [
            Element(name=el, atomic_number=(0 if el == "Z" else symbol_to_atomic[el]))
            for el in working_elements
        ]
        self.fluid.elements = self.element_objects
        self.fluid.num_elements = len(self.element_objects)
        self.fluid.element_names = working_elements

        # === Extract fluid formula matrix ===
        fluid_species = [
            name
            for name in species_names
            if self.species_in_phase[name] in [p.name for p in self.fluid.phases]
        ]
        fluid_indices = [species_names.index(name) for name in fluid_species]

        # Extract submatrix for fluid species
        fluid_formula_matrix = matrix[:, fluid_indices]

        # Assign to self.fluid
        self.fluid.fluid_formula_matrix = fluid_formula_matrix
        self.fluid.fluid_species_names = fluid_species  # Optional: for reference

    def set_kinetic_reaction_rates(
        self, reactions: Sequence[Reaction]
    ) -> Sequence[Reaction]:
        """Sets the reaction rates for kinetic reactions.

        Parameters:
            reactions: A list of Reaction objects defining the chemical reactions.
        This needs to be overridden to provide actual reaction rates.
        """

        def rr(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return pp.ad.Scalar(0.0, "synthetic_kinetic_reaction_rate")

        for reaction in reactions:
            if reaction.is_kinetic:
                reaction.reaction_rate = rr
            else:

                def rr_eq(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    return pp.ad.Scalar(0.0, "equilibrium_reaction_rate")

                reaction.reaction_rate = rr_eq

        return reactions

    def get_reactions(self) -> Sequence[pp.Reaction]:
        """Method to return a list of modelled components.

        The default implementation takes the user-provided or default fluid component
        found in the model ``params`` and returns a single component.

        Override this method via mixin to provide a more complex component context for
        the :attr:`fluid`.

        """
        # Should be available after SolutionStrategy.set_materials()
        # Getting the user-passed or default fluid component to create the default fluid
        # component.
        mc = self.params.get("material_constants", {})
        # Common: not provided yet
        if "reactions" not in mc or mc["reactions"] is None:
            return []
        reactions  = mc["reactions"]
        # Single reaction
        if isinstance(reactions, pp.Reaction):
            return [cast(pp.Reaction, reactions)]

        # Many reactions
        if isinstance(reactions, (list, tuple)):
            # Validate elements
            for r in reactions:
                if not isinstance(r, pp.Reaction):
                    raise TypeError(
                        "model.params['material_constants']['reactions'] must be a Reaction "
                        "or a Sequence[Reaction]. Found element of type "
                        f"{type(r)}."
                    )
            return cast(Sequence[pp.Reaction], list(reactions))

        raise TypeError(
            "model.params['material_constants']['reactions'] must be a Reaction, "
            "a Sequence[Reaction], or None. Found type "
            f"{type(reactions)}."
        )



    def set_reactions(self, reactions: Sequence[pp.Reaction]) -> None:
        """Sets the reactions for the chemical system and updates the fluid accordingly.

        Parameters:
            reactions: A list of Reaction objects defining the chemical reactions.
        """

        self.fluid.reactions = reactions
        self.fluid.num_reactions = len(reactions)
        self.fluid.stoichiometric_matrix = self.build_stoichiometric_matrix(reactions)

        # reactions = self.set_kinetic_reaction_rates(reactions)

        self.reactions = reactions

    _COEFF_RE = re.compile(
        r"""
        ^\s*
        (?:
            (?P<coeff> -? (?:\d+(?:\.\d*)?|\.\d+) (?:[eE][+-]?\d+)? )  # number like 2, 0.5, .5, 1e-3
            (?:\s*\*\s*)?                                             # optional '*'
        )?
        (?P<species> \S.*\S|\S )                                      # the rest (non-empty, trims later)
        \s*$
        """,
        re.VERBOSE,
    )

    def _parse_side(self, side: str, sign: int) -> Tuple[List[str], List[float]]:
        species: List[str] = []
        coeffs: List[float] = []
        # Split on ' + ' with at least one space on each side, so 'H3O+' stays intact
        tokens = re.split(r"\s+\+\s+", side.strip())
        for tok in tokens:
            if not tok:
                continue
            m = self._COEFF_RE.match(tok)
            if not m:
                raise ValueError(f"Could not parse token: {tok!r}")
            coeff_str = m.group("coeff")
            sp = m.group("species").strip()
            coeff = float(coeff_str) if coeff_str is not None else 1.0
            species.append(sp)
            coeffs.append(sign * coeff)
        return species, coeffs

    def parse_reaction(self, reaction: Reaction) -> tuple[list[str], list[float]]:
        reaction_str = reaction.formula
        """
        Parses a reaction string and returns species and their stoichiometric coefficients.
        Reactants get negative coefficients, products get positive ones.

        Example input: 'Halite = Na+ Cl-'
        Output: (['Halite', 'Na+', 'Cl-'], [-1, 1, 1])
        """
        # Split into left-hand side (reactants) and right-hand side (products)
        if "=" not in reaction_str:
            raise ValueError("Reaction must contain '=' separating LHS and RHS")

        lhs, rhs = reaction_str.split("=", 1)
        lhs_species, lhs_coeffs = self._parse_side(lhs, sign=-1)
        rhs_species, rhs_coeffs = self._parse_side(rhs, sign=+1)
        return lhs_species + rhs_species, lhs_coeffs + rhs_coeffs

    def build_stoichiometric_matrix(
        self, reactions: Sequence[pp.Reaction]
    ) -> np.ndarray:
        """
        Build a stoichiometric matrix S with shape (n_reactions, n_species).

        Columns are ordered exactly as in self.species_names.
        Rows follow the input order of `reactions`.

        Requires:
            - self.species_names: list[str]
            - self.parse_reaction(reaction): -> (species: list[str], coeffs: list[float])
            - each `reaction` has a `.formula` attribute (or is accepted by parse_reaction)

        Raises:
            ValueError if a species in any reaction is not present in self.species_names.
        """
        if not hasattr(self, "species_names") or not self.species_names:
            raise ValueError("self.species_names must be a non-empty list of species.")

        # Map species name -> column index using your specified order
        col_index = {sp: j for j, sp in enumerate(self.species_names)}

        n_rxn = len(reactions)
        n_sp = len(self.species_names)

        S = np.zeros((n_rxn, n_sp), dtype=float)

        if n_rxn == 0:
            for comp in self.fluid.components:
                comp.is_equilibrium_species = "Non-reactive"
            return S

        # Initialize flags and tracking
        is_equilibrium_species = ["Non-reactive"] * n_sp
        reaction_formulas = [""] * n_rxn
        seen_species = set()

        for i, rxn in enumerate(reactions):
            # Your own parser (reactants negative, products positive)
            reaction_formulas[i] = rxn.formula
            sp_list, coeffs = self.parse_reaction(rxn)

            # Accumulate duplicates within the same reaction
            row_acc = {}
            for sp, c in zip(sp_list, coeffs):
                if sp not in col_index:
                    # Fail fast with a helpful message
                    raise ValueError(
                        f"Species '{sp}' in reaction {i} is not in self.species_names. "
                        f"Known species: {self.species_names}"
                    )
                seen_species.add(sp)
                row_acc[sp] = row_acc.get(sp, 0.0) + float(c)

                j = col_index[sp]
                if not rxn.is_kinetic:
                    is_equilibrium_species[j] = "equilibrium"
                    self.fluid.enable_chemical_equilibrium = True

            # Write this reaction row
            for sp, c in row_acc.items():
                S[i, col_index[sp]] = c

        self.reaction_formulas = reaction_formulas

        for comp in self.fluid.components:
            if comp.name in self.species_names:
                j = col_index[comp.name]
                if comp.name in seen_species and is_equilibrium_species[j] != "equilibrium":
                    is_equilibrium_species[j] = "kinetic"
                comp.is_equilibrium_species = is_equilibrium_species[j]

        return S


class ActivityModels(pp.PorePyModel):
    def universal_gas_constant():
        """
        Returns the universal gas constant in J/(mol*K).
        """
        return 8.3144621

    def water_molar_mass(self):
        "The molar mass of water (in kg/mol)"
        return 0.01801528

    def activity_model(
        self, component: pp.Component, phase: pp.Phase
    ) -> ExtendedDomainFunctionType:
        """
        Ideal activity model for species in a solution.

        Returns
        -------
        activity : float
            Ln of the activity of the species in the solution.

        ln a_i = ln \gamma_i + ln m_i, where ln \gamma_i = 0 for ideal solution,
        m_i is the molality of the species.
        For minerals, the activity is 1.0.
        """
        gamma = pp.ad.Scalar(1.0, "activity_coefficient_ideal")
        water_mole_mass = pp.ad.Scalar(self.water_molar_mass())
        if self.fluid.num_fluid_phases > 1:
            raise NotImplementedError("Multiphase flow not implemented yet.")
        if phase.name == "aqueous":

            def activity(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                if component.name == "H2O":
                    # Special case for water, molality is defined as mole fraction in the aqueous phase.
                    molality = phase.partial_fraction_of[component](domains)
                else:
                    for comp in self.fluid.components:
                        if comp.name == "H2O":
                            water_fraction = phase.partial_fraction_of[comp](domains)

                    molality = phase.partial_fraction_of[component](domains) / (
                        water_fraction * water_mole_mass
                    )
                # op = pp.ad.log(molality)
                op = gamma * molality
                op.set_name(f"activity_ideal_{component.name}_in_{phase.name}")
                return op

            return activity
        else:
            # For minerals, the activity is 1.0
            def activity(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                op = pp.ad.Scalar(1.0, "activity_mineral")
                return op

            return activity


class ReactionRatesKineticArrhenius:
    def set_kinetic_reaction_rates(
        self, reactions: Sequence[pp.Reaction]
    ) -> Sequence[pp.Reaction]:
        """Sets the reaction rates for kinetic reactions.

        Parameters:
            reactions: A list of Reaction objects defining the chemical reactions.
        This needs to be overridden to provide actual reaction rates.
        """
        S = self.fluid.stoichiometric_matrix
        reaction_formulas = self.reaction_formulas
        for reaction in reactions:
            if reaction.is_kinetic:
                K = self.equilibrium_constant(reaction)
                k_0 = 1000.0  # unit: mol/m2/s
                Abar = 6.0  # unit: m2/m3
                rxn_index = reaction_formulas.index(reaction.formula)

                nu = S[rxn_index, :]
                reactive_species = []
                reactive_coeffs = []
                for comp in self.fluid.components:
                    if comp.name in self.species_names:
                        sp_index = self.species_names.index(comp.name)
                        if nu[sp_index] != 0:
                            # Build subarrays for reactive species and their coefficients in this reaction
                            reactive_species.append(comp)
                            reactive_coeffs.append(nu[sp_index])
                reactive_activities = {}
                # finding the activities of the reactive species
                for phase in self.fluid.phases:
                    if phase.state == PhysicalState.solid:
                        mineral_count = 0
                    for comp in reactive_species:
                        if comp in phase.components:
                            if phase.state == PhysicalState.solid:
                                mineral_count += 1
                            reactive_activities[comp.name] = phase.activity_of[comp]
                if mineral_count > 1:
                    raise NotImplementedError(
                        "Multiple minerals in one reaction not implemented yet."
                    )

                def rr(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    # get the corrsponding stoichiometric coefficients

                    # calculated the ion activity product
                    # lnQ = pp.ad.sum_operator_list(
                    #     [
                    #         reactive_activities[comp.name](domains)
                    #         * pp.ad.Scalar(reactive_coeffs[i])
                    #         for i, comp in enumerate(reactive_species)
                    #     ]
                    # )
                    # Q = pp.ad.exp(lnQ)
                    for i, comp in enumerate(reactive_species):
                        if i == 0:
                            Q = pp.ad.Scalar(1.0)
                        Q = Q * reactive_activities[comp.name](domains) ** pp.ad.Scalar(
                            reactive_coeffs[i]
                        )
                    Omega = Q / pp.ad.Scalar(K)
                    for comp in reactive_species:
                        if comp in self.fluid.solid_components:
                            A = comp.mineral_saturation(domains) * pp.ad.Scalar(
                                self.solid.total_porosity * Abar
                            )
                    r = pp.ad.Scalar(k_0) * A * (pp.ad.Scalar(1.0) - Omega)
                    return r

                reaction.reaction_rate = rr
            else:

                def rr_eq(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    return pp.ad.Scalar(0.0, "equilibrium_reaction_rate")

                reaction.reaction_rate = rr_eq

        return reactions

    def equilibrium_constant(self, reaction: pp.Reaction):
        return 2.5e6
        # set a fake equilibrium constant




class ReactionRatesKineticFirstOrder:
    def set_kinetic_reaction_rates(
        self, reactions: Sequence[pp.Reaction]
    ) -> Sequence[pp.Reaction]:
        """Sets the reaction rates for kinetic reactions.

        Parameters:
            reactions: A list of Reaction objects defining the chemical reactions.
        This needs to be overridden to provide actual reaction rates.
        """
        S = self.fluid.stoichiometric_matrix
        reaction_formulas = self.reaction_formulas
        for reaction in reactions:
            if reaction.is_kinetic:
                k_0 = self.reaction_constant()
                rxn_index = reaction_formulas.index(reaction.formula)

                nu = S[rxn_index, :]
                reactive_species = []
                reactive_coeffs = []
                for comp in self.fluid.components:
                    if comp.name in self.species_names:
                        sp_index = self.species_names.index(comp.name)
                        if nu[sp_index] != 0:
                            # Build subarrays for reactive species and their coefficients in this reaction
                            reactive_species.append(comp)
                            reactive_coeffs.append(nu[sp_index])
                # finding the activities of the reactive species
                for phase in self.fluid.phases:
                    if phase.state == PhysicalState.solid:
                        mineral_count = 0
                    for comp in reactive_species:
                        if comp in phase.components:
                            if phase.state == PhysicalState.solid:
                                mineral_count += 1
                                if mineral_count > 1:
                                    raise NotImplementedError(
                                        "Multiple minerals in one reaction not implemented yet."
                                    )

                def rr(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    for comp in reactive_species:
                        if comp in self.fluid.components:
                            if comp.name=="CO2":
                                concentration_CO2=self.total_molar_concentration(domains)*comp.fraction(domains)
                    r = pp.ad.Scalar(k_0) * concentration_CO2
                    return r

                reaction.reaction_rate = rr
            else:

                def rr_eq(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    return pp.ad.Scalar(0.0, "equilibrium_reaction_rate")

                reaction.reaction_rate = rr_eq

        return reactions

    def reaction_constant(self):
        return 1.0
    
    def C_exact(self,x, t, C_in, u, k):
        """Analytical CO2 solution for the constant-inflow benchmark."""
        x = np.asarray(x)
        C = np.zeros_like(x, dtype=float)
        front = u * t
        mask = x <= front
        C[mask] = C_in * np.exp(-k * x[mask] / u)
        return C

    def D_exact(self,x, t, C_in, u, k):
        """Analytical H2CO3 solution for the constant-inflow benchmark."""
        x = np.asarray(x)
        D = np.zeros_like(x, dtype=float)
        front = u * t
        mask = x <= front
        D[mask] = C_in * (1.0 - np.exp(-k * x[mask] / u))
        return D

    def C_exact_gaussian(self,x, t, u, L, alpha, x0,k):
        """
        Exact solution of C_t + u C_x = 0 with periodic BC and Gaussian IC.
        x: array of cell centers
        t: scalar time
        """
        x = np.asarray(x)
        # shift backward by ut and wrap into [0,L)
        x_shift = (x - u * t) % L
        return np.exp(-alpha * (x_shift - x0)**2) * np.exp(-k*t)

    def C_exact_pulse(self,x, t, u, a, b, C0):
        x = np.asarray(x)
        left = a + u * t
        right = b + u * t
        vals = np.zeros_like(x)
        mask = (x >= left) & (x <= right)
        vals[mask] = C0
        return vals

    def pulse_cell_average(self,x_faces, t, u, a, b, C0):
        # x_faces: array of length nx+1
        left = a + u*t
        right = b + u*t

        xL = x_faces[:-1]
        xR = x_faces[1:]
        overlap = np.maximum(0.0, np.minimum(xR, right) - np.maximum(xL, left))
        return C0 * overlap / (xR - xL)



class ModifiedSourceAsWells:
    def fluid_source(self,sd:list[pp.Grid])-> pp.ad.Operator:
        """Source term for fluid mass balance equation.

        Parameters:
            sd: List of subdomain grids in the mixed-dimensional grid.

        Returns:
            An operator representing the fluid source term.
        """
        # Create a source term for each subdomain
        internal_sources: pp.ad.Operator = super().fluid_source(sd)

        external_sources = pp.ad.sum_operator_list(
            [
                self.element_injection_source(element, sd)
                for element in self.fluid.elements
            ],
            "fluid_injection_sources",
        )

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        # Combine the source terms into a single operator
        return source


    def component_source(
        self, component: pp.Component, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Source term in a component's mass balance equation.

        Analogous to
        :meth:`~porepy.models.fluid_mass_balance.FluidMassBalanceEquations.fluid_source`
        , but using :meth:`interface_component_flux` and :meth:`well_component_flux` to
        obtain the correct component flux accross
        interfaces.

        Parameters:
            component: A component in the :attr:`fluid`.
            subdomains: A list of subdomains in the :attr:`mdg`.

        Returns:
            The base method returns the sources corresponding to the fluxes in the
            mixed-dimensional setting.

        """
        # Interdimensional fluxes manifest as source terms in lower-dimensional
        # subdomains.
        internal_sources: pp.ad.Operator = super().component_source(component, subdomains)
        external_sources = self.component_injection_source(component, subdomains)
        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("component_source_" + component.name)

        # Combine the source terms into a single operator
        return source

    def component_injection_source(
        self, component: pp.Component, subdomains: list[pp.Grid]
    )  -> np.ndarray|pp.ad.Operator:
        injection_terms = []
        for subdomain in subdomains:
            # Define the source term for the subdomain
            if subdomain.dim == self.mdg.dim_max():
                if component in self.fluid.solid_components:
                    injection_terms.append(np.zeros(subdomain.num_cells, dtype=np.float64))
                else:
                    values=np.zeros(subdomain.num_cells, dtype=np.float64)
                    # Define injection and production cells
                    injection_mask, production_mask = self.well_masks_from_coordinates(subdomain)
                    injected_concentration = self.ic_values_species_concentration(component, subdomain) / self.solid.total_porosity
                    values[injection_mask] = self.injection_and_production_rates("injection") * injected_concentration[injection_mask]  # Injection rate (positive)
                    injection_terms.append(values)
            else:
                injection_terms.append(np.zeros(subdomain.num_cells, dtype=np.float64))

        external_sources = pp.wrap_as_dense_ad_array(np.hstack(injection_terms))
        external_sources.set_name(f"component_source_{component.name}_injection_and_production_wells")
        
        #compute production source term

        production_source_terms= pp.ad.sum_operator_list([
            phase.molar_density(subdomains) * phase.saturation(subdomains) * phase.partial_fraction_of[component](subdomains)
            for phase in self.fluid.phases if component in phase.components
        ])
        mask_inj,mask_prod = self.global_well_masks(subdomains)
        mask_prod_ad = pp.wrap_as_dense_ad_array(mask_prod.astype(float))
        production_source_terms = mask_prod_ad * pp.ad.Scalar(self.injection_and_production_rates("production"))* production_source_terms
        external_sources += production_source_terms
        return external_sources


    #constant production concentration
    # def component_injection_source(
    #     self, component: pp.Component, subdomains: list[pp.Grid]
    # )  -> np.ndarray|pp.ad.Operator:
    #     injection_terms = []
    #     for subdomain in subdomains:
    #         # Define the source term for the subdomain
    #         if subdomain.dim == self.mdg.dim_max():
    #             if component in self.fluid.solid_components:
    #                 injection_terms.append(np.zeros(subdomain.num_cells, dtype=np.float64))
    #             else:
    #                 values=np.zeros(subdomain.num_cells, dtype=np.float64)
    #                 # Define injection and production cells
    #                 injection_mask, production_mask = self.well_masks_from_coordinates(subdomain)
    #                 injected_concentration = self.ic_values_species_concentration(component, subdomain) / self.solid.total_porosity
    #                 values[injection_mask] = self.injection_and_production_rates("injection") * injected_concentration[injection_mask]  # Injection rate (positive)
    #                 values[production_mask] = self.injection_and_production_rates("production")*injected_concentration[production_mask]  # Production rate (negative)
    #                 injection_terms.append(values)
    #         else:
    #             injection_terms.append(np.zeros(subdomain.num_cells, dtype=np.float64))

    #     external_sources = pp.wrap_as_dense_ad_array(np.hstack(injection_terms))
    #     external_sources.set_name(f"component_source_{component.name}_injection_and_production_wells")
        
    #     #compute production source term


    #     return external_sources




    def element_source(
        self, element: pp.Element, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Source term in an element's mass balance equation.

        Analogous to
        :meth:`~porepy.models.fluid_mass_balance.FluidMassBalanceEquations.fluid_source`
        , but using :meth:`interface_element_flux` and :meth:`well_element_flux` to
        obtain the correct element flux accross
        interfaces.

        Parameters:
            element: An element in the :attr:`fluid`.
            subdomains: A list of subdomains in the :attr:`mdg`.

        Returns:
            The base method returns the sources corresponding to the fluxes in the
            mixed-dimensional setting.

        """
        # Interdimensional fluxes manifest as source terms in lower-dimensional
        # subdomains.
        internal_sources: pp.ad.Operator = super().element_source(element, subdomains)

        external_sources = self.element_injection_source(element, subdomains)

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("element_source_" + element.name)

        # Combine the source terms into a single operator
        return source


    def element_injection_source(
        self, element: pp.Element, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        W = self.fluid.fluid_formula_matrix  # shape (E, C)
        species_names = self.fluid.fluid_species_names
        components = self.fluid.components

        # Map species name -> AD function
        z_funcs = {
            comp.name: self.component_injection_source(comp, subdomains)
            for comp in components
        }

        # Evaluate z_ξ(subdomains) to get a list of Operators
        try:
            z_ops = [z_funcs[name] for name in species_names]  # shape (C,)
        except KeyError as e:
            raise KeyError(
                f"Species name '{e.args[0]}' not found in fluid components."
            )

        # Extract row for the element
        # find the row index according to element name
        try:
            element_index = self.fluid.element_names.index(element.name)
        except ValueError as e:
            raise ValueError(
                f"Element name '{e.args[0]}' not found in fluid elements."
            )
        W_row = W[element_index, :]  # shape (C,)
        # Compute ∑_{e} ∑_{ξ} W[e, ξ] * z_ξ
        external_sources = pp.ad.sum_operator_list(
            [pp.ad.Scalar(w) * z for w, z in zip(W_row, z_ops)]
        )

        external_sources.set_name(f"element_injection_source_{element.name}")
        return external_sources


    def injection_and_production_rates(self,well_type:str):
        """Define injection and production rates for wells.
        unit: m3/s
        Parameters:
            well_type: A string indicating the type of well ("injection" or "production").

        Returns:
            A dictionary containing the injection and production rates for the wells.
        """
        if well_type == "injection":
            return self.params.get("injection_rate", 1e-14)  # Injection rate (positive)
        elif well_type == "production":
            return self.params.get("production_rate", -1e-14)  # Production rate (negative)
        else:
            raise ValueError("Invalid well type. Must be 'injection' or 'production'.")



    def well_region_masks(
        self,
        subdomain: pp.Grid,
        depth_range=(1000.0, 1500.0),
        north_inj_range=(500.0,1000.0),
        north_prod_range=(1500.0,2000.0),
        east_range=(1000.0, 1500.0),
    ):
        """
        Return injection and production boolean masks based on geometric filters.

        Coordinates convention:

        """

        cc = subdomain.cell_centers
        
        east = cc[0, :]
        north = cc[1, :]
        depth = cc[2, :]

        zmin, zmax = depth_range
        emin, emax = east_range

        deep_mask = (depth >= zmin) & (depth <= zmax)
        east_mask = (east >= emin) & (east <= emax)
        injection_north_mask=(north >= north_inj_range[0]) & (north <= north_inj_range[1])
        production_north_mask=(north >= north_prod_range[0]) & (north <= north_prod_range[1])

        injection_mask = (
            deep_mask
            & injection_north_mask
            & east_mask
        )

        production_mask = (
            deep_mask
            & production_north_mask
            & east_mask
        )

        return injection_mask, production_mask


    def global_well_masks(
        self,
        subdomains: list[pp.Grid],
    ):
        """
        Build global injection/production masks over a list of subdomains.
        - For grids with dim == self.mdg.dim_max(): use geometric masks.
        - For all other grids: all False.

        Returns:
            inj_global: np.ndarray[bool] of length sum(sd.num_cells)
            prod_global: np.ndarray[bool] of length sum(sd.num_cells)
        """
        # inj_parts = []
        # prod_parts = []

        # dim_max = self.mdg.dim_max()

        # for sd in subdomains:
        #     if sd.dim == dim_max:
        #         inj, prod = self.well_region_masks(
        #             sd,
        #         )
        #     else:
        #         inj = np.zeros(sd.num_cells, dtype=bool)
        #         prod = np.zeros(sd.num_cells, dtype=bool)

        #     inj_parts.append(inj)
        #     prod_parts.append(prod)

        # inj_global = np.hstack(inj_parts) if len(inj_parts) else np.array([], dtype=bool)
        # prod_global = np.hstack(prod_parts) if len(prod_parts) else np.array([], dtype=bool)

        # return inj_global, prod_global
        return self.global_well_masks_from_coordinates(
            subdomains,)


    def energy_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermal source."""

        # Internal sources are inherited from parent class.
        # Zero internal sources for unfractured domains, but we keep it to maintain
        # good code practice
        internal_sources: pp.ad.Operator = super().energy_source(subdomains)


        external_sources = pp.ad.Scalar(0.0)
        #compute production source term

        production_source_terms= pp.ad.sum_operator_list([
            phase.density(subdomains) * phase.specific_enthalpy(subdomains)
            for phase in self.fluid.phases if phase.state != PhysicalState.solid
        ])
        mask_inj,mask_prod = self.global_well_masks(subdomains)
        mask_prod_ad = pp.wrap_as_dense_ad_array(mask_prod.astype(float))
        production_source_terms = mask_prod_ad * pp.ad.Scalar(self.injection_and_production_rates("production"))* production_source_terms
        mask_inj_ad= pp.wrap_as_dense_ad_array(mask_inj.astype(float))
        injected_temperature=self.params.get("injected_temperature", 300)
        injected_enthalpy= self.fluid.reference_component.specific_heat_capacity * (injected_temperature - self.reference_variable_values.temperature)
        injection_source_terms = mask_inj_ad * pp.ad.Scalar(self.injection_and_production_rates("injection")*injected_enthalpy*self.fluid.reference_component.density) 
        external_sources += injection_source_terms
        external_sources += production_source_terms



        # Add up contribution of internal and external sources of energy.
        thermal_sources = internal_sources + external_sources
        thermal_sources.set_name("Energy source")

        return thermal_sources

    def nearest_cell_mask(self,subdomain: pp.Grid, xyz: tuple[float, float, float]) -> np.ndarray:
        """
        Return a boolean mask with exactly one True: the cell whose center is
        closest (Euclidean) to xyz = (north, east, depth).

        If xyz lies outside the min/max bounds, this still picks the nearest boundary cell.
        """
        cc = subdomain.cell_centers  # shape (3, num_cells)
        target = np.array(xyz, dtype=float).reshape(3, 1)

        # Squared Euclidean distance per cell
        d2 = np.sum((cc - target) ** 2, axis=0)
        
        min_val=np.min(d2)
        candidates= np.where(d2 == min_val)[0]
        idx = candidates[0]  # In case of ties, pick the first one. This is deterministic and consistent with the "nearest" definition.


        mask = np.zeros(subdomain.num_cells, dtype=bool)
        mask[idx] = True
        return mask


    def well_masks_from_coordinates(
        self,
        subdomain: pp.Grid,
        inj_xyz: tuple[float, float, float] | None = None,
        prod_xyz: tuple[float, float, float] | None = None,

    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (injection_mask, production_mask) where each mask has exactly one True.
        """
        # domain_sizes = self.params.get("domain_sizes", None)
        # if domain_sizes is None:
        #     raise KeyError(
        #         "Missing parameter 'domain_sizes' in self.params. "
        #         "Set self.params['domain_sizes'] = [Lx, Ly, Lz]."
        #     )
        domain=self._domain

        Lx, Ly, Lz = domain.side_lengths()

        # Auto coordinates only if not provided
        if inj_xyz is None:
            inj_xyz = (0.5 * Lx, (1.0 / 3.0) * Ly, 0.5 * Lz)
        if prod_xyz is None:
            prod_xyz = (0.5 * Lx, (2.0 / 3.0) * Ly, 0.5 * Lz)

        inj_mask = self.nearest_cell_mask(subdomain, inj_xyz)
        prod_mask = self.nearest_cell_mask(subdomain, prod_xyz)

        if np.any(inj_mask & prod_mask):
            raise ValueError("Injection and production snapped to the same cell. Move wells apart.")

        return inj_mask, prod_mask

    def global_well_masks_from_coordinates(
        self,
        subdomains: list[pp.Grid],
    ) -> tuple[np.ndarray, np.ndarray]:
        inj_parts, prod_parts = [], []
        dim_max = self.mdg.dim_max()

        for sd in subdomains:
            if sd.dim == dim_max:
                inj, prod = self.well_masks_from_coordinates(sd)
            else:
                inj = np.zeros(sd.num_cells, dtype=bool)
                prod = np.zeros(sd.num_cells, dtype=bool)
            inj_parts.append(inj)
            prod_parts.append(prod)

        return np.hstack(inj_parts), np.hstack(prod_parts)
