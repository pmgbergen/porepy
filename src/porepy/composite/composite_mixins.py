"""A module containing equilibrium formulations for fluid mixtures using PorePy's AD
framework.

"""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np

import porepy as pp

from .base import AbstractEoS, Component, Mixture, Phase
from .composite_utils import safe_sum
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
    """Mixin class for modelling a mixture."""

    fluid_mixture: Mixture
    """The fluid mixture set by this class during :meth:`set_mixture`."""

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

    def set_mixture(self) -> None:
        subdomains = self.mdg.subdomains()
        p = self.pressure(subdomains)
        T = self.temperature(subdomains)

        components = self.get_components()
        phases: list[Phase] = list()
        for config in self.get_phase_configuration():
            eos, type_, name = config
            phases.append(Phase(eos, type_, name))

        self.set_components_in_phases(components, phases)

        self.fluid_mixture = Mixture(components, phases)
        self.fluid_mixture.set_up_ad(
            self.equation_system,
            subdomains,
            p,
            T,
            self.eliminate_reference_phase,
            self.eliminate_reference_component,
        )

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

    def get_secondary_equation_names(self) -> list[str]:
        """Returns a list of secondary equations introduced by this mixin.

        The base method returns the names of density relations, since they are included
        by default.

        Important:
            Override this method and append names of additionally included, secondary
            equations in :meth:`set_secondary_equations`.

        """
        names: list[str] = list()
        rphase = self.fluid_mixture.reference_phase
        # names of density relations
        for phase in self.fluid_mixture.phases:
            if phase == rphase and self.eliminate_reference_phase:
                continue
            names.append(f"density-relation-{phase.name}")
        return names

    def set_secondary_equations(self) -> None:
        """Override this method to set secondary expressions in equation form

        .. math::

            f(x) = 0

        by setting the left-hand side as an equation in the Ad framework.

        """

    def set_density_relations_for_phases(self) -> None:
        """Introduced the mass relations for phases into the AD system.

        This method is separated, because it has another meaning when coupling the
        equilibrium problem with flow and transport.

        In multiphase flow in porous media, saturations must always be provided.
        Hense even if there are no isochoric specifications in the flash, the model
        necessarily introduced the saturations as unknowns.

        The mass relations per phase close the system, by relating molar phase fractions
        to saturations. Hence rendering the system solvable.

        Important:
            Isochoric equilibrium specifiation include these relations by default.
            Hence do not call this method in that case, otherwise the equations are
            introduced twice.

        """
        rphase = self.fluid_mixture.reference_phase
        subdomains = self.mdg.subdomains()
        for phase in self.fluid_mixture.phases:
            if phase == rphase and self.eliminate_reference_phase:
                continue
            equ = self.density_relation_for_phase(phase)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})

    def density_relation_for_phase(self, phase: Phase) -> pp.ad.Operator:
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

        Returns:
            The left-hand side of above equations.

            If normalization of state constraints is set in the solution strategy,
            it returns the normalized form.

        """
        if self.normalize_state_constraints:
            equ = (
                phase.fraction
                - phase.saturation * phase.density / self.fluid_mixture.density
            )
        else:
            equ = (
                phase.fraction * self.fluid_mixture.density
                - phase.saturation * phase.density
            )
        equ.set_name(f"density-relation-{phase.name}")
        return equ


class EquilibriumEquationsMixin:
    """Basic class introducing the fluid phase equilibrium equations into the model."""

    create_compositional_fractions: bool
    """**IMPORTANT**

    A flag indicates whether (extended) fractions of components in phases should
    be created or not. This must be True, if :attr:`equilibrium_type` is not None.

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
           :attr:`create_compositional_fractions` is True.

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
    volume: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """TODO Not covered so far."""

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

    def get_equilibrium_equation_names(self) -> list[str]:
        """Get a list of equation names introduced into the AD framework by this class
        depending on :attr:`equilibrium_type`.

        Returns an empty list if ``equilibrium_type==None``.

        """
        names: list[str] = list()
        if self.equilibrium_type is not None:
            # local mass conservation names
            for component in self.fluid_mixture.components:
                # skip for reference component if eliminated
                if (
                    component == self.fluid_mixture.reference_component
                    and self.eliminate_reference_component
                ):
                    continue

                names.append(f"mass-constraint-{component.name}")

            rphase = self.fluid_mixture.reference_phase
            for phase in self.fluid_mixture.phases:
                # complementary conditions
                if self.use_semismooth_complementarity:
                    names.append(f"semismooth-complementary-condition-{phase.name}")
                else:
                    names.append(f"complementary-condition-{phase.name}")

                # isofugacity constraints
                if phase == rphase:
                    continue

                for component in self.fluid_mixture.components:
                    if not (
                        component in phase.components and component in rphase.components
                    ):
                        continue

                    names.append(
                        f"isofugacity-constraint-"
                        + f"{component.name}-{phase.name}-{rphase.name}"
                    )

            # equatiosn constraining the thermodynamic state
            if self.equilibrium_type == "p-h":
                names.append("mixture-enthalpy-constraint")
            if self.equilibrium_type == "v-h":
                names.append("mixture-enthalpy-constraint")
                names.append("mixture-volume-constraint")

        return names

    def set_equations(self) -> None:
        """Introduced the local equilibrium equations into the AD framework.

        All equations are scalar, single, cell-wise equations on each subdomains.

        """
        subdomains = self.mdg.subdomains()
        ## starting with equations common to all equilibrium definitions
        # local mass constraint per component
        for component in self.fluid_mixture.components:
            # skip for reference component if eliminated
            if (
                component == self.fluid_mixture.reference_component
                and self.eliminate_reference_component
            ):
                continue
            equ = self.mass_constraint_for_component(component)
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
                    component, phase
                )
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # complementarity conditions
        for phase in self.fluid_mixture.phases:
            equ = self.complementarity_condition_for_phase(phase)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # No more equations for p-T based flash
        if self.equilibrium_type == "p-T":
            pass
        # 1 more equation for p-h based flash (T unknown)
        elif self.equilibrium_type == "p-h":
            # here T is another unknown, but h is fixed. Introduce 1 more equations
            h = self.enthalpy(subdomains)
            equ = self.mixture_enthalpy_constraint(h)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})
        # 2 + num_phase - 1 more equations for v-h flash (p, T, s_j unknown)
        elif self.equilibrium_type == "v-h":
            h = self.enthalpy(subdomains)
            v = self.volume(subdomains)
            equ = self.mixture_enthalpy_constraint(h)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})
            equ = self.mixture_volume_constraint(v)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})

    def mass_constraint_for_component(self, component: Component) -> pp.ad.Operator:
        """Constructs the local mass constraint for a component :math:`i`.

        .. math::

            z_i - \\sum_j x_{ij} y_j = 0.

        - :math:`z` : Component :attr:`~porepy.composite.base.Component.fraction`
        - :math:`y` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`x` : Phase :attr:`~porepy.composite.base.Phase.fraction_of` component

        The above sum is performed over all phases the component is present in.

        Parameter:
            component: The component represented by the overall fraction :math:`z_i`

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
            component.fraction,
            [phase.fraction_of[component] for phase in phases],
            [phase.fraction for phase in phases],
        )  # type:ignore
        equ.set_name(f"mass-constraint-{component.name}")
        return equ

    def complementarity_condition_for_phase(self, phase: Phase) -> pp.ad.Operator:
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

        Returns:
            The left-hand side of above equation. If the semi-smooth form is
            requested by the solution strategy, then the :math:`\\min\\{\\}` operator is
            used.

        """

        unity: pp.ad.Operator = 1.0 - pp.ad.sum_operator_list(
            [phase.fraction_of[comp] for comp in phase.components]
        )

        if self.use_semismooth_complementarity:
            equ = pp.ad.SemiSmoothMin(phase.fraction, unity)
            equ.set_name(f"semismooth-complementary-condition-{phase.name}")
        else:
            equ = phase.fraction * unity
            equ.set_name(f"complementary-condition-{phase.name}")
        return equ

    def isofugacity_constraint_for_component_in_phase(
        self, component: Component, phase: Phase
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

        equ = (
            phase.fraction_of[component] * phase.fugacity_of[component]
            - self.fluid_mixture.reference_phase.fraction_of[component]
            * self.fluid_mixture.reference_phase.fugacity_of[component]
        )
        equ.set_name(
            f"isofugacity-constraint-"
            + f"{component.name}-{phase.name}-{self.fluid_mixture.reference_phase.name}"
        )
        return equ

    def mixture_enthalpy_constraint(self, h: pp.ad.Operator) -> pp.ad.Operator:
        """Constructs the enthalpy constraint for the mixture enthalpy and the
        transported enthalpy variable.

        .. math::

            \\sum_j y_j h_j  - h = 0~,~
            (\\sum_j y_j h_j) / h - 1= 0~

        - :math:`y_j` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`h_j` : Phase :attr:`~porepy.composite.base.Phase.enthalpy`

        The first term represents the mixture enthalpy based on the thermodynamic state.
        The second term represents the target enthalpy in the equilibrium problem.
        The target enthalpy is a transportable quantity in flow and transport.

        Parameters:
            h: An operator representing the target enthalpy.

        Returns:
            The left-hand side of above equations. If the normalization of state
            constraints is required by the solution strategy, the second form is
            returned.

        """
        if self.normalize_state_constraints:
            equ: pp.ad.Operator = self.fluid_mixture.enthalpy / h - 1.0
        else:
            equ: pp.ad.Operator = self.fluid_mixture.enthalpy - h
        equ.set_name("mixture-enthalpy-constraint")
        return equ

    def mixture_volume_constraint(self, v: pp.ad.Operator) -> pp.ad.Operator:
        """Constructs the volume constraint using the reciprocal of the mixture density.

        .. math::

            \\dfrac{1}{\\sum_j s_j \\rho_j} - v = 0~,~
            v \\left(\\sum_j s_j \\rho_j\\right) - 1 = 0.

        - :math:`s_j` : Phase :attr:`~porepy.composite.base.Phase.saturation`
        - :math:`\\rho_j` : Phase :attr:`~porepy.composite.base.Phase.density`

        Parameters:
            v: An operator representing the target mixture volume :math:`v`.

        Returns:
            The left-hand side of above equations. If the normalization of state
            constraints is required by the solution strategy, the second form is
            returned.

        """
        if self.normalize_state_constraints:
            equ: pp.ad.Operator = v * self.fluid_mixture.density - 1.0
        else:
            equ: pp.ad.Operator = self.fluid_mixture.density ** (-1) - v
        equ.set_name("mixture-volume-constraint")
        return equ

    def fractional_state_from_vector(
        self,
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
                component.fraction.value(self.equation_system, state)
                for component in self.fluid_mixture.components
            ]
        )

        y = np.array(
            [
                phase.fraction.value(self.equation_system, state)
                for phase in self.fluid_mixture.phases
            ]
        )

        sat = np.array(
            [
                phase.saturation.value(self.equation_system, state)
                for phase in self.fluid_mixture.phases
            ]
        )

        x = [
            np.array(
                [
                    phase.fraction_of[component].value(self.equation_system, state)
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

    fractional_state_from_vector: Callable[[Optional[np.ndarray]], FluidState]
    """Provided by :class:`EquilibriumEquationsMixin`."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by
    :class:`~porepy.models.compositional_balance.VariablesCompositionalFlow`."""

    def set_up_flasher(self) -> None:
        """Method to introduce the flash class, if an equilibrium is defined.

        This method is called by the solution strategy after the model is set up.

        """
        if self.equilibrium_type is not None:
            raise NotImplementedError(
                f"No flash set-up implemented for {self.equilibrium_type} equilibrium."
            )

    def equilibriate_fluid(
        self, state: Optional[np.ndarray] = None
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
        # Extracting the current, iterative state to use as initial guess for the flash
        subdomains = self.mdg.subdomains()
        fluid_state = self.fractional_state_from_vector(state)
        fluid_state.p = self.pressure(subdomains).value(self.equation_system, state)
        fluid_state.T = self.temperature(subdomains).value(self.equation_system, state)
        fluid_state.h = self.enthalpy(subdomains).value(self.equation_system, state)
        fluid_state.v = self.fluid_mixture.volume.value(self.equation_system, state)

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
            # TODO change v to the right volume based on domain
            v = self.fluid_mixture.volume.value(self.equation_system, state)
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
