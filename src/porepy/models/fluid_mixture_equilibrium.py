"""A module containing equilibrium formulations for fluid mixtures using PorePy's AD
framework.

"""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Sequence

import porepy as pp
import porepy.composite as ppc

from ..composite.composite_utils import safe_sum

__all__ = [
    "evaluate_homogenous_constraint",
    "MixtureMixin",
    "EquilibriumMixin",
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

    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Usually defined in a mixin class
    defining the geometry."""

    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]

    fluid_mixture: ppc.Mixture
    """The fluid mixture set by this class during :meth:`set_mixture`."""

    eliminate_reference_phase: bool
    """A flag indicating whether the molar phase fraction and saturation of the
    reference phase should be eliminated by unity or not. Usually defined in a mixin
    class for the solution strategy."""

    eliminate_reference_component: bool
    """A flag indicating whether the overall fraction of the reference component should
    be eliminated by unity. Also, if eliminated, the local mass constraint for the
    reference component is not constructed. Usually defined in the mixin class
    for the solution strategy."""

    def set_mixture(self) -> None:
        subdomains = self.mdg.subdomains()
        p = self.pressure(subdomains)
        T = self.temperature(subdomains)

        components = self.get_components()
        phases: list[ppc.Phase] = list()
        for config in self.get_phase_configuration():
            eos, type_, name = config
            phases.append(ppc.Phase(eos, type_, name))

        self.set_components_in_phases(components, phases)

        self.fluid_mixture = ppc.Mixture(components, phases)
        self.fluid_mixture.set_up_ad(
            self.equation_system,
            subdomains,
            p,
            T,
            self.eliminate_reference_phase,
            self.eliminate_reference_component,
        )

    def get_components(self) -> Sequence[ppc.Component]:
        """Method to return a list of modelled components.

        Raises:
            NotImplementedError: If not overwritten to return a list of components.

        """
        raise NotImplementedError("No components defined in mixture mixin.")

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.AbstractEoS, int, str]]:
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
        self, components: Sequence[ppc.Component], phases: Sequence[ppc.Phase]
    ) -> None:
        """Method to implement a strategy for which components are added to which phase.

        By default, the unified assumption is applied: All phases contain all
        components.

        Overwrite to do otherwise.

        Parameters:
            components: The list of components modelled by :meth:`get_components`.
            phases: The list of phases modelled by :meth:`get_phases`.

        """
        for phase in phases:
            phase.components = components


class EquilibriumMixin:
    """Basic class introducing the fluid phase equilibrium equations into the model."""

    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Usually defined in a mixin class
    defining the geometry."""

    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy."""

    fluid_mixture: ppc.Mixture
    """A mixture containing all modelled phases and components, and required fluid
    properties as a combination of phase properties. Usually defined in a mixin class
    defining the mixture."""

    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    volume: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]

    equilibrium_type: Literal["p-T", "p-h", "v-h"]
    """A string denoting the two state functions which are assumed constant at
    equilibrium. Usually defined in the solution strategy."""

    eliminate_reference_component: bool
    """A flag indicating whether the overall fraction of the reference component should
    be eliminated by unity. Also, if eliminated, the local mass constraint for the
    reference component is not constructed. Usually defined in the mixin class
    for the solution strategy."""

    eliminate_reference_phase: bool
    """A flag indicating whether the molar phase fraction and saturation of the
    reference phase should be eliminated by unity or not. Usually defined in a mixin
    class for the solution strategy."""

    use_semismooth_complementarity: bool
    """Flag indicating whether the complementarity conditions for each phase should be
    assembled in semi-smooth form using a :math:`\\min` operator. Usually defined
    in the solution strategy."""

    normalize_state_constraints: bool
    """A flag indicating whether the (local) equilibrium equations representing
    constraints of some state function should be normalized with the target value.
    Usually defined in the solution strategy."""

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

            self.set_density_relations_for_phases()

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

    def mass_constraint_for_component(self, component: ppc.Component) -> pp.ad.Operator:
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

    def complementarity_condition_for_phase(self, phase: ppc.Phase) -> pp.ad.Operator:
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
        self, component: ppc.Component, phase: ppc.Phase
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

    def density_relation_for_phase(self, phase: ppc.Phase) -> pp.ad.Operator:
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
