"""Module containing equation classes for introducing the local, unified equilibrium
problem into a PorePy model.

Local equilibrium equations are single, cell-wise algebraic equations, introducing
the thermodynamically consistent approach to modelling secondary expressions like
phase densities and closing a CF model.

Instances of :class:`UnifiedEquilibriumMixin` require the ``'equilibrium_type'`` model
parameter to be *not* ``None``. This is to inform the remaining framework
that local equilibrium assumptions (instead of some constitutive laws) were introduced.

"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import porepy as pp
from porepy.compositional.utils import CompositionalModellingError
from porepy.models.abstract_equations import EquationMixin

__all__ = [
    "EnthalpyTemperatureRelation",
    "UnifiedPhaseEquilibriumEquations",
    "Unified_pT_Equilibrium",
    "Unified_ph_Equilibrium",
    "Unified_vh_Equilibrium",
]


class EnthalpyTemperatureRelation(EquationMixin):
    r"""A single, local equation for closing models with independent enthalpy and
    temperature variables.

    It relates the independent enthalpy variable to the enthalpy of the fluid

    .. math::

        \tilde{h} = \sum_j y_j h_j  - h = (\sum_j y_j h_j) / h - 1= 0~,

    where the user can pass ``params['normalize_state_constraints']`` to determine which
    form should be used.

    Additionally, a float ``params['relaxation_enthalpy_constraint']`` can be passed to
    introduce a temporal relaxation

    .. math::

        \frac{\partial}{\partial t} \tilde{h} + \frac{a}{\Delta t} \tilde{h} = 0~,

    with :math:`a` being the respective parameters.

    Note:
        Use this equation only in models where the local equilibrium is not defined in
        terms of enthalpy. Otherwise the regular enthalpy constraint should be used.

    """

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.EnthalpyVariable`."""

    def set_equations(self):
        """Introduces the (relaxed) enthalpy-temperature relation on all subdomains."""
        super().set_equations()

        subdomains = self.mdg.subdomains()

        h_mix = self.fluid.specific_enthalpy(subdomains)
        h_target = self.enthalpy(subdomains)
        if self.params.get("normalize_state_constraints", True):
            equ = h_mix / h_target - pp.ad.Scalar(1.0)
        else:
            equ = h_mix - h_target

        relaxation_parameter = self.params.get("relaxation_enthalpy_constraint", None)
        if relaxation_parameter is not None:
            equ = (
                pp.ad.dt(equ, self.ad_time_step)
                + (pp.ad.Scalar(relaxation_parameter) / self.ad_time_step) * equ
            )
            equ.set_name("relaxed_local_fluid_enthalpy_constraint")
        else:
            equ.set_name("local_fluid_enthalpy_constraint")

        self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class UnifiedPhaseEquilibriumEquations(pp.PorePyModel):
    """Base class for introducing local phase equilibrium equations into a model using
    the unified formulation.

    The base class provides means to assemble required equations, as well as a
    verification of model assumptions for the unified formulation.

    A :class:`~porepy.compositional.utils.CompositionalModellingError` will be raised
    if any of the following assumptions is violated:

    1. At least 2 components and 2 phases are modelled.
    2. The model's ``params['equilibrium_type']`` is not None and contains the keyword
       ``'unified'``.
    3. All phases have all components set in them (all extended partial fractions are
       defined and introduced).

    If the reference phase was not eliminated (dangling variables), a warning is raised.

    """

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.EnthalpyVariable`."""
    volume: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.compositional_flow.SolidSkeletonCF`."""

    @property
    def _normalize_constraints(self) -> bool:
        """Returns the flags set in model parameters indicating whether local
        constraints should be normalized (non-dimensional equations)."""
        return bool(self.params.get("normalize_state_constraints", True))

    def set_equations(self) -> None:
        """The base class method without defined equilibrium type performs a model
        validation to ensure that the assumptions for the unified flash are fulfilled.
        """
        assert isinstance(self, EquationMixin)
        super().set_equations()  # type:ignore[safe-super]

        ncomp = self.fluid.num_components
        nphase = self.fluid.num_phases

        if not pp.compositional.has_unified_equilibrium(self):
            raise CompositionalModellingError(
                "Must define a `equilibrium_type` model parameter containing the"
                + " keyword `unified` when using the Unified Equilibrium Mixin."
            )

        if nphase < 2:
            raise CompositionalModellingError(
                "Unified equilibrium models need at least to modelled phases,"
                + f" {nphase} given."
            )
        if ncomp < 2:
            raise CompositionalModellingError(
                "Unified equilibrium models require at least to components in the fluid"
                + f" mixture, {ncomp} given."
            )
        if not self._is_reference_phase_eliminated():
            warnings.warn(
                "Unified equilibrium model included, but reference phase not"
                + " eliminated. Check model closedness."
            )

        all_comps = set(self.fluid.components)
        for phase in self.fluid.phases:
            phase_comps = set(phase)
            if all_comps.symmetric_difference(phase_comps):
                raise CompositionalModellingError(
                    f"Unified equilibrium assumption violated for phase: {phase.name}."
                    + " All phases must have all components modelled in them."
                )

    def mass_constraint_for_component(
        self, component: pp.FluidComponent, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the local mass constraint for a component :math:`i`.

        .. math::

            z_i - \\sum_j x_{ij} y_j = 0.

        - :math:`z` : Component :attr:`~porepy.compositional.base.Component.fraction`
        - :math:`y` : Phase :attr:`~porepy.compositional.base.Phase.fraction`
        - :math:`x` : :attr:`~porepy.compositional.base.Phase.extended_fraction_of` the
          component in a phase.

        The above sum is performed over all phases the component is present in.

        Parameter:
            component: The component represented by the overall fraction :math:`z_i`.
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            An operator representing the left-hand side of above equation.

        """
        # get all phases the component is present in
        phases = [phase for phase in self.fluid.phases if component in phase]

        # create operators for fractions
        z_i = component.fraction(subdomains)
        y_j = [phase.fraction(subdomains) for phase in phases]
        x_ij = [phase.extended_fraction_of[component](subdomains) for phase in phases]

        equ = z_i - pp.ad.sum_operator_list([x * y for x, y in zip(x_ij, y_j)])

        equ.set_name(f"local_mass_constraint_{component.name}")
        return equ

    def complementarity_condition_for_phase(
        self, phase: pp.Phase, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the complementarity condition for a given phase.

        .. math::

            y_j (1 - \\sum_i x_{ij}) = 0~,~
            \\min \\{y_j, (1 - \\sum_i x_{ij}) \\} = 0.

        - :math:`y` : Phase :attr:`~porepy.compositional.base.Phase.fraction`
        - :math:`x` : :attr:`~porepy.compositional.base.Phase.extended_fraction_of` the
          components in the phase.

        The sum is performed over all components modelled in that phase
        (see :attr:`~porepy.compositional.base.Phase.components`).

        Parameters:
            phase: The phase for which the condition is assembled.
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equation. If the semi-smooth form is
            requested by the solution strategy, then the :math:`\\min\\{\\}` operator is
            used. The semi-smooth form can be requested via
            ``params['use_semismooth_complementarity']`` and defaults to True.

        """

        unity: pp.ad.Operator = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
            [phase.extended_fraction_of[comp](subdomains) for comp in phase]
        )

        minimum = lambda x, y: pp.ad.maximum(-x, -y)
        ssmin = pp.ad.Function(minimum, "semi-smooth-minimum")

        if self.params.get("use_semismooth_complementarity", True):
            equ = ssmin(phase.fraction(subdomains), unity)
            equ.set_name(f"semismooth_complementary_condition_{phase.name}")
        else:
            equ = phase.fraction(subdomains) * unity
            equ.set_name(f"complementary_condition_{phase.name}")
        return equ

    def isofugacity_constraint_for_component_in_phase(
        self,
        component: pp.FluidComponent,
        phase: pp.Phase,
        subdomains: Sequence[pp.Grid],
    ) -> pp.ad.Operator:
        """Construct the local isofugacity constraint for a component between a given
        phase and the reference phase.

        .. math::

            x_{ij} \\varphi_{ij} - x_{iR} \\varphi_{iR} = 0.

        - :math:`x_{ij}` : :attr:`~porepy.compositional.base.Phase.extended_fraction_of`
          component
        - :math:`\\varphi_{ij}` : Phase
          :attr:`~porepy.compositional.base.Phase.fugacity_coefficient_of` component

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
        rphase = self.fluid.reference_phase
        if phase == rphase:
            raise ValueError(
                "Cannot construct isofugacity constraint between reference phase and "
                + "itself."
            )
        assert component in phase, "Passed component not modelled in passed phase."
        assert component in rphase, "Passed component not modelled in reference phase."

        x_ij = phase.extended_fraction_of[component](subdomains)
        x_ir = rphase.extended_fraction_of[component](subdomains)
        phi_ij = phase.fugacity_coefficient_of[component](subdomains)
        phi_ir = rphase.fugacity_coefficient_of[component](subdomains)

        equ = x_ij * phi_ij - x_ir * phi_ir

        equ.set_name(
            f"isofugacity_constraint_{component.name}_{phase.name}_{rphase.name}"
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

        - :math:`y_j`: Phase :attr:`~porepy.compositional.base.Phase.fraction`.
        - :math:`h_j`: Phase :attr:`~porepy.compositional.base.Phase.specific_enthalpy`.
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
        h_mix = self.fluid.specific_enthalpy(subdomains)
        h_target = self.enthalpy(subdomains)

        if self._normalize_constraints:
            equ = h_mix / h_target - pp.ad.Scalar(1.0)
        else:
            equ = h_mix - h_target

        equ.set_name("local_fluid_enthalpy_constraint")
        return equ

    def mixture_volume_constraint(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the volume constraint using the reciprocal of the mixture density.

        .. math::

            \\dfrac{1}{\\sum_j s_j \\rho_j} - v = 0~,~
            v \\left(\\sum_j s_j \\rho_j\\right) - 1 = 0.

        - :math:`s_j` : Phase :attr:`~porepy.compositional.base.Phase.saturation`
        - :math:`\\rho_j` : Phase :attr:`~porepy.compositional.base.Phase.density`

        Parameters:
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equations. If the normalization of state
            constraints is required by the solution strategy, the second form is
            returned.

        """
        if self._normalize_constraints:
            equ = self.volume(subdomains) * self.fluid.density(
                subdomains
            ) - pp.ad.Scalar(1.0)
        else:
            equ = self.volume(subdomains) - self.fluid.specific_volume(subdomains)
        equ.set_name("local_fluid_volume_constraint")
        return equ

    def mass_constraint_for_phase(
        self, phase: pp.Phase, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs a type of local mass constraint based on a relation between
        mixture density, saturated phase density and phase fractions.

        For a phase :math:`j` it holds:

        .. math::

            y_j \\rho - s_j \\rho_j = 0~,~
            y_j - s_j \\dfrac{\\rho_j}{rho} = 0

        with the mixture density :math:`\\rho = \\sum_k s_k \\rho_k`, assuming
        :math:`\\rho_k` is the density of a phase when saturated.

        - :math:`y` : Phase :attr:`~porepy.compositional.base.Phase.fraction`
        - :math:`s` : Phase :attr:`~porepy.compositional.base.Phase.saturation`
        - :math:`\\rho` : Fluid mixture :attr:`~porepy.compositional.base.Fluid.
          density`
        - :math:`\\rho_j` : Phase:attr:`~porepy.compositional.base.Phase.density`

        Note:
            These equations can be used to close the model if molar phase fractions and
            saturations are independent variables.

            They also appear in the unified flash with isochoric specifications.

        Parameters:
            phase: A phase for which the equation should be assembled.
            subdomains: A list of subdomains on which the equation is defined.

        Returns:
            The left-hand side of above equations.

            If normalization of state constraints is set in the solution strategy,
            it returns the normalized form.

        """
        if self._normalize_constraints:
            equ = phase.fraction(subdomains) - phase.saturation(
                subdomains
            ) * phase.density(subdomains) / self.fluid.density(subdomains)
        else:
            equ = phase.fraction(subdomains) * self.fluid.density(
                subdomains
            ) - phase.saturation(subdomains) * phase.density(subdomains)
        equ.set_name(f"local_density_conservation_{phase.name}")
        return equ


class Unified_pT_Equilibrium(UnifiedPhaseEquilibriumEquations):
    """Mixin class modelling the unified p-T flash.

    The unified p-T flash consists of

    - ``num_components - 1`` local mass constraints for components
    - ``(num_phases - 1) * num_components`` isofugacity constraints
    - ``num_phases`` semi-smooth complementarity conditions.

    I.e., for ``num_phase - 1`` independent molar phase fractions and
    ``num_components * num_phases`` extended molar fractions of components in phases,
    the local model is closed.

    """

    def set_equations(self) -> None:
        """Introduces the equations into the equation system on all subdomains."""
        assert isinstance(self, EquationMixin)
        super().set_equations()  # type:ignore[safe-super]

        subdomains = self.mdg.subdomains()

        ## starting with equations common to all equilibrium definitions
        # local mass constraint per independent component
        for comp in self.fluid.components:
            # skipping reference component according to unified assumptions
            if comp != self.fluid.reference_component:
                equ = self.mass_constraint_for_component(comp, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # isofugacity constraints
        rphase = self.fluid.reference_phase
        for phase in self.fluid.phases:
            if phase != rphase:
                for comp in self.fluid.components:
                    equ = self.isofugacity_constraint_for_component_in_phase(
                        comp, phase, subdomains
                    )
                    self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # complementarity conditions
        for phase in self.fluid.phases:
            equ = self.complementarity_condition_for_phase(phase, subdomains)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class Unified_ph_Equilibrium(Unified_pT_Equilibrium):
    """Unified equilibrium equations where temperature is treated as an unknown.

    To close the system, this class introduces a local enthalpy constraint atop the
    standard equations set up by the unified p-T flash.
    This equation mixin introduces a local enthalpy constraint, constraining
    the fluid mixture enthalpy to a given enthalpy value.

    Compared to the p-T model, it has hence 1 equation and 1 unknown more, and is
    closed.

    """

    def set_equations(self) -> None:
        """Introduces the local  enthalpy constraint, atop the equations introduced
        by :class:`Unified_pT_Equilibrium`, on all subdomains."""
        assert isinstance(self, EquationMixin)
        super().set_equations()  # type:ignore[safe-super]
        subdomains = self.mdg.subdomains()
        equ = self.mixture_enthalpy_constraint(subdomains)
        self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class Unified_vh_Equilibrium(Unified_ph_Equilibrium):
    """Unified equilibrium model if pressure and temperature are unknown at equilibrium.

    It extends the unified p-h equilibrium formulation by introducing a local
    volume constraint, and ``num_phases - 1`` phase density relations.

    If volume is given, the saturation of independent phases
    (volumetric fractions) are required to define a fluid volume as the reciprocal
    of the mixture density.

    In total, this system has ``1 + (num_phases - 1)`` additional unknowns and equations
    and is hence closed.

    """

    def set_equations(self) -> None:
        assert isinstance(self, EquationMixin)
        super().set_equations()  # type:ignore[safe-super]
        subdomains = self.mdg.subdomains()
        equ = self.mixture_volume_constraint(subdomains)
        self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        for phase in self.fluid.phases:
            if phase != self.fluid.reference_phase:
                equ = self.mass_constraint_for_phase(phase, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})
