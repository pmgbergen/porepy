"""Module containing mixin classes for introducing the local equilibrium problem into a
PorePy model.

Local equilibrium equations are single, cell-wise algebraic equations, introducing
the thermodynamically consistent approach to modelling secondary expressions like
phase densities.
They also introduce necessarily new variables into the system (fractions).

Instances of :class:`UnifiedEquilibriumMixin` require an attribite ``equilibrium_type``
to be set, which must not be ``None``. This is to inform the remaining framework
that local equilibrium assumptions (instead of some constitutive laws) were introduced.

"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np

import porepy as pp

from .base import Component, FluidMixture, Phase
from .flash import Flash
from .states import FluidState
from .utils import CompositionalModellingError, safe_sum

__all__ = [
    "evaluate_homogenous_constraint",
    "UnifiedPhaseEquilibriumMixin",
    "Unified_pT_Equilibrium",
    "Unified_ph_Equilibrium",
    "Unified_vh_Equilibrium",
    "FlashMixin",
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
        assert len(weights) == len(
            phi_i
        ), "Need equal amount of weights and partial quantities"
        return phi - safe_sum([phi_ * w for phi_, w in zip(phi_i, weights)])
    else:
        return phi - safe_sum(phi_i)


class UnifiedPhaseEquilibriumMixin:
    """Base class for introducing local phase equilibrium equations into a model using
    the unified formulation.

    The base class provides means to assemble required equations, as well as to define
    the :attr:`equilibrium_type`.

    The solution strategy sets this value to None, if not defined by a class here.

    Important:
        This class assumes the mixture is fully set up, including all properties and
        variables.

    """

    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """A string denoting the two state functions which are assumed constant in the
    local (phase) equilibrium problem.

    Must be set to a value in models using some local equilibrium equations.

    """

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    fluid_mixture: FluidMixture
    """Provided by :class:`FluidMixtureMixin`."""

    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by
    :class:`~porepy.models.compositional_flow.VariablesCF`."""
    volume: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Provided by :class:`~porepy.models.compositional_flow.SolidSkeletonCF`."""

    eliminate_reference_component: bool
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""
    eliminate_reference_phase: bool
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""
    use_semismooth_complementarity: bool
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""
    normalize_state_constraints: bool
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""

    ad_time_step: pp.ad.Operator
    """Provided by solutions trategy."""

    def set_equations(self) -> None:
        """The base class method without defined equilibrium type performs a model
        validation to ensure that the assumptions for the unified flash are fulfilled.
        """
        ncomp = self.fluid_mixture.num_components
        nphase = self.fluid_mixture.num_phases

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
        if not self.eliminate_reference_component:
            warnings.warn(
                "Unified equilibrium model included, but reference phase not"
                + " eliminated. Check model closedness."
            )
        if not self.eliminate_reference_phase:
            warnings.warn(
                "Unified equilibrium model included, but reference component not"
                + " eliminated. Check model closedness."
            )

        all_comps = set(self.fluid_mixture.components)
        for phase in self.fluid_mixture.phases:
            phase_comps = set(phase)
            if all_comps.symmetric_difference(phase_comps):
                raise CompositionalModellingError(
                    f"Unified equilibrium assumption violated for phase: {phase.name}."
                    + " All phases must have all components modelled in them."
                )

    def mass_constraint_for_component(
        self, component: Component, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the local mass constraint for a component :math:`i`.

        .. math::

            z_i - \\sum_j x_{ij} y_j = 0.

        - :math:`z` : Component :attr:`~porepy.compositional.base.Component.fraction`
        - :math:`y` : Phase :attr:`~porepy.compositional.base.Phase.fraction`
        - :math:`x` : Phase :attr:`~porepy.compositional.base.Phase.fraction_of` component

        The above sum is performed over all phases the component is present in.

        Parameter:
            component: The component represented by the overall fraction :math:`z_i`.
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            An operator representing the left-hand side of above equation.

        """
        # get all phases the component is present in
        phases = [phase for phase in self.fluid_mixture.phases if component in phase]

        equ: pp.ad.Operator = evaluate_homogenous_constraint(
            component.fraction(subdomains),
            [phase.extended_fraction_of[component](subdomains) for phase in phases],
            [phase.fraction(subdomains) for phase in phases],
        )  # type:ignore
        equ.set_name(f"local_mass_constraint_{component.name}")
        return equ

    def complementarity_condition_for_phase(
        self, phase: Phase, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the complementarity condition for a given phase.

        .. math::

            y_j (1 - \\sum_i x_{ij}) = 0~,~
            \\min \\{y_j, (1 - \\sum_i x_{ij}) \\} = 0.

        - :math:`y` : Phase :attr:`~porepy.compositional.base.Phase.fraction`
        - :math:`x` : Phase :attr:`~porepy.compositional.base.Phase.fraction_of` component

        The sum is performed over all components modelled in that phase
        (see :attr:`~porepy.compositional.base.Phase.components`).

        Parameters:
            phase: The phase for which the condition is assembled.
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equation. If the semi-smooth form is
            requested by the solution strategy, then the :math:`\\min\\{\\}` operator is
            used.

        """

        unity: pp.ad.Operator = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
            [phase.extended_fraction_of[comp](subdomains) for comp in phase]
        )

        minimum = lambda x, y: pp.ad.maximum(-x, -y)
        ssmin = pp.ad.Function(minimum, "semi-smooth-minimum")

        if self.use_semismooth_complementarity:
            equ = ssmin(phase.fraction(subdomains), unity)
            equ.set_name(f"semismooth_complementary_condition_{phase.name}")
        else:
            equ = phase.fraction(subdomains) * unity
            equ.set_name(f"complementary_condition_{phase.name}")
        return equ

    def isofugacity_constraint_for_component_in_phase(
        self, component: Component, phase: Phase, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Construct the local isofugacity constraint for a component between a given
        phase and the reference phase.

        .. math::

            x_{ij} \\varphi_{ij} - x_{iR} \\varphi_{iR} = 0.

        - :math:`x_{ij}` : :attr:`~porepy.compositional.base.Phase.fraction_of` component
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
        rphase = self.fluid_mixture.reference_phase
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
            f"isofugacity_constraint_" + f"{component.name}_{phase.name}_{rphase.name}"
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
        - :math:`h_j`: Phase :attr:`~porepy.compositional.base.Phase.enthalpy`.
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

        h_mix = self.fluid_mixture.specific_enthalpy(subdomains)
        h_target = self.enthalpy(subdomains)
        if self.normalize_state_constraints:
            equ = h_mix / h_target - pp.ad.Scalar(1.0)
        else:
            equ = h_mix - h_target

        if self.equilibrium_type == "p-T":  # TODO should I leave it for now or later?
            equ = (
                pp.ad.dt(equ, self.ad_time_step)
                + (pp.ad.Scalar(1 / 5) / self.ad_time_step) * equ
            )
            equ.set_name("relaxed_local_fluid_enthalpy_constraint")
        else:
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
        if self.normalize_state_constraints:
            equ = self.volume(subdomains) * self.fluid_mixture.density(
                subdomains
            ) - pp.ad.Scalar(1.0)
        else:
            equ = self.volume(subdomains) - self.fluid_mixture.specific_volume(
                subdomains
            )
        equ.set_name("local_fluid_volume_constraint")
        return equ

    def density_relation_for_phase(
        self, phase: Phase, subdomains: Sequence[pp.Grid]
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
        - :math:`\\rho` : Fluid mixture :attr:`~porepy.compositional.base.FluidMixture.density`
        - :math:`\\rho_j` : Phase:attr:`~porepy.compositional.base.Phase.density`

        Note:
            These equations can be used to close the model if molar phase fractions and
            saturations are independent variables.

            They also appear in the unified flash with isochoric specificitations.

        Parameters:
            phase: A phase for which the equation should be assembled.
            subdomains: A list of subdomains on which the equation is defined.

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
        equ.set_name(f"local_density_conservation_{phase.name}")
        return equ


class Unified_pT_Equilibrium(UnifiedPhaseEquilibriumMixin):
    """Mixin class modelling the unified p-T flash.

    The unified p-T flash consists of

    - ``num_components - 1`` local mass constraints
    - ``(num_phases - 1) * num_components`` isofugacity constraints
    - ``num_phases`` semi-smooth complementarity conditions.

    I.e., for ``num_phase - 1`` independent molar phase fractions and
    ``num_components * num_phases`` extended molar fractions of components in phases,
    the local model is closed.

    """

    equilibrium_type = "p-T"
    """THe quantities fixed at equilibrium are pressure, temperature and overall
    fractions."""

    def set_equations(self) -> None:
        """Introduces the equations into the equation system on all subdomains."""
        # Perform model validations
        UnifiedPhaseEquilibriumMixin.set_equations(self)
        subdomains = self.mdg.subdomains()

        ## starting with equations common to all equilibrium definitions
        # local mass constraint per independent component
        for comp in self.fluid_mixture.components:
            # skip for reference component if eliminated
            if comp != self.fluid_mixture.reference_component:
                equ = self.mass_constraint_for_component(comp, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # isofugacity constraints
        rphase = self.fluid_mixture.reference_phase
        for phase in self.fluid_mixture.phases:
            if phase != rphase:
                for comp in self.fluid_mixture.components:
                    equ = self.isofugacity_constraint_for_component_in_phase(
                        comp, phase, subdomains
                    )
                    self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        # complementarity conditions
        for phase in self.fluid_mixture.phases:
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

    equilibrium_type = "p-h"
    """The quantities fixed at equilibrium are pressure, specific enthalpy and
    overall fractions."""

    def set_equations(self) -> None:
        """Introduces the local  enthalpy constraint, atop the equations introduced
        by :class:`Unified_pT_Equilibrium`, on all subdomains."""
        Unified_pT_Equilibrium.set_equations(self)
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

    equilibrium_type = "v-h"
    """The quantities fixed at equilibrium are specific volume, specific enthalpy
    and overall fractions."""

    def set_equations(self) -> None:
        Unified_ph_Equilibrium.set_equations(self)
        subdomains = self.mdg.subdomains()
        equ = self.mixture_volume_constraint(subdomains)
        self.equation_system.set_equation(equ, subdomains, {"cells": 1})

        for phase in self.fluid_mixture.phases:
            if phase != self.fluid_mixture.reference_phase:
                equ = self.density_relation_for_phase(phase, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class FlashMixin:
    """Mixin class to introduce the unified flash procedure into the solution strategy.

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

    flash_params: dict = dict()
    """The dictionary to be passed to a flash algorithm, whenever it is called."""

    mdg: pp.MixedDimensionalGrid
    """Provided by :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """Provided by :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    fluid_mixture: FluidMixture
    """Provided by :class:`FluidMixtureMixin`."""

    fractional_state_from_vector: Callable[
        [Sequence[pp.Grid], Optional[np.ndarray]], FluidState
    ]
    """Provided by :class:`CompositeVariables`."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Provided by
    :class:`~porepy.models.compositional_flow.VariablesCF`."""
    volume: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Provided by :class:`~porepy.models.compositional_flow.SolidSkeletonCF`."""

    equilibrium_type: Optional[Literal["p-T", "p-h", "v-h"]]
    """Provided by
    :class:`~porepy.models.compositional_flow.SolutionStrategyCF`."""

    def set_up_flasher(self) -> None:
        """Method to introduce the flash class, if an equilibrium is defined.

        This method is called by the solution strategy after the model is set up.

        """
        raise CompositionalModellingError(
            "Call to mixin method. No flash object defined."
        )

    def get_fluid_state(
        self, subdomains: Sequence[pp.Grid], state: Optional[np.ndarray] = None
    ) -> FluidState:
        """Method to assemble a fluid state in the iterative procedure, which
        should be passed to :meth:`equilibriate_fluid`.

        This method provides room to pre-process data before the flash is called with
        the returned fluid state as the initial guess.

        Parameters:
            subdomains: Subdomains for which the state functions should be evaluated
            state: ``default=None``

                Global state vector to be passed to the Ad framework when evaluating the
                current state (fractions, pressure, temperature, enthalpy,..)

        Returns:
            The base method returns a fluid state containing the current iterate value
            of the unknowns of respective flash subproblem (p-T, p-h,...).

        """

        # Extracting the current, iterative state to use as initial guess for the flash
        fluid_state = self.fractional_state_from_vector(subdomains, state)

        # Evaluate temperature as initial guess, if not fixed in equilibrium type
        if "T" not in self.equilibrium_type:
            # initial guess for T from iterate
            fluid_state.T = self.temperature(subdomains).value(
                self.equation_system, state
            )
        # evaluate pressure, if volume is fixed. NOTE saturations are also fractions
        # and already included
        if "v" in self.equilibrium_type:
            fluid_state.p = self.pressure(subdomains).value(self.equation_system, state)

        return fluid_state

    def equilibriate_fluid(
        self,
        subdomains: Sequence[pp.Grid],
        state: Optional[np.ndarray] = None,
        initial_fluid_state: Optional[FluidState] = None,
    ) -> tuple[FluidState, np.ndarray]:
        """Convenience method perform the flash based on model specifications.

        This method is called in
        :meth:`~porepy.models.compositional_flow.SolutionStrategyCF.
        before_nonlinear_iteration` to use the flash as a predictor during nonlinear
        iterations.

        Parameters:
            subdomains: Subdomains on which to evaluate the target state functions.
            state: ``default=None``

                Global state vector to be passed to the Ad framework when evaluating the
                state functions.
            initial_fluid_state: ``default=None``

                Initial guess passed to :meth:`~porepy.compositional.flash.Flash.flash`.
                Note that if None, the flash computes the initial guess itself.

        Returns:
            The equilibriated state of the fluid and an indicator where the flash was
            successful (or not).

            For more information on the `success`-indicators, see respective flash
            object.

        """

        if initial_fluid_state is None:
            z = np.array(
                [
                    comp.fraction(subdomains).value(self.equation_system)
                    for comp in self.fluid_mixture.components
                ]
            )
        else:
            z = initial_fluid_state.z

        flash_kwargs = {
            "z": z,
            "initial_state": initial_fluid_state,
            "parameters": self.flash_params,
        }

        if self.equilibrium_type == "p-T":
            flash_kwargs.update(
                {
                    "p": self.pressure(subdomains).value(self.equation_system, state),
                    "T": self.temperature(subdomains).value(
                        self.equation_system, state
                    ),
                }
            )
        elif self.equilibrium_type == "p-h":
            flash_kwargs.update(
                {
                    "p": self.pressure(subdomains).value(self.equation_system, state),
                    "h": self.enthalpy(subdomains).value(self.equation_system, state),
                }
            )
        elif self.equilibrium_type == "v-h":
            flash_kwargs.update(
                {
                    "v": self.volume(subdomains).value(self.equation_system, state),
                    "h": self.enthalpy(subdomains).value(self.equation_system, state),
                }
            )
        else:
            raise CompositionalModellingError(
                "Attempting to equilibriate fluid with uncovered equilibrium type"
                + f" {self.equilibrium_type}."
            )

        if "p" in flash_kwargs:
            flash_kwargs["p"] = flash_kwargs["p"]
        if "h" in flash_kwargs:
            flash_kwargs["h"] = flash_kwargs["h"]
        result_state, succes, _ = self.flash.flash(**flash_kwargs)

        return result_state, succes

    def postprocess_failures(
        self, subdomain: pp.Grid, fluid_state: FluidState, success: np.ndarray
    ) -> FluidState:
        """A method called after :meth:`equilibriate_fluid` to post-process failures if
        any.

        Parameters:
            subdomain: A grid for which ``fluid_state`` contains the values.
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
            raise ValueError(
                "Flash strategy did not succeed in"
                + f" {(success > 0).sum()} / {len(success)} cases."
            )
