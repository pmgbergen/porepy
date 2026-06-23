"""Module containing an equation class for introducing the local, persistent-variable
equilibrium problem (PVE) into a PorePy model.

Local equilibrium equations are single, cell-wise algebraic equations, introducing
the thermodynamically consistent approach to modelling secondary expressions like
phase densities and closing a CF model.

"""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import Callable, Sequence

import porepy as pp
import porepy.compositional as pc
from porepy.models.abstract_equations import EquationMixin

__all__ = [
    "PersistentEquilibriumEquations",
]

_ad_capped_log: pp.ad.Function = pp.ad.Function(
    lambda x: pp.ad.functions.log(pp.ad.maximum(x, 1e-14)),
    "ad_capped_log",
)


class PersistentEquilibriumEquations(EquationMixin):
    """Base class for introducing local phase equilibrium equations into a model using
    the persisten-variable formulation.

    The base class provides means to assemble required equations, as well as a
    verification of model assumptions for the formulation.

    A :class:`~porepy.compositional.utils.CompositionalModellingError` will be raised
    if any of the following assumptions is violated:

    1. At least 2 components and 2 phases are modelled.
    2. The model's ``params['equilibrium_specification']`` is not None and contains the
       keyword ``'persistent-variables'``.
    3. All phases have all components set in them (all extended partial fractions are
       defined and introduced).

    If the reference phase was not eliminated (dangling variables), a warning is raised.

    Equilibrium equations are introduced based on the global equilibrium specification
    found in ``model.params["equilibrium_specification"]``.
    Ensures the variables for the equilibrium target state are defined and introduces
    respective local equations.

    Includes closure relations in case saturations and fractions for a phase are
    variables.

    """

    specific_fluid_enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    specific_fluid_internal_energy: Callable[
        [pp.SubdomainsOrBoundaries], pp.ad.Operator
    ]
    specific_fluid_volume: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    has_independent_saturation: Callable[[pp.Phase], bool]
    has_independent_fraction: Callable[[pp.Phase | pp.Component], bool]

    @cached_property
    def equilibrium_equation_names(self) -> list[str]:
        """List of introduced equilibrium equations."""
        return []

    def set_equations(self) -> None:
        """The base class method without defined equilibrium condition performs a model
        validation to ensure that the assumptions for the persistent-variable
        formulation are fulfilled.

        """
        super().set_equations()

        nphase = self.fluid.num_phases

        if not pp.compositional.is_persistent_variable_form(self):
            raise pc.CompositionalModellingError(
                "Must define a `equilibrium_specification` model parameter containing"
                + " the keyword `persistent-variables`."
            )

        if nphase < 2:
            raise pc.CompositionalModellingError(
                "Unified equilibrium models need at least to modelled phases,"
                + f" {nphase} given."
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
                raise pc.CompositionalModellingError(
                    f"Persistent-variables assumption violated for phase: {phase.name}."
                    + " All phases must have all components modelled in them."
                )

        equ_specs = pc.get_equilibrium_specifications(self)
        if pc.FlashSpec.none in equ_specs:
            raise pc.CompositionalModellingError(
                "No global equilibrium conditions specified."
            )

        spec = [s for s in equ_specs if isinstance(s, pc.FlashSpec)][0]
        subdomains = self.mdg.subdomains()

        equations: list[pp.ad.Operator] = []

        # First, set volume-constraint for isochoric specifications.
        if spec >= pc.FlashSpec.vT:
            assert isinstance(self, pp.fluid_mass_balance.VariablesSinglePhaseFlow), (
                f"Model {type(self)} does not inherit the flow variables."
            )

            if not self.has_fluid_volume_variable:
                raise pc.CompositionalModellingError(
                    "Expecting fluid volume variable for isochoric equilibrium."
                )

            equations.append(self.local_fluid_volume_constraint(subdomains))

        # Second, set energy-constraint for non-isothermal specifications.
        if spec not in [pc.FlashSpec.pT, pc.FlashSpec.vT]:
            assert isinstance(self, pp.energy_balance.VariablesEnergyBalance), (
                f"Model {type(self)} does not inherit the energy variables."
            )

            if self.has_fluid_internal_energy_variable and spec == pc.FlashSpec.vu:
                equations.append(
                    self.local_fluid_internal_energy_constraint(subdomains)
                )
            elif self.has_fluid_enthalpy_variable and spec in [
                pc.FlashSpec.ph,
                pc.FlashSpec.vh,
            ]:
                equations.append(self.local_fluid_enthalpy_constraint(subdomains))
            else:
                raise pc.CompositionalModellingError(
                    "Failed to resolve energetic equilibrium specification:\n"
                    f"Has internal energy: {self.has_fluid_internal_energy_variable}\n"
                    f"Has enthalpy: {self.has_fluid_enthalpy_variable}\n"
                    f"Global specification: {spec.name}"
                )

        # What follows are equations shared by all equilibrium systems.
        # Third, isofugacity constraints.
        for phase in self.fluid.phases:
            if phase != self.fluid.reference_phase:
                for comp in self.fluid.components:
                    equations.append(
                        self.isofugacity_constraint_for_component_in_phase(
                            comp, phase, subdomains
                        )
                    )

        # Fourth, local mass constraints.
        for comp in self.fluid.components:
            # skipping reference component according to assumptions
            if comp != self.fluid.reference_component:
                equations.append(
                    self.local_mass_constraint_for_component(comp, subdomains)
                )

        # Fifth, complementarity conditions.
        for phase in self.fluid.phases:
            equations.append(
                self.complementarity_condition_for_phase(phase, subdomains)
            )

        # Finally, the closure relations for phase-related fractions.
        for phase in self.fluid.phases:
            if self.has_independent_fraction(phase) and self.has_independent_saturation(
                phase
            ):
                equations.append(self.mass_constraint_for_phase(phase, subdomains))

        # Set all equations in the equation system and store the names.
        for equ in equations:
            self.equilibrium_equation_names.append(equ.name)
            self.equation_system.set_equation(equ, subdomains, {"cells": 1})

    def local_mass_constraint_for_component(
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

        equ = pp.ad.sum_operator_list([x * y for x, y in zip(x_ij, y_j)]) - z_i

        equ.set_name(f"local_component_mass_constraint_{component.name}")
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
            The left-hand side of above equation. The :math:`\\min\\{\\}` operator is
            used by default (semi-smooth form).

        """

        unity: pp.ad.Operator = pp.ad.Scalar(1.0) - pp.ad.sum_operator_list(
            [phase.extended_fraction_of[comp](subdomains) for comp in phase]
        )

        minimum = lambda x, y: pp.ad.maximum(-x, -y)
        ssmin = pp.ad.Function(minimum, "semi-smooth-minimum")

        equ = ssmin(phase.fraction(subdomains), unity)
        equ.set_name(f"semismooth_complementary_condition_{phase.name}")
        return equ

    def isofugacity_constraint_for_component_in_phase(
        self,
        component: pp.FluidComponent,
        phase: pp.Phase,
        subdomains: Sequence[pp.Grid],
    ) -> pp.ad.Operator:
        r"""Construct the local isofugacity constraint for a component between a given
        phase and the reference phase in the log space.

        .. math::

            \log{x_{ij}} + \log{\varphi_{ij}} - \log{x_{iR}} -  \log{\varphi_{iR}} = 0.

        - :math:`x_{ij}` : :attr:`~porepy.compositional.base.Phase.extended_fraction_of`
          component
        - :math:`\varphi_{ij}` : Phase
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

        equ = _ad_capped_log(x_ij) + phi_ij - _ad_capped_log(x_ir) - phi_ir

        equ.set_name(
            f"isofugacity_constraint_{component.name}_{phase.name}_{rphase.name}"
        )
        return equ

    def local_fluid_enthalpy_constraint(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the enthalpy constraint for the mixture enthalpy and the
        transported enthalpy variable.

        .. math::

            \\sum_j y_j h_j  - h = 0~,~
            (\\sum_j y_j h_j) / h - 1= 0~

        - :math:`y_j`: Phase :attr:`~porepy.compositional.base.Phase.fraction`.
        - :math:`h_j`: Phase :attr:`~porepy.compositional.base.Phase.specific_enthalpy`.
        - :math:`h`: The transported enthalpy.

        The first term represents the mixture enthalpy based on the thermodynamic state.
        The second term represents the target enthalpy in the equilibrium problem.
        The target enthalpy is a transportable quantity in flow and transport.

        Parameters:
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equation.

        """
        equ = self.fluid.specific_enthalpy(subdomains) / self.specific_fluid_enthalpy(
            subdomains
        ) - pp.ad.Scalar(1.0)
        equ.set_name("local_fluid_enthalpy_constraint")
        return equ

    def local_fluid_internal_energy_constraint(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Constructs the internal energy constraint for the mixture internal energy and
        the transported energy variable.

        .. math::

            \\sum_j y_j u_j  - u = 0~,~
            (\\sum_j y_j u_j) / u - 1= 0~

        - :math:`y_j`: Phase :attr:`~porepy.compositional.base.Phase.fraction`.
        - :math:`u_j`: Phase :attr:`~porepy.compositional.base.Phase.
          specific_internal_energy`.
        - :math:`u`: The transported energy.

        The first term represents the mixture energy based on the thermodynamic state.
        The second term represents the target energy in the equilibrium problem.
        The target energy is a transportable quantity in flow and transport.

        Parameters:
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equation.

        """
        equ = self.fluid.specific_internal_energy(
            subdomains
        ) / self.specific_fluid_internal_energy(subdomains) - pp.ad.Scalar(1.0)
        equ.set_name("local_fluid_internal_energy_constraint")
        return equ

    def local_fluid_volume_constraint(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns the constraint on the specific volume of the fluid in log-space

        .. math::

            \\log{\\hat{v}} - \\log{v}~,

        with :math:`v` being :meth:`~porepy.compositional.base.FluidMixture.
        specific_volume` and :math:`\\hat{v}` the transported volume.

        Parameters:
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equation.

        """
        equ = _ad_capped_log(self.specific_fluid_volume(subdomains)) - _ad_capped_log(
            self.fluid.specific_volume(subdomains)
        )
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
        equ = phase.fraction(subdomains) * self.fluid.density(
            subdomains
        ) / phase.density(subdomains) - phase.saturation(subdomains)
        equ.set_name(f"local_phase_mass_constraint_{phase.name}")
        return equ
