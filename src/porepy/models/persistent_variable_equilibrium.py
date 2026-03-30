"""Module containing equation classes for introducing the local, persistent-variable
equilibrium problem (PVE) into a PorePy model.

Local equilibrium equations are single, cell-wise algebraic equations, introducing
the thermodynamically consistent approach to modelling secondary expressions like
phase densities and closing a CF model.

Instances of :class:`UnifiedEquilibriumMixin` require the
``'equilibrium_specification'`` model parameter to be *not* none. This is to inform the
remaining framework that local equilibrium assumptions were introduced.

"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import porepy as pp
from porepy.compositional.utils import CompositionalModellingError
from porepy.models.abstract_equations import EquationMixin

__all__ = [
    "PVEEquations",
    "PT_PVEEquations",
    "PH_PVEEquations",
    "VT_PVEEquations",
]


class PVEEquations(EquationMixin):
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

    """

    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.EnthalpyVariable`."""
    fluid_specific_volume: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.fluid_mass_balance.FluidVolumeVariable`."""

    __ad_capped_log: pp.ad.Function = pp.ad.Function(
        # pp.ad.log,
        lambda x: pp.ad.functions.log(pp.ad.maximum(x, 1e-14)),
        "ad_capped_log",
    )

    def set_equations(self) -> None:
        """The base class method without defined equilibrium condition performs a model
        validation to ensure that the assumptions for the persistent-variable
        formulation are fulfilled.

        """
        super().set_equations()

        nphase = self.fluid.num_phases

        if not pp.compositional.is_persistent_variable_form(self):
            raise CompositionalModellingError(
                "Must define a `equilibrium_specification` model parameter containing"
                + " the keyword `persistent-variables`."
            )

        if nphase < 2:
            raise CompositionalModellingError(
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
                raise CompositionalModellingError(
                    f"Persistent-variables assumption violated for phase: {phase.name}."
                    + " All phases must have all components modelled in them."
                )

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

        equ = self.__ad_capped_log(x_ij) + phi_ij - self.__ad_capped_log(x_ir) - phi_ir

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
        - :math:`h`: The transported enthalpy :attr:`enthalpy`.

        The first term represents the mixture enthalpy based on the thermodynamic state.
        The second term represents the target enthalpy in the equilibrium problem.
        The target enthalpy is a transportable quantity in flow and transport.

        Parameters:
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equations.

        """
        equ = self.fluid.specific_enthalpy(subdomains) / self.enthalpy(
            subdomains
        ) - pp.ad.Scalar(1.0)
        equ.set_name("local_fluid_enthalpy_constraint")
        return equ

    def local_fluid_volume_constraint(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        """Returns the constraint on the specific volume of the fluid in log-space

        .. math::

            \\log{\\hat{v}} - \\log{v}~,

        with :math:`v` being :meth:`~porepy.compositional.base.FluidMixture.
        specific_volume` and :math:`\\hat{v}` some mixed in method returning the
        available specific volume.

        Parameters:
            subdomains: A list of subdomains on which to define the equation.

        Returns:
            The left-hand side of above equations.

        """
        equ = self.__ad_capped_log(
            self.fluid_specific_volume(subdomains)
        ) - self.__ad_capped_log(self.fluid.specific_volume(subdomains))
        equ.set_name("local_fluid_volume_constraint")
        return equ


class PT_PVEEquations(PVEEquations):
    """Mixin class modelling the persistent p-T flash.

    This local system of equations consists of:

    - ``num_components - 1`` local mass constraints for components
    - ``(num_phases - 1) * num_components`` isofugacity constraints
    - ``num_phases`` semi-smooth complementarity conditions.

    I.e., for ``num_phase - 1`` independent molar phase fractions and
    ``num_components * num_phases`` extended molar fractions of components in phases,
    the local model is closed.

    """

    def set_equations(self) -> None:
        """Introduces the equations into the equation system on all subdomains."""
        super().set_equations()

        subdomains = self.mdg.subdomains()

        ## starting with equations common to all equilibrium definitions
        # local mass constraint per independent component
        for comp in self.fluid.components:
            # skipping reference component according to assumptions
            if comp != self.fluid.reference_component:
                equ = self.local_mass_constraint_for_component(comp, subdomains)
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


class PH_PVEEquations(PT_PVEEquations):
    """Equilibrium system where temperature is treated as an unknown.

    To close the system, this class introduces a local fluid enthalpy constraint on top
    of the standard equations set up by the p-T system.
    It constraints the enthalpy of the fluid mixture to a (presumed) enthalpy variable.

    """

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()
        equ = self.local_fluid_enthalpy_constraint(subdomains)
        self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class VT_PVEEquations(PT_PVEEquations):
    """Analogous to :class:`PH_PVEEquations` but instead of a local enthalpy constraint
    it introduces a local fluid volume constraint, coupling the specific volume of
    the fluid mixture to a presumed variable for the fluid volume.

    The equation is formulated in the log-space.

    """

    def set_equations(self):
        super().set_equations()
        subdomains = self.mdg.subdomains()
        equ = self.local_fluid_volume_constraint(subdomains)
        self.equation_system.set_equation(equ, subdomains, {"cells": 1})
