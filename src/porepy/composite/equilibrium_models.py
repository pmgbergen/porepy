"""A module containing equilibrium formulations for mixtures using PorePy's AD framework
to define first-order conditions for equilibrium.

"""
from __future__ import annotations

from typing import Any, Optional

import porepy as pp

from .base import Component, Mixture, Phase
from .composite_utils import safe_sum

__all__ = ["MixtureUNR", "evaluate_homogenous_constraint"]


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


class MixtureUNR(Mixture):
    """A class modelling Unified equilibrium conditions for Non-Reactive mixtures.

    This is a layer-2 mixture implementing equilibrium equations on top of the
    standard expressions and variables in the base class.

    It models the following equations using PorePy's AD framework.

    - Static mass constraints (fixed feed fractions)
    - Equilibrium equations formulated with respect to the reference phase
      (isofugacity constraints).
    - Complementary conditions for phase fractions using the unified framework.

    Note:
        If the reference phase is eliminated, above set equations is a closed model for
        the p-T flash.
        In this equilibrium formulation, the unknowns are independent phase fractions
        and phase compositions, a total of
        ``num_phase - 1 + num_comp * (num_phase - 1)`` unknowns.
        This is also the number of equations modelled by this class.

    Note:
        For other flash configurations like the p-h formulation, an additional
        enthalpy constraint is required, including the target enthalpy.

        While not created by default, respective equations can be assembled using this
        class.

        This class assembles only above mentioned equations, which are common in all
        unified formulations.

    """

    def __init__(self, components: list[Component], phases: list[Phase]) -> None:
        super().__init__(components, phases)

        self.mass_constraints: dict[Component, pp.ad.Operator]
        """A map containing mass constraints per components (key), except for the
        reference component.

        .. math::

            z_i - \\sum_j x_{ij} y_j~,~\\forall i \\neq r~,

        - :math:`z` : Component :attr:`~porepy.composite.base.Component.fraction`
        - :math:`y` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`x` : Phase :attr:`~porepy.composite.base.Phase.fraction_of` component

        This operator is created in :meth:`set_up`.

        """

        self.equilibrium_equations: dict[Component, dict[Phase, pp.ad.Operator]]
        """A map containing equilibrium equations per component (key).

        Equilibrium equations are formulated between a phase (second key) and the
        reference phase.

        Per component, there are ``num_phases -1`` equilibrium equations.

        .. math::

            x_{ij} \\varphi_{ij} - x_{iR} \\varphi_{iR}~,~\\forall i~,~j \\neq R~.

        - :math:`z` : Component :attr:`~porepy.composite.base.Component.fraction`
        - :math:`y` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`x` : Phase :attr:`~porepy.composite.base.Phase.fraction_of` component
        - :math:`\\varphi` : Phase :attr:`~porepy.composite.base.Phase.fugacity_of`
          component

        This dictionary is filled in :meth:`set_up`.

        """

        self.complementary_conditions: dict[Phase, pp.ad.Operator]
        """A map containing complementary conditions per phase (key) as per the unified
        setting.

        .. math::

            y_j (1 - \\sum_i x_{ij})~,
            \\min \\{y_j, (1 - \\sum_i x_{ij}) \\}~\\forall j~.

        - :math:`y` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`x` : Phase :attr:`~porepy.composite.base.Phase.fraction_of` component

        Note that fraction of the reference phase :math:`y_R` is possibly represented
        through unity.

        Complementary conditions are either given as is, or in semi-smooth form
        (see this class' :meth:`set_up`).

        This dictionary is filled in :meth:`set_up`.

        """

    def set_up(
        self,
        ad_system: Optional[pp.ad.EquationSystem] = None,
        subdomains: list[pp.Grid] = None,
        eliminate_ref_phase: bool = True,
        eliminate_ref_feed_fraction: bool = True,
        semismooth_complementarity: bool = True,
    ) -> list[pp.Grid]:
        """Performs on top of the base class set-up the creation of equations relevant
        for non-reactive mixtures.

        These include:

        - mass constraints (:attr:`mass_constraints`)
        - equilibrium equations (:attr:`equilibrium_equations`)
        - complementary conditions (:attr:`complementary_conditions`)

        The equations are introduced into :attr:`system`.
        Names of set equations are stored :attr:`equations`.

        Parameters:
            semismooth_complementarity: ``default=True``

                If True, the complementary conditions are set using a semi-smooth
                min-function (see :attr:`complementary_conditions`).

        """
        domains = super().set_up(
            ad_system=ad_system,
            subdomains=subdomains,
            eliminate_ref_phase=eliminate_ref_phase,
            eliminate_ref_feed_fraction=eliminate_ref_feed_fraction,
        )

        # Setting up mass constraints
        mass_constraints: dict[Component, pp.ad.Operator] = dict()
        y_j = [phase.fraction for phase in self.phases]
        for comp in self.components:
            if comp != self.reference_component:
                constraint: pp.ad.Operator = evaluate_homogenous_constraint(
                    comp.fraction,
                    [phase.fraction_of[comp] for phase in self.phases],
                    y_j,
                )  # type: ignore
                constraint.set_name(f"mass-constraint-{comp.name}")
                mass_constraints.update({comp: constraint})
        self.mass_constraints = mass_constraints

        # Setting up equilibrium equations
        equilibrium: dict[Component, dict[Phase, pp.ad.Operator]] = dict()
        for comp in self.components:
            comp_equ: dict[Phase, pp.ad.Operator] = dict()
            for phase in self.phases:
                if phase != self.reference_phase:
                    equ = (
                        phase.fraction_of[comp] * phase.fugacity_of[comp]
                        - self.reference_phase.fraction_of[comp]
                        * self.reference_phase.fugacity_of[comp]
                    )
                    equ.set_name(
                        f"isofugacity-constraint-"
                        + f"{comp.name}-{phase.name}-{self.reference_phase.name}"
                    )
                    comp_equ.update({phase: equ})
            equilibrium.update({comp: comp_equ})
        self.equilibrium_equations = equilibrium

        # Setting up complementary conditions
        ss_min: pp.ad.Operator = pp.ad.SemiSmoothMin()
        cc_conditions: dict[Phase, pp.ad.Operator] = dict()
        for phase in self.phases:
            comp_unity: pp.ad.Operator = 1.0 - safe_sum(
                [phase.fraction_of[c] for c in self.components]
            )  # type: ignore
            if semismooth_complementarity:
                equ = ss_min(phase.fraction, comp_unity)
                equ.set_name(f"semismooth-complementary-condition-{phase.name}")
            else:
                equ = phase.fraction * comp_unity
                equ.set_name(f"complementary-condition-{phase.name}")
            cc_conditions.update({phase: equ})
        self.complementary_conditions = cc_conditions

        return domains

    def get_enthalpy_constraint(
        self, h: pp.ad.Operator, normalize: bool = False
    ) -> pp.ad.Operator:
        """Assembles the enthalpy constraint in AD operator form.

        .. math::

            \\sum_j y_j h_j  - h = 0~,~
            (\\sum_j y_j h_j) / h - 1= 0~

        - :math:`y_j` : Phase :attr:`~porepy.composite.base.Phase.fraction`
        - :math:`h_j` : Phase :attr:`~porepy.composite.base.Phase.enthalpy`

        Parameters:
            h: An operator representing the target mixture enthalpy :math:`h`.
            normalize: ``default=False``

                A flag to normalize above equation by dividing by the target enthalpy

        Returns:
            The right-hand side of above equation.

        """
        if normalize:
            constr: pp.ad.Operator = self.enthalpy / h - 1.0
        else:
            constr: pp.ad.Operator = self.enthalpy - h
        constr.set_name("enthalpy-constraint")
        return constr

    def get_volume_constraint(
        self, v: pp.ad.Operator, normalize: bool = False
    ) -> pp.ad.Operator:
        """Assembles the volume constraint in AD operator form using the reciprocal of
        the mixture density.

        .. math::

            \\dfrac{1}{\\sum_j s_j \\rho_j} - v = 0~,~
            v \\left(\\sum_j s_j \\rho_j\\right) - 1 = 0.

        - :math:`s_j` : Phase :attr:`~porepy.composite.base.Phase.saturation`
        - :math:`\\rho_j` : Phase :attr:`~porepy.composite.base.Phase.density`

        Parameters:
            v: An operator representing the target mixture volume :math:`v`.
            normalize: ``default=False``

                A flag to normalize above equation by dividing by the target volume

        Returns:
            The right-hand side of above equation.

        """
        if normalize:
            constr: pp.ad.Operator = v * self.density - 1.0
        else:
            constr: pp.ad.Operator = self.density ** (-1) - v
        constr.set_name("volume-constraint")
        return constr

    def get_density_conservation_for(
        self, phase: Phase, normalize: bool = False
    ) -> pp.ad.Operator:
        """Assembles the conservation of densities for a given phase.

        .. math::

            y_j \\rho - s_j \\rho_j = 0~,~
            y_j - s_j \\dfrac{\\rho_j}{rho} = 0

        with the mixture density :math:`\\rho = \\sum_k s_k \\rho_k`, assuming
        :math:`\\rho_k` is the density of a phase when saturated.

        These equations are required for isochoric flash specifications, which introduce
        necessarily another ``num_phases - 1`` unknowns, the saturations.

        Parameters:
            phase: A phase in this mixture.
            normalize: ``default=False``

                A flag to normalize above equation by dividing by the mixture density.

        Returns:
            The right-hand side of above equation.

        """
        assert phase in self._phases, "Unknown phase object."
        if normalize:
            constr = phase.fraction - phase.saturation * phase.density / self.density
        else:
            constr = phase.fraction * self.density - phase.saturation * phase.density
        constr.set_name(f"density-constraint-{phase.name}")
        return constr
