"""This module contains a class representing a multiphase multicomponent mixture
using the unified formulation for phase stability and split calculations.

The base class is a starting point to derive custom mixtures using an
equations of state, which must yield formulae for phase densities, specific
enthalpies and fugacity coefficients.
As of now, only those two thermodynamic properties of phases are relevant for the
p-T and p-h flash.

Fractional values in the compositional framework are set and initiated by respective
phase and component classes.

The user has to ensure that values for secondary variables are set
prior to calling a flash. This involves setting values for

- pressure
- temperature
- (specific) enthalpy
- feed fraction per component

Note:
    The temperature is a variable in the p-h flash.
    But the enthalpy is **not** a variable in the p-T flash. It can be evaluated after
    the flash is performed.

Note:
    Saturations are never variables in the flash. They (volumetric fraction) can be
    evaluated after the flash using (molar) phase fractions and the thermodynamic state.

Note:
    Due to various unity constraints, mixtures have a reference component and a
    reference phase.

    The fractions of the reference phase are eliminated by default by unity.

    The mass constraint of the reference component is also eliminated by unity of
    component feed fractions.

The thermodynamic variables can be accessed using
:func:`~porepy.composite.composition.Composition.p`,
:func:`~porepy.composite.composition.Composition.T` and
:func:`~porepy.composite.composition.Composition.h`.

Feed fractions variables can be accessed using
:func:`~porepy.composite.component.Component.fraction` of
:class:`~porepy.composite.component.Component`.

Phases have to be modelled for each composition class using the respective EoS,
see :class:`~porepy.composite.phase.Phase`.

Warning:
    As of now it is recommended to use more than one component.

    Due to the phase rule ``F=C-P+2``, the thermodynamic degree of freedom reduces to 1
    if ``C=1`` and ``P=2`` f.e.,
    causing the unified formulation to lose its injectivity.
    This leads to a potentially singular Jacobian of the system.

"""
from __future__ import annotations

import abc
from typing import Any, Generator, Literal, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS
from .component import Component, Compound
from .composite_utils import safe_sum
from .phase import Phase

__all__ = ["Mixture", "MixtureAD"]

FlashSystemDict = dict[
    Literal["equations", "primary-variables", "secondary-varsiables"], list[str]
]
"""A type alias for subsystem dictionaries which contain:

- 'equations': A list of names of equations belonging to this subsystem.
- 'primary_vars': A list of names of primary variables in this subsystem.
- 'secondary_vars': A list of names of secondary variables in this subsystem.

"""


class MixtureAD:
    """A storage class for mixtures, containing references to various relevant
    equations and/or definitions in the form of
    :class:`~porepy.numerics.ad.operators.Operator`.

    It also contains some generic operations which hold for numbers, arrays, AD-arrays
    and operators.

    This class is meant to be instantiated by a :class:`Composition` during
    construction.

    :meth:`set_up` can be used to couple the mixture with other model problems e.g.,
    flow and transport.

    Note:
        All created operators have their own, unique
        :meth:`~porepy.numerics.ad.operators.Operator.name` assigned here.
        I.e., this name can be used to add the operator as an equation to the respective
        :class:`~porepy.numerics.ad.equation_system.EquationSystem`.

    Parameters:
        mixture: The mixture class for which the equations should be assembled.
        ad_system: The AD system and its computational domain

    """

    def __init__(
        self,
        mixture: Mixture,
        ad_system: pp.ad.EquationSystem,
    ) -> None:

        self._mix: Mixture = mixture

        ### PUBLIC

        self.pT_subsystem: FlashSystemDict = dict()
        """A dictionary representing the subsystem for the p-T flash.

        The equations of the p-T subsystem are:

        - mass constraint per component, except reference component
          (``num_components - 1``). If only one component is present, one equation is
          set up.
        - equilibrium equations per component, between each phase and the reference
          phase
          (``num_components * (num_phases - 1)``)

        The primary variables are:

        - molar phase fraction, except reference phase (``num_phases - 1``)
        - fractions of components in phases (``num_components * num_phases``)

        """

        self.ph_subsystem: FlashSystemDict = dict()
        """A dictionary representing the subsystem for the p-h flash.

        Additionally to the information in :data:`pT_subsystem`, the p-h subsystem
        contains

        - enthalpy constraint (1 equation)
        - temperature (1 primary variable)

        """

        self.system: pp.ad.EquationSystem = ad_system

        self.reference_phase_eliminated: bool
        """The flag passed at instantiation."""

        self.p: pp.ad.Operator
        """The pressure variable for the thermodynamic state.

        The values are assumed to represent values at equilibrium and are therefore
        constant during any flash procedure.

        The name of this variable is composed of the general symbol
        (see :data:`~porepy.composite._composite_utils.VARIABLE_SYMBOLS`).

        | Math. Dimension:        scalar
        | Phys. Dimension:        [MPa] = [MN / m^2]

        """

        self.h: pp.ad.Operator
        """The specific molar enthalpy variable for the thermodynamic state.

        For the isenthalpic flash, the values are assumed to represent values at
        equilibrium.
        For the isothermal flash, the enthalpy changes based on the results
        (composition) and should be evaluated afterwards using
        :meth:`evaluate_specific_enthalpy`.

        The name of this variable is composed of the general symbol
        (see :data:`~porepy.composite._composite_utils.VARIABLE_SYMBOLS`).

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kJ / mol / K]

        """

        self.T: pp.ad.Operator
        """The temperature variable for the thermodynamic state.

        For the isothermal flash, the values are assumed to represent values at
        equilibrium.
        For the isenthalpic flash, the temperature varies and depends on the enthalpy
        and the composition. Its values are determined by the isenthalpic flash
        procedure.

        The name of this variable is composed of the general symbol
        (see :data:`~porepy.composite._composite_utils.VARIABLE_SYMBOLS`).

        | Math. Dimension:        scalar
        | Phys. Dimension:        [K]

        """

        self.z_R: pp.ad.Operator
        """A representation of the
        :meth:`~porepy.composite.component.Component.fraction` of the
        :meth:`~Composition.reference_component` by unity, using the fractions
        of other present components.

        """

        self.y_R: pp.ad.Operator
        """A representation of the
        :meth:`~porepy.composite.phase.Phase.fraction` of the
        :meth:`~Composition.reference_phase` by unity, using the fractions
        of other present phases.

        """

        self.s_R: pp.ad.Operator
        """A representation of the
        :meth:`~porepy.composite.phase.Phase.saturation` of the
        :meth:`~Composition.reference_phase` by unity, using the saturations
        of other present phases.

        """

        self.component_fraction_unity: pp.ad.Operator
        """A representation of the unity of component feed fractions."""

        self.phase_fraction_unity: pp.ad.Operator
        """A representation of the unity of phase fractions."""

        self.phase_saturation_unity: pp.ad.Operator
        """A representation of the unity of phase saturations."""

        self.enthalpy_constraint: pp.ad.Operator
        """A representation of the mixture enthalpy constraint."""

        self.composition_unity_of_phase: dict[Phase, pp.ad.Operator] = dict()
        """A dictionary containing per present phase (key) its composition unity in
        operator form. """

        self.mass_constraint_of_component: dict[Component, pp.ad.Operator] = dict()
        """A dictionary containing per present component (key) its mass constraint in
        operator form.

        Note:
            For the reference phase, the phase fraction is represented by unity itself,
            if ``eliminate_ref_phase`` is ``True``.

        """

        self.complementary_condition_of_phase: dict[Phase, pp.ad.Operator] = dict()
        """A dictionary containing per present phase (key) its complementary condition
        in operator form.

        The CC is formed by the multiplication of the phase fraction with the
        respective phase composition unity.

        Note:
            For the reference phase, the phase fraction is represented by unity itself,
            if ``eliminate_ref_phase`` is ``True``.

        """

        self.saturation_equation_of_phase: dict[Phase, pp.ad.Operator] = dict()
        """A dictionary containing per present phase (key) its nonlinear relation
        between the molar and volumetric fraction (saturation), depending on the
        ratio between mixture density and phase density."""

        self.equilibrium_equations: dict[
            Component, dict[Phase, pp.ad.Operator]
        ] = dict()
        """A dictionary containing per component (key) another dictionary,
        which contains an equilibrium equation between a phase (second-key) and the
        reference phase."""

        self.rho_mix: pp.ad.Operator
        """An operator representing the mixture density."""

        self.h_mix: pp.ad.Operator
        """An operator representing the mixture enthalpy as a function of the phase
        enthalpies and phase fractions"""

    def set_up(
        self,
        pressure_var: Optional[pp.ad.MixedDimensionalVariable] = None,
        temperature_var: Optional[pp.ad.MixedDimensionalVariable] = None,
        enthalpy_var: Optional[pp.ad.MixedDimensionalVariable] = None,
        eliminate_ref_phase: bool = True,
    ):
        """Perform the AD set-up:
        assembly of equation and assigning of various operators.

        Important:
            The thermodynamic state in terms of pressure, temperature and enthalpy can
            be given by the input.

            If they are not given, this method will create AD representations of
            respective variables.

            This done so that the user can couple the compositional framework
            with p-T-h variables used in other models such as flow and transport.

        Parameters:
            pressure_var: ``default=None``

                The pressure variable in the AD system.
            temperature_var: ``default=None``

                The temperature variable in the AD system.
            enthalpy_var: ``default=None``

                The enthalpy variable in the AD system.
            eliminate_reference_phase: ``default=True``

                An optional flag to eliminate reference phase variables from the
                system, and hence reduce the system.

                If True, no assembled equation or operator will have a dependency
                on the molar and volumetric fractions of the reference-phase.


        Raises:
            AssertionError: If the mixture is empty (no components).
            AssertionError: If less than 2 phases are modelled.

        """
        MIX = self._mix
        # assert non-empty mixture
        assert MIX.num_components >= 2, "Mixture modelled with only one component."
        # assert there are at least 2 phases modelled
        assert MIX.num_phases >= 2, "Mixture modelled with only one phase."

        self.reference_phase_eliminated = eliminate_ref_phase

        subdomains = self.system.mdg.subdomains()
        if pressure_var is None:
            pressure_var = self.system.create_variables(
                COMPOSITIONAL_VARIABLE_SYMBOLS["pressure"], subdomains=subdomains
            )
        if temperature_var is None:
            temperature_var = self.system.create_variables(
                COMPOSITIONAL_VARIABLE_SYMBOLS["temperature"], subdomains=subdomains
            )
        if enthalpy_var is None:
            enthalpy_var = self.system.create_variables(
                COMPOSITIONAL_VARIABLE_SYMBOLS["enthalpy"], subdomains=subdomains
            )

        self.p = pressure_var
        self.T = temperature_var
        self.h = enthalpy_var

        ### Creation of AD operators
        phases = list(MIX.phases)
        components = list(MIX.components)
        reference_phase = MIX.reference_phase
        reference_component = MIX.reference_component

        # reference feed fraction by unity
        z_R = self.evaluate_unity(
            *[
                component.fraction
                for component in components
                if component != reference_component
            ]
        )
        z_R.set_name("ref-feed-frac-by-unity")
        self.z_R = z_R

        # reference phase fraction by unity
        y_R = self.evaluate_unity(
            *[phase.fraction for phase in phases if phase != reference_phase]
        )
        y_R.set_name("ref-phase-frac-by-unity")
        self.y_R = y_R

        # reference phase saturation by unity
        s_R = self.evaluate_unity(
            *[phase.saturation for phase in phases if phase != reference_phase]
        )
        s_R.set_name("ref-phase-sat-by-unity")
        self.s_R = s_R

        # component feed fraction unity
        feed_unity = self.evaluate_unity(
            *[component.fraction for component in components]
        )
        feed_unity.set_name("component-fraction-unity")
        self.component_fraction_unity = feed_unity

        # phase fraction unity
        phase_frac_unity = self.evaluate_unity(*[phase.fraction for phase in phases])
        phase_frac_unity.set_name("phase-fraction-unity")
        self.phase_fraction_unity = phase_frac_unity

        # phase saturation unity
        phase_saturation_unity = self.evaluate_unity(
            *[phase.saturation for phase in phases]
        )
        phase_saturation_unity.set_name("phase-saturation-unity")
        self.phase_saturation_unity = phase_saturation_unity

        # phase composition unities
        for phase in phases:
            unity = self.evaluate_unity(
                *[phase.fraction_of_component(component) for component in components]
            )
            unity.set_name(f"composition-unity-phase-{phase.name}")
            self.composition_unity_of_phase.update({phase: unity})

        # mass constraints
        Y = [
            self.y_R
            if phase == reference_phase and eliminate_ref_phase
            else phase.fraction
            for phase in phases
        ]
        for component in components:
            X = [phase.fraction_of_component(component) for phase in phases]
            mass_constraint = self.evaluate_mass_constraint(component.fraction, Y, X)
            mass_constraint.set_name(f"mass-constraint-component-{component.name}")
            self.mass_constraint_of_component.update({component: mass_constraint})

        # complementary conditions
        # semi-smooth min operator
        min: pp.ad.Operator = pp.ad.SemiSmoothMin()
        for y, phase in zip(Y, phases):
            cc = min(y, self.composition_unity_of_phase[phase])
            cc.set_name(f"complementary-condition-phase-{phase.name}")
            self.complementary_condition_of_phase.update({phase: cc})

        # mixture density
        S = [
            self.s_R
            if phase == reference_phase and eliminate_ref_phase
            else phase.saturation
            for phase in phases
        ]
        X = dict(
            [
                (
                    phase,
                    [
                        phase.normalized_fraction_of_component(component)
                        for component in components
                    ],
                )
                for phase in phases
            ]
        )
        self.rho_mix = safe_sum(
            [
                s * phase.density(self.p, self.T, *X[phase])
                for s, phase in zip(S, phases)
            ]
        )  # type: ignore

        # saturation equations
        for y, s, phase in zip(Y, S, phases):
            s_equ = self.rho_mix * y - phase.density(self.p, self.T, *X[phase]) * s
            s_equ.set_name(f"saturation-equation-{phase.name}")
            self.saturation_equation_of_phase.update({phase: s_equ})

        # enthalpy constraint
        phase_enthalpies = list()
        for y, phase in zip(Y, phases):
            phase_enthalpies.append(
                y * phase.specific_enthalpy(self.p, self.T, *X[phase])
            )
        self.h_mix = safe_sum(phase_enthalpies)
        self.h_mix.set_name("mixture-enthalpy-as-function")
        self.enthalpy_constraint = self.h - self.h_mix
        self.enthalpy_constraint.set_name("mixture-enthalpy-constraint")

        # equilibrium equations
        for component in components:
            self.equilibrium_equations[component] = dict()
            for other_phase in phases:
                if other_phase != reference_phase:
                    equilibrium = other_phase.fugacity_of(
                        component, self.p, self.T, *X[other_phase]
                    ) * other_phase.fraction_of_component(
                        component
                    ) - reference_phase.fugacity_of(
                        component, self.p, self.T, *X[reference_phase]
                    ) * reference_phase.fraction_of_component(
                        component
                    )
                    equilibrium.set_name(
                        f"equilibrium-{component.name}-"
                        + f"{phase.name}-{reference_phase.name}"
                    )
                    self.equilibrium_equations[component][phase] = equilibrium

        ### Adding relevant equations to the AD-system
        # allocating subsystems
        equations: list[pp.ad.Operator] = list()
        pT_subsystem: FlashSystemDict = {
            "equations": list(),
            "primary-variables": list(),
            "secondary-varsiables": list(),
        }
        ph_subsystem: FlashSystemDict = {
            "equations": list(),
            "primary-variables": list(),
            "secondary-varsiables": list(),
        }
        self._set_subsystem_vars(ph_subsystem, pT_subsystem)

        # Mass constraint per component, except reference component
        for component in MIX.components:
            if component != MIX.reference_component:
                equation = self.mass_constraint_of_component[component]
                equations.append(equation)
                pT_subsystem["equations"].append(equation.name)
                ph_subsystem["equations"].append(equation.name)

        # enthalpy constraint for p-H flash
        equation = self.enthalpy_constraint
        equations.append(equation)
        ph_subsystem["equations"].append(equation.name)

        # equilibrium equation
        for component in MIX.components:
            for phase in MIX.phases:
                if phase != reference_phase:
                    equation = self.equilibrium_equations[component][phase]
                    equations.append(equation)
                    pT_subsystem["equations"].append(equation.name)
                    ph_subsystem["equations"].append(equation.name)

        # semi-smooth complementary conditions
        # NOTE: They are part of the system, but not of the subsystems.
        # They are only used by the newton-min in the flash procedure.
        for phase in MIX.phases:
            equation = self.complementary_condition_of_phase[phase]
            equations.append(equation)
            # pT_subsystem["equations"].append(equation.name)
            # ph_subsystem["equations"].append(equation.name)

        # adding equations to system
        # every equation in the unified flash is a cell-wise scalar equation
        for equation in equations:
            self.system.set_equation(
                equation,
                grids=subdomains,
                equations_per_grid_entity={"cells": 1},
            )

        # storing references to the subsystems
        self.pT_subsystem = pT_subsystem
        self.ph_subsystem = ph_subsystem

    def _set_subsystem_vars(
        self,
        ph_subsystem: FlashSystemDict,
        pT_subsystem: FlashSystemDict,
    ) -> None:
        """Auxiliary function to set the variables in respective subsystems."""

        MIX = self._mix
        ### FLASH SECONDARY VARIABLES
        # pressure is always a secondary var in the flash
        pT_subsystem["secondary-varsiables"].append(self.p.name)
        ph_subsystem["secondary-varsiables"].append(self.p.name)
        # enthalpy is always a secondary var in the flash
        ph_subsystem["secondary-varsiables"].append(self.h.name)
        pT_subsystem["secondary-varsiables"].append(self.h.name)
        # Temperature is only secondary in the p-T flash because it is fixed
        # It varies in the p-h flash.
        pT_subsystem["secondary-varsiables"].append(self.T.name)
        # feed fractions are always secondary vars
        for component in MIX.components:
            pT_subsystem["secondary-varsiables"].append(component.fraction.name)
            ph_subsystem["secondary-varsiables"].append(component.fraction.name)
        # saturations are always secondary vars
        for phase in MIX.phases:
            pT_subsystem["secondary-varsiables"].append(phase.saturation.name)
            ph_subsystem["secondary-varsiables"].append(phase.saturation.name)
        # molar fraction of the reference phase is always a secondary var
        pT_subsystem["secondary-varsiables"].append(MIX.reference_phase.fraction.name)
        ph_subsystem["secondary-varsiables"].append(MIX.reference_phase.fraction.name)
        # solute fractions in compounds are always secondary vars in the flash
        for component in MIX.components:
            if isinstance(component, Compound):
                for solute in component.solutes:
                    solute_fraction_name = component.solute_fraction_name(solute)
                    pT_subsystem["secondary-varsiables"].append(solute_fraction_name)
                    ph_subsystem["secondary-varsiables"].append(solute_fraction_name)

        ### FLASH PRIMARY VARIABLES
        # for the p-h flash, T is an additional var
        ph_subsystem["primary-variables"].append(self.T.name)
        # phase fractions
        for phase in MIX.phases:
            if phase != MIX.reference_phase:
                pT_subsystem["primary-variables"].append(phase.fraction.name)
                ph_subsystem["primary-variables"].append(phase.fraction.name)
            # phase composition
            for component in phase:
                var_name = phase.fraction_of_component(component).name
                pT_subsystem["primary-variables"].append(var_name)
                ph_subsystem["primary-variables"].append(var_name)

    def assemble_Gibbs(
        self,
        variables: list[str],
        other_equations: Optional[list[str]] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Assembles the first-order condition for the Gibbs energy including the
        equality constraints for mass.

        The (by default) evaluated operators are

        - :data:`mass_constraint_of_component`, except for the reference component
        - :data:`equilibrium_equations`, which is the unconstrained first-order cond.

        It calls :meth:`Mixture.precompute` as a first step by passing ``state``.

        Parameters:
            variables: List of variables w.r.t. which the derivative should be returned.

                Use this to include f.e. temperature for the p-h flash.
            other_equations: ``default=None``

                Use this to include other equations in the Ad system to shorten the
                assembly time e.g., the enthalpy constraint for the p-h flash
            state: ``default=None``

                A state vector w.r.t. which the system should be assembled
                (see :meth:`~porepy.numerics.ad.ad_system.assemble_subsystem`).

        Returns:
            The linearized system. The order of equations is as found in
            :data:`pT_subsystem`, followed by the equations in ``other_equations``.

        """
        equations = self.pT_subsystem["equations"]
        self._mix.precompute(state=state)
        return self.system.assemble_subsystem(
            equations + other_equations if other_equations else equations,
            variables,
            state=state,
        )

    def evaluate_Gibbs(
        self,
        p: NumericType,
        T: NumericType,
        Z: list[NumericType],
        Y: list[NumericType],
        X: list[list[NumericType]],
        h: Optional[NumericType] = None,
    ) -> NumericType:
        """Evaluate the first-order conditions for the Gibbs energy, including the
        mass constraints, for a given thermodynamic state.

        Optionally the enthalpy constraint can be evaluated as well.

        This functions intends to give an alternative to the AD framework, bypassing
        its evaluation by directly calling respective functions.

        Its intended use is for the quick evaluation of the residual of the system.

        Note:
            You can use :meth:`get_compositional_state` to obtain the right input for
            this method.

        Parameters:
            p: Pressure values.
            T: Temperature values.
            Z: A list of component feed fractions in correct order.
            Y: A list of phase molar fractions in correct order.
            X: Phase compositions, where the outer list follows to the order of
                phases and the inner lists follow the order of components.
            h: ``default=None``

                If given, the enthalpy constraint will be evaluated as well and
                appended.

        Returns:
            The evaluation of the mass constraints, excluding the reference component,
            and the equilibrium conditions (fugacities).

            If ``h`` is given, the enthalpy constraint is evaluated and appended at the
            end.

        """
        # evaluating the mass constraints
        ads = self.system
        components = list(self._mix.components)
        phases = list(self._mix.phases)

        mass_constraints = list()

        r_idx = components.index(self._mix.reference_component)
        for i, z in enumerate(Z):
            if i != r_idx:
                X_i = [X_[i] for X_ in X]
                mass_constraints.append(self.evaluate_mass_constraint(z, Y, X_i))

    @staticmethod
    def evaluate_unity(*x: tuple[Any]) -> Any:
        """Returns ``1 - sum(x)`` with 1 as a Python integer."""
        return 1 - safe_sum(x)

    @staticmethod
    def evaluate_mass_constraint(z: Any, y: list[Any], x: list[Any]) -> Any:
        """Returns ``z - sum(y*x)``."""
        return z - safe_sum([y_ * x_ for y_, x_ in zip(y, x)])

    @staticmethod
    def evaluate_complementary_condtion(y: Any, x: list[Any]) -> Any:
        """Returns ``y * (1 - sum(x))`` with 1 as a Python integer."""
        return y * (1 - safe_sum(x))

    def get_compositional_state(
        self,
        state: Optional[np.ndarray] = None,
        derivatives: Optional[list[str]] = None,
        as_ad: bool = False,
    ) -> tuple[
        NumericType,
        NumericType,
        NumericType,
        list[NumericType],
        list[NumericType],
        list[list[NumericType]],
    ]:
        """Extract compositional variables from a global AD state vector.

        Parameters:
            state: ``default=None``

                A state vector from which the compositional state should be extracted.
                (see :meth:`~porepy.numerics.ad.ad_system.assemble_subsystem`).

                If ``None``, the values stored as ``ITERATE`` are obtained.
            derivatives: ``default=None``

                A list of variables for which derivatives in the AD-array should be
                included, if ``as_ad`` is ``True``.

                If given, the Jacobian of the assembled AD-arrays will be sliced
                accordingly.
            as_ad: ``default=False``

                If ``True``, the variables are returned as
                :class:`~porepy.numerics.ad.forward_mode.Ad_array` containing
                the global derivatives, as assigned by the AD framework.

                If ``False``, only the values in form of numpy arrays are returned.

        Returns:
            A tuple containing

            1. pressure,
            2. temperature,
            3. enthalpy,
            4. list of feed fractions,
            5. list of phase fractions
            6. list of lists of phase compositions

            in respective numeric format.

            The order of phase fractions corresponds to the order of phases given
            by :meth:`phases`.

            The list of lists, contains phase compositions per phase in :meth:`phases`.
            The compositions are again ordered as given by :meth:`components`.

        """
        ads = self.system
        components = list(self._mix.components)
        phases = list(self._mix.phases)
        # If derivatives requested, utilize the AD framework to get the correct
        # derivatives
        if as_ad:
            p = self.p.evaluate(ads, state)
            T = self.T.evaluate(ads, state)
            h = self.h.evaluate(ads, state)
            Z = [component.fraction.evaluate(ads, state) for component in components]
            Y = [phase.fraction.evaluate(ads, state) for phase in phases]
            X = [
                [
                    phase.fraction_of_component(component).evaluate(ads, state)
                    for component in components
                ]
                for phase in phases
            ]

            # if only certain derivatives are requested, slice the Jacobians
            if derivatives:
                projection = ads.projection_to(derivatives).transpose()
                p.jac = p.jac * projection
                T.jac = T.jac * projection
                h.jac = h.jac * projection
                Z = [pp.ad.Ad_array(z.val, z.jac * projection) for z in Z]
                Y = [pp.ad.Ad_array(y.val, y.jac * projection) for y in Y]
                X = [
                    [pp.ad.Ad_array(x.val, x.jac * projection) for x in X_] for X_ in X
                ]
        # Otherwise we extract only the values using functions of the AD system
        else:
            if state:
                p = state[ads.dofs_of([self.p.name])]
                T = state[ads.dofs_of([self.T.name])]
                h = state[ads.dofs_of([self.h.name])]
                Z = [
                    state[ads.dofs_of([component.fraction.name])]
                    for component in components
                ]
                Y = [state[ads.dofs_of([phase.fraction.name])] for phase in phases]
                X = [
                    [
                        state[
                            ads.dofs_of([phase.fraction_of_component(component).name])
                        ]
                        for component in components
                    ]
                    for phase in phases
                ]

                # if this happens, above index slicing extracts numbers from arrays
                # wrap values back into arrays in that case
                if ads.mdg.num_subdomain_cells() == 1:
                    p = np.array([p])
                    T = np.array([T])
                    h = np.array([h])
                    Z = [np.array([z]) for z in Z]
                    Y = [np.array([y]) for y in Y]
                    X = [[np.array(x) for x in X_] for X_ in X]
            else:
                p = ads.get_variable_values([self.p.name], True)
                T = ads.get_variable_values([self.T.name], True)
                h = ads.get_variable_values([self.h.name], True)
                Z = [
                    ads.get_variable_values([component.fraction.name], True)
                    for component in components
                ]
                Y = [
                    ads.get_variable_values([phase.fraction.name], True)
                    for phase in phases
                ]
                X = [
                    [
                        ads.get_variable_values(
                            [phase.fraction_of_component(component).name], True
                        )
                        for component in components
                    ]
                    for phase in phases
                ]

        return p, T, h, Z, Y, X


class Mixture(abc.ABC):
    """Base class for all multiphase, multicomponent mixture models
    in the unified setting.

    This class can be used to model mixtures s which implement a specific
    equations of state.

    It serves as a container for various phase and component classes,
    as well as for PorePy's Ad functionality.

    The equations in AD form are created and stored in an instance of
    :class:`MixtureAD` (see :meth:`AD`).

    Notes:
        - The first, added phase is treated as the reference phase.
          Its molar fraction will not be part of the primary variables.
        - The first, added component is set as reference component.
          Its mass conservation will not be part of the flash equations.
        - Choice of reference phase and component influence the choice of equations and
          variables, keep that in mind. It might have numeric implications.

    Important:
        If the user wants to model a single-component mixture, a dummy component must be
        added as the first component (reference component for elimination),
        with a feed fraction close to machine precision.

        This approximates a single-component mixture.

        This is due to the flash system being inherently singular in this case.
        Numerical issues can appear if done so!

    Parameters:
        ad_system: ``default=None``

            If given, this class will use the AD system and the respective
            mixed-dimensional domain to represent all involved variables cell-wise in
            each subdomain.

            If not given (None), a single-cell domain and respective AD system are
            created.
        nc: ``default=1``

            Number of cells for the default AD system (and its grid).

            Use this to vectorize the flash procedure, such that multiple different
            thermodynamic states are set in vector form and the flash system
            is assembled in a block-diagonal manner.

            Only used if `ad_system=None` and the default system is created.

            Warning:
                In some problematic cases, the vectorization causes a purely
                mathematical coupling between the formally independent flash cases.

                This is due to the condition number of the Flash system being inherently
                high. Over-iterations necessary for problematic vector-components
                can cause convergence issues for other, already converged
                vector-components.

    """

    def __init__(
        self, ad_system: Optional[pp.ad.EquationSystem] = None, nc: int = 1
    ) -> None:

        if ad_system is None:
            sg = pp.CartGrid([nc, 1], [1, 1])
            mdg = pp.MixedDimensionalGrid()
            mdg.add_subdomains(sg)
            mdg.compute_geometry()

            ad_system = pp.ad.EquationSystem(mdg)  # type: ignore

        # modelled phases and components
        self._components: list[Component] = list()
        """A list containing all modelled components."""
        self._phases: list[Phase] = list()
        """A list containing all modelled phases."""

        # storage for AD operators
        # TODO This is a memory leak due to mutual reference
        self._AD: MixtureAD = MixtureAD(self, ad_system)

    def __str__(self) -> str:
        """Returns string representation of the composition,
        with information about present components.
        """
        out = f"Composition with {self.num_components} components:"
        for component in self.components:
            out += f"\n\t{component.name}"
        out += f"\nand {self.num_phases} phases:"
        for phase in self.phases:
            out += f"\n\t{phase.name}"
        return out

    ### Composition Management ---------------------------------------------------------

    @property
    def AD(self) -> MixtureAD:
        """The storage class for this mixture's equations and variables.

        Raises:
            AssertionError: If the mixture is not initialized, i.e. respective operators
                where not assigned.

        """
        assert (
            self._AD is not None
        ), "Mixture not initialized: AD operators not assigned."
        return self._AD

    @property
    def num_components(self) -> int:
        """Number of components in the composition."""
        return len(self._components)

    @property
    def num_phases(self) -> int:
        """Number of *modelled* phases in the composition."""
        return len(self._phases)

    @property
    def num_equilibrium_equations(self) -> int:
        """Number of necessary equilibrium equations for this composition, based on the
        number of added components and modelled phases."""
        return self.num_components * (self.num_phases - 1)

    @property
    def components(self) -> Generator[Component, None, None]:
        """
        Yields:
            Components added to the composition.

        """
        for C in self._components:
            yield C

    @property
    def phases(self) -> Generator[Phase, None, None]:
        """
        Yields:
            Phases modelled by the composition class.

        """
        for P in self._phases:
            yield P

    @property
    def reference_phase(self) -> Phase:
        """Returns the reference phase.

        As of now, the first, added phase is declared as reference phase.

        The fraction of the reference can be eliminated from the system using respective
        flags.

        Raises:
            AssertionError: If no phases were added to the mixture.

        """
        # assert the child classes has a non-empty list of phases
        assert self._phases, "No phases present in mixture."
        return self._phases[0]

    @property
    def reference_component(self) -> Component:
        """Returns the reference component.

        As of now, the first, added component is declared as reference component.

        The mass balance of the reference component can be eliminated from the system.

        Raises:
            AssertionError: If no components were added to the mixture

        """
        assert self._components, "No components present in mixture."
        return self._components[0]

    def add(self, components: list[Component], phases: list[Phase]) -> None:
        """Adds one or multiple components and phases to the mixture.

        Components and phases must be added before the system can be set up.

        Important:
            This method is meant to be called only once per mixture!

            This is due to the creation of respective AD variables.
            By calling it twice, their reference is overwritten and the previous
            variables remain as dangling parts in the AD system.

        Parameters:
            component: Component(s) to be added to this mixture.
            phases: Phase(s) to be added to this mixture.

        Raises:
            ValueError: If a component or phase was instantiated using a different
                AD system than the one used for this composition.

        """
        doubles = list()

        for comp in components:
            # sanity check when using the AD framework
            if self.AD.system != comp.ad_system:
                raise ValueError(
                    f"Component '{comp.name}' instantiated with a different AD system."
                )
            # Avoid double components
            if comp in doubles:
                # already added components are skipped
                continue
            else:
                doubles.append(comp)

            # add component
            self._components.append(comp)

        for phase in phases:
            # sanity check when using the AD framework
            if self.AD.system != phase.ad_system:
                raise ValueError(
                    f"Phase '{phase.name}' instantiated with a different AD system."
                )
            # avoid double phases
            if phase in doubles:
                continue
            else:
                doubles.append(phase)

            # add component
            self._phases.append(phase)

        # adding all components to every phase, according to unified procedure
        for phase in self.phases:
            phase.components = list(self.components)

    ### Computational methods ----------------------------------------------------------

    def precompute(
        self,
        state: Optional[np.ndarray] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """(Optional) Helper method for computing thermodynamic properties of the
        mixture.

        It is called inside :meth:`MixtureAD.linearize_subsystem`.

        This method is intended as a helper method for the user and the flash,
        to perform potentially expensive computations only once.

        This is motivated by the fact that many properties are related to some extent,
        and some parts can be computed only once for all of them.
        Otherwise the AD-framework will force the user to perform the computations
        once per equation in the model, which might be too expensive.

        Override this method for specific mixture models.

        It is intentionally not declared as an abstract method.

        Parameters:
            state: ``default=None``

                An optional (global) state vector for the AD system, containing the
                thermodynamic state of the system.

                Important:
                    The vector is *global* in the AD sense. It can contain elements
                    of other variables not related to the compositional framework.

                    Use :meth:`compositional_variables_from_state` to get relevant
                    quantities.

        """
        pass
