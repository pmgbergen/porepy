"""Experimentel code for efficient unified flash calculations using numba and sympy."""
from __future__ import annotations

from typing import Any, Generator, Literal, Optional, overload

import numpy as np
import scipy.sparse as sps
import sympy as sm

import porepy as pp

from .._core import COMPOSITIONAL_VARIABLE_SYMBOLS as SYMBOLS
from ..component import Component, Compound
from ..composite_utils import normalize_fractions, safe_sum
from ..phase import Phase, PhaseProperties
from ..mixture import NonReactiveMixture, BasicMixture

from .eos import PengRobinsonEoS
from .mixing import VanDerWaals

__all__ = ['MixtureSymbols', 'PR_Compiler']


class MixtureSymbols:
    """A class containing basic symbols (thermodynamic properties and variables) for a
    mixture represented using ``sympy``.
    
    It is meant for symbols which are considered primary or independent e.g., pressure
    or molar fractions, not for higher expressions.
    
    """

    def __init__(self, mixture: BasicMixture) -> None:
        
        self._all: dict[str, dict] = dict()
        """Shortcut map between symbol names and the public dictionary they are stored
        in."""

        self.mixture_properties: dict[str, sm.Symbol] = {
            "pressure": sm.Symbol(f"{SYMBOLS['pressure']}"),
            "temperature": sm.Symbol(f"{SYMBOLS['temperature']}"),
            "enthalpy": sm.Symbol(f"{SYMBOLS['enthalpy']}"),
            "volume": sm.Symbol(f"{SYMBOLS['volume']}"),
        }
        """A map containing the symbols for mixture properties considered uniform across
        all phases."""

        self.feed_fractions: dict[str, sm.Symbol] = {}
        """A map containing the symbols for feed fractions (overall molar fraction) per
        component.
        
        The feed fraction of the reference component is eliminated by unity.
        
        """

        self.phase_fractions: dict[str, sm.Symbol] = {}
        """A map containing the symbols for phase molar fractions per phase.

        The molar fraction of the reference phase is eliminated by unity.

        """

        self.phase_saturations: dict[str, sm.Symbol] = {}
        """A map containing the symbols for phase volumetric fractions per phase.

        The saturation of the reference phase is eliminate by unity.

        """

        self.phase_compositions: dict[str, sm.Symbol] = dict()
        """A map containing the symbols for molar component fractions in phases."""

        # expression for reference component
        # we do this inconvenient way to preserve the order of components
        s_ = list()
        name_r = ''
        for comp in mixture.components:
            name = f"{SYMBOLS['component_fraction']}_{comp.name}"
            if comp == mixture.reference_component:
                name_r = name
                self.feed_fractions.update({name: 0})
            else:
                s = sm.Symbol(name)
                self.feed_fractions.update({name: s})
                s_.append(s)
        name = f"{SYMBOLS['component_fraction']}_{mixture.reference_component.name}"
        self.feed_fractions.update(
            {name_r: 1 - safe_sum(s_)}
        )

        # expressions for reference phase
        y_ = list()
        s_ = list()
        name_y_r = ''
        name_s_r = ''
        for phase in mixture.phases:
            # symbols for fractions of components in phases
            for comp in mixture.components:
                name = f"{SYMBOLS['phase_composition']}_{comp.name}_{phase.name}"
                self.phase_compositions.update({name: sm.Symbol(name)})

            name_y = f"{SYMBOLS['phase_fraction']}_{phase.name}"
            name_s = f"{SYMBOLS['phase_saturation']}_{phase.name}"
            # skip reference phase due to dependency
            if phase == mixture.reference_phase:
                name_y_r = name_y
                name_s_r = name_s
                self.phase_fractions.update({name_y: 0})
                self.phase_saturations.update({name_s: 0})
            else:
            # symbols for fractions related to phases
                s = sm.Symbol(name_y)
                self.phase_fractions.update({name_y: s})
                y_.append(s)

                s = sm.Symbol(name_s)
                self.phase_saturations.update({name: s})
                s_.append(s)
        self.phase_fractions.update({name_y_r: 1 - safe_sum(y_)})
        self.phase_saturations.update({name_s_r: 1 - safe_sum(s_)})

        # construct shortcut mapping for general access functionality
        for k in self.mixture_properties.keys():
            self._all.update({k: self.mixture_properties})
        for k in self.feed_fractions.keys():
            self._all.update({k: self.feed_fractions})
        for k in self.phase_fractions.keys():
            self._all.update({k: self.phase_fractions})
        for k in self.phase_saturations.keys():
            self._all.update({k: self.phase_saturations})
        for k in self.phase_compositions.keys():
            self._all.update({k: self.phase_compositions})

    def __call__(self, symbol_name: str) -> sm.Symbol:
        """
        Parameters:
            symbol_name: A name of any symbol contained in this class.

        Raises:
            KeyError: If ``symbol_name`` is unknown.

        Returns:
            The symbol with the passed name.

        """
        d = self._all.get(symbol_name, None)

        if d is None:
            raise KeyError(f"No symbol with name '{symbol_name}' available.")
        else:
            return d[symbol_name]


class PR_Compiler:
    """Class implementing JIT-compiled representation of the equilibrium equations
    using numba and sympy, based on the Peng-Robinson EoS.

    It uses the no-python mode of numba to produce compiled code with near-C-efficiency.
    
    """

    def __init__(self, mixture: NonReactiveMixture) -> None:
        
        self._n_p = mixture.num_phases
        self._n_c = mixture.num_components

        self.symbols: MixtureSymbols = MixtureSymbols(mixture)
        """A container class for symbols representing (potentially) independent
        variables in the equilibrium problem."""

        self.unknowns: dict[str, list[sm.Symbol]] = dict()
        """A map between flash-types (p-T, p-h,...) and the respective list of
        independent variables as symbols."""

        self.arguments: dict[str, list[sm.Symbol]] = dict()
        """A map between flash-types (p-T, p-h,...) and the respective list of
        arguments required for the evaluation of the equations and Jacobians.
        
        The arguments include the fixed values and the unknowns.
        The fixed values for each flash type are the respective specifications e.g.,
        for the p-T flash it includes the symbol for p and T, followed by the symbolic
        unknowns for the p-T flash."""

        self.equations: dict[str, sm.Expr] = dict()
        """A map between flash-types (p-T, p-h,...) and the equilibrium equations
        represented by multivariate, vector-valued callables."""

        self.jacobians: dict[str, sm.Expr] = dict()
        """A map between flash-types (p-T, p-h,...) and the Jacobian of respective
        equilibrium equations."""


        self._define_unknowns_and_arguments(mixture)
        self._define_equations_and_jacobians(mixture)

    def _define_unknowns_and_arguments(self, mixture: NonReactiveMixture) -> None:
        """Set the list of symbolic unknowns per flash type."""

        X_pT: list[sm.Symbol] = list()
        X_ph: list[sm.Symbol] = list()
        X_hv: list[sm.Symbol] = list()

        # unknowns for p-T flash
        for phase, y in zip(mixture.phases, self.symbols.phase_fractions.values()):
            if phase != mixture.reference_phase:
                X_pT.append(y)

        X_pT += [x_ij for x_ij in self.symbols.phase_compositions.values()]

        # unknowns for p-h flash
        X_ph += [self.symbols('temperature')] + X_pT

        # unknowns for h-v flash
        X_hv += [self.symbols('pressure'), self.symbols('temperature')]
        for phase, s in zip(mixture.phases, self.symbols.phase_saturations.values()):
            if phase != mixture.reference_phase:
                X_hv.append(s)
        X_hv += X_pT

        self.unknowns.update(
            {
                'p-T': X_pT,
                'p-h': X_ph,
                'h-v': X_hv,
            }
        )

        args_z = [
            z
            for z, comp in zip(self.symbols.feed_fractions.values(), mixture.components)
            if comp != mixture.reference_component
        ]

        # arguments for the p-T system
        args_pT = [self.symbols('pressure'), self.symbols('temperature')] + X_pT

        # arguments for the p-h system
        args_ph = [self.symbols('pressure'), self.symbols('enthalpy')] + X_ph

        # arguments for the h-v system
        args_hv = [self.symbols('enthalpy'), self.symbols('volume')] + X_hv

        self.arguments.update(
            {
                'p-T': args_z + args_pT,
                'p-h': args_z + args_ph,
                'h-v': args_z + args_hv,
            }
        )

    def _define_equations_and_jacobians(self, mixture: NonReactiveMixture):
        """Constructs the equilibrium equatins for each flash specification."""

        # list of fractional unknowns in symbolic form
        X_ = list(self.symbols.phase_compositions.values())
        Y = list(self.symbols.phase_fractions.values())
        Z = list(self.symbols.feed_fractions.values())

        X: list[list[sm.Symbol]] = list()
        for j in range(self._n_p):
            X_j = X_[j * self._n_c : (j + 1) * self._n_c]
            X.append(X_j)

        # list of mass conservation per component, except reference component
        mass_conservation: list[sm.Expr] = list()

        for i, C in enumerate(zip(mixture.components, Z)):
            comp, z_i = C
            if comp != mixture.reference_component:
                conservation_i = z_i
                for j, y in enumerate(Y):
                    conservation_i -= y * X[j][i]

                mass_conservation.append(conservation_i)

        # list of complementary conditions
        complement_cond: list[sm.Expr] = list()

        for j, y in enumerate(Y):
            cc_j = y * (1 - safe_sum(X[j]))
            complement_cond.append(cc_j)


        ### symbolic computation of EoS specific terms.
        p = self.symbols('pressure')
        T = self.symbols('temperature')

        # critical covolume per component
        b_i_crit = [
            PengRobinsonEoS.b_crit(comp.p_crit, comp.T_crit)
            for comp in mixture.components
        ]

        # mixed covolume per phase
        b_j = [VanDerWaals.covolume(x_j, b_i_crit) for x_j in X]
        B_j = [PengRobinsonEoS.B(b_j_, p, T) for b_j_ in b_j]

        # cohesion per phase

        # TODO this is not safe. Special BIP implementations are not sympy compatible
        # and it needs an instantiation of a PR EOS
        bips, dT_bips = mixture.reference_phase.eos._compute_bips(T)

        a_i_crit = [
            PengRobinsonEoS.a_crit(comp.p_crit, comp.T_crit)
            for comp in mixture.components
        ]

        k_i = [
            PengRobinsonEoS.a_correction_weight(comp.omega)
            for comp in mixture.components
        ]

        a_i_correction = [
            1 + k * (1 - sm.sqrt(T / comp.T_crit)) for k, comp in zip(k_i, mixture.components)
        ]

        # cohesion term per component
        a_i = [a * corr**2 for a, corr in zip(a_i_crit, a_i_correction)]

        # mixed cohesion terms per phase
        a_j = [VanDerWaals.cohesion_s(x_j, a_i, bips) for x_j in X]
        A_j = [PengRobinsonEoS.A(a, p, T) for a in a_j]

        pass