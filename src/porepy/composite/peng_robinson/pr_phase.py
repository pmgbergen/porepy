"""This module contains an extension of the abstract phase class for the
Peng-Robinson EoS."""
from __future__ import annotations

from typing import Callable

import numpy as np

import porepy as pp

from .._composite_utils import R_IDEAL
from ..phase import Phase
from .pr_component import PR_Component
from .pr_mixing import VdW_a_ij, dT_VdW_a_ij
from .pr_utils import Leaf, _exp, _log, _power


class PR_Phase(Phase):
    """Representation of a phase using the Peng-Robinson EoS and the Van der Waals
    mixing rule.

    This class is not intended to be used or instantiated except by the respective
    composition class.

    """

    def __init__(self, ad_system: pp.ad.EquationSystem, name: str = "") -> None:
        super().__init__(ad_system, name=name)

        self._h: Callable

        self.Z: Leaf = Leaf(f"Compressibility {self.name}")
        """An operator representing the compressibility factor of this phase.

        The respective composition class assigns values to this operator upon computing
        them.

        Due to performance/ algorithmic reasons,
        the computation of Z is modularized and done in a special way.

        """

    def cohesion(self, T: pp.ad.MixedDimensionalVariable) -> pp.ad.Operator:
        """An operator representing ``a`` in the Peng-Robinson EoS using the component
        molar fractions in this phase, and the Van der Waals mixing rule.

        Parameters:
            T: The temperature variable of the mixture.

        Returns:
            An operator representing ``a`` of this phase.

            If the phase is empty (no components), a wrapped zero is returned.

        """
        if self.num_components > 0:
            components: list[PR_Component] = [c for c in self]  # type: ignore

            # First we sum over the diagonal elements of the mixture matrix,
            # starting with the first component
            comp_0 = components[0]
            a = VdW_a_ij(T, comp_0, comp_0) * _power(
                self.normalized_fraction_of_component(comp_0), pp.ad.Scalar(2)
            )

            if len(components) > 1:
                # add remaining diagonal elements
                for comp in components[1:]:
                    a += VdW_a_ij(T, comp, comp) * _power(
                        self.normalized_fraction_of_component(comp), pp.ad.Scalar(2)
                    )

                # adding off-diagonal elements, including BIPs
                for comp_i in components:
                    for comp_j in components:
                        if comp_i != comp_j:
                            # computing the cohesion between components i and j
                            a += (
                                VdW_a_ij(T, comp_i, comp_j)
                                * self.normalized_fraction_of_component(comp_i)
                                * self.normalized_fraction_of_component(comp_j)
                            )

            # store cohesion parameters
            return a
        else:
            return pp.ad.Scalar(0.0)

    def dT_cohesion(self, T: pp.ad.MixedDimensionalVariable) -> pp.ad.Operator:
        """Returns an operator representing the temperature-derivative of
        :meth:`cohesion`."""
        if self.num_components > 0:
            components: list[PR_Component] = [c for c in self]  # type: ignore

            # First we sum over the diagonal elements of the mixture matrix,
            # starting with the first component
            comp_0 = components[0]
            dT_a = dT_VdW_a_ij(T, comp_0, comp_0) * _power(
                self.normalized_fraction_of_component(comp_0), pp.ad.Scalar(2)
            )

            if len(components) > 1:
                # add remaining diagonal elements
                for comp in components[1:]:
                    dT_a += dT_VdW_a_ij(T, comp, comp) * _power(
                        self.normalized_fraction_of_component(comp), pp.ad.Scalar(2)
                    )

                # adding off-diagonal elements, including BIPs
                for comp_i in components:
                    for comp_j in components:
                        if comp_i != comp_j:
                            # computing the cohesion between components i and j
                            dT_a += (
                                dT_VdW_a_ij(T, comp_i, comp_j)
                                * self.normalized_fraction_of_component(comp_i)
                                * self.normalized_fraction_of_component(comp_j)
                            )

            # store cohesion parameters
            return dT_a
        else:
            return pp.ad.Scalar(0.0)

    @property
    def covolume(self) -> pp.ad.Operator:
        """An operator representing ``b`` in the Peng-Robinson EoS using the component
        molar fractions in this phase, and the Van der Waals mixing rule.

        If the phase is empty (no components), a wrapped zero is returned."""
        if self.num_components > 0:
            components: list[PR_Component] = [c for c in self]  # type: ignore
            comp_0 = components[0]

            # updating covolume term
            b = self.normalized_fraction_of_component(comp_0) * comp_0.covolume
            if len(components) > 1:
                for comp in components[1:]:
                    b += self.normalized_fraction_of_component(comp) * comp.covolume

            return b
        else:
            return pp.ad.Scalar(0.0)

    def fugacity_of(
        self,
        p: pp.ad.MixedDimensionalVariable,
        T: pp.ad.MixedDimensionalVariable,
        component: PR_Component,
    ) -> pp.ad.Operator:
        """
        References:
            [1]: `Zhu et al. (2014), equ. A-4
                 <https://doi.org/10.1016/j.fluid.2014.07.003>`_
            [2]: `ePaper <https://www.yumpu.com/en/document/view/36008448/
                 1-derivation-of-the-fugacity-coefficient-for-the-peng-robinson-fet>`_
            [3]: `therm <https://thermo.readthedocs.io/
                 thermo.eos_mix.html#thermo.eos_mix.PRMIX.fugacity_coefficients>`_

        """
        if component in self:
            log_phi_c_e = self._log_phi_c_e(p, T, component)
            phi_c_e = _exp(log_phi_c_e)

            return phi_c_e
        else:
            return pp.ad.Scalar(0.0)

    def _log_phi_c_e(
        self,
        p: pp.ad.MixedDimensionalVariable,
        T: pp.ad.MixedDimensionalVariable,
        component: PR_Component,
    ) -> pp.ad.Operator:
        """Auxiliary function implementing the logarithmic fugacity coefficients for a
        ``component``."""
        # index c for component
        # index m for mixture
        b_c = component.covolume
        b_m = sum(
            [
                self.normalized_fraction_of_component(comp) * comp.covolume
                for comp in self
            ]
        )
        B_m = (b_m * p) / (R_IDEAL * T)

        a_c = sum(
            [
                self.normalized_fraction_of_component(other_c)
                * VdW_a_ij(T, component, other_c)
                for other_c in self
            ]
        )
        a_m = sum(
            [
                self.normalized_fraction_of_component(comp_i)
                * self.normalized_fraction_of_component(comp_j)
                * VdW_a_ij(T, comp_i, comp_j)
                for comp_i in self
                for comp_j in self
            ]
        )

        log_phi_c_e = (
            b_c / b_m * (self.Z - 1)
            - _log(self.Z - B_m)
            + _log(
                (self.Z + (1 + np.sqrt(2)) * B_m) / (self.Z + (1 - np.sqrt(2)) * B_m)
            )
            * a_m
            / (b_m * R_IDEAL * T * np.sqrt(8))
            * (b_c / b_m - 2 * a_c / a_m)
        )

        return log_phi_c_e

    def density(self, p, T):
        """ideal gas law modified by the compressibility factor."""
        return p / (R_IDEAL * T * self.Z)

    def specific_enthalpy(self, p, T):
        """
        References:
            [1]: `Connolly et al. (2021), eq. B-15 <https://doi.org/10.1016/
                 j.ces.2020.116150>`_
        """

        B = (self.covolume * p) / (R_IDEAL * T)

        h_departure = _power(self.covolume, pp.ad.Scalar(-1 / 2)) / (2 * np.sqrt(2)) * (
            T * self.dT_cohesion(T) - self.cohesion(T)
        ) * _log(
            (self.Z + (1 - np.sqrt(2)) * B) / (self.Z + (1 + np.sqrt(2)) * B)
        ) + R_IDEAL * T * (
            self.Z - 1
        )

        h_ideal = 0.0  # TODO enter formula

        return h_ideal + h_departure

    def dynamic_viscosity(self, p, T):  # TODO
        return pp.ad.Scalar(1.0)

    def thermal_conductivity(self, p, T):  # TODO
        return pp.ad.Scalar(1.0)
