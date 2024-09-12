"""
This module contains functionality for testing the flow benchmarks implementations.
"""

from typing import Callable, Union

import numpy as np

import porepy as pp
from porepy.models.protocol import (
    PorePyModel,
    FluidMassBalanceProtocol,
    MixedDimensionalProtocol,
)


class EffectivePermeability(
    PorePyModel, FluidMassBalanceProtocol, MixedDimensionalProtocol
):
    """Mixin that contains the computation of effective permeabilities."""

    def effective_tangential_permeability(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Retrieves the effective tangential permeability, see Eq. 6a from [1].

        This method implicitly assumes that, in each subdomain, the effective
        tangential permeability is isotropic.

        The effective tangential permeability is the permeability tensor multiplied
        by the specific volume. PorePy "transforms" the intrinsic permeability into
        an effective one using the method `operator_to_SecondOrderTensor` defined in
        the mixin class `~porepy.models.constitutive_laws.SecondOrderTensorUtils`.

        Parameters:
            subdomains: List of subdomain grids for which the permeability is fetched.

        Returns:
            Wrapped ad operator containing the effective tangential permeabilities
            for the given list of subdomains.

        """
        values = []
        size = self.mdg.num_subdomain_cells()
        for sd in subdomains:
            d = self.mdg.subdomain_data(sd)
            val_loc = d[pp.PARAMETERS][self.darcy_keyword][
                "second_order_tensor"
            ].values[0][0]
            values.append(val_loc)
        return pp.wrap_as_dense_ad_array(
            np.hstack(values), size, "effective_tangential_permeability"
        )

    def effective_normal_permeability(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """
        Computes the effective normal permeability, see Eq. 6b from [1].

        The effective normal permeability is the scalar that multiplies the pressure
        jump in the continuous interface law.

        Parameters:
            interfaces: List of mortar grids where the normal permeability is computed.

        Returns:
            Wrapped ad operator containing the effective normal permeabilities for the
            given list of interfaces.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )

        effective_normal_permeability = (
            self.specific_volume(interfaces)
            * self.normal_permeability(interfaces)
            * normal_gradient
        )
        effective_normal_permeability.set_name("effective_normal_permeability")

        return effective_normal_permeability
