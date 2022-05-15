""" Contains the class representing a computational subdomain.
"""
from __future__ import annotations

import numpy as np

import porepy as pp

from .substance import SolidSubstance

__all__ = ["MaterialSubdomain"]


class MaterialSubdomain:
    """
    Class representing a physical extension of :class:`~porepy.Grid`.

    It combines the physical properties of substances with
    the space discretization of the geometry.

    In future, this class can also serve as the single point of implementation
    for heuristic laws.

    NOTE: We assume currently only a single substance. Mixed substance Domains remain a
    question for future development.
    """

    def __init__(self, grid: pp.Grid, substances: SolidSubstance) -> None:
        """Constructor stores the parameters for future access.

        :param grid: the discretization for which variables and substance parameter arrays
            should be provided
        :type grid: :class:`porepy.Grid`
        :param substance: substance representative with access to physical values
        :type substance: :class:`~porepy.composite.SolidSubstance`
        """

        self.grid: pp.Grid = grid
        self.substance: SolidSubstance = substances

    def __str__(self) -> str:
        """String representation combining information about geometry and material."""
        out = "Material subdomain made of %s on grid:\n" % (self.substance.name)
        return out + str(self.grid)

    def base_porosity(self) -> "pp.ad.Operator":
        """
        :return: AD representation of the base porosity
        :rtype: :class:`~porepy.numerics.ad.operators.Array`
        """
        arr = np.ones(self.grid.num_cells) * self.substance.base_porosity()

        return pp.ad.Array(arr)

    def base_permeability(self) -> "pp.ad.Operator":
        """
        :return: AD representation of the base permeability
        :rtype: :class:`~porepy.numerics.ad.operators.Array`
        """
        arr = np.ones(self.grid.num_cells) * self.substance.base_permeability()

        return pp.ad.Array(arr)

    # ------------------------------------------------------------------------------
    ### HEURISTIC LAWS NOTE all heuristic laws can be modularized somewhere and referenced here
    # ------------------------------------------------------------------------------

    def porosity(
        self,
        law: str,
        pressure: "pp.ad.MergedVariable",
        enthalpy: "pp.ad.MergedVariable",
        **kwargs,
    ) -> "pp.ad.Operator":
        """
        Currently supported heuristic laws (values for 'law'):
            - 'pressure':      linear model using base porosity and reference pressure

        Inherit this class and overwrite this method if you want to implement special models
        for the phase density.
        Use keyword arguments 'kwargs' to provide arguments for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [-] (fractional)

        :return: Ad object representing the porosity
        :rtype: :class:`porepy.ad.Operator`
        """

        law = str(law)
        if law == "pressure":
            p_ref = kwargs["reference_pressure"]
            return pp.ad.Function(
                lambda p: self.substance.base_porosity() * (p - p_ref),
                "porosity-%s-%s" % (law, self.substance.name),
            )(pressure)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for subdomain porosity.: %s \n" % (law)
                + "Available: 'pressure,'"
            )

    def relative_permeability(
        self, law: str, saturation: "pp.ad.MergedVariable", **kwargs
    ) -> "pp.ad.Operator":
        """
        Currently supported heuristic laws (values for 'law'):
            - 'quadratic':      quadratic power law for saturation

        Inherit this class and overwrite this method if you want to implement special models
        for the relative permeability.
        Use keyword arguments 'kwargs' to provide arguments for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [-] (fractional)

        :param law: name of the law to be applied (see valid values above)
        :type law: str

        :return: relative permeability using the respectie law
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        law = str(law)
        if law == "quadratic":
            return pp.ad.Function(
                lambda S: S**2, "rel-perm-%s-%s" % (law, self.substance.name)
            )(saturation)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for rel.Perm.: %s \n" % (law)
                + "Available: 'quadratic,'"
            )
