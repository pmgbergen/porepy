""" Contains the class representing a computational subdomain.
"""
from __future__ import annotations

import numpy as np

import porepy as pp
from porepy.composite.substance import SolidSubstance

from ._composite_utils import COMPUTATIONAL_VARIABLES

__all__ = ["MaterialSubdomain"]


class MaterialSubdomain:
    """
    Class representing a physical extension of :class:`~porepy.Grid`.

    It combines the physical properties of substances with
    the space discretization of the geometry.

    It has also functionalities to instantiate primary variables defined in
    :data:`~porepy.params.computational_variables.COMPUTATIONAL_VARIABLES`.

    In future, this class can also serve as the single point of implementation
    for heuristic laws.

    NOTE: It is unclear so far, how to proceed with the combination of Grids and substances
    (instead of GridBuckets)
    and how to combine the functionalities with the ComputationalDomain.
    Currently, one needs the whole ComputationalDomain instance to call the SolidSubstance.
    (In general, traces of a substance can be expected anywhere in a domain)
    This SolidSubstance instance is then passed to the MaterialSubdomain in the instantiation.
    This is a weird knot in the reference logic of these different types.

    NOTE: We assume currently only a single substance. Mixed substance Domains remain a
    question for future development.

    """

    def __init__(self, grid: pp.Grid, substances: SolidSubstance) -> None:
        """Constructor stores the parameters for future access.

        :param grid: the discretization for which variables and substance parameter arrays
        should be provided
        :type grid: :class:`porepy.Grid`
        :param substance: substance representative with access to physical values
        :type grid: :class:`~porepy.composite.component.SolidSkeletonSubstance`

        """

        self.grid: pp.Grid = grid
        self.substance: SolidSubstance = substances

    def __str__(self) -> str:
        """String representation combining information about geometry and material."""
        out = "Material subdomain made of " + self.substance.name + " on grid:\n"
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

    def porosity(self, law: str, *args, **kwargs) -> "pp.ad.Operator":
        """
        Currently supported heuristic laws (values for 'law'):
            - 'pressure':      expects one positional argument in 'args', namely the reference
                               pressure
            - 'solvent':       uses the unmodified solvent density

        Inherit this class and overwrite this method if you want to implement special models
        for the phase density.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments
        for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        dimensionsless, fractional

        :return: Ad object representing the porosity
        :rtype: :class:`porepy.ad.Operator`
        """

        law = str(law)
        if law == "pressure":
            p_ref = args[0]
            pressure = self.cd(COMPUTATIONAL_VARIABLES["pressure"])
            return pp.ad.Function(
                lambda p: self.substance.base_porosity() * (p - p_ref),
                "porosity-%s-%s" % (law, self.substance.name),
            )(pressure)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for subdomain porosity.: %s \n" % (law)
                + "Available: 'pressure,'"
            )

    def relative_permeability(self, law: str, *args, **kwargs) -> "pp.ad.Operator":
        """
        Currently supported heuristic laws (values for 'law'):
            - 'brooks_corey':   Brook-Corey model TODO finish
            - 'quadratic':      quadratic power law for saturation

        Inherit this class and overwrite this method if you want to implement special models
        for the relative permeability.
        Use positional arguments 'args' and keyword arguments 'kwargs' to provide arguments
        for the heuristic law.

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
                lambda S: S**2, "rel-perm-%s-%s" % (law, self.name)
            )(self.saturation)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for rel.Perm.: %s \n" % (law)
                + "Available: 'quadratic,'"
            )
