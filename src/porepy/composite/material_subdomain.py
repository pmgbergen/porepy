""" Contains the class representing a computational subdomain.
"""

from typing import Union, Iterable, List, Optional

import porepy as pp
import numpy as np

from .substance import SolidSubstance


class MaterialSubdomain:
    """ Class representing an physical extension of :class:`~porepy.Grid`.
    
    It combines the physical properties of substances with the space discretization of the geometry.

    Has also functionalities to instantiate and store primary variables defined in
    :data:`~porepy.params.computational_variables.COMPUTATIONAL_VARIABLES`.

    In future, this class can also serve as the single point of implementation for heuristic laws.

    """
    def __init__(self, grid: pp.Grid, substances: List[SolidSubstance]) -> None:
        """Constructor stores the parameters for future access.
        
        :param grid: the discretization for which variables and substance parameter arrays should be provided
        :type grid: :class:`porepy.Grid`
        :param substance: substance representative with access to physical values
        :type grid: List[:class:`~porepy.composite.component.SolidSkeletonSubstance`]
        """
        
        self.grid: pp.Grid = grid
        # NOTE currently we assume only a single substance. Mixed substance Domains remain a question for future development
        self.substance: SolidSubstance = substances[0]

    def base_porosity(self) -> pp.ad.Array:
        """
        :return: AD representation of the base porosity
        :rtype: :class:`~porepy.numerics.ad.operators.Array`
        """
        arr = np.ones(self.grid.num_cells) * self.substance.base_porosity()

        return pp.ad.Array(arr)
        