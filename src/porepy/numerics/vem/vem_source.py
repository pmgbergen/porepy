"""
Discretization of the source term of an equation tailored for a dual
(flux-pressure) system. The sources are assigned to the rows starting from
sd.num_faces, that is, to those rows in the saddle point system that represents
conservation.

"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization


class DualScalarSource(Discretization):
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = - param.get_source.keyword in a saddle point fashion.
    """

    def __init__(self, keyword: str = "flow") -> None:
        self.keyword = keyword

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.
        In this case number of faces plus number of cells.

        Args:
            sd: grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.

        """
        return sd.num_faces + sd.num_cells

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the (null) matrix and right-hand side for a discretization of the
        integrated source term. Also discretize the necessary operators if the data
        dictionary does not contain a source term.

        Args:
            sd (pp.Grid): grid, or a subclass, with geometry fields computed.
            data (dict): dictiotary to store the data.

        Returns:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs.
            sources (array, self.ndof): Right-hand side vector.

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.

        """
        return self.assemble_matrix(sd, data), self.assemble_rhs(sd, data)

    def assemble_matrix(self, sd: pp.Grid, data: dict) -> sps.spmatrix:
        """Return the (null) matrix and for a discretization of the integrated source
        term. Also discretize the necessary operators if the data dictionary does not
        contain a source term.

        Args:
            sd (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): Null system matrix of this
                discretization.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        return matrix_dictionary["source"]

    def assemble_rhs(self, sd: pp.Grid, data: dict) -> np.ndarray:
        """Return the rhs for a discretization of the integrated source term. Also
        discretize the necessary operators if the data dictionary does not contain a
        source term.

        Args:
            sd (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.array (self.ndof): Right hand side vector representing the
                source.

        """
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        sources = parameter_dictionary["source"]
        if not sources.size == sd.num_cells:
            raise ValueError("There should be one source value for each cell")

        # The sources are assigned to the rows representing conservation.
        rhs = np.zeros(self.ndof(sd))
        is_p = np.hstack(
            (np.zeros(sd.num_faces, dtype=bool), np.ones(sd.num_cells, dtype=bool))
        )
        # A minus sign is apparently needed here to be consistent with the user
        # side convention of the finite volume method
        rhs[is_p] = -sources
        return rhs

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        """Discretize an integrated source term.

        Args:
            sd : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Stores:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs, stored as
                self._key() + "source".

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.

        """
        lhs = sps.csc_matrix((self.ndof(sd), self.ndof(sd)))
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        matrix_dictionary["source"] = lhs
