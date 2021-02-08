"""
Discretization of the source term of an equation tailored for a dual
(flux-pressure) system. The sources are assigned to the rows starting from
g.num_faces, that is, to those rows in the saddle point system that represents
conservation.

"""

import numpy as np
import scipy.sparse as sps

import porepy as pp

module_sections = ["numerics", "discretization", "assembly"]


class DualScalarSource(pp.numerics.discretization.Discretization):
    """
    Discretization of the integrated source term
    int q * dx
    over each grid cell.

    All this function does is returning a zero lhs and
    rhs = - param.get_source.keyword in a saddle point fashion.
    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, keyword="flow"):
        self.keyword = keyword

    @pp.time_logger(sections=module_sections)
    def ndof(self, g):
        return g.num_cells + g.num_faces

    @pp.time_logger(sections=module_sections)
    def assemble_matrix_rhs(self, g, data):
        """Return the (null) matrix and right-hand side for a discretization of the
        integrated source term. Also discretize the necessary operators if the data
        dictionary does not contain a source term.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs.
            sources (array, self.ndof): Right-hand side vector.

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.

        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    @pp.time_logger(sections=module_sections)
    def assemble_matrix(self, g, data):
        """Return the (null) matrix and for a discretization of the integrated source
        term. Also discretize the necessary operators if the data dictionary does not
        contain a source term.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): Null system matrix of this
                discretization.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        return matrix_dictionary["source"]

    @pp.time_logger(sections=module_sections)
    def assemble_rhs(self, g, data):
        """Return the rhs for a discretization of the integrated source term. Also
        discretize the necessary operators if the data dictionary does not contain a
        source term.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.array (self.ndof): Right hand side vector representing the
                source.

        """
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        sources = parameter_dictionary["source"]
        if not sources.size == g.num_cells:
            raise ValueError("There should be one source value for each cell")

        # The sources are assigned to the rows representing conservation.
        rhs = np.zeros(self.ndof(g))
        is_p = np.hstack(
            (np.zeros(g.num_faces, dtype=bool), np.ones(g.num_cells, dtype=bool))
        )
        # A minus sign is apparently needed here to be consistent with the user
        # side convention of the finite volume method
        rhs[is_p] = -sources
        return rhs

    @pp.time_logger(sections=module_sections)
    def discretize(self, g, data):
        """Discretize an integrated source term.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Stores:
            lhs (sparse dia, self.ndof x self.ndof): Null lhs, stored as
                self._key() + "source".

        The names of data in the input dictionary (data) are:
        param (Parameter Class) with the source field set for self.keyword. The assigned
            source values are assumed to be integrated over the cell volumes.

        """
        lhs = sps.csc_matrix((self.ndof(g), self.ndof(g)))
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        matrix_dictionary["source"] = lhs
