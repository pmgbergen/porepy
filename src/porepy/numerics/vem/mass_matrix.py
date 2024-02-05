"""
Mass matrix classes for a discretization of a L2-mass bilinear form with constant test
and trial functions for mixed methods (e.g. RT0, MVEM).

The discretization takes into account cell volumes and the mass_weight given in
the parameters (the mass weight can again incorporate porosity, time step,
apertures etc),  so that the mass matrix (shape (sd.num_faces + sd.num_cells)^2)
has the following diagonal for the cell_dof:

    sd.cell_volumes * mass_weight

The right hand side is null.
There is also a class for the inverse of the mass matrix.

Note that the matrix equals the discretization operator in this case, and so is stored
directly in the data as
self._key() + "mixed_mass" or self._key() + "inv_mixed_mass".
The corresponding (null) rhs vectors are stored as
self._key() + "bound_mixed_mass" or self._key() + "bound_inv_mixed_mass", respectively.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp


class MixedMassMatrix:
    """Class that provides the discretization of a L2-mass bilinear form with constant
    test and trial functions for mixed methods (e.g. RT0, MVEM).
    """

    def __init__(self, keyword: str = "flow") -> None:
        """Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

    def _key(self) -> str:
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.
        In this case number of faces plus number of cells.

        Parameter:
            sd: grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.

        """
        return sd.num_faces + sd.num_cells

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side (null) for a discretization of a
        L2-mass bilinear form with constant test and trial functions. Also
        discretize the necessary operators if the data dictionary does not contain
        a mass matrix.

        Args:
            sd (pp.Grid) : grid, or a subclass, with geometry fields computed.
            data (dict): dictionary to store the data.

        Returns:
            matrix (sparse dia, self.ndof x self.ndof): Mass matrix obtained from the
                discretization.
            rhs (array, self.ndof): zero right-hand side.

        The names of data in the input dictionary (data) are given in the documentation of
        discretize, see there.
        """
        return self.assemble_matrix(sd, data), self.assemble_rhs(sd, data)

    def assemble_matrix(self, sd: pp.Grid, data: dict) -> sps.spmatrix:
        """Return the matrix for a discretization of a L2-mass bilinear form with
        constant test and trial functions.

        Args:
            sd (pp.Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): System matrix of this
                discretization.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        return matrix_dictionary["mixed_mass"]

    def assemble_rhs(self, sd: pp.Grid, data: dict) -> np.ndarray:
        """Return the (null) right-hand side for a discretization of a L2-mass bilinear
        form with constant test and trial functions.

        Args:
            sd (pp.Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray (self.ndof): zero right hand side vector with representation of
                boundary conditions.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        rhs = matrix_dictionary["bound_mixed_mass"]
        return rhs

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        """Discretize a L2-mass bilinear form with constant test and trial functions.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]

        parameter_dictionary contains the entries:
            mass_weight: (array, self.g.num_cells): Scalar values which may e.g.
                represent the porosity or heat capacity. The disrcetization
                will multiply this weight with the cell volumes.

        matrix_dictionary will be updated with the following entries:
            mixed_mass: sps.dia_matrix (sparse dia, self.ndof x self.ndof): Mass matrix
                obtained from the discretization.
            bound_mass: all zero np.ndarray (self.ndof)

        Args:
            sd (pp.Grid) : A grid with geometry fields computed.
            data (dict): dictionary to store the data.

        """
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        ndof = self.ndof(sd)
        w = parameter_dictionary["mass_weight"]
        volumes = sd.cell_volumes
        coeff = np.hstack((np.zeros(sd.num_faces), volumes * w))

        matrix_dictionary["mixed_mass"] = sps.dia_matrix((coeff, 0), shape=(ndof, ndof))
        matrix_dictionary["bound_mixed_mass"] = np.zeros(ndof)


class MixedInvMassMatrix:
    """Class that provides the discretization of an inverse L2-mass bilinear form with constant
    test and trial functions for mixed methods (e.g. RT0, MVEM).
    """

    def __init__(self, keyword: str = "flow") -> None:
        """
        Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

    def _key(self) -> str:
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, sd: pp.Grid):
        """Return the number of degrees of freedom associated to the method.

        Args:
            sd (pp.Grid): A grid.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.num_faces + sd.num_cells

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the inverse of the matrix and right-hand side (null) for a
        discretization of a L2-mass bilinear form with constant test and trial
        functions.

        Args:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            matrix (sparse dia, self.ndof x self.ndof): Mass matrix obtained from the
                discretization.
            rhs (array, self.ndof):
                zero right-hand side.

        """
        return self.assemble_matrix(sd, data), self.assemble_rhs(sd, data)

    def assemble_matrix(self, sd: pp.Grid, data: dict) -> sps.spmatrix:
        """Return the inverse of the matrix for a discretization of a L2-mass bilinear
        form with constant test and trial functions. Also discretize the necessary
        operators if the data dictionary does not contain a discrete inverse mass
        matrix.

        Args:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): System matrix of this
                discretization.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        return matrix_dictionary["inv_mixed_mass"]

    def assemble_rhs(self, sd: pp.Grid, data: dict) -> np.ndarray:
        """Return the (null) right-hand side for a discretization of the inverse of a
        L2-mass bilinear form with constant test and trial functions. Also discretize
        the necessary operators if the data dictionary does not contain a discretization
        of the boundary term.

        Args:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions: A null vector of length g.num_faces.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        return matrix_dictionary["bound_inv_mixed_mass"]

    def discretize(self, sd: pp.Grid, data: dict) -> None:
        """Discretize the inverse of a L2-mass bilinear form with constant test and
        trial functions.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]

        parameter_dictionary contains the entries:
            mass_weight: (array, self.g.num_cells): Scalar values which may e.g.
                represent the porosity or heat capacity. The discretization
                will multiply this weight with the cell volumes.

        matrix_dictionary will be updated with the following entries:
            mixed_mass: sps.dia_matrix (sparse dia, self.ndof x self.ndof): Mass matrix
                obtained from the discretization.
            bound_mass: all zero np.ndarray (self.ndof)

        Args:
            sd (pp.Grid) : grid, or a subclass, with geometry fields computed.
            data (dict): dictionary to store the data.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        mass = MixedMassMatrix(keyword=self.keyword)
        mass.discretize(sd, data)
        M, rhs = mass.assemble_matrix_rhs(sd, data)
        coeff = M.diagonal()
        coeff[sd.num_faces :] = 1.0 / coeff[sd.num_faces :]

        matrix_dictionary["inv_mixed_mass"] = sps.dia_matrix((coeff, 0), shape=M.shape)
        matrix_dictionary["bound_inv_mixed_mass"] = rhs
