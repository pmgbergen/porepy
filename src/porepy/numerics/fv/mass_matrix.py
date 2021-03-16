"""
Mass matrix classes for a discretization of a L2-mass bilinear form with constant test
and trial functions.

The discretization takes into account cell volumes and the mass_weight given in
the parameters (the mass weight can again incorporate porosity, time step,
apertures etc),  so that the mass matrix (shape g.num_cells^2) has the
following diagonal:

    g.cell_volumes * mass_weight

The right hand side is null.
There is also a class for the inverse of the mass matrix.

Note that the matrix equals the discretization operator in this case, and so is stored
directly in the data as
self._key() + "mass" or self._key() + "inv_mass".
The corresponding (null) rhs vectors are stored as
self._key() + "bound_mass" or self._key() + "bound_inv_mass", respectively.
"""
from typing import Dict, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

module_sections = ["numerics", "disrcetization"]


class MassMatrix(pp.numerics.discretization.Discretization):
    """Class that provides the discretization of a L2-mass bilinear form with constant
    test and trial functions.
    """

    def __init__(self, keyword: str = "flow") -> None:
        """Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword: str = keyword
        self.mass_matrix_key: str = "mass"
        self.bound_mass_matrix_key: str = "bound_mass"

    def _key(self) -> str:
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, g: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.
        In this case number of cells.

        Parameter:
            g: grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.

        """
        return g.num_cells

    @pp.time_logger(sections=module_sections)
    def assemble_matrix_rhs(
        self, g: pp.Grid, data: Dict
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side (null) for a discretization of a
        L2-mass bilinear form with constant test and trial functions. Also
        discretize the necessary operators if the data dictionary does not contain
        a mass matrix.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            matrix (sparse dia, self.ndof x self.ndof): Mass matrix obtained from the
                discretization.
            rhs (array, self.ndof): zero right-hand side.

        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g: pp.Grid, data: Dict) -> sps.spmatrix:
        """Return the matrix for a discretization of a L2-mass bilinear form with
        constant test and trial functions.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): System matrix of this
                discretization.

        """
        matrix_dictionary: Dict[str, Union[sps.spmatrix, np.ndarray]] = data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]

        M = matrix_dictionary[self.mass_matrix_key]
        return M

    def assemble_rhs(self, g: pp.Grid, data: Dict) -> np.ndarray:
        """Return the (null) right-hand side for a discretization of a L2-mass bilinear
        form with constant test and trial functions.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray (self.ndof): zero right hand side vector with representation of
                boundary conditions.

        """
        matrix_dictionary: Dict[str, Union[sps.spmatrix, np.ndarray]] = data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]

        rhs: np.ndarray = matrix_dictionary[self.bound_mass_matrix_key]
        return rhs

    def discretize(self, g: pp.Grid, data: Dict) -> None:
        """Discretize a L2-mass bilinear form with constant test and trial functions.

        Note that the porosity is not included in the volumes, and should be included
        in the mass weight if appropriate.

        We assume the following two sub-dictionaries to be present in the data
        dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]

        parameter_dictionary contains the entries:
            mass_weight: (array, self.g.num_cells): Scalar values which may e.g.
                represent the porosity, apertures (for lower-dimensional
                grids), or heat capacity. The discretization will multiply this
                weight with the cell volumes.
            num_components (int, optional): Number of components to be accumulated.
                Defaults to 1.

        matrix_dictionary will be updated with the following entries:
            mass: sps.dia_matrix (sparse dia, self.ndof x self.ndof): Mass matrix
                obtained from the discretization.
            bound_mass: all zero np.ndarray (self.ndof)

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        """
        parameter_dictionary: Dict = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary: Dict[str, Union[sps.spmatrix, np.ndarray]] = data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]
        ndof = self.ndof(g)

        w = parameter_dictionary["mass_weight"]
        volumes = g.cell_volumes
        coeff = volumes * w

        # Expand to cover number of components
        num_components: int = parameter_dictionary.get("num_components", 1)
        coeff_exp = np.tile(coeff, (num_components, 1)).ravel("F")
        tot_ndof = num_components * ndof

        matrix_dictionary[self.mass_matrix_key] = sps.dia_matrix(
            (coeff_exp, 0), shape=(tot_ndof, tot_ndof)
        )
        matrix_dictionary[self.bound_mass_matrix_key] = np.zeros(ndof)


class InvMassMatrix:
    """Class that provides the discretization of a L2-mass bilinear form with constant
    test and trial functions.
    """

    def __init__(self, keyword: str = "flow") -> None:
        """
        Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword: str = keyword
        self.mass_matrix_key: str = "inv_mass"
        self.bound_mass_matrix_key: str = "inv_bound_mass"

    def _key(self) -> str:
        """Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + "_"

    def ndof(self, g: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.
        In this case number of cells.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    def assemble_matrix_rhs(
        self, g: pp.Grid, data: Dict
    ) -> Tuple[sps.spmatrix, np.ndarray]:
        """Return the inverse of the matrix and right-hand side (null) for a
        discretization of a L2-mass bilinear form with constant test and trial
        functions. Also discretize the necessary operators if the data dictionary does
        not contain a discrete inverse mass matrix.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            matrix (sparse dia, self.ndof x self.ndof): Mass matrix obtained from the
                discretization.
            rhs (array, self.ndof):
                zero right-hand side.

        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g: pp.Grid, data: Dict) -> sps.spmatrix:
        """Return the inverse of the matrix for a discretization of a L2-mass bilinear
        form with constant test and trial functions. Also discretize the necessary
        operators if the data dictionary does not contain a discrete inverse mass
        matrix.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): System matrix of this
                discretization.
        """
        matrix_dictionary: Dict[str, Union[sps.spmatrix, np.ndarray]] = data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]

        M = matrix_dictionary[self.mass_matrix_key]
        return M

    def assemble_rhs(self, g: pp.Grid, data: Dict) -> np.ndarray:
        """Return the (null) right-hand side for a discretization of the inverse of a
        L2-mass bilinear form with constant test and trial functions. Also discretize
        the necessary operators if the data dictionary does not contain a discretization
        of the boundary term.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions: A null vector of length g.num_faces.
        """
        matrix_dictionary: Dict[str, Union[sps.spmatrix, np.ndarray]] = data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]

        rhs = matrix_dictionary[self.bound_mass_matrix_key]
        return rhs

    def discretize(self, g: pp.Grid, data: Dict) -> None:
        """Discretize the inverse of a L2-mass bilinear form with constant test and
        trial functions. Calls the MassMatrix().discretize() method and takes the
        inverse for the lhs.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Stores:
            matrix (sparse dia, self.ndof x self.ndof): Mass matrix obtained from the
                discretization, stored as           self._key() + "inv_mass".
            rhs (array, self.ndof):
                zero right-hand side, stored as     self._key() + "bound_inv_mass".

        The names of data in the input dictionary (data) are:
            mass_weight: (array, self.g.num_cells): Scalar values which may e.g.
                represent the porosity, apertures (for lower-dimensional
                grids), or heat capacity. The discretization will multiply this
                weight with the cell volumes.

        """
        matrix_dictionary: Dict[str, Union[sps.spmatrix, np.ndarray]] = data[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]

        mass = MassMatrix(keyword=self.keyword)
        mass.discretize(g, data)
        M, rhs = mass.assemble_matrix_rhs(g, data)

        matrix_dictionary[self.mass_matrix_key] = sps.dia_matrix(
            (1.0 / M.diagonal(), 0), shape=M.shape
        )

        matrix_dictionary[self.bound_mass_matrix_key] = rhs
