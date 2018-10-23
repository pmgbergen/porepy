"""
Mass matrix classes for a discretization of a L2-mass bilinear form with constant test
and trial functions.

The discretization takes into account cell volumes, porosity, time step and aperture,
so that the mass matrix (shape g.num_cells^2) has the following diagonal:
g.cell_volumes * porosity * aperture / deltaT
The right hand side is null.
There is also a class for the inverse of the mass matrix.

Note that the matrix equals the discretization operator in this case, and so is stored
directly in the data as
self._key() + "mass" or self._key() + "inv_mass".
The corresponding (null) rhs vectors are stored as
self._key() + "bound_mass" or self._key() + "bound_inv_mass", respectively.
"""
import numpy as np
import scipy.sparse as sps

from porepy.numerics.mixed_dim.solver import Solver, SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler

# ------------------------------------------------------------------------------#


class MassMatrixMixedDim(SolverMixedDim):
    def __init__(self, physics="flow"):
        self.physics = physics

        self.discr = MassMatrix(self.physics)

        self.solver = Coupler(self.discr)


# ------------------------------------------------------------------------------#


class MassMatrix(Solver):
    """ Class that provides the discretization of a L2-mass bilinear form with constant
    test and trial functions.
    """
    # ------------------------------------------------------------------------------#

    def __init__(self, keyword="flow"):
        """ Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

        # The physics keyword is kept for consistency for now, but will soon be purged.
        self.physics = keyword

    # ------------------------------------------------------------------------------#

    def _key(self):
        """ Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + '_'

    # ------------------------------------------------------------------------------#

    def ndof(self, g):
        """ Return the number of degrees of freedom associated to the method.
        In this case number of cells.

        Parameter:
            g: grid, or a subclass.

        Returns:
            int: the number of degrees of freedom.

        """
        return g.num_cells

    # ------------------------------------------------------------------------------#

    def assemble_matrix_rhs(self, g, data):
        """ Return the matrix and right-hand side (null) for a discretization of a
        L2-mass bilinear form with constant test and trial functions. Also
        discretize the necessary operators if the data dictionary does not contain
        a mass matrix.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Returns:
            matrix (sparse dia, self.ndof x self.ndof): Mass matrix obtained from the
                discretization.
            rhs (array, self.ndof): Null right-hand side.

        The names of data in the input dictionary (data) are:
        param (Parameter Class): Contains the following parameters:
            porosity: (array, self.g.num_cells): Scalar values which represent the
                porosity. If not given assumed unitary.
            apertures (ndarray, g.num_cells): Apertures of the cells for scaling of
                the face normals. If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme. If not given
            assumed unitary.
        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    # ------------------------------------------------------------------------------#

    def assemble_matrix(self, g, data):
        """ Return the matrix for a discretization of a L2-mass bilinear form with
        constant test and trial functions. Also discretize the necessary operators
        if the data dictionary does not contain a mass matrix.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix (self.ndof x self.ndof): System matrix of this
                discretization.
        """
        if not self._key() + "mass" in data.keys():
            self.discretize(g, data)

        M = data[self._key() + "mass"]
        return M

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """ Return the (null) right-hand side for a discretization of a L2-mass bilinear
        form with constant test and trial functions. Also discretize the necessary
        operators if the data dictionary does not contain a discretization of the
        boundary equation.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray (self.ndof): Null right hand side vector with representation of
                boundary conditions.
        """
        if not self._key() + 'bound_mass' in data.keys():
            self.discretize(g, data)

        rhs = data[self._key() + "bound_mass"]
        return rhs

    # ------------------------------------------------------------------------------#

    def discretize(self, g, data, faces=None):
        """ Discretize a L2-mass bilinear form with constant test and trial functions.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Stores:
            matrix (sparse dia, self.ndof x self.ndof): Mass matrix obtained from the
                discretization, stored as           self._key() + "mass".
            rhs (array, self.ndof):
                Null right-hand side, stored as     self._key() + "bound_mass".

        The names of data in the input dictionary (data) are:
        param (Parameter Class): Contains the following parameters:
            porosity: (array, self.g.num_cells): Scalar values which represent the
                porosity. If not given assumed unitary.
            apertures (ndarray, g.num_cells): Apertures of the cells for scaling of
                the face normals. If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme. If not given
            assumed unitary.
        """
        ndof = self.ndof(g)
        phi = data["param"].get_porosity()
        aperture = data["param"].get_aperture()
        coeff = g.cell_volumes * phi / data["deltaT"] * aperture
        data[self._key() + "mass"] = sps.dia_matrix((coeff, 0), shape=(ndof, ndof))
        data[self._key() + "bound_mass"] = np.zeros(ndof)


##########################################################################


class InvMassMatrixMixedDim(SolverMixedDim):

    def __init__(self, physics="flow"):
        self.physics = physics

        self.discr = InvMassMatrix(self.physics)

        self.solver = Coupler(self.discr)


# ------------------------------------------------------------------------------#


class InvMassMatrix(Solver):
    """ Class that provides the discretization of a L2-mass bilinear form with constant
    test and trial functions.
    """

    def __init__(self, keyword="flow"):
        """
        Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

        # The physics keyword is kept for consistency for now, but will soon be purged.
        self.physics = keyword

    # ------------------------------------------------------------------------------#

    def _key(self):
        """ Get the keyword of this object, on a format friendly to access relevant
        fields in the data dictionary

        Returns:
            String, on the form self.keyword + '_'.

        """
        return self.keyword + '_'

    # ------------------------------------------------------------------------------#

    def ndof(self, g):
        """ Return the number of degrees of freedom associated to the method.
        In this case number of cells.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    # ------------------------------------------------------------------------------#

    def assemble_matrix_rhs(self, g, data):
        """ Return the inverse of the matrix and right-hand side (null) for a
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
                Null right-hand side.

        The names of data in the input dictionary (data) are:
        param (Parameter Class): Contains the following parameters:
            porosity: (array, self.g.num_cells): Scalar values which represent the
                porosity. If not given assumed unitary.
            apertures (ndarray, g.num_cells): Apertures of the cells for scaling of
                the face normals. If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme. If not given
            assumed unitary.
        """
        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    # ------------------------------------------------------------------------------#

    def assemble_matrix(self, g, data):
        """ Return the inverse of the matrix for a discretization of a L2-mass bilinear
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
        if not self._key() + "inv_mass" in data.keys():
            self.discretize(g, data)

        M = data[self._key() + "inv_mass"]
        return M

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """ Return the (null) right-hand side for a discretization of the inverse of a
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
        if not self._key() + "bound_inv_mass" in data.keys():
            self.discretize(g, data)

        rhs = data[self._key() + "bound_inv_mass"]
        return rhs

    # ------------------------------------------------------------------------------#

    def discretize(self, g, data, faces=None):
        """ Discretize the inverse of a L2-mass bilinear form with constant test and
        trial functions. Calls the MassMatrix().discretize() method and takes the
        inverse for the lhs.

        Parameters:
            g : grid, or a subclass, with geometry fields computed.
            data: dictionary to store the data.

        Stores:
            matrix (sparse dia, self.ndof x self.ndof): Mass matrix obtained from the
                discretization, stored as           self._key() + "inv_mass".
            rhs (array, self.ndof):
                Null right-hand side, stored as     self._key() + "bound_inv_mass".

        The names of data in the input dictionary (data) are:
        param (Parameter Class): Contains the following parameters:
            porosity: (array, self.g.num_cells): Scalar values which represent the
                porosity. If not given assumed unitary.
            apertures (ndarray, g.num_cells): Apertures of the cells for scaling of
                the face normals. If not given assumed unitary.
        deltaT: Time step for a possible temporal discretization scheme. If not given
            assumed unitary.
        """
        M, rhs = MassMatrix(keyword=self.keyword).assemble_matrix_rhs(g, data)
        data[self._key() + "inv_mass"] = sps.dia_matrix((1. / M.diagonal(), 0),
                                                       shape=M.shape)
        data[self._key() + "bound_inv_mass"] = rhs


# ------------------------------------------------------------------------------#
