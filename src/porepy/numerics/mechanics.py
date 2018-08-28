"""
Module for initializing, assigning data, solve, and save linear elastic problem
with fractures.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import time
import logging

from porepy.numerics.fv import mpsa, fvutils
from porepy.numerics.linalg.linsolve import Factory as LSFactory
from porepy.grids.grid import Grid
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.viz.exporter import Exporter

# Module-wide logger
logger = logging.getLogger(__name__)


class StaticModel:
    """
    Class for solving an static elasticity problem:
     \nabla \sigma = 0,
    where nabla is the stress tensor.

    Parameters in Init:
    gb: (Grid) a grid object.
    data (dictionary): dictionary of data. Should contain a Parameter class
                       with the keyword 'Param'
    physics: (string): defaults to 'mechanics'

    Functions:
    solve(): Calls reassemble and solves the linear system.
             Returns: the displacement d.
             Sets attributes: self.x
    step(): Same as solve, but without reassemble of the matrices
    reassemble(): Assembles the lhs matrix and rhs array.
            Returns: lhs, rhs.
            Sets attributes: self.lhs, self.rhs
    stress_disc(): Defines the discretization of the stress term.
            Returns stress discretization object (E.g., Mpsa)
    grid(): Returns: the Grid or GridBucket
    data(): Returns: Data dictionary
    traction(name='traction'): Calculate the traction over each
                face in the grid and assigne it to the data dictionary as
                keyword name.
    save(): calls split('d'). Then export the pressure to a vtk file to the
            folder kwargs['folder_name'] with file name
            kwargs['file_name'], default values are 'results' for the folder and
            physics for the file name.
    """

    def __init__(self, gb, data, physics="mechanics", **kwargs):
        self.physics = physics
        self._gb = gb
        if not isinstance(self._gb, Grid):
            raise ValueError("StaticModel only defined for Grid class")

        self._data = data

        self.lhs = []
        self.rhs = []
        self.x = []

        file_name = kwargs.get("file_name", physics)
        folder_name = kwargs.get("folder_name", "results")

        tic = time.time()
        logger.info("Create exporter")
        self.exporter = Exporter(self._gb, file_name, folder_name)
        logger.info("Elapsed time: " + str(time.time() - tic))

        self._stress_disc = self.stress_disc()

        self.displacement_name = "displacement"
        self.frac_displacement_name = "frac_displacement"
        self.is_factorized = False

    def solve(self, max_direct=40000, callback=False, discretize=True, **kwargs):
        """ Reassemble and solve linear system.

        After the funtion has been called, the attributes lhs and rhs are
        updated according to the parameter states. Also, the attribute x
        gives the pressure given the current state.

        The function attempts to set up the best linear solver based on the
        system size. The setup and parameter choices here are still
        experimental.

        Parameters:
            max_direct (int): Maximum number of unknowns where a direct solver
                is applied. If a direct solver can be applied this is usually
                the most efficient option. However, if the system size is
                too large compared to available memory, a direct solver becomes
                extremely slow.
            callback (boolean, optional): If True iteration information will be
                output when an iterative solver is applied (system size larger
                than max_direct)

        Returns:
            np.array: Pressure state.

        """
        # Discretize
        tic = time.time()
        if discretize:
            logger.info("Discretize")
            self.lhs, self.rhs = self.reassemble(**kwargs)
            self.is_factorized = False
            logger.info("Done. Elapsed time " + str(time.time() - tic))
        else:
            self.rhs = self._stress_disc.rhs(self.grid(), self.data())
        # Solve
        tic = time.time()
        ls = LSFactory()

        if self.rhs.size < max_direct:
            logger.info("Solve linear system using direct solver")
            if not self.is_factorized:
                logger.info("Making LU decomposition")
                self.lhs = self.lhs.tocsc()
                self.lhs = sps.linalg.factorized(self.lhs)
                self.is_factorized = True
                logger.info("Done. Elapsed time " + str(time.time() - tic))
            logger.info("Solve linear system using direct solver")
            tic = time.time()
            self.x = self.lhs(self.rhs)
        else:
            logger.info("Solve linear system using GMRES")
            precond = self._setup_preconditioner()
            #            precond = ls.ilu(self.lhs)
            slv = ls.gmres(self.lhs)
            self.x, info = slv(
                self.rhs,
                M=precond,
                callback=callback,
                maxiter=10000,
                restart=1500,
                tol=1e-8,
            )
            if info == 0:
                logger.info("GMRES succeeded.")
            else:
                logger.error("GMRES failed with status " + str(info))

        logger.info("Done. Elapsed time " + str(time.time() - tic))
        return self.x

    def step(self, **kwargs):
        """
        Calls self.solve(**kwargs)
        """
        return self.solve(**kwargs)

    def reassemble(self, discretize=True):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system.
        """
        self.lhs, self.rhs = self._stress_disc.matrix_rhs(
            self.grid(), self.data(), discretize
        )
        return self.lhs, self.rhs

    def stress_disc(self):
        """
        Define the stress discretization.
        Returns:
            FracturedMpsa (Solver object)
        """
        return mpsa.FracturedMpsa(physics=self.physics)

    def grid(self):
        """
        get the model grid
        Returns:
            gb (Grid object)
        """
        return self._gb

    def data(self):
        """
        get data
        Returns:
            data (Dictionary)
        """
        return self._data

    def displacement(self, displacement_name="displacement"):
        """
        Save the cell displacement to the data dictionary. The displacement
        will be saved as a (3,  self.grid().num_cells) array
        Parameters:
        -----------
        displacement_name:    (string) Defaults to 'displacement'. Defines the
                              keyword for the saved displacement in the data
                              dictionary
        Returns:
        --------
        d:  (ndarray) the displacement as a (3, self.grid().num_cells) array
        """

        self.displacement_name = displacement_name
        d = self._stress_disc.extract_u(self.grid(), self.x)
        d = d.reshape((3, -1), order="F")
        self._data[self.displacement_name] = d
        return d

    def frac_displacement(self, frac_displacement_name="frac_displacement"):
        """
        Save the fracture displacement to the data dictionary. This is the
        displacement on the fracture facers. The displacement
        will be saved as a (3,  self.grid().num_cells) array
        Parameters:
        -----------
        frac_displacement_name:
            (string) Defaults to 'frac_displacement'. Defines the keyword for
            the saved displacement in the data dictionary

        Returns:
        --------
        d:  (ndarray) the displacement of size (3, #number of fracture faces)
        """
        self.frac_displacement_name = frac_displacement_name
        d = self._stress_disc.extract_frac_u(self.grid(), self.x)
        d = d.reshape((3, -1), order="F")
        self._data[self.frac_displacement_name] = d
        return d

    def traction(self, traction_name="traction"):
        """
        Save the  traction on faces to the data dictionary. This is the
        area scaled traction on the fracture facers. The displacement
        will be saved as a (3,  self.grid().num_cells) array
        Parameters:
        -----------
        traction_name
            (string) Defaults to 'traction'. Defines the keyword for the
            saved traction in the data dictionary

        Returns:
        --------
        d:  (ndarray) the traction as a (3, self.grid().num_faces) array
        """
        T = self._stress_disc.traction(self.grid(), self._data, self.x)
        T = T.reshape((self.grid().dim, -1), order="F")
        T_b = np.zeros(T.shape)
        sigma = self._data["param"].get_background_stress(self.physics)
        if np.any(sigma):
            normals = self.grid().face_normals
            for i in range(normals.shape[1]):
                T_b[:, i] = np.dot(normals[:, i], sigma[i])
        else:
            T_b = 0
        self._data[traction_name] = T + T_b

    def save(self, variables=None, time_step=None):
        """
        Save the result as vtk.

        Parameters:
        ----------
        variables: (list) Optional, defaults to None. If None, only the grid
            will be exported. A list of strings where each element defines a
            keyword in the data dictionary to be saved.
        time_step: (float) optinal, defaults to None. The time step of the
            variable(s) that is saved
        """

        if variables is None:
            self.exporter.write_vtk()
        else:
            variables = {k: self._data[k] for k in variables if k in self._data}
            self.exporter.write_vtk(variables, time_step=time_step)

    ### Helper functions for linear solve below
    def _setup_preconditioner(self):
        solvers, ind, not_ind = self._assign_solvers()

        def precond(r):
            x = np.zeros_like(r)
            for s, i, ni in zip(solvers, ind, not_ind):
                x[i] += s(r[i])
            return x

        def precond_mult(r):
            x = np.zeros_like(r)
            A = self.lhs
            for s, i, ni in zip(solvers, ind, not_ind):
                r_i = r[i] - A[i, :][:, ni] * x[ni]
                x[i] += s(r_i)
            return x

        M = lambda r: precond(r)
        return spl.LinearOperator(self.lhs.shape, M)

    def _assign_solvers(self):
        mat, ind = self._obtain_submatrix()
        all_ind = np.arange(self.rhs.size)
        not_ind = [np.setdiff1d(all_ind, i) for i in ind]

        factory = LSFactory()
        num_mat = len(mat)
        solvers = np.empty(num_mat, dtype=np.object)
        for i, A in enumerate(mat):
            sz = A.shape[0]
            if sz < 5000:
                solvers[i] = factory.direct(A)
            else:
                # amg solver is pyamg is installed, if not ilu
                try:
                    solvers[i] = factory.amg(A, as_precond=True)
                except ImportError:
                    solvers[i] = factory.ilu(A)

        return solvers, ind, not_ind

    def _obtain_submatrix(self):
        return [self.lhs], [np.arange(self.grid().num_cells)]


# ------------------------------------------------------------------------------#
class StaticDataAssigner:
    """
    Class for setting data to a linear elastic static problem:
    \nabla \sigma = 0,
    This class creates a Parameter object and assigns the data to this object
    by calling StaticDataAssigner's functions.

    To change the default values, create a class that inherits from
    StaticDataAssigner. Then overload the values you whish to change.

    Parameters in Init:
    gb (Grid): a grid object
    data (dictionary): Dictionary which Parameter will be added to with keyword
                       'param'
    physics: (string): defaults to 'mechanics'

    Functions that assign data to Parameter class:
        bc(): defaults to neumann boundary condition
             Returns: (Object) boundary condition
        bc_val(): defaults to 0
             returns: (ndarray) boundary condition values
        stress_tensor(): defaults to 1
             returns: (tensor.FourthOrderTensor) Stress tensor

    Utility functions:
        grid(): returns: the grid

    """

    def __init__(self, g, data, physics="mechanics"):
        self._g = g
        self._data = data

        self.physics = physics
        self._set_data()

    def bc(self):
        return bc.BoundaryCondition(self.grid())

    def bc_val(self):
        return np.zeros(self.grid().dim * self.grid().num_faces)

    def background_stress(self):
        sigma = -np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        return sigma

    def stress_tensor(self):
        return None

    def data(self):
        return self._data

    def grid(self):
        return self._g

    def _set_data(self):
        if "param" not in self._data:
            self._data["param"] = Parameters(self.grid())
        self._data["param"].set_bc(self.physics, self.bc())
        self._data["param"].set_bc_val(self.physics, self.bc_val())
        self._data["param"].set_background_stress(
            self.physics, self.background_stress()
        )
        if self.stress_tensor() is not None:
            self._data["param"].set_tensor(self.physics, self.stress_tensor())
