'''
Module for initializing, assigning data, solve, and save linear elastic problem 
with fractures.
'''
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


class StaticModel():
    '''
    Class for solving an static elasticity problem flow problem:
     \nabla \sigma = 0,
    where nabla is the stress tensor.

    Parameters in Init:
    gb: (Grid) a grid object.
    data: (dictionary) Defaults to None. Only used if gb is a Grid. Should
          contain a Parameter class with the keyword 'Param'
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
    '''

    def __init__(self, gb, data=None, physics='mechanics', **kwargs):
        self.physics = physics
        self._gb = gb
        if not isinstance(self._gb, Grid):
            raise ValueError('StaticModel only defined for Grid class')

        self._data = data

        self.lhs = []
        self.rhs = []
        self.x = []

        file_name = kwargs.get('file_name', physics)
        folder_name = kwargs.get('folder_name', 'results')

        tic = time.time()
        logger.info('Create exporter')
        self.exporter = Exporter(self._gb, file_name, folder_name)
        logger.info('Elapsed time: ' + str(time.time() - tic))

        self._stress_disc = self.stress_disc()

        self.displacement_name = 'displacement'
        self.frac_displacement_name = 'frac_displacement'

    def solve(self, max_direct=40000, callback=False, **kwargs):
        """ Reassemble and solve linear system.

        After the funtion has been called, the attributes lhs and rhs are
        updated according to the parameter states. Also, the attribute x
        gives the pressure given the current state.

        TODO: Provide an option to save solver information if multiple
        systems are to be solved with the same left hand side.

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
        logger.info('Discretize')
        self.lhs, self.rhs = self.reassemble()
        logger.info('Done. Elapsed time ' + str(time.time() - tic))

        # Solve
        tic = time.time()
        ls = LSFactory()
        if self.rhs.size <  max_direct:
            logger.info('Solve linear system using direct solver')
            self.x = ls.direct(self.lhs,self.rhs)
        else:
            logger.info('Solve linear system using GMRES')
            precond = self._setup_preconditioner()
#            precond = ls.ilu(self.lhs)
            slv = ls.gmres(self.lhs)
            self.x, info = slv(self.rhs, M=precond, callback=callback,
                               maxiter=10000, restart=1500, tol=1e-8)
            if info == 0:
                logger.info('GMRES succeeded.')
            else:
                logger.error('GMRES failed with status ' + str(info))

        logger.info('Done. Elapsed time ' + str(time.time() - tic))
        return self.x

    def step(self):
        return self.solve()

    def reassemble(self):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system.
        """
        self.lhs, self.rhs = self._discretize(self._stress_disc)
        return self.lhs, self.rhs

    def stress_disc(self):
        return mpsa.FracturedMpsa(physics=self.physics)

    def _discretize(self, discr):
        return discr.matrix_rhs(self.grid(), self.data())

    def grid(self):
        return self._gb

    def data(self):
        return self._data

    def displacement(self, displacement_name='displacement'):
        self.displacement_name = displacement_name
        d = self._stress_disc.extract_u(self.grid(), self.x)
        self._data[self.displacement_name] = d.reshape((3, -1),order='F')

    def frac_displacement(self, frac_displacement_name='frac_displacement'):
        self.frac_displacement_name =frac_displacement_name
        self._data[self.frac_displacement_name] = \
            self._stress_disc.extract_frac_u(self.grid(), self.x)

    def traction(self, traction_name='traction'):
        T = self._stress_disc.traction(self.grid(),
                                       self._data,
                                       self.x)
        self._data[traction_name] = T.reshape((self.grid().dim, -1), order='F')

    def save(self, variables=None):
        if variables is None:
            self.exporter.write_vtk()
        else: 
            variables = {k: self._data[k] for k in variables \
                         if k in self._data}
            self.exporter.write_vtk(variables)

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
