'''
Module for initializing, assigning data, solve, and save an elliptic pde.
Ths can for example be incompressible flow
problem assuming darcy's law. Please see the tutorial darcy's equation on the
porepy github: https://github.com/pmgbergen/porepy
'''
import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import tpfa, source, fvutils
from porepy.grids.grid_bucket import GridBucket
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.viz.exporter import export_vtk


class Elliptic():
    '''
    Class for solving an incompressible flow problem:
    \nabla K \nabla p = q,
    where K is the second order permeability tenser, p the fluid pressure
    and q sinks and sources.

    Parameters in Init:
    gb: (Grid /GridBucket) a grid or grid bucket object. If gb = GridBucket
        a Parameter class should be added to each grid bucket data node with
        keyword 'param'.
    data: (dictionary) Defaults to None. Only used if gb is a Grid. Should
          contain a Parameter class with the keyword 'Param'
    physics: (string): defaults to 'flow'

    Functions:
    solve(): Calls reassemble and solves the linear system.
             Returns: the pressure p.
             Sets attributes: self.p
    step(): Same as solve, but without reassemble of the matrices
    reassemble(): Assembles the lhs matrix and rhs array.
            Returns: lhs, rhs.
            Sets attributes: self.lhs, self.rhs
    source_disc(): Defines the discretization of the source term.
            Returns Source discretization object
    flux_disc(): Defines the discretization of the flux term.
            Returns Flux discretization object (E.g., Tpfa)
    grid(): Returns: the Grid or GridBucket
    data(): Returns: Data dictionary
    split(name): Assignes the pressure self.p to the data dictionary at each
                 node in the GridBucket.
                 Parameters:
                    name: (string) The keyword assigned to the pressure
    discharge(): Calls split('p'). Then calculate the discharges over each
                 face in the grids and between edges in the GridBucket
    save(): calls split('p'). Then export the pressure to a vtk file to the
            folder self.parameters['folder_name'] with file name
            self.parameters['file_name']
    '''

    def __init__(self, gb, data=None, physics='flow'):
        self.physics = physics
        self._gb = gb
        self._data = data
        self.lhs = []
        self.rhs = []
        self.p = np.zeros(self.flux_disc().ndof(self.grid()))
        self.parameters = {'file_name': physics}
        self.parameters['folder_name'] = 'results'

    def solve(self):
        self.lhs, self.rhs = self.reassemble()
        self.p = sps.linalg.spsolve(self.lhs, self.rhs)
        return self.p

    def step(self):
        return self.solve()

    def reassemble(self):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system.
        """
        lhs_flux, rhs_flux = self._discretize(self.flux_disc())
        lhs_source, rhs_source = self._discretize(self.source_disc())
        assert lhs_source.nnz == 0, 'Source lhs different from zero!'
        self.lhs = lhs_flux
        self.rhs = rhs_flux + rhs_source
        return self.lhs, self.rhs

    def source_disc(self):
        if isinstance(self.grid(), GridBucket):
            source_discr = source.IntegralMultiDim(physics=self.physics)
        else:
            source_discr = source.Integral(physics=self.physics)
        return source_discr

    def flux_disc(self):
        if isinstance(self.grid(), GridBucket):
            diffusive_discr = tpfa.TpfaMixDim(physics=self.physics)
        else:
            diffusive_discr = tpfa.Tpfa(physics=self.physics)
        return diffusive_discr

    def _discretize(self, discr):
        if isinstance(self.grid(), GridBucket):
            lhs, rhs = discr.matrix_rhs(self.grid())
        else:
            lhs, rhs = discr.matrix_rhs(self.grid(), self.data())
        return lhs, rhs

    def grid(self):
        return self._gb

    def data(self):
        return self._data

    def split(self, name):
        self.flux_disc().split(self.grid(), name, self.p)

    def discharge(self):
        self.split('p')
        fvutils.compute_discharges(self.grid())

    def save(self, save_every=None):
        self.split('p')
        folder = self.parameters['folder_name']
        f_name = self.parameters['file_name']
        export_vtk(self.grid(), f_name, ['p'], folder=folder)


class EllipticData():
    '''
    Class for setting data to an incompressible flow problem:
    \nabla K \nabla p = q,
    where K is the second order permeability tenser, p the fluid pressure
    and q sinks and sources. This class creates a Parameter object and
    assigns the data to this object by calling EllipticData's functions.

    To change the default values create a class that inherits from EllipticData.
    Then overload the values you whish to change.

    Parameters in Init:
    gb: (Grid /GridBucket) a grid or grid bucket object
    data: (dictionary) Dictionary which Parameter will be added to with keyword
          'param'
    physics: (string): defaults to 'flow'

    Functions that assign data to Parameter class:
        bc(): defaults to neumann boundary condition
             Returns: (Object) boundary condition
        bc_val(): defaults to 0
             returns: (ndarray) boundary condition values
        porosity(): defaults to 1
             returns: (ndarray) porosity of each cell
        apperture(): defaults to 1
             returns: (ndarray) aperture of each cell
        permeability(): defaults to 1
             returns: (tensor.SecondOrder) Permeabillity tensor
        source(): defaults to 0
             returns: (ndarray) The source and sinks

    Utility functions:
        grid(): returns: the grid

    '''

    def __init__(self, g, data, physics='flow'):
        self._g = g
        self._data = data

        self.physics = physics
        self._set_data()

    def bc(self):
        return bc.BoundaryCondition(self.grid())

    def bc_val(self):
        return np.zeros(self.grid().num_faces)

    def porosity(self):
        '''Returns apperture of each cell. If None is returned, default
        Parameter class value is used'''
        return None

    def aperture(self):
        '''Returns apperture of each cell. If None is returned, default
        Parameter class value is used'''
        return None

    def permeability(self):
        kxx = np.ones(self.grid().num_cells)
        return tensor.SecondOrder(self.grid().dim, kxx)

    def source(self):
        return np.zeros(self.grid().num_cells)

    def data(self):
        return self._data

    def grid(self):
        return self._g

    def _set_data(self):
        if 'param' not in self._data:
            self._data['param'] = Parameters(self.grid())
        self._data['param'].set_tensor(self.physics, self.permeability())
        self._data['param'].set_bc(self.physics, self.bc())
        self._data['param'].set_bc_val(self.physics, self.bc_val())
        self._data['param'].set_source(self.physics, self.source())

        if self.porosity() is not None:
            self._data['param'].set_porosity(self.porosity())
        if self.aperture() is not None:
            self._data['param'].set_aperture(self.aperture())
