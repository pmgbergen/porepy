'''
Module for initializing, assigning data, solve, and save an elliptic pde.
Ths can for example be incompressible flow
problem assuming darcy's law. Please see the tutorial darcy's equation on the
porepy github: https://github.com/pmgbergen/porepy
'''
import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import tpfa, source, fvutils
from porepy.numerics.vem import vem_dual, vem_source
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
             Sets attributes: self.x
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
    split(name): Assignes the solution self.x to the data dictionary at each
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
        self.is_GridBucket = isinstance(self._gb, GridBucket)
        self._data = data

        self.lhs = []
        self.rhs = []
        self.x = []

        self.parameters = {'file_name': physics}
        self.parameters['folder_name'] = 'results'

        self._flux_disc = self.flux_disc()
        self._source_disc = self.source_disc()

    def solve(self):
        self.x = sps.linalg.spsolve(*self.reassemble())
        return self.x

    def step(self):
        return self.solve()

    def reassemble(self):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system.
        """
        lhs_flux, rhs_flux = self._discretize(self._flux_disc)
        lhs_source, rhs_source = self._discretize(self._source_disc)
        assert lhs_source.nnz == 0, 'Source lhs different from zero!'
        self.lhs = lhs_flux
        self.rhs = rhs_flux + rhs_source
        return self.lhs, self.rhs

    def source_disc(self):
        if self.is_GridBucket:
            return source.IntegralMixDim(physics=self.physics)
        else:
            return source.Integral(physics=self.physics)

    def flux_disc(self):
        if self.is_GridBucket:
            return tpfa.TpfaMixDim(physics=self.physics)
        else:
            return tpfa.Tpfa(physics=self.physics)

    def _discretize(self, discr):
        if self.is_GridBucket:
            return discr.matrix_rhs(self.grid())
        else:
            return discr.matrix_rhs(self.grid(), self.data())

    def grid(self):
        return self._gb

    def data(self):
        return self._data

    def split(self, x_name='solution'):
        self.x_name = x_name
        self._flux_disc.split(self.grid(), self.x_name, self.x)

    def pressure(self, pressure_name='pressure'):
        self.pressure_name = pressure_name
        if self.is_GridBucket:
            self.split(self.pressure_name)
        else:
            self._data[self.pressure_name] = self.x

    def discharge(self, discharge_name='discharge'):
        if self.is_GridBucket:
            fvutils.compute_discharges(self.grid(), self.physics,\
                                       self.pressure_name)
        else:
            fvutils.compute_discharges(self.grid(), self.physics,\
                                       self.pressure_name,
                                       self._data)

    def save(self, variables, save_every=None):
        folder_name = self.parameters['folder_name']
        file_name = self.parameters['file_name']
        if not self.is_GridBucket:
            variables = {k: self._data[k] for k in variables if k in self._data}
        export_vtk(self.grid(), file_name, variables, folder=folder_name)

#------------------------------------------------------------------------------#

class DualElliptic(Elliptic):

    def __init__(self, gb, data=None, physics='flow'):
        Elliptic.__init__(self, gb, data, physics)

        self.discharge_name = str()
        self.projected_discharge_name = str()

    def source_disc(self):
        if self.is_GridBucket:
            return vem_source.IntegralMixDim(physics=self.physics)
        else:
            return vem_source.Integral(physics=self.physics)

    def flux_disc(self):
        if self.is_GridBucket:
            return vem_dual.DualVEMMixDim(physics=self.physics)
        else:
            return vem_dual.DualVEM(physics=self.physics)

    def pressure(self, pressure_name='pressure'):
        self.pressure_name = pressure_name
        if self.is_GridBucket:
            self._flux_disc.extract_p(self._gb, self.x_name, self.pressure_name)
        else:
            pressure = self._flux_disc.extract_p(self._gb, self.x)
            self._data[self.pressure_name] = pressure

    def discharge(self, discharge_name="discharge"):
        self.discharge_name = discharge_name
        if self.is_GridBucket:
            self._flux_disc.extract_u(self._gb, self.x_name, self.discharge_name)
            [d['param'].set_discharge(d[self.discharge_name]) \
                                                        for _, d in self._gb]
        else:
            discharge = self._flux_disc.extract_u(self._gb, self.x)
            self._data['param'].set_discharge(discharge)
            self._data[self.discharge_name] = discharge

    def project_discharge(self, projected_discharge_name="P0u"):
        self.projected_discharge_name = projected_discharge_name
        if self.is_GridBucket:
            self._flux_disc.project_u(self._gb, self.discharge_name,
                                      self.projected_discharge_name)
        else:
            discharge = self._data[self.discharge_name]
            projected_discharge = self._flux_disc.project_u(self._gb,
                                                          discharge, self._data)
            self._data[self.projected_discharge_name] = projected_discharge

#------------------------------------------------------------------------------#

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
