import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import tpfa, source, fvutils
from porepy.grids.grid_bucket import GridBucket
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.viz.exporter import export_vtk


class Darcy():
    def __init__(self, gb, data=None, physics='flow'):
        self.physics = physics
        self._gb = gb
        self._data = data
        self.lhs, self.rhs = self.reassemble()
        self.p = np.zeros(self.flux_disc().ndof(self.grid()))
        self.parameters = {'file_name': physics}
        self.parameters['folder_name'] = 'results'

    def solve(self):
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
            diffusive_discr = tpfa.TpfaMultiDim(physics=self.physics)
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


class DarcyData():
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
        return np.ones(self.grid().num_cells)

    def aperture(self):
        return np.ones(self.grid().num_cells)

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
        self._data['param'].set_porosity(self.porosity())
        self._data['param'].set_bc(self.physics, self.bc())
        self._data['param'].set_bc_val(self.physics, self.bc_val())
        self._data['param'].set_source(self.physics, self.source())
        self._data['param'].set_aperture(self.aperture())
