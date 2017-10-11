import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import tpfa, source
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

    def save(self):
        self.flux_disc().split(self.grid(), 'p', self.p)
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


if __name__ == '__main__':
    from porepy.grids import structured
    from porepy.fracs import meshing
    from porepy.viz.plot_grid import plot_grid
    from porepy.grids.grid import FaceTag
    g = structured.CartGrid([11, 11])
    g.compute_geometry()

    param = Parameters(g)
    dir_bound = np.ravel(np.argwhere(
        (g.has_face_tag(FaceTag.DOMAIN_BOUNDARY))))
    bc_cond = bc.BoundaryCondition(g, dir_bound, ['dir'] * dir_bound.size)
    bc_val = g.face_centers[0]

    src = np.zeros(g.num_cells)
    src[60] = 0

    param.set_bc('flow', bc_cond)
    param.set_bc_val('flow', bc_val)
    param.set_source('flow', src)
    d = {'param': param}

    problem = Darcy(g, d)
    p = problem.solve()
    import pdb
    pdb.set_trace()

    plot_grid(g, p)

    def assign_darcy_data(gb):
        gb.add_node_props(['problem'])
        for g, d in gb:
            d['problem'] = Matrix(g, d)

    class Matrix(DarcyData):
        def __init__(self, g, d):
            DarcyData.__init__(self, g, d)

        def bc_val(self):
            east = bc.face_on_side(self.grid(), 'East')
            west = bc.face_on_side(self.grid(), 'West')
            val = np.zeros(self.grid().num_faces)
            val[east] = 1
            val[west] = -1
            return val

    g = structured.CartGrid([10, 10])
    g.compute_geometry()
    d = {'param': Parameters(g)}
    gb = meshing.cart_grid([], [10, 10])
    gb.assign_node_ordering()

    assign_darcy_data(gb)
    problem = Darcy(gb)
    problem_mono = Darcy(g, d)
    p_mono = problem_mono.solve()
    plot_grid(g, p_mono)
    p = problem.solve()
    problem.flux_disc().split(problem.grid(), 'pressure', p)
    plot_grid(problem.grid(), 'pressure')
