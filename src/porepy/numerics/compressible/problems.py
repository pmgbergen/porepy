import numpy as np
import scipy.sparse as sps

from porepy.grids import structured
from porepy.numerics.fv import tpfa, mass_matrix
from porepy.numerics.compressible import solvers
from porepy.params.data import Parameters
from porepy.params import tensor
from porepy.params import bc
from porepy.viz.exporter import export_vtk, export_pvd


class SlightlyCompressible():
    """
    Base-class for slightly compressible flow. Initialize all needed
    attributes for a slightly compressible solver.
    """

    def __init__(self):
        self._data = dict()
        self._set_data()
        self.solver = solvers.Implicit(self)
        self.solver.parameters['store_results'] = True
        self.parameters = {'file_name': 'pressure'}
        self.parameters['folder_name'] = 'results'

    #---------Discretization---------------

    def flux_disc(self):
        """
        Returns the flux discretization.
        """
        return tpfa.Tpfa()

    def time_disc(self):
        """
        Returns the flux discretization.
        """
        class TimeDisc(mass_matrix.MassMatrix):
            def matrix_rhs(self, g, data):
                lhs, rhs = mass_matrix.MassMatrix.matrix_rhs(self, g, data)
                return lhs * data['compressibility'], rhs * data['compressibility']
        return TimeDisc()

    def solve(self):
        """
        Call the solver
        """
        self.data = self.solver.solve()
        return self.data

    def update(self, t):
        source = self.source(t)
        bc_val = self.bc_val(t)
        self.data()['param'].set_source(self.flux_disc(), source)
        self.data()['param'].set_bc_val(self.flux_disc(), bc_val)

    #-----Parameters------------
    def porosity(self):
        return np.ones(self.grid().num_cells)

    def compressibility(self):
        return 1

    def permeability(self):
        kxx = np.ones(self.grid().num_cells)
        return tensor.SecondOrder(self.grid().dim, kxx)

    #----Boundary Condition-----------

    def bc(self):
        dir_bound = np.array([])
        return bc.BoundaryCondition(self.grid(), dir_bound,
                                    ['dir'] * dir_bound.size)

    def bc_val(self, t):
        return np.zeros(self.grid().num_faces)

    #-----Sink and sources-----------

    def source(self, t):
        f = np.zeros(self.grid().num_cells)
        return f

    #-----Data-----------------

    def data(self):
        return self._data

    def _set_data(self):
        self._data['param'] = Parameters(self.grid())
        self._data['deltaT'] = self.time_step()
        self._data['end_time'] = self.end_time()
        self._data['param'].set_tensor(self.flux_disc(), self.permeability())
        self._data['param'].set_porosity(self.porosity())
        self._data['compressibility'] = self.compressibility()
        self._data['param'].set_bc(self.flux_disc(), self.bc())
        self._data['param'].set_bc_val(self.flux_disc(), self.bc_val(0.0))
    #--------Initial conditions---------------

    def initial_pressure(self):
        return np.zeros(self.grid().num_cells)

    #---------Overloaded Functions-----------------

    def grid(self):
        raise NotImplementedError('subclass must overload function grid()')

    #--------Time stepping------------

    def time_step(self):
        return 1.0

    def end_time(self):
        return 1.0

    def save_results(self):
        pressures = self.data['pressure']
        times = np.array(self.data['times'])
        folder = self.parameters['folder_name']
        f_name = self.parameters['file_name']
        for i, p in enumerate(pressures):
            data_to_plot = {'pressure': p}
            export_vtk(
                self.grid(), f_name, data_to_plot, time_step=i, folder=folder)

        export_pvd(
            self.grid(), self.parameters['file_name'], times, folder=folder)
