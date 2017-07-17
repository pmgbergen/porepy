import numpy as np
import scipy.sparse as sps

from porepy.grids import structured
from porepy.numerics.fv import tpfa
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
        self.solver = solvers.Implicit(self)
        self.solver.parameters['store_results'] = True
        self.parameters = {'file_name': 'pressure'}
        self.parameters['folder_name'] = 'results'
        self.data = dict()
    #---------Discretization---------------

    def flux_disc(self):
        """
        Returns the flux discretization. 
        """
        return tpfa.Tpfa()

    def solve(self):
        """
        Call the solver
        """
        self.data = self.solver.solve()
        return self.data

    #-----Parameters------------
    def porosity(self):
        return np.ones(self.grid().num_cells)

    def compressibility(self):
        return 1

    def permeability(self):
        kxx = np.ones(self.grid().num_cells)
        return tensor.SecondOrder(self.grid().dim, kxx)

    #--------Inn/outflow terms---------------

    def initial_pressure(self):
        return np.zeros(self.grid().num_cells)

    def source(self, t):
        return np.zeros(self.g.num_cells)

    def bc(self):
        dir_bound = np.array([])
        return bc.BoundaryCondition(self.grid(), dir_bound,
                                    ['dir'] * dir_bound.size)

    def bc_val(self, t):
        return np.zeros(self.grid().num_faces)

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
