import numpy as np
import scipy.sparse as sps

from porepy.grids import structured
from porepy.numerics.fv import tpfa, mass_matrix
from porepy.numerics.compressible import solvers
from porepy.numerics.mixed_dim.coupler import Coupler
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

    def save(self):
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


###############################################################################


class SubProblem(SlightlyCompressible):
    def __init__(self, g):
        self.g = g
        SlightlyCompressible.__init__(self)

    def grid(self):
        return self.g


class TpfaComp(tpfa.Tpfa):
    def __init__(self, physics='flow'):
        tpfa.Tpfa.__init__(self, physics)

    def matrix_rhs(self, g, data):
        param = data['problem'].data()
        return tpfa.Tpfa.matrix_rhs(self, g, param)


class TpfaCompCoupling(tpfa.TpfaCoupling):
    def __init__(self, solver):
        tpfa.TpfaCoupling.__init__(self, solver)

    def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
        param_h = data_h['problem'].data()
        param_l = data_l['problem'].data()
        return tpfa.TpfaCoupling.matrix_rhs(
            self, g_h, g_l, param_h, param_l, data_edge)


class SlightlyCompressibleMultiDim():
    """
    Base-class for slightly compressible flow. Initialize all needed
    attributes for a slightly compressible solver.
    """

    def __init__(self):
        self._data = dict()
        self.set_sub_problems()
        self.solver = solvers.Implicit(self)
        self.solver.parameters['store_results'] = True
        self.parameters = {'file_name': 'pressure'}
        self.parameters['folder_name'] = 'results'

    #---------Discretization---------------

    def flux_disc(self):
        """
        Returns the flux discretization.
        """
        discr = TpfaComp()
        coupling_conditions = TpfaCompCoupling(discr)
        return Coupler(discr, coupling_conditions)

    def time_disc(self):
        """
        Returns the flux discretization.
        """
        class TimeDisc(mass_matrix.MassMatrix):
            def __init__(self, deltaT):
                self.deltaT = deltaT

            def matrix_rhs(self, g, data):
                sub_d = data['problem'].data()
                sub_d['deltaT'] = self.deltaT
                lhs, rhs = mass_matrix.MassMatrix.matrix_rhs(self, g, sub_d)
                return lhs * sub_d['compressibility'], rhs * sub_d['compressibility']
        single_dim_discr = TimeDisc(self.time_step())
        multi_dim_discr = Coupler(single_dim_discr)
        return multi_dim_discr

    def solve(self):
        """
        Call the solver
        """
        self.data = self.solver.solve()
        return self.data

    def update(self, t):
        for _, d in self.grid():
            d['problem'].update(t)

    #-----Data-----------------

    def data(self):
        return self._data

    def set_sub_problems(self):
        self.grid().add_node_props(['problem'])
        for g, d in self.grid():
            d['problem'] = SubProblem(g)

    #--------Initial conditions---------------

    def initial_pressure(self):
        for _, d in self.grid():
            d['pressure'] = d['problem'].initial_pressure()

        global_pressure = self.time_disc().merge(self.grid(), 'pressure')
        return global_pressure

    #---------Overloaded Functions-----------------

    def grid(self):
        raise NotImplementedError('subclass must overload function grid()')

    #--------Time stepping------------

    def time_step(self):
        return 1.0

    def end_time(self):
        return 1.0

    def save(self, save_every=1):
        pressures = self.data['pressure'][::save_every]
        times = np.array(self.data['times'])[::save_every]
        folder = self.parameters['folder_name']
        f_name = self.parameters['file_name']

        for i, p in enumerate(pressures):
            self.time_disc().split(self.grid(), 'pressure', p)
            data_to_plot = ['pressure']
            export_vtk(
                self.grid(), f_name, data_to_plot, time_step=i, folder=folder)

        export_pvd(
            self.grid(), self.parameters['file_name'], times, folder=folder)
