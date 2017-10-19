import numpy as np
import scipy.sparse as sps

from porepy.params.data import Parameters
from porepy.params import tensor, bc
from porepy.fracs import meshing
from porepy.numerics.fv import fvutils, tpfa, mass_matrix, source
from porepy.numerics.fv.transport import upwind, upwind_coupling
from porepy.numerics import pde_solver
from porepy.numerics.mixed_dim import coupler
from porepy.viz.exporter import export_vtk, export_pvd


class ParabolicProblem():
    '''
    Base class for solving general pde problems. This class solves equations of
    the type:
    dT/dt + v*\nabla T - \nabla K \nabla T = q

    Init:
    - gb (Grid/GridBucket) Grid or grid bucket for the problem
    - physics (string) Physics key word. See Parameters class for valid physics

    Functions:
    data(): returns data dictionary. Is only used for single grids (I.e. not
            GridBucket)
    solve(): solve problem
    step(): take one time step
    update(t): update parameters to time t
    reassemble(): reassemble matrices and right hand side
    solver(): initiate solver (see numerics.pde_solver)
    advective_disc(): discretization of the advective term
    diffusive_disc(): discretization of the diffusive term
    soruce_disc(): discretization of the source term, q
    space_disc(): returns one or more of the above discretizations. If
                  advective_disc(), source_disc() are returned we solve
                  the problem without diffusion
    time_disc(): returns the time discretization
    initial_condition(): returns the initial condition for global variable
    grid(): returns the grid bucket for the problem
    time_step(): returns time step length
    end_time(): returns end time
    save(save_every=1): save solution. Parameter: save_every, save only every
                                                  save_every time steps

    Example:
    # We create a problem with default data, neglecting the advective term

    class ExampleProblem(ParabolicProblem):
        def __init__(self, gb):
            self._g = gb
            ParabolicProblem.__init__(self)

        def space_disc(self):
            return self.source_disc(), self.diffusive_discr()
    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['problem'] = ParabolicData(g, d)
    problem = ExampleProblem(gb)
    problem.solve()
    '''

    def __init__(self, gb, physics='transport'):
        self._gb = gb
        self.physics = physics
        self._data = dict()
        self._set_data()
        self._solver = self.solver()
        self._solver.parameters['store_results'] = True
        self.parameters = {'file_name': physics}
        self.parameters['folder_name'] = 'results'

    def data(self):
        'Get data dictionary'
        return self._data

    def _set_data(self):
        for _, d in self.grid():
            d['deltaT'] = self.time_step()

    def solve(self):
        'Solve problem'
        return self._solver.solve()

    def step(self):
        'Take one time step'
        return self._solver.step()

    def update(self, t):
        'Update parameters to time t'
        for g, d in self.grid():
            d['problem'].update(t)

    def reassemble(self):
        'Reassemble matrices and rhs'
        return self._solver.reassemble()

    def solver(self):
        'Initiate solver'
        return pde_solver.Implicit(self)

    def advective_disc(self):
        'Discretization of term v * \nabla T'
        advection_discr = upwind.Upwind(physics=self.physics)
        advection_coupling = upwind_coupling.UpwindCoupling(advection_discr)
        advection_solver = coupler.Coupler(advection_discr, advection_coupling)
        return advection_solver

    def diffusive_disc(self):
        'Discretization of term \nabla K \nabla T'
        diffusive_discr = tpfa.TpfaMultiDim(physics=self.physics)
        return diffusive_discr

    def source_disc(self):
        'Discretization of source term, q'
        return source.IntegralMultiDim(physics=self.physics)

    def space_disc(self):
        '''Space discretization. Returns the discretization terms that should be
        used in the model'''
        return self.advective_disc(), self.diffusive_disc(), self.source_disc()

    def time_disc(self):
        """
        Returns the flux discretization.
        """
        mass_matrix_discr = mass_matrix.MassMatrix(physics=self.physics)
        multi_dim_discr = coupler.Coupler(mass_matrix_discr)
        return multi_dim_discr

    def initial_condition(self):
        'Returns initial condition for global variable'
        for _, d in self.grid():
            d[self.physics] = d['problem'].initial_condition()

        global_variable = self.time_disc().merge(self.grid(), self.physics)
        return global_variable

    def grid(self):
        'Returns grid/grid_bucket'
        return self._gb

    def time_step(self):
        'Returns the time step'
        return 1.0

    def end_time(self):
        'Returns the end time'
        return 1.0

    def save(self, save_every=1):
        'Saves the solution'
        variables = self.data()[self.physics][::save_every]
        times = np.array(self.data()['times'])[::save_every]
        folder = self.parameters['folder_name']
        f_name = self.parameters['file_name']

        for i, p in enumerate(variables):
            self.time_disc().split(self.grid(), self.physics, p)
            data_to_plot = [self.physics]
            export_vtk(
                self.grid(), f_name, data_to_plot, time_step=i, folder=folder)

        export_pvd(
            self.grid(), self.parameters['file_name'], times, folder=folder)


class ParabolicData():
    '''
    Base class for assigning valid data to a grid.
    Init:
    - g    (Grid) Grid that data should correspond to
    - d    (dictionary) data dictionary that data will be assigned to
    - physics (string) Physics key word. See Parameters class for valid physics

    Functions:
        update(t): Update source and bc term to time t
        bc: Set boundary condition
        bc_val(t): boundary condition value at time t
        initial_condition(): initial condition for problem
        source(): source term for problem
        porosity(): porosity of each cell
        diffusivity(): second order diffusivity tensor
        aperture(): the aperture of each cell
        data(): returns data dictionary
        grid(): returns the grid g

    Example:
    # We set an inflow and outflow boundary condition by overloading the
    # bc_val term
    class ExampleData(ParabolicData):
        def __init__(g, d):
            ParabolicData.__init__(self, g, d)
        def bc_val(self):
            left = self.grid().nodes[0] < 1e-6
            right = self.grid().nodes[0] > 1 - 1e-6
            val = np.zeros(g.num_faces)
            val[left] = 1
            val[right] = -1
            return val
    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['problem'] = ExampleData(g, d)
    '''

    def __init__(self, g, data, physics='transport'):
        self._g = g
        self._data = data
        self.physics = physics
        self._set_data()

    def update(self, t):
        'Update source and bc_val term to time step t'
        source = self.source(t)
        bc_val = self.bc_val(t)
        self.data()['param'].set_source(self.physics, source)
        self.data()['param'].set_bc_val(self.physics, bc_val)

    def bc(self):
        'Define boundary condition'
        dir_bound = np.array([])
        return bc.BoundaryCondition(self.grid(), dir_bound,
                                    ['dir'] * dir_bound.size)

    def bc_val(self, t):
        'Returns boundary condition values at time t'
        return np.zeros(self.grid().num_faces)

    def initial_condition(self):
        'Returns initial condition'
        return np.zeros(self.grid().num_cells)

    def source(self, t):
        'Returns source term'
        return np.zeros(self.grid().num_cells)

    def porosity(self):
        'Returns porosity'
        return np.ones(self.grid().num_cells)

    def diffusivity(self):
        'Returns diffusivity tensor'
        kxx = np.ones(self.grid().num_cells)
        return tensor.SecondOrder(self.grid().dim, kxx)

    def aperture(self):
        'Returns apperture of each cell'
        return np.ones(self.grid().num_cells)

    def data(self):
        'Returns data dictionary'
        return self._data

    def grid(self):
        'Returns grid'
        return self._g

    def _set_data(self):
        '''Create a Parameter object and assign data based on the returned 
        values from the functions (e.g., self.source(t))
        '''
        if 'param' not in self._data:
            self._data['param'] = Parameters(self.grid())
        self._data['param'].set_tensor(self.physics, self.diffusivity())
        self._data['param'].set_porosity(self.porosity())
        self._data['param'].set_bc(self.physics, self.bc())
        self._data['param'].set_bc_val(self.physics, self.bc_val(0.0))
        self._data['param'].set_source(self.physics, self.source(0.0))
        self._data['param'].set_aperture(self.aperture())
