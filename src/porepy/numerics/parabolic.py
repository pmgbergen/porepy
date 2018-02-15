import numpy as np
import logging
import time
import scipy.sparse as sps

from porepy.params.data import Parameters
from porepy.params import tensor, bc
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.numerics.fv import tpfa, mass_matrix, source
from porepy.numerics.fv.transport import upwind
from porepy.numerics import time_stepper
from porepy.numerics.mixed_dim import coupler
from porepy.viz.exporter import Exporter
from porepy.grids.grid_bucket import GridBucket

logger = logging.getLogger(__name__)


class ParabolicModel():
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

    def __init__(self, gb, physics='transport',time_step=1.0, end_time=1.0, **kwargs):
        self._gb = gb
        self.is_GridBucket = isinstance(self._gb, GridBucket)
        self.physics = physics
        self._data = kwargs.get('data', dict())
        self._time_step = time_step
        self._end_time = end_time
        self._set_data()

        self._solver = self.solver()

        logger.info('Create exporter')
        tic = time.time()
        file_name = kwargs.get('file_name', 'solution')
        folder_name = kwargs.get('folder_name', 'results')
        self.exporter = Exporter(self._gb, file_name, folder_name)
        logger.info('Done. Elapsed time: ' + str(time.time() - tic))

        self.x_name = 'solution'
        self._time_disc = self.time_disc()

    def data(self):
        'Get data dictionary'
        return self._data

    def _set_data(self):
        if self.is_GridBucket:
            for _, d in self.grid():
                d['deltaT'] = self.time_step()
        else:
            self.data()['deltaT'] = self.time_step()

    def solve(self, save_as=None, save_every=1):
        '''Solve problem

        Arguments:
        save_as (string), defaults to None. If a string is given, the solution
                          variable is saved to a vtk-file as save_as
        save_every (int), defines which time steps to save. save_every=2 will
                          store every second time step.
        '''
        tic = time.time()
        logger.info('Solve problem')
        s = self._solver.solve(save_as, save_every)
        logger.info('Done. Elapsed time: ' + str(time.time() - tic))
        return s

    def step(self):
        'Take one time step'
        return self._solver.step()

    def update(self, t):
        'Update parameters to time t'
        if self.is_GridBucket:
            for g, d in self.grid():
                d[self.physics + '_data'].update(t)
        else:
            self.data()[self.physics + '_data'].update(t)

    def split(self, x_name='solution'):
        self.x_name = x_name
        self._time_disc.split(self.grid(), self.x_name, self._solver.p)

    def reassemble(self):
        'Reassemble matrices and rhs'
        return self._solver.reassemble()

    def solver(self):
        'Initiate solver'
        return time_stepper.Implicit(self)

    def advective_disc(self):
        'Discretization of fluid_density*fluid_specific_heat * v * \nabla T'

        class WeightedUpwindDisc(upwind.Upwind):
            def __init__(self):
                self.physics = 'transport'

            def matrix_rhs(self, g, data):
                lhs, rhs = upwind.Upwind.matrix_rhs(self, g, data)
                factor = data['param'].fluid_specific_heat\
                       * data['param'].fluid_density
                lhs *= factor
                rhs *= factor
                return lhs, rhs

        class WeightedUpwindCoupler(upwind.UpwindCoupling):
            def __init__(self, discr):
                self.physics = 'transport'
                upwind.UpwindCoupling.__init__(self, discr)

            def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
                cc = upwind.UpwindCoupling.matrix_rhs(self, g_h, g_l, data_h,
                                                      data_l, data_edge)
                factor = data_h['param'].fluid_specific_heat \
                       * data_h['param'].fluid_density
                return cc * factor

        class WeightedUpwindMixedDim(upwind.UpwindMixedDim):

            def __init__(self):
                self.physics = 'transport'

                self.discr = WeightedUpwindDisc()
                self.discr_ndof = self.discr.ndof
                self.coupling_conditions = WeightedUpwindCoupler(self.discr)

                self.solver = coupler.Coupler(self.discr,
                                             self.coupling_conditions)

        if self.is_GridBucket:
            upwind_discr = WeightedUpwindMixedDim()
        else:
            upwind_discr = WeightedUpwindDisc()
        return upwind_discr

    def diffusive_disc(self):
        'Discretization of term \nabla K \nabla T'
        if self.is_GridBucket:
            diffusive_discr = tpfa.TpfaMixedDim(physics=self.physics)
        else:
            diffusive_discr = tpfa.Tpfa(physics=self.physics)
        return diffusive_discr

    def source_disc(self):
        'Discretization of source term, q'
        if self.is_GridBucket:
            return source.IntegralMixedDim(physics=self.physics)
        else:
            return source.Integral(physics=self.physics)

    def space_disc(self):
        '''Space discretization. Returns the discretization terms that should be
        used in the model'''
        return self.advective_disc(), self.diffusive_disc(), self.source_disc()

    def time_disc(self):
        """
        Returns the time discretization.
        """
        class TimeDisc(mass_matrix.MassMatrix):
            def __init__(self, deltaT):
                self.deltaT = deltaT

            def matrix_rhs(self, g, data):
                ndof = g.num_cells
                aperture = data['param'].get_aperture()
                coeff = g.cell_volumes * aperture / self.deltaT

                factor_fluid = data['param'].fluid_specific_heat\
                             * data['param'].fluid_density\
                             * data['param'].porosity
                factor_rock = data['param'].rock_specific_heat\
                             * data['param'].rock_density\
                             * (1 - data['param'].porosity)
                factor = sps.dia_matrix((factor_fluid + factor_rock, 0),
                                        shape=(ndof, ndof))

                lhs = sps.dia_matrix((coeff, 0), shape=(ndof, ndof))
                rhs = np.zeros(ndof)
                return factor * lhs, factor * rhs

        single_dim_discr = TimeDisc(self.time_step())
        if self.is_GridBucket:
            time_discretization = coupler.Coupler(single_dim_discr)
        else:
            time_discretization = TimeDisc(self.time_step())
        return time_discretization

    def initial_condition(self):
        'Returns initial condition for global variable'
        if self.is_GridBucket:
            for _, d in self.grid():
                d[self.physics] = d[self.physics + '_data'].initial_condition()
            global_variable = self.time_disc().merge(self.grid(), self.physics)
        else:
            global_variable = self._data[self.physics + '_data'].initial_condition()
        return global_variable

    def grid(self):
        'Returns grid/grid_bucket'
        return self._gb

    def time_step(self):
        'Returns the time step'
        return self._time_step

    def end_time(self):
        'Returns the end time'
        return self._end_time


class ParabolicDataAssigner():
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
        rock_specific_heat(): Specific heat of the rock. Constant.
        fluid_specific_heat(): Specific heat of the fluid. Constant.
        rock_density(): Density of the rock. Constant.
        fluid_density(): Density of the fluid. Constant.
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
        '''Returns apperture of each cell. If None is returned, default
        Parameter class value is used'''
        return None

    def diffusivity(self):
        'Returns diffusivity tensor'
        kxx = np.ones(self.grid().num_cells)
        return tensor.SecondOrder(self.grid().dim, kxx)

    def aperture(self):
        '''Returns apperture of each cell. If None is returned, default
        Parameter class value is used'''
        return None

    def rock_specific_heat(self):
        """ Returns *constant* specific heat capacity of rock. If None is
        returned, default Parameter class value is used.
        """
        return None

    def fluid_specific_heat(self):
        """ Returns *constant* specific heat capacity of fluid. If None is
        returned, default Parameter class value is used.
        """
        return None

    def rock_density(self):
        """ Returns *constant* density of rock. If None is
        returned, default Parameter class value is used.
        """
        return None

    def fluid_density(self):
        """ Returns *constant* density of fluid. If None is
        returned, default Parameter class value is used.
        """
        return None

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
        self._data['param'].set_bc(self.physics, self.bc())
        self._data['param'].set_bc_val(self.physics, self.bc_val(0.0))
        self._data['param'].set_source(self.physics, self.source(0.0))

        if self.porosity() is not None:
            self._data['param'].set_porosity(self.porosity())
        if self.aperture() is not None:
            self._data['param'].set_aperture(self.aperture())
        if self.rock_specific_heat() is not None:
            self._data['param'].set_rock_specific_heat(self.rock_specific_heat())
        if self.fluid_specific_heat() is not None:
            self._data['param'].set_fluid_specific_heat(self.fluid_specific_heat())
        if self.rock_density() is not None:
            self._data['param'].set_rock_density(self.rock_density())
        if self.fluid_density() is not None:
            self._data['param'].set_fluid_density(self.fluid_density())

