import numpy as np
import logging
import time
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.fv.transport.upwind import UpwindCoupling
logger = logging.getLogger(__name__)


class ParabolicModel():
    '''
    Base class for discretizing general pde problems. This class solves equations of
    the type:
    dT/dt + v*\nabla T - \nabla K \nabla T = q

    Init:
    - gb (Grid/GridBucket) Grid or grid bucket for the problem
    - physics (string) Physics key word. See Parameters class for valid physics

    Functions:
    data(): returns data dictionary. Is only used for single grids (I.e. not
            GridBucket)
    update(t=0): update parameters to time t. Assumes a ParabolicDataAssigner
                 is assigned to data with keyword: physics + '_data'
    mass_disc(): discretization of the mass term (should give out the mass
                 mastrix but not divided by the time step)
    advective_disc(): discretization of the advective term
    diffusive_disc(): discretization of the diffusive term
    soruce_disc(): discretization of the source term, q
    space_disc(): returns one or more of the above discretizations. If
                  advective_disc(), source_disc() are returned we solve
                  the problem without diffusion
    initial_condition(): returns the initial condition for global variable. 
                         Assumes a ParabolicDataAssigner is assigned to data
                         with keyword: physics + '_data'
    grid(): returns the grid bucket for the problem

    Example:
    # We create a problem with default data, neglecting the advective term.
    # We define an explicit time stepper 

    class ExampleModel(ParabolicModel):
        def space_disc(self):
            return self.source_disc(), self.diffusive_discr()
    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['transport_data'] = ParabolicData(g, d)
    model = ExampleModel(gb)
    stepper = Implicit(model, dt=0.2, end_time=1.0)
    stepper.solve()
    '''

    def __init__(self, gb, physics='transport', **kwargs):
        self._gb = gb
        self.is_GridBucket = isinstance(self._gb, pp.GridBucket)
        self.physics = physics
        self._data = kwargs.get('data', dict())
        
        self.x_name = physics
        self._mass_disc = self.mass_disc()

    def data(self):
        'Get data dictionary'
        return self._data

    def update(self, t=0):
        'Update parameters to time t'
        if self.is_GridBucket:
            for _, d in self.grid():
                d[self.physics + '_data'].update(t)
        else:
            self.data()[self.physics + '_data'].update(t)

    def split(self, x, x_name='solution'):
        self.x_name = x_name
        self._mass_disc.split(self.grid(), self.x_name, x)

    def advective_disc(self):
        'Discretization of fluid_density*fluid_specific_heat * v * \nabla T'

        class WeightedUpwindDisc(pp.Upwind):
            def __init__(self, physics):
                self.physics = physics

            def matrix_rhs(self, g, data):
                lhs, rhs = pp.Upwind.matrix_rhs(self, g, data)
                factor = data['param'].fluid_specific_heat\
                       * data['param'].fluid_density
                lhs *= factor
                rhs *= factor
                return lhs, rhs

        class WeightedUpwindCoupler(UpwindCoupling):
            def __init__(self, discr):
                UpwindCoupling.__init__(self, discr)

            def matrix_rhs(self, g_h, g_l, data_h, data_l, data_edge):
                cc = UpwindCoupling.matrix_rhs(self, g_h, g_l, data_h,
                                                      data_l, data_edge)
                factor = data_h['param'].fluid_specific_heat \
                       * data_h['param'].fluid_density
                return cc * factor

        class WeightedUpwindMixedDim(pp.UpwindMixedDim):

            def __init__(self, physics):
                self.physics = physics

                self.discr = WeightedUpwindDisc(self.physics)
                self.discr_ndof = self.discr.ndof
                self.coupling_conditions = WeightedUpwindCoupler(self.discr)

                self.solver = pp.Coupler(self.discr,
                                         self.coupling_conditions)

        if self.is_GridBucket:
            upwind_discr = WeightedUpwindMixedDim(self.physics)
        else:
            upwind_discr = WeightedUpwindDisc(self.physics)
        return upwind_discr

    def diffusive_disc(self):
        'Discretization of term \nabla K \nabla T'
        if self.is_GridBucket:
            diffusive_discr = pp.TpfaMixedDim(physics=self.physics)
        else:
            diffusive_discr = pp.Tpfa(physics=self.physics)
        return diffusive_discr

    def source_disc(self):
        'Discretization of source term, q'
        if self.is_GridBucket:
            return pp.IntegralMixedDim(physics=self.physics)
        else:
            return pp.Integral(physics=self.physics)

    def space_disc(self):
        '''Space discretization. Returns the discretization terms that should be
        used in the model'''
        return self.advective_disc(), self.diffusive_disc(), self.source_disc()

    def mass_disc(self):
        """
        Returns the time discretization.
        """
        class TimeDisc(pp.MassMatrix):

            def matrix_rhs(self, g, data):
                ndof = g.num_cells
                aperture = data['param'].get_aperture()
                coeff = g.cell_volumes * aperture

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

        single_dim_discr = TimeDisc()
        if self.is_GridBucket:
            time_discretization = pp.Coupler(single_dim_discr)
        else:
            time_discretization = TimeDisc()
        return time_discretization

    def initial_condition(self):
        'Returns initial condition for global variable'
        if self.is_GridBucket:
            for _, d in self.grid():
                d[self.physics] = d[self.physics + '_data'].initial_condition()
            global_variable = self.mass_disc().merge(self.grid(), self.physics)
        else:
            global_variable = self._data[self.physics + '_data'].initial_condition()
        return global_variable

    def grid(self):
        'Returns grid/grid_bucket'
        return self._gb


class ParabolicDataAssigner():
    '''
    Base class for assigning valid data to a grid.
    Init:
    - g    (Grid) Grid that data should correspond to
    - d    (dictionary) data dictionary that data will be assigned to
    - physics (string) Physics key word. See Parameters class for valid physics

    Functions:
        update(t=0): Update source and bc term to time t
        bc: Set boundary condition
        bc_val(t=0): boundary condition value at time t
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

    def update(self, t=0.):
        'Update source and bc_val term to time step t'
        source = self.source(t)
        bc_val = self.bc_val(t)
        self.data()['param'].set_source(self.physics, source)
        self.data()['param'].set_bc_val(self.physics, bc_val)

    def bc(self):
        'Define boundary condition'
        dir_bound = np.array([])
        return pp.BoundaryCondition(self.grid(), dir_bound,
                                    ['dir'] * dir_bound.size)

    def bc_val(self, t=0.):
        'Returns boundary condition values at time t'
        return np.zeros(self.grid().num_faces)

    def initial_condition(self):
        'Returns initial condition'
        return np.zeros(self.grid().num_cells)

    def source(self, t=0.):
        'Returns source term'
        return np.zeros(self.grid().num_cells)

    def porosity(self):
        '''Returns apperture of each cell. If None is returned, default
        Parameter class value is used'''
        return None

    def diffusivity(self):
        'Returns diffusivity tensor'
        kxx = np.ones(self.grid().num_cells)
        return pp.SecondOrderTensor(self.grid().dim, kxx)

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
            self._data['param'] = pp.Parameters(self.grid())
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

