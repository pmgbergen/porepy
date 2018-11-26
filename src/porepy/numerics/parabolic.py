import numpy as np
import logging
import time
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class ParabolicModel:
    """
    Base class for solving general pde problems. This class solves equations of
    the type:
    dT/dt + v*\nabla T - \nabla K \nabla T = q

    Init:
    - gb (Grid/GridBucket) Grid or grid bucket for the problem
    - keyword (string) Keyword key word. See Parameters class for valid keyword

    Functions:
    data(): returns data dictionary. Is only used for single grids (I.e. not
            GridBucket)
    solve(): solve problem
    step(): take one time step
    update(t): update parameters to time t
    reassemble(): reassemble matrices and right hand side
    time_stepper(): initiate solver (see numerics.time_stepper)
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
    """

    def __init__(
        self,
        gb,
        keyword="transport",
        time_step=1.0,
        end_time=1.0,
        callback=None,
        **kwargs
    ):
        self._gb = gb
        self.is_GridBucket = isinstance(self._gb, pp.GridBucket)
        self.keyword = keyword
        self._data = kwargs.get("data", dict())
        self._time_step = time_step
        self._end_time = end_time

        self._disc_name = None
        self._time_name = "time_disc"

        self.callback = callback
        self.x_name = "solution"
        self.assembler = self._set_discretization()
        _, _, block_dof, full_dof = self.assembler._initialize_matrix_rhs(gb, None)
        self._block_dof = block_dof
        self._full_dof = full_dof

        self.time_stepper = self.time_stepper()

        logger.info("Create exporter")
        tic = time.time()
        file_name = kwargs.get("file_name", "solution")
        folder_name = kwargs.get("folder_name", "results")
        self.exporter = pp.Exporter(self._gb, file_name, folder_name)
        logger.info("Done. Elapsed time: " + str(time.time() - tic))

    def data(self):
        "Get data dictionary"
        return self._data

    def solve(self, save_as=None, save_every=1):
        """Solve problem

        Arguments:
        save_as (string), defaults to None. If a string is given, the solution
                          variable is saved to a vtk-file as save_as
        save_every (int), defines which time steps to save. save_every=2 will
                          store every second time step.
        """
        tic = time.time()
        logger.info("Solve problem, saving every " + str(save_every))
        s = self.time_stepper.solve(save_as, save_every)
        logger.info("Done. Elapsed time: " + str(time.time() - tic))
        return s

    def step(self):
        "Take one time step"
        return self.time_stepper.step()

    def update(self, t):
        "Update parameters to time t"
        if self.is_GridBucket:
            for g, d in self.grid():
                d[self.keyword + "_data"].update(t)
        else:
            self.data()[pp.keywords.PARAMETERS][self.keyword].update(t)

        if self.callback is not None:
            self.callback(self)

    def split(self):
        self.assembler.distribute_variable(
            self.grid(), self.time_stepper.p, self._block_dof, self._full_dof
        )

    def reassemble(self):
        "Reassemble matrices and rhs"
        return self.time_stepper.reassemble()

    def time_stepper(self):
        "Initiate solver"
        return pp.Implicit(self)

    def advective_disc(self):
        "Discretization of fluid_density*fluid_specific_heat * v * \nabla T"

        class WeightedUpwindDisc(pp.Upwind):
            def __init__(self, keyword):
                self.keyword = keyword
                pp.Upwind.__init__(self, self.keyword)

            def matrix_rhs(self, g, data):
                lhs, rhs = pp.Upwind.matrix_rhs(self, g, data)
                parameter_dictionary = data[pp.keywords.PARAMETERS][self.keyword]
                factor = (
                    parameter_dictionary["fluid_specific_heat"]
                    * parameter_dictionary["fluid_density"]
                )
                lhs *= factor
                rhs *= factor
                return lhs, rhs

        class WeightedUpwindCoupler(pp.UpwindCoupling):
            def __init__(self, keyword):
                self.keyword = keyword
                pp.UpwindCoupling.__init__(self, self.keyword)

            def matrix_rhs(self, matrix, g_h, g_l, data_h, data_l, data_edge):
                cc = pp.UpwindCoupling.matrix_rhs(
                    self, matrix, g_h, g_l, data_h, data_l, data_edge
                )
                parameter_dictionary = data_h[pp.keywords.PARAMETERS][self.keyword]
                factor = (
                    parameter_dictionary["fluid_specific_heat"]
                    * parameter_dictionary["fluid_density"]
                )
                return (cc - matrix) * factor + matrix

        return (WeightedUpwindDisc(self.keyword), WeightedUpwindCoupler(self.keyword))

    def diffusive_disc(self):
        "Discretization of term \nabla K \nabla T"
        tpfa = pp.Tpfa(self.keyword)
        return (tpfa, pp.RobinCoupling(self.keyword, tpfa))

    def source_disc(self):
        "Discretization of source term, q"
        return (pp.Integral(self.keyword), None)

    def space_disc(self):
        """Space discretization. Returns the discretization terms that should be
        used in the model"""
        return [self.advective_disc(), self.diffusive_disc(), self.source_disc()]

    def time_disc(self):
        """
        Returns the time discretization.
        """

        class TimeDisc(object):
            def __init__(self, time_step, keyword):
                self.keyword = keyword
                self.time_step = time_step

            def assemble_matrix_rhs(self, g, data):
                ndof = g.num_cells
                parameter_dictionary = data[pp.keywords.PARAMETERS][self.keyword]
                aperture = parameter_dictionary["aperture"]
                coeff = g.cell_volumes * aperture / self.time_step

                factor_fluid = (
                    parameter_dictionary["fluid_specific_heat"]
                    * parameter_dictionary["fluid_density"]
                    * parameter_dictionary["porosity"]
                )
                factor_rock = (
                    parameter_dictionary["rock_specific_heat"]
                    * parameter_dictionary["rock_density"]
                    * (1 - parameter_dictionary["porosity"])
                )
                factor = sps.dia_matrix(
                    (factor_fluid + factor_rock, 0), shape=(ndof, ndof)
                )

                lhs = sps.dia_matrix((coeff, 0), shape=(ndof, ndof))
                rhs = np.zeros(ndof)
                return factor * lhs, factor * rhs

        return (TimeDisc(self.time_step(), self.keyword), None)

    def discretize(self):
        for g, d in self._gb:
            self._time_name

        lhs, rhs, self._block_dof, self._full_dof = self.assembler.assemble_matrix_rhs(
            self.grid(), add_matrices=False
        )

        time = self._time_name + "_" + self.keyword
        lhs_time = lhs[time]
        rhs_time = rhs[time]

        lhs_space = []
        rhs_space = []
        for key in lhs.keys():
            if key != time:
                lhs_space.append(lhs[key])
                rhs_space.append(rhs[key])
        lhs_space = sum(lhs_space)
        rhs_space = sum(rhs_space)

        return lhs_time, lhs_space, rhs_time, rhs_space

    def _set_discretization(self):
        if self.is_GridBucket:
            key = self.keyword
            var = self.x_name
            node_disc = {}
            edge_disc = []
            disc_name = []
            for i, disc in enumerate(self.space_disc()):
                loc_name = "disc_" + str(i)
                node_disc[loc_name] = disc[0]
                edge_disc.append(disc[1])
                disc_name.append(loc_name)
            self._disc_name = disc_name
            time_disc, _ = self.time_disc()
            for g, d in self._gb:
                # Choose discretization and define the solver
                d[pp.keywords.PRIMARY_VARIABLES] = {var: {"cells": 1}}
                node_disc[self._time_name] = time_disc
                d[pp.keywords.DISCRETIZATION] = {key: node_disc}

            for e, d in self._gb.edges():
                g_slave, g_master = self._gb.nodes_of_edge(e)
                num_mortars = 0
                d[pp.keywords.PRIMARY_VARIABLES] = {}
                d[pp.keywords.COUPLING_DISCRETIZATION] = {}
                for i, disc in enumerate(edge_disc):
                    if disc is None:
                        continue
                    d[pp.keywords.COUPLING_DISCRETIZATION][disc_name[i]] = {
                        g_slave: (key, disc_name[i]),
                        g_master: (key, disc_name[i]),
                        e: (key + "_mortar", disc),
                    }
                    d[pp.keywords.PRIMARY_VARIABLES][var + "_mortar"] = {"cells": 1}

            assembler = pp.Assembler()
            return assembler
        else:
            raise NotImplementedError()

    def initial_condition(self):
        "Returns initial condition for global variable"
        if self.is_GridBucket:
            for _, d in self.grid():
                d[self.keyword] = d[self.keyword + "_data"].initial_condition()
            global_variable = self.assembler.merge_variable(
                self.grid(), self.keyword, self._block_dof, self._full_dof
            )
        else:
            global_variable = self._data[self.keyword + "_data"].initial_condition()
        return global_variable

    def grid(self):
        "Returns grid/grid_bucket"
        return self._gb

    def time_step(self):
        "Returns the time step"
        return self._time_step

    def end_time(self):
        "Returns the end time"
        return self._end_time

    def save(self, variables=None, save_every=1):
        if variables is None:
            self.exporter.write_vtk()
        else:
            if not self.is_GridBucket:
                variables = {k: self._data[k] for k in variables if k in self._data}

            time = self.time_stepper.data["times"][::save_every].copy()

            for time_step, current_time in enumerate(time):
                for v in variables:
                    v_data = self.time_stepper.data[self.keyword][time_step]
                    self._time_disc.split(self.grid(), v, v_data)
                self.exporter.write_vtk(variables, time_step=time_step)
            self.exporter.write_pvd(time)


class ParabolicDataAssigner:
    """
    Base class for assigning valid data to a grid.
    Init:
    - g    (Grid) Grid that data should correspond to
    - d    (dictionary) data dictionary that data will be assigned to
    - keyword (string) Keyword key word. See Parameters class for valid keyword

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
    """

    def __init__(self, g, data, keyword="transport"):
        self._g = g
        self._data = data
        self.keyword = keyword
        self._set_data()

    def update(self, t):
        "Update source and bc_val term to time step t"
        source = self.source(t)
        bc_val = self.bc_val(t)
        self.data()[pp.keywords.PARAMETERS][self.keyword]["source"] = source
        self.data()[pp.keywords.PARAMETERS][self.keyword]["bc_values"] = bc_val

    def bc(self):
        "Define boundary condition"
        return pp.BoundaryCondition(self.grid())

    def bc_val(self, t):
        "Returns boundary condition values at time t"
        return np.zeros(self.grid().num_faces)

    def initial_condition(self):
        "Returns initial condition"
        return np.zeros(self.grid().num_cells)

    def source(self, t):
        "Returns source term"
        return np.zeros(self.grid().num_cells)

    def porosity(self):
        """Returns apperture of each cell. If None is returned, default
        Parameter class value is used"""
        return np.ones(self.grid().num_cells)

    def diffusivity(self):
        "Returns diffusivity tensor"
        kxx = np.ones(self.grid().num_cells)
        return pp.SecondOrderTensor(self.grid().dim, kxx)

    def aperture(self):
        """Returns apperture of each cell. If None is returned, default
        Parameter class value is used"""
        return np.ones(self.grid().num_cells)

    def rock_specific_heat(self):
        """ Returns *constant* specific heat capacity of rock. If None is
        returned, default Parameter class value is used.
        """
        return np.ones(self.grid().num_cells)

    def fluid_specific_heat(self):
        """ Returns *constant* specific heat capacity of fluid. If None is
        returned, default Parameter class value is used.
        """
        return np.ones(self.grid().num_cells)

    def rock_density(self):
        """ Returns *constant* density of rock. If None is
        returned, default Parameter class value is used.
        """
        return np.ones(self.grid().num_cells)

    def fluid_density(self):
        """ Returns *constant* density of fluid. If None is
        returned, default Parameter class value is used.
        """
        return np.ones(self.grid().num_cells)

    def data(self):
        "Returns data dictionary"
        return self._data

    def grid(self):
        "Returns grid"
        return self._g

    def _set_data(self):
        """Create a Parameter object and assign data based on the returned
        values from the functions (e.g., self.source(t))
        """
        if pp.keywords.PARAMETERS not in self._data:
            self._data[pp.keywords.PARAMETERS] = pp.Parameters(
                self.grid(), [self.keyword], [{}]
            )
        self._data[pp.keywords.PARAMETERS].update_dictionaries([self.keyword], [{}])
        parameter_dictionary = self._data[pp.keywords.PARAMETERS][self.keyword]

        parameter_dictionary["second_order_tensor"] = self.diffusivity()
        parameter_dictionary["bc"] = self.bc()
        parameter_dictionary["bc_values"] = self.bc_val(0.0)
        parameter_dictionary["source"] = self.source(0.0)

        if self.porosity() is not None:
            parameter_dictionary["porosity"] = self.porosity()
        if self.aperture() is not None:
            parameter_dictionary["aperture"] = self.aperture()
        if self.rock_specific_heat() is not None:
            parameter_dictionary["rock_specific_heat"] = self.rock_specific_heat()
        if self.fluid_specific_heat() is not None:
            parameter_dictionary["fluid_specific_heat"] = self.fluid_specific_heat()
        if self.rock_density() is not None:
            parameter_dictionary["rock_density"] = self.rock_density()
        if self.fluid_density() is not None:
            parameter_dictionary["fluid_density"] = self.fluid_density()
