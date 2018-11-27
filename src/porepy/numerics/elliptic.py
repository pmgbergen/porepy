"""
Module for initializing, assigning data, solve, and save an elliptic pde.
Ths can for example be incompressible flow
problem assuming darcy's law. Please see the tutorial darcy's equation on the
porepy github: https://github.com/pmgbergen/porepy
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import time
import logging

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)


class EllipticModel:
    """
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
    keyword: (string): defaults to 'flow'

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
    discharge(): Calls split('pressure'). Then calculate the discharges over each
                 face in the grids and between edges in the GridBucket
    save(): calls split('pressure'). Then export the pressure to a vtk file to the
            folder kwargs['folder_name'] with file name
            kwargs['file_name'], default values are 'results' for the folder and
            keyword for the file name.
    """

    def __init__(self, gb, data=None, keyword="flow", **kwargs):
        self.keyword = keyword
        self._gb = gb
        self.is_GridBucket = isinstance(self._gb, pp.GridBucket)
        self._data = data

        self.lhs = []
        self.rhs = []
        self.x = []

        file_name = kwargs.get("file_name", keyword)
        folder_name = kwargs.get("folder_name", "results")
        mesh_kw = kwargs.get("mesh_kw", {})

        tic = time.time()
        logger.info("Create exporter")
        self.exporter = pp.Exporter(self._gb, file_name, folder_name, **mesh_kw)
        logger.info("Elapsed time: " + str(time.time() - tic))

        self._discr = self._set_discretization()

    def solve(self, max_direct=40000, callback=False, **kwargs):
        """ Reassemble and solve linear system.

        After the funtion has been called, the attributes lhs and rhs are
        updated according to the parameter states. Also, the attribute x
        gives the pressure given the current state.

        TODO: Provide an option to save solver information if multiple
        systems are to be solved with the same left hand side.

        The function attempts to set up the best linear solver based on the
        system size. The setup and parameter choices here are still
        experimental.

        Parameters:
            max_direct (int): Maximum number of unknowns where a direct solver
                is applied. If a direct solver can be applied this is usually
                the most efficient option. However, if the system size is
                too large compared to available memory, a direct solver becomes
                extremely slow.
            callback (boolean, optional): If True iteration information will be
                output when an iterative solver is applied (system size larger
                than max_direct)

        Returns:
            np.array: Pressure state.

        """
        logger.error("Solve elliptic model")
        # Discretize
        tic = time.time()
        logger.warning("Discretize")
        self.lhs, self.rhs = self.reassemble()
        logger.warning("Done. Elapsed time " + str(time.time() - tic))

        # Solve
        tic = time.time()
        ls = pp.numerics.linalg.linsolve.Factory()
        if self.rhs.size < max_direct:
            logger.warning("Solve linear system using direct solver")
            self.x = ls.direct(self.lhs, self.rhs)
        else:
            logger.warning("Solve linear system using GMRES")
            precond = self._setup_preconditioner()
            #            precond = ls.ilu(self.lhs)
            slv = ls.gmres(self.lhs)
            self.x, info = slv(
                self.rhs,
                M=precond,
                callback=callback,
                maxiter=10000,
                restart=1500,
                tol=1e-8,
            )
            if info == 0:
                logger.warning("GMRES succeeded.")
            else:
                logger.warning("GMRES failed with status " + str(info))

        logger.warning("Done. Elapsed time " + str(time.time() - tic))
        return self.x

    def step(self):
        return self.solve()

    def reassemble(self):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system.
        """
        lhs, rhs = self._discretize()
        self.lhs = lhs
        self.rhs = rhs
        return self.lhs, self.rhs

    def _set_discretization(self):
        if self.is_GridBucket:
            key = self.keyword

            tpfa = pp.Tpfa(self.keyword)
            source = pp.Integral(self.keyword)
            for g, d in self._gb:
                # Choose discretization and define the solver
                d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
                d[pp.DISCRETIZATION] = {key: {"flux": tpfa, "source": source}}

            for e, d in self._gb.edges():
                g_slave, g_master = self._gb.nodes_of_edge(e)
                d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
                d[pp.COUPLING_DISCRETIZATION] = {
                    "flux": {
                        g_slave: (key, "flux"),
                        g_master: (key, "flux"),
                        e: (key, pp.RobinCoupling(key, tpfa)),
                    }
                }
            assembler = pp.Assembler()

            return assembler
        else:
            return pp.Tpfa(keyword=self.keyword)

    def _discretize(self):

        if self.is_GridBucket:
            A, b, block_dof, full_dof = self._discr.assemble_matrix_rhs(self.grid())
            self._block_dof = block_dof
            self._full_dof = full_dof
            return A, b
        else:
            A, b = self._discr.assemble_matrix_rhs(self.grid(), self.data())
            return A, b

    def grid(self):
        return self._gb

    def data(self):
        return self._data

    def split(self, x_name="solution"):
        self.x_name = x_name
        if self.is_GridBucket:
            self._discr.distribute_variable(
                self.grid(), self.x, self._block_dof, self._full_dof
            )
        else:
            self._flux_disc.split(self.grid(), self.x_name, self.x)

    def pressure(self, pressure_name="pressure"):
        self.pressure_name = pressure_name
        if self.is_GridBucket:
            self.split(self.pressure_name)
        else:
            self._data[self.pressure_name] = self.x

    def discharge(self, discharge_name="discharge"):
        if self.is_GridBucket:
            pp.numerics.fv.fvutils.compute_discharges(
                self.grid(), self.keyword, p_name=self.pressure_name
            )
        else:
            pp.numerics.fv.fvutils.compute_discharges(
                self.grid(),
                self.keyword,
                discharge_name,
                self.pressure_name,
                self._data,
            )

    def permeability(self, perm_names=["kxx", "kyy", "kzz"]):
        """ Assign permeability to self._data, ready for export to vtk.

        For the moment, we only dump the main diagonals of the permeabliity.
        Extensions should be trivial if needed.

        Parameters:
            perm_names (list): Which components to export. Defaults to kxx,
                kyy and xzz.

        """

        def get_ind(n):
            if n == "kxx":
                return 0
            elif n == "kyy":
                return 1
            elif n == "kzz":
                return 2
            else:
                raise ValueError("Unknown perm keyword " + n)

        for n in perm_names:
            ind = get_ind(n)
            if self.is_GridBucket:
                for _, d in self.grid():
                    d[n] = d[pp.PARAMETERS][self.keyword]["second_order_tensor"].values[
                        ind, ind, :
                    ]
            else:
                self._data[n] = self._data[pp.PARAMETERS][self.keyword][
                    "second_order_tensor"
                ].values[ind, ind, :]

    def porosity(self, poro_name="porosity"):
        if self.is_GridBucket:
            for _, d in self.grid():
                d[poro_name] = d[pp.PARAMETERS][self.keyword]["porosity"]
        else:
            self._data[poro_name] = self._data[pp.PARAMETERS][self.keyword]["porosity"]

    def save(self, variables=None, save_every=None):
        if variables is None:
            self.exporter.write_vtk()
        else:
            if not self.is_GridBucket:
                variables = {k: self._data[k] for k in variables if k in self._data}
            self.exporter.write_vtk(variables)

    # Helper functions for linear solve below
    def _setup_preconditioner(self):
        solvers, ind, not_ind = self._assign_solvers()

        def precond(r):
            x = np.zeros_like(r)
            for s, i, ni in zip(solvers, ind, not_ind):
                x[i] += s(r[i])
            return x

        def precond_mult(r):
            x = np.zeros_like(r)
            A = self.lhs
            for s, i, ni in zip(solvers, ind, not_ind):
                r_i = r[i] - A[i, :][:, ni] * x[ni]
                x[i] += s(r_i)
            return x

        def M(r):
            return precond(r)

        return spl.LinearOperator(self.lhs.shape, M)

    def _assign_solvers(self):
        mat, ind = self._obtain_submatrix()
        all_ind = np.arange(self.rhs.size)
        not_ind = [np.setdiff1d(all_ind, i) for i in ind]

        factory = pp.numerics.linalg.linsolve.Factory()
        num_mat = len(mat)
        solvers = np.empty(num_mat, dtype=np.object)
        for i, A in enumerate(mat):
            sz = A.shape[0]
            if sz < 5000:
                solvers[i] = factory.direct(A)
            else:
                # amg solver is pyamg is installed, if not ilu
                try:
                    solvers[i] = factory.amg(A, as_precond=True)
                except ImportError:
                    solvers[i] = factory.ilu(A)

        return solvers, ind, not_ind

    def _obtain_submatrix(self):

        if isinstance(self.grid(), pp.GridBucket):
            gb = self.grid()
            fd = self.flux_disc()
            mat = []
            sub_ind = []
            for g, _ in self.grid():
                ind = fd.solver.dof_of_grid(gb, g)
                A = self.lhs[ind, :][:, ind]
                mat.append(A)
                sub_ind.append(ind)
            return mat, sub_ind
        else:
            return [self.lhs], [np.arange(self.grid().num_cells)]


# ------------------------------------------------------------------------------#


class DualEllipticModel(EllipticModel):
    def __init__(self, gb, data=None, keyword="flow", **kwargs):
        EllipticModel.__init__(self, gb, data, keyword, **kwargs)

        self.discharge_name = str()
        self.projected_discharge_name = str()

    def _set_discretization(self):
        discr = pp.MVEM(self.keyword)
        source = pp.DualSource(self.keyword)

        if self.is_GridBucket:
            key = self.keyword

            for _, d in self._gb:
                # Choose discretization and define the solver
                d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1, "faces": 1}}
                d[pp.DISCRETIZATION] = {key: {"flux": discr, "source": source}}

            for e, d in self._gb.edges():
                g_slave, g_master = self._gb.nodes_of_edge(e)
                d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
                d[pp.COUPLING_DISCRETIZATION] = {
                    "flux": {
                        g_slave: (key, "flux"),
                        g_master: (key, "flux"),
                        e: (key, pp.RobinCoupling(key, discr)),
                    }
                }
            return pp.Assembler()
        else:
            return discr

    def solve(self):
        """ Discretize and solve linear system by a direct solver.

        The saddle point structure of the dual discretization implies that
        other block solvers are needed. TODO.

        """
        logger.error("Solve elliptic model")
        # Discretize
        tic = time.time()
        logger.warning("Discretize")
        self.lhs, self.rhs = self.reassemble()
        logger.warning("Done. Elapsed time " + str(time.time() - tic))

        # Solve
        tic = time.time()
        ls = pp.numerics.linalg.linsolve.Factory()
        logger.warning("Solve linear system using direct solver")
        self.x = ls.direct(self.lhs, self.rhs)
        np.set_printoptions(linewidth=3000)
        logger.warning("Done. Elapsed time " + str(time.time() - tic))

        return self.x

    def pressure(self, pressure_name="pressure"):
        self.pressure_name = pressure_name
        if self.is_GridBucket:
            for g, d in self._gb:
                discr = d[pp.DISCRETIZATION][self.keyword]["flux"]
                d[self.pressure_name] = discr.extract_pressure(g, d[self.x_name])
        else:
            pressure = self._discr.extract_pressure(self._gb, self.x)
            self._data[self.pressure_name] = pressure

    def discharge(self, discharge_name="discharge"):
        self.discharge_name = discharge_name
        if self.is_GridBucket:
            for g, d in self._gb:
                discr = d[pp.DISCRETIZATION][self.keyword]["flux"]
                d[self.discharge_name] = discr.extract_flux(g, d[self.x_name])

            # for e, d in self._gb.edges():
            #    g_h = self._gb.nodes_of_edge(e)[1]
            #    d[discharge_name] = self._gb.node_props(g_h, discharge_name)

        else:
            discharge = self._discr.extract_flux(self._gb, self.x)
            self._data[self.discharge_name] = discharge

    def project_discharge(self, projected_discharge_name="P0u"):
        if self.discharge_name is str():
            self.discharge()
        self.projected_discharge_name = projected_discharge_name
        if self.is_GridBucket:
            self._flux_disc.project_u(
                self._gb, self.discharge_name, self.projected_discharge_name
            )
        else:
            discharge = self._data[self.discharge_name]
            projected_discharge = self._discr.project_flux(
                self._gb, discharge, self._data
            )
            self._data[self.projected_discharge_name] = projected_discharge


# ------------------------------------------------------------------------------#


class EllipticDataAssigner:
    """
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
    keyword: (string): defaults to 'flow'

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
             returns: (tensor.SecondOrderTensor) Permeabillity tensor
        source(): defaults to 0
             returns: (ndarray) The source and sinks

    Utility functions:
        grid(): returns: the grid

    """

    def __init__(self, g, data, keyword="flow"):
        self._g = g
        self._data = data

        self.keyword = keyword
        self._set_data()

    def bc(self):
        return pp.BoundaryCondition(self.grid())

    def bc_val(self):
        return np.zeros(self.grid().num_faces)

    def porosity(self):
        """Returns apperture of each cell. If None is returned, default
        Parameter class value is used"""
        return None

    def aperture(self):
        """Returns apperture of each cell. If None is returned, default
        Parameter class value is used"""
        return None

    def permeability(self):
        kxx = np.ones(self.grid().num_cells)
        return pp.SecondOrderTensor(self.grid().dim, kxx)

    def source(self):
        return np.zeros(self.grid().num_cells)

    def data(self):
        return self._data

    def grid(self):
        return self._g

    def _set_data(self):
        if pp.PARAMETERS not in self._data:
            self._data[pp.PARAMETERS] = pp.Parameters(self.grid(), [self.keyword], [{}])
        self._data[pp.PARAMETERS].update_dictionaries([self.keyword], [{}])
        parameter_dictionary = self._data[pp.PARAMETERS][self.keyword]

        parameter_dictionary["second_order_tensor"] = self.permeability()
        parameter_dictionary["bc"] = self.bc()
        parameter_dictionary["bc_values"] = self.bc_val()
        parameter_dictionary["source"] = self.source()

        if self.porosity() is not None:
            parameter_dictionary["porosity"] = self.porosity()
        if self.aperture() is not None:
            parameter_dictionary["aperture"] = self.aperture()
