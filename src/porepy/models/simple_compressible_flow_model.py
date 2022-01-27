"""This module contains a simple extension of the incompressible flow by
including a time derivative of the pressure and constant compressibility.
"""

import logging
import time
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class SimpleCompressibleFlow(pp.models.incompressible_flow_model.IncompressibleFlow):
    """ This class extends the Incompressible flow model by including a
    cummulative term expressed through pressure and a constant compressibility
    coefficient. For a full documenation refer to the parent class.

    The simulation starts at time t=0.

    Overwritten methods include:
        1. prepare_simulation:
        Starting time set to 0, end time and timestep size optional paramters
        2. _set_parameters: compressibility added

    New methods:
        1. _compressibility: constant compressibility per cell

    Attributes:
        end_time (float): Upper limit of considered time interval
        time_step (float): time step size
        time_index (int): number of time loops passed.
    """

    def __init__(
        self,
        end_time: Optional[float] = 10.,
        time_step: Optional[float] = [0.1],
        params: Optional[Dict] = None
    ) -> None:
        """ 
        Parameters:
            end_time (optional float): specifices end time for simulation.
                Simulation stops when first time value bigger than end_time is 
                reached.
            time_step (optional float ): Timestep size. Currently only
                uniform timestepping is supported.
        """
        super().__init__(params)

        # attributes
        # assure positivity and correct type and format of input
        self.end_time = abs(float(end_time))
        self.time_step = abs(float(time_step))
        self.time_index = 0

    # def run_simulation(self) -> None:
    #     """ Runs the explicit Euler scheme until endtime is reached.
    #     """
    #     if not self._is_set_up:
    #         raise RuntimeError("Simulation started before set-up was completed")
        
    #     t = 0.
    #     T = self._endtime

    #     while t < T:
    #         t =+ self._tstep_size

    def _set_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameters fields of the data dictionaries are updated for all
        subdomains and edges (of codimension 1).
        """
        super()._set_parameters()

        for g, d in self.gb:

            mass_weight = self._compressibility(g)

            pp.initialize_data(
                g,
                d,
                self.parameter_key,
                {
                    "mass_weight": mass_weight
                },
            )

    def _compressibility(self, g: pp.Grid) -> np.ndarray:
        """Unitary compressibility.

        Units: Pa^(-1)
        """
        return np.ones(g.num_cells)

    def _source(self, g: pp.Grid) -> np.ndarray:
        """Zero source term. TODO introduce source

        Units: m^3 / s
        """
        return np.zeros(g.num_cells)

    def _initial_condition(self) -> None:
        """Set initial guess for the variables."""
        initial_values = np.zeros(self.dof_manager.num_dofs())
        self.dof_manager.distribute_variable(initial_values)
        self.dof_manager.distribute_variable(initial_values, to_iterate=True)

    def _assign_discretizations(self) -> None:
        """Define equations through discretizations.

        Uses the Mpfa discretization of the parent class for the elliptic part
        and same boundary conditions coupling on internal boundaries.

        Implements an Explicit Euler discretization in time.
        """

        super()._assign_discretizations()
        gb = self.gb
        grid_list = [g for g, _ in gb.nodes()]


        # Ad representation of discretizations
        mass_mat = pp.MassMatrix(self.parameter_key)
        mass_mat.discretize(gb)

        # Ad variables
        p = self._eq_manager.merge_variables([(g, self.variable) for g in grid_list])

        # Update equations
        subdomain_flow_eq = self._eq_manager.equations["subdomain_flow"]
        interface_flow_eq = self._eq_manager.equations["interface_flow"]

        mass = (
            mass_mat * (p - p.previous_timestep()) / self.time_step
        )


        subdomain_flow_eq += mass
        interface_flow_eq += mass

        # dict update not necessary if retrieved by reference
        self._eq_manager.equations.update(
            {
                "subdomain_flow": subdomain_flow_eq,
                "interface_flow": interface_flow_eq,
            }
        )

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """Use a direct solver for the linear system."""
        A, b = self._eq_manager.assemble()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min"
            + f" {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )
        tic = time.time()
        x = sps.linalg.spsolve(A, b)
        logger.info("Solved linear system in {} seconds".format(time.time() - tic))
        return x

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._use_ad:
            solution = self.dof_manager.assemble_variable(from_iterate=True)

        self.assembler.distribute_variable(solution)
        self.convergence_status = True
        self._export()

    ## Methods required by AbstractModel but irrelevant for static problems:
    def before_newton_loop(self):
        self._nonlinear_iteration = 0

    def after_simulation(self):
        pass
