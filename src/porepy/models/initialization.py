"""Initialization strategies utilizing auxiliary simulations.

TODO:
* Once time stepping is updated. Make here use of provided structures.
* User control on hard coded parameters.
* Allow to reuse solvers etc.?

"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path

import porepy as pp
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceStatus,
    SimulationStatus,
)

logger = logging.getLogger(__name__)


class InitializationStrategy(pp.PorePyModel):
    def prepare_simulation(self) -> None:
        """Include initialization in the preparation of the simulation."""
        super().prepare_simulation()
        self.initialization()

    @abstractmethod
    def initialization(self) -> None:
        raise NotImplementedError


class QuasiStaticReferenceStateInitialization(InitializationStrategy):

    def initialization(self) -> None:
        """Run initialization with strategy-specific update placement."""
        self._run_initialization(update_reference_after_solve=True)

    def _run_initialization(self, update_reference_after_solve: bool) -> None:
        """Run initialization with strategy-specific update placement."""

        # Get initialization parameters.
        init_config = self.params.get("initialization", {})

        # Define nonlinear solver for initialization.
        solver_params = init_config.get("solver_params", {})
        solver = pp.NewtonSolver(solver_params)


        # Get initialization parameters
        use_export = init_config.get("use_export", True)
        convergence_tol = init_config.get("convergence_inc_atol", 1e-6)
        pseudo_time_step = init_config.get("pseudo_time_step", 1000 * pp.YEAR)

        # Define exporter and export initial state.
        exporter = None
        if use_export:
            folder = Path(self.params["folder_name"])
            folder_iterations = folder.parent / (folder.name + "_initialization")
            exporter = pp.Exporter(
                self.mdg,
                file_name=self.params["file_name"],
            folder_name=folder_iterations,
            length_scale=self.units.m,
            )
            exporter.write_vtu(
                self.data_to_export(),
                time_dependent=True,
                time_step=0,
            )

        # Artificial time control for quasi-static initialization.
        self.time_manager.dt = pseudo_time_step

        # Perform a pseudo time stepping to initialize the reference state.
        iteration = 0
        while True:
            # Advance iter.
            iteration += 1

            # Communicate dt to the model and update time-dependent arrays and
            # derived quantities.
            self.before_time_step()

            # Solve pseudo time step.
            self.initialize_nonlinear_solution()
            solver_status = solver.solve(self)

            # React to solver_status.
            if solver_status.is_successful():
                # Evaluate initialization status based on total increments.
                convergence_status = self.check_initialization_convergence(
                    convergence_tol
                )
                if convergence_status.is_converged():
                    initialization_status = SimulationStatus.SUCCESSFUL
                else:
                    initialization_status = SimulationStatus.IN_PROGRESS

                # Shift solution for next computation.
                self.update_time_step_solution()
                if update_reference_after_solve:
                    self.update_reference()

                # Update the time step magnitude if the dynamic scheme is used.
                if not self.time_manager.is_constant:
                    assert isinstance(
                        self.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
                    )  # For type checking, to ensure the method is available.
                    self.time_manager.compute_time_step(
                        iterations=self.nonlinear_solver_statistics.num_iterations
                    )

            elif solver_status.is_failed():
                if self.time_manager.is_constant:
                    initialization_status = SimulationStatus.STOPPED

                else:
                    try:
                        initialization_status = SimulationStatus.FAILED
                        self.model.time_manager.compute_time_step(
                            recompute_solution=True
                        )
                    except Exception as e:
                        logger.warning(str(e))
                        initialization_status = SimulationStatus.STOPPED
            elif solver_status.is_stopped():
                initialization_status = SimulationStatus.STOPPED

            else:
                raise ValueError("Unrecognized solver status.")

            # Export initialization iterates.
            if exporter is not None:
                exporter.write_vtu(
                    self.data_to_export(),
                    time_dependent=True,
                    time_step=iteration,
                )

            # Stop initialization.
            if (
                initialization_status.is_successful()
                or initialization_status.is_stopped()
            ):
                break

        # Revert exporter-internal counter (initiated in prepare_simulation) and
        # save updated initial data.
        # TODO: Revisit. Exporter has own counter which needs reset - prone to error.
        self.exporter._time_step_counter = 0
        self.save_data_time_step()

        # Reset time manager as possibly redefined during initialization.
        self.time_manager.dt = self.time_manager.dt_init

        logger.info("\033[92mInitialization completed.\033[0m")

    def check_initialization_convergence(self, tol: float) -> ConvergenceStatus:
        """Check convergence of the initialization state.

        Uses simple criterion based on the change in the state variables between
        the current and previous iteration.

        Parameters:
            tol: Tolerance for convergence.

        Returns:
            ConvergenceStatus: Enum indicating whether the initialization state is
                converged or not.

        """
        # Define a simple convergence criterion aiming for checking change in updates.
        criterion = pp.IncrementBasedAbsoluteCriterion(
            tol=tol,
            metric=pp.EuclideanMetric(),
        )

        # Define the increment to be the change of (all) states in time.
        state = self.equation_system.get_variable_values(iterate_index=0)
        prev_state = self.equation_system.get_variable_values(time_step_index=0)
        increment = state - prev_state

        # Check convergence based on increment
        convergence_status, _ = criterion.check(increment=increment)

        return convergence_status


class QuasiStaticPreviousStateInitialization(QuasiStaticReferenceStateInitialization):
    """Update the reference state at the beginning of the simulation."""

    def initialization(self) -> None:
        """Run initialization with strategy-specific update placement."""
        self._run_initialization(update_reference_after_solve=False)