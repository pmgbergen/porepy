"""This module contains the `IterativeSolverMixin` class, which provides the capabilitiy
of using iterative linear solvers to a PorePy model.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import time
from typing import Callable, TypedDict

import numpy as np
import porepy as pp
import scipy.sparse as sps
from porepy.viz.solver_statistics import SolverStatistics

from porepy.numerics.linear_solver.block_linear_system import (
    BlockLinearSystem,
    LinearSystemIndexer,
    concatenate_dof_indices,
)
from porepy.numerics.linear_solver.dof_manager import DofManager

# from pp_solvers.options_parsers import initialize_petsc_ksp
from porepy.numerics.linear_solver.linear_solver import LinearSolverConfiguration

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LinearSolverParams(TypedDict, total=False):
    pass


class IterativeSolverMixin(pp.PorePyModel):
    def solve_linear_system(self) -> np.ndarray:
        rhs = self.bmat.rhs
        if np.any(np.isnan(rhs) | np.isinf(rhs)):
            # This should never be the case, as this situation should cut off by the
            # nonlinear convergence criterion from the earliear nonlinear iteration. We
            # keep this safeguard until the iterative solver is in a more mature state.
            raise ValueError("RHS contains NaN or Inf values")

        t0 = time()
        try:
            solver = initialize_petsc_ksp(
                block_linear_system=self.bmat,
                dof_manager=self._dof_manager,
                petsc_ksp_pc_configuration=self._petsc_ksp_pc_configuration,
                user_options=solver_options,
            )
        except Exception:
            logger.exception(
                "Failed to build a PETSc linear solver based on the given linear "
                "system.",
                solver_options,
            )
            return np.full_like(rhs, np.nan), -9999
        elapsed = time() - t0
        self.nonlinear_solver_statistics.linsolve_construction_time.append(elapsed)
        logger.info("Linear solver constructed in %.2f seconds.", elapsed)

        # Project the right hand side to the local block matrix ordering, as was done
        # for the block matrix during assembly. We need to do this on the reordered rhs
        # vector (with contact eqs reordered).
        t0 = time()
        x_loc = solver.solve(rhs)
        elapsed = time() - t0
        self.linear_solver_statistics.linsolve_solve_time.append(elapsed)
        num_it = len(solver.get_residuals())
        logger.info(
            "Linear system solved in %.2f seconds with %d iterations.",
            elapsed,
            num_it,
        )

        info: PETScKspConvergedReason = solver.ksp.getConvergedReason()
        if info <= 0:
            logger.warning(
                f"Linear solver did not converge. Reason: %d. "
                "Check the solver options and the problem setup. "
                "See detailed description of PETSc error codes: "
                "https://petsc.org/release/manualpages/KSP/KSPConvergedReason/",
                info,
            )
        # Transform the solution back to the global (PorePy) ordering.
        for transformation in reversed(self._transformations):
            x_loc = transformation.transform_solution(x_loc)

        _, proj_col = self._dof_manager.build_projection()
        x = np.zeros_like(x_loc)
        x[concatenate_dof_indices(proj_col)] = x_loc

        # x = self.bmat.permute_right_vector_to_original(x_loc)  # YZ: This fails 02.06

        self.linear_solver_statistics.petsc_converged_reason.append(info)
        self.linear_solver_statistics.num_krylov_iters.append(num_it)
        if self.linear_solver_params().get("delete_matrices", True):
            del self.bmat

        return np.atleast_1d(x), info

    def assemble_linear_system(self):
        super().assemble_linear_system()  # type: ignore[misc]

        dof_manager = self._dof_manager
        # Get the linear system from the equation system.

        # TODO: Replace this with a different type of plugin
        mat, rhs = self.linear_system
        assert mat.getformat() == "csr"

        # Creating the indices of DoFs for the BlockLinearSystem class.
        linear_system = BlockLinearSystem(
            mat=mat,
            rhs=rhs,
            indexer=LinearSystemIndexer(
                dofs_row=dof_manager.eq_dofs(),
                dofs_col=dof_manager.var_dofs(),
                group_names_row=dof_manager.equation_names(),
                group_names_col=dof_manager.variable_names(),
            ),
        )

        # By calling [:], rearrange the blocks (and thereby the underlying matrix) to
        # match the ordering defined by the `dof_manager`.
        linear_system = linear_system[:]

        # Apply transformations to the linear systems before passing it to the solver.
        # TODO: Unit tests!
        for transformation in self._transformations:
            linear_system = transformation.transform_matrix_rhs(
                linear_system, dof_manager=dof_manager
            )

        self.bmat = linear_system

        # Delete the original linear system to save memory unless instructed not to.
        if self.linear_solver_params().get("delete_matrices", True):
            del self.linear_system

    def _initialize_linear_solver(self):
        # Set up preconditioner.

        # Add fields for the linear solver statistics to the nonlinear solver statistics
        # object.
        self.nonlinear_solver_statistics.linsolve_construction_time = []
        self.nonlinear_solver_statistics.linsolve_solve_time = []
        self.nonlinear_solver_statistics.petsc_converged_reason = []
        self.nonlinear_solver_statistics.num_krylov_iters = []

        linear_solver_params = self.linear_solver_params()
        configuration_factory = linear_solver_params.get("preconditioner_factory", None)
        if configuration_factory is None:
            configuration_factory = default_preconditioner_factory(self)

        configuration = configuration_factory()

        self._petsc_ksp_pc_configuration = configuration.solver
        self._transformations = configuration.transformations
        self._dof_manager = DofManager(model=self, groups=configuration.groups)

    def set_nonlinear_solver_statistics(self) -> None:
        """Override the method to set the solver statistics, so that we also get fields
        for the linear solver.

        This is certainly not the intended way of doing this, and it hacky, but the
        current PorePy implementation only caters to statistics objects being sent
        as part of the parameter class, which would require modification of all
        runscripts. Instead, we do it dirty for now.

        """
        super().set_nonlinear_solver_statistics()  # type: ignore[misc]
        # The name of the attribute is really not meaningful..
        self.linear_solver_statistics = LinearSolverStatistics()


def default_preconditioner_factory(
    model: pp.PorePyModel,
) -> Callable[[], LinearSolverConfiguration]:
    if isinstance(model, pp.SinglePhaseFlow):
        return mass_balance_factory
    if isinstance(model, pp.MomentumBalance):
        return momentum_balance_factory
    if isinstance(model, pp.MassAndEnergyBalance):
        return th_factory
    if isinstance(model, pp.Poromechanics):
        return hm_factory
    if isinstance(model, pp.Thermoporomechanics):
        return thm_factory
    raise ValueError(f"Unknown model:", type(model))
