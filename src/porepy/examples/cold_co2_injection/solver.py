"""Contains the global solver used in this example."""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import lstsq

import porepy as pp
import porepy.models.compositional_flow as cf

logger = logging.getLogger(__name__)


class AndersonAcceleration:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(
        self,
        dimension: int,
        depth: int,
        constrain_acceleration: bool = False,
        regularization_parameter: float = 0.0,
    ) -> None:
        self._dimension = int(dimension)
        self._depth = int(depth)
        self._constrain_acceleration: bool = bool(constrain_acceleration)
        self._reg_param: float = float(regularization_parameter)

        # Initialize arrays for iterates.
        self.reset()
        self._fkm1: np.ndarray = np.zeros(self._dimension)
        self._gkm1: np.ndarray = np.zeros(self._dimension)

    def reset(self) -> None:
        self._Fk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in increments
        self._Gk: np.ndarray = np.zeros(
            (self._dimension, self._depth)
        )  # changes in fixed point applications

    def apply(self, gk: np.ndarray, fk: np.ndarray, iteration: int) -> np.ndarray:
        """Apply Anderson acceleration.

        Parameters:
            gk: application of some fixed point iteration onto approximation xk, i.e.,
                g(xk).
            fk: residual g(xk) - xk; in general some increment.
            iteration: current iteration count.

        Returns:
            Modified application of fixed point approximation after acceleration, i.e.,
            the new iterate xk+1.

        """

        if iteration == 0:
            self.reset()

        mk = min(iteration, self._depth)

        # Apply actual acceleration (not in the first iteration).
        if mk > 0:
            # Build matrices of changes.
            col = (iteration - 1) % self._depth
            self._Fk[:, col] = fk - self._fkm1
            self._Gk[:, col] = gk - self._gkm1

            # Solve least squares problem.
            A = self._Fk[:, 0:mk]
            b = fk
            if self._constrain_acceleration:
                A = np.vstack((A, np.ones((1, self._depth))))
                b = np.concatenate((b, np.ones(1)))

            direct_solve = False

            if self._reg_param > 0:
                b = A.T @ b
                A = A.T @ A + self._reg_param * np.eye(A.shape[1])
                direct_solve = np.linalg.matrix_rank(A) >= A.shape[1]

            if direct_solve:
                gamma_k = np.linalg.solve(A, b)
            else:
                gamma_k = lstsq(A, b)[0]

            # Do the mixing
            x_k_plus_1 = gk - np.dot(self._Gk[:, 0:mk], gamma_k)
        else:
            x_k_plus_1 = gk

        # Store values for next iteration.
        self._fkm1 = fk.copy()
        self._gkm1 = gk.copy()

        return x_k_plus_1


class NewtonArmijoAndersonSolver(pp.NewtonSolver, AndersonAcceleration):
    """Newton solver with Armijo line search.

    The residual objective function is tailored to models where phase properties are
    assumed to be surrogate factories and require an update before evaluating the
    objective function.

    """

    def __init__(self, params: dict | None = None):
        pp.NewtonSolver.__init__(self, params)
        if params is None:
            params = {}
        depth = int(params.get("anderson_acceleration_depth", 3))
        dimension = int(params["anderson_acceleration_dimension"])
        constrain = params.get("anderson_acceleration_constrained", False)
        reg_param = params.get("anderson_acceleration_regularization_parameter", 0.0)
        AndersonAcceleration.__init__(
            self,
            dimension,
            depth,
            constrain_acceleration=constrain,
            regularization_parameter=reg_param,
        )

    def iteration(self, model: pp.PorePyModel):
        """An iteration consists of performing the Newton step and obtaining the step
        size from the line search."""
        # dx = super().iteration(model)

        dx = pp.NewtonSolver.iteration(self, model)

        dx *= self.armijo_line_search(model, dx)
        dx = self.appleyard_chop(model, dx)

        if self.params.get("anderson_acceleration", False):
            iteration = model.nonlinear_solver_statistics.num_iteration
            x = model.equation_system.get_variable_values(iterate_index=0)
            x_temp = x + dx
            if not (np.any(np.isnan(x_temp)) or np.any(np.isinf(x_temp))):
                try:
                    xp1 = self.apply(x_temp, dx.copy(), iteration)
                    res = model.equation_system.assemble(evaluate_jacobian=False)
                    # TODO Wrong reference residual
                    res_norm = model.compute_residual_norm(res, res)  # type:ignore
                    if res_norm <= self.params.get(
                        "anderson_start_after_residual_reaches", np.inf
                    ) and res_norm >= self.params.get(
                        "anderson_stop_after_residual_reaches", 0.0
                    ):
                        dx = xp1 - x
                        print("Applying Anderson")
                    else:
                        print("Not applying anderson")
                except Exception:
                    logger.warning(
                        f"Resetting Anderson acceleration at"
                        f" T={model.time_manager.time}; i={iteration} due to failure."
                    )
                    self.reset()

        return dx

    def armijo_line_search(self, model: pp.PorePyModel, dx: np.ndarray) -> float:
        """Performs the Armijo line search."""
        res = model.equation_system.assemble(evaluate_jacobian=False)
        # TODO Wrong reference residual
        res_norm = model.compute_residual_norm(res, res)  # type:ignore
        if (
            not self.params.get("armijo_line_search", False)
            or (res_norm <= self.params.get("armijo_stop_after_residual_reaches", 0.0))
            or (
                res_norm
                >= self.params.get("armijo_start_after_residual_reaches", np.inf)
            )
        ):
            print("Not Applying Armijo")
            return 1.0
        else:
            print("applying Armijo")

        rho = float(self.params.get("armijo_line_search_weight", 0.9))
        kappa = float(self.params.get("armijo_line_search_incline", 0.4))
        N = int(self.params.get("armijo_line_search_max_iterations", 50))

        pot_0 = self.armijo_objective_function(model, dx, 0.0)
        rho_i = rho
        n = 0

        for i in range(N):
            n = i
            rho_i = rho**i

            pot_i = self.armijo_objective_function(model, dx, rho_i)
            if pot_i <= (1 - 2 * kappa * rho_i) * pot_0:
                break

        if hasattr(model.nonlinear_solver_statistics, "num_iteration_armijo"):
            model.nonlinear_solver_statistics.num_iteration_armijo += n  # type:ignore
        logger.info(f"Armijo line search determined weight: {rho_i} ({n})")
        return rho_i

    def armijo_objective_function(
        self, model: pp.PorePyModel, dx: np.ndarray, weight: float
    ) -> float:
        """The objective function to be minimized is the norm of the residual squared
        and divided by 2."""
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        state = x_0 + weight * dx
        # model.update_thermodynamic_properties_of_phases(state)
        cf.SolutionStrategyPhaseProperties.update_thermodynamic_properties_of_phases(
            model,  # type:ignore[arg-type]
            state,
        )
        residual = model.equation_system.assemble(state=state, evaluate_jacobian=False)
        return float(np.dot(residual, residual) / 2)

    def appleyard_chop(self, model: pp.PorePyModel, dx: np.ndarray) -> np.ndarray:
        """ "Simple chopping of updates for saturatons such that their absolute values
        is not larger than a defined value ``params['appleyard_chop']``.

        By default, no chop is applied.

        """

        m = self.params.get("appplyard_chop", None)
        if isinstance(m, float):
            assert 0 < m < 1, "Chopping limit for saturations must be strictly in (0,1)"
            if hasattr(model, "saturation_variables"):
                dofs = model.equation_system.dofs_of(model.saturation_variables)
                ds = dx[dofs]

                chop = np.abs(ds) > m
                if np.any(chop):
                    logger.info(f"Applying Appleyard chop in {int(chop.sum())} cells.")
                    ds[chop] = m * np.sign(ds[chop])
                    dx[dofs] = ds

                if model.phase_fraction_variables:
                    dofs = model.equation_system.dofs_of(model.phase_fraction_variables)
                    dy = dx[dofs]
                    chop = np.abs(dy) > m
                    if np.any(chop):
                        dy[chop] = m * np.sign(dy[chop])
                        dx[dofs] = dy

        return dx
