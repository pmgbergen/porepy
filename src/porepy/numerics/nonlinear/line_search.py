"""Module for line search algorithms for nonlinear solvers.

The algorithm is described in the manuscript at https://arxiv.org/abs/2407.01184.

Main active classes, combined in the NonlinearSolver class:
- LineSearchNewtonSolver - extends NewtonSolver with line search and implements a basic
    line search based on the residual norm.
- SplineInterpolationLineSearch - implements a line search based on spline interpolation.
- ConstraintLineSearch - implements a line search based on constraint functions for contact
    mechanics.

The functionality is invoked by specifying the solver in the model parameters, e.g.:

    ```python
    model.params["nonlinear_solver"] = NonlinearSolver
    ```

The solver can be further customized by specifying parameters in the model parameters. It also
requires implementation of the constraint functions in the model as methods called
"opening_indicator" and "sliding_indicator", see model_setup.ContactIndicators.

"""

import logging
from typing import Any, Callable, Optional, Sequence, cast

import numpy as np
import scipy

import porepy as pp

logger = logging.getLogger(__name__)


class LineSearchNewtonSolver(pp.NewtonSolver):
    """Class for relaxing a nonlinear iteration update using a line search.

    This class extends the iteration method to include a line search and implements
    a line search based on the residual norm.

    """

    def iteration(self, model) -> np.ndarray:
        """A single nonlinear iteration.

        Add line search to the iteration method. First, call the super method to
        compute a nonlinear iteration update. Then, perform a line search along
        that update direction.

        Parameters:
            model: The simulation model.

        Returns:
            The solution update.

        """
        dx = super().iteration(model)
        relaxation_vector = self.nonlinear_line_search(model, dx)

        # Update the solution
        sol = relaxation_vector * dx
        model._current_update = sol
        return sol

    def nonlinear_line_search(self, model, dx: np.ndarray) -> np.ndarray:
        """Perform a line search along the Newton step.

        Parameters:
            model: The model.
            dx: The nonlinear iteration update.

        Returns:
            The step length vector, one for each degree of freedom. The relaxed update
            is obtained by multiplying the nonlinear update step by the relaxation vector.

        """
        return self.residual_line_search(model, dx)

    def residual_line_search(self, model, dx: np.ndarray) -> np.ndarray:
        """Compute the relaxation factors for the current iteration based on the residual.

        Parameters:
            model: The model.
            dx: The nonlinear iteration update.

        Returns:
            The step length vector, one value for each degree of freedom.

        """
        if not model.params.get("Global_line_search", False):
            return np.ones_like(dx)

        def objective_function(weight):
            """Objective function for the trial update relaxed by
            the current weight.

            Parameters:
                weight: The relaxation factor.

            Returns:
                The objective function value. Specified as the norm of the residual
                in this implementation, may be overridden.

            """
            return self.residual_objective_function(model, dx, weight)

        tol = 1e-1
        f_0 = objective_function(0)
        f_1 = objective_function(1)
        if f_1 < model.params["nl_convergence_tol_res"] or (f_1 < f_0 / 1e4):
            # The objective function is zero at the full nonlinear step. This
            # means that the nonlinear step is a minimum of the objective
            # function. We can use the update without any relaxation.
            return np.ones_like(dx)

        def f_terminate(vals):
            """Terminate the recursion if the objective function is increasing."""
            return vals[-1] > vals[-2]

        alpha = self.recursive_weight_from_sampling(
            0, 1, f_terminate, objective_function, f_0, f_1, 3, tol
        )
        # safeguard against zero weights
        return np.maximum(alpha, tol / 10) * np.ones_like(dx)

    def recursive_weight_from_sampling(
        self,
        a: float,
        b: float,
        condition_function: Callable[[Sequence], bool],
        function: Callable,
        f_a: Optional[float] = None,
        f_b: Optional[float] = None,
        num_steps: int = 5,
        step_size_tolerance: float = 1e-1,
    ) -> float:
        """Recursive function for finding a weight satisfying a condition.

        The function is based on sampling the function in the interval [a, b] and
        recursively narrowing down the interval until the interval is smaller than the
        tolerance and the condition is satisfied. It returns the smallest tested value
        not satisfying the condition.

        Parameters:
            a, b: The interval.
            condition_function: The condition function. It takes a sequence of line search
                evaluations as argument and returns True if the condition is satisfied,
                indicating that the recursion should be terminated.
            function: The function to be tested. Returns a scalar or vector, must be compatible
                with the condition function's parameter.
            f_a: The value of the function at a. If not given, it is computed.
            num_steps: The number of sampling points in the interval [a, b].
            step_size_tolerance: The tolerance for the step size. If the step size is smaller
                than this, the recursion is terminated.

        Returns:
            The smallest tested value not satisfying the condition.

        """
        if f_a is None:
            f_a = function(a)
        terminate_condition = False
        sampling_points = np.linspace(a, b, num_steps)
        step_size = (b - a) / (num_steps - 1)
        f_vals = [f_a]
        for c in sampling_points[1:]:
            if np.isclose(c, b) and f_b is not None:
                f_c = f_b
            else:
                f_c = function(c)
            f_vals.append(f_c)
            terminate_condition = condition_function(f_vals)
            if not terminate_condition:
                f_a = f_c
                a = c
            else:
                # There is a local minimum in the narrowed-down interval [a, c]
                if step_size > step_size_tolerance:
                    # Find it to better precision
                    return self.recursive_weight_from_sampling(
                        a,
                        c,
                        condition_function,
                        function,
                        f_a=f_a,
                        num_steps=num_steps,
                        step_size_tolerance=step_size_tolerance,
                    )
                else:
                    # We're happy with the precision, return the minimum
                    return c

        # We went through the whole interval without finding a local minimum.
        # Thus, we assume that the minimum lies in [c, b]. If we have reached
        # the tolerance, we return b. Otherwise, we search in [c, b].
        if step_size < step_size_tolerance:
            return b
        else:
            return self.recursive_weight_from_sampling(
                sampling_points[-2],
                b,
                condition_function,
                function,
                f_a=f_vals[-2],
                num_steps=num_steps,
                step_size_tolerance=step_size_tolerance,
            )

    def residual_objective_function(
        self, model, dx: np.ndarray, weight: float
    ) -> np.floating[Any]:
        """Compute the objective function for the current iteration.

        The objective function is the norm of the residual.

        Parameters:
            model: The model.
            dx: The nonlinear iteration update.
            weight: The relaxation factor.

        Returns:
            The objective function value.

        """
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        residual = model.equation_system.assemble(
            state=x_0 + weight * dx, evaluate_jacobian=False
        )
        return np.linalg.norm(residual)


class SplineInterpolationLineSearch:
    """Class for computing relaxation factors based on spline interpolation.

    This class could be seen as a tool for the technical step of performing a
    line search. It also specifies that this choice should be used for the
    constraint weights.

    """

    def compute_constraint_violation_weights(
        self,
        model,
        solution_update: np.ndarray,
        constraint_function: pp.ad.Operator,
        crossing_inds: np.ndarray,
        f_0: np.ndarray,
        max_weight: float = 1.0,
        interval_target_size=1e-3,
    ) -> float:
        """Specify that the constraint weights are computed based on spline interpolation.

        Parameters:
            model: The model.
            solution_update: The nonlinear iteration update.
            constraint_function: The constraint function. Returns a scalar or vector and
                changes sign at the zero crossing.
            crossing_inds: The indices where the constraint function has changed sign.
            f_0: The value of the constraint function at the initial point.
            max_weight: The maximum value for the constraint weights.
            interval_target_size: The target size of the interval for the root finding. The
                recursion is terminated when the interval is smaller than this.

        Returns:
            The step length weight.

        """
        if not np.any(crossing_inds):
            return 1.0
        # If the indicator has changed, we need to compute the relaxation
        # factors. We do this by recursively narrowing down the interval until
        # the interval is smaller than the tolerance using a spline
        # interpolation.
        a, b = 0.0, max_weight
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        f_0 = f_0[crossing_inds]
        # Compute the constraint function at the new weights. Specify that return value
        # is an array.
        f_1_vals = cast(
            np.ndarray,
            constraint_function.value(model.equation_system, x_0 + solution_update * b),
        )
        f_1 = f_1_vals[crossing_inds]

        def f(x):
            return constraint_function.value(
                model.equation_system, x_0 + solution_update * x
            )[crossing_inds]

        alpha, a, b = self.recursive_spline_interpolation(
            a,
            b,
            f,
            f_0,
            f_1,
            interval_target_size=interval_target_size,
            method="roots",
        )
        return alpha

    def recursive_spline_interpolation(
        self,
        a: float,
        b: float,
        function: Callable,
        f_a: Optional[float | np.ndarray] = None,
        f_b: Optional[float | np.ndarray] = None,
        num_pts: int = 5,
        interval_target_size: float = 1e-1,
        method="minimize_scalar",
    ) -> tuple[float, float, float]:
        """Recursive function for finding a weight satisfying a condition.

        Returns both the optimum/root (see method) and the minimal interval in
        which it is assumed to lie.

        Parameters:
            a, b: The interval.
            function: The function to be tested. Returns a scalar or vector defined on
                the interval [a, b].
            f_a: The value of the function at a. If not given, it is computed.
            f_b: The value of the function at b. If not given, it is computed.
            num_pts: The number of sampling points in the interval [a, b].
            step_size_tolerance: The tolerance for the step size. If the step size is smaller
                than this, the recursion is terminated.
            method: The method for finding the minimum of the spline. Either "minimize_scalar"
                or "roots".

        Returns:
            Tuple containing:
                The minimum of the function in the interval [a, b].
                The lower bound of the interval in which the minimum is assumed to lie.
                The upper bound of the interval in which the minimum is assumed to lie.
        """
        counter = 0
        while b - a > interval_target_size or counter < 1:
            alpha, x, y = self.optimum_from_spline(
                function,
                a,
                b,
                f_a,
                f_b,
                num_pts=num_pts,
                method=method,
            )
            x = np.linspace(a, b, num_pts)
            # Find the indices on either side of alpha
            ind = np.searchsorted(x, alpha)
            if ind == 0:
                b = x[1]
                f_b = y[1]
            elif ind == num_pts:
                a = x[ind - 1]
                f_a = y[ind - 1]
            else:
                a = x[ind - 1]
                b = x[ind]
                f_a = y[ind - 1]
                f_b = y[ind]
            counter += 1

        return alpha, a, b

    def optimum_from_spline(
        self,
        f: Callable,
        a: float,
        b: float,
        f_a=None,
        f_b=None,
        num_pts: int = 5,
        method="minimize_scalar",
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the minimum/root of the spline interpolation of the function.

        Parameters:
            f: The function to be interpolated.
            a, b: The interval.
            f_a: The value of the function at a. If not given, it is computed.
            f_b: The value of the function at b. If not given, it is computed.
            num_pts: The number of sampling points in the interval [a, b].
            method: The method for finding the minimum of the spline. Either "minimize_scalar"
                or "roots".

        Returns:
            Tuple containing:
                The minimum of the function in the interval [a, b].
                The x values of the spline interpolation.
                The y values of the spline interpolation.

        """

        x = np.linspace(a, b, num_pts)
        y_list = []

        for pt in x:
            if f_a is not None and np.isclose(pt, a):
                f_pt = f_a
            elif f_b is not None and np.isclose(pt, b):
                f_pt = f_b
            else:
                f_pt = f(pt)
            if np.any(np.isnan(f_pt)):
                # If we get overflow, truncate the x vector
                x = x[: np.where(x == pt)[0][0]]
                break
            # Collect function values, scalar or vector
            y_list.append(f_pt)
        if isinstance(y_list[0], np.ndarray):
            y = np.vstack(y_list)
        else:
            y = np.array(y_list)

        def compute_and_postprocess_single(poly, a: float, b: float) -> float:
            if method == "minimize_scalar":
                minimum = scipy.optimize.minimize_scalar(
                    lambda s: poly(s), bounds=[a, b], method="bounded"
                )
                min_x = minimum.x
            elif method == "roots":
                min_x = poly.roots()
            if min_x.size == 0:
                return b
            else:
                # Find smallest root inside [a, b]
                min_x = min_x[(min_x >= a) & (min_x <= b)]
                if min_x.size == 0:
                    return b
                else:
                    return np.min(min_x)

        # Find minima of the spline

        if isinstance(y_list[0], np.ndarray):
            all_minima = []
            for i in range(y.shape[1]):
                poly = scipy.interpolate.PchipInterpolator(x, y[:, i])
                this_min = compute_and_postprocess_single(poly, a, b)
                all_minima.append(this_min)
            alpha = np.min(all_minima)
        else:
            poly = scipy.interpolate.PchipInterpolator(x, y)
            alpha = compute_and_postprocess_single(poly, a, b)

        return alpha, x, y


class ConstraintLineSearch:
    """Class for computing relaxation weights based on constraint functions
    for contact mechanics.

    The contract with the Model class is that the constraint functions are
    defined in the model as Operator returning methods called "opening_indicator"
    and "sliding_indicator".

    """

    compute_constraint_violation_weights: Callable[..., float]
    """Method for computing the constraint weights.

    This method specifies the algorithm for computing the constraint weights by a line
    search method. Current option is based on spline interpolation.

    """

    residual_line_search: Callable[[Any, np.ndarray], np.ndarray]
    """Method for computing the relaxation factors for the current iteration
    based on the residual."""

    @property
    def use_fracture_minimum(self):
        return True

    @property
    def min_line_search_weight(self):
        """Minimum weight for the relaxation weights."""
        return 1e-12

    def nonlinear_line_search(self, model, dx: np.ndarray) -> np.ndarray:
        """Perform a line search along the Newton step.

        First, call super method using the global residual as the objective function.
        Then, compute the constraint weights.

        Parameters:
            model: The model.
            dx: The Newton step.

        Returns:
            The step length vector, one for each degree of freedom.

        """
        residual_weight = self.residual_line_search(model, dx)
        if model.params.get("Local_line_search", False):
            return self.constraint_line_search(model, dx, residual_weight.min())
        else:
            return residual_weight

    def constraint_line_search(
        self, model, dx: np.ndarray, max_weight: float = 1
    ) -> np.ndarray:
        """Perform line search along the Newton step for the constraints.

        This method defines the constraint weights for the contact mechanics and
        how they are combined to a global weight. For more advanced combinations,
        this method can be overridden or the stored weights can be accessed elsewhere.

        Parameters:
            model: The model.
            dx: The Newton step.
            max_weight: The maximum weight for the constraint weights.

        Returns:
            The step length vector, one for each degree of freedom.

        """

        subdomains = model.mdg.subdomains(dim=model.nd - 1)

        sd_weights = {}
        global_weight = max_weight
        for sd in subdomains:
            sd_list = [sd]
            # Compute the relaxation factors for the normal and tangential
            # component of the contact mechanics.
            normal_weights = self.constraint_weights(
                model,
                dx,
                model.opening_indicator(sd_list),
                max_weight=max_weight,
            )
            tangential_weights = self.constraint_weights(
                model,
                dx,
                model.sliding_indicator(sd_list),
                max_weight=np.minimum(max_weight, normal_weights).min(),
            )
            # For each cell, take minimum of tangential and normal weights
            combined_weights = np.vstack((tangential_weights, normal_weights))
            min_weights = np.min(combined_weights, axis=0)
            # Store the weights for the subdomain. This facilitates more advanced
            # combinations of the constraint weights, e.g. using local weights for
            # each cell.
            sd_weights[sd] = min_weights
            model.mdg.subdomain_data(sd).update({"constraint_weights": min_weights})
            # Combine the weights for all subdomains to a global minimum weight.
            global_weight = np.minimum(global_weight, min_weights.min())
        # Return minimum of all weights.
        weight = np.ones_like(dx) * global_weight
        return weight

    def constraint_weights(
        self,
        model,
        solution_update: np.ndarray,
        constraint_function: pp.ad.Operator,
        max_weight: float = 1,
    ) -> np.ndarray:
        """Compute weights for a given constraint.

        This method specifies the algorithm for computing the constraint weights:
        - Identify the indices where the constraint function has changed sign.
        - Compute the relaxation factors for these indices, allowing transition beyond
            zero by a tolerance given by the constraint_violation_tolerance parameter.
        - Reassess the constraint function at the new weights and repeat the process if
            too many indices are still transitioning.

        Parameters:
            model: The model.
            solution_update: The Newton step.
            constraint_function: The constraint function.
            max_weight: The maximum weight for the constraint weights.

        Returns:
            The step length vector, one for each degree of freedom of the constraint.

        """
        # If the sign of the function defining the regions has not changed, we
        # use unitary relaxation factors
        x_0 = model.equation_system.get_variable_values(iterate_index=0)
        violation_tol = model.params.get("constraint_violation_tolerance", 3e-1)
        relative_cell_tol = model.params.get(
            "relative_constraint_transition_tolerance", 2e-1
        )
        # Compute the constraint function at the maximum weights. Specify that return value
        # is an array to avoid type errors.
        f_1 = cast(
            np.ndarray,
            constraint_function.value(
                model.equation_system, x_0 + max_weight * solution_update
            ),
        )
        weight = max_weight
        weights = max_weight * np.ones(f_1.shape)
        f_0 = constraint_function.value(model.equation_system, x_0)
        active_inds = np.ones(f_1.shape, dtype=bool)
        for i in range(10):
            # Only consider dofs where the constraint violation has changed sign
            violation = violation_tol * np.sign(f_1)
            f = constraint_function - pp.wrap_as_dense_ad_array(violation)
            # Absolute tolerance should be safe, as constraints are assumed to be
            # scaled to approximately 1.
            roundoff = 1e-8
            inds = np.logical_and(np.abs(f_1) > violation_tol, f_0 * f_1 < -roundoff)
            if i > 0:  # Ensure at least one iteration.
                if sum(active_inds) < max(1, relative_cell_tol * active_inds.size):
                    # Only a few indices are active, and the set of active indices
                    # does not contain any new indices. We can stop the iteration.
                    break

                else:
                    logger.info(
                        f"Relaxation factor {weight} is too large for {sum(active_inds)}"
                        + " indices. Reducing constraint violation tolerance."
                    )

            f_0_v = f_0 - violation
            # Compute the relaxation factors for the indices where the constraint
            # violation has changed sign. The weight corresponds to the point at
            # which the constraint function crosses zero.
            crossing_weight = self.compute_constraint_violation_weights(
                model,
                solution_update,
                f,
                inds,
                f_0_v,
                max_weight=max_weight,
                interval_target_size=1e-3,
            )
            weight = np.clip(
                crossing_weight, a_max=max_weight, a_min=self.min_line_search_weight
            )
            weights[inds] = weight

            if not self.use_fracture_minimum:
                break  # Experimental.
            # Check how many indices are active for current weight
            f_1 = cast(
                np.ndarray,
                constraint_function.value(
                    model.equation_system, x_0 + weight * solution_update
                ),
            )
            active_inds = np.logical_and(
                np.abs(f_1) > violation_tol, f_0 * f_1 < -roundoff
            )
            if i == 9:
                logger.info(
                    "Maximum number of iterations reached. "
                    + "Returning current weights."
                )
            max_weight = weight
            violation_tol = violation_tol / 2

        return weights


# class NonlinearSolver(
#     ConstraintLineSearch,
#     SplineInterpolationLineSearch,
#     LineSearchNewtonSolver,
# ):
#     pass
