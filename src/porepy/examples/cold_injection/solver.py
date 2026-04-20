"""Contains the global solver used in this example.

References:

    `NTRDC <https://doi.org/10.1016/j.advwatres.2022.104285>`_

"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, TypeAlias, TypedDict, cast

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sla
from numpy.typing import NDArray
from scipy.linalg import lstsq

import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle

logger = logging.getLogger(__name__)

CFLEModel: TypeAlias = cfle.EnthalpyBasedCFLETemplate | cfle.EnthalpyBasedCFFLETemplate


def condest(A: sps.csr_matrix) -> float:
    # Get LU decomposition
    decomposition = sla.splu(A)

    # Define matrix-vector and reverse matrix-vector products for the inverse
    def matvec(rhs):
        return decomposition.solve(rhs, trans="N")

    def rmatvec(rhs):
        return decomposition.solve(rhs, trans="H")

    # Create a linear operator for the matrix inverse
    op = sla.LinearOperator(A.shape, matvec=matvec, rmatvec=rmatvec)

    # Compute 1-norm of the matrix and its inverse
    nrm_ori = sla.onenormest(A)
    nrm_inv = sla.onenormest(op)

    return float(nrm_ori * nrm_inv)


class CFSolverParams(TypedDict):
    """Parameters for the compositional flow solver. They are used additionally to the
    parameters of the default Newton solver."""

    logp_clip: tuple[float, float] | None
    """Clipping logarithmic pressure update, if 
    ``model.params["use_logp_nonlinear_rpc"] == True``."""
    newton_chop: float | None
    """Global chop for raw Newton update."""
    appleyard_chop: float | None
    """Chop for saturation and phase fraction variables. If None, chop is deactivated.
    Otherwise it must be a number between 0 and 1."""

    atol_objective: float
    """Absolute tolerance for objective function. If its 2-norm falls below this value,
    line search or trust region methods abort truncation of iterates considering the
    solution to be close enough for Newton to safely converge."""

    atol_inc: float
    """Absolute tolerance for increment. Values below this are treated as noise."""

    do_armijo_line_search: bool
    """Activate Armijo line search."""
    armijo_line_search_weight: float
    """Initial Armijo step size."""
    armijo_line_search_incline: float
    """Incline parameter for Armijo line search (kappa in the Armijo condition)."""
    armijo_line_search_max_iterations: int
    """Maximal number of line search iterations."""
    armijo_stop_after_residual_reaches: float
    """Stop line search if residual 2-norm reaches this value."""
    armijo_start_after_residual_reaches: float
    """Start line search if residual 2-norm reaches this value."""
    armijo_least_squares_form: bool
    """Whether to use the least squares form of the Armijo condition, i.e.,
    ``||F(x+rho*dx)||^2 <= (1 - 2*kappa*rho)*||F(x)||^2``, instead of the more common
    form ``F(x+rho*dx) <= F(x) + kappa*rho*F'(x)*dx``.
    """

    do_anderson_acceleration: bool
    """Activate Anderson acceleration."""
    anderson_acceleration_depth: int
    """Depth of Anderson acceleration."""
    anderson_acceleration_constrained: bool
    """Adds the constraint that the coefficients of the Anderson acceleration must
    sum up to 1 explicitly in the least squares problem."""
    anderson_acceleration_regularization_parameter: float
    """Regularization parameter for Anderson acceleration least squares problem."""
    anderson_acceleration_relaxation_parameter: float
    """Relaxation parameter for Anderson acceleration. If larger than 0, the Anderson
    acceleration step is relaxed towards the previous iterate."""
    anderson_start_after_residual_reaches: float
    """Start Anderson acceleration only after the residual 2-norm reaches this value."""
    anderson_stop_after_residual_reaches: float
    """Stop Anderson acceleration after the residual 2-norm reaches this value."""

    do_ntrdc: bool
    """Activate Newton Trust-Region Dogleg-Cauchy."""
    ntrdc_delta_0: float
    """Initial trust region radius relative to norm of initial iterate."""
    ntrdc_delta_tol: float
    """Criterion for trust-region radius to exit iteration and return nans
    (time step failed to find direction of sufficient decrease)."""
    ntrdc_eta_1: float
    """Criterion for approximate objective improvement to break the iterations."""
    ntrdc_eta_2: float
    """Criterion for approximate objective improvement to decrease trust-region
    radius."""
    ntrdc_eta_3: float
    """Criterion for approximate objective improvement to increase trust-region
    radius."""
    ntrdc_t_1: float
    """Scaling factor for decreasing trust-region radius."""
    ntrdc_t_2: float
    """Scaling factor for increasing trust-region radius."""
    ntrdc_scale_with_inf: bool
    """If true, scales the system with the inf-norm of the update locally in the trust-
    region iterations."""
    ntrdc_return_nan: bool
    """Returns nans when trust-region-radius limit is reached to trigger time step
    cutting. Otherwise it returns the tiny step as is."""


class ModAndersonAcceleration:
    """Anderson acceleration as described by Walker and Ni in doi:10.2307/23074353."""

    def __init__(
        self,
        params: Optional[dict] = None,
    ) -> None:
        if params is None:
            params = {}
        self._depth = int(params.get("anderson_acceleration_depth", 3))
        self._constrain_acceleration: bool = bool(
            params.get("anderson_acceleration_constrained", False)
        )
        self._reg_param: float = float(
            params.get("anderson_acceleration_regularization_parameter", 0.0)
        )
        self._beta: float = float(
            params.get("anderson_acceleration_relaxation_parameter", 0.0)
        )

        assert 0 <= self._reg_param < 1
        assert 0 <= self._beta < 1

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
            dimension = gk.size
            assert dimension == fk.size
            self._Fk: np.ndarray = np.zeros((dimension, self._depth))
            self._Gk: np.ndarray = np.zeros((dimension, self._depth))
            self._xk: np.ndarray = np.zeros((dimension, self._depth))
            self._fkm1: np.ndarray = np.zeros(dimension)
            self._gkm1: np.ndarray = np.zeros(dimension)

        mk = min(iteration, self._depth)

        # Apply actual acceleration (not in the first iteration).
        if mk > 0:
            # Build matrices of changes.
            col = (iteration - 1) % self._depth
            self._Fk[:, col] = fk  # - self._fkm1
            self._Gk[:, col] = gk  # - self._gkm1

            # Solve least squares problem.
            A = self._Fk[:, 0:mk]
            b = fk
            if self._constrain_acceleration:
                A = np.vstack((A, np.ones((1, mk))))
                b = np.concatenate((b, np.ones(1)))

            direct_solve = False

            if self._reg_param > 0:
                b = A.T @ b
                A = A.T @ A + self._reg_param * np.eye(mk)
                direct_solve = np.linalg.matrix_rank(A) >= mk

            if direct_solve:
                gamma_k = np.linalg.solve(A, b)
            else:
                gamma_k = lstsq(A, b)[0]

            # Do the mixing
            # x_k_plus_1 = gk - np.dot(self._Gk[:, 0:mk], gamma_k)
            x_k_plus_1 = np.dot(self._Gk[:, 0:mk], gamma_k)
            if self._beta > 0:
                x_k_plus_1 *= self._beta
                x_k_plus_1 += (1 - self._beta) * np.dot(self._xk[:, 0:mk], gamma_k)
        else:
            x_k_plus_1 = gk

        self._xk[:, :-1] = self._xk[:, 1:]
        self._xk[:, -1] = x_k_plus_1
        # Store values for next iteration.
        # self._fkm1 = fk.copy()
        # self._gkm1 = gk.copy()

        return x_k_plus_1


class CFLESolver(pp.NewtonSolver):
    """Numerical methods on top of the raw Newton solver for compositional flow
    problems.

    See :class:`CFSolverParams` for the parameters that can be used to control the
    behavior of the solver.
    """

    def __init__(self, params: dict | None = None):
        if params is None:
            params = {}
        assert isinstance(params, dict)
        default_params: dict[Any, Any] = {}
        default_params.update(self.default_params())
        default_params.update(params)
        super().__init__(params)

        self._anderson: ModAndersonAcceleration = ModAndersonAcceleration(params)

        self._J: sps.csr_matrix
        """Current Jacobian matrix."""
        self._F: np.ndarray
        """Current residual vector."""
        self._grad_pot: np.ndarray
        """Current gradient of the potential (i.e., Jac.T times residual)."""
        self._F_norm: float | np.floating
        """Current residual 2-norm."""
        self._xk: np.ndarray
        """Current iterate."""
        self._xk_norm: float | np.floating
        """Current iterate 2-norm."""
        self._pot: float | np.floating
        """Current objective function value."""

        self._delta: float | np.floating
        """Initial trust-region radius set in the first iteration."""
        self._delta_max: float | np.floating
        """Maximal trust-region radius, set in the first iteration."""
        self._delta_min: float | np.floating
        """Minimal trust-region radius, set in the first iteration. The algorithm is
        aborted if the trust-region radius falls below this value."""

        self._state_changes: dict[pp.Grid, NDArray[np.bool_]] = {}
        """Boolean indicators per cell for each grid whether a state change in phase
        configuration is detected or not. Empty dictionary indiciates no state
        change."""

        self.params = cast(CFSolverParams, default_params)

    @staticmethod
    def default_params() -> CFSolverParams:
        return CFSolverParams(
            logp_clip=None,
            newton_chop=None,
            appleyard_chop=None,
            atol_objective=1e-8,
            atol_inc=1e-10,
            do_armijo_line_search=True,
            armijo_line_search_weight=0.9,
            armijo_line_search_incline=1e-4,
            armijo_line_search_max_iterations=20,
            armijo_stop_after_residual_reaches=0.0,
            armijo_start_after_residual_reaches=np.inf,
            armijo_least_squares_form=False,
            do_anderson_acceleration=False,
            anderson_acceleration_depth=3,
            anderson_acceleration_constrained=False,
            anderson_acceleration_regularization_parameter=0.0,
            anderson_acceleration_relaxation_parameter=0.0,
            anderson_start_after_residual_reaches=np.inf,
            anderson_stop_after_residual_reaches=0.0,
            do_ntrdc=False,
            ntrdc_delta_0=0.2,
            ntrdc_delta_tol=1e-10,
            ntrdc_eta_1=1e-3,
            ntrdc_eta_2=0.25,
            ntrdc_eta_3=0.75,
            ntrdc_t_1=0.25,
            ntrdc_t_2=2.0,
            ntrdc_scale_with_inf=True,
            ntrdc_return_nan=True,
        )

    def _plot(
        self, model: CFLEModel, dx: np.ndarray | None = None, suffix: str = ""
    ) -> None:
        """Plotting function for debugging purposes."""

        is_update = False
        if dx is not None:
            is_update = True
            # NOTE Rescale separately, because wrong values are stored in state.
            col_scales = model._column_scales()
            use_logp = model._uses_logp()
            dp: np.ndarray | None = None
            dofs_p: np.ndarray | None = None

            vec = dx.copy()

            if use_logp and isinstance(
                model, pp.fluid_mass_balance.VariablesSinglePhaseFlow
            ):
                # If the logp nonlinear RPC is used, the nonlinear increment is a logp
                # and needs to be transformed back to a pressure increment before being
                # added to the solution.
                dofs_p = model.equation_system.dofs_of([model.pressure_variable])
                p_k = self._xk[dofs_p]
                p_k1p = p_k * np.exp(vec[dofs_p])
                dp = p_k1p - p_k

            if isinstance(col_scales, np.ndarray):
                vec *= col_scales
            if dp is not None and dofs_p is not None:
                vec[dofs_p] = dp
        else:
            vec = self._xk

        model.plot_from_vec(vec, "pressure", is_update, suffix)  # type:ignore
        model.plot_from_vec(vec, "fluid_specific_volume", is_update, suffix)  # type:ignore
        model.plot_from_vec(vec, "s_G", is_update, suffix)  # type:ignore
        # model.plot_from_vec(vec, "y_G", is_update, suffix)

    def iteration(self, model: CFLEModel) -> np.ndarray:  # type:ignore[override]
        """An iteration consists of performing the Newton step and obtaining the step
        size from the line search."""

        # Raw Newton update.
        dx = super().iteration(model)  # type:ignore[arg-type]
        dxn_raw = np.linalg.norm(dx)

        # Catch initial bad iterates. Goal is to trigger nan-divergence criterion early
        # enough if any method failes.
        if np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
            return np.full_like(dx, np.nan)  # Trigger NanDivergence criterion.
        diverged: bool | np.bool | np.bool_ = False

        do_armijo = self.params["do_armijo_line_search"]
        do_anderson = self.params["do_anderson_acceleration"]
        do_ntrdc = self.params["do_ntrdc"]
        least_squares = self.params["armijo_least_squares_form"]

        self._xk = model.equation_system.get_variable_values(iterate_index=0)
        col_scales = model._column_scales()
        # Calcualate norm of current iterate in scaled space.
        pk = None
        if model._uses_logp():
            pk = model.equation_system.get_variable_values(
                [model.pressure_variable], iterate_index=0
            )
        xk = self._xk.copy()
        if isinstance(col_scales, np.ndarray):
            xk /= col_scales
            if pk is not None:
                dofsp = model.equation_system.dofs_of([model.pressure_variable])
                xk[dofsp] = np.log(pk)

        self._xk_norm = np.linalg.norm(xk)

        if do_ntrdc or (do_armijo and not least_squares):
            A, b = model.equation_system.assemble(evaluate_jacobian=True)
            if isinstance(col_scales, np.ndarray):
                assert A.shape[1] == col_scales.size
                if not sps.isspmatrix_csr(A):
                    A = sps.csr_matrix(A)
                A.data *= col_scales[A.indices]
            self._J = A
            self._F = -b
            self._F_norm = np.linalg.norm(self._F)
            self._pot = self._F_norm**2 * 0.5
            self._grad_pot = self._J.transpose() @ self._F
        else:
            self._F = -model.equation_system.assemble(evaluate_jacobian=False)
            self._F_norm = np.linalg.norm(self._F)
            self._pot = self._F_norm**2 * 0.5

        # if model.time_manager.time_index == 28:
        #     self._plot(model, None)
        #     self._plot(model, dx, "raw")
        dx = self.apply_chops(model, dx)
        # if model.time_manager.time_index == 28:
        #     self._plot(model, dx, "after chops")

        if do_ntrdc:
            dx = self.ntrdc(model, dx)
            diverged = np.any(np.isnan(dx))
            # if model.time_manager.time_index == 28:
            #     self._plot(model, dx, "after TR")

        if do_armijo and not diverged:
            dx *= self.armijo_line_search(model, dx)
            diverged = np.any(np.isnan(dx))

        # Manuel dereferencing for clean-up
        self._F = np.empty(0)
        self._grad_pot = np.empty(0)
        self._J = sps.csr_matrix(0)

        if do_anderson and not diverged:
            dx = self.anderson_acceleration(model, dx)
            diverged = np.any(np.isnan(dx))

        # NOTE: Reverse the changes of the state which happened during the procedures
        # here.
        model.equation_system.set_variable_values(self._xk, iterate_index=0)
        if diverged:
            return np.full_like(dx, np.nan)
        else:
            logger.debug(
                f"Delta increment norm: {dxn_raw:.4e} -> ({np.linalg.norm(dx):.4e})"
            )
            return dx

    def objective_function(
        self, model: CFLEModel, trial_increment: np.ndarray
    ) -> float:
        """Objective function for the residual depending on a state vector."""
        xk1p = self._xk + model._rescale_increment(trial_increment)
        model.equation_system.set_variable_values(xk1p, iterate_index=0)
        model.update_derived_quantities()
        res = model.equation_system.assemble(evaluate_jacobian=False)
        return float(np.dot(res, res)) * 0.5

    def apply_chops(self, model: CFLEModel, dx: np.ndarray) -> np.ndarray:
        if self.params["newton_chop"] is not None:
            dx *= self.params["newton_chop"]
        if self.params["appleyard_chop"] is not None:
            dx = self.appleyard_chop(model, dx)
        if model._uses_logp():
            c = self.params["logp_clip"]
            if c is not None:
                assert c[0] < 0, "Log-p lower clip must be smaller than 0."
                assert c[1] > 0, "Log-p upper clip must be greater than 1."
                dofs = model.equation_system.dofs_of([model.pressure_variable])
                dx[dofs] = np.clip(dx[dofs], c[0], c[1])
        return dx

    def appleyard_chop(self, model: CFLEModel, dx: np.ndarray) -> np.ndarray:
        """ "Simple chopping of updates for saturatons such that their absolute values
        is not larger than a defined value ``params['appleyard_chop']``.

        By default, no chop is applied.

        """
        if hasattr(model, "saturation_variables"):
            chop = cast(float, self.params["appleyard_chop"])
            dofs = model.equation_system.dofs_of(model.saturation_variables)
            ds = dx[dofs]

            idx = np.abs(ds) > chop
            if np.any(idx):
                logger.info(f"Appleyard chop on saturations in {int(idx.sum())} cells.")
                ds[idx] = chop * np.sign(ds[idx])
                dx[dofs] = ds

            if model.phase_fraction_variables:  # type:ignore[attr-defined]
                dofs = model.equation_system.dofs_of(model.phase_fraction_variables)  # type:ignore[attr-defined]
                dy = dx[dofs]
                idx = np.abs(dy) > chop
                if np.any(idx):
                    logger.info(
                        f"Appleyard chop on phase fractions in {int(idx.sum())} cells."
                    )
                    dy[idx] = chop * np.sign(dy[idx])
                    dx[dofs] = dy

        return dx

    def armijo_line_search(self, model: CFLEModel, dx: np.ndarray) -> float:
        """Performs the Armijo line search."""
        F_upper = self.params["armijo_start_after_residual_reaches"]
        F_lower = self.params["armijo_stop_after_residual_reaches"]
        if not (F_lower <= self._F_norm <= F_upper):
            return 1.0

        rho_0 = self.params["armijo_line_search_weight"]
        kappa = self.params["armijo_line_search_incline"]
        N = self.params["armijo_line_search_max_iterations"]
        least_squares = self.params["armijo_least_squares_form"]
        atol = self.params["atol_objective"]

        tol_pot = max(np.finfo(np.float64).eps, atol**2 / 2)

        pot_0 = self._pot

        lin_decrease = 0.0
        if not least_squares:
            lin_decrease = float(np.dot(self._grad_pot, dx))

        if pot_0 <= tol_pot:
            logger.info(f"Armijo line search potential below cap. Returning 1.")
            return 1.0
        rho = rho_0

        start = time.time()
        for i in range(N):
            rho = rho_0**i

            try:
                pot_i = self.objective_function(model, rho * dx)
            except:
                # In case this was the last evaluation and it failed, return nan to flag
                # divergence. Avoid errors downstream.
                rho = np.nan
                continue

            if least_squares:
                break_condition = pot_i <= (1 - 2 * kappa * rho) * pot_0
            else:
                break_condition = pot_i <= pot_0 + kappa * rho * lin_decrease

            if break_condition or pot_i <= tol_pot:
                break

        model.nonlinear_solver_statistics.log_custom_data(
            append=True, armijo_clocktime=time.time() - start
        )
        model.nonlinear_solver_statistics.log_custom_data(
            append=True, armijo_iterations=i
        )

        if np.isnan(rho):
            logger.warning("Armijo line search failed. Returning nan.")
        else:
            logger.info(f"Armijo line search determined weight: {rho:.4f} ({i})")

        return rho

    def ntrdc(self, model: CFLEModel, dx_n: np.ndarray) -> np.ndarray:
        """Newton Trust-Region Dogleg-Cauchy method."""

        eps = np.finfo(np.float64).eps

        # Extract parameters.
        delta_tol = self.params["ntrdc_delta_tol"]
        delta_0 = self.params["ntrdc_delta_0"]
        eta_1 = self.params["ntrdc_eta_1"]
        eta_2 = self.params["ntrdc_eta_2"]
        eta_3 = self.params["ntrdc_eta_3"]
        t_1 = self.params["ntrdc_t_1"]
        t_2 = self.params["ntrdc_t_2"]
        atol = self.params["atol_objective"]
        atol_inc = self.params["atol_inc"]
        do_scale = self.params["ntrdc_scale_with_inf"]

        atol_pot = max(eps, atol**2 / 2)
        rtol_inc = np.sqrt(eps)
        eps_noise = 100.0 * eps

        pot_0 = self._pot
        dx_ns = dx_n.copy()
        g = self._grad_pot
        B = sps.csr_matrix(self._J.transpose() @ self._J)

        scales = np.ones(model.equation_system.num_dofs())
        if do_scale:

            def apply_scale(name: str) -> None:
                idx = model.equation_system.dofs_of([name])
                s = np.linalg.norm(dx_n[idx], ord=np.inf)
                scales[idx] = 1.0 if s <= atol_inc else s

            # Apply scaling for variables with physical dimensions.
            if isinstance(model, pp.fluid_mass_balance.VariablesSinglePhaseFlow):
                apply_scale(model.pressure_variable)
            if isinstance(model, pp.fluid_mass_balance.FluidVolumeVariable):
                apply_scale(model.fluid_volume_variable)
            if isinstance(model, pp.energy_balance.VariablesEnergyBalance):
                apply_scale(model.temperature_variable)
            if isinstance(model, pp.energy_balance.EnthalpyVariable):
                apply_scale(model.enthalpy_variable)

            dx_ns /= scales
            g = self._grad_pot * scales
            # Column and row scaling (J D)T (J D) = D (JT J) D
            B.data *= scales[B.indices]
            B.data *= scales[np.repeat(np.arange(B.shape[0]), np.diff(B.indptr))]

        # Initialize trust-region radius based on initial values.
        # Update minimal trust-region radius based on current solution.
        # Then fetch current delta from self._delta.
        # NOTE: Iteration counter is increased before Newton iteration. Likewise the
        # time step index.
        if self.iteration_index == 1:
            rel_tol = max(1.0, self._xk_norm)
            self._delta_min = delta_tol * rel_tol
            if model.time_manager.time_index == 1:
                self._delta = delta_0 * rel_tol
                self._delta_max = rel_tol
                logger.info(
                    f"NTRDC initial trust-region radius: {self._delta:.4e}. "
                    # f"maximal trust-region radius: {self._delta_max:.4e}. "
                )
            logger.info(
                "NTRDC minimal trust-region radius for time step "
                f"{model.time_manager.time_index}: {self._delta_min:.4e}."
            )

        delta = self._delta
        dx_ks = dx_ns
        gBg = np.dot(g, B @ g)
        g_norm = np.linalg.norm(g)
        g_gBg = g_norm**2 / gBg
        k = 0
        start = time.time()

        while True:
            if np.linalg.norm(dx_ns) <= delta:
                dx_ks = dx_ns
            else:
                alpha = min(delta / g_norm, g_gBg)
                dx_c = -alpha * g
                if np.linalg.norm(dx_c) >= delta:
                    dx_ks = dx_c
                else:
                    dx_ = dx_ns - dx_c
                    a = np.dot(dx_, dx_)
                    b = 2.0 * np.dot(dx_c, dx_)
                    c = np.dot(dx_c, dx_c) - delta**2
                    d = np.sqrt(b**2 - 4 * a * c)
                    n = 1.0 / (2 * a)
                    tau = max((-b + d) * n, (-b - d) * n)
                    dx_ks = dx_c + tau * dx_

            dx_k = dx_ks * scales if do_scale else dx_ks
            pot_k = self.objective_function(model, dx_k)
            m_k = pot_0 + np.dot(g, dx_ks) + 0.5 * np.dot(dx_ks, B @ dx_ks)

            if (
                pot_0 <= atol_pot
                and np.linalg.norm(dx_k) <= rtol_inc * (1 + self._xk_norm)
                and pot_k <= pot_0 + eps_noise * (1 + pot_0)
            ):
                logger.info(
                    "NTRDC stopping criterion reached: potential and step below "
                    "noise level. Returning current step."
                )
                break

            # Approximate improvement.
            dpot = pot_0 - pot_k
            dm = pot_0 - m_k
            rho = dpot / dm
            if dpot < 0 and dm < 0:
                rho *= -1.0

            # Adaption of trust-region radius.
            # If quadratic model is a bad approximation (rho << 1), decrease radius.
            if rho < eta_2:
                delta *= t_1
            # If quadratic model is a good approximation (rho ~/> 1), increase radius.
            elif rho > eta_3:
                delta *= t_2
                # delta = min(self._delta_max, t_2 * delta)

            if rho > eta_1:  # Success condition.
                logger.info(f"NTRDC accepted step with improvement {rho:.3e}.")
                break

            k += 1
            if delta < self._delta_min:  # Failure condition.
                msg = f"NTRDC reached minimal trust-region radius {delta:.4e}."
                if self.params["ntrdc_return_nan"]:
                    msg += " Returning nans."
                    dx_k = np.full_like(dx_n, np.nan)
                logger.warning(msg)
                break

        # NOTE: Store delta for next iterations and time steps. Ensure it is not smaller
        # than minimum.
        model.nonlinear_solver_statistics.log_custom_data(
            append=True, ntrdc_clocktime=time.time() - start
        )
        logger.info(f"NTRDC change in delta: {self._delta:.4e} -> {delta:.4e} ({k})")
        self._delta = max(delta, self._delta_min)
        model.nonlinear_solver_statistics.log_custom_data(
            append=True, ntrdc_iterations=k
        )
        return dx_k

    def anderson_acceleration(self, model: CFLEModel, dx: np.ndarray) -> np.ndarray:
        """Apply the anderson acceleration."""

        F_upper = self.params["anderson_start_after_residual_reaches"]
        F_lower = self.params["anderson_stop_after_residual_reaches"]
        if F_lower <= self._F_norm <= F_upper:
            logger.info("Applying Anderson acceleration.")

            xk = self._xk.copy()
            pk = None
            dofsp = None
            uses_logp = model._uses_logp()
            if uses_logp:
                dofsp = model.equation_system.dofs_of([model.pressure_variable])
                pk = np.log(xk[dofsp])
            col_scales = model._column_scales()
            if isinstance(col_scales, np.ndarray):
                xk /= col_scales
            if isinstance(pk, np.ndarray):
                xk[dofsp] = pk

            xk1p = self._anderson.apply(xk + dx, dx, self.iteration_index)
            return xk1p - xk
        else:
            return dx

    def get_equilibrated_trial_step(
        self, mode: CFLEModel, dx: np.ndarray
    ) -> np.ndarray:
        """Returns a modified update step ``dx_mod`` such that ``x_k + dx_mod`` is a
        solution to the equilibrium system, i.e. the sub-residual of the equilibrium
        equations is zero."""

        raise NotImplementedError("")
