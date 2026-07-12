"""Contains the global solver used in this example.

References:

    `NTRDC <https://doi.org/10.1016/j.advwatres.2022.104285>`_

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, TypedDict, cast, no_type_check

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sla
from numpy.typing import NDArray
from scipy.linalg import lstsq

import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.compositional import FlashSpec

if TYPE_CHECKING:
    from .config import CIModel

logger = logging.getLogger(__name__)


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

    pressure_clip: tuple[float, float] | None
    """Clipping pressure update fractionally with respect to last pressure value
    ``p_k``.
    
    The first value clips the pressure update from below to ``(c - 1)*p_k``, the second
    value clips it from above.
    
    Example:
        ``(0.8, 1.2)`` - pressure can vary at most 20% up and down from current iterate
        value.
    
    """

    volume_clip: tuple[float, float] | tuple[float, float, float] | None
    """Analogous to pressure clipping. Allows additionally to define a lower value
    for volume. Updates going below that will be clipped."""

    energy_clip: tuple[float, float] | None
    """Analogous to pressure clipping for energy-related variables."""

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

    in_physical_space: bool
    """If True, all additional methods after the linear solver are performed on
    the iterate and nonlinear increment in the physical space (as it is stored in the
    grid data dictionaries). If False, the scaling introduced by linear
    right-preconditioning is kept until the increment is stored."""

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
        self._xks_norm: float | np.floating
        """Current iterate 2-norm in scaled space."""
        self._pot: float | np.floating
        """Current objective function value."""

        self._delta: float | np.floating
        """Initial trust-region radius set in the first iteration."""
        self._delta_max: float | np.floating
        """Maximal trust-region radius, set in the first iteration."""
        self._delta_min: float | np.floating
        """Minimal trust-region radius, set in the first iteration. The algorithm is
        aborted if the trust-region radius falls below this value."""

        self.params = cast(CFSolverParams, default_params)

    @staticmethod
    def default_params() -> CFSolverParams:
        return CFSolverParams(
            pressure_clip=None,
            volume_clip=None,
            energy_clip=None,
            newton_chop=None,
            appleyard_chop=None,
            atol_objective=1e-7,
            atol_inc=1e-8,
            in_physical_space=False,
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

    @no_type_check
    def _plot(
        self,
        model: CIModel,
        dx: np.ndarray | None = None,
        suffix: str = "",
        log_pv: bool = False,
    ) -> None:
        """Plotting function for debugging purposes."""

        is_update = False
        if dx is not None:
            is_update = True
            vec = model._scale_back_state(dx, is_increment=True)
        else:
            vec = self._xk.copy()
            if log_pv:
                dofsp = model.equation_system.dofs_of(["pressure"])
                vec[dofsp] = np.log(vec[dofsp])
                # dofsv = model.equation_system.dofs_of(["specific_fluid_volume"])
                # vec[dofsv] = np.log(vec[dofsv])

        model.plot_from_vec(vec, "pressure", is_update, suffix)  # type:ignore
        model.plot_from_vec(vec, "specific_fluid_volume", is_update, suffix)  # type:ignore
        model.plot_from_vec(vec, "s_G", is_update, suffix)  # type:ignore
        # model.plot_from_vec(vec, "temperature", is_update, suffix)

    @no_type_check
    def _print_cond_p_block(self, model: CIModel):

        print("Total")
        print(f"Cond: {np.linalg.cond(self._J.todense()):.5e}")
        svd = np.linalg.svdvals(self._J.todense())
        print(f"SVs min max: {svd.min():.5e} {svd.max():.5e}")

        print("Whole p-block")
        sds = model.mdg.subdomains()
        p = model.pressure(sds)
        v = model.specific_fluid_volume(sds)
        dofs = model.equation_system.dofs_of([p, v])
        A0, _ = model.equation_system.assemble(
            True,
            [
                "mass_balance_equation",
                "production_pressure_constraint",
                "local_fluid_volume_constraint",
            ],
        )
        A1 = A0[:, dofs]
        print(f"Cond: {np.linalg.cond(A1.todense()):.5e}")
        svd = np.linalg.svdvals(A1.todense())
        print(f"SVs min max: {svd.min():.5e} {svd.max():.5e}")

        print("Rock")
        sds = model.mdg.subdomains(dim=2)
        p = model.pressure(sds)
        v = model.specific_fluid_volume(sds)
        dofs = model.equation_system.dofs_of([p, v])
        As = model.equation_system.evaluate(
            [
                model.mass_balance_equation(sds),
                model.local_fluid_volume_constraint(sds),
            ],
            derivative=True,
        )
        A0 = sps.vstack([A.jac for A in As])
        A1 = A0[:, dofs]
        print(f"Cond: {np.linalg.cond(A1.todense()):.5e}")
        svd = np.linalg.svdvals(A1.todense())
        print(f"SVs min max: {svd.min():.5e} {svd.max():.5e}")

        print("Fracture")
        sds = model.mdg.subdomains(dim=1)
        p = model.pressure(sds)
        v = model.specific_fluid_volume(sds)
        dofs = model.equation_system.dofs_of([p, v])
        As = model.equation_system.evaluate(
            [
                model.mass_balance_equation(sds),
                model.local_fluid_volume_constraint(sds),
            ],
            derivative=True,
        )
        A0 = sps.vstack([A.jac for A in As])
        A1 = A0[:, dofs]
        print(f"Cond: {np.linalg.cond(A1.todense()):.5e}")
        svd = np.linalg.svdvals(A1.todense())
        print(f"SVs min max: {svd.min():.5e} {svd.max():.5e}")

        print("Injector")
        sds = [sd for sd in model.mdg.subdomains(dim=0) if "injection_well" in sd.tags]
        p = model.pressure(sds)
        v = model.specific_fluid_volume(sds)
        dofs = model.equation_system.dofs_of([p, v])
        As = model.equation_system.evaluate(
            [
                model.mass_balance_equation(sds),
                model.local_fluid_volume_constraint(sds),
            ],
            derivative=True,
        )
        A0 = sps.vstack([A.jac for A in As])
        A1 = A0[:, dofs]
        print(f"Cond: {np.linalg.cond(A1.todense()):.5e}")
        svd = np.linalg.svdvals(A1.todense())
        print(f"SVs min max: {svd.min():.5e} {svd.max():.5e}")

        print("Producer")
        sds = [sd for sd in model.mdg.subdomains(dim=0) if "production_well" in sd.tags]
        p = model.pressure(sds)
        v = model.specific_fluid_volume(sds)
        dofs = model.equation_system.dofs_of([p, v])
        As = model.equation_system.evaluate(
            [
                model.pressure_constraint_at_production_wells(sds),
                model.local_fluid_volume_constraint(sds),
            ],
            derivative=True,
        )
        A0 = sps.vstack([A.jac for A in As])
        A1 = A0[:, dofs]
        print(f"Cond: {np.linalg.cond(A1.todense()):.5e}")
        svd = np.linalg.svdvals(A1.todense())
        print(f"SVs min max: {svd.min():.5e} {svd.max():.5e}")

    def iteration(self, model: CIModel) -> np.ndarray:  # type:ignore[override]
        """An iteration consists of performing the Newton step and obtaining the step
        size from the line search."""

        in_physical_space = self.params["in_physical_space"]
        # Raw Newton update.
        try:
            dx = super().iteration(model)  # type:ignore[arg-type]
        except:
            logger.warning(
                f"Assembly or lin-solve failure at t={model.time_manager.time} and "
                f"iter={self.iteration_index}"
            )
            dx = np.full((model.equation_system.num_dofs(),), np.nan)

        if in_physical_space:
            dx = model._scale_back_state(dx, is_increment=True)
            model.current_column_scales = None
        dxn_raw = np.linalg.norm(dx)

        # Catch initial bad iterates. Goal is to trigger nan-divergence criterion early
        # enough if any method failes.
        diverged: bool | np.bool_ = np.any(np.isnan(dx)) or np.any(np.isinf(dx))
        if diverged:
            return np.full_like(dx, np.nan)  # Trigger NanDivergence criterion.

        self._xk = model.equation_system.get_variable_values(iterate_index=0)
        self._xks_norm = np.linalg.norm(model._scale_state(self._xk))

        self.apply_chops(model, dx)
        change = self.identify_phase_change(model, dx)

        if change:
            try:
                dx = self.get_equilibrated_trial_step(model, dx)
            except:
                logger.warning(
                    f"Trial step equilibration failure at t={model.time_manager.time}"
                    f" and iter={self.iteration_index}"
                )
                dx = np.full((model.equation_system.num_dofs(),), np.nan)

        do_armijo = self.params["do_armijo_line_search"]
        do_anderson = self.params["do_anderson_acceleration"]
        do_ntrdc = self.params["do_ntrdc"]
        least_squares = self.params["armijo_least_squares_form"]

        if do_ntrdc or (do_armijo and not least_squares):
            A, b = model.equation_system.assemble(evaluate_jacobian=True)
            if not in_physical_space:
                A = model.params.get("linear_right_preconditioner", lambda x: x)([A])[0]  # type:ignore

            self._J = A
            self._F = -b
            self._F_norm = np.linalg.norm(self._F)
            self._pot = self._F_norm**2 * 0.5
            self._grad_pot = self._J.transpose() @ self._F
        else:
            self._F = -model.equation_system.assemble(evaluate_jacobian=False)
            self._F_norm = np.linalg.norm(self._F)
            self._pot = self._F_norm**2 * 0.5

        # self.resolve_md_flux_update(model, dx)
        diverged = np.any(np.isnan(dx))
        if do_ntrdc and not diverged:
            dx = self.ntrdc(model, dx)
            diverged = np.any(np.isnan(dx))

        if do_armijo and not diverged:
            dx = self.armijo_line_search(model, dx)
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
            return model._scale_back_state(dx, is_increment=True)

    def objective_function(self, model: CIModel, trial_increment: np.ndarray) -> float:
        """Objective function for the residual depending on a state vector."""
        xk1p = self._xk + model._scale_back_state(trial_increment, is_increment=True)
        model.equation_system.set_variable_values(xk1p, iterate_index=0)
        model.update_derived_quantities()
        res = model.equation_system.assemble(evaluate_jacobian=False)
        return float(np.dot(res, res)) * 0.5

    def apply_chops(self, model: CIModel, dx: np.ndarray) -> None:
        """Applies chops to the raw Newton update.

        This includes:

        - Global Newton chop
        - Appleyard chop
        - Pressure clip

        Parameters:
            model: A CFLE model.
            dx: Global nonlinear increment.

        """
        if self.params["newton_chop"] is not None:
            dx *= self.params["newton_chop"]
        if self.params["appleyard_chop"] is not None:
            self.appleyard_chop(model, dx)
        if self.params["pressure_clip"] is not None:
            self.pressure_clip(model, dx)
        if self.params["volume_clip"] is not None and model.has_fluid_volume_variable:
            self.relative_clip(
                model,
                dx,
                self.params["volume_clip"],
                model.specific_fluid_volume_variable,
            )
        if self.params["energy_clip"] is not None and isinstance(
            model, pp.energy_balance.VariablesEnergyBalance
        ):
            if model.has_fluid_internal_energy_variable:
                name = model.specific_fluid_internal_energy_variable
            elif model.has_fluid_enthalpy_variable:
                name = model.specific_fluid_enthalpy_variable
            else:
                name = model.temperature_variable

            self.relative_clip(
                model,
                dx,
                self.params["energy_clip"],
                name,
            )

    def identify_phase_change(
        self, model: CIModel, dx: np.ndarray
    ) -> list[tuple[pp.Grid, NDArray[np.bool_]]]:
        """Returns per grid a boolean array identifying cells where the active set
        of phases changed with ``self._xk + dx``."""
        xk = self._xk
        xk1 = xk + model._scale_back_state(dx, is_increment=True)

        tol_p = 1e-8  # Tolerance for flagging phase present
        tol_a = 1e-10  # Tolerance for flagging phase absent
        change: list[tuple[pp.Grid, NDArray[np.bool_]]] = []

        for grid in model.mdg.subdomains():
            idx = np.zeros(grid.num_cells, dtype=np.bool_)

            for phase in model.fluid.phases:
                yk = model.equation_system.evaluate(phase.fraction([grid]), state=xk)
                yk1 = model.equation_system.evaluate(phase.fraction([grid]), state=xk1)

                # Phase appearing
                idx = idx | ((yk1 > tol_p) & (yk < tol_p))
                # Phase disappearing
                idx = idx | ((yk1 < tol_p) & (yk > tol_p))

            if np.any(idx):
                logger.info(
                    f"Detected phase change in {idx.sum()} cells on grid {grid.id}"
                )
                change.append((grid, idx))

        return change

    def appleyard_chop(self, model: CIModel, dx: np.ndarray) -> None:
        """Simple chopping of updates for saturatons and phase fractions such that their
        absolute values is not larger than a defined value ``params['appleyard_chop']``.

        Parameters:
            model: A CFLE model.
            dx: Global nonlinear increment.

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

    def pressure_clip(self, model: CIModel, dx: np.ndarray) -> None:
        """Applies the pressure clip.

        Parameters:
            model: A CFLE model.
            dx: Global nonlinear increment.

        """
        c = cast(tuple[float, float], self.params["pressure_clip"])
        assert 0 < c[0] < 1, "Lower p-clip must be in (0, 1)."
        assert 1 < c[1], "Upper p-clip must be greater than 1."

        dofs = model.equation_system.dofs_of([model.pressure_variable])

        if model._uses_logp() and not self.params["in_physical_space"]:
            dx[dofs] = np.clip(dx[dofs], np.log(c[0]), np.log(c[1]))
        else:
            p_k = self._xk[dofs]
            dxs = model._scale_back_state(dx, is_increment=True)
            dxs[dofs] = np.clip(dxs[dofs], (c[0] - 1) * p_k, (c[1] - 1) * p_k)
            dx[dofs] = model._scale_state(dxs, is_increment=True)[dofs]

    def relative_clip(
        self, model: CIModel, dx: np.ndarray, c: tuple[float, ...], name: str
    ) -> None:
        """Applies the relative clip for a variable.

        Parameters:
            model: A CFLE model.
            dx: Global nonlinear increment.
            c: Tuple containing change limits (lower and upper) relative to current
                iterate state.
            name: Variable name.

        """
        assert 0 < c[0] < 1, "Lower relative clip must be in (0, 1)."
        assert 1 < c[1], "Upper relative clip must be greater than 1."

        dofs = model.equation_system.dofs_of([name])  # type:ignore

        v_k = self._xk[dofs]
        dxs = model._scale_back_state(dx, is_increment=True)
        dxs[dofs] = np.clip(dxs[dofs], (c[0] - 1) * v_k, (c[1] - 1) * v_k)
        if len(c) > 2:
            dxs[dofs] = np.maximum(dxs[dofs], c[2] - v_k)
        dx[dofs] = model._scale_state(dxs, is_increment=True)[dofs]

    def armijo_line_search(self, model: CIModel, dx: np.ndarray) -> np.ndarray:
        """Performs the Armijo line search."""
        F_upper = self.params["armijo_start_after_residual_reaches"]
        F_lower = self.params["armijo_stop_after_residual_reaches"]
        if not (F_lower <= self._F_norm <= F_upper):
            return dx

        rho_0 = self.params["armijo_line_search_weight"]
        kappa = self.params["armijo_line_search_incline"]
        N = self.params["armijo_line_search_max_iterations"]
        least_squares = self.params["armijo_least_squares_form"]
        atol = self.params["atol_objective"]

        pot_0 = self._pot

        lin_decrease = 0.0
        if not least_squares:
            lin_decrease = float(np.dot(self._grad_pot, dx))

        if pot_0 <= atol:
            logger.info(f"Armijo line search potential below cap. Returning 1.")
            return dx
        rho = rho_0
        dx_i = dx.copy()
        start = time.time()
        for i in range(N):
            rho = rho_0**i

            try:
                # dx_i = self.get_equilibrated_trial_step(model, rho * dx)
                pot_i = self.objective_function(model, dx_i)
            except:
                # In case this was the last evaluation and it failed, return nan to flag
                # divergence. Avoid errors downstream.
                rho = np.nan
                continue

            if least_squares:
                break_condition = pot_i <= (1 - 2 * kappa * rho) * pot_0
            else:
                break_condition = pot_i <= pot_0 + kappa * rho * lin_decrease

            if break_condition or pot_i <= atol:
                break

        model.nonlinear_solver_statistics.log_custom_data(
            append=True, armijo_clocktime=time.time() - start
        )
        model.nonlinear_solver_statistics.log_custom_data(
            append=True, armijo_iterations=i
        )

        if np.isnan(rho):
            logger.warning("Armijo line search failed. Returning nan.")
            dx_i = np.full_like(dx, np.nan)
        else:
            logger.info(f"Armijo line search determined weight: {rho:.4f} ({i})")

        return dx_i

    def ntrdc(self, model: CIModel, dx_n: np.ndarray) -> np.ndarray:
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
            xks = model._scale_state(self._xk)

            def apply_scale(name: str) -> None:
                idx = model.equation_system.dofs_of([name])
                s = np.linalg.norm(xks[idx], ord=np.inf)
                scales[idx] = 1.0 if s <= atol_inc else s

            # Apply scaling for variables with physical dimensions.
            if isinstance(model, pp.fluid_mass_balance.VariablesSinglePhaseFlow):
                apply_scale(model.pressure_variable)
                apply_scale(model.interface_darcy_flux_variable)
                apply_scale(model.well_flux_variable)
                if model.has_fluid_volume_variable:
                    apply_scale(model.specific_fluid_volume_variable)
            if isinstance(model, pp.energy_balance.VariablesEnergyBalance):
                apply_scale(model.temperature_variable)
                apply_scale(model.interface_enthalpy_flux_variable)
                apply_scale(model.well_enthalpy_flux_variable)
                apply_scale(model.interface_fourier_flux_variable)
                if model.has_fluid_enthalpy_variable:
                    apply_scale(model.specific_fluid_enthalpy_variable)
                if model.has_fluid_internal_energy_variable:
                    apply_scale(model.specific_fluid_internal_energy_variable)

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
            rel_tol = max(1.0, self._xks_norm)
            self._delta_min = delta_tol * rel_tol
            # if model.time_manager.time_index == 1:
            self._delta = delta_0 * rel_tol
            self._delta_max = delta_0 * rel_tol
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
            # dx_k = self.get_equilibrated_trial_step(model, dx_k)
            # dx_ks = dx_k / scales if do_scale else dx_k
            pot_k = self.objective_function(model, dx_k)
            m_k = pot_0 + np.dot(g, dx_ks) + 0.5 * np.dot(dx_ks, B @ dx_ks)

            # if (
            #     pot_0 <= atol_pot
            #     and np.linalg.norm(dx_k) <= rtol_inc * (1 + self._xks_norm)
            #     and pot_k <= pot_0 + eps_noise * (1 + pot_0)
            # ):
            if pot_k < atol:
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
            # If quadratic model is exact approximation (rho ~ 1), increase radius
            # aggressively.
            elif rho > 0.999:
                delta *= max(10.0, t_2)
                # delta = self._delta_max
            # If quadratic model is a good approximation, increase radius.
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

    def anderson_acceleration(self, model: CIModel, dx: np.ndarray) -> np.ndarray:
        """Apply the anderson acceleration."""

        F_upper = self.params["anderson_start_after_residual_reaches"]
        F_lower = self.params["anderson_stop_after_residual_reaches"]
        if F_lower <= self._F_norm <= F_upper:
            logger.info("Applying Anderson acceleration.")
            xk = model._scale_state(self._xk)
            xk1p = self._anderson.apply(xk + dx, dx, self.iteration_index)
            return xk1p - xk
        else:
            return dx

    def get_equilibrated_trial_step(self, model: CIModel, dx: np.ndarray) -> np.ndarray:
        """Returns a modified update step ``dx_mod`` such that ``x_k + dx_mod`` is a
        solution to the equilibrium system, i.e. the sub-residual of the equilibrium
        equations is zero."""

        dxs = model._scale_back_state(dx, is_increment=True)
        xk1p = self._xk + dxs
        xk1p_e = xk1p.copy()  # equilibrated state.

        for grid in model.mdg.subdomains():
            results, mask = model.local_equilibrium(
                grid,
                update_secondary_quantities=False,
                state=xk1p,
            )
            # if results.specification == FlashSpec.vu:
            #     mask[results.exitcode < 3] = True
            self.populate_state_with_flash_results(model, xk1p_e, results, [grid], mask)

        _, sec_vars = self.primary_secondary_thermodynamic_variable_names(model)
        sec_dofs = model.equation_system.dofs_of(sec_vars)
        dxs[sec_dofs] = xk1p_e[sec_dofs] - self._xk[sec_dofs]
        return model._scale_state(dxs, is_increment=True)

    def primary_secondary_thermodynamic_variable_names(
        self, model: CIModel
    ) -> tuple[list[str], list[str]]:
        """Returns the primary and secondary variables in the thermodynamic sense,
        based on how the local equilibrium is specified.

        Parameters:
            model: A CFLE model.

        Returns:
            The lists of thermodynamic primary and secondary variable names.
            The secondary variables are all unknowns of the thermodynamic subproblems.
            This includes all fractional quantities (except overall fractions),
            temperature if non-isothermal and local equilibrium is not
            temperature-based, and pressure if isochoric local equilibrium.

            The primary variables are the set-difference with all other variables.
            Most notably, they include also the flux variables.

        """
        sec_vars = (
            model.phase_fraction_variables
            + model.fraction_in_phase_variables
            + model.saturation_variables
        )

        flash_spec = [
            x
            for x in pp.compositional.get_equilibrium_specifications(model)
            if isinstance(x, FlashSpec)
        ][0]

        if (flash_spec not in [FlashSpec.pT, FlashSpec.vT]) and isinstance(
            model, pp.energy_balance.VariablesEnergyBalance
        ):
            sec_vars.append(model.temperature_variable)

        if flash_spec >= FlashSpec.vT and isinstance(
            model, pp.fluid_mass_balance.VariablesSinglePhaseFlow
        ):
            sec_vars.append(model.pressure_variable)

        prim_vars = list(
            set([var.name for var in model.equation_system.variables]).difference(
                sec_vars
            )
        )
        return prim_vars, sec_vars

    def populate_state_with_flash_results(
        self,
        model: CIModel,
        x: np.ndarray,
        results: cfle.FlashResults,
        subdomains: list[pp.Grid],
        cell_mask: NDArray[np.bool_],
    ) -> None:
        """"""

        def update(var: pp.ad.Operator, new_val: np.ndarray) -> None:
            assert isinstance(var, (pp.ad.MixedDimensionalVariable, pp.ad.Variable)), (
                f"Operator {var.name} not independent variable."
            )
            dofs = model.equation_system.dofs_of([var])
            x[dofs[cell_mask]] = new_val[cell_mask]
            # omit update where not sucessful.
            # TODO do not cancel, but damp update!!!!!!!!!!!!!!!!!!!! Or average over stencil

            cancel = ~cell_mask
            x[dofs[cancel]] = self._xk[dofs[cancel]]

        # Updating variables which are always unknowns in the equilibrium problem.
        for j, phase in enumerate(model.fluid.phases):
            if model.has_independent_fraction(phase):
                update(phase.fraction(subdomains), results.y[j])

            if model.has_independent_saturation(phase):
                update(phase.saturation(subdomains), results.sat[j])

            for i, comp in enumerate(phase.components):
                if model.has_independent_extended_fraction(comp, phase):
                    var = phase.extended_fraction_of[comp](subdomains)
                elif model.has_independent_partial_fraction(comp, phase):
                    var = phase.partial_fraction_of[comp](subdomains)
                else:
                    continue

                update(var, results.phases[j].x[i])

        # Updating state variables. If isochoric, update pressure. If isobaric, update
        # fluid volume.
        if isinstance(model, pp.fluid_mass_balance.VariablesSinglePhaseFlow):
            if results.specification >= FlashSpec.vT:
                update(model.pressure(subdomains), results.p)
            elif model.has_fluid_volume_variable:
                update(model.specific_fluid_volume(subdomains), results.v)

        # Update energy-related variables if applicable.
        if isinstance(model, pp.energy_balance.VariablesEnergyBalance):
            if results.specification not in [FlashSpec.pT, FlashSpec.vT]:
                update(model.temperature(subdomains), results.T)

            if (
                results.specification not in [FlashSpec.ph, FlashSpec.vh]
            ) and model.has_fluid_enthalpy_variable:
                update(model.specific_fluid_enthalpy(subdomains), results.h)

            if (
                results.specification != FlashSpec.vu
                and model.has_fluid_internal_energy_variable
            ):
                update(model.specific_fluid_internal_energy(subdomains), results.u)

    def resolve_md_flux_update(self, model: CIModel, dx: np.ndarray) -> None:
        """Modifies the update for interface and well fluxes such that the
        interface equations, which are linear in the respective variables, are
        fullfilled exactly.

        This is critical if any intensive state variable update was modified, or rapid
        changes in the rock occur.

        """
        if hasattr(model, "isochoric_npc_done"):
            if not model.isochoric_npc_done:
                return

        # Set intermediate state and update secondary quantities like properties and
        # upwind discretizations.
        rpc = model.params.get("linear_right_preconditioner", lambda x: x)
        dx_f = np.zeros_like(dx)
        xk1p = self._xk + model._scale_back_state(dx, is_increment=True)
        model.equation_system.set_variable_values(xk1p, iterate_index=0)
        model.update_derived_quantities()

        codim_1_interfaces = model.mdg.interfaces(codim=1)
        codim_2_interfaces = model.mdg.interfaces(codim=2)

        vars: list[pp.ad.MixedDimensionalVariable] = []
        advals: list[pp.ad.AdArray] = []

        # Interface Darcy Flux

        vars.append(model.interface_darcy_flux(codim_1_interfaces))
        advals.append(
            model.equation_system.evaluate(
                model.interface_darcy_flux_equation(codim_1_interfaces),
                derivative=True,
                state=None,
            )
        )

        # Well flux

        vars.append(model.well_flux(codim_2_interfaces))
        advals.append(
            model.equation_system.evaluate(
                model.well_flux_equation(codim_2_interfaces),
                derivative=True,
                state=None,
            )
        )

        if isinstance(model, pp.energy_balance.VariablesEnergyBalance):
            # Fourier flux

            vars.append(model.interface_fourier_flux(codim_1_interfaces))
            advals.append(
                model.equation_system.evaluate(
                    model.interface_fourier_flux_equation(codim_1_interfaces),
                    derivative=True,
                    state=None,
                )
            )

            # Interface enthalpy Flux

            vars.append(model.interface_enthalpy_flux(codim_1_interfaces))
            advals.append(
                model.equation_system.evaluate(
                    model.interface_enthalpy_flux_equation(codim_1_interfaces),
                    derivative=True,
                    state=None,
                )
            )

            # Well enthalpy Flux

            vars.append(model.well_enthalpy_flux(codim_2_interfaces))
            advals.append(
                model.equation_system.evaluate(
                    model.well_enthalpy_flux_equation(codim_2_interfaces),
                    derivative=True,
                    state=None,
                )
            )

        # Preconditioning Jacobians.

        res = [v.val for v in advals]
        Jacs = [v.jac for v in advals]
        if not self.params["in_physical_space"]:
            Jacs = rpc(Jacs)  # type:ignore[operator]

        for v, r, J in zip(vars, res, Jacs):
            P = model.equation_system.projection_to([v]).transpose()
            dx_f = -(J * P).transpose() @ r
            dofs = model.equation_system.dofs_of([v])
            dx[dofs] = dx_f
