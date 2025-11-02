"""Module containing an implementation of the unified flash using (parallel) compiled
functions created with numba.

The flash system, including a non-parametric interior point method, is assembled and
compiled using :func:`numba.njit`, to enable an efficient solution of the equilibrium
problem.

The compiled functions are such that they can be used to solve multiple flash problems
in parallel.

Parallelization is achieved by applying Newton in parallel for multiple input.
The intended use is for larg compositional flow problems, where an efficient solution
to the local equilibrium problem is required.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_

"""

from __future__ import annotations

import copy
import logging
import time
from typing import Callable, Literal, Optional, Sequence, cast

# NOTE: numba.typed has a spurious py.typed file which confuses mypy and
# makes it render an endless amount of errors related to attributes of the
# numba package. Importing numba.typed like this makes mypy ignore that file.
import numba
import numba.typed
import numpy as np

import porepy as pp

from ..compiled_eos import CompiledEoS
from ..utils import _chainrule_fractional_derivatives, normalize_rows
from .abstract_flash import AbstractFlash, FlashResults, StateSpecType
from .flash_initializer import FlashInitializer
from .solvers import DEFAULT_SOLVER_PARAMS, MULTI_SOLVERS, SOLVERS
from .uniflash_equations import (
    complementary_conditions_jac,
    complementary_conditions_res,
    first_order_constraint_jac,
    first_order_constraint_res,
    generic_arg_from_flash_results,
    isofugacity_constraints_jac,
    isofugacity_constraints_res,
    mass_conservation_jac,
    mass_conservation_res,
    parse_generic_arg,
    parse_vectorized_generic_arg,
    phase_mass_constraints_jac,
    phase_mass_constraints_res,
)

__all__ = ["CompiledPersistentVariableFlash"]


logger = logging.getLogger(__name__)


class CompiledPersistentVariableFlash(AbstractFlash):
    """A class providing efficient unified flash calculations using numba-compiled
    functions.

    It uses the no-python mode of numba to produce highly efficient, compiled code.

    Flash equations are represented by callable residuals and Jacobians. Various
    flash types are assembled in a modular way by combining required, compiled equations
    into a solvable system.

    Since each system depends on the modelled phases and components, significant
    parts of the equilibrium problem must be compiled on the fly.

    This is a one-time action once the modelling process is completed.

    The supported flash types are than available until destruction.

    Supported flash types/specifications:

    1. ``'pT'``: state definition in terms of pressure and temperature
    2. ``'ph'``: state definition in terms of pressure and specific mixture enthalpy
    3. ``'vh'``: state definition in terms of specific volume and enthalpy of the
       mixture

    Important:
        The isenthalpic-isochoric flash is as of now not robust for some tricky areas.
        Use with care.

    Multiple flash problems can be solved in parallel by passing vectorized state
    definitions.

    Parameters:
        fluid: A mixture model containing modelled components and phases.

    Raises:
        AssertionError: If any of the following assumptions is violated

            - Exactly two phases modelled
            - At least two components modelled (non-singular)
            - All components present in all phases (unified assumption)

    """

    SUPPORTED_SPECIFICATIONS: tuple[Literal["pT"], Literal["ph"], Literal["vh"]] = (
        "pT",
        "ph",
        "vh",
    )
    """Supported flash types. Used for checking flash input."""

    def __init__(
        self,
        fluid: pp.Fluid[pp.FluidComponent, pp.Phase[pp.FluidComponent]],
        params: Optional[dict] = None,
    ) -> None:
        super().__init__(fluid, params)

        assert set(self.params["components_per_phase"]) == set(
            [self.params["num_components"]]
        ), "Supports only unified mixtures (all components in all phases)."

        assert len(set([p.eos for p in fluid.phases])) == 1, (
            "All phases must have the same EoS instance."
        )

        """A list containing acentric factors per component."""
        self._phasestates: Sequence[pp.compositional.PhysicalState] = [
            phase.state for phase in fluid.phases
        ]
        """A sequence containing the physical phase state per phase."""

        eos = fluid.reference_phase.eos
        assert isinstance(eos, CompiledEoS)
        self._eos: CompiledEoS = eos
        """Compiled EoS of the reference phase, assuming all phases have the same EoS.
        """

        self.initializer: FlashInitializer = self.params.get(
            "initializer", FlashInitializer(fluid)
        )
        """Flash initializer passed during instantiation.

        If not given, the heuristic :class:`~porepy.compositional.flash.
        flash_initializer.FlashInitializer` is assigned.

        """

        self.residuals: dict[str, Callable[[np.ndarray], np.ndarray]] = dict()
        """Contains per flash configuration the respective residuals as a callable."""

        self.jacobians: dict[str, Callable[[np.ndarray], np.ndarray]] = dict()
        """Contains per flash configuration the respective Jacobian as a callable."""

        self._nb_solver_params: dict[str, float]
        """Numba typed dict which can be passed to compiled functions. Created during
        first call to :meth:`_convert_solver_params`."""

    def _parse_and_complete_results(
        self,
        resultsarray: np.ndarray,
        results: FlashResults,
        phase_property_params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> None:
        """Helper function to fill a fluid state with the equilibrium results from the
        flash and evaluate all fluid properties using the values at equilibrium."""
        nphase = self.params["num_phases"]
        ncomp = self.params["num_components"]

        s, x, y, _, p, T, *_ = parse_vectorized_generic_arg(
            resultsarray, (nphase, ncomp), results.specification.name
        )

        results.y = y
        if "T" not in results.specification.name:
            results.T = T
        if "p" not in results.specification.name:
            results.p = p
        if "v" in results.specification.name:
            results.sat = s

        # Computing states for each phase after filling p, T and x
        results.phases = list()
        for j in range(nphase):
            results.phases.append(
                self._eos.compute_phase_properties(
                    self._phasestates[j],
                    results.p,
                    results.T,
                    x[j, :, :],
                    params=phase_property_params,
                )
            )

        # If v not defined, evaluate saturations based on rho and y.
        if "v" not in results.specification.name:
            results.evaluate_saturations()
        # Evaluate extensive properties of the fluid mixture at equilibrium values.
        results.evaluate_extensive_state()

    def _convert_solver_params(self, solver_params: dict[str, float]) -> None:
        """Helper method to convert the solver parameters dictionary into a
        numba-conformal type."""

        if not hasattr(self, "_nb_solver_params"):
            d = numba.typed.Dict.empty(
                key_type=numba.types.unicode_type, value_type=numba.types.float64
            )
            self._nb_solver_params = cast(dict[str, float], d)

        for k, v in solver_params.items():
            self._nb_solver_params[k] = float(v)

    def compile(self, *args: Literal["pT", "ph", "vh"]) -> None:
        """Triggers the assembly and compilation of equilibrium equations, as well as
        the EoS if not already compiled.

        This can take a considerable amount of time.

        Parameters:
            *args: Specify subset of flash types which should be compiled to safe time.

        """

        # If not specified, compile all.
        if not args:
            args = self.SUPPORTED_SPECIFICATIONS

        if not self._eos.is_compiled:
            self._eos.compile()

        self.initializer.compile(*args)

        # Setting outer scope variables to avoid referencing self in JIT functions.
        nphase = self.params["num_phases"]
        ncomp = self.params["num_components"]
        npnc = (nphase, ncomp)
        phasestates = np.array(
            [
                # Depending on the environment, the enum value is sometimes already
                # evaluated, sometimes not... (pytest)
                state if isinstance(state, int) else state.value
                for state in self._phasestates
            ],
            dtype=np.int8,
        )

        prearg_val_c = self._eos.funcs["prearg_val"]
        prearg_jac_c = self._eos.funcs["prearg_jac"]
        phis_c = self._eos.funcs["phis"]
        dphis_c = self._eos.funcs["dphis"]
        h_c = self._eos.funcs["h"]
        dh_c = self._eos.funcs["dh"]
        rho_c = self._eos.funcs["rho"]
        drho_c = self._eos.funcs["drho"]

        logger.info(f"Compiling {args} flash systems ...")
        start = time.time()

        if "pT" in args and "pT" not in self.residuals:
            logger.debug("Compiling pT flash ...")

            @numba.njit(numba.f8[:](numba.f8[:]))
            def F_pT(X_gen: np.ndarray) -> np.ndarray:
                gen_arg = parse_generic_arg(X_gen, npnc, "pT")

                x = gen_arg[1]
                y = gen_arg[2]
                z = gen_arg[3]
                p = gen_arg[4]
                T = gen_arg[5]
                params = gen_arg[8]

                # EoS specific computations
                xn = normalize_rows(x)
                phis = np.empty(npnc, dtype=np.float64)
                for j in range(nphase):
                    pre_res_j = prearg_val_c(phasestates[j], p, T, xn[j], params)
                    phis[j] = phis_c(pre_res_j, p, T, xn[j])

                res_1 = mass_conservation_res(x, y, z)
                res_2 = isofugacity_constraints_res(x, phis)
                res_3 = complementary_conditions_res(x, y)

                return np.hstack((res_1, res_2, res_3))

            @numba.njit(numba.f8[:, :](numba.f8[:]))
            def DF_pT(X_gen: np.ndarray) -> np.ndarray:
                gen_arg = parse_generic_arg(X_gen, npnc, "pT")

                x = gen_arg[1]
                y = gen_arg[2]
                p = gen_arg[4]
                T = gen_arg[5]
                params = gen_arg[8]

                # EoS specific computations
                xn = normalize_rows(x)
                phis = np.empty(npnc, dtype=np.float64)
                dphis = np.empty((nphase, ncomp, 2 + ncomp), dtype=np.float64)
                for j in range(nphase):
                    pre_res_j = prearg_val_c(phasestates[j], p, T, xn[j], params)
                    pre_jac_j = prearg_jac_c(phasestates[j], p, T, xn[j], params)
                    phis[j] = phis_c(pre_res_j, p, T, xn[j])
                    d_phi_j = dphis_c(pre_res_j, pre_jac_j, p, T, xn[j])
                    for i in range(ncomp):
                        dphis[j, i, :] = _chainrule_fractional_derivatives(
                            d_phi_j[i], x[j]
                        )

                jac_1 = mass_conservation_jac(x, y)
                jac_2 = isofugacity_constraints_jac(x, phis, dphis)
                jac_3 = complementary_conditions_jac(x, y)

                # Stack Jacobians and return only derivatives w.r.t. y and x
                return np.vstack((jac_1, jac_2, jac_3))[:, 2 + nphase - 1 :]

            self.residuals["pT"] = F_pT
            self.jacobians["pT"] = DF_pT

        if "ph" in args and "ph" not in self.residuals:
            logger.debug("Compiling ph flash ...")

            @numba.njit(numba.f8[:](numba.f8[:]))
            def F_ph(X_gen: np.ndarray) -> np.ndarray:
                gen_arg = parse_generic_arg(X_gen, npnc, "ph")

                x = gen_arg[1]
                y = gen_arg[2]
                z = gen_arg[3]
                p = gen_arg[4]
                T = gen_arg[5]
                h_target = gen_arg[7]
                params = gen_arg[8]

                # EoS specific computations
                xn = normalize_rows(x)
                phis = np.empty(npnc, dtype=np.float64)
                h = np.empty(nphase, dtype=np.float64)
                for j in range(nphase):
                    pre_res_j = prearg_val_c(phasestates[j], p, T, xn[j], params)
                    phis[j] = phis_c(pre_res_j, p, T, xn[j])
                    h[j] = h_c(pre_res_j, p, T, xn[j])

                res_1 = mass_conservation_res(x, y, z)
                res_2 = isofugacity_constraints_res(x, phis)
                # Adding additional term with T appearing in first order conditions.
                res_3 = first_order_constraint_res(h_target, y, h) / T**2
                # Non-dimensional scaling of enthalpy constraint.
                res_3 /= h_target
                res_4 = complementary_conditions_res(x, y)

                return np.hstack((res_1, res_2, res_3, res_4))

            @numba.njit(numba.f8[:, :](numba.f8[:]))
            def DF_ph(X_gen: np.ndarray) -> np.ndarray:
                gen_arg = parse_generic_arg(X_gen, npnc, "ph")

                x = gen_arg[1]
                y = gen_arg[2]
                p = gen_arg[4]
                T = gen_arg[5]
                h_target = gen_arg[7]
                params = gen_arg[8]

                # EoS specific computations
                xn = normalize_rows(x)
                phis = np.empty(npnc, dtype=np.float64)
                dphis = np.empty((nphase, ncomp, 2 + ncomp), dtype=np.float64)
                hs = np.empty(nphase, dtype=np.float64)
                dhs = np.empty((nphase, 2 + ncomp), dtype=np.float64)
                for j in range(nphase):
                    pre_res_j = prearg_val_c(phasestates[j], p, T, xn[j], params)
                    pre_jac_j = prearg_jac_c(phasestates[j], p, T, xn[j], params)
                    phis[j] = phis_c(pre_res_j, p, T, xn[j])
                    d_phi_j = dphis_c(pre_res_j, pre_jac_j, p, T, xn[j])
                    for i in range(ncomp):
                        dphis[j, i, :] = _chainrule_fractional_derivatives(
                            d_phi_j[i], x[j]
                        )
                    hs[j] = h_c(pre_res_j, p, T, xn[j])
                    dhs[j] = _chainrule_fractional_derivatives(
                        dh_c(pre_res_j, pre_jac_j, p, T, xn[j]), x[j]
                    )

                jac_1 = mass_conservation_jac(x, y)
                jac_2 = isofugacity_constraints_jac(x, phis, dphis)
                # Product rule for extra term 1/T**2.
                jac_3 = first_order_constraint_jac(y, hs, dhs, 1) / T**2
                h_res = first_order_constraint_res(h_target, y, hs)[0]
                jac_3[0, 1] -= 2.0 / T**3 * h_res
                # Scaling of constraint with target value.
                jac_3 /= h_target
                jac_4 = complementary_conditions_jac(x, y)

                # No derivatives w.r.t. pressure and saturations.
                jac = np.vstack((jac_1, jac_2, jac_3, jac_4))
                # NOTE, this is cumbersome, but Numba does not allow stacking of
                # single column (1D array) with other columns (2D array). So we slice
                # out only the columns belonging to saturations, and stack. Final slice
                # which removes column belonging to p is done after stack.
                return np.hstack((jac[:, :2], jac[:, 2 + nphase - 1 :]))[:, 1:]

            self.residuals["ph"] = F_ph
            self.jacobians["ph"] = DF_ph

        if "vh" in args and "vh" not in self.residuals:
            logger.debug("Compiling vh flash ...")

            @numba.njit(numba.f8[:](numba.f8[:]))
            def F_vh(X_gen: np.ndarray) -> np.ndarray:
                s, x, y, z, p, T, v_target, h_target, params = parse_generic_arg(
                    X_gen, npnc, "vh"
                )

                # EoS specific computations
                xn = normalize_rows(x)
                phis = np.empty(npnc, dtype=np.float64)
                hs = np.empty(nphase, dtype=np.float64)
                rhos = np.empty(nphase, dtype=np.float64)
                for j in range(nphase):
                    pre_res_j = prearg_val_c(phasestates[j], p, T, xn[j], params)
                    phis[j] = phis_c(pre_res_j, p, T, xn[j])
                    hs[j] = h_c(pre_res_j, p, T, xn[j])
                    rhos[j] = rho_c(pre_res_j, p, T, xn[j])

                res_1 = mass_conservation_res(x, y, z)
                res_2 = isofugacity_constraints_res(x, phis)
                res_3 = first_order_constraint_res(h_target, y, hs) / T**2
                # Non-dimensional scaling of first order constraints.
                res_3 /= h_target
                # res_4 *= v_target
                # NOTE due to v * rho = 1, the scaling of the volume constraint is
                # performed differently than for the enthalpy constraint.
                res_4 = first_order_constraint_res(1.0, s, v_target * rhos)
                res_5 = phase_mass_constraints_res(s, y, rhos)
                res_6 = complementary_conditions_res(x, y)

                return np.hstack((res_1, res_2, res_3, res_4, res_5, res_6))

            @numba.njit(numba.f8[:, :](numba.f8[:]))
            def DF_vh(X_gen: np.ndarray) -> np.ndarray:
                s, x, y, _, p, T, v_target, h_target, params = parse_generic_arg(
                    X_gen, npnc, "vh"
                )

                # EoS specific computations
                xn = normalize_rows(x)
                phis = np.empty(npnc, dtype=np.float64)
                dphis = np.empty((nphase, ncomp, 2 + ncomp), dtype=np.float64)
                hs = np.empty(nphase, dtype=np.float64)
                dhs = np.empty((nphase, 2 + ncomp), dtype=np.float64)
                rhos = np.empty(nphase, dtype=np.float64)
                drhos = np.empty((nphase, 2 + ncomp), dtype=np.float64)
                for j in range(nphase):
                    pre_res_j = prearg_val_c(phasestates[j], p, T, xn[j], params)
                    pre_jac_j = prearg_jac_c(phasestates[j], p, T, xn[j], params)
                    phis[j] = phis_c(pre_res_j, p, T, xn[j])
                    d_phi_j = dphis_c(pre_res_j, pre_jac_j, p, T, xn[j])
                    for i in range(ncomp):
                        dphis[j, i, :] = _chainrule_fractional_derivatives(
                            d_phi_j[i], x[j]
                        )
                    hs[j] = h_c(pre_res_j, p, T, xn[j])
                    dhs[j] = _chainrule_fractional_derivatives(
                        dh_c(pre_res_j, pre_jac_j, p, T, xn[j]), x[j]
                    )
                    rhos[j] = rho_c(pre_res_j, p, T, xn[j])
                    drhos[j] = _chainrule_fractional_derivatives(
                        drho_c(pre_res_j, pre_jac_j, p, T, xn[j]), x[j]
                    )

                jac_1 = mass_conservation_jac(x, y)
                jac_2 = isofugacity_constraints_jac(x, phis, dphis)
                # Product rule for extra term 1/T**2.
                jac_3 = first_order_constraint_jac(y, hs, dhs, 1) / T**2
                h_res = first_order_constraint_res(h_target, y, hs)[0]
                jac_3[0, 1] -= 2.0 / T**3 * h_res
                jac_4 = first_order_constraint_jac(s, rhos, drhos, 0)
                # Non-dimensional scaling of constraints.
                jac_3 /= h_target
                jac_4 *= v_target
                jac_5 = phase_mass_constraints_jac(s, y, rhos, drhos)
                jac_6 = complementary_conditions_jac(x, y)

                return np.vstack((jac_1, jac_2, jac_3, jac_4, jac_5, jac_6))

            self.residuals["vh"] = F_vh
            self.jacobians["vh"] = DF_vh

        logger.info(
            f"{nphase}-phase, {ncomp}-component flash compiled"
            + " (elapsed time: %.5f (s))." % (time.time() - start)
        )

    def flash(
        self,
        specification: StateSpecType,
        z: Optional[Sequence[np.ndarray | pp.number]] = None,
        /,
        *,
        initial_state: Optional[pp.compositional.FluidProperties] = None,
        params: Optional[dict] = None,
        **kwargs,
    ) -> FlashResults:
        """Performes the flash for given feed fractions and supported equilibrium
        specifications (see :attr:`SUPPORTED_SPECIFICATIONS`).

        Supported parameters:

        - ``'mode'``: Mode of solving the equilibrium problems for multiple state
          definitions given by vectorized input.

          - ``'sequential'``: A classical loop over state defintions (row-wise).
          - ``'parallel'``: A parallelized loop, intended for larger amounts of
            problems.

            Defaults to ``'sequential'``.
        - ``'solver'``: selected solver (see
          :data:`~porepy.compositional.flash.solvers.SOLVERS`)
        - ``'solver_params'``: Custom solver parameters for single run. Otherwise the
          instance- :attr:`solver_params` are used.
        - ``'gen_arg_params'``: A sequence of arrays to be added as parameters
          to the generic flash argument. Can also contain floats, which will be
          broadcasted into the vectorized argument.
        - ``'phase_property_params'``: A sequence of arrays or floats to be used when
          calling :meth:`~porepy.compositional.compiled_eos.CompiledEoS.
          compute_phase_properties` after the flash is performed to evaluate dependent
          state functions and fluid properties.

        Raises:
            NotImplementedError: If an unsupported combination or insufficient number of
                of thermodynamic states is passed.

        """

        if params is None:
            params = {"mode": "sequential", "solver": "npipm"}

        mode = params.get("mode", "sequential")
        assert mode in MULTI_SOLVERS, f"Unsupported mode {mode}."
        solver = params.get("solver", "npipm")
        assert solver in SOLVERS, f"Unsupported solver {solver}."

        # Get default solver params.
        solver_params = copy.deepcopy(DEFAULT_SOLVER_PARAMS[solver])
        # Update params with params from instance.
        solver_params.update(self.solver_params)
        # Updating solver params for local run, if provided.
        solver_params.update(params.get("solver_params", {}))

        results = self.parse_flash_arguments(
            specification, z, initial_state=initial_state
        )
        logger.debug(
            f"{results.size} {results.specification} flash target state(s) parsed; "
            + f"DOFs: {results.dofs}; Solver: {solver} ({mode});"
        )
        assert results.specification in self.SUPPORTED_SPECIFICATIONS, (
            f"Unsupported flash type {results.specification}."
        )

        # Compile if not already compiled.
        if results.specification not in self.residuals:
            # NOTE ignore because parsing of flash type in base class supports more
            # configurations, while compile will do only what it can.
            self.compile(results.specification)  # type:ignore[arg-type]

        # Vectorized, generic flash argument as initial guess.
        X0 = generic_arg_from_flash_results(
            results,
            bool(initial_state),
            params.get("gen_arg_params", None),
        )

        # Compute initial guess if not provided.
        if initial_state is None:
            start = time.time()
            # NOTE Same ignore note as above on flash_type
            X0 = self.initializer[results.specification.name](X0)  # type:ignore[index]
            logger.debug(
                "Initial values computed (elapsed time: %.5f (s))."
                % (time.time() - start)
            )

        # Convert local solver params to numba-conform type
        solver_params["f_dim"] = results.dofs
        self._convert_solver_params(solver_params)

        start = time.time()
        resultsarray, success, num_iter = MULTI_SOLVERS[mode](
            X0,
            self.residuals[results.specification.name],
            self.jacobians[results.specification.name],
            SOLVERS[solver],
            self._nb_solver_params,
        )

        results.exitcode = success
        results.num_iter = num_iter

        logger.info(
            f"{results.size} {results.specification} flash solved"
            + " (elapsed time: %.5f (s))." % (time.time() - start)
        )
        logger.debug(
            f"Success: {results.converged.sum()} / {results.size};"
            + f" Max iter: {results.max_iter_reached.sum()};"
            + f" Diverged: {results.diverged.sum()};"
            + f" Failures: {np.sum(success > 2)};"
            + f" Iterations: {num_iter.max()} (max) "
            + f"{np.mean(num_iter):.2f} (avg);"
        )
        results.exitcode = success
        results.num_iter = num_iter

        self._parse_and_complete_results(
            resultsarray,
            results,
            phase_property_params=params.get("phase_property_params", None),
        )
        return results
