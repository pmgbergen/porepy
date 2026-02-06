"""Module containing an implementation of the persistent-variables flash using parallel
compiled functions created with numba.

Equations are assembled in a modular fashion depending on the flash specification.
They are always in the following order:

1. Mass conservation equations
2. Isofugacity equations
3. First-order optimality conditions (w.r.t. energy/enthalpy and/or volume).
4. Complementary conditions for phase fractions.

Each compiled flash is tailored to a fluid mixture with a given number of components and
phases.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Lipovac et al. (2024) <https://doi.org/10.1016/j.fluid.2023.113991>`_

"""

from __future__ import annotations

import copy
import logging
import time
from typing import Callable, Optional, Sequence

import numba as nb
import numpy as np

import porepy as pp

from .._numba_interface import get_empty_numba_dict, njit
from ..compiled_eos import CompiledEoS
from ..utils import (
    FlashSpec,
    FlashSpec_NUMBA_TYPE,
    _chainrule_fractional_derivatives,
    normalize_rows,
)
from .abstract_flash import AbstractFlash, FlashResults, StateSpecDict
from .flash_equations import (
    complementary_conditions_jac,
    complementary_conditions_res,
    first_order_constraint_jac,
    first_order_constraint_res,
    generic_arg_from_flash_results,
    isofugacity_constraints_jac,
    isofugacity_constraints_res,
    mass_constraint_jac,
    mass_constraint_res,
    parse_generic_arg,
    parse_vectorized_generic_arg,
    phase_mass_constraints_jac,
    phase_mass_constraints_res,
)
from .flash_initializer import FlashInitializer, UniformFlashInitializer
from .solvers import (
    DEFAULT_SOLVER_PARAMS,
    FLASH_JACOBIAN_SIGNATURE,
    FLASH_RESIDUAL_SIGNATURE,
    MULTI_SOLVERS,
    SOLVERS,
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

    SUPPORTED_SPECIFICATIONS: tuple[FlashSpec, ...] = (
        FlashSpec.pT,
        FlashSpec.ph,
        FlashSpec.vh,
        FlashSpec.vu,
    )
    """Supported flash specifications. Used for checking flash input."""

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

        states = tuple([phase.state for phase in fluid.phases])
        if np.any(
            [state == pp.compositional.PhysicalState.undefined for state in states]
        ):
            raise ValueError(
                "All phases must have a defined physical state in the "
                "persistent-variables flash."
            )
        self._phasestates: tuple[pp.compositional.PhysicalState, ...] = states
        """A sequence containing the physical phase state per phase."""

        eos = fluid.reference_phase.eos
        assert isinstance(eos, CompiledEoS), "Expecting compiled EoS."
        self._eos: CompiledEoS = eos
        """Compiled EoS of the reference phase, assuming all phases have the same EoS.
        """

        initializer: type[FlashInitializer] = self.params.get(
            "initializer", UniformFlashInitializer
        )
        self.initializer: FlashInitializer = initializer(fluid)
        """Flash initializer passed during instantiation.

        If not given, the heuristic :class:`~porepy.compositional.flash.
        flash_initializer.FlashInitializer` is assigned.

        """

        self.residuals: dict[FlashSpec, Callable[[np.ndarray], np.ndarray]] = dict()
        """Contains per flash configuration the respective residuals as a callable."""

        self.jacobians: dict[FlashSpec, Callable[[np.ndarray], np.ndarray]] = dict()
        """Contains per flash configuration the respective Jacobian as a callable."""

        self._nb_solver_params: dict[str, float]
        """Numba typed dict which can be passed to compiled functions. Created during
        first call to :meth:`_convert_solver_params`."""

        self._template_res: Callable[[np.ndarray, FlashSpec], np.ndarray]
        """Compiled flash residual template returning a uniform system depending on
        flash specification and flash argument."""

        self._template_jac: Callable[[np.ndarray, FlashSpec], np.ndarray]
        """Compiled flash residual template returning a uniform system depending on
        flash specification and flash argument."""

        # Setting default solver parameters.
        self.params["rpc_T_default"] = np.array(
            [c.critical_temperature for c in fluid.components]
        ).max()
        self.params["rpc_p_default"] = np.array(
            [c.critical_pressure for c in fluid.components]
        ).max()

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
            resultsarray, ncomp, nphase, results.specification
        )

        results.y = y
        if results.specification not in [FlashSpec.pT, FlashSpec.vT]:
            results.T = T
        if results.specification >= FlashSpec.vT:
            results.p = p
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

        # If not isochoric, evaluate saturations based on rho and y.
        if results.specification < FlashSpec.vT:
            results.evaluate_saturations()
        # Evaluate extensive properties of the fluid mixture at equilibrium values.
        results.evaluate_extensive_state()

    def _convert_solver_params(self, solver_params: dict[str, float]) -> None:
        """Helper method to convert the solver parameters dictionary into a
        numba-conformal type."""

        if not hasattr(self, "_nb_solver_params"):
            self._nb_solver_params = get_empty_numba_dict()

        for k, v in solver_params.items():
            self._nb_solver_params[k] = float(v)

    def compile(self, *args: FlashSpec) -> None:
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

        no_supp: list[str] = [
            s.name for s in args if s not in self.SUPPORTED_SPECIFICATIONS
        ]
        if no_supp:
            raise NotImplementedError(f"Specifications {no_supp} not supported.")

        self.initializer.compile(*args)

        # Setting outer scope variables to avoid referencing self in JIT functions.
        nphase = self.params["num_phases"]
        ncomp = self.params["num_components"]
        phasestates = self._phasestates

        prearg_val_c = self._eos.funcs["prearg_val"]
        prearg_jac_c = self._eos.funcs["prearg_jac"]
        phis_c = self._eos.funcs["phis"]
        dphis_c = self._eos.funcs["dphis"]
        h_c = self._eos.funcs["h"]
        dh_c = self._eos.funcs["dh"]
        rho_c = self._eos.funcs["rho"]
        drho_c = self._eos.funcs["drho"]
        u_c = self._eos.funcs["u"]
        du_c = self._eos.funcs["du"]

        logger.info(f"Compiling {[a.name for a in args]} flash systems ...")
        start = time.time()

        if not hasattr(self, "_template_res"):
            logger.debug("Compiling template flash residual ...")

            @njit(nb.f8[:](nb.f8[:], FlashSpec_NUMBA_TYPE))
            def template_res(X_gen: np.ndarray, spec: FlashSpec) -> np.ndarray:
                n_P = int(nphase)
                n_C = int(ncomp)
                states = nb.literal_unroll(phasestates)

                sat, x, y, z, p, T, s1, s2, xp = parse_generic_arg(
                    X_gen, n_C, n_P, spec
                )

                # EoS specific computations.
                xn = normalize_rows(x)
                phis = np.empty((n_P, n_C))  # Fugacities.
                es = np.empty(n_P)  # Energies.
                rhos = np.empty(n_P)  # Densities.

                for j in range(n_P):
                    pre_res_j = prearg_val_c(states[j], p, T, xn[j], xp)
                    phis[j] = phis_c(pre_res_j, p, T, xn[j])

                    if spec >= FlashSpec.vT:
                        rhos[j] = rho_c(pre_res_j, p, T, xn[j])

                    if spec in (FlashSpec.ph, FlashSpec.vh):
                        es[j] = h_c(pre_res_j, p, T, xn[j])
                    elif spec == FlashSpec.vu:
                        es[j] = u_c(pre_res_j, p, T, xn[j])

                # Block which all flashes have in common.
                res = np.hstack(
                    (
                        isofugacity_constraints_res(x, phis),
                        mass_constraint_res(x, y, z),
                        complementary_conditions_res(x, y),
                    )
                )

                # Pre-append volume block for isochoric specifications.
                if spec >= FlashSpec.vT:
                    res = np.hstack(
                        (
                            # NOTE: Scaling volume constraint with target volume s1
                            first_order_constraint_res(1.0, sat, s1 * rhos),
                            phase_mass_constraints_res(sat, y, rhos),
                            res,
                        )
                    )

                # Pre-append energy block for non-isothermal specifications.
                if spec in (FlashSpec.ph, FlashSpec.vh, FlashSpec.vu):
                    res_e = first_order_constraint_res(s2, y, es)
                else:
                    res_e = np.zeros((0,))

                # Scaling of energy residual.
                # NOTE analytically, this is correct, but numerically high T-values
                # allow for non-physical solutions because the energy residual is
                # scaled down.
                # if res_e.size > 0:
                #     res_e /= T**2
                # Non-dimensional scaling of energy residual.
                if np.abs(s2) > 1.0:
                    res_e /= s2

                return np.hstack((res_e, res))

            self._template_res = template_res

        if not hasattr(self, "_template_jac"):
            logger.debug("Compiling template flash Jacobian ...")

            @njit(nb.f8[:, :](nb.f8[:], FlashSpec_NUMBA_TYPE))
            def template_jac(X_gen: np.ndarray, spec: FlashSpec) -> np.ndarray:
                n_P = int(nphase)
                n_C = int(ncomp)
                states = nb.literal_unroll(phasestates)

                # Analogous to template_res, but for derivatives.
                sat, x, y, _, p, T, s1, s2, xp = parse_generic_arg(
                    X_gen, n_C, n_P, spec
                )

                # EoS specific computations.
                xn = normalize_rows(x)
                phis = np.empty((n_P, n_C))
                es = np.empty(n_P)
                rhos = np.empty(n_P)

                dphis = np.empty((n_P, n_C, 2 + n_C))
                des = np.empty((n_P, 2 + n_C))
                drhos = np.empty((n_P, 2 + n_C))

                for j in range(n_P):
                    pre_res_j = prearg_val_c(states[j], p, T, xn[j], xp)
                    pre_jac_j = prearg_jac_c(pre_res_j, p, T, xn[j], xp)
                    phis[j] = phis_c(pre_res_j, p, T, xn[j])

                    d_phi_j = dphis_c(pre_res_j, pre_jac_j, p, T, xn[j])
                    for i in range(n_C):
                        dphis[j, i, :] = _chainrule_fractional_derivatives(
                            d_phi_j[i], x[j]
                        )

                    if spec >= FlashSpec.vT:
                        rhos[j] = rho_c(pre_res_j, p, T, xn[j])
                        drhos[j] = _chainrule_fractional_derivatives(
                            drho_c(pre_res_j, pre_jac_j, p, T, xn[j]), x[j]
                        )

                    if spec in (FlashSpec.ph, FlashSpec.vh):
                        es[j] = h_c(pre_res_j, p, T, xn[j])
                        des[j] = _chainrule_fractional_derivatives(
                            dh_c(pre_res_j, pre_jac_j, p, T, xn[j]), x[j]
                        )
                    elif spec == FlashSpec.vu:
                        es[j] = u_c(pre_res_j, p, T, xn[j])
                        des[j] = _chainrule_fractional_derivatives(
                            du_c(pre_res_j, pre_jac_j, p, T, xn[j]), x[j]
                        )

                # Common block.
                jac = np.vstack(
                    (
                        isofugacity_constraints_jac(x, phis, dphis),
                        mass_constraint_jac(x, y),
                        complementary_conditions_jac(x, y),
                    )
                )

                # Pre-append volume block for isochoric specifications.
                if spec >= FlashSpec.vT:
                    jac = np.vstack(
                        (
                            first_order_constraint_jac(sat, rhos, drhos, False) * s1,
                            phase_mass_constraints_jac(sat, y, rhos, drhos),
                            jac,
                        )
                    )

                # Pre-append energy block for non-isothermal specifications.
                if spec in (FlashSpec.ph, FlashSpec.vh, FlashSpec.vu):
                    jac_e = first_order_constraint_jac(y, es, des, True)
                    # res_e = first_order_constraint_res(s2, y, es)[0]
                else:
                    jac_e = np.empty((0, jac.shape[1]))
                    # res_e = 0.0

                # if jac_e.size > 0:
                #     TT = T**2
                #     jac_e /= TT
                #     jac_e[0, 1] -= 2.0 / (TT * T) * res_e

                if np.abs(s2) > 1.0:
                    jac_e /= s2

                return np.vstack((jac_e, jac))

            self._template_jac = template_jac

        template_res = self._template_res
        template_jac = self._template_jac

        res_compiler = njit(FLASH_RESIDUAL_SIGNATURE)
        jac_compiler = njit(FLASH_JACOBIAN_SIGNATURE)

        # NOTE Cannot compile in loop over args, only reference of last is stored.

        if FlashSpec.pT in args:
            if FlashSpec.pT not in self.residuals:
                logger.debug("Compiling pT-flash residual ...")

                @res_compiler
                def res_pT(X_gen: np.ndarray) -> np.ndarray:
                    return template_res(X_gen, FlashSpec.pT)

                self.residuals[FlashSpec.pT] = res_pT

            if FlashSpec.pT not in self.jacobians:
                logger.debug("Compiling pT-flash Jacobian ...")

                @jac_compiler
                def jac_pT(X_gen: np.ndarray) -> np.ndarray:
                    n_P = int(nphase)
                    J = template_jac(X_gen, FlashSpec.pT)
                    return J[:, 2 + n_P - 1 :]

                self.jacobians[FlashSpec.pT] = jac_pT

        if FlashSpec.ph in args:
            if FlashSpec.ph not in self.residuals:
                logger.debug("Compiling ph-flash residual ...")

                @res_compiler
                def res_ph(X_gen: np.ndarray) -> np.ndarray:
                    return template_res(X_gen, FlashSpec.ph)

                self.residuals[FlashSpec.ph] = res_ph

            if FlashSpec.ph not in self.jacobians:
                logger.debug("Compiling ph-flash Jacobian ...")

                @jac_compiler
                def jac_ph(X_gen: np.ndarray) -> np.ndarray:
                    n_P = int(nphase)
                    J = template_jac(X_gen, FlashSpec.ph)
                    idx = np.zeros(J.shape[1], dtype=np.bool_)
                    idx[1] = True
                    idx[2 + n_P - 1 :] = True
                    return J[:, idx]

                self.jacobians[FlashSpec.ph] = jac_ph

        if FlashSpec.vu in args:
            if FlashSpec.vu not in self.residuals:
                logger.debug("Compiling vu-flash residual ...")

                @res_compiler
                def res_vu(X_gen: np.ndarray) -> np.ndarray:
                    return template_res(X_gen, FlashSpec.vu)

                self.residuals[FlashSpec.vu] = res_vu

            if FlashSpec.vu not in self.jacobians:
                logger.debug("Compiling vu-flash Jacobian ...")

                @jac_compiler
                def jac_vu(X_gen: np.ndarray) -> np.ndarray:
                    return template_jac(X_gen, FlashSpec.vu)

                self.jacobians[FlashSpec.vu] = jac_vu

        if FlashSpec.vh in args:
            if FlashSpec.vh not in self.residuals:
                logger.debug("Compiling vh-flash residual ...")

                @res_compiler
                def res_vh(X_gen: np.ndarray) -> np.ndarray:
                    return template_res(X_gen, FlashSpec.vh)

                self.residuals[FlashSpec.vh] = res_vh

            if FlashSpec.vh not in self.jacobians:
                logger.debug("Compiling vh-flash Jacobian ...")

                @jac_compiler
                def jac_vh(X_gen: np.ndarray) -> np.ndarray:
                    return template_jac(X_gen, FlashSpec.vh)

                self.jacobians[FlashSpec.vh] = jac_vh

        logger.info(
            f"{nphase}-phase, {ncomp}-component flash compiled"
            + " (elapsed time: %.5f (s))." % (time.time() - start)
        )

    def flash(
        self,
        specification: StateSpecDict,
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

        global SOLVERS, MULTI_SOLVERS, DEFAULT_SOLVER_PARAMS

        if params is None:
            params = {"mode": "sequential", "solver": "npipm"}

        mode = params.get("mode", "sequential")
        assert mode in MULTI_SOLVERS, f"Unsupported mode {mode}."
        solver = params.get("solver", "npipm")
        assert solver in SOLVERS, f"Unsupported solver {solver}."

        results = self.parse_flash_arguments(
            specification, z, initial_state=initial_state
        )
        logger.debug(
            f"{results.size} {results.specification.name} flash target state(s) parsed;"
            + f" DOFs: {results.dofs}; Solver: {solver} ({mode});"
        )
        assert results.specification in self.SUPPORTED_SPECIFICATIONS, (
            f"Unsupported flash type {results.specification.name}."
        )

        # Get default solver params.
        solver_params = copy.deepcopy(DEFAULT_SOLVER_PARAMS[solver])
        # Update params with params from instance.
        solver_params.update(self.solver_params)
        # Update right-preconditioning defaults for non-isothermal specs.
        if results.specification not in [FlashSpec.pT, FlashSpec.vT]:
            solver_params["rpc_T"] = self.params["rpc_T_default"]
        if results.specification >= FlashSpec.vT:
            solver_params["rpc_p"] = self.params["rpc_p_default"]
        # Updating solver params for local run, if provided.
        solver_params.update(params.get("solver_params", {}))

        # Compile if not already compiled.
        if results.specification not in self.residuals:
            # NOTE ignore because parsing of flash type in base class supports more
            # configurations, while compile will do only what it can.
            self.compile(results.specification)  # type:ignore[arg-type]

        # Vectorized, generic flash argument as initial guess.
        X0 = generic_arg_from_flash_results(
            results,
            self.params["num_components"],
            self.params["num_phases"],
            bool(initial_state),
            params.get("gen_arg_params", None),
        )

        # Compute initial guess if not provided.
        if initial_state is None:
            start = time.time()
            X0 = self.initializer[results.specification](X0)
            logger.debug(
                "Initial values computed (elapsed time: %.5f (s))."
                % (time.time() - start)
            )

        # Convert local solver params to numba-conform type
        solver_params["f_dim"] = results.dofs
        self._convert_solver_params(solver_params)

        start = time.time()
        resultsarray, exitcodes, num_iter = MULTI_SOLVERS[mode](
            np.ascontiguousarray(X0),
            self.residuals[results.specification],
            self.jacobians[results.specification],
            SOLVERS[solver],
            self._nb_solver_params,
            results.specification,
        )

        results.exitcode = exitcodes
        results.num_iter = num_iter

        logger.debug(
            f"{results.size} {results.specification.name} flash solved"
            + " (elapsed time: %.5f (s))." % (time.time() - start)
        )

        self._parse_and_complete_results(
            resultsarray,
            results,
            phase_property_params=params.get("phase_property_params", None),
        )
        return results
