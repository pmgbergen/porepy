"""This module contains functionality to solve the equilibrium problem numerically."""
from __future__ import annotations

import logging
import numbers
from dataclasses import asdict
from typing import Any, Callable, Literal, Optional

import numpy as np
import pypardiso
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ._core import (
    _rr_pole,
    rachford_rice_feasible_region,
    rachford_rice_potential,
    rachford_rice_vle_inversion,
)
from .composite_utils import safe_sum
from .heuristics import K_val_Wilson
from .mixture import NonReactiveMixture, ThermodynamicState
from .phase import Phase, PhaseProperties

__all__ = ["FlashSystemNR", "FlashNR"]

logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_handler.terminator = ""
logger.addHandler(log_handler)


def _pos(var: NumericType) -> NumericType:
    """Returns a numeric value where the negative parts have been set to zero."""
    if isinstance(var, pp.ad.Ad_array):
        eliminate = var.val < 0.0
        out_val = np.copy(var.val)
        out_val[eliminate] = 0
        out_jac = var.jac.tolil()
        out_jac[eliminate] = 0
        return pp.ad.Ad_array(out_val, out_jac.tocsr())
    elif isinstance(var, np.ndarray):
        eliminate = var < 0.0
        out = np.copy(var)
        out[eliminate] = 0
        return out
    elif isinstance(var, numbers.Real):
        if var < 0:
            return 0.0
        else:
            return var
    else:
        raise TypeError(f"Argument {var} is not numeric.")


def _neg(var: NumericType) -> NumericType:
    """Returns a numeric value where the positive parts have been set to zero."""
    if isinstance(var, pp.ad.Ad_array):
        eliminate = var.val > 0.0
        out_val = np.copy(var.val)
        out_val[eliminate] = 0
        out_jac = var.jac.tolil()
        out_jac[eliminate] = 0
        return pp.ad.Ad_array(out_val, out_jac.tocsr())
    elif isinstance(var, np.ndarray):
        eliminate = var > 0.0
        out = np.copy(var)
        out[eliminate] = 0
        return out
    elif isinstance(var, numbers.Real):
        if var > 0:
            return 0.0
        else:
            return var
    else:
        raise TypeError(f"Argument {var} is not numeric.")


class FlashSystemNR(ThermodynamicState):
    """A callable class representing the unified flash system for non-reactive mixtures.

    The call evaluates the equilibrium system and returns the value. This can be
    utilized to linearize the system for a stored, thermodynamic state.

    Note:
        Reference phase fractions are always treated as eliminated by unity.

    Parameters:
        mixture: The mixture this flash state represents.
        num_vals: Number of values per state function (for vectorized flash).
        flash_type: ``default='p-T'``

            A string-representation of the flash type. If e.g. ``'p-h'`` is passed,
            temperature is treated as an unknown and the flash system contains the
            enthalpy constraint.
        eos_kwargs: ``default={}``

                Keyword-arguments to be passed to
                :meth:`~porepy.composite.mixture.BasicMixture.compute_properties`.
        npipm: ``default=None``

            A bool indicating if the flash system is extended by the NPIPM method.

        **kwargs: Thermodynamic state values according to the parent data class.

    """

    # TODO should NPIPM slack equation be part of potential in Armijo line search?
    # TODO composition guess must not be regularized, if one phase is missing
    # otherwise nu in npipm will be instantiated with zero (1-sum x = 0)

    def __init__(
        self,
        mixture: NonReactiveMixture,
        num_vals: int,
        flash_type: str,
        eos_kwargs: dict,
        npipm: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._num_comp: int = mixture.num_components
        self._num_phases: int = mixture.num_phases
        self._num_vals: int = num_vals

        self._num_vars: int = self._num_comp * self._num_phases + self._num_phases - 1
        """Number of independent variables in the flash system."""

        self._nu: NumericType = 0.0
        """NPIPM slack variable."""

        self._T_unknown: bool = False
        """Flag if T is a variable."""

        self._p_unknown: bool = False
        """Flag if p is a variable."""

        # NPIPM parameters
        self._eta: float = 0.0
        self._u: float = 0.0
        self._kappa: float = 0.0

        # flags for which constraints to assemble
        self._assemble_h_constraint: bool = False
        """Flag if enthalpy constraint should be assembled."""
        self._assemble_v_constraint: bool = False
        """Flag if volume constraint should be assembled."""

        if "T" not in flash_type:  # If temperature is unknown
            self._T_unknown = True
            self._num_vars += 1
            if "h" in flash_type:
                self._assemble_h_constraint = True
            else:
                raise NotImplementedError("FlashState undefined. T and h missing.")
        if "p" not in flash_type:  # If pressure is unknown
            self._p_unknown = True
            self._num_vars += 1
            self._assemble_v_constraint = True

        # Finally, check if NPIPM slack var should be introduced
        if npipm:
            self._num_vars += 1  # slack variable is independent
            self._eta = npipm.get("eta", 0.5)
            self._u = npipm.get("u", 1.0)
            self._kappa = npipm.get("kappa", 1.0)

            # assembling slack variable
            jac_nu = sps.lil_matrix((self._num_vals, self._num_vals * self._num_vars))
            jac_nu[:, -self._num_vals :] = sps.identity(3, format="lil")

            val_nu: NumericType = safe_sum(
                [y_ * (1 - safe_sum(self.X[j])) for j, y_ in enumerate(self.y)]
            )  # type: ignore
            val_nu = val_nu.val / self._num_phases
            self._nu = pp.ad.Ad_array(val_nu, jac_nu.tocsr())

            # Adding zero-block to independent variables and convert to csr
            empty_block = sps.lil_matrix((self._num_vals, self._num_vals))
            for j in range(self._num_phases):
                self.y[j].jac = sps.hstack([self.y[j].jac, empty_block], format="csr")
                for i in range(self._num_comp):
                    self.X[j][i].jac = sps.hstack(
                        [self.X[j][i].jac, empty_block], format="csr"
                    ).tocsr()

            if self._T_unknown:
                self.T.jac = sps.hstack([self.T.jac, empty_block], format="csr")
            if self._p_unknown:
                self.p.jac = sps.hstack([self.p.jac, empty_block], format="csr")
                for j in range(self._num_phases):
                    self.s[j].jac = sps.hstack(
                        [self.s[j].jac, empty_block], format="csr"
                    )

        self.mixture: NonReactiveMixture = mixture
        """Passed at instantiation."""
        self.eos_kwargs: dict = eos_kwargs
        """Passed at instantiation."""

    def __call__(
        self, vec: Optional[np.ndarray] = None, with_derivatives: Optional[bool] = None
    ) -> NumericType:
        """Implements the assembly/ evaluation of the equilibrium problem.

        Parameters:
            vec: ``default=None``

                A state vector for this flash system configuration.

                If not given, the current :meth:`state` is used.
            with_derivatives: ``default=None``

                If True, the system is evaluated using AD-arrays, hence providing
                a linearized system.

                If True, the equilibrium system is only evaluated and the return value
                represent the residual.

        """
        if vec is None:
            vec = self.state

        idx = 0  # index counter from top to bottom
        p: NumericType = 0.0
        T: NumericType = 0.0
        z: list[NumericType] = [0.0] * self._num_comp
        y: list[NumericType] = [0.0] * self._num_phases
        s: list[NumericType] = [0.0] * self._num_phases
        X: list[list[NumericType]] = [[0.0] * self._num_comp] * self._num_phases
        nu: NumericType = 0.0

        if self._p_unknown:
            p = vec[idx : idx + self._num_vals]
            idx += self._num_vals
            if self._T_unknown:
                T = vec[idx : idx + self._num_vals]
                idx += self._num_vals
            else:
                T = self.T
                s_0 = 0.0
                for j in range(1, self._num_phases):
                    s[j] = vec[idx : idx + self._num_vals]
                    s_0 += vec[idx : idx + self._num_vals]
                    idx += self._num_vals
                s[0] = 1 - s_0  # reference phase saturation by unity
        else:
            p = self.p
            if self._T_unknown:
                T = vec[idx : idx + self._num_vals]
            else:
                T = self.T

        y_0 = 0.0
        for j in range(1, self._num_phases):
            y[j] = vec[idx : idx + self._num_vals]
            y_0 += vec[idx : idx + self._num_vals]
            idx += self._num_vals
        y[0] = 1 - y_0  # reference phase fraction by unity

        for j in range(self._num_phases):
            for i in range(self._num_comp):
                X[j][i] = vec[idx : idx + self._num_vals]
                idx += self._num_vals

        if self._nu:
            nu = vec[idx : idx + self._num_vals]

        # if the the linearized system is to be assembled, get the stored Jacobians.
        if with_derivatives:
            if self._p_unknown:
                p = pp.ad.Ad_array(p, self.p.jac)
                for j in range(self._num_phases):
                    s[j] = pp.ad.Ad_array(s[j], self.s[j].jac)
            if self._T_unknown:
                T = pp.ad.Ad_array(T, self.T.jac)
            for j in range(self._num_phases):
                y[j] = pp.ad.Ad_array(y[j], self.y[j].jac)
                for i in range(self._num_comp):
                    X[j][i] = pp.ad.Ad_array(X[j][i], self.X[j][i].jac)

        # computing properties necessary to assemble the system
        phase_props: list[PhaseProperties] = self.mixture.compute_properties(
            p, T, X, False, **self.eos_kwargs
        )

        ## Assembling values of flash system
        equations: list[NumericType] = []

        # TODO below loops can be made more compact for the cost of an arbitrary order
        # of equations

        # First, evaluate mass balance for components except reference component
        for i in range(1, self._num_comp):
            x_i = [X[j][i] for j in range(self._num_phases)]
            equ = self.mixture.evaluate_homogenous_constraint(z[i], y, x_i)
            equations.append(equ)
        
        # Second, evaluate equilibrium equations for each component
        # between independent phases and reference phase
        for i in range(self._num_comp):
            for j in range(1, self._num_phases):
                equ = (
                    phase_props[j].phis[i] * X[j][i]
                    - phase_props[0].phis[i] * X[0][i]
                )
                equations.append(equ)

        # Third, enthalpy constraint if temperature unknown
        if self._T_unknown and self._assemble_h_constraint:
            h_j = [phase_props[j].h for j in range(self._num_phases)]
            equ = self.mixture.evaluate_homogenous_constraint(self.h, y, h_j)

        # Third, volume constraint if pressure is unknown
        if self._p_unknown and self._assemble_v_constraint:
            rho_j = [phase_props[j].rho for j in range(self._num_phases)]
            rho = self.mixture.evaluate_weighed_sum(rho_j, s)
            equ = self.v - rho**-1
            equations.append(equ)
            for j in range(1, self._num_phases):
                equ = y[j] * rho - s[j] * rho_j[j]
                equations.append(equ)

        # Fourth, complementary conditions or NPIPM equations
        if nu:
            # NPIPM coupling
            composition_unities: list[NumericType] = []
            # Storage of regularization terms according to Vu et al.
            regularization = []

            for j in range(self._num_phases):
                unity_j: NumericType = self.mixture.evaluate_unity(X[j])  # type: ignore
                composition_unities.append(unity_j)
                equ = y[j] * unity_j - nu
                regularization.append(
                    (y[j].val * unity_j.val * self._u / self._num_phases**2)
                )
                equations.append(equ)

            # NPIPM slack equation
            equ = self.evaluate_npipm_slack(y, composition_unities, nu)

            # Regularizing the slack equation: Gauss-elimination steps using the
            # coupling equations and some factor 
            for j, reg in enumerate(regularization):
                equ_j = equations[-(j + 1)]
                equ.jac = equ.jac - sps.diags(reg) * equ_j.jac
                equ.val = equ.val - reg * equ_j.val
            equations.append(equ)
        else:
            # Semi-Smooth Newton
            for j in range(self._num_phases):
                # semi-smooth min(a,b) = max(-a, -b)
                equ = pp.ad.maximum(
                    (-1) * y[j], (-1) * self.mixture.evaluate_unity(X[j])
                )

        # Assemble the global system
        if with_derivatives:
            # with derivatives we need to stack values and Jacobians separately
            vals = []
            jacs = []
            for equ in equations:
                vals.append(equ.val)
                jacs.append(equ.jac)
            return pp.ad.Ad_array(np.hstack(vals), sps.vstack(jacs, format='csr'))
        else:
            return np.hstack(equations)

    def evaluate_npipm_slack(
        self, V: list[NumericType], W: list[NumericType], nu: NumericType
    ) -> NumericType:
        """Auxiliary method evaluating the slack equation in the NPIPM.

        For an explanation of the input arguments, see
        `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_ .

        Arguments ``V`` and ``W`` must be of length ``num_phases``.

        """
        positivity_penalty = list()
        # regularization = list()

        negativity_penalty = 0.

        for j in range(self._num_phases):
            v = V[j]
            w = W[j]
            positivity_penalty.append(v * w)
            negativity_penalty += _neg(v) * _neg(v) + _neg(w) * _neg(w)

        dot_part = pp.ad.power(
            _pos(safe_sum(positivity_penalty)), 2
        ) * self._u / self._num_phases**2

        f = (
            self._eta * nu
            + nu * nu
            + (negativity_penalty + dot_part) / 2
        )
        return f

    @property
    def state(self) -> np.ndarray:
        """The vector of unknowns for this flash system is given by

        1. (optional) pressure values,
        2. (optional) temperature values,
        3. (optional) saturation values for each phase except reference phase,
        4. phase fraction values for each phase except reference phase,
        5. phase compositions, ordered per phase per component,
        6. (optional) NPIPM slack variable.

        Parameters:
            vec: The unknowns can be set using this property. This should only be done
                at the end of a procedure.

        Returns:
            The vector of unknowns for this flash system.

        """
        if self._nu:
            nu = [self._nu.val]
        else:
            nu = []
        x = [
            self.X[j][i].val
            for j in range(self._num_phases)
            for i in range(self._num_comp)
        ]
        y = [self.y[j] for j in range(1, self._num_phases)]
        if self._p_unknown:
            p = [self.p.val]
            s = [self.s[j] for j in range(1, self._num_phases)]
        else:
            p = []
            s = []
        if self._T_unknown:
            T = [self.T.val]

        return np.hstack(p + T + s + y + x + nu)

    @state.setter
    def state(self, vec: np.ndarray) -> np.ndarray:
        idx = 0  # index counter from top to bottom

        if self._p_unknown:
            self.p.val = vec[idx : idx + self._num_vals]
            idx += self._num_vals
            if self._T_unknown:
                self.T.val = vec[idx : idx + self._num_vals]
                idx += self._num_vals
            else:
                s_0 = 0.0
                for j in range(1, self._num_phases):
                    self.s[j].val = vec[idx : idx + self._num_vals]
                    s_0 += vec[idx : idx + self._num_vals]
                    idx += self._num_vals
                self.s[0].val = 1 - s_0  # reference phase saturation by unity
        else:
            if self._T_unknown:
                self.T.val = vec[idx : idx + self._num_vals]

        y_0 = 0.0
        for j in range(1, self._num_phases):
            self.y[j].val = vec[idx : idx + self._num_vals]
            y_0 += vec[idx : idx + self._num_vals]
            idx += self._num_vals
        self.y[0].val = 1 - y_0  # reference phase saturation by unity

        for j in range(self._num_phases):
            for i in range(self._num_comp):
                self.X[j][i].val = vec[idx : idx + self._num_vals]
                idx += self._num_vals

        if self._nu:
            self._nu.val = vec[idx : idx + self._num_vals]

    def export_state(self) -> ThermodynamicState:
        """Returns the values of the currently stored, thermodynamic state."""
        return ThermodynamicState(**asdict(self)).values()

    def evaluate_saturations(self, eps: float = 1e-10) -> None:
        """Function to evaluate the saturations based on molar fractions and
        saturations.

        The densities will be computed using currently stored values for compositions,
        pressure and temperature.

        Parameters:
            eps: ``default=1e-8``

                Tolerance for detection of saturated phases.

        """
        if self._num_phases == 1:
            self.s[0] = np.ones(self._num_vals)
        else:
            p = self.p.val if isinstance(self.p, pp.ad.Ad_array) else self.p
            T = self.T.val if isinstance(self.T, pp.ad.Ad_array) else self.T
            X = [
                [self.X[j][i].val for i in range(self._num_comp)]
                for j in range(self._num_phases)
            ]

            phase_pros: list[PhaseProperties] = self.mixture.compute_properties(
                p, T, X, store=False
            )

            densities: list[NumericType] = [props.rho for props in phase_pros]

            if self._num_phases == 2:
                # 2-phase saturation evaluation can be done analytically
                rho_1, rho_2 = densities
                y_1, y_2 = self.y
                
                s_1 = pp.ad.Ad_array(
                    np.zeros(y_1.val.shape), sps.lil_matrix(y_1.jac.shape)
                )
                s_2 = pp.ad.Ad_array(
                    np.zeros(y_2.val.shape), sps.lil_matrix(y_2.jac.shape)
                )

                phase_1_saturated = np.logical_and(
                    y_1 > 1 - eps, y_1 < 1 + eps
                )
                phase_2_saturated = np.logical_and(
                    y_2 > 1 - eps, y_2 < 1 + eps
                )
                idx = np.logical_not(phase_2_saturated)
                y2_idx = y_2[idx]
                rho1_idx = rho_1[idx]
                rho2_idx = rho_2[idx]
                s_1[idx] = 1.0 / (1.0 + y2_idx / (1.0 - y2_idx) * rho1_idx / rho2_idx)
                s_1[phase_1_saturated] = 1.0
                s_1[phase_2_saturated] = 0.0

                idx = np.logical_not(phase_1_saturated)
                y1_idx = y_1[idx]
                rho1_idx = rho_1[idx]
                rho2_idx = rho_2[idx]
                s_2[idx] = 1.0 / (1.0 + y1_idx / (1.0 - y1_idx) * rho2_idx / rho1_idx)
                s_2[phase_1_saturated] = 0.0
                s_2[phase_2_saturated] = 1.0

                self.s = [s_1, s_2]
            else:
                # More than 2 phases requires the inversion of the matrix given by
                # phase fraction relations
                mats = list()

                # list of indicators per phase, where the phase is fully saturated
                saturated = list()
                # where one phase is saturated, the other vanish
                vanished = [
                    np.zeros(self._num_vals, dtype=bool)
                    for _ in range(self._num_phases)
                ]

                for j2 in range(self._num_phases):
                    # get the DOFS where one phase is fully saturated
                    # TODO check sensitivity of this
                    saturated_j = np.logical_and(
                        self.y[j2] > 1 - eps, self.y[j2] < 1 + eps
                    )
                    saturated.append(saturated_j)
                    # store information that other phases vanish at these DOFs
                    for j2 in range(self._num_phases):
                        if j2 != j2:
                            # where phase j is saturated, phase j2 vanishes
                            # logical OR acts cumulatively
                            vanished[j2] = np.logical_or(vanished[j2], saturated_j)

                # stacked indicator which DOFs
                saturated = np.hstack(saturated)
                # staacked indicator which DOFs vanish
                vanished = np.hstack(vanished)
                # all other DOFs are in multiphase regions
                multiphase = np.logical_not(np.logical_or(saturated, vanished))

                # construct the matrix for saturation flash
                # first loop, per block row (equation per phase)
                for j in range(self._num_phases):
                    mats_row = list()
                    # second loop, per block column (block per phase per equation)
                    for j2 in range(self._num_phases):
                        # diagonal values are zero
                        # This matrix is just a placeholder
                        if j == j2:
                            mats.append(sps.diags([np.zeros(self._num_vals)]))
                        # diagonals of blocks which are not on the main diagonal,
                        # are non-zero
                        else:
                            denominator = 1 - self.y[j].val
                            # to avoid a division by zero error, we set it to one
                            # this is arbitrary, but respective matrix entries are
                            # sliced out since they correspond to cells where one phase
                            # is saturated,
                            # i.e. the respective saturation is 1., the other 0.
                            denominator[denominator == 0.0] = 1.0
                            d = (
                                1.0
                                + densities[j2].val / densities[j].val
                                * self.y[j].val / denominator
                            )
                            mats_row.append(sps.diags([d]))

                    # row-block per phase fraction relation
                    mats.append(sps.hstack(mats_row, format='csr'))

                # Stack matrices per equation on each other
                # This matrix corresponds to the vector of stacked saturations per phase
                mat = sps.vstack(mats, format='csr')
                # TODO Matrix has large band width

                # projection matrix to DOFs in multiphase region
                # start with identity in CSR format
                projection: sps.spmatrix = sps.diags([np.ones(len(multiphase))]).tocsr()
                # slice image of canonical projection out of identity
                projection = projection[multiphase]
                projection_transposed = projection.transpose()

                # get sliced system
                rhs = projection * np.ones(self._num_vals * self._num_phases)
                mat = projection * mat * projection_transposed

                s = pypardiso.spsolve(mat, rhs)

                # prolongate the values from the multiphase region to global DOFs
                saturations = projection_transposed * s
                # set values where phases are saturated or have vanished
                saturations[saturated] = 1.0
                saturations[vanished] = 0.0

                # distribute results to the saturation variables
                for j in range(self._num_phases):
                    vals = saturations[j * self._num_vals : (j + 1) * self._num_vals]
                    if isinstance(self.s[j], pp.ad.Ad_array):
                        self.s[j] = vals
                    else:
                        self.s[j] = vals

    def validate_fractions(self, tol: float = 1e-8, raise_error: bool = False) -> None:
        """A method to validate fractions and trim numerical artifacts.
        
        For numerical reasons, fractions can lie outside the bound ``[0, 1]``.
        Use ``tol`` to determine how much is allowed.

        The fraction will be trimmed to the interval, **only** if it does not violate
        the tolerance. This can help mitigate error propagation.

        Everything above tolerance will log a warning or throw an error.

        Parameters:
            tol: ``default=1e-8``

                The fractions are checked to lie in the interval ``[-tol, 1 + tol]``.
            raise_error: ``default=False``

        """
        def _trim(vec: np.ndarray):
            vec[vec < 0] = 0.
            vec[vec > 1] = 1.

        if raise_error:
            def action(msg):
                raise AssertionError(msg)
        else:
            def action(msg):
                logger.warn(msg)

        msg = []

        for j in range(self._num_phases):
            if np.any(self.y[j] < -tol) or np.any(self.y[j] > 1 + tol):
                msg.append(f"\n\t- Phase fraction {j}\n")
            else:
                _trim(self.y[j].val)

            if np.any(self.s[j] < -tol) or np.any(self.s[j] > 1 + tol):
                msg.append(f"\n\t- Phase saturation {j}\n")
            else:
                _trim(self.s[j].val)

            for i in range(self._num_comp):
                if np.any(self.X[j][i] < -tol) or np.any(self.X[j][i] > 1 + tol):
                    msg.append(f"\n\t- Fraction of component {i} in phase {j}\n")
                else:
                    _trim(self.X[j][i].val)

        if msg:
            msg = f"\nDetected fractions outside bound [-{tol}, 1 + {tol}]:"
            msg += safe_sum(msg)
            action(msg)


class FlashNR:
    """A basic flash class for non-reactive mixtures computing fluid phase equilibria in
    the unified setting.

    Two methods are implemented:

    - A (semi-smooth) Newton using applied directly to equations provided by a mixture.
    - A non-parametric interior point method (NPIPM).

    Notes:
        1. The mixture set-up must be performed using
           ``semismooth_complementarity=True`` for the semi-smooth method to work as
           intended (see :meth:`~porepy.composite.mixture.NonReactiveMixture.set_up`).
        2. The reference phase fraction must be eliminated
           (``eliminate_ref_phase=True`` during
           :meth:`~porepy.composite.mixture.NonReactiveMixture.set_up`).

    References:
        [1]: `Lauser et al. (2011) <https://doi.org/10.1016/j.advwatres.2011.04.021>`_
        [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_

    Parameters:
        mixture: A non-reactive mixture class.

    """

    def __init__(self, mixture: NonReactiveMixture) -> None:

        ### PUBLIC

        self.mixture: NonReactiveMixture = mixture
        """The mixture class passed at instantiation."""

        self.history: list[dict[str, Any]] = list()
        """Contains chronologically stored information about performed flash procedures.
        """

        self.max_history: int = 100
        """Maximal number of flash history entries (FiFo). Defaults to 100."""

        self.tolerance: float = 1e-7
        """Convergence criterion for the flash algorithm. Defaults to ``1e-7``."""

        self.max_iter: int = 100
        """Maximal number of iterations for the flash algorithms. Defaults to 100."""

        self.eps: float = 1e-10
        """Small number to define the numerical zero. Defaults to ``1e-10``."""

        self.use_armijo: bool = True
        """A bool indicating if an Armijo line-search should be performed after an
        update direction has been found. Defaults to True."""

        self.newton_update_chop: float = 1.0
        """A number in ``[0, 1]`` to scale the Newton update resulting from
        solving the linearized system. Defaults to 1."""

        self.npipm_parameters: dict[str, float] = {
            "eta": 0.5,
            "u": 1,
            "kappa": 1.0,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the NPIPM:

        - ``'eta': 0.5``
        - ``'u': 1.``
        - ``'kappa': 1.``

        Values can be set directly by modifying the values of this dictionary.

        See Also:
            `Vu et al. (2021), Section 6.
            <https://doi.org/10.1016/j.matcom.2021.07.015>`_

        """

        self.armijo_parameters: dict[str, float] = {
            "kappa": 0.4,
            "rho": 0.99,
            "j_max": 150,
            "return_max": False,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the Armijo line-search:

        - ``'kappa': 0.4``
        - ``'rho': 0.99``
        - ``'j_max': 150`` (maximal number of Armijo iterations)
        - ``'return_max': False`` (returns last iterate if maximum number is reached)

        Values can be set directly by modifying the values of this dictionary.

        """

    def flash(
        self,
        state: dict[Literal["p", "T", "v", "u", "h"], pp.ad.Operator | NumericType],
        eos_kwargs: dict = dict(),
        method: Literal["newton-min", "npipm"] = "npipm",
        guess_from_state: bool = False,
        to_state: bool = False,
        verbosity: bool = 0,
    ) -> tuple[bool, ThermodynamicState]:
        """Performs a flash procedure based on the arguments.

        Note:
            Passing a state with ``n`` values, and leaving
            ``to_state, guess_from_state=False``, allows the user to calculate a
            a smaller flash problem, even if the mixture is defined in a system
            with ``m != n`` DOFs per state function!

            In theory this allows the user to partition the flash problem
            into domain-wise sub-problems (if for example the underlying
            mixed-dimensional domain in :attr:`~porepy.composite.mixture.Mixture.system`
            is large).

            Also, different flash-types can be performed on different partitions.

        Parameters:
            state: The thermodynamic state.

                Besides the feed :attr:`~porepy.composite.component.Component.fraction`,
                the thermodynamic state must be fixed with exactly 2 other quantities.

                Currently supported are

                - p-T
                - p-h
                - v-h

                The state can be passed using PorePy's AD operators, or numerical
                values. In either way, it the size must be compatible with the
                system used to model the mixture.
            eos_kwargs: ``default={}``

                Keyword-arguments to be passed to
                :meth:`~porepy.composite.mixture.BasicMixture.compute_properties`.
            method: ``default='npipm'``

                A string indicating the chosen algorithm:

                - ``'newton-min'``: A semi-smooth Newton method, where the complementary
                  conditions and and their derivatives are evaluated using a semi-smooth
                  min function [1].
                - ``'npipm'``: A Non-Parametric Interior Point Method [2].

            guess_from_state: ``default=False``

                If ``True``, the flash will take the values currently stored in the AD
                framework (``ITERATE`` or ``STATE``) as the initial guess for fractional
                values (**only**). The values passed in ``state`` must be compatible
                with what the AD framework returns in this case.

                If ``False``, an initial guess for fractional values (and for pressure
                and temperature if required) is computed internally.

            to_state: ``default=False``

                **If** the flash is successful, writes the results to ``'STATE'``
                for each variable in the AD framework.
            verbosity: ``default=0``

                Verbosity for logging. Per default, only warnings and logs of higher
                severity are printed to the console.

        Raises:
            NotImplementedError: If flash-type is not supported (determined by passed
                ``state``).
            AssertionError: If ``method`` is not among supported ones.
            AssertionError: If the reference phase fractions were not eliminated.
            AssertionError: If the number of DOFs of passed state function values is
                inconsistent with the number of DOFs the mixture was modelled with.

        Returns:
            A 2-tuple containing

            - A bool indicating if flash was successful or not.
            - A data structure containing the resulting thermodynamic state.
              The returned state contains the passed values from ``state``, as well as
              resulting fractional values, and optionally other resulting states.
              E.g., a p-h flash returns a state structure with resulting temperature
              values.

        """

        # setting logging verbosity
        if verbosity:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # currently only 2-phase flash is supported
        assert (
            self.mixture.num_phases == 2
        ), "Currently only 2-phase flashes are supported."
        # check requested method
        assert method in ["netwon-min", "npipm"], f"Unsupported method '{method}'."
        assert (
            self.mixture.reference_phase_eliminated
        ), "Reference phase fractions must be eliminated."

        # parsing state arguments
        flash_type, state_args = self._parse_flash_input_state(state)

        num_vals = len(state_args[0])  # number of values per state function
        num_comp = self.mixture.num_components
        num_phases = self.mixture.num_phases
        success = False  # success flag

        # declaring state structures
        state_from_ad: Optional[ThermodynamicState] = None
        thd_state: ThermodynamicState

        if guess_from_state:
            state_from_ad = self.mixture.get_fractional_state_from_vector()
            # assert consistency
            # TODO this functionality can be extended with a projection
            assert num_vals == self.mixture.dofs, (
                f"Inconsistent number of state values passed ({num_vals})"
                + " for initial guess from stored state."
                + f"\nMixture is set up with {self.mixture.dofs} per state function."
            )

        logger.info(f"\nInitiating state..\n")
        # Computation of initial guess values
        if flash_type == "p-T":

            p, T = state_args
            # initialize local AD system

            if state_from_ad:
                thd_state = ThermodynamicState.initialize(
                    num_comp, num_phases, num_vals, True, values_from=state_from_ad
                )
                thd_state.p = p
                thd_state.T = T
            else:
                thd_state = ThermodynamicState.initialize(
                    num_comp, num_phases, num_vals, True
                )
                thd_state.p = p
                thd_state.T = T
                thd_state = self._guess_fractions(thd_state, guess_K_values=True)

        elif flash_type == "p-h":

            p, h = state_args

            if state_from_ad:
                thd_state = ThermodynamicState.initialize(
                    num_comp,
                    num_phases,
                    num_vals,
                    True,
                    ["T"],
                    values_from=state_from_ad,
                )
                thd_state.p = p
                thd_state.h = h
            else:
                thd_state = ThermodynamicState.initialize(
                    num_comp, num_phases, num_vals, True, ["T"]
                )
                thd_state.p = p
                thd_state.h = h
                # initial temperature guess using pseudo-critical temperature
                thd_state = self._guess_temperature(thd_state, True)
                thd_state = self._guess_fractions(thd_state, guess_K_values=True)

            # Alternating guess for fractions and temperature
            # successive-substitution-like
            for _ in range(3):
                thd_state = self._guess_fractions(thd_state, False)
                thd_state = self._guess_temperature(thd_state, False, 3)
            # final temperature guess
            thd_state = self._guess_temperature(thd_state, False, 3)

        elif flash_type == "v-h":
            NotImplementedError("Initial guess for v-h flash not implemented.")

        logger.info(
            f"\nStarting {flash_type} flash\n"
            + f"Method: {method}\n"
            + f"Using Armijo line search: {self.use_armijo}\n"
            + f"Initial guess from state: {guess_from_state}\n\n"
        )

        flash_system = FlashSystemNR(
            mixture=self.mixture,
            num_vals=num_vals,
            flash_type=flash_type,
            eos_kwargs=eos_kwargs,
            npipm=self.npipm_parameters if method == "npipm" else None,
            **asdict(thd_state),
        )

        # Perform Newton iterations with above F(x)
        success, iter_final, solution = self._newton_iterations(
            X_0=flash_system.state,
            F=flash_system,
        )

        flash_system.state = solution
        # TODO calculate saturations if not variables

        # storing the state in the AD framework if successful and requested
        if to_state and success:
            self._store_state_in_ad(thd_state)

        logger.info(
            f"\n{flash_type} flash done.\n"
            + f"SUCCESS: {success}\n"
            + f"Iterations: {iter_final}\n\n"
        )
        # append history entry
        self._history_entry(
            flash=flash_type,
            method="newton-min",
            iterations=iter_final,
            success=success,
        )

        return success, flash_system.export_state()

    ### misc methods -------------------------------------------------------------------

    def _parse_flash_input_state(
        self,
        state: dict[Literal["p", "T", "v", "u", "h"], pp.ad.Operator | NumericType],
    ) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
        """Auxiliary function to parse the given state into numeric format.

        Returns:
            A 2-tuple consisting of

            1. The flash-type, characterized by a string symbols for state variables
               which are passed, f.e. ``'p-T'``.
            2. Exactly two state function values.

        """
        # currently available state input for flash
        available_state_args = ["p", "T", "v", "u", "h"]

        parsed_state = [state.get(var, None) for var in available_state_args]

        # check if state is not over-constrained (unsupported)
        num_states = len([s for s in parsed_state if s is not None])
        assert num_states == 2, "Thermodynamic state is over-constrained (need 2)."

        # determining flash type
        flash_type: str
        state_1: pp.ad.Operator | NumericType
        state_2: pp.ad.Operator | NumericType
        if parsed_state[0] is not None and parsed_state[1] is not None:
            flash_type = "p-T"
            state_1 = parsed_state[0]
            state_2 = parsed_state[1]
        elif parsed_state[0] is not None and parsed_state[4] is not None:
            flash_type = "p-h"
            state_1 = parsed_state[0]
            state_2 = parsed_state[4]
        elif parsed_state[2] is not None and parsed_state[4] is not None:
            flash_type = "h-v"
            state_1 = parsed_state[4]  # mind the order!
            state_2 = parsed_state[0]
        else:
            flash_type = "-".join(
                [
                    available_state_args[i]
                    for i in range(len(available_state_args))
                    if parsed_state[i] is not None
                ]
            )
            raise NotImplementedError(
                f"Unsupported flash procedure for state given by {flash_type}"
            )

        parsed_state = [state_1, state_2]

        # converting AD operators to numeric format
        ad_args = [isinstance(var, pp.ad.Operator) for var in parsed_state]
        if np.any(ad_args):
            assert np.all(
                ad_args
            ), "If any state argument is an AD operator, all must be."

            parsed_state = [var.evaluate(self.mixture.system) for var in parsed_state]

        # if any state is an Ad_array, take only val
        parsed_state = [
            val.val if isinstance(val, pp.ad.Ad_array) else val for val in parsed_state
        ]
        # if any state is a number, convert to vector
        parsed_state = [
            np.array([val]) if isinstance(val, numbers.Real) else val
            for val in parsed_state
        ]

        # last sanity check, this should always hold if user is not messing around
        assert isinstance(state_1, np.ndarray) and isinstance(
            state_2, np.ndarray
        ), "Failed to parse input state to arrays."
        # last compatibility check that enough values are provided
        assert len(state_1) == len(state_2), "Must provide equally sized state values."

        return flash_type, (state_1, state_2)

    def _store_state_in_ad(self, state: ThermodynamicState) -> None:
        """Auxiliary function to store fractional values after a successful flash
        in the AD framework."""
        ads = self.mixture.system

        for j, phase in enumerate(self.mixture.phases):
            # storing phase fractions of independent phases
            if phase != self.mixture.reference_phase:
                ads.set_variable_values(state.y[j], [phase.fraction.name], True, True)
            # storing phase compositions
            for i, comp in enumerate(self.mixture.components):
                ads.set_variable_values(
                    state.X[j][i], [phase.fraction_of[comp].name], True, True
                )

    def _history_entry(
        self,
        flash: str = "p-T",
        method: str = "standard",
        iterations: int = 0,
        success: bool = False,
        **kwargs,
    ) -> None:
        """Makes an entry in the flash history."""

        if kwargs:
            other = safe_sum([f"\n\t{str(k)} : {str(v)}" for k, v in kwargs.items()])
        else:
            other = ""

        self.history.append(
            {
                "flash": str(flash),
                "method": str(method),
                "iterations": str(iterations),
                "success": str(success),
                "other": other,
            }
        )
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _print_system(
        self, A: sps.spmatrix, b: np.ndarray, print_dense: bool = False
    ) -> None:
        """Helper function to print a linearized system ``Ax=b`` to the console,
        including some numerical information."""
        print("---")
        print("||Res||: ", np.linalg.norm(b))
        print("Cond: ", np.linalg.cond(A.todense()))
        print("Rank: ", np.linalg.matrix_rank(A.todense()))
        print("---")
        if print_dense:
            print("Residual:")
            print(b)
            print("Jacobian:")
            print(A.todense())
            print("---")

    ### Initial guess strategies -------------------------------------------------------

    def _guess_fractions(
        self,
        state: ThermodynamicState,
        guess_K_values: bool = True,
    ) -> ThermodynamicState:
        """Computes a guess for phase fractions and compositions, based on
        feed fractions, pressure and temperature.

        This methodology uses some iterations on the Rachford-Rice equations, starting
        from K-values computed by the Wilson correlation.

        Parameters:
            state: A thermodynamic state data structure.
            guess_K_values: ``default=True``

                If True, computes first K-value guesses using the Wilson correlation.
                If False, computes the first K-values using the phase EoS.

        Returns:
            Argument ``state`` with updated fractions.

        """

        # declaration of returned objects
        nc = len(state.p)  # number of cells/ flashes
        nph = self.mixture.num_phases
        ncp = self.mixture.num_components
        # K-values per independent phase (nph - 1)
        K: list[list[NumericType]]
        # assert data structure for phase compositions is of correct length
        # First sub-list is always the composition of the reference phase
        state.X = [[0 for _ in range(ncp)] for _ in range(nph)]

        if guess_K_values:
            # TODO can we use Wilson for all independent phases if more than 2 phases?
            K = [
                [
                    K_val_Wilson(state.p, comp.p_crit, state.T, comp.T_crit, comp.omega)
                    for comp in self.mixture.components
                ]
                for _ in range(1, nph)
            ]
        else:
            phase_props: list[PhaseProperties] = [
                phase.compute_properties(state.p, state.T, state.X[j], store=False)
                for j, phase in enumerate(self.mixture.phases)
            ]
            K = [
                [phase_props[0].phis[i] / phase_props[j].phis[i] for i in range(ncp)]
                for j in range(1, nph)
            ]

        independent_phases: list[Phase] = list(self.mixture.phases[1:])  # type: ignore

        for i in range(3):
            # Computing phase fraction estimates,
            # depending on number of independent phases
            if nph == 2:
                y = rachford_rice_vle_inversion(state.z, K[0])
                negative_flash = np.logical_or(0.0 > y, y > 1.0)

                feasible_reg = rachford_rice_feasible_region(state.z, [y], K)
                rr_pot = rachford_rice_potential(state.z, [np.ones(nc)], K)

                y = np.where((rr_pot > 0.0) & negative_flash, np.zeros(nc), y)
                y = np.where(
                    feasible_reg & (rr_pot < 0.0) & negative_flash,
                    np.ones(nc),
                    y,
                )

                assert not np.any(
                    np.logical_or(0.0 > y, y > 1.0)
                ), "y fraction estimate outside bound [0, 1]."
                state.y = [1 - y, y]
            else:
                raise NotImplementedError(
                    f"p-T-based guess for {nph} phase fractions not available."
                )

            for i in range(ncp):
                t_i = _rr_pole(i, state.y[1:], K)
                # compute composition of independent phases
                for j in range(nph - 1):
                    x_ce = state.z[i] * K[j][i] / t_i
                    state.X[j + 1][i] = x_ce
                # compute composition of reference phase
                state.X[0][i] = state.z[i] / t_i

            # Re-normalizing compositions
            for j in range(nph):
                total_j = safe_sum(state.X[j])
                for i in range(ncp):
                    state.X[j][i] = state.X[j][i] / total_j

            # update K values from EoS
            props_r: PhaseProperties = self.mixture.reference_phase.compute_properties(
                state.p, state.T, state.X[0], store=False
            )
            for j, phase in enumerate(independent_phases):
                props: PhaseProperties = phase.compute_properties(
                    state.p, state.T, state.X[j], store=False
                )
                for i in range(ncp):
                    K[j][i] = props_r.phis[i] / props.phis[i] + 1e-12

        # Re-compute phase compositions with updated K-values, after above iterations
        for i in range(ncp):
            t_i = _rr_pole(i, state.y[1:], K)
            # compute composition of independent phases
            for j in range(nph - 1):
                state.X[j + 1][i] = state.z[i] * K[j][i] / t_i
            # compute composition of reference phase
            state.X[0][i] = state.z[i] / t_i
        # Re-normalizing compositions
        for j in range(nph):
            total_j = safe_sum(state.X[j])
            for i in range(ncp):
                state.X[j][i] = state.X[j][i] / total_j

        return state

    def _guess_temperature(
        self, state: ThermodynamicState, use_pseudocritical: bool = False, iter: int = 1
    ) -> ThermodynamicState:
        """Computes a temperature guess for a mixture.

        Parameters:
            state: A thermodynamic state data structure.
            use_pseudocritical: ``default=False``

                If True, the pseudo-critical temperature is computed, a sum of critical
                temperatures of components weighed with feed fractions.
                If False, the enthalpy constraint is solved using newton iterations.
            iter: ``default=1``

                Number of iterations for when solving the enthalpy constraint.

        Returns:
            Argument ``state`` with updated fractions.

        """
        if use_pseudocritical:
            T_pc = safe_sum(
                [
                    comp.T_crit * state.z[i]
                    for i, comp in enumerate(self.mixture.components)
                ]
            )
            state.T = T_pc  # type: ignore
        else:
            pass

    ### Numerical methods --------------------------------------------------------------

    def _l2_potential(self, vec: np.ndarray) -> float:
        """Auxiliary method implementing the potential function which is to be
        minimized in the line search. Currently it uses the least-squares-potential.

        Parameters:
            vec: Vector for which the potential should be computed

        Returns:
            Value of potential.

        """
        return float(np.dot(vec, vec) / 2)

    def _Armijo_line_search(
        self,
        X_k: np.ndarray,
        b_k: np.ndarray,
        DX: np.ndarray,
        F: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """Performs the Armijo line-search for a given function ``F(X)``
        and a preliminary update ``DX`` starting from ``X_k``,
        using the least-square potential.

        Parameters:
            X_k: Last iterate.
            b_k: Last right-hand side of the lin. system s.t. ``DX= DF[X_k] \ b_k``.
            DX: Preliminary update to iterate.
            F: A callable representing the function for which a potential-reducing
                step-size should be found.

        Raises:
            RuntimeError: If line-search in defined interval does not yield any results.
                (Applies only if ``'return_max'`` is set to ``False``)

        Returns:
            The step-size resulting from the line-search algorithm.

        """
        # get relevant parameters
        kappa = self.armijo_parameters["kappa"]
        rho = self.armijo_parameters["rho"]
        j_max = self.armijo_parameters["j_max"]
        return_max = self.armijo_parameters["return_max"]

        # get starting point from current ITERATE state at iteration k
        b_k_pot = self._l2_potential(b_k)

        pot_j = b_k_pot
        rho_j = rho

        logger.info(f"\nArmijo line search potential: {b_k_pot}")

        # if maximal line-search interval defined, use for-loop
        if j_max:
            for j in range(1, j_max + 1):
                # new step-size
                rho_j = rho**j

                # compute system state at preliminary step-size
                try:
                    b_j = F(X_k + rho_j * DX)
                except AssertionError:
                    logger.warn(f"Armijo line search j={j}: evaluation failed")
                    continue

                pot_j = self._l2_potential(b_j)

                logger.info(f"Armijo line search j={j}: potential = {pot_j}")

                # check potential and return if reduced.
                if pot_j <= (1 - 2 * kappa * rho_j) * b_k_pot:
                    logger.info(f"Armijo line search j={j}: success\n")
                    return rho_j

            # if for-loop did not yield any results, raise error if requested
            if return_max:
                logger.info(f"Armijo line search: reached max iter\n")
                return rho_j
            else:
                raise RuntimeError(
                    f"Armijo line-search did not yield results after {j_max} steps."
                )
        # if no j_max is defined, use while loop
        # NOTE: If system is bad in some sense,
        # this might not finish in feasible time.
        else:
            # prepare for while loop
            j = 1
            # while potential not decreasing, compute next step-size
            while pot_j > (1 - 2 * kappa * rho_j) * b_k_pot:
                # next power of step-size
                rho_j *= rho
                try:
                    b_j = F(X_k + rho_j * DX)
                except AssertionError:
                    logger.warn(f"Armijo line search j={j}: evaluation failed")
                    j += 1
                    continue
                j += 1
                pot_j = self._l2_potential(b_j)

                logger.info(f"Armijo line search j={j}: potential = {pot_j}")
            # if potential decreases, return step-size
            else:
                logger.info(f"Armijo line search j={j}: success\n")
                return rho_j

    def _newton_iterations(
        self,
        X_0: np.ndarray,
        F: Callable[[NumericType, Optional[bool]], NumericType],
    ) -> tuple[bool, int, np.ndarray]:
        """Performs standard Newton iterations using the matrix and rhs-vector returned
        by ``F``, until (possibly) the L2-norm of the rhs-vector reaches the convergence
        criterion.

        Parameters:
            X_0: Starting point for iterations.
            F: Function to be minimized. Must be able AD-capable, i.e. return an
                AD-array if a second, boolean True-argument is passed.

        Returns:
            A 3-tuple containing

            1. a bool representing the success-indicator,
            2. an integer representing the the number of iteration performed,
            3. The found root (Or the last iterate if max iter reached).

        """

        success: bool = False
        iter_final: int = 0

        F_k = F(X_0, True)
        X_k = X_0

        res_norm = np.linalg.norm(F_k.val)
        # if residual is already small enough
        if res_norm <= self.tolerance:
            logger.info("Newton iteration 0: success\n")
            success = True
        else:
            for i in range(1, self.max_iter + 1):
                logger.info(f"Newton iteration {i}: residual norm = {res_norm}")

                DX = pypardiso.spsolve(F_k.jac, -F_k.val) * self.newton_update_chop

                if self.use_armijo:
                    # get step size using Armijo line search
                    step_size = self._Armijo_line_search(X_k, -F_k.val, DX, F)
                    DX = step_size * DX

                X_k = X_k + DX
                F_k = F(X_k, True)
                res_norm = np.linalg.norm(F_k.val)

                # in case of convergence
                if res_norm <= self.tolerance:
                    # counting necessary number of iterations
                    iter_final = i + 1  # shift since range() starts with zero
                    logger.info(f"\nNewton iteration {iter_final}: success\n")
                    success = True
                    break

        return success, iter_final, X_k
