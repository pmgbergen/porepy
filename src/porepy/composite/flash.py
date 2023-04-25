"""This module contains functionality to solve the equilibrium problem numerically
(flash)."""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import numpy as np
import pypardiso
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ._core import rachford_rice_equation, rachford_rice_potential
from .composite_utils import safe_sum
from .heuristics import K_val_Wilson
from .mixture import Mixture
from .peng_robinson.pr_utils import Leaf
from .phase import Phase

# import time


__all__ = ["Flash"]


def _del_log() -> None:
    # \r does not delete the printed characters in some consoles
    # use whitespace to overwrite them
    # obviously, this does not work for really long lines...
    print(
        "\r                                                                         \r",
        end="",
        flush=True,
    )


def _reg_smoother(x: NumericType) -> NumericType:
    return x / (x + 1)


def _pos(var):
    if isinstance(var, pp.ad.Ad_array):
        eliminate = var.val < 0.0
        out_val = np.copy(var.val)
        out_val[eliminate] = 0
        out_jac = var.jac.tolil()
        out_jac[eliminate] = 0
        return pp.ad.Ad_array(out_val, out_jac.tocsr())
    else:
        eliminate = var < 0.0
        out = np.copy(var)
        out[eliminate] = 0
        return out


def _neg(var):
    if isinstance(var, pp.ad.Ad_array):
        eliminate = var.val > 0.0
        out_val = np.copy(var.val)
        out_val[eliminate] = 0
        out_jac = var.jac.tolil()
        out_jac[eliminate] = 0
        return pp.ad.Ad_array(out_val, out_jac.tocsr())
    else:
        eliminate = var > 0.0
        out = np.copy(var)
        out[eliminate] = 0
        return out


class Flash:
    """A class containing various flash methods.

    Notes:
        - Two flash procedures are implemented: p-T flash and p-h flash.
        - Two numerical procedures are implemented (see below references).
        - Equilibrium calculations can be done cell-wise.
          I.e. the computations can be done smarter or parallelized.
          This is not exploited here.

    While the molar fractions are the actual unknowns in the flash procedure,
    the saturation values can be computed once the equilibrium converges using
    :meth:`post_process_fractions`.

    The specific enthalpy can be evaluated directly after a p-T flash using
    :meth:`evaluate_specific_enthalpy`.

    References:
        [1]: `Lauser et al. (2011) <https://doi.org/10.1016/j.advwatres.2011.04.021>`_
        [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_

    Parameters:
        composition: A mixture class with a set up AD system.
        auxiliary_npipm: ``default=False``

            An optional flag to enlarge the NPIPM system by introducing phase-specific
            slack variables and equations. This introduces the size of the system
            by ``2 * num_phases`` equations and variables and should be used carefully.

            This feature is left here to reflect the original algorithm
            introduced by Vu.

    """

    def __init__(
        self,
        mixture: Mixture,
        auxiliary_npipm: bool = False,
    ) -> None:

        self._MIX: Mixture = mixture
        """The mixture class passed at instantiation."""

        self._max_history: int = 100
        """Maximal number of flash history entries (FiFo)."""

        self._V_of_phase: dict[Phase, pp.ad.Operator] = dict()
        """A dictionary containing the NPIPM extension variable ``V`` for each phase,
        if used. Contains the phase fraction otherwise."""

        self._W_of_phase: dict[Phase, pp.ad.Operator] = dict()
        """A dictionary containing the NPIPM extension variable ``W`` for each phase,
        if used. Contains the phase composition unity otherwise."""

        self._nu: pp.ad.MixedDimensionalVariable
        """Slack variable ``nu`` representing the NPIPM parameter."""

        self._regularization_param = Leaf("reg")
        """A regularization parameter for the NPIPM to condition the system."""

        ### PUBLIC

        self.flash_history: list[dict[str, Any]] = list()
        """Contains chronologically stored information about performed flash procedures.
        """

        self.flash_tolerance: float = 1e-7
        """Convergence criterion for the flash algorithm. Defaults to ``1e-7``."""

        self.max_iter_flash: int = 100
        """Maximal number of iterations for the flash algorithms. Defaults to 100."""

        self.eps: float = 1e-10
        """Small number to define the numerical zero. Defaults to ``1e-10``."""

        self.use_armijo: bool = True
        """A bool indicating if an Armijo line-search should be performed after an
        update direction has been found. Defaults to True."""

        self.use_auxiliary_npipm_vars: bool = auxiliary_npipm
        """A bool indicating if the auxiliary variables ``V`` and ``W`` should
        be used in the NPIPM algorithm of Vu et al.. Passed at instantiation."""

        self.newton_update_chop: float = 1.0
        """A number in ``[0, 1]`` to scale the Newton update ``dx`` resulting from
        solving the linearized system. Defaults to 1."""

        self.npipm_parameters: dict[str, float] = {
            "eta": 0.5,
            "u": 1,
            "kappa": 1.0,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the NPIPM:

        - ``'eta'``
        - ``'u'``
        - ``'kappa'``

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

        - ``'kappa'``
        - ``'rho'``
        - ``'j_max'``
        - ``'return_max'``

        Values can be set directly by modifying the values of this dictionary.

        """

        ### setting if flash equations
        self.complementary_equations: list[str] = [
            equ.name for equ in self._MIX.AD.complementary_condition_of_phase.values()
        ]
        """A list of strings representing names of semi-smooth complementary conditions
        for the unified flash problem."""

        npipm_eqn, npipm_vars = self._set_npipm_eqn_vars()
        self.npipm_variables: list[str] = npipm_vars
        """A list containing algorithmic variables in for the NPIPM, which are
        neither physical nor must they be used in extended problems e.g., flow.

        """
        self.npipm_equations: list[str] = npipm_eqn
        """A list containing equations for the NPIPM, which are
        neither physical nor must they be used in extended problems e.g., flow.

        They result from specific algorithms chosen for the flash.

        """

    def _set_npipm_eqn_vars(self) -> list[str]:
        """Auxiliary function for the constructor to set equations and variables
        for the NPIPM method.

        Returns:
            A 2-tuple containing a list of equations names and a list of variable names
            introduced here.

        """
        ads = self._MIX.AD.system
        subdomains = ads.mdg.subdomains()
        nc = ads.mdg.num_subdomain_cells()

        npipm_vars: list[str] = list()
        npipm_equations: list[pp.ad.Operator] = list()

        # set initial values of zero for parameter nu
        self._nu = ads.create_variables("NPIPM-nu", subdomains=subdomains)
        npipm_vars.append(self._nu.name)
        ads.set_variable_values(
            np.zeros(nc), variables=[self._nu.name], to_iterate=True, to_state=True
        )

        # Defining and setting V per phase
        for phase in self._MIX.phases:
            # Instantiating V as a variable, if requested
            if self.use_auxiliary_npipm_vars:
                V_e = ads.create_variables(
                    f"NPIPM-V-{phase.name}", subdomains=subdomains
                )
                npipm_vars.append(V_e.name)
                ads.set_variable_values(
                    np.zeros(nc), variables=[V_e.name], to_iterate=True, to_state=True
                )
                self._V_of_phase[phase] = V_e
            # else we eliminate them by their respective definition
            else:
                if phase == self._MIX.reference_phase:
                    self._V_of_phase[phase] = self._MIX.AD.y_R
                else:
                    self._V_of_phase[phase] = phase.fraction

        # Defining and setting W per phase
        for phase in self._MIX.phases:
            # Instantiating W as a variable, if requested
            if self.use_auxiliary_npipm_vars:
                W_e = ads.create_variables(
                    f"NPIPM-W-{phase.name}", subdomains=subdomains
                )
                npipm_vars.append(W_e.name)
                ads.set_variable_values(
                    np.zeros(nc), variables=[W_e.name], to_iterate=True, to_state=True
                )
                self._W_of_phase[phase] = W_e
            # else we eliminate them by their respective definition
            else:
                self._W_of_phase[phase] = self._MIX.AD.composition_unity_of_phase[phase]

        # If V and W are requested as separate variables,
        # introduce the extension equations
        if self.use_auxiliary_npipm_vars:
            # V_e extension equations, create and store
            for phase in self._MIX.phases:
                if phase == self._MIX.reference_phase:
                    v_extension = self._MIX.AD.y_R - self._V_of_phase[phase]
                else:
                    v_extension = phase.fraction - self._V_of_phase[phase]
                v_extension.set_name(f"NPIPM-V-slack-{phase.name}")
                npipm_equations.append(v_extension)

            # W_e extension equations, create and store
            for phase in self._MIX.phases:
                w_extension = (
                    self._MIX.AD.composition_unity_of_phase[phase]
                    - self._W_of_phase[phase]
                )
                w_extension.set_name(f"NPIPM-W-slack-{phase.name}")
                npipm_equations.append(w_extension)

        # Equations coupling the KKT with the slack variable
        # NOTE: If nu becomes zero and V and W are not used,
        # this is reduced to the KKT conditions itself.
        # I.e. they are the same equations as for the Newton-min, except the derivative.
        for phase in self._MIX.phases:

            v = self._V_of_phase[phase]
            w = self._W_of_phase[phase]
            coupling = self.npipm_coupling(v, w, self._nu)

            if self.use_auxiliary_npipm_vars:
                coupling.set_name(f"NPIPM-coupling-VW-{phase.name}")
            else:
                coupling.set_name(f"NPIPM-coupling-cc-{phase.name}")
            npipm_equations.append(coupling)

        # NPIPM parameter equation
        f_args = [self._V_of_phase[phase] for phase in self._MIX.phases]
        f_args += [self._W_of_phase[phase] for phase in self._MIX.phases]
        f_args += [self._nu]

        equation = pp.ad.Function(func=self.npipm_f, name="NPIPM-AD-F")(*f_args)
        equation.set_name("NPIPM-slack-equation")
        npipm_equations.append(equation)

        # adding new equations to the ad system
        for eq in npipm_equations:
            ads.set_equation(
                eq,
                grids=subdomains,
                equations_per_grid_entity={"cells": 1},
            )

        return [eq.name for eq in npipm_equations], npipm_vars

    @staticmethod
    def npipm_coupling(v: NumericType, w: NumericType, nu: NumericType) -> NumericType:
        """Auxiliary method to evaluate the coupling between the
        complementary conditions and the slack variable ``nu``."""
        return v * w - nu

    def npipm_f(self, *X: tuple[NumericType]) -> NumericType:
        """Auxiliary method implementing the slack equation in the NPIPM.

        Parameters:
            *X: ``len=2*num_phases + 1``

                ``*X`` is assumed to be of length ``2*num_phases + 1``,
                where the first ``num_phases`` are ``v`` per phase,
                the second ``num_phases`` are ``w`` per phase
                and the last entry is ``nu``.

        Returns:
            Tue value of ``f(v, w, nu)`` in the NPIPM.

        """
        m = self._MIX.num_phases
        eta = self.npipm_parameters["eta"]
        coeff = self.npipm_parameters["u"] / m**2
        V = X[:m]
        W = X[m:-1]
        nu = X[-1]

        norm_parts = list()
        dot_parts = list()
        # regularization = list()

        for j in range(m):
            v = V[j]
            w = W[j]

            dot_parts.append(v * w)
            norm_parts.append(_neg(v) * _neg(v) + _neg(w) * _neg(w))

            # regularization.append(_reg_smoother(_neg(v) / (nu + 1)))

        dot_part = pp.ad.power(_pos(safe_sum(dot_parts)), 2) * coeff

        f = (
            eta * nu
            + nu * nu
            + (safe_sum(norm_parts) + dot_part) / 2
            # + _safe_sum(regularization) * self._regularization_param.value
        )
        return f

    ### printer methods ----------------------------------------------------------------

    def print_last_flash_results(self) -> None:
        """Prints the result of the last flash calculation."""
        entry = self.flash_history[-1]
        msg = "\nProcedure: %s\n" % (str(entry["flash"]))
        msg += "SUCCESS: %s\n" % (str(entry["success"]))
        msg += "Method: %s\n" % (str(entry["method"]))
        msg += "Iterations: %s\n" % (str(entry["iterations"]))
        msg += "Remarks: %s" % (str(entry["other"]))
        print(msg)

    def print_ordered_vars(self, vars):
        all_vars = [var.name for var in self._MIX.AD.system._variable_numbers]
        print(list(sorted(set(vars), key=lambda x: all_vars.index(x))), flush=True)

    def print_ph_system(self, print_dense: bool = False):
        print("---")
        print("Flash Variables:")
        self.print_ordered_vars(self._MIX.AD.ph_subsystem["primary-variables"])
        for equ in self._MIX.AD.ph_subsystem["equations"]:
            A, b = self._MIX.AD.system.assemble_subsystem(
                [equ], self._MIX.AD.ph_subsystem["primary-variables"]
            )
            print("---")
            print(equ)
            self.print_system(A, b, print_dense)
        print("---")

    def print_pT_system(self, print_dense: bool = False):
        print("---")
        print("Flash Variables:")
        self.print_ordered_vars(self._MIX.AD.pT_subsystem["primary-variables"])
        for equ in self._MIX.AD.pT_subsystem["equations"]:
            A, b = self._MIX.AD.system.assemble_subsystem(
                [equ], self._MIX.AD.pT_subsystem["primary-variables"]
            )
            print("---")
            print(equ)
            self.print_system(A, b, print_dense)
        print("---")

    def print_system(self, A, b, print_dense: bool = False):
        print("---")
        print("||Res||: ", np.linalg.norm(b))
        print("Cond: ", np.linalg.cond(A.todense()))
        # print("Eigvals: ", np.linalg.eigvals(A.todense()))
        print("Rank: ", np.linalg.matrix_rank(A.todense()))
        print("---")
        if print_dense:
            print("Residual:")
            print(b)
            print("Jacobian:")
            print(A.todense())
            print("---")

    def print_state(self, from_iterate: bool = False):
        """Print all information on the thermodynamic state of the mixture.

        Parameters:
            from_iterate: ``default=False``

                Take values from ``ITERATE``, instead of ``STATE`` by default.

        """
        filler = "---"
        ads = self._MIX.AD.system
        if from_iterate:
            print("ITERATE:")
        else:
            print("STATE:")
        print(filler)
        print("Pressure:")
        print(
            "\t"
            + str(
                ads.get_variable_values(
                    variables=[self._MIX.AD.p.name], from_iterate=from_iterate
                )
            )
        )
        print("Temperature:")
        print(
            "\t"
            + str(
                ads.get_variable_values(
                    variables=[self._MIX.AD.T.name], from_iterate=from_iterate
                )
            )
        )
        print("Enthalpy:")
        print(
            "\t"
            + str(
                ads.get_variable_values(
                    variables=[self._MIX.AD.h.name], from_iterate=from_iterate
                )
            )
        )
        print("Feed fractions:")
        for component in self._MIX.components:
            print(f"{component.fraction.name}: ")
            print(
                "\t"
                + str(
                    ads.get_variable_values(
                        variables=[component.fraction.name],
                        from_iterate=from_iterate,
                    )
                )
            )
        print(filler)
        print("Phase fractions:")
        for phase in self._MIX.phases:
            print(f"{phase.name}: ")
            print(
                "\t"
                + str(
                    ads.get_variable_values(
                        variables=[phase.fraction.name], from_iterate=from_iterate
                    )
                )
            )
        print("Saturations:")
        for phase in self._MIX.phases:
            print(f"{phase.name}: ")
            print(
                "\t"
                + str(
                    ads.get_variable_values(
                        variables=[phase.saturation.name], from_iterate=from_iterate
                    )
                )
            )
        print(filler)
        print("Composition:")
        for phase in self._MIX.phases:
            print(f"{phase.name}: ")
            for component in self._MIX.components:
                print(f"{phase.fraction_of_component(component).name}: ")
                print(
                    "\t"
                    + str(
                        ads.get_variable_values(
                            variables=[phase.fraction_of_component(component).name],
                            from_iterate=from_iterate,
                        )
                    )
                )
        print(filler)
        print("Algorithmic Variables:")
        for var in self.npipm_variables:
            print(f"\t{var}:")
            print(
                "\t"
                + str(
                    ads.get_variable_values(
                        variables=[var],
                        from_iterate=from_iterate,
                    )
                )
            )

    def _history_entry(
        self,
        flash: str = "isenthalpic",
        method: str = "standard",
        iterations: int = 0,
        success: bool = False,
        **kwargs,
    ) -> None:
        """Makes an entry in the flash history."""

        self.flash_history.append(
            {
                "flash": flash,
                "method": method,
                "iterations": iterations,
                "success": success,
                "other": str(kwargs),
            }
        )
        if len(self.flash_history) > self._max_history:
            self.flash_history.pop(0)

    ### Flash methods ------------------------------------------------------------------

    def flash(
        self,
        flash_type: Literal["pT", "ph"] = "pT",
        method: Literal["newton-min", "npipm"] = "npipm",
        initial_guess: Literal["iterate", "feed", "uniform", "rachford_rice"]
        | str = "iterate",
        copy_to_state: bool = False,
        do_logging: bool = False,
    ) -> bool:
        """Performs a flash procedure based on the arguments.

        Parameters:
            flash_type: ``default='pT'``

                A string representing the chosen flash type:

                - ``'pT'``: The composition is determined based on given
                  temperature, pressure and feed fractions per component.
                  Enthalpy is not considered and can be evaluated upon success using
                  :meth:`evaluate_specific_enthalpy`.
                - ``'ph'``: The composition **and** temperature are determined
                  based on given pressure and enthalpy.
                  Compared to the isothermal flash, the temperature in the isenthalpic
                  flash is an additional variable and an enthalpy constraint is
                  introduced into the system as an additional equation.

            method: ``default='npipm'``

                A string indicating the chosen algorithm:

                - ``'newton-min'``: A semi-smooth Newton method, where the KKT-
                  conditions and and their derivatives are evaluated using a semi-smooth
                  min function [1].
                - ``'npipm'``: A Non-Parametric Interior Point Method [2].

            initial_guess: ``default='iterate'``

                Strategy for choosing the initial guess:

                - ``'iterate'``: values from ITERATE or STATE, if ITERATE not existent.
                - ``'feed'``: feed fractions and k-values are used to compute initial
                  guesses.
                - ``'uniform'``: uniform fractions adding up to 1 are used as initial
                  guesses.

                Note:
                    For performance reasons, k-values are always evaluated using the
                    Wilson-correlation for computing initial guesses.

            copy_to_state: Copies the values to the STATE of the AD variables,
                additionally to ITERATE.

                Note:
                    If not successful, the ITERATE will **not** be copied to the STATE,
                    even if flagged ``True`` by ``copy_to_state``.

            do_logging: ``default=False``

                A bool indicating if progress logs should be printed.

        Raises:
            ValueError: If either ``flash_type``, ``method`` or ``initial_guess`` are
                unsupported keywords.

        Returns:
            A bool indicating if flash was successful or not.

        """
        success = False
        AD = self._MIX.AD

        other_equations = []
        variables = []
        preprocessor = None

        if flash_type == "pT":
            variables += [v for v in AD.pT_subsystem["primary-variables"]]
        elif flash_type == "ph":
            variables += [v for v in AD.ph_subsystem["primary-variables"]]
            other_equations += [AD.enthalpy_constraint.name]
        else:
            raise ValueError(f"Unknown flash type {flash_type}.")

        if do_logging:
            print("+++")
            print(f"Flash procedure: {flash_type}")
            print(f"Method: {method}")
            print(f"Using Armijo line search: {self.use_armijo}")
            print(f"Setting initial guesses: {initial_guess}")

        self._set_initial_guess(initial_guess)

        if method == "newton-min":
            other_equations += self.complementary_equations
        elif method == "npipm":
            variables += self.npipm_variables
            other_equations += self.npipm_equations
            preprocessor = self._npipm_pre_processor
            if do_logging:
                print("Setting initial NPIPM variables.", flush=True)
            self._set_NPIPM_initial_guess()
        else:
            raise ValueError(f"Unknown method {method}.")

        if do_logging:
            print("+++", flush=True)

        def assembler(
            X: Optional[np.ndarray] = None,
        ) -> tuple[sps.spmatrix, np.ndarray]:
            return AD.assemble_Gibbs(
                variables,
                other_equations=other_equations if other_equations else None,
                state=X,
            )

        # Perform Newton iterations with above F(x)
        success, iter_final = self._newton_iterations(
            assembler=assembler,
            global_prolongation=AD.system.projection_to(variables).transpose(),
            do_logging=do_logging,
            preprocessor=preprocessor,
        )

        # append history entry
        self._history_entry(
            flash=flash_type,
            method="newton-min",
            iterations=iter_final,
            success=success,
        )

        # setting STATE to newly found solution
        if copy_to_state and success:
            X = self._MIX.AD.system.get_variable_values(
                variables=variables, from_iterate=True
            )
            self._MIX.AD.system.set_variable_values(
                X, variables=variables, to_state=True
            )

        return success

    def evaluate_specific_enthalpy(self, copy_to_state: bool = True) -> None:
        """Evaluates the specific molar enthalpy of the composition,
        based on current pressure, temperature and phase fractions.

        Warning:
            To be used after an **isothermal** flash.

            Use with care, if the equilibrium problem is coupled with an energy balance.

        Parameters:
            copy_to_state: ``default=True``

                Copies the values to the STATE of the AD variable,
                additionally to ITERATE. Defaults to True.

        """
        # obtain values by forward evaluation
        h_mix = self._MIX.AD.h_mix.evaluate(self._MIX.AD.system)

        # write values in local var form
        self._MIX.AD.system.set_variable_values(
            h_mix.val,
            variables=[self._MIX.AD.h.name],
            to_iterate=True,
            to_state=copy_to_state,
        )

    def post_process_fractions(self, copy_to_state: bool = False) -> None:
        """Post-processes fractions after a flash procedure.

        1. Evaluates reference phase molar fraction
        2. Evaluates phase saturations
        3. Removes numerical artifacts from (fractions are bound in ``[0, 1]``).
           Tolerance :data:`eps` is applied.

        Removes numerical artifacts from all fractional values.

        Fractional values are supposed to be between 0 and 1 after any valid flash
        result. Molar phase fractions and phase compositions are post-processed
        respectively.

        Note:
            Components overall fractions (feed fractions) are not changed!
            The amount of moles belonging to a certain component are not supposed
            to change at all during a flash procedure without reactions.

        Parameters:
            copy_to_state: ``default=False``

                Copies the values to the STATE of the AD variable,
                additionally to ITERATE. Defaults to True.

        """
        ads = self._MIX.AD.system
        # evaluate reference phase fractions
        y_R = self._MIX.AD.y_R.evaluate(ads).val
        ads.set_variable_values(
            y_R,
            variables=[self._MIX.reference_phase.fraction.name],
            to_iterate=True,
            to_state=copy_to_state,
        )

        # remove numerical artifacts from molar phase fractions
        for phase in self._MIX.phases:
            y = ads.get_variable_values([phase.fraction.name], from_iterate=True)
            y = self._trim_fraction(y, phase.fraction.name)
            ads.set_variable_values(
                y,
                variables=[phase.fraction.name],
                to_iterate=True,
                to_state=copy_to_state,
            )

        # evaluate saturations
        self._evaluate_saturations(copy_to_state)

        # remove numerical artifacts from volumetric phase fractions
        for phase in self._MIX.phases:
            s = ads.get_variable_values([phase.saturation.name], from_iterate=True)
            s = self._trim_fraction(s, phase.saturation.name)
            ads.set_variable_values(
                s,
                variables=[phase.saturation.name],
                to_iterate=True,
                to_state=copy_to_state,
            )

        # remove numerical artifacts in phase compositions
        for phase in self._MIX.phases:
            for component in self._MIX.components:
                x = ads.get_variable_values(
                    [phase.fraction_of_component(component).name],
                    from_iterate=True,
                )
                x = self._trim_fraction(x, phase.fraction_of_component(component).name)
                # write values
                ads.set_variable_values(
                    x,
                    variables=[phase.fraction_of_component(component).name],
                    to_iterate=True,
                    to_state=copy_to_state,
                )

    def _trim_fraction(self, frac: np.ndarray, name: str) -> np.ndarray:
        """Auxiliary function to ensure the boundedness of a fraction in ``[0, 1]``.

        Raises errors otherwise.

        """
        assert np.all(frac >= -self.eps), f"Lower bound for fraction {name} violated."
        assert np.all(
            frac <= 1 + self.eps
        ), f"Upper bound for fraction {name} violated."
        frac[frac < 0] = 0.0
        frac[frac > 1] = 1.0
        return frac

    def _evaluate_saturations(self, copy_to_state: bool) -> None:
        """Evaluates the volumetric phase fractions (saturations) based on the number of
        modelled phases and the thermodynamic state of the mixture.

        To be used after any flash procedure.

        If no phases are present (e.g. before any flash procedure),
        this method does nothing.

        Parameters:
            copy_to_state: Copies the values to the STATE of the AD variable,
                additionally to ITERATE.

        """
        if self._MIX.num_phases == 1:
            self._single_phase_saturation_evaluation(copy_to_state)
        if self._MIX.num_phases == 2:
            self._2phase_saturation_evaluation(copy_to_state)
        elif self._MIX.num_phases >= 3:
            self._multi_phase_saturation_evaluation(copy_to_state)

    ### Initial guess strategies -------------------------------------------------------

    def _set_NPIPM_initial_guess(self) -> None:
        """Sets the initial guesses for ``V_e``, ``W_e`` and ``nu`` according to
        Vu (2021), section 3.3."""
        # initial guess for nu is constructed from V and W
        V: list[np.ndarray] = list()
        W: list[np.ndarray] = list()

        ads = self._MIX.AD.system

        for phase in self._MIX.phases:
            # if requested, set initial guess for auxiliary NPIPM vars V and W
            if self.use_auxiliary_npipm_vars:
                if phase == self._MIX.reference_phase:
                    v_e = self._MIX.AD.y_R.evaluate(ads).val
                else:
                    v_e = phase.fraction.evaluate(ads).val
                w_e = self._MIX.AD.composition_unity_of_phase[phase].evaluate(ads).val

                ads.set_variable_values(
                    v_e,
                    variables=[self._V_of_phase[phase].name],
                    to_iterate=True,
                    to_state=True,
                )
                ads.set_variable_values(
                    w_e,
                    variables=[self._W_of_phase[phase].name],
                    to_iterate=True,
                    to_state=True,
                )
            # else evaluate respective operators which define V and W
            else:
                v_e = self._V_of_phase[phase].evaluate(ads).val
                w_e = self._W_of_phase[phase].evaluate(ads).val

            # store value for initial guess for nu
            V.append(v_e)
            W.append(w_e)

        # initial guess for nu is cell-wise scalar product between concatenated V and W
        nu = safe_sum([v * w for v, w in zip(V, W)])
        nu = nu / self._MIX.num_phases
        # nu[nu < 0] = 0

        ads.set_variable_values(
            nu, variables=[self._nu.name], to_iterate=True, to_state=False
        )

        # some starting value for regularization
        self._regularization_param.value = 0.9 * np.ones(len(nu))

    def _set_initial_guess(self, initial_guess: str) -> None:
        """Auxillary function to set the initial values for phase fractions,
        phase compositions and temperature, based on the chosen strategy."""

        ads = self._MIX.AD.system
        nc = ads.mdg.num_subdomain_cells()

        if initial_guess == "iterate":
            # DofManager takes by default values from ITERATE, than from STATE if not found
            pass
        elif initial_guess == "uniform":

            # uniform values for phase fraction
            val_phases = 1.0 / self._MIX.num_phases
            for phase in self._MIX.phases:
                ads.set_variable_values(
                    val_phases * np.ones(nc),
                    variables=[phase.fraction.name],
                    to_iterate=True,
                )
                # uniform values for composition of this phase
                val = 1.0 / phase.num_components
                for component in self._MIX.components:
                    ads.set_variable_values(
                        val * np.ones(nc),
                        variables=[phase.fraction_of_component(component).name],
                        to_iterate=True,
                    )
        elif initial_guess == "feed":
            phases = [p for p in self._MIX.phases if p != self._MIX.reference_phase]
            # store preliminary phase composition
            composition: dict[Any, dict] = dict()
            # storing feed fractions
            feeds: dict[Any, np.ndarray] = dict()
            pressure = ads.get_variable_values(
                variables=[self._MIX.AD.p.name], from_iterate=True
            )
            temperature = ads.get_variable_values(
                variables=[self._MIX.AD.T.name], from_iterate=True
            )

            for comp in self._MIX.components:
                # evaluate feed fractions
                z_c = ads.get_variable_values(
                    variables=[comp.fraction.name], from_iterate=True
                )
                feeds[comp] = z_c

                # set fractions in reference phase to feed
                ads.set_variable_values(
                    np.copy(z_c),
                    variables=[
                        self._MIX.reference_phase.fraction_of_component(comp).name
                    ],
                    to_iterate=True,
                )

            # for phases except ref phase,
            # change values using the k-value estimates, s.t. initial guess
            # fulfils the equilibrium equations
            for phase in phases:
                composition[phase] = dict()
                for comp in self._MIX.components:
                    k_ce = (
                        comp.critical_pressure()
                        / pressure
                        * pp.ad.exp(
                            5.37
                            * (1 + comp.acentric_factor)
                            * (1 - comp.critical_temperature() / temperature)
                        )
                    )

                    x_ce = feeds[comp] * k_ce

                    composition[phase].update({comp: x_ce})

            # normalize initial guesses and set values in phases except ref phase.
            # feed fractions (composition of ref phase)
            # are assumed to be already normalized
            for phase in phases:
                for comp in self._MIX.components:
                    # normalize
                    x_ce = composition[phase][comp] / safe_sum(
                        composition[phase].values()
                    )
                    # set values
                    ads.set_variable_values(
                        x_ce,
                        variables=[phase.fraction_of_component(comp).name],
                        to_iterate=True,
                    )

            # use the feed fraction of the reference component to set an initial guess
            # for the phase fractions
            # re-normalize to set fractions fulfilling the unity constraint
            feed_R = feeds[self._MIX.reference_component] / len(phases)
            feed_R = np.ones(nc) * 0.9
            for phase in phases:
                ads.set_variable_values(
                    np.copy(feed_R), variables=[phase.fraction.name], to_iterate=True
                )
            # evaluate reference phase fraction by unity
            ads.set_variable_values(
                self._MIX.AD.y_R.evaluate(ads).val,
                variables=[self._MIX.reference_phase.fraction.name],
                to_iterate=True,
            )
        elif initial_guess == "rachford_rice":

            assert (
                self._MIX.num_phases == 2
            ), "Rachford-Rice initial guess only supported for liquid-gas-equilibrium."

            pressure = self._MIX.AD.p.evaluate(ads)
            temperature = self._MIX.AD.T.evaluate(ads)
            z_c = [comp.fraction.evaluate(ads) for comp in self._MIX.components]

            # Collects initial K values from Wilson's correlation
            K = [
                K_val_Wilson(
                    pressure.val,
                    comp.critical_pressure(),
                    temperature.val,
                    comp.critical_temperature(),
                    comp.acentric_factor,
                )
                for comp in self._MIX.components
            ]

            def ResidualRR(Y, z_c, K):
                res = np.zeros_like(Y)
                for z_i, K_i in zip(z_c, K):
                    res = res + (z_i.val * (K_i - 1)) / (1 + Y * (K_i - 1))
                return res

            def FunctionRR(Y, z_c, K):
                # A New Algorithm for Rachford-Rice for Multiphase Compositional Simulation
                # R. Okuno, R.T. Johns, and K. Sepehrnoori,
                # SPE, The University of Texas at Austin
                # TODO: Generalization to n_phases
                functionRR = np.zeros_like(Y)
                for z_i, K_i in zip(z_c, K):
                    functionRR = functionRR - z_i.val * np.log(
                        np.abs((1 + Y * (K_i - 1)))
                    )
                return functionRR

            def YConstraints(Y, z_c, K):
                # A New Algorithm for Rachford-Rice for Multiphase Compositional Simulation
                # Non-negativity of phase fracttion
                # R. Okuno, R.T. Johns, and K. Sepehrnoori,
                # SPE, The University of Texas at Austin
                t_vals = 1 + Y * (np.array(K) - 1.0)
                cond_1 = np.array([t - z_c[i].val for i, t in enumerate(t_vals)]) > 0
                cond_2 = (
                    np.array([t - K[i] * z_c[i].val for i, t in enumerate(t_vals)]) > 0
                )
                return np.all(np.logical_and(cond_1, cond_2), axis=0)

            def YPhaseFraction(z_c, K):
                # TODO: Generalize the case for multidimensional bisection
                # as an example check Efficient and Robust Three-Phase SplitComputations
                # Kjetil B. Haugen and Abbas Firoozabad

                # Since this is still a two-phase the inverse function is available
                # For the three-phase the inverse is still possible but more complicated
                d = (-1 + K[0]) * (-1 + K[1])
                n = z_c[0].val - K[0] * z_c[0].val + z_c[1].val - K[1] * z_c[1].val
                y = n / d
                return y

            Y = np.zeros(nc)
            composition: dict[Any, dict] = dict()
            for i in range(3):

                Y = YPhaseFraction(z_c, K)
                invalid_state = np.logical_or(0.0 > Y, Y > 1.0)

                gas_feasible_q = YConstraints(np.ones(nc), z_c, K)
                function_RR_val = FunctionRR(np.ones(nc), z_c, K)

                Y = np.where((function_RR_val > 0.0) & (invalid_state), np.zeros(nc), Y)
                Y = np.where(
                    (gas_feasible_q) & (function_RR_val < 0.0) & (invalid_state),
                    np.ones(nc),
                    Y,
                )

                assert not np.any(np.logical_or(0.0 > Y, Y > 1.0))

                for phase in self._MIX.phases:
                    composition[phase] = dict()
                    for i, comp in enumerate(self._MIX.components):
                        if phase.eos.gaslike:
                            x_ce = z_c[i] * K[i] / (1 + Y * (K[i] - 1))
                            composition[phase].update({comp: x_ce})
                        else:
                            x_ce = z_c[i] / (1 + Y * (K[i] - 1))
                            composition[phase].update({comp: x_ce})

                for phase in self._MIX.phases:
                    total = safe_sum(list(composition[phase].values()))
                    for comp in self._MIX.components:
                        x_ce = composition[phase][comp] / total
                        composition[phase].update({comp: x_ce})

                # update K values from EoS
                phi_L = None
                phi_G = None
                for phase in self._MIX.phases:
                    x_phase = [
                        composition[phase][comp] for comp in self._MIX.components
                    ]
                    phase.eos.compute(pressure, temperature, *x_phase)
                    if phase.eos.gaslike:
                        phi_G = list(phase.eos.phi.values())
                    else:
                        phi_L = list(phase.eos.phi.values())

                for i, pair in enumerate(zip(phi_L, phi_G)):
                    K[i] = (pair[0] / (pair[1] + 1.0e-12)).val

            # TODO: It seems x_ce is contextual
            # sometimes it is extended and sometimes partial.
            # Consider the possibility of having separate instances
            # for extended fractions and partial fractions
            for phase in self._MIX.phases:
                composition[phase] = dict()
                for i, comp in enumerate(self._MIX.components):
                    if phase.eos.gaslike:
                        x_ce = z_c[i] * K[i] / (1 + Y * (K[i] - 1))
                        composition[phase].update({comp: x_ce})
                    else:
                        x_ce = z_c[i] / (1 + Y * (K[i] - 1))
                        composition[phase].update({comp: x_ce})

            # set values.
            for phase in self._MIX.phases:
                for comp in self._MIX.components:
                    # set values
                    x_ce = composition[phase][comp].val
                    ads.set_variable_values(
                        x_ce,
                        variables=[phase.fraction_of_component(comp).name],
                        to_iterate=True,
                    )
            # set phase fraction
            for phase in self._MIX.phases:
                ads.set_variable_values(
                    np.copy(Y), variables=[phase.fraction.name], to_iterate=True
                )
            # evaluate reference phase fraction by unity
            ads.set_variable_values(
                self._MIX.AD.y_R.evaluate(ads).val,
                variables=[self._MIX.reference_phase.fraction.name],
                to_iterate=True,
            )
        else:
            raise ValueError(f"Unknown initial-guess-strategy {initial_guess}.")

    ### Numerical methods --------------------------------------------------------------

    def _update_reg(self):

        reg_k = self._regularization_param.value
        reg_geo = 0.5 * reg_k
        reg_pow = reg_k**2

        # TODO consider third option <V,W>+ / m

        # Don't use hstack, to avoid arrays with shape (n,) (second axis must exist)
        self._regularization_param.value = np.min(
            np.stack([reg_geo, reg_pow], axis=1), axis=1
        )

    def _npipm_pre_processor(
        self, A: sps.spmatrix, b: np.ndarray, prolongation: sps.spmatrix
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Pre-conditioning the NPIPM system by performing a Gauss elimination step
        involving the complementary conditions and the slack equation, and additionally
        regularizing the slack equation."""

        ads = self._MIX.AD.system
        nc = ads.mdg.num_subdomain_cells()
        u = self.npipm_parameters["u"]
        m = self._MIX.num_phases

        # reg_k = self._regularization_param
        # reg_geo = 0.5 * reg_k
        # reg_pow = reg_k**2
        # # TODO consider third option <V,W>+ / m

        # # Use stack not hstack, to avoid arrays with shape (n,) (second axis must exist)
        # self._regularization_param = np.min(
        #     np.stack([reg_geo, reg_pow], axis=1), axis=1
        # )

        # nu = self._nu.evaluate(self._C.ad_system)

        # The pre-conditioning consists of multiplying the coupling equation (per phase)
        # with dot_V_W**+ * u / m**2 and subtracting them from the slack equation for nu
        # Essentially a Gaussian elimination step as a pre-conditioning.
        A = A.tolil()  # for performance reasons

        # TODO: Augmentation performed under assumption that the order of the
        # equations is as constructed and added to the AD system
        # (hence the direct accessing of indices)
        # This should be done more generically with, in case the system changes
        # when coupled with something else or modified.

        for p, phase in enumerate(self._MIX.phases):
            v_phase = self._V_of_phase[phase].evaluate(ads)
            w_phase = self._W_of_phase[phase].evaluate(ads)
            factor = u / m**2 * v_phase.val * w_phase.val
            # positive part
            factor[factor < 0] = 0.0
            # last nc rows belong to slack equation
            # chose the the nc-long block rows above per phase for multiplication
            # sps.diags assures that factor (vector) is multiplied with each column
            # of the block-row
            A[-nc:] -= sps.diags(factor) * A[-(p + 2) * nc : -(p + 1) * nc]
            b[-nc:] -= factor * b[-(p + 2) * nc : -(p + 1) * nc]

            # NOTE: blocks of A belonging to the slack equation and V and W still
            # contain V- and W- respectively, as opposed to Vhu
            # They can be canceled by incorporating the following code:
            # if self.use_auxiliary_npipm_vars:
            #     A[-nc:, -m * 2 * nc :] = 0
            # else:
            #     A[-nc:, :-nc] = 0

            # regularization of slack equation
            # neg_v = v_phase.val > 0
            # v_phase.val[neg_v] = 0.0
            # v_phase.jac = v_phase.jac.tolil()
            # v_phase.jac[neg_v] = 0.0
            # neg_w = w_phase.val > 0.0
            # w_phase.val[neg_w] = 0.0
            # w_phase.jac = w_phase.jac.tolil()
            # w_phase.jac[neg_w] = 0.0

            # regularization = smoother(
            #     w_phase / (self._regularization_param + 1)
            # ) + smoother(v_phase / ( nu+ 1))
            # regularization = regularization * self._regularization_param

            # A[-nc:] += regularization.jac * prolongation
            # b[-nc:] -= regularization.val

        # back to csr and eliminate zeros
        A = A.tocsr()
        A.eliminate_zeros()

        return A, b

    def _Armijo_line_search(
        self,
        DX: np.ndarray,
        F: Callable[[Optional[np.ndarray]], tuple[sps.spmatrix, np.ndarray]],
        do_logging: bool,
    ) -> float:
        """Performs the Armijo line-search for a given function ``F(X)``
        and a preliminary update ``DX``, using the least-square potential.

        Parameters:
            DX: Preliminary update to solution vector.
            F: A callable representing the function for which a potential-reducing
                step-size should be found.

                The callable ``F(X)`` must be such that the input argument ``X`` is
                optional (None), which makes the AD-system chose values stored as
                ``ITERATE``.

        Raises:
            RuntimeError: If line-search in defined interval does not yield any results.
                (Applies only if `return_max` is set to `False`)

        Returns:
            The step-size resulting from the line-search algorithm.

        """
        # get relevant parameters
        kappa = self.armijo_parameters["kappa"]
        rho = self.armijo_parameters["rho"]
        j_max = self.armijo_parameters["j_max"]
        return_max = self.armijo_parameters["return_max"]

        # get starting point from current ITERATE state at iteration k
        _, b_k = F()
        b_k_pot = self._Armijo_potential(b_k)
        X_k = self._MIX.AD.system.get_variable_values(from_iterate=True)

        if do_logging:
            print(f"Armijo line search initial potential: {b_k_pot}")

        # if maximal line-search interval defined, use for-loop
        if j_max:
            for j in range(1, j_max + 1):
                # new step-size
                rho_j = rho**j

                # compute system state at preliminary step-size
                try:
                    _, b_j = F(X_k + rho_j * DX)
                except AssertionError:
                    if do_logging:
                        _del_log()
                        print(
                            f"Armijo line search j={j}: evaluation failed",
                            end="",
                            flush=True,
                        )
                    continue

                pot_j = self._Armijo_potential(b_j)

                if do_logging:
                    _del_log()
                    print(
                        f"Armijo line search j={j}: potential = {pot_j}",
                        end="",
                        flush=True,
                    )

                # check potential and return if reduced.
                if pot_j <= (1 - 2 * kappa * rho_j) * b_k_pot:
                    if do_logging:
                        _del_log()
                        print(f"Armijo line search j={j}: success", flush=True)
                    return rho_j

            # if for-loop did not yield any results, raise error if requested
            if return_max:
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
                    _, b_j = F(X_k + rho_j * DX)
                except AssertionError:
                    if do_logging:
                        _del_log()
                        print(
                            f"Armijo line search j={j}: evaluation failed",
                            end="",
                            flush=True,
                        )
                    j += 1
                    continue
                j += 1
                pot_j = self._Armijo_potential(b_j)

                if do_logging:
                    _del_log()
                    print(
                        f"Armijo line search j={j}: potential = {pot_j}",
                        end="",
                        flush=True,
                    )
            # if potential decreases, return step-size
            else:
                if do_logging:
                    _del_log()
                    print(f"Armijo line search j={j}: success", flush=True)
                return rho_j

    def _Armijo_potential(self, vec: np.ndarray) -> float:
        """Auxiliary method implementing the potential function which is to be
        minimized in the line search. Currently it uses the least-squares-potential.

        Parameters:
            vec: Vector for which the potential should be computed

        Returns:
            Value of potential.

        """
        return float(np.dot(vec, vec) / 2)

    def _newton_iterations(
        self,
        assembler: Callable[[Optional[np.ndarray]], tuple[sps.spmatrix, np.ndarray]],
        global_prolongation: sps.spmatrix,
        do_logging: bool,
        preprocessor: Optional[
            Callable[
                [sps.spmatrix, np.ndarray, sps.spmatrix],
                tuple[sps.spmatrix, np.ndarray],
            ]
        ] = None,
    ) -> tuple[bool, int]:
        """Performs standard Newton iterations using the matrix and rhs-vector returned
        by ``F``, until (possibly) the L2-norm of the rhs-vector reaches the convergence
        criterion.

        Parameters:
            assembler: A callable returning the Jacobian and residual of the function
                for which the roots should be found.

                The callable must be such that the input vector ``X`` is
                optional (None), which makes the AD-system chose values stored as
                ``ITERATE``.

                It further more must return a tuple containing its Jacobian matrix and
                the negative residual vector.

                The user must ensure that the Jacobian is invertible.
            global_prolongation: The prolongation matrix which maps the variables
                in this Newton method to the global DOF vector in the AD system.
            do_logging: Flag to print status logs in the console.
            preprocessor: ``default=None``

                An optional callable to pre-process the matrix and right-hand side
                of the linearized system returned by ``F``.

                This is called directly before each solve-command for the linear system

        Returns:
            A 2-tuple containing a bool and an integer, representing the success-
            indicator and the final number of iteration performed.

        """

        success: bool = False
        iter_final: int = 0
        ads = self._MIX.AD.system
        if self.use_armijo:
            logging_end = "\n"
        else:
            logging_end = ""

        A, b = assembler()
        self.cond_start = np.linalg.cond(A.todense())

        # if residual is already small enough
        if np.linalg.norm(b) <= self.flash_tolerance:
            if do_logging:
                print("Newton iteration 0: success", flush=True)
            success = True
        else:
            for i in range(1, self.max_iter_flash + 1):
                if do_logging:
                    # if self.use_armijo:
                    #     print("", end="\n")
                    _del_log()
                    print(
                        f"Newton iteration {i}: residual norm = {np.linalg.norm(b)}",
                        end=logging_end,
                        flush=True,
                    )

                if preprocessor:
                    A, b = preprocessor(A, b, global_prolongation)

                dx = pypardiso.spsolve(A, b)
                DX = self.newton_update_chop * global_prolongation * dx

                if self.use_armijo:
                    # get step size using Armijo line search
                    step_size = self._Armijo_line_search(DX, assembler, do_logging)
                    DX = step_size * DX

                ads.set_variable_values(
                    DX,
                    to_iterate=True,
                    additive=True,
                )

                self._update_reg()
                A, b = assembler()

                # in case of convergence
                if np.linalg.norm(b) <= self.flash_tolerance:
                    # counting necessary number of iterations
                    iter_final = i + 1  # shift since range() starts with zero
                    if do_logging:
                        if not self.use_armijo:
                            _del_log()
                        print(f"\nNewton iteration {iter_final}: success", flush=True)
                    success = True
                    break
        self.cond_end = np.linalg.cond(A.todense())
        return success, iter_final

    ### Saturation evaluation methods --------------------------------------------------

    def _single_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """If only one phase is present, we assume it occupies the whole pore space."""
        phase = self._MIX.reference_phase
        ads = self._MIX.AD.system
        values = np.ones(ads.mdg.num_subdomain_cells())
        ads.set_variable_values(
            values,
            variables=[phase.saturation.name],
            to_iterate=True,
            to_state=copy_to_state,
        )

    def _2phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        In the case of 2 phases, the evaluation is straight forward:

            ``s_i = 1 / (1 + y_j / (1 - y_j) * rho_i / rho_j) , i != j``.

        """
        # get reference to phases
        phase1, phase2 = (phase for phase in self._MIX.phases)
        ads = self._MIX.AD.system
        # get phase molar fraction values
        y1 = ads.get_variable_values(
            variables=[phase1.fraction.name], from_iterate=True
        )
        y2 = ads.get_variable_values(
            variables=[phase2.fraction.name], from_iterate=True
        )
        p = ads.get_variable_values(variables=[self._MIX.AD.p.name], from_iterate=True)
        T = ads.get_variable_values(variables=[self._MIX.AD.T.name], from_iterate=True)
        X1 = [
            phase1.normalized_fraction_of_component(comp).evaluate(ads).val
            for comp in self._MIX.components
        ]
        X2 = [
            phase2.normalized_fraction_of_component(comp).evaluate(ads).val
            for comp in self._MIX.components
        ]

        # get density values for given pressure and enthalpy
        rho1 = phase1.density(p, T, *X1)
        if isinstance(rho1, pp.ad.Ad_array):
            rho1 = rho1.val
        rho2 = phase2.density(p, T, *X2)
        if isinstance(rho2, pp.ad.Ad_array):
            rho2 = rho2.val

        # allocate saturations, size must be the same
        s1 = np.zeros(y1.size)
        s2 = np.zeros(y1.size)

        phase1_saturated = np.isclose(y1, 1.0, rtol=0.0, atol=self.eps)
        phase2_saturated = np.isclose(y2, 1.0, rtol=0.0, atol=self.eps)

        # calculate only non-saturated cells to avoid division by zero
        # set saturated or "vanishing" cells explicitly to 1., or 0. respectively
        idx = np.logical_not(phase2_saturated)
        y2_idx = y2[idx]
        rho1_idx = rho1[idx]
        rho2_idx = rho2[idx]
        s1[idx] = 1.0 / (1.0 + y2_idx / (1.0 - y2_idx) * rho1_idx / rho2_idx)
        s1[phase1_saturated] = 1.0
        s1[phase2_saturated] = 0.0

        idx = np.logical_not(phase1_saturated)
        y1_idx = y1[idx]
        rho1_idx = rho1[idx]
        rho2_idx = rho2[idx]
        s2[idx] = 1.0 / (1.0 + y1_idx / (1.0 - y1_idx) * rho2_idx / rho1_idx)
        s2[phase1_saturated] = 0.0
        s2[phase2_saturated] = 1.0

        # write values to AD system
        ads.set_variable_values(
            s1,
            variables=[phase1.saturation.name],
            to_iterate=True,
            to_state=copy_to_state,
        )
        ads.set_variable_values(
            s2,
            variables=[phase2.saturation.name],
            to_iterate=True,
            to_state=copy_to_state,
        )

    def _multi_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        Valid for compositions with at least 3 phases.

        In this case a linear system has to be solved for each multiphase cell.

        It holds for all i = 1... m, where m is the number of phases:

            ``1 = sum_{j != i} (1 + rho_j / rho_i * chi_i / (1 - chi_i)) s_j``.

        """
        ads = self._MIX.AD.system
        nc = ads.mdg.num_subdomain_cells()
        # molar fractions per phase
        y = [
            ads.get_variable_values(variables=[phase.fraction.name], from_iterate=True)
            for phase in self._MIX.phases
        ]
        p = ads.get_variable_values(variables=[self._MIX.AD.p.name], from_iterate=True)
        T = ads.get_variable_values(variables=[self._MIX.AD.T.name], from_iterate=True)
        X = [
            [
                phase.normalized_fraction_of_component(comp).evaluate(ads).val
                for comp in self._MIX.components
            ]
            for phase in self._MIX.phases
        ]
        # densities per phase
        rho = list()
        for x, phase in zip(X, self._MIX.phases):
            rho_e = phase.density(p, T, *x)
            if isinstance(rho_e, pp.ad.Ad_array):
                rho_e = rho_e.val
            rho.append(rho_e)

        mat_per_eq = list()

        # list of indicators per phase, where the phase is fully saturated
        saturated = list()
        # where one phase is saturated, the other vanish
        vanished = [np.zeros(nc, dtype=bool) for _ in self._MIX.phases]

        for i in range(self._MIX.num_phases):
            # get the DOFS where one phase is fully saturated
            # TODO check sensitivity of this
            saturated_i = y[i] == 1.0
            saturated.append(saturated_i)

            # store information that other phases vanish at these DOFs
            for j in range(self._MIX.num_phases):
                if j == i:
                    # a phase can not vanish and be saturated at the same time
                    continue
                else:
                    # where phase i is saturated, phase j vanishes
                    # Use OR to accumulate the bools per i-loop without overwriting
                    vanished[j] = np.logical_or(vanished[j], saturated_i)

        # indicator which DOFs are saturated for the vector of stacked saturations
        saturated = np.hstack(saturated)
        # indicator which DOFs vanish
        vanished = np.hstack(vanished)
        # all other DOFs are in multiphase regions
        multiphase = np.logical_not(np.logical_or(saturated, vanished))

        # construct the matrix for saturation flash
        # first loop, per block row (equation per phase)
        for i in range(self._MIX.num_phases):
            mats = list()
            # second loop, per block column (block per phase per equation)
            for j in range(self._MIX.num_phases):
                # diagonal values are zero
                # This matrix is just a placeholder
                if i == j:
                    mats.append(sps.diags([np.zeros(nc)]))
                # diagonals of blocks which are not on the main diagonal, are non-zero
                else:
                    denominator = 1 - y[i]
                    # to avoid a division by zero error, we set it to one
                    # this is arbitrary, but respective matrix entries are be sliced out
                    # since they correspond to cells where one phase is saturated,
                    # i.e. the respective saturation is 1., the other 0.
                    denominator[denominator == 0.0] = 1.0
                    d = 1.0 + rho[j] / rho[i] * y[i] / denominator

                    mats.append(sps.diags([d]))

            # rectangular matrix per equation
            mat_per_eq.append(np.hstack(mats))

        # Stack matrices per equation on each other
        # This matrix corresponds to the vector of stacked saturations per phase
        mat = np.vstack(mat_per_eq)
        # TODO permute DOFS to get a block diagonal matrix.
        # This one has a large band width
        mat = sps.csr_matrix(mat)

        # projection matrix to DOFs in multiphase region
        # start with identity in CSR format
        projection: sps.spmatrix = sps.diags([np.ones(len(multiphase))]).tocsr()
        # slice image of canonical projection out of identity
        projection = projection[multiphase]
        projection_transposed = projection.transpose()

        # get sliced system
        rhs = projection * np.ones(nc * self._MIX.num_phases)
        mat = projection * mat * projection_transposed

        s = pypardiso.spsolve(mat.tocsr(), rhs)

        # prolongate the values from the multiphase region to global DOFs
        saturations = projection_transposed * s
        # set values where phases are saturated or have vanished
        saturations[saturated] = 1.0
        saturations[vanished] = 0.0

        # distribute results to the saturation variables
        for i, phase in enumerate(self._MIX.phases):
            vals = saturations[i * nc : (i + 1) * nc]
            ads.set_variable_values(
                vals,
                variables=[phase.saturation.name],
                to_iterate=True,
                to_state=copy_to_state,
            )
