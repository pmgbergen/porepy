"""This module contains functionality to solve the equilibrium problem numerically
(flash)."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .composition import Composition
from .phase import Phase

__all__ = ["Flash"]


class Flash:
    """A class containing various methods for the isenthalpic and isothermal flash.

    Notes:
        - Two flash procedures are implemented: p-T flash and p-h flash.
        - Two numerical procedures are implemented (see below references).
        - Equilibrium calculations can be done cell-wise.
          I.e. the computations can be done smarter or parallelized.
          This is not yet exploited.

    While the molar fractions are the actual unknowns in the flash procedure,
    the saturation values can be computed once the equilibrium converges using
    :meth:`evaluate_saturations`.

    The specific enthalpy can be evaluated directly after a p-T flash using
    :meth:`evaluate_specific_enthalpy`.

    References:
        [1]: `Lauser et al. (2011) <https://doi.org/10.1016/j.advwatres.2011.04.021>`_
        [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_

    Parameters:
        composition: An **initialized** composition class representing the
            modelled mixture.

    """

    def __init__(self, composition: Composition) -> None:

        self._C: Composition = composition
        """The composition class passed at instantiation"""

        self._max_history: int = 100
        """Maximal number of flash history entries (FiFo)."""

        self._ss_min: pp.ad.Operator = pp.ad.SemiSmoothMin()
        """An operator representing the semi-smooth min function in AD."""

        self._y_R: pp.ad.Operator = composition.get_reference_phase_fraction_by_unity()
        """An operator representing the reference phase fraction by unity."""

        self._V_name: str = "NPIPM_var_V"
        """Name of the variable ``V`` in the NPIPM."""

        self._W_name: str = "NPIPM_var_W"
        """Name of the variable ``W`` in the NPIPM."""

        self._nu_name: str = "NPIPM_var_nu"
        """Name of the variable ``nu`` in the NPIPM."""

        self._V_of_phase: dict[Phase, pp.ad.MergedVariable] = dict()
        """A dictionary containing the NPIPM extension variable ``V`` for each phase."""

        self._W_of_phase: dict[Phase, pp.ad.MergedVariable] = dict()
        """A dictionary containing the NPIPM extension variable ``W`` for each phase."""

        self._nu: pp.ad.MergedVariable = self._C.ad_system.create_variable(
            self._nu_name
        )
        """Variable ``nu`` representing the IPM parameter."""

        ### setting if flash equations
        cc_eqn = self._set_complementary_conditions()
        npipm_eqn, npipm_vars = self._set_npipm_eqn_vars()

        ### PUBLIC

        self.flash_history: list[dict[str, Any]] = list()
        """Contains chronologically stored information about performed flash procedures.
        """

        self.flash_tolerance: float = 1e-7
        """Convergence criterion for the flash algorithm."""

        self.max_iter_flash: int = 100
        """Maximal number of iterations for the flash algorithms."""

        self.eps: float = 1e-12
        """Small number to define the numerical zero."""

        self.npipm_parameters: dict[str, float] = {
            "eta": 0.5,
            "u": 1,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the NPIPM.

        Values can be set directly by modifying the values of this dictionary.

        See Also:
            `Vu et al. (2021), Section 6.
            <https://doi.org/10.1016/j.matcom.2021.07.015>`_

        """

        self.armijo_parameters: dict[str, float] = {
            "kappa": 0.4,
            "rho": 0.99,
            "j_max": 20,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the Armijo line-search.

        Values can be set directly by modifying the values of this dictionary.

        """

        self.complementary_equations: list[str] = cc_eqn
        """A list of strings representing names of complementary conditions (KKT)
        for the unified flash problem."""

        self.algorithmic_variables: list[str] = npipm_vars
        """A list containing algorithmic variables in the AD framework, which are
        neither physical nor must they be used in extended problems e.g., flow.

        Warning:
            This list is filled only upon initialization.

        """

        self.algorithmic_equations: list[str] = npipm_eqn
        """A list containing equations in the AD framework, which are
        neither physical nor must they be used in extended problems e.g., flow.

        They result from specific algorithms chosen for the flash.

        Warning:
            This list is filled only upon initialization.

        """

    def _set_complementary_conditions(self) -> list[str]:
        """Auxiliary function for the constructor to set equations representing
        the complementary conditions.

        Returns:
            Names of the set equations.

        """
        equations = dict()  # storage of additional flash equations
        cc_eqn: list[str] = []  # storage for complementary constraints

        ## Semi-smooth complementary conditions per phase
        for phase in self._C.phases:

            # name of the equation
            name = f"flash_KKT_{phase.name}"

            if phase == self._C.reference_phase:
                # skip the reference phase KKT condition if only one component,
                # otherwise the system is overdetermined.
                if self._C.num_components == 1:
                    continue

                # the reference phase fraction is replaced by unity
                _, lagrange = self._C.get_complementary_condition_for(
                    self._C.reference_phase
                )
                constraint = self._C.get_reference_phase_fraction_by_unity()
            # for other phases, 'constraint' is the phase fraction
            else:
                constraint, lagrange = self._C.get_complementary_condition_for(phase)

            # instantiate semi-smooth min in AD form with skewing factor
            equation = self._ss_min(constraint, lagrange)
            equations.update({name: equation})

            # store name for it to be excluded in NPIPM
            cc_eqn.append(name)

        # adding flash-specific equations to AD system
        # every equation in the unified flash is a cell-wise scalar equation
        image_info = dict()
        for sd in self._C.ad_system.dof_manager.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        for name, equ in equations.items():
            self._C.ad_system.set_equation(name, equ, num_equ_per_dof=image_info)

        return cc_eqn

    def _set_npipm_eqn_vars(self) -> list[str]:
        """Auxiliary function for the constructor to set equations and variables
        for the NPIPM method.

        Returns:
            A 2-tuple containing a list of equations names and a list of variable names
            introduced here.

        """
        npipm_eqn: list[str] = []
        npipm_vars: list[str] = []

        # every equation in the unified flash is a cell-wise scalar equation
        image_info = dict()
        for sd in self._C.ad_system.dof_manager.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})

        ### NPIPM variables and equations
        for phase in self._C.phases:

            # create V_e
            name = f"{self._V_name}_{phase.name}"
            V_e = self._C.ad_system.create_variable(name)
            npipm_vars.append(name)
            self._V_of_phase[phase] = V_e

            # create W_e
            name = f"{self._W_name}_{phase.name}"
            W_e = self._C.ad_system.create_variable(name)
            npipm_vars.append(name)
            self._W_of_phase[phase] = W_e

            # V_e extension equation, create and store
            v_extension = phase.fraction - V_e
            name = f"NPIPM_V_{phase.name}"
            self._C.ad_system.set_equation(
                name, v_extension, num_equ_per_dof=image_info
            )
            npipm_eqn.append(name)

            # W_e extension equation, create and store
            w_extension = self._C.get_composition_unity_for(phase) - W_e
            name = f"NPIPM_W_{phase.name}"
            self._C.ad_system.set_equation(
                name, w_extension, num_equ_per_dof=image_info
            )
            npipm_eqn.append(name)

            # V-W-nu coupling for this phase
            coupling = V_e * W_e - self._nu
            name = f"NPIPM_coupling_{phase.name}"
            self._C.ad_system.set_equation(name, coupling, num_equ_per_dof=image_info)
            npipm_eqn.append(name)

        # NPIPM parameter equation
        eta = pp.ad.Scalar(self.npipm_parameters["eta"])
        coeff = pp.ad.Scalar(self.npipm_parameters["u"] / self._C.num_phases**2)
        neg = pp.ad.SemiSmoothNegative()
        pos = pp.ad.SemiSmoothPositive()
        dot = pp.ad.ScalarProduct()

        phase_parts = list()
        for phase in self._C.phases:
            v_e = self._V_of_phase[phase]
            w_e = self._W_of_phase[phase]

            phase_parts.append(
                dot(neg(v_e), neg(v_e))
                + dot(neg(w_e), neg(w_e))
                + coeff * pos(dot(v_e, w_e)) * pos(dot(v_e, w_e))
            )

        equation = eta * self._nu + self._nu * self._nu + sum(phase_parts) / 2
        self._C.ad_system.set_equation(
            "NPIPM_param", equation, num_equ_per_dof=image_info
        )
        npipm_eqn.append("NPIPM_param")

        return npipm_eqn, npipm_vars

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
        all_vars = [block[1] for block in self._C.ad_system.dof_manager.block_dof]
        print("Variables:")
        print(list(sorted(set(vars), key=lambda x: all_vars.index(x))))

    def print_ph_system(self, print_dense: bool = False):
        print("---")
        print("Flash Variables:")
        self.print_ordered_vars(self._C.ph_subsystem["primary_vars"])
        for equ in self._C.ph_subsystem["equations"]:
            A, b = self._C.ad_system.assemble_subsystem(
                equ, self._C.ph_subsystem["primary_vars"]
            )
            print("---")
            print(equ)
            self.print_system(A, b, print_dense)
        print("---")

    def print_pT_system(self, print_dense: bool = False):
        print("---")
        print("Flash Variables:")
        self.print_ordered_vars(self._C.pT_subsystem["primary_vars"])
        for equ in self._C.pT_subsystem["equations"]:
            A, b = self._C.ad_system.assemble_subsystem(
                equ, self._C.pT_subsystem["primary_vars"]
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

    def _history_entry(
        self,
        flash: str = "isenthalpic",
        method: str = "standard",
        iterations: int = 0,
        success: bool = False,
        variables: list[str] = list(),
        equations: list[str] = list(),
        **kwargs,
    ) -> None:
        """Makes an entry in the flash history."""

        self.flash_history.append(
            {
                "flash": flash,
                "method": method,
                "iterations": iterations,
                "success": success,
                "variables": str(variables),
                "equations": str(equations),
                "other": str(kwargs),
            }
        )
        if len(self.flash_history) > self._max_history:
            self.flash_history.pop(0)

    ### Flash methods ------------------------------------------------------------------

    def flash(
        self,
        flash_type: Literal["isothermal", "isenthalpic"] = "isothermal",
        method: Literal["newton-min", "npipm"] = "newton-min",
        initial_guess: Literal["iterate", "uniform"] | str = "iterate",
        copy_to_state: bool = False,
    ) -> bool:
        """Performs a flash procedure based on the arguments.

        Parameters:
            flash_type: ``default='isothermal'``

                A string representing the chosen flash type:

                - ``'isothermal'``: The composition is determined based on given
                  temperature, pressure and feed fractions per component.
                  Enthalpy is not considered and can be evaluated upon success using
                  :meth:`evaluate_specific_enthalpy`.
                - ``'isenthalpic'``: The composition **and** temperature are determined
                  based on given pressure and enthalpy.
                  Compared to the isothermal flash, the temperature in the isenthalpic
                  flash is an additional variable and an enthalpy constraint is
                  introduced into the system as an additional equation.

            method: ``default='newton-min'``

                A string indicating the chosen algorithm:

                - ``'newton-min'``: A semi-smooth Newton method, where the KKT-
                  conditions and and their derivatives are used using a semi-smooth
                  min function [1].
                - ``'npipm'``: A Non-Parametric Interior Point Method [2].

            initial_guess: ``default='iterate'``

                Strategy for choosing the initial guess:

                - ``'iterate'``: values from ITERATE or STATE, if ITERATE not existent.
                - ``'uniform'``: uniform fractions adding up to 1 are used as initial
                  guesses.

            copy_to_state: Copies the values to the STATE of the AD variables,
                additionally to ITERATE.

                Note:
                    If not successful, the ITERATE will **not** be copied to the STATE,
                    even if flagged ``True`` by ``copy_to_state``.

        Raises:
            ValueError: If either `flash_type`, `method` or `initial_guess` are
                unsupported keywords.

        Returns:
            A bool indicating if flash was successful or not.

        """
        success = False

        if flash_type == "isothermal":
            subsystem = self._C.pT_subsystem
        elif flash_type == "isenthalpic":
            subsystem = self._C.ph_subsystem
        else:
            raise ValueError(f"Unknown flash type {flash_type}.")

        self._set_initial_guess(initial_guess)

        if method == "newton-min":
            success = self._Newton_min(subsystem)
        elif method == "npipm":
            self._set_NPIPM_initial_guess()
            success = self._NPIPM(subsystem)
        else:
            raise ValueError(f"Unknown method {method}.")

        # setting STATE to newly found solution
        if copy_to_state and success:
            var_names = subsystem["primary_vars"]
            X = self._C.ad_system.dof_manager.assemble_variable(
                variables=var_names, from_iterate=True
            )
            self._C.ad_system.dof_manager.distribute_variable(X, variables=var_names)

        return success

    def evaluate_saturations(self, copy_to_state: bool = True) -> None:
        """Evaluates the volumetric phase fractions (saturations) based on the number of
        modelled phases and the thermodynamic state of the mixture.

        To be used after any flash procedure.

        If no phases are present (e.g. before any flash procedure),
        this method does nothing.

        Parameters:
            copy_to_state: ``default=True``

                Copies the values to the STATE of the AD variable,
                additionally to ITERATE. Defaults to True.

        """
        if self._C.num_phases == 1:
            self._single_phase_saturation_evaluation(copy_to_state)
        if self._C.num_phases == 2:
            self._2phase_saturation_evaluation(copy_to_state)
        elif self._C.num_phases >= 3:
            self._multi_phase_saturation_evaluation(copy_to_state)

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
        equ_ = list()
        for phase in self._C.phases:
            equ_.append(phase.fraction * phase.specific_enthalpy(self._C.p, self._C.T))
        equ = sum(equ_)

        # if no phase present (list empty) zero is returned and enthalpy is zero
        if equ == 0:
            h = np.zeros(self._C.ad_system.dof_manager.mdg.num_subdomain_cells())
        # else evaluate this operator
        elif isinstance(equ, pp.ad.Operator):
            h = equ.evaluate(self._C.ad_system.dof_manager).val
        else:
            raise RuntimeError("Something went terribly wrong.")
        # write values in local var form
        self._C.ad_system.set_var_values(self._C.h_name, h, copy_to_state)

    def post_process_fractions(self, copy_to_state: bool = True) -> None:
        """Evaluates the fraction of the reference phase and removes numerical artifacts
        from all fractional values.

        Fractional values are supposed to be between 0 and 1 after any valid flash
        result. Molar phase fractions and phase compositions are post-processed
        respectively.

        Note:
            A components overall fraction (feed fraction) is not changed!
            The amount of moles belonging to a certain component are not supposed
            to change at all during a flash procedure without reactions.

        Parameters:
            copy_to_state: ``default=True``

                Copies the values to the STATE of the AD variable,
                additionally to ITERATE. Defaults to True.

        """

        # evaluate reference phase fractions
        y_R = self._y_R.evaluate(self._C.ad_system.dof_manager).val
        self._C.ad_system.set_var_values(
            self._C.reference_phase.fraction_name, y_R, copy_to_state
        )

        for phase_e in self._C.phases:
            # remove numerical artifacts on phase fractions y
            y_e = self._C.ad_system.get_var_values(phase_e.fraction_name)
            y_e[y_e < 0.0] = 0.0
            y_e[y_e > 1.0] = 1.0
            self._C.ad_system.set_var_values(phase_e.fraction_name, y_e, copy_to_state)

            # remove numerical artifacts in phase compositions
            for comp_c in phase_e:
                xi_ce = self._C.ad_system.get_var_values(
                    phase_e.fraction_of_component_name(comp_c)
                )
                xi_ce[xi_ce < 0.0] = 0.0
                xi_ce[xi_ce > 1.0] = 1.0
                # write values
                self._C.ad_system.set_var_values(
                    phase_e.fraction_of_component_name(comp_c),
                    xi_ce,
                    copy_to_state,
                )

    ### Initial guess strategies -------------------------------------------------------

    def _set_NPIPM_initial_guess(self) -> None:
        """Sets the initial guesses for ``V_e``, ``W_e`` and ``nu`` according to
        Vu (2021), section 3.3."""
        # initial guess for nu is constructed from V and W
        V_mat: list[np.ndarray] = list()
        W_mat: list[np.ndarray] = list()

        ad_system = self._C.ad_system

        for phase in self._C.phases:
            # initial value for V_e
            v_name = self._V_name + phase.name
            val_v = phase.fraction.evaluate(ad_system.dof_manager).val
            ad_system.set_var_values(v_name, val_v, True)
            # initial value for W_e
            w_name = self._W_name + phase.name
            val_w = (
                self._C.get_composition_unity_for(phase)
                .evaluate(ad_system.dof_manager)
                .val
            )
            ad_system.set_var_values(w_name, val_w, True)
            # store value for initial guess for nu
            V_mat.append(val_v)
            W_mat.append(val_w)

        # initial guess for nu is cell-wise scalar product between concatenated V and W
        # for each phase
        V = np.ndarray(V_mat).T  # num_cells X num_phases
        W = np.ndarray(W_mat)  # num_phases X num_cells
        # the diagonal of the product of above returns the cell-wise scalar product
        # TODO can this be optimized using a for loop over diagonal elements of product?
        nu_mat = V * W
        nu = np.diag(nu_mat)
        nu = nu / self._C.num_phases

        ad_system.set_var_values(self._nu_name, nu, True)

    def _set_initial_guess(self, initial_guess: str) -> None:
        """Auxillary function to set the initial values for phase fractions,
        phase compositions and temperature, based on the chosen strategy."""

        ad_system = self._C.ad_system

        if initial_guess == "iterate":
            # DofManager takes by default values from ITERATE, than from STATE if not found
            pass
        elif initial_guess == "uniform":
            nc = ad_system.dof_manager.mdg.num_subdomain_cells()
            # uniform values for phase fraction
            val_phases = 1.0 / self._C.num_phases
            for phase in self._C.phases:
                ad_system.set_var_values(phase.fraction_name, val_phases * np.ones(nc))
                # uniform values for composition of this phase
                val = 1.0 / phase.num_components
                for component in self._C.components:
                    ad_system.set_var_values(
                        phase.fraction_of_component_name(component),
                        val * np.ones(nc),
                    )
        else:
            raise ValueError(f"Unknown initial-guess-strategy {initial_guess}.")

    ### Numerical methods --------------------------------------------------------------

    def _Armijo_line_search(
        self, DX: np.ndarray, equations: list[str], variables: list[str]
    ) -> float:
        """Performs the Armijo line-search for a given system of equations, variables
        and a preliminary update, using the least-square potential.

        By default, the values stored as ``ITERATE`` are used for evaluation.

        Parameters:
            DX: Preliminary update to solution vector.
            equations: Names of equations in system for which the search is requested.
            var_names: Names of variables in the system.

        Raises:
            RuntimeError: If line-search in defined interval does not yield any results.

        Returns:
            The step-size resulting from the line-search algorithm.

        """
        # get relevant parameters
        kappa = self.armijo_parameters["kappa"]
        rho = self.armijo_parameters["rho"]
        j_max = self.armijo_parameters["j_max"]

        ad_system = self._C.ad_system

        # get starting point from current ITERATE state at iteration k
        _, b_k = ad_system.assemble_subsystem(equations, variables)
        b_k_pot = np.dot(b_k, b_k) / 2  # -b0 since above method returns rhs
        X_k = ad_system.dof_manager.assemble_variable(from_iterate=True)

        _, b_1 = ad_system.assemble_subsystem(
            equations, variables, state=(X_k + rho * DX)
        )
        # start with first step size. If sufficient, return rho
        if np.dot(b_1, b_1) <= (1 - 2 * kappa * rho) * b_k_pot:
            return rho
        else:
            # if maximal line-search interval defined, use for-loop
            if j_max:
                for j in range(2, j_max + 1):
                    # new step-size
                    rho_j = rho**j

                    # compute system state at preliminary step-size
                    _, b_j = ad_system.assemble_subsystem(
                        equations, variables, state=(X_k + rho_j * DX)
                    )

                    # check potential and return if reduced.
                    if np.dot(b_j, b_j) <= (1 - 2 * kappa * rho_j) * b_k_pot:
                        return rho_j

                # if for-loop did not yield any results, raise error
                raise RuntimeError(
                    f"Armijo line-search did not yield results after {j_max} steps."
                )
            # if no j_max is defined, use while loop
            # NOTE: If system is wrong in some sense, this might possible never finish.
            else:
                # prepare for while loop
                rho_j = rho * rho
                # compute system state at preliminary step-size
                _, b_j = ad_system.assemble_subsystem(
                    equations, variables, state=(X_k + rho_j * DX)
                )

                # while potential not decreasing, compute next step-size
                while np.dot(b_j, b_j) > (1 - 2 * kappa * rho_j) * b_k_pot:
                    # next power of step-size
                    rho_j *= rho
                    _, b_j = ad_system.assemble_subsystem(
                        equations, variables, state=(X_k + rho_j * DX)
                    )
                # if potential decreases, return step-size
                else:
                    return rho_j

    ### Saturation evaluation methods --------------------------------------------------

    def _single_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """If only one phase is present, we assume it occupies the whole pore space."""
        phase = self._C.reference_phase
        values = np.ones(self._C.ad_system.dof_manager.mdg.num_subdomain_cells())
        self._C.ad_system.set_var_values(phase.saturation_name, values, copy_to_state)

    def _2phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        In the case of 2 phases, the evaluation is straight forward:

            ``s_i = 1 / (1 + y_j / (1 - y_j) * rho_i / rho_j) , i != j``.

        """
        # get reference to phases
        phase1, phase2 = (phase for phase in self._C.phases)
        # shortening the name space
        dm = self._C.ad_system.dof_manager
        # get phase molar fraction values
        y1 = self._C.ad_system.get_var_values(phase1.fraction_name)
        y2 = self._C.ad_system.get_var_values(phase2.fraction_name)

        # get density values for given pressure and enthalpy
        rho1 = phase1.density(self._C.p, self._C.T).evaluate(dm)
        if isinstance(rho1, pp.ad.Ad_array):
            rho1 = rho1.val
        rho2 = phase2.density(self._C.p, self._C.T).evaluate(dm)
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
        self._C.ad_system.set_var_values(phase1.saturation_name, s1, copy_to_state)
        self._C.ad_system.set_var_values(phase2.saturation_name, s2, copy_to_state)

    def _multi_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        Valid for compositions with at least 3 phases.

        In this case a linear system has to be solved for each multiphase cell.

        It holds for all i = 1... m, where m is the number of phases:

            ``1 = sum_{j != i} (1 + rho_j / rho_i * chi_i / (1 - chi_i)) s_j``.

        """
        # shortening name space
        dm = self._C.ad_system.dof_manager
        nc = dm.mdg.num_subdomain_cells()
        # molar fractions per phase
        y = [
            self._C.ad_system.get_var_values(phase.saturation_name)
            for phase in self._C.phases
        ]
        # densities per phase
        rho = list()
        for phase in self._C.phases:
            rho_e = phase.density(self._C.p, self._C.T).evaluate(dm)
            if isinstance(rho_e, pp.ad.Ad_array):
                rho_e = rho_e.val
            rho.append(rho_e)

        mat_per_eq = list()

        # list of indicators per phase, where the phase is fully saturated
        saturated = list()
        # where one phase is saturated, the other vanish
        vanished = [np.zeros(nc, dtype=bool) for _ in self._C.phases]

        for i in range(self._C.num_phases):
            # get the DOFS where one phase is fully saturated
            # TODO check sensitivity of this
            saturated_i = y[i] == 1.0
            saturated.append(saturated_i)

            # store information that other phases vanish at these DOFs
            for j in range(self._C.num_phases):
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
        for i in range(self._C.num_phases):
            mats = list()
            # second loop, per block column (block per phase per equation)
            for j in range(self._C.num_phases):
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
        rhs = projection * np.ones(nc * self._C.num_phases)
        mat = projection * mat * projection_transposed

        s = sps.linalg.spsolve(mat.tocsr(), rhs)

        # prolongate the values from the multiphase region to global DOFs
        saturations = projection_transposed * s
        # set values where phases are saturated or have vanished
        saturations[saturated] = 1.0
        saturations[vanished] = 0.0

        # distribute results to the saturation variables
        for i, phase in enumerate(self._C.phases):
            vals = saturations[i * nc : (i + 1) * nc]
            self._C.ad_system.set_var_values(phase.saturation_name, vals, copy_to_state)
