"""This module contains functionality to solve the equilibrium problem numerically
(flash)."""
from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .composition import Composition
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


class Flash:
    """A class containing various methods for the isenthalpic and isothermal flash.

    Notes:
        - Two flash procedures are implemented: p-T flash and p-h flash.
        - Two numerical procedures are implemented (see below references).
        - Equilibrium calculations can be done cell-wise.
          I.e. the computations can be done smarter or parallelized.
          This is not yet exploited.

    Warning:
        The flash methods here are focused on vapor liquid equilibria, assuming the
        liquid phase to be the reference phase.

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
        auxiliary_npipm: ``default=False``

            An optional flag to enlarge the NPIPM system by introducing phase-specific
            auxiliary variables and equations. This introduces the size of the system
            by ``2 * num_phases`` equations and variables and should be used carefully.

            This feature is left here to reflect the original algorithm
            introduced by Vu.
        npipm_param_as_var: ``default=True``

            An optional flag to introduce the NPIPM parameter ``nu`` as a variable
            and its update as an equation.

            If False, the update for ``nu`` is performed after each iteration and
            contributes to the right-hand side of the NPIPM system.

    """

    def __init__(
        self,
        composition: Composition,
        auxiliary_npipm: bool = False,
        npipm_param_as_var: bool = True,
    ) -> None:

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

        self._V_of_phase: dict[Phase, pp.ad.MixedDimensionalVariable] = dict()
        """A dictionary containing the NPIPM extension variable ``V`` for each phase."""

        self._W_of_phase: dict[Phase, pp.ad.MixedDimensionalVariable] = dict()
        """A dictionary containing the NPIPM extension variable ``W`` for each phase."""

        self._nu: Union[pp.ad.MixedDimensionalVariable, np.ndarray]
        """Slack variable ``nu`` representing the NPIPM parameter."""

        ### PUBLIC

        self.flash_history: list[dict[str, Any]] = list()
        """Contains chronologically stored information about performed flash procedures.
        """

        self.flash_tolerance: float = 1e-7
        """Convergence criterion for the flash algorithm. Defaults to ``1e-7``."""

        self.max_iter_flash: int = 100
        """Maximal number of iterations for the flash algorithms. Defaults to 100."""

        self.eps: float = 1e-12
        """Small number to define the numerical zero. Defaults to ``1e-12``."""

        self.use_armijo: bool = True
        """A bool indicating if an Armijo line-search should be performed after an
        update direction has been found. Defaults to True."""

        self.use_auxiliary_npipm_vars: bool = auxiliary_npipm
        """A bool indicating if the auxiliary variables ``V`` and ``W`` should
        be used in the NPIPM algorithm of Vu et al.. Passed at instantiation."""

        self.npipm_param_as_var: bool = npipm_param_as_var
        """A bool indicating if ``nu`` in the NPIPM is introduced as a variable.
        Passed at instantiation."""

        self.newton_update_chop: float = 1.0
        """A number in ``[0, 1]`` to scale the Newton update ``dx`` resulting from
        solving the linearized system. Defaults to 1."""

        self.npipm_parameters: dict[str, float] = {
            "eta": 0.5,
            "u": 1,
            "kappa": 1.0,
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
            "j_max": 150,
            "return_max": False,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the Armijo line-search.

        Values can be set directly by modifying the values of this dictionary.

        """

        ### setting if flash equations
        self.complementary_equations: list[str] = self._set_complementary_conditions()
        """A list of strings representing names of complementary conditions (KKT)
        for the unified flash problem."""

        self._nu2 = Leaf("npipm reg")
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

    def _set_complementary_conditions(self) -> list[str]:
        """Auxiliary function for the constructor to set equations representing
        the complementary conditions.

        Returns:
            Names of the set equations.

        """
        # storage of additional flash equations
        equations: dict[str, pp.ad.Operator] = dict()
        # storage for complementary constraints
        cc_eqn: list[str] = []

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
        for name, equ in equations.items():
            equ.set_name(name)
            self._C.ad_system.set_equation(
                equ,
                grids=self._C.ad_system.mdg.subdomains(),
                equations_per_grid_entity={"cells": 1},
            )

        return cc_eqn

    def _set_npipm_eqn_vars(self) -> list[str]:
        """Auxiliary function for the constructor to set equations and variables
        for the NPIPM method.

        Returns:
            A 2-tuple containing a list of equations names and a list of variable names
            introduced here.

        """
        nc = self._C.ad_system.mdg.num_subdomain_cells()
        npipm_vars: list[str] = []
        npipm_eqn: list[str] = []

        # set initial values of zero for parameter nu
        if self.npipm_param_as_var:
            self._nu = self._C.ad_system.create_variables(
                self._nu_name, subdomains=self._C.ad_system.mdg.subdomains()
            )
            npipm_vars.append(self._nu_name)
            self._C.ad_system.set_variable_values(
                np.zeros(nc), variables=[self._nu_name], to_iterate=True, to_state=True
            )
        else:
            self._nu = np.zeros(nc)

        # Defining and setting V per phase
        for phase in self._C.phases:
            # Instantiating V as a variable, if requested
            if self.use_auxiliary_npipm_vars:
                # create V_e
                name = f"{self._V_name}_{phase.name}"
                V_e = self._C.ad_system.create_variables(
                    name, subdomains=self._C.ad_system.mdg.subdomains()
                )
                npipm_vars.append(name)
                self._C.ad_system.set_variable_values(
                    np.zeros(nc), variables=[name], to_iterate=True, to_state=True
                )
                self._V_of_phase[phase] = V_e
            # else we eliminate them by their respective definition
            else:

                if phase == self._C.reference_phase:
                    self._V_of_phase[phase] = self._y_R
                else:
                    self._V_of_phase[phase] = phase.fraction

        # Defining and setting W per phase
        for phase in self._C.phases:
            # Instantiating W as a variable, if requested
            if self.use_auxiliary_npipm_vars:
                # create W_e
                name = f"{self._W_name}_{phase.name}"
                W_e = self._C.ad_system.create_variables(
                    name, subdomains=self._C.ad_system.mdg.subdomains()
                )
                npipm_vars.append(name)
                self._C.ad_system.set_variable_values(
                    np.zeros(nc), variables=[name], to_iterate=True, to_state=True
                )
                self._W_of_phase[phase] = W_e
            # else we eliminate them by their respective definition
            else:
                self._W_of_phase[phase] = self._C.get_composition_unity_for(phase)

        # If V and W are requested as separate variables,
        # introduce the extension equations
        if self.use_auxiliary_npipm_vars:
            # V_e extension equations, create and store
            for phase in self._C.phases:
                if phase == self._C.reference_phase:
                    v_extension = self._y_R - V_e
                else:
                    v_extension = phase.fraction - V_e
                name = f"NPIPM_V_{phase.name}"
                v_extension.set_name(name)
                self._C.ad_system.set_equation(
                    v_extension,
                    grids=self._C.ad_system.mdg.subdomains(),
                    equations_per_grid_entity={"cells": 1},
                )
                npipm_eqn.append(name)

            # W_e extension equations, create and store
            for phase in self._C.phases:
                w_extension = self._C.get_composition_unity_for(phase) - W_e
                name = f"NPIPM_W_{phase.name}"
                w_extension.set_name(f"NPIPM_W_{phase.name}")
                self._C.ad_system.set_equation(
                    w_extension,
                    grids=self._C.ad_system.mdg.subdomains(),
                    equations_per_grid_entity={"cells": 1},
                )
                npipm_eqn.append(name)

        # Equations coupling the KKT with the slack variable
        # NOTE: If nu is not a variable and V and W are not used,
        # this is reduced to the KKT conditions itself.
        # I.e. they are the same equations as for the Newton-min.

        def smoother(X):
            return X / (X + 1)

        for phase in self._C.phases:
            if self.npipm_param_as_var:
                assert isinstance(self._nu, pp.ad.MixedDimensionalVariable)
                coupling = self._V_of_phase[phase] * self._W_of_phase[phase] - self._nu
                # coupling = self._nu * (
                #     smoother(self._V_of_phase[phase] / self._nu)
                #     + smoother(self._W_of_phase[phase] / self._nu)
                #     - pp.ad.Scalar(1) / self._nu
                # )
            else:
                coupling = self._V_of_phase[phase] * self._W_of_phase[phase]
            name = f"NPIPM_coupling_{phase.name}"
            coupling.set_name(name)
            self._C.ad_system.set_equation(
                coupling,
                grids=self._C.ad_system.mdg.subdomains(),
                equations_per_grid_entity={"cells": 1},
            )
            npipm_eqn.append(name)

        # NPIPM parameter equation
        if self.npipm_param_as_var:
            eta = pp.ad.Scalar(self.npipm_parameters["eta"])
            kappa = pp.ad.Scalar(self.npipm_parameters["kappa"])
            coeff = pp.ad.Scalar(self.npipm_parameters["u"] / self._C.num_phases**2)
            neg = pp.ad.SemiSmoothNegative()
            pos = pp.ad.SemiSmoothPositive()
            one = pp.ad.Scalar(1.0)

            norm_parts = list()
            dot_parts = list()
            test_parts = list()
            for phase in self._C.phases:
                v_e = self._V_of_phase[phase]
                w_e = self._W_of_phase[phase]

                dot_parts.append(v_e * w_e)
                norm_parts.append(neg(v_e) * neg(v_e) + neg(w_e) * neg(w_e))

                test_parts.append(
                    smoother(neg(w_e) / (self._nu2 + one))
                    + smoother(neg(v_e) / (self._nu + one))
                )

            dot_part = pos(sum(dot_parts))
            dot_part *= dot_part * coeff

            equation = (
                eta * self._nu
                + self._nu * self._nu
                + (sum(norm_parts) + dot_part) / 2
                + sum(test_parts) * kappa * self._nu2
            )
            equation.set_name("NPIPM_param")
            self._C.ad_system.set_equation(
                equation,
                grids=self._C.ad_system.mdg.subdomains(),
                equations_per_grid_entity={"cells": 1},
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
        all_vars = [var.name for var in self._C.ad_system._variable_numbers]
        print(list(sorted(set(vars), key=lambda x: all_vars.index(x))), flush=True)

    def print_ph_system(self, print_dense: bool = False):
        print("---")
        print("Flash Variables:")
        self.print_ordered_vars(self._C.ph_subsystem["primary_vars"])
        for equ in self._C.ph_subsystem["equations"]:
            A, b = self._C.ad_system.assemble_subsystem(
                [equ], self._C.ph_subsystem["primary_vars"]
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
                [equ], self._C.pT_subsystem["primary_vars"]
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
        sys = self._C.ad_system
        if from_iterate:
            print("ITERATE:")
        else:
            print("STATE:")
        print(filler)
        print("Pressure:")
        print(
            "\t"
            + str(
                sys.get_variable_values(
                    variables=[self._C.p_name], from_iterate=from_iterate
                )
            )
        )
        print("Temperature:")
        print(
            "\t"
            + str(
                sys.get_variable_values(
                    variables=[self._C.T_name], from_iterate=from_iterate
                )
            )
        )
        print("Enthalpy:")
        print(
            "\t"
            + str(
                sys.get_variable_values(
                    variables=[self._C.h_name], from_iterate=from_iterate
                )
            )
        )
        print("Feed fractions:")
        for component in self._C.components:
            print(f"{component.fraction_name}: ")
            print(
                "\t"
                + str(
                    sys.get_variable_values(
                        variables=[component.fraction_name],
                        from_iterate=from_iterate,
                    )
                )
            )
        print(filler)
        print("Phase fractions:")
        for phase in self._C.phases:
            print(f"{phase.name}: ")
            print(
                "\t"
                + str(
                    sys.get_variable_values(
                        variables=[phase.fraction_name], from_iterate=from_iterate
                    )
                )
            )
        print("Saturations:")
        for phase in self._C.phases:
            print(f"{phase.name}: ")
            print(
                "\t"
                + str(
                    sys.get_variable_values(
                        variables=[phase.saturation_name], from_iterate=from_iterate
                    )
                )
            )
        print(filler)
        print("Composition:")
        for phase in self._C.phases:
            print(f"{phase.name}: ")
            for component in self._C.components:
                print(f"{phase.fraction_of_component_name(component)}: ")
                print(
                    "\t"
                    + str(
                        sys.get_variable_values(
                            variables=[phase.fraction_of_component_name(component)],
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
                    sys.get_variable_values(
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
        flash_type: Literal["isothermal", "isenthalpic"] = "isothermal",
        method: Literal["newton-min", "npipm"] = "newton-min",
        initial_guess: Literal["iterate", "feed", "uniform", "rachford_rice"]
        | str = "iterate",
        copy_to_state: bool = False,
        do_logging: bool = False,
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
            ValueError: If either `flash_type`, `method` or `initial_guess` are
                unsupported keywords.

        Returns:
            A bool indicating if flash was successful or not.

        """
        success = False

        if flash_type == "isothermal":
            var_names = self._C.pT_subsystem["primary_vars"]
            equations = self._C.pT_subsystem["equations"]
        elif flash_type == "isenthalpic":
            var_names = self._C.ph_subsystem["primary_vars"]
            equations = self._C.ph_subsystem["equations"]
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
            if do_logging:
                print(f"Variables: {var_names}")
                print(f"Equations: {equations}")
                print("+++", flush=True)
            success = self._Newton_min(flash_type, do_logging)
        elif method == "npipm":
            if do_logging:
                print("Setting initial NPIPM variables.")
                print(f"Variables: {var_names + self.npipm_variables}")
                print(f"Equations: {equations + self.npipm_equations}")
                print("+++", flush=True)
            self._set_NPIPM_initial_guess()
            success = self._NPIPM(flash_type, do_logging)
        else:
            raise ValueError(f"Unknown method {method}.")

        # setting STATE to newly found solution
        if copy_to_state and success:
            X = self._C.ad_system.get_variable_values(
                variables=var_names, from_iterate=True
            )
            self._C.ad_system.set_variable_values(X, variables=var_names, to_state=True)

            # evaluate reference phase fractions
            y_R = self._y_R.evaluate(self._C.ad_system).val
            self._C.ad_system.set_variable_values(
                y_R,
                variables=[self._C.reference_phase.fraction_name],
                to_iterate=True,
                to_state=True,
            )

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
        equ = sum(
            [
                phase.fraction
                * phase.specific_enthalpy(
                    self._C.p,
                    self._C.T,
                    *[phase.normalized_fraction_of_component(comp) for comp in phase],
                )
                for phase in self._C.phases
            ]
        )

        # if no phase present (list empty) zero is returned and enthalpy is zero
        if equ == 0:
            h = np.zeros(self._C.ad_system.mdg.num_subdomain_cells())
        # else evaluate this operator
        elif isinstance(equ, pp.ad.Operator):
            h = equ.evaluate(self._C.ad_system).val
        else:
            raise RuntimeError("Something went terribly wrong.")
        # write values in local var form
        self._C.ad_system.set_variable_values(
            h, variables=[self._C.h_name], to_iterate=True, to_state=copy_to_state
        )

    def post_process_fractions(self, copy_to_state: bool = True) -> None:
        """Removes numerical artifacts from all fractional values.

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
        y_R = self._y_R.evaluate(self._C.ad_system).val
        self._C.ad_system.set_variable_values(
            y_R,
            variables=[self._C.reference_phase.fraction_name],
            to_iterate=True,
            to_state=copy_to_state,
        )

        for phase_e in self._C.phases:
            # remove numerical artifacts on phase fractions y
            y_e = self._C.ad_system.get_variable_values(
                [phase_e.fraction_name], from_iterate=True
            )
            y_e[y_e < 0.0] = 0.0
            y_e[y_e > 1.0] = 1.0
            self._C.ad_system.set_variable_values(
                y_e,
                variables=[phase_e.fraction_name],
                to_iterate=True,
                to_state=copy_to_state,
            )

            # remove numerical artifacts in phase compositions
            for comp_c in phase_e:
                xi_ce = self._C.ad_system.get_variable_values(
                    [phase_e.fraction_of_component_name(comp_c)],
                    from_iterate=True,
                )
                xi_ce[xi_ce < 0.0] = 0.0
                xi_ce[xi_ce > 1.0] = 1.0
                # write values
                self._C.ad_system.set_variable_values(
                    xi_ce,
                    variables=[phase_e.fraction_of_component_name(comp_c)],
                    to_iterate=True,
                    to_state=copy_to_state,
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
            # initial value for V_e, W_e
            val_v = phase.fraction.evaluate(ad_system).val
            val_w = self._C.get_composition_unity_for(phase).evaluate(ad_system).val

            # if requested, set initial guess for auxiliary NPIPM vars
            if self.use_auxiliary_npipm_vars:
                v_name = f"{self._V_name}_{phase.name}"
                w_name = f"{self._W_name}_{phase.name}"
                ad_system.set_variable_values(
                    val_v, variables=[v_name], to_iterate=True, to_state=True
                )
                ad_system.set_variable_values(
                    val_w, variables=[w_name], to_iterate=True, to_state=True
                )

            # store value for initial guess for nu
            V_mat.append(val_v)
            W_mat.append(val_w)

        # initial guess for nu is cell-wise scalar product between concatenated V and W
        # for each phase
        V = np.vstack(V_mat).T  # num_cells X num_phases
        W = np.vstack(W_mat)  # num_phases X num_cells
        # the diagonal of the product of above returns the cell-wise scalar product
        # TODO can this be optimized using a for loop over diagonal elements of product?
        nu_mat = np.matmul(V, W)
        nu = np.diag(nu_mat)
        nu = nu / self._C.num_phases
        nu[nu < 0] = 0

        self._nu2.value = 0.9 * np.ones(len(nu))

        if self.npipm_param_as_var:
            ad_system.set_variable_values(
                nu, variables=[self._nu_name], to_iterate=True, to_state=True
            )
        else:
            self._nu = nu

    def _set_initial_guess(self, initial_guess: str) -> None:
        """Auxillary function to set the initial values for phase fractions,
        phase compositions and temperature, based on the chosen strategy."""

        ad_system = self._C.ad_system
        nc = ad_system.mdg.num_subdomain_cells()

        if initial_guess == "iterate":
            # DofManager takes by default values from ITERATE, than from STATE if not found
            pass
        elif initial_guess == "uniform":

            # uniform values for phase fraction
            val_phases = 1.0 / self._C.num_phases
            for phase in self._C.phases:
                ad_system.set_variable_values(
                    val_phases * np.ones(nc),
                    variables=[phase.fraction_name],
                    to_iterate=True,
                )
                # uniform values for composition of this phase
                val = 1.0 / phase.num_components
                for component in self._C.components:
                    ad_system.set_variable_values(
                        val * np.ones(nc),
                        variables=[phase.fraction_of_component_name(component)],
                        to_iterate=True,
                    )
        elif initial_guess == "feed":
            phases = [p for p in self._C.phases if p != self._C.reference_phase]
            # store preliminary phase composition
            composition: dict[Any, dict] = dict()
            # storing feed fractions
            feeds: dict[Any, np.ndarray] = dict()
            pressure = self._C.p.evaluate(ad_system).val
            temperature = self._C.T.evaluate(ad_system).val

            for comp in self._C.components:
                # evaluate feed fractions
                z_c = comp.fraction.evaluate(ad_system).val
                feeds[comp] = z_c

                # set fractions in reference phase to feed
                ad_system.set_variable_values(
                    np.copy(z_c),
                    variables=[
                        self._C.reference_phase.fraction_of_component_name(comp)
                    ],
                    to_iterate=True,
                )

                # # use feed fractions as first value for all phase compositions
                # # values are initiated as zero, thats is why this step is necessary
                # # to avoid division by zero when evaluating k-values.
                # for phase in self._C.phases:
                #     ad_system.set_variable_values(
                #         np.copy(z_c),
                #         variables=[phase.fraction_of_component_name(comp)],
                #         to_iterate=True,
                #     )

            # for phases except ref phase,
            # change values using the k-value estimates, s.t. initial guess
            # fulfils the equilibrium equations
            for phase in phases:
                composition[phase] = dict()
                for comp in self._C.components:
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
                for comp in self._C.components:
                    # normalize
                    x_ce = composition[phase][comp] / sum(composition[phase].values())
                    # set values
                    ad_system.set_variable_values(
                        x_ce,
                        variables=[phase.fraction_of_component_name(comp)],
                        to_iterate=True,
                    )

            # use the feed fraction of the reference component to set an initial guess
            # for the phase fractions
            # re-normalize to set fractions fulfilling the unity constraint
            feed_R = feeds[self._C.reference_component] / len(phases)
            feed_R = np.ones(nc) * 0.9
            for phase in phases:
                ad_system.set_variable_values(
                    np.copy(feed_R), variables=[phase.fraction_name], to_iterate=True
                )
            # evaluate reference phase fraction by unity
            ad_system.set_variable_values(
                self._y_R.evaluate(ad_system).val,
                variables=[self._C.reference_phase.fraction_name],
                to_iterate=True,
            )
        elif initial_guess == "rachford_rice":

            assert (
                self._C.num_phases == 2
            ), "Rachford-Rice initial guess only supported for liquid-gas-equilibrium."

            pressure = self._C.p.evaluate(ad_system)
            temperature = self._C.T.evaluate(ad_system)
            z_c = [comp.fraction.evaluate(ad_system) for comp in self._C.components]

            # Collects initial K values from Wilson's correlation
            K = [
                comp.critical_pressure()
                / pressure.val
                * pp.ad.exp(
                    5.37
                    * (1 + comp.acentric_factor)
                    * (1 - comp.critical_temperature() / temperature.val)
                )
                + 1.0e-12
                for comp in self._C.components
            ]

            def ResidualRR(Y, z_c, K):
                res = np.zeros_like(Y)
                for z_i, K_i in zip(z_c, K):
                    res = res + (z_i.val * (K_i - 1)) / (1 + Y * (K_i - 1))
                return res

            def FindPhaseFraction(a, b, z_c, K):
                n = 5
                for i in range(n):
                    x = (a + b) / 2
                    prod = ResidualRR(a, z_c, K) * ResidualRR(x, z_c, K)
                    b = np.where(prod < 0, x, b)
                    a = np.where(prod > 0, x, a)
                return x

            Y = np.zeros(nc)
            composition: dict[Any, dict] = dict()
            # phases = [p for p in self._C.phases]

            for i in range(3):

                Y = FindPhaseFraction(np.zeros(nc), np.ones(nc), z_c, K)

                res_L_Q = ResidualRR(np.zeros(nc), z_c, K)
                res_G_Q = ResidualRR(np.ones(nc), z_c, K)
                Y = np.where(res_L_Q < 0.0, np.zeros(nc), Y)
                Y = np.where(res_G_Q > 0.0, np.ones(nc), Y)
                invalid = np.logical_and(0.0 > Y, Y > 1.0)
                assert not np.any(invalid)

                for phase in self._C.phases:
                    composition[phase] = dict()
                    for i, comp in enumerate(self._C.components):
                        if phase.eos.gaslike:
                            x_ce = z_c[i] * K[i] / (1 + Y * (K[i] - 1))
                            composition[phase].update({comp: x_ce})
                        else:
                            x_ce = z_c[i] / (1 + Y * (K[i] - 1))
                            composition[phase].update({comp: x_ce})

                for phase in self._C.phases:
                    total = sum(composition[phase].values())
                    for comp in self._C.components:
                        x_ce = composition[phase][comp] / total
                        composition[phase].update({comp: x_ce})

                # update K values from EoS
                phi_L = None
                phi_G = None
                for phase in self._C.phases:
                    x_phase = [composition[phase][comp] for comp in self._C.components]
                    phase.eos.compute(pressure, temperature, *x_phase)
                    if phase.eos.gaslike:
                        phi_G = list(phase.eos.phi.values())
                    else:
                        phi_L = list(phase.eos.phi.values())

                for i, pair in enumerate(zip(phi_L, phi_G)):
                    K[i] = (pair[0] / (pair[1] + 1.0e-12)).val

            # set values.
            for phase in self._C.phases:
                for comp in self._C.components:
                    # set values
                    x_ce = composition[phase][comp].val
                    ad_system.set_variable_values(
                        x_ce,
                        variables=[phase.fraction_of_component_name(comp)],
                        to_iterate=True,
                    )
            # set phase fraction
            for phase in self._C.phases:
                ad_system.set_variable_values(
                    np.copy(Y), variables=[phase.fraction_name], to_iterate=True
                )
            # evaluate reference phase fraction by unity
            ad_system.set_variable_values(
                self._y_R.evaluate(ad_system).val,
                variables=[self._C.reference_phase.fraction_name],
                to_iterate=True,
            )
        else:
            raise ValueError(f"Unknown initial-guess-strategy {initial_guess}.")

    ### Numerical methods --------------------------------------------------------------

    def _Newton_min(
        self, flash_type: Literal["isothermal", "isenthalpic"], do_logging: bool
    ) -> bool:
        """Performs a semi-smooth newton (Newton-min),
        where the complementary conditions are the semi-smooth part.

        Note:
            This looks exactly like a regular Newton since the semi-smooth part,
            since the assembly of the sub-gradients are wrapped in a special operator.

        Parameters:
            flash_type: Name of the flash type.
            do_logging: Flag for printing progress logs.

        Returns:
            A bool indicating the success of the method.

        """
        success = False

        # at this point we know by logic which values flash_type can have.
        if flash_type == "isothermal":
            var_names = self._C.pT_subsystem["primary_vars"]
        else:
            var_names = self._C.ph_subsystem["primary_vars"]

        # Construct a callable representing the chosen subsystem (which is square),
        # including the complementary conditions.
        # The dependency on X is optional, since the AD framework uses ITERATE/STATE
        # by default, but we include it for Armijo line-search.
        def F(X: Optional[np.ndarray] = None) -> tuple[sps.spmatrix, np.ndarray]:
            return self._C.linearize_subsystem(
                flash_type, other_eqns=self.complementary_equations, state=X
            )

        # Perform Newton iterations with above F(x)
        success, iter_final = self._newton_iterations(F, var_names, do_logging)

        # append history entry
        self._history_entry(
            flash=flash_type,
            method="newton-min",
            iterations=iter_final,
            success=success,
        )

        return success

    def _NPIPM(
        self,
        flash_type: Literal["isothermal", "isenthalpic"],
        do_logging: bool,
    ) -> bool:
        """Performs a non-parametric interior point algorithm to find the solution
        inside the compositional space.

        Includes an Armijo line-search to find a descending step size.

        Root-finding is still performed using Newton iterations, with semi-smooth parts.

        Parameters:
            flash_type: Name of the flash type.
            do_logging: Flag for printing progress logs.

        Returns:
            A bool indicating the success of the method.

        """
        success = False

        # at this point we know by logic which values flash_type can have.
        if flash_type == "isothermal":
            var_names = self._C.pT_subsystem["primary_vars"]
        else:
            var_names = self._C.ph_subsystem["primary_vars"]

        # Construct a callable representing the chosen subsystem (which is square),
        # including the NPIPM variables and equations.
        # The dependency on X is optional, since the AD framework uses ITERATE/STATE
        # by default, but we include it for Armijo line-search.
        def F(X: Optional[np.ndarray] = None) -> tuple[sps.spmatrix, np.ndarray]:
            return self._C.linearize_subsystem(
                flash_type,
                other_vars=self.npipm_variables,
                other_eqns=self.npipm_equations,
                state=X,
            )

        # the NPIPM has some algorithmic variables
        success, iter_final = self._newton_iterations(
            F,
            var_names + self.npipm_variables,
            do_logging,
            pre_processor=self._npipm_pre_processor,
        )

        # append history entry
        self._history_entry(
            flash=flash_type,
            method="npipm",
            iterations=iter_final,
            success=success,
        )

        return success

    def _npipm_pre_processor(
        self, A: sps.spmatrix, b: np.ndarray
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Modifying some parts of the NPIPM matrix according to
        Vu et al. (2021), proposition 3.1.

        Also, if ``nu`` is not a variable, modifies the the rhs."""

        nc = self._C.ad_system.mdg.num_subdomain_cells()
        u = self.npipm_parameters["u"]
        eta = self.npipm_parameters["eta"]
        m = self._C.num_phases

        dot_V_W = sum(
            [
                self._V_of_phase[phase].evaluate(self._C.ad_system).val
                * self._W_of_phase[phase].evaluate(self._C.ad_system).val
                for phase in self._C.phases
            ]
        )

        # If nu is a variable and the slack equation is part of the system.
        # the pre-conditioning consists of multiplying the coupling equation (per phase)
        # with dot_V_W**+ * u / m**2 and subtracting them from the slack equation for nu
        # Essentially a Gaussian elimination step as a pre-conditioning.
        if self.npipm_param_as_var:
            A = A.tolil()  # for performance reasons

            # TODO: Augmentation performed under assumption that the order of the
            # equations is as constructed and added to the AD system
            # (hence the direct accessing of indices)
            # This should be done more generically with, in case the system changes
            # when coupled with something else or modified.

            for p, phase in enumerate(self._C.phases):
                factor = (
                    u
                    / m**2
                    * self._V_of_phase[phase].evaluate(self._C.ad_system).val
                    * self._W_of_phase[phase].evaluate(self._C.ad_system).val
                )
                # positive part
                factor[factor < 0] = 0.0
                # last nc rows belong to slack equation
                # chose the the nc-long block rows above per phase for multiplication
                # sps.diags assures that factor (vector) is multiplied with each column
                # of the block-row
                A[-nc:] -= sps.diags(factor) * A[-(p + 2) * nc : -(p + 1) * nc]
                b[-nc:] -= factor * b[-(p + 2) * nc : -(p + 1) * nc]

            # ## First modification: Eliminate derivatives w.r.t. V and W in the slack equ.
            # # According to the set-up, the very last num_cells equations represent
            # # the slack equation involving nu -> last num_cells rows [-nc:]
            # # The column indices are given by the last num_phases * 2 * num_cells,

            # # If the auxiliary vars V and W are not used, we know that the bottom right
            # # block belongs to the slack variable nu
            # if self.use_auxiliary_npipm_vars:
            #     A[-nc:, -m * 2 * nc :] = 0
            # else:
            #     A[-nc:, :-nc] = 0

            # ## Second modification: Augment the derivative of the slack equ w.r.t. nu
            # # by adding the term u * <V,W> / m
            # # the columns of the nu derivative are given by the num_cells block before the
            # # above blocks
            # augmentation = dot_V_W * u / m**2

            # # augment the block in the last equation belonging to nu
            # if self.use_auxiliary_npipm_vars:
            #     A[
            #         -nc:,
            #         -(nc + m * 2 * nc) : -m * 2 * nc,
            #     ] += augmentation
            # else:
            #     A[-nc:, -nc:] += augmentation

            # back to csr and eliminate zeros
            A = A.tocsr()
            A.eliminate_zeros()

        else:
            # If nu is not a variable, the slack equation is not present.
            # In this case we subtract the value of the slack variable from the rhs for
            # each KKT condition (per phase)
            b[-nc:] -= self._nu
            for p in range(1, m):
                b[-(p + 1) * nc : -(p) * nc] -= self._nu

            # Perform an update to nu, based on current values of fractions for the next
            # iteration
            dnu = (
                eta * self._nu
                + self._nu**2
                - u * (dot_V_W / (2 * m) - self._nu) * dot_V_W / m
            ) / (eta + 2 * self._nu + u * dot_V_W / m)
            self._nu -= dnu

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
        X_k = self._C.ad_system.get_variable_values(from_iterate=True)

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
                except:
                    if do_logging:
                        _del_log()
                        print(
                            f"Armijo line search j={j}; evaluation failed.",
                            end="",
                            flush=True,
                        )
                    continue

                pot_j = self._Armijo_potential(b_j)

                if do_logging:
                    _del_log()
                    print(
                        f"Armijo line search j={j}; potential: {pot_j}",
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
                except:
                    if do_logging:
                        _del_log()
                        print(
                            f"Armijo line search j={j}; evaluation failed.",
                            flush=True,
                        )
                    j += 1
                    continue
                j += 1
                pot_j = self._Armijo_potential(b_j)

                if do_logging:
                    _del_log()
                    print(
                        f"Armijo line search j={j}; potential: {pot_j}",
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

    def _update_nu2(self):

        VW = list()

        for phase in self._C.phases:
            VW.append(
                self._V_of_phase[phase].evaluate(self._C.ad_system).val
                * self._W_of_phase[phase].evaluate(self._C.ad_system).val
            )

        VW = sum(VW) / self._C.num_phases

        nu_k = self._nu2.value
        nu_geo = 0.5 * nu_k
        nu_pow = nu_k**2

        # Use stack not hstack, to avoid arrays with shape (n,) (second axis must exist)
        self._nu2.value = np.min(np.stack([nu_geo, nu_pow], axis=1), axis=1)

    def _newton_iterations(
        self,
        F: Callable[[Optional[np.ndarray]], tuple[sps.spmatrix, np.ndarray]],
        var_names: list[str],
        do_logging: bool,
        pre_processor: Optional[
            Callable[[sps.spmatrix, np.ndarray], tuple[sps.spmatrix, np.ndarray]]
        ] = None,
    ) -> tuple[bool, int]:
        """Performs standard Newton iterations using the matrix and rhs-vector returned
        by ``F``, until (possibly) the L2-norm of the rhs-vector reaches the convergence
        criterion.

        Parameters:
            F: A callable representing the function for which the roots should be found.

                The callable ``F(X)`` must be such that the input vector ``X`` is
                optional (None), which makes the AD-system chose values stored as
                ``ITERATE``.

                It further more must return a tuple containing its Jacobian matrix and
                the residual vector, in this case ``-F(X)``.

                The user must ensure that the Jacobian is invertible.
            var_names: A list of variable names, such that ``F`` is solvable.
                This list is used to construct a prolongation matrix and to update
                respective components of the global solution vector.
            do_logging: Flag to print status logs in the console.
            pre_processor: ``default=None``

                An optional callable to pre-process the matrix and right-hand side
                of the linearized system returned by ``F``.

                This is called directly before each solve-command for the linear system

        Returns:
            A 2-tuple containing a bool and an integer, representing the success-
            indicator and the final number of iteration performed.

        """
        success: bool = False
        iter_final: int = 0
        if self.use_armijo:
            logging_end = "\n"
        else:
            logging_end = ""

        # # NOTE: structure for  time measurement
        # self.iter_times=list()
        # self.assembly_times=list()
        # self.solver_times=list()
        # assemble linear system of eq
        # start_assembly = time.time()
        A, b = F()
        # stop_assembly = time.time()
        # self.assembly_times.append(stop_assembly - start_assembly)

        # if residual is already small enough
        if np.linalg.norm(b) <= self.flash_tolerance:
            if do_logging:
                print("Newton iteration 0: success", flush=True)
            success = True
        else:
            # column slicing to relevant variables
            prolongation = self._C.ad_system.projection_to(var_names).transpose()

            for i in range(1, self.max_iter_flash + 1):
                # start_iter = time.time()

                if do_logging:
                    if self.use_armijo:
                        print("", end="\n")
                    _del_log()
                    print(
                        f"Newton iteration {i}; residual norm: {np.linalg.norm(b)}",
                        end=logging_end,
                        flush=True,
                    )

                if pre_processor:
                    A, b = pre_processor(A, b)

                # start_solve = time.time()
                dx = sps.linalg.spsolve(A, b)
                # stop_solve = time.time()
                # self.solver_times.append(stop_solve - start_solve)

                DX = self.newton_update_chop * prolongation * dx

                if self.use_armijo:
                    # get step size using Armijo line search
                    step_size = self._Armijo_line_search(DX, F, do_logging)
                    DX = step_size * DX

                self._C.ad_system.set_variable_values(
                    DX,
                    to_iterate=True,
                    additive=True,
                )

                # start_assembly = time.time()
                self._update_nu2()
                A, b = F()
                # stop_assembly = time.time()
                # self.assembly_times.append(stop_assembly - start_assembly)
                # stop_iter = time.time()
                # self.iter_times.append(stop_iter - start_iter)

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

        return success, iter_final

    ### Saturation evaluation methods --------------------------------------------------

    def _single_phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """If only one phase is present, we assume it occupies the whole pore space."""
        phase = self._C.reference_phase
        values = np.ones(self._C.ad_system.mdg.num_subdomain_cells())
        self._C.ad_system.set_variable_values(
            values,
            variables=[phase.saturation_name],
            to_iterate=True,
            to_state=copy_to_state,
        )

    def _2phase_saturation_evaluation(self, copy_to_state: bool = True) -> None:
        """Calculates the saturation value assuming phase molar fractions are given.
        In the case of 2 phases, the evaluation is straight forward:

            ``s_i = 1 / (1 + y_j / (1 - y_j) * rho_i / rho_j) , i != j``.

        """
        # get reference to phases
        phase1, phase2 = (phase for phase in self._C.phases)
        # get phase molar fraction values
        y1 = self._C.ad_system.get_variable_values(
            variables=[phase1.fraction_name], from_iterate=True
        )
        y2 = self._C.ad_system.get_variable_values(
            variables=[phase2.fraction_name], from_iterate=True
        )

        # get density values for given pressure and enthalpy
        rho1 = phase1.density(self._C.p, self._C.T).evaluate(self._C.ad_system)
        if isinstance(rho1, pp.ad.Ad_array):
            rho1 = rho1.val
        rho2 = phase2.density(self._C.p, self._C.T).evaluate(self._C.ad_system)
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
        self._C.ad_system.set_variable_values(
            s1,
            variables=[phase1.saturation_name],
            to_iterate=True,
            to_state=copy_to_state,
        )
        self._C.ad_system.set_variable_values(
            s2,
            variables=[phase2.saturation_name],
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
        nc = self._C.ad_system.mdg.num_subdomain_cells()
        # molar fractions per phase
        y = [
            self._C.ad_system.get_variable_values(
                variables=[phase.saturation_name], from_iterate=True
            )
            for phase in self._C.phases
        ]
        # densities per phase
        rho = list()
        for phase in self._C.phases:
            rho_e = phase.density(self._C.p, self._C.T).evaluate(self._C.ad_system)
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
            self._C.ad_system.set_variable_values(
                vals,
                variables=[phase.saturation_name],
                to_iterate=True,
                to_state=copy_to_state,
            )
