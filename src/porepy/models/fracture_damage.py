from functools import partial
from typing import Callable, cast
import abc

import numpy as np

import porepy as pp


class FractureDamageHistoryVariables(pp.PorePyModel):
    """Base class for fracture damage variables.

    Sets up the damage variables for the model. The damage variable is defined on
    fractures and used to compute damage evolution of fracture parameters such as
    dilation and friction. The damage variable is computed from a damage equation, see
    :class:`FractureDamageEquation`.

    """

    friction_damage_history_variable = "friction_damage_history"
    dilation_damage_history_variable = "dilation_damage_history"

    interface_displacement_variable: str
    """Interface displacement variable."""

    contact_traction_variable: str
    """Contact traction variable."""

    def friction_damage_history(self, subdomains: list[pp.Grid]) -> pp.ad.Variable:
        """Fracture friction damage history [-].

        Parameters:
            subdomains: List of subdomains where the damage is defined. Should be of co-
                dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture friction damage.
        """
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Damage only defined on fractures")

        return self.equation_system.md_variable(
            self.friction_damage_history_variable, subdomains
        )

    def dilation_damage_history(self, subdomains: list[pp.Grid]) -> pp.ad.Variable:
        """Fracture dilation damage history [-].

        Parameters:
            subdomains: List of subdomains where the damage is defined. Should be of co-
                dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture dilation damage.
        """
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Damage only defined on fractures")

        return self.equation_system.md_variable(
            self.dilation_damage_history_variable, subdomains
        )

    def create_variables(self) -> None:
        """Create variables for the model."""
        super().create_variables()

        fractures = self.mdg.subdomains(dim=self.nd - 1)

        self.equation_system.create_variables(
            dof_info={"cells": 1},
            name=self.friction_damage_history_variable,
            subdomains=fractures,
            tags={"si_units": "-"},
        )

        self.equation_system.create_variables(
            dof_info={"cells": 1},
            name=self.dilation_damage_history_variable,
            subdomains=fractures,
            tags={"si_units": "-"},
        )

    def update_solution(self, solution: np.ndarray) -> None:
        """Update the solution with the damage variables.

        Parameters:
            solution: Solution to update.

        """
        assert isinstance(self, pp.SolutionStrategy), (
            "The FractureDamageHistoryVariables class should be combined with the "
            "SolutionStrategy class."
        )
        # Check that the only other class in the model implementing this method is
        # pp.SolutionStrategy. This is done since the below method is implemented under
        # that assumption. A more sophisticated approach to updating the solution is
        # needed if this is not the case. Specifically, some variables may need to be
        # stored at, say, two time steps for other purposes than computing the damage
        # history.
        for cls in self.__class__.__mro__:
            if cls is FractureDamageHistoryVariables:
                continue
            if cls is pp.SolutionStrategy:
                continue
            # Check if the class has its own implementation of update_solution.
            update_solution_method = cls.__dict__.get("update_solution", None)
            if update_solution_method is not None:
                raise AssertionError(
                    f"""The class {cls.__name__} implements update_solution, but the
                    FractureDamageHistoryVariables class assumes only
                    pp.SolutionStrategy implements this method."""
                )

        damage_variables = self.variables_stored_all_time_steps()
        other_vars = [
            var for var in self.equation_system.variables if var not in damage_variables
        ]
        # Need to store all time steps to compute the damage history.
        self.equation_system.shift_time_step_values(
            max_index=None, variables=damage_variables
        )
        # Then proceed as usual with the other variables.
        self.equation_system.shift_time_step_values(
            max_index=len(self.time_step_indices), variables=other_vars
        )

        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )

    def variables_stored_all_time_steps(self) -> list[pp.ad.Variable]:
        """Return the variables stored at all time steps.

        This method defines which variables to store at all time steps for computation
        of the damage history. The default implementation includes the contact traction
        and interface displacement variables. The reason we need traction in addition to
        displacement is that the damage history is computed from the plastic
        displacement jump, which requires the contact traction in the case of a nonzero
        elastic jump.

        Note that if used with a pure contact mechanics model, the contact traction
        variable is the only variable stored at all time steps, since the interface
        displacement is not included in the model. In that case, the method should be
        overridden.

        Returns:
            List of variables.

        """
        return self.equation_system.get_variables(
            variables=[
                self.interface_displacement_variable,
                self.contact_traction_variable,
            ]
        )


class FractureDamageEquations(pp.PorePyModel, abc.ABC):
    """Base class for fracture damage equations.

    The equations implemented herein are on the form of convolution integral equations

    .. math::
        \\phi(t) = \\int_0^t K(t - s) f(s) \\, ds + \\phi(0),

    where :math:`\\phi(t)` is the damage variable, :math:`K(t - s)` is the kernel
    function, and :math:`f(s)` is the integrand function. The kernel function
    represents the memory effect of the damage variable, and the integrand function
    represents the contribution of the displacement jump at time :math:`s` to the
    damage variable at time :math:`t`.
    """

    friction_damage_equation_name = "friction_damage_equation"
    dilation_damage_equation_name = "dilation_damage_equation"

    characteristic_displacement: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function to compute the characteristic displacement."""
    dilation_damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Method returning the dilation damage variable."""
    friction_damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Method returning the friction damage variable."""
    dilation_damage_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method to compute the damage coefficient for dilation damage."""

    friction_damage_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method to compute the damage coefficient for friction damage."""

    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function to compute the plastic displacement jump on fractures."""

    def set_equations(self):
        """Set the damage equations for friction and dilation."""
        super().set_equations()
        fractures = self.mdg.subdomains(dim=self.nd - 1)

        friction_eq = self.friction_damage_equation(fractures)
        friction_eq.set_name(self.friction_damage_equation_name)
        self.equation_system.set_equation(friction_eq, fractures, {"cells": 1})

        dilation_eq = self.dilation_damage_equation(fractures)
        dilation_eq.set_name(self.dilation_damage_equation_name)
        self.equation_system.set_equation(dilation_eq, fractures, {"cells": 1})

    def before_nonlinear_loop(self):
        """Reset damage history equations to include new terms."""
        super().before_nonlinear_loop()
        fractures = self.mdg.subdomains(dim=self.nd - 1)

        self.equation_system._equations[self.friction_damage_equation_name] = (
            self.friction_damage_equation(fractures)
        )
        self.equation_system._equations[self.dilation_damage_equation_name] = (
            self.dilation_damage_equation(fractures)
        )

    def _convolution_integral_equation(
        self,
        damage_history_var: pp.ad.Variable,
        integrand_func: Callable[
            [int, list[pp.Grid]], tuple[pp.ad.Operator, pp.ad.Operator]
        ],
        subdomains: list[pp.Grid],
        tolerance: float = 1e-14,
    ) -> pp.ad.Operator:
        """Helper method for convolution integral equations.

        Parameters:
            damage_history_var: The damage variable (friction or dilation).
            integrand_func: Function that takes time step index and subdomains,
                returns (contribution, constant_part) tuple.
            subdomains: List of fracture subdomains.
            tolerance: Tolerance for checking if constant part is non-zero.

        Returns:
            Operator for the damage equation.
        """
        num_steps = self.time_manager.time_index

        # Current time step contribution (implicit part)
        current_contribution, _ = integrand_func(0)  # 0 = current time step
        eq = current_contribution - damage_history_var

        # Previous time steps contributions (explicit part)
        for i in range(1, num_steps):
            # i = number of steps back in time
            eq = self.add_integration_term(eq, i, subdomains, integrand_func, tolerance)

        return eq

    def add_integration_term(
        self,
        eq: pp.ad.Operator,
        time_step_index: int,
        subdomains: list[pp.Grid],
        integrand_func: Callable[[int], tuple[pp.ad.Operator, pp.ad.Operator]],
        tolerance,
    ) -> pp.ad.Operator:
        """Add an integration term to the equation.

        The term is added only if the constant part of the integrand is non-zero,
        under the assumption that the contribution is the product of the constant part
        and some variable. If the constant part is zero, the contribution is also zero,
        regardless of the variable's value.
        If this method is used with an integrand that does not follow this assumption,
        the tolerance can be set negative or the constant part returned can be set to an
        arbitrary non-zero value.

        Parameters:
            eq: The equation to which the term is added.
            time_step_index: Index of the time step.
            subdomains: List of fracture subdomains.
            integrand_func: Function that takes time step index and returns
                (contribution, constant_part) tuple.
            tolerance: Tolerance for checking if constant part is non-zero.

        Returns:
            The updated equation with the new term added.
        """
        contribution_i, constant_part_i = integrand_func(time_step_index)
        # Check if constant part is non-zero before adding. Otherwise, skip it,
        # under the assumption that the contribution is the product of the constant
        # part and some variable. Provided this, constant_part_i = 0 implies
        # contribution_i = 0 regardless of the variable's value.
        constant_value = cast(np.ndarray, constant_part_i.value(self.equation_system))
        if np.any(np.abs(constant_value) > tolerance):  # tolerance for zero check
            eq += contribution_i
        return eq

    def dilation_damage_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Anisotropic dilation damage equation."""

        return self._convolution_integral_equation(
            self.dilation_damage_history(subdomains),
            partial(
                self.damage_integrand,
                subdomains=subdomains,
                degradation=self.dilation_damage_coefficient(subdomains),
            ),  # Now only the time step index is passed to the integrand function.
            subdomains,
        )

    def friction_damage_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Anisotropic friction damage equation."""

        return self._convolution_integral_equation(
            self.friction_damage_history(subdomains),
            partial(
                self.damage_integrand,
                subdomains=subdomains,
                degradation=self.friction_damage_coefficient(subdomains),
            ),  # Now only the time step index is passed to the integrand function.
            subdomains,
        )

    @abc.abstractmethod
    def damage_integrand(
        self,
        time_step_index: int,
        subdomains: list[pp.Grid],
        degradation: pp.ad.Operator,
    ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
        """Integrand for the damage equation.

        Parameters:
            time_step_index: Index of the time step.
            subdomains: List of subdomains where the damage is defined.
            degradation: Function to compute the degradation factor.

        Returns:
            Tuple containing the contribution to the equation and the displacement
            increment at the specified time step. If the displacement increment is zero,
            the full contribution is also zero.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses to define the damage "
            "integrand."
        )


class AnisotropicFractureDamageEquations(FractureDamageEquations):
    """Anisotropic damage equations for both friction and dilation."""

    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method to compute the contact traction on fractures."""

    dilation_damage_history_variable: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Method returning the dilation damage variable."""

    friction_damage_history_variable: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Method returning the friction damage variable."""

    def damage_integrand(
        self,
        time_step_index: int,
        subdomains: list[pp.Grid],
        degradation: pp.ad.Operator,
    ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
        """Integrand for the anisotropic damage equation.

        Parameters:
            time_step_index: Index of the time step.
            subdomains: List of subdomains where the damage is defined.
            degradation: Operator representing the degradation factor.

        Returns:
            Tuple containing the contribution to the equation and the displacement
            increment at the specified time step. If the displacement increment is zero,
            the full contribution is also zero.
        """
        # Get displacement increment.
        u_t: pp.ad.Operator = self.tangential_component(
            subdomains
        ) @ self.plastic_displacement_jump(subdomains)
        # Note that this amounts to a endpoint rule for this interval of the integral.
        if time_step_index == 0:
            degradation_increment = pp.ad.time_increment(degradation)
        else:
            degradation_increment = degradation.previous_timestep(
                time_step_index
            ) - degradation.previous_timestep(time_step_index + 1)

        m_t = self.normalized_tangential_plastic_jump(subdomains)
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        tangential_to_scalar = pp.ad.sum_projection_list(
            [e_i.T for e_i in tangential_basis]
        )

        zero_val = 0.5  # TODO: Verify if this is the correct zero value.
        f_Heaviside = pp.ad.Function(
            partial(pp.ad.functions.heaviside, zero_val), "Heaviside"
        )
        f_abs = pp.ad.Function(pp.ad.abs, "abs_function")

        contribution = f_Heaviside(tangential_to_scalar @ (m_t * u_t)) * f_abs(
            tangential_to_scalar @ (m_t * degradation_increment)
        )

        return contribution, degradation_increment

    def normalized_tangential_plastic_jump(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Normalized tangential plastic jump [-].

        Parameters:
            subdomains: List of subdomains where the jump is defined. Should be of co-
                dimension one, i.e. fractures.

        Returns:
            Normalized tangential plastic jump.
        """
        # Operators for the tangential basis and the tangential component in local
        # coordinates.
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(subdomains)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        # Compute the tangential plastic displacement jump.
        u_t = nd_vec_to_tangential @ self.plastic_displacement_jump(subdomains)

        # Define the functions for the norm and zero-division-safe power.
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        zero_tol = 1e-12 * cast(
            float,
            self.equation_system.evaluate(self.characteristic_displacement(subdomains)),
        )
        f_power = pp.ad.Function(
            partial(pp.ad.safe_power, -1, 1 / np.sqrt(self.nd - 1), zero_tol),
            "safe power",
        )
        # Compute normalized tangential displacement. First, compute the norm of the
        # displacement jump.
        norm_u_t = scalar_to_tangential @ f_norm(u_t)
        # Then, normalize the jump by multiplying it by the inverse of the norm. The
        # safe power is used to handle division by zero.
        m_t = f_power(norm_u_t) * u_t
        return m_t


class IsotropicFractureDamageEquations(FractureDamageEquations):
    """Isotropic damage equations for both friction and dilation."""

    dilation_damage_history_variable: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Method returning the dilation damage variable."""

    def damage_integrand(
        self,
        time_step_index: int,
        subdomains: list[pp.Grid],
        degradation: pp.ad.Operator,
    ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
        """Integrand for the isotropic damage equation.

        Parameters:
            time_step_index: Index of the time step.
            subdomains: List of subdomains where the damage is defined.
            degradation: Function to compute the degradation factor.

        Returns:
            Tuple containing the contribution to the equation and the displacement
            increment at the specified time step. If the displacement increment is zero,
            the full contribution is also zero.
        """

        if time_step_index == 0:
            degradation_increment = pp.ad.time_increment(degradation)
        else:
            degradation_increment = degradation.previous_timestep(
                time_step_index
            ) - degradation.previous_timestep(time_step_index + 1)

        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        contribution = f_norm(degradation_increment)

        return contribution, degradation_increment
