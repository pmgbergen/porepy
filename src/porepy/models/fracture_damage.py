from functools import partial
from typing import Callable, cast
import abc

import numpy as np

import porepy as pp


class FractureDamageVariables(pp.PorePyModel):
    """Base class for fracture damage variables.

    Common functionality for fracture damage variables. Currently related to storing of
    multiple time steps of the variables entering the history integral.

    """

    interface_displacement_variable: str
    """Interface displacement variable."""

    contact_traction_variable: str
    """Contact traction variable."""

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
            if cls is FractureDamageVariables:
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

        damage_variables = cast(
            FractureDamageVariables, self
        ).variables_stored_all_time_steps()
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


class DilationDamageVariable(FractureDamageVariables):
    """Dilation damage variable for fractures.

    Defines the variable and sets it to the equation system.

    """

    dilation_damage_history_variable = "dilation_damage_history"

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

        self.equation_system.create_variables(
            dof_info={"cells": 1},
            name=self.dilation_damage_history_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
            tags={"si_units": "-"},
        )


class FrictionDamageVariable(FractureDamageVariables):
    """Friction damage variable for fractures.

    Defines the variable and sets it to the equation system.
    """

    friction_damage_history_variable = "friction_damage_history"

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

    def create_variables(self) -> None:
        """Create variables for the model."""
        super().create_variables()

        self.equation_system.create_variables(
            dof_info={"cells": 1},
            name=self.friction_damage_history_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
            tags={"si_units": "-"},
        )


class FractureDamageEquations(pp.PorePyModel, abc.ABC):
    """Base class for fracture damage equations.

    Provides shared helpers for damage convolution-based equations. Subclasses should
    implement specific equations (friction or dilation) and override `set_equations` and
    `before_nonlinear_loop` to register the equations they provide.
    """

    characteristic_displacement: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function to compute the characteristic displacement."""
    contact_mechanics_open_state_characteristic: Callable[
        [list[pp.Grid]], pp.ad.Operator
    ]
    """Method to compute the open/closed state characteristic for contact mechanics."""
    damage_length: Callable[[list[pp.Grid], int], tuple[pp.ad.Operator, pp.ad.Operator]]
    """Method returning the damage length operator."""
    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function to compute the plastic displacement jump on fractures."""

    def damage_convolution_integral(
        self,
        length_function: Callable[
            [list[pp.Grid], int], tuple[pp.ad.Operator, pp.ad.Operator]
        ],
        damage_coefficient_function: Callable[[list[pp.Grid]], pp.ad.Operator],
        subdomains: list[pp.Grid],
        tolerance: float = 1e-14,
    ) -> pp.ad.Operator:
        """Helper method for convolution integral equations.

        Parameters:
            length_function: Function that takes (subdomains, time_step_index) and
                returns (contribution, constant_part) tuple.
            damage_coefficient_function: Function returning the damage coefficient
                operator for the current time step.
            subdomains: List of fracture subdomains.
            tolerance: Tolerance for checking if constant part is non-zero.

        Returns:
            Operator for the damage equation.
        """
        num_steps = self.time_manager.time_index

        # Current time step contribution (implicit part). 0 = current time step.
        damage_coefficient = damage_coefficient_function(subdomains)
        length, _ = length_function(subdomains, 0)
        eq = damage_coefficient * length

        # Previous time steps contributions (explicit part).
        for i in range(1, num_steps):
            # i = number of steps back in time.
            damage_coefficient_i = damage_coefficient.previous_timestep(i)
            length_i, constant_part_i = length_function(subdomains, i)
            # Provided the contribution is the product of the constant part and some
            # variable, constant_part_i = 0 implies contribution_i = 0 regardless of the
            # variable's value. The damage coefficient is also treated as constant (it
            # is evaluated at the previous time step).
            constant_value = cast(
                np.ndarray,
                (constant_part_i * damage_coefficient_i).value(self.equation_system),
            )
            if np.any(np.abs(constant_value) > tolerance):  # tolerance for zero check
                eq += length_i * damage_coefficient_i

        return eq


class DilationDamageEquation(FractureDamageEquations):
    """Mixin class that provides the dilation damage equation and registration."""

    dilation_damage_equation_name = "dilation_damage_equation"
    dilation_damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Method returning the dilation damage variable."""
    dilation_damage_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method to compute the damage coefficient for dilation damage."""

    def set_equations(self):
        """Set the dilation damage equation."""
        super().set_equations()
        fractures = self.mdg.subdomains(dim=self.nd - 1)

        dilation_eq = self.dilation_damage_equation(fractures)
        dilation_eq.set_name(self.dilation_damage_equation_name)
        self.equation_system.set_equation(dilation_eq, fractures, {"cells": 1})

    def before_nonlinear_loop(self):
        """Update the dilation damage equation to include new term."""
        super().before_nonlinear_loop()
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        self.equation_system._equations[self.dilation_damage_equation_name] = (
            self.dilation_damage_equation(fractures)
        )

    def dilation_damage_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Dilation damage equation.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Dilation damage equation operator.
        """
        # If the contact mechanics state is open, use the open state characteristic to
        # enforce no update of the damage history. Otherwise, the standard version of
        # the damage equation is used (characteristic=0).
        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)

        eq = (
            (pp.ad.Scalar(1.0) - characteristic)
            * self.damage_convolution_integral(
                self.damage_length,
                self.dilation_damage_coefficient,
                subdomains=subdomains,
            )
            - self.dilation_damage_history(subdomains)
            + characteristic
            * self.dilation_damage_history(subdomains).previous_timestep(1)
        )
        eq.set_name("dilation_damage_equation")
        return eq


class FrictionDamageEquation(FractureDamageEquations):
    """Mixin class that provides the friction damage equation and registration."""

    friction_damage_equation_name = "friction_damage_equation"
    friction_damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Method returning the friction damage variable."""
    friction_damage_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method to compute the damage coefficient for friction damage."""

    def set_equations(self):
        """Set the friction damage equation."""
        super().set_equations()
        fractures = self.mdg.subdomains(dim=self.nd - 1)

        friction_eq = self.friction_damage_equation(fractures)
        friction_eq.set_name(self.friction_damage_equation_name)
        self.equation_system.set_equation(friction_eq, fractures, {"cells": 1})

    def before_nonlinear_loop(self):
        """Update the friction damage equation to include new term."""
        super().before_nonlinear_loop()
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        self.equation_system._equations[self.friction_damage_equation_name] = (
            self.friction_damage_equation(fractures)
        )

    def friction_damage_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction damage equation.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Friction damage equation operator.
        """
        # If the contact mechanics state is open, use the open state characteristic to
        # enforce no update of the damage history. Otherwise, the standard version of
        # the damage equation is used (characteristic=0).
        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)
        eq = (
            (pp.ad.Scalar(1.0) - characteristic)
            * self.damage_convolution_integral(
                self.damage_length,
                self.friction_damage_coefficient,
                subdomains=subdomains,
            )
            - self.friction_damage_history(subdomains)
            + characteristic
            * self.friction_damage_history(subdomains).previous_timestep(1)
        )
        eq.set_name("friction_damage_equation")
        return eq


class AnisotropicFractureDamageLength(pp.PorePyModel):
    """Anisotropic damage equations for both friction and dilation.

    When combined with both ``class:FrictionDamageEquation`` and
    ``class:DilationDamageEquation``, the use of a single damage length method implies a
    unified treatment of damage in both friction and dilation.
    """

    characteristic_displacement: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method to compute the characteristic displacement on fractures."""

    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method to compute the contact traction on fractures."""

    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method returning the plastic displacement jump variable."""

    def damage_length(
        self,
        subdomains: list[pp.Grid],
        time_step_index: int,
    ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
        r"""Integrand for the anisotropic damage equation.

        The damage length is defined as the difference between the positive part of the
        values of the tangential displacement along the update direction m at time n and
        the previous time step n-1:

        .. math::

            L_d = \max(0, m \cdot u_t_{n}) - \max(0, m \cdot u_t_{n-1})



        Parameters:
            subdomains: List of subdomains where the damage is defined.
            time_step_index: Index of the time step.

        Returns:
            Tuple containing the contribution to the equation and the displacement
            increment at the specified time step. If the displacement increment is zero,
            the full contribution is also zero.
        """
        # Fracture coordinate basis functions.
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        tangential_to_scalar = pp.ad.sum_projection_list(
            [e_i.T for e_i in tangential_basis]
        )

        # Get variables.
        u_t: pp.ad.Operator = self.tangential_component(
            subdomains
        ) @ self.plastic_displacement_jump(subdomains)
        m_t = self.normalized_tangential_plastic_jump(subdomains)
        # Derived previous time step values. If time_step_index is 0, u_t_0 is the
        # actual variable.
        u_t_1 = u_t.previous_timestep(time_step_index + 1)
        u_t_0 = u_t.previous_timestep(time_step_index)

        # Length is evaluated using the ramp function max(x, 0)
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        zero = pp.ad.Scalar(0.0)
        max_0 = f_max(
            tangential_to_scalar @ (m_t * u_t_0),
            zero,
        )
        max_1 = f_max(
            tangential_to_scalar @ (m_t * u_t_1),
            zero,
        )
        f_abs = pp.ad.Function(pp.ad.abs, "abs_function")
        contribution = f_abs(max_1 - max_0)
        # If time_step_index > 0, we can safely disregard the contribution if the
        # displacement increment is zero. Return increment for checking before adding
        # the contribution.
        increment = u_t_0 - u_t_1
        return contribution, tangential_to_scalar @ increment

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
        zero_tol = 1e-10 * cast(
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


class IsotropicFractureDamageLength(pp.PorePyModel):
    """Isotropic damage equations for both friction and dilation.

    When combined with both ``class:FrictionDamageEquation`` and
    ``class:DilationDamageEquation``, the use of a single damage length method
    implies a unified treatment of damage in both friction and dilation.
    """

    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method returning the plastic displacement jump variable."""

    def damage_length(
        self,
        subdomains: list[pp.Grid],
        time_step_index: int,
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
        nd_vec_to_tangential = self.tangential_component(subdomains)
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        tangential_to_scalar = pp.ad.sum_projection_list(
            [e_i.T for e_i in tangential_basis]
        )
        u_t = nd_vec_to_tangential @ self.plastic_displacement_jump(subdomains)
        u_t_increment = u_t.previous_timestep(time_step_index) - u_t.previous_timestep(
            time_step_index + 1
        )

        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        contribution = f_norm(u_t_increment)

        return contribution, tangential_to_scalar @ u_t_increment
