from functools import partial
from typing import Callable, cast

import numpy as np

import porepy as pp


class DamageHistoryVariable(pp.PorePyModel):
    """Base class for damage history variables.

    Sets up the damage history variable for the model. The damage history is defined on
    fractures and used to compute damage evolution of fracture parameters such as
    dilation and friction. The damage history variable is computed from a history
    equation, see :class:`DamageHistoryEquation`.

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
            subdomains: List of subdomains where the damage history is defined. Should
            be of co-dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture friction damage history.
        """
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Damage history only defined on fractures")

        return self.equation_system.md_variable(
            self.friction_damage_history_variable, subdomains
        )

    def dilation_damage_history(self, subdomains: list[pp.Grid]) -> pp.ad.Variable:
        """Fracture dilation damage history [-].

        Parameters:
            subdomains: List of subdomains where the damage history is defined. Should
            be of co-dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture dilation damage history.
        """
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Damage history only defined on fractures")

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
        """Update the solution with the damage history variable.

        Parameters:
            solution: Solution to update.

        """
        assert isinstance(self, pp.SolutionStrategy), (
            "The DamageHistoryVariable class should be combined with the "
            "SolutionStrategy class."
        )
        # Check that the only other class in the model implementing this method is
        # pp.SolutionStrategy. This is done since the below method is implemented under
        # that assumption. A more sophisticated approach to updating the solution is
        # needed if this is not the case. Specifically, some variables may need to be
        # stored at, say, two time steps for other purposes than computing the damage
        # history.
        for cls in self.__class__.__mro__:
            if cls is DamageHistoryVariable:
                continue
            if cls is pp.SolutionStrategy:
                continue
            # Check if the class has its own implementation of update_solution
            update_solution_method = cls.__dict__.get("update_solution", None)
            if update_solution_method is not None:
                raise AssertionError(
                    f"The class {cls.__name__} implements update_solution, but the "
                    "DamageHistoryVariable class assumes only pp.SolutionStrategy "
                    "implements this method."
                )

        history_variables = self.variables_stored_all_time_steps()
        other_vars = [
            var
            for var in self.equation_system.variables
            if var not in history_variables
        ]
        # Need to store all time steps to compute the damage history.
        self.equation_system.shift_time_step_values(
            max_index=None, variables=history_variables
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


class DamageHistoryEquation(pp.PorePyModel):
    """Base class for damage history equations."""

    friction_damage_history_equation_name = "friction_damage_history_equation"
    dilation_damage_history_equation_name = "dilation_damage_history_equation"

    def set_equations(self):
        super().set_equations()
        fractures = self.mdg.subdomains(dim=self.nd - 1)

        friction_eq = self.friction_damage_history_equation(fractures)
        friction_eq.set_name(self.friction_damage_history_equation_name)
        self.equation_system.set_equation(friction_eq, fractures, {"cells": 1})

        dilation_eq = self.dilation_damage_history_equation(fractures)
        dilation_eq.set_name(self.dilation_damage_history_equation_name)
        self.equation_system.set_equation(dilation_eq, fractures, {"cells": 1})

    def before_nonlinear_loop(self):
        """Reset damage history equations to include new terms."""
        super().before_nonlinear_loop()
        fractures = self.mdg.subdomains(dim=self.nd - 1)

        self.equation_system._equations[self.friction_damage_history_equation_name] = (
            self.friction_damage_history_equation(fractures)
        )
        self.equation_system._equations[self.dilation_damage_history_equation_name] = (
            self.dilation_damage_history_equation(fractures)
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
            damage_history_var: The damage history variable (friction or dilation).
            integrand_func: Function that takes time step index and subdomains,
                returns (contribution, constant_part) tuple.
            subdomains: List of fracture subdomains.
            tolerance: Tolerance for checking if constant part is non-zero.

        Returns:
            Operator for the damage history equation.
        """
        num_steps = self.time_manager.time_index

        # Current time step contribution (implicit part)
        current_contribution, _ = integrand_func(0, subdomains)  # 0 = current time step
        eq = damage_history_var - current_contribution

        # Previous time steps contributions (explicit part)
        for i in range(1, num_steps):
            # i = number of steps back in time
            contribution_i, constant_part_i = integrand_func(i, subdomains)
            # Check if constant part is non-zero before adding. Otherwise, skip it,
            # under the assumption that the contribution is the product of the constant
            # part and some variable. Provided this, constant_part_i = 0 implies
            # contribution_i = 0 regardless of the variable's value.
            constant_value = constant_part_i.value(self.equation_system)
            if np.any(np.abs(constant_value) > tolerance):  # tolerance for zero check
                eq -= contribution_i

        return eq

    def friction_damage_history_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Return the friction damage history equation."""
        raise NotImplementedError("Subclass must implement this method.")

    def dilation_damage_history_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Return the dilation damage history equation."""
        raise NotImplementedError("Subclass must implement this method.")


class AnisotropicHistoryEquation(DamageHistoryEquation):
    """Anisotropic damage history equations for both friction and dilation."""

    def friction_damage_history_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Anisotropic friction damage history equation."""

        def friction_integrand(
            time_step_index: int, sds: list[pp.Grid]
        ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
            # Get displacement increment at the specified time step.
            nd_vec_to_tangential = self.tangential_component(sds)
            u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
                sds
            )

            if time_step_index == 0:
                u_t_increment = pp.ad.time_increment(u_t)
            else:
                u_t_increment = u_t.previous_timestep(
                    time_step_index
                ) - u_t.previous_timestep(time_step_index + 1)

            # For friction: max(m_t · δu_t, 0) where m_t is normalized tangential
            # increment.
            m_t = self._normalized_tangential_plastic_jump(sds)
            tangential_basis = self.basis(sds, dim=self.nd - 1)
            tangential_to_scalar = pp.ad.sum_operator_list(
                [e_i.T for e_i in tangential_basis]
            )

            inner = tangential_to_scalar @ (m_t * u_t_increment)
            f_max = pp.ad.Function(pp.ad.functions.maximum, "max")
            zero = pp.ad.Scalar(0.0)
            contribution = f_max(inner, zero)

            return contribution, u_t_increment

        return self._convolution_integral_equation(
            self.friction_damage_history(subdomains), friction_integrand, subdomains
        )

    def dilation_damage_history_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Anisotropic dilation damage history equation."""

        def dilation_integrand(
            time_step_index: int, sds: list[pp.Grid]
        ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
            # Get displacement increment at the specified time step.
            nd_vec_to_tangential = self.tangential_component(sds)
            u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
                sds
            )

            if time_step_index == 0:
                u_t_increment = pp.ad.time_increment(u_t)
            else:
                u_t_increment = u_t.previous_timestep(
                    time_step_index
                ) - u_t.previous_timestep(time_step_index + 1)

            # For dilation: same as friction for anisotropic case (can be customized).
            m_t = self._normalized_tangential_plastic_jump(sds)
            tangential_basis = self.basis(sds, dim=self.nd - 1)
            tangential_to_scalar = pp.ad.sum_operator_list(
                [e_i.T for e_i in tangential_basis]
            )

            inner = tangential_to_scalar @ (m_t * u_t_increment)
            f_max = pp.ad.Function(pp.ad.functions.maximum, "max")
            zero = pp.ad.Scalar(0.0)
            contribution = f_max(inner, zero)

            return contribution, u_t_increment

        return self._convolution_integral_equation(
            self.dilation_damage_history(subdomains), dilation_integrand, subdomains
        )

    def _normalized_tangential_plastic_jump(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Normalized tangential plastic jump [-].

        Parameters:
            subdomains: List of subdomains where the jump is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Normalized tangential plastic jump.

        """
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(subdomains)

        u_t = nd_vec_to_tangential @ self.plastic_displacement_jump(subdomains)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        # Define the functions for the norm and safe power.
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
        # Then, normalize the jump by multiplying it by the inverse of the norm.
        m_t = f_power(norm_u_t) * u_t
        return m_t


class IsotropicHistoryEquation(DamageHistoryEquation):
    """Isotropic damage history equations for both friction and dilation."""

    def friction_damage_history_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Isotropic friction damage history equation."""

        def friction_integrand(
            time_step_index: int, sds: list[pp.Grid]
        ) -> tuple[pp.ad.Operator, pp.ad.Operator]:
            """Integrand for the friction damage history equation.

            Parameters:
                time_step_index: Index of the time step.
                sds: List of subdomains where the damage history is defined.

            Returns:
                Tuple containing the contribution to the equation and the displacement
                increment at the specified time step. If the displacement increment
                is zero, the full contribution is also zero.

            """
            # Get displacement increment at the specified time step
            nd_vec_to_tangential = self.tangential_component(sds)
            u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
                sds
            )

            if time_step_index == 0:
                u_t_increment = pp.ad.time_increment(u_t)
            else:
                u_t_increment = u_t.previous_timestep(
                    time_step_index
                ) - u_t.previous_timestep(time_step_index + 1)

            # For friction: ||δu_t||
            f_norm = pp.ad.Function(
                partial(pp.ad.l2_norm, self.nd - 1), "norm_function"
            )
            contribution = f_norm(u_t_increment)

            return contribution, u_t_increment

        return self._convolution_integral_equation(
            self.friction_damage_history(subdomains), friction_integrand, subdomains
        )

    def dilation_damage_history_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Isotropic dilation damage history equation."""

        def dilation_integrand(
            time_step_index: int, sds: list[pp.Grid]
        ) -> pp.ad.Operator:
            # Get displacement increment at the specified time step
            nd_vec_to_tangential = self.tangential_component(sds)
            u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
                sds
            )

            if time_step_index == 0:
                u_t_increment = pp.ad.time_increment(u_t)
            else:
                u_t_increment = u_t.previous_timestep(
                    time_step_index
                ) - u_t.previous_timestep(time_step_index + 1)

            # For dilation: ||δu_t|| (same as friction for isotropic case)
            f_norm = pp.ad.Function(
                partial(pp.ad.l2_norm, self.nd - 1), "norm_function"
            )
            return f_norm(u_t_increment)

        return self._convolution_integral_equation(
            self.dilation_damage_history(subdomains), dilation_integrand, subdomains
        )
