from functools import partial
from typing import Callable

import numpy as np

import porepy as pp


class DamageHistoryVariable(pp.PorePyModel):
    damage_history_variable = "damage_history"

    interface_displacement_variable: str
    """Interface displacement variable.
    TODO: This is needed for mypy. However, perhaps the better solution is to use a
    MomentumBalanceProtocol to define the interface displacement variable and pacify
    the complaint about safe-super when calling update_solution."""

    def damage_history(self, subdomains: list[pp.Grid]) -> pp.ad.Variable:
        """Fracture damage history [-].

        Parameters:
            subdomains: List of subdomains where the damage history is defined. Should
            be of co-dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture damage history.

        """
        # Check that the subdomains are fractures.
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Damage history only defined on fractures")

        return self.equation_system.md_variable(
            self.damage_history_variable, subdomains
        )

    def create_variables(self) -> None:
        """Create variables for the model."""
        if not isinstance(self, pp.MomentumBalance):
            raise TypeError(
                "DamageHistoryVariable must be used in combination with a "
                "MomentumBalance model."
            )
        super().create_variables()
        self.equation_system.create_variables(
            dof_info={"cells": 1},
            name=self.damage_history_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
            tags={"si_units": "-"},
        )

    def update_solution(self, solution: np.ndarray) -> None:
        """Update the solution with the damage history variable.

        Parameters:
            solution: Solution to update.

        """
        history_var = self.equation_system.get_variables(
            variables=[self.interface_displacement_variable]
        )
        other_vars = [
            var for var in self.equation_system.variables if var not in history_var
        ]
        # Need to store all time steps to compute the damage history.
        self.equation_system.shift_time_step_values(
            max_index=None, variables=history_var
        )
        # Then proceed as usual with the other variables.
        self.equation_system.shift_time_step_values(
            max_index=len(self.time_step_indices), variables=other_vars
        )

        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )


class DamageHistoryEquation(pp.PorePyModel):
    """Base class for damage history equations.

    Sets up the damage history equation for the model. Since the equation considers the
    full history, we reset the equation at each time step to include the new term.

    """

    damage_history_equation_name = "damage_history_equation"
    """Name of the damage history equation."""

    def set_equations(self):
        super().set_equations()
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        eq = self.damage_history_equation(fractures)
        eq.set_name(self.damage_history_equation_name)
        self.equation_system.set_equation(eq, fractures, {"cells": 1})

    def before_nonlinear_loop(self):
        """Reset the damage history equation to include new term.

        This needs to be done *after* time manager is updated, which happens in the
        super class method.

        """
        super().before_nonlinear_loop()
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        self.equation_system._equations[self.damage_history_equation_name] = (
            self.damage_history_equation(fractures)
        )

    def damage_history_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Return the damage history equation.

        Parameters:
            subdomains: List of subdomains where the damage history is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Operator for the damage history equation.

        """
        raise NotImplementedError("Subclass must implement this method.")


class AnisotropicHistoryEquation(DamageHistoryEquation):
    r"""Anisotropic damage history equation.

    The anisotropic damage history equation is given by

    .. math::

        h = \int{\tau} max(m_t \cdot \delta u_t, 0) \, d\tau,

    where :math:`\delta u_t` is the tangential plastic displacement jump increment and
    :math:`m_t` is the normalized tangential plastic displacement jump, while h is the
    damage history variable.

    """

    damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Damage history variable on ad form."""

    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method returning plastic displacement jump."""

    def damage_history_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Anisotropic damage history equation.

        Parameters:
            subdomains: List of subdomains where the damage history is defined. Should
                be of co-dimension one, i.e. fractures.

            Returns:
                Operator for the anisotropic damage history equation.

        """
        # Get the tangential component of the plastic displacement jump.
        nd_vec_to_tangential = self.tangential_component(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )
        # The time increment of the tangential displacement jump.
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        # Prepare for taking the inner product sum.
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        tangential_to_scalar = pp.ad.sum_operator_list(
            [e_i.T for e_i in tangential_basis]
        )

        m_t = self._normalized_tangential_plastic_jump_increment(subdomains)
        num_steps = self.time_manager.time_index
        # The first term in the equation, the implicit part.
        current_inner = tangential_to_scalar @ (m_t * u_t_increment)
        # We are only interested in positive values of the inner product.
        f_max = pp.ad.Function(pp.ad.functions.maximum, "max")
        zero = pp.ad.Scalar(0.0)
        eq = self.damage_history(subdomains) - f_max(current_inner, zero)
        # Then add the explicit part, i.e., the sum of the inner product of m with the u_t
        # increment from all previous time steps. The sum starts at 1 since the first
        # increment is already included in the implicit part.
        for i in range(1, num_steps):
            u_t_increment_i = u_t.previous_timestep(i) - u_t.previous_timestep(i + 1)
            inner = tangential_to_scalar @ (m_t * u_t_increment_i)
            contr_i = f_max(inner, zero)
            eq -= contr_i

        return eq

    def _normalized_tangential_plastic_jump_increment(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Normalized tangential plastic jump increment [-].

        Parameters:
            subdomains: List of subdomains where the jump is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Normalized tangential plastic jump increment.

        """
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        nd_vec_to_tangential = self.tangential_component(subdomains)

        u_t_increment = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        # TODO: Reconsider this function and its implementation.
        f_power = pp.ad.Function(
            partial(pp.ad.safe_power, -1, 1 / np.sqrt(self.nd - 1)), "safe power"
        )
        # Compute normalised tangential displacement increment. First, compute the norm
        # of the increment.
        norm_u_t_increment = scalar_to_tangential @ f_norm(u_t_increment)
        # Then, normalise the increment by multiplying it by the inverse of the norm.
        m_t = f_power(norm_u_t_increment) * u_t_increment
        return m_t


class IsotropicHistoryEquation(pp.PorePyModel):
    r"""Isotropic damage history equation.

    The isotropic damage history equation is given by

    .. math::

        h = \int_{\tau} ||\delta u_t|| \, d\tau,

    where :math:`\delta u_t` is the plastic displacement jump increment and h is the
    damage history variable.
    """

    damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Damage history variable on ad form."""

    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method returning plastic displacement jump."""

    def damage_history_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Isotropic damage history equation.

        Parameters:
            subdomains: List of subdomains where the damage history is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Operator for the isotropic damage history equation.

        """
        nd_vec_to_tangential = self.tangential_component(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        num_steps = self.time_manager.time_index
        # The first term in the equation, the implicit part.
        eq = self.damage_history(subdomains) - f_norm(u_t_increment)
        # Then add the explicit part, i.e., the sum of the inner product of m with the u_t
        # increment from all previous time steps. The sum starts at 1 since the first
        # increment is already included in the implicit part.
        for i in range(1, num_steps):
            u_t_increment_i = u_t.previous_timestep(i) - u_t.previous_timestep(i + 1)
            eq -= f_norm(u_t_increment_i)

        return eq
