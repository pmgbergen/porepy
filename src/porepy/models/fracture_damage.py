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

    damage_history_variable = "damage_history"

    interface_displacement_variable: str
    """Interface displacement variable."""

    contact_traction_variable: str
    """Contact traction variable."""

    def damage_history(self, subdomains: list[pp.Grid]) -> pp.ad.Variable:
        """Fracture damage history [-].

        Parameters:
            subdomains: List of subdomains where the damage history is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Variable for fracture damage history.

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
        super().create_variables()  # type: ignore[safe-super]
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
        self.equation_system.set_equation(
            eq, fractures, {"cells": 1}, is_nonlinear=True
        )

    def before_nonlinear_loop(self):
        """Reset the damage history equation to include new term from previous time
        step.

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

        h = \int{\tau} H(m_t \cdot u_t) ||m_t \cdot \delta u_t|| \, d\tau,

    where :math:`u_t` is the tangential plastic displacement jump, :math:`\delta u_t` is
    its increment, and :math:`m_t` is the normalized tangential plastic displacement
    jump, i.e. unit vector in that direction. h is the damage history variable. See J.
    White (2014) https://doi.org/10.1002/nag.2247 for more details.

    """

    damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Damage history variable on AD form."""

    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method returning plastic displacement jump."""

    characteristic_displacement: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Characteristic displacement."""

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
        # Prepare for taking the inner product sums below.
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        tangential_to_scalar = pp.ad.sum_projection_list(
            [e_i.T for e_i in tangential_basis]
        )

        # Compute the normalized tangential displacement jump (current time step).
        m_t = self._normalized_tangential_plastic_jump(subdomains)
        # The first term in the equation, the implicit part.
        current_inner = tangential_to_scalar @ (m_t * u_t)
        # We are only interested in non-negative values of the inner product. Specify
        # the value of the heaviside function at zero. The choice follows White (2014).
        zero_value = 1.0
        f_heaviside = pp.ad.Function(
            partial(pp.ad.functions.heaviside, zero_value), "max"
        )
        f_abs = pp.ad.Function(pp.ad.functions.abs, "abs")
        # Initialize the equation with the implicit part and the history variable.
        eq = self.damage_history(subdomains) - f_heaviside(current_inner) * f_abs(
            tangential_to_scalar @ (m_t * u_t_increment)
        )
        # Then add the explicit part, i.e., the sum of the inner product of m with the
        # u_t increment from all previous time steps. The sum starts at 1 since the
        # first increment is already included in the implicit part.
        num_steps = self.time_manager.time_index
        for i in range(1, num_steps):
            u_t_i = u_t.previous_timestep(i)
            u_t_increment_i = u_t_i - u_t.previous_timestep(i + 1)
            # Check if the contribution is zero. If it is, we skip the term to avoid
            # unnecessary computations. Set a strict tolerance to avoid neglecting
            # small terms. It's better to err on the side of caution here. Note that the
            # contribution is linear in the increment, so small increments will give
            # small contributions and should not impact the solution significantly. For
            # long simulations with deformation on some time steps only, this could save
            # non-negligible amounts of time.
            tol = 1e-12 * cast(
                float,
                self.equation_system.evaluate(
                    self.characteristic_displacement(subdomains)
                ),
            )
            if np.allclose(self.equation_system.evaluate(u_t_increment_i), 0, atol=tol):
                # The contribution is zero, so we skip it to avoid unnecessary
                # computations.
                continue
            inner_u = tangential_to_scalar @ (m_t * u_t_i)
            inner_u_increment = tangential_to_scalar @ (m_t * u_t_increment_i)

            contr_i = f_heaviside(inner_u) * f_abs(inner_u_increment)
            eq -= contr_i

        return eq

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

        # Get the tangential component of the plastic displacement jump.
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


class IsotropicHistoryEquation(pp.PorePyModel):
    r"""Isotropic damage history equation.

    The isotropic damage history equation is given by

    .. math::

        h = \int_{\tau} ||\delta u_t|| \, d\tau,

    where :math:`\delta u_t` is the plastic displacement jump increment and h is the
    damage history variable. See J. White (2014) https://doi.org/10.1002/nag.2247 for
    more details.

    """

    damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Damage history variable on AD form."""

    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method returning plastic displacement jump."""

    characteristic_displacement: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Characteristic displacement."""

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
        # The time increment of the tangential displacement jump.
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        num_steps = self.time_manager.time_index
        # The first term in the equation, the implicit part.
        eq = self.damage_history(subdomains) - f_norm(u_t_increment)
        # Then add the explicit part, i.e., the sum of the inner product of m with the
        # u_t increment from all previous time steps. The sum starts at 1 since the
        # first increment is already included in the implicit part.
        for i in range(1, num_steps):
            u_t_increment_i = u_t.previous_timestep(i) - u_t.previous_timestep(i + 1)
            # Check if the contribution is zero. If it is, we skip the term to avoid
            # unnecessary computations. Set a strict tolerance to avoid neglecting
            # small terms. It's better to err on the side of caution here. Note that the
            # contribution is linear in the increment, so small increments will give
            # small contributions and should not impact the solution significantly. For
            # long simulations with deformation on some time steps only, this could save
            # non-negligible amounts of time.
            tol = 1e-12 * cast(
                float,
                self.equation_system.evaluate(
                    self.characteristic_displacement(subdomains)
                ),
            )
            if np.allclose(self.equation_system.evaluate(u_t_increment_i), 0, atol=tol):
                # The contribution is zero, so we skip it to avoid unnecessary
                # computations.
                continue
            eq -= f_norm(u_t_increment_i)

        return eq
