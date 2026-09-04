r"""Fracture damage models.

The formulation used here is described in Stefansson et al. in preparation. It covers
two damage channels, dilation and friction, either of which may be activated on its own
or together with the other, and the damage may be isotropic or anisotropic.

The main components are the following:
    1. History variables.
    2. Equations for the history variables, which are convolution integrals over the
       history of the plastic displacement jump. The equations contain a damage
       evolution coefficient as well as a function describing the length of shear,
       respectively k and l in the paper.
        .. math::

            \Lambda(t) = \int_0^t k(s)\, \ell(t, s)\, \mathrm{d}s,

        where :math:`t` is the current time and :math:`s` the integration variable. The
        length function depends on both: in the anisotropic case it is projected onto
        the accumulated slip direction evaluated at the *current* time, which is why the
        history cannot be accumulated incrementally. The damage evolution coefficient is
        specified in constitutive_laws.py. A single history serves both channels, since
        the driver is common to them.
    3. Constitutive laws that compute the damage states from the history variable and
       modify the friction and dilation accordingly. l depends on the damage being
       anisotropic or isotropic. The damage states are

        .. math::

            d^{\alpha} = d_0^{\alpha}
                + (1 - d_0^{\alpha}) \exp(-\Lambda / \Lambda_c^{\alpha}),

        where :math:`d_0^{\alpha}` is the *residual* damage state for type
        :math:`\alpha`: :math:`d^{\alpha}` starts at one and decays towards
        :math:`d_0^{\alpha}`, which is the reverse of White's (2014) convention, where
        the multiplier decays from a value above one towards one. The wear energy scale
        :math:`\Lambda_c^{\alpha}` is what distinguishes the two channels.

       Each channel applies its damage state multiplicatively: dilation to the intact
       shear dilation gap, friction to the intact friction coefficient. See
       :class:`~porepy.constitutive_laws.FractureDamage`.
"""

from functools import partial
from typing import Callable, cast

import numpy as np

import porepy as pp


class FractureDamageVariable(pp.PorePyModel):
    """Fracture damage history variable.

    Defines the variable and sets it to the equation system, and arranges for the
    variables entering the history integral to be stored at all time steps.

    A single history serves both damage channels: the driver is common to them, so two
    histories would carry the same number. The channels are distinguished by their wear
    energy scales in the softening functions, see
    :class:`~porepy.constitutive_laws.FractureDamage`.

    """

    damage_history_variable = "damage_history"

    interface_displacement_variable: str
    """Interface displacement variable."""

    contact_traction_variable: str
    """Contact traction variable."""

    def damage_history(self, subdomains: list[pp.Grid]) -> pp.ad.Variable:
        """Fracture damage history [-].

        Parameters:
            subdomains: List of subdomains where the damage is defined. Should be of co-
                dimension one, i.e. fractures.

        Returns:
            Variable for the nondimensionalized fracture damage history.

        """
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Damage only defined on fractures")

        return self.equation_system.md_variable(
            self.damage_history_variable, subdomains
        )

    def create_variables(self) -> None:
        """Create variables for the model."""
        # Call super to create variables defined by other mixin classes. Mypy only sees
        # the protocol's trivial body here; the call resolves to a sibling mixin at
        # runtime.
        super().create_variables()  # type: ignore[safe-super]

        self.equation_system.create_variables(
            dof_info={"cells": 1},
            name=self.damage_history_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
            tags={"si_units": "-"},
        )

    def update_time_step_solution(self) -> None:
        """Update the solution with the damage variables."""
        assert isinstance(self, pp.SolutionStrategy), (
            "The FractureDamageVariable class should be combined with the "
            "SolutionStrategy class."
        )
        # Check that the only other class in the model implementing this method is
        # pp.SolutionStrategy. This is done since the below method is implemented under
        # that assumption. A more sophisticated approach to updating the solution is
        # needed if this is not the case. Specifically, some variables may need to be
        # stored at, say, two time steps for other purposes than computing the damage
        # history.
        for cls in self.__class__.__mro__:
            if cls is FractureDamageVariable:
                continue
            if cls is pp.SolutionStrategy:
                continue
            # Check if the class has its own implementation of update_solution.
            update_solution_method = cls.__dict__.get("update_time_step_solution", None)
            if update_solution_method is not None:
                raise AssertionError(
                    f"""The class {cls.__name__} implements update_time_step_solution,
                    but the FractureDamageVariable class assumes only
                    pp.SolutionStrategy implements this method."""
                )

        damage_variables = cast(
            FractureDamageVariable, self
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
        # Finally, update the solution with the new time step values for all variables.
        solution = self.equation_system.get_variable_values(iterate_index=0)
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

        The history variable itself is also included, so that a damage evolution
        coefficient depending on the damage state can be evaluated at the step it
        belongs to. :meth:`FractureDamageEquation.damage_convolution_integral` reaches
        past steps by pushing the whole coefficient back with ``previous_timestep``,
        which rewrites every variable in it, so what is stored is the history rather
        than a separately cached coefficient.

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
                self.damage_history_variable,
            ]
        )


class FractureDamageEquation(pp.PorePyModel):
    """Convolution equation for the fracture damage history.

    One equation serves both damage channels, matching the single history variable of
    :class:`FractureDamageVariable`.
    """

    damage_equation_name = "damage_equation"

    characteristic_displacement: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function to compute the characteristic displacement."""
    contact_mechanics_open_state_characteristic: Callable[
        [list[pp.Grid]], pp.ad.Operator
    ]
    """Method to compute the open/closed state characteristic for contact mechanics."""
    damage_length: Callable[[list[pp.Grid], int], tuple[pp.ad.Operator, pp.ad.Operator]]
    """Method returning the damage length operator."""
    damage_history: Callable[[list[pp.Grid]], pp.ad.Variable]
    """Method returning the damage history variable."""
    damage_evolution_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method to compute the damage evolution coefficient."""

    def damage_convolution_integral(
        self,
        length_function: Callable[
            [list[pp.Grid], int], tuple[pp.ad.Operator, pp.ad.Operator]
        ],
        damage_coefficient_function: Callable[[list[pp.Grid]], pp.ad.Operator],
        subdomains: list[pp.Grid],
        tolerance: float = 1e-14,
    ) -> pp.ad.Operator:
        r"""Helper method for convolution integral equations.

        Discrete counterpart of

        .. math::

            \Lambda^{\alpha}(t) = \int_0^t k^{\alpha}(s)\, \ell(t, s)\, \mathrm{d}s,

        with :math:`t` the current time and :math:`s` the integration variable. The
        contribution of the current time step is implicit; earlier steps are evaluated
        at their own time index and are therefore constant within the nonlinear
        iteration. Note that the sum runs over the whole history rather than updating a
        stored value: the length function is re-evaluated against the current state at
        every step, which for the anisotropic length means re-projection onto the
        present slip direction.

        Parameters:
            length_function: Function that takes (subdomains, time_step_index) and
                returns a (contribution, increment_norm) tuple.
            damage_coefficient_function: Function returning the damage coefficient
                operator for the current time step.
            subdomains: List of fracture subdomains.
            tolerance: Tolerance for checking if the increment norm is non-zero.

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
            length_i, increment_norm_i = length_function(subdomains, i)
            # A vanishing slip increment gives a vanishing contribution for both length
            # functions, regardless of the current state they are evaluated against, so
            # such terms can be dropped from the sum. Both factors are evaluated at a
            # previous time step and are therefore constant.
            constant_value = cast(
                np.ndarray,
                (increment_norm_i * damage_coefficient_i).value(self.equation_system),
            )
            if np.any(np.abs(constant_value) > tolerance):  # tolerance for zero check
                eq += length_i * damage_coefficient_i

        return eq

    def set_equations(self):
        """Set the damage equation."""
        super().set_equations()
        fractures = self.mdg.subdomains(dim=self.nd - 1)

        eq = self.damage_equation(fractures)
        eq.set_name(self.damage_equation_name)
        self.equation_system.set_equation(eq, fractures, {"cells": 1})

    def before_nonlinear_loop(self):
        """Update the damage equation to include the new term."""
        super().before_nonlinear_loop()
        fractures = self.mdg.subdomains(dim=self.nd - 1)
        self.equation_system.update_equation(
            equation_name=self.damage_equation_name,
            new_equation=self.damage_equation(fractures),
            grids=fractures,
            equations_per_grid_entity={"cells": 1},
        )

    def damage_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Damage equation.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Damage equation operator.

        """
        # If the contact mechanics state is open, use the open state characteristic to
        # enforce no update of the damage history. Otherwise, the standard version of
        # the damage equation is used (characteristic=0).
        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)

        eq = (
            (pp.ad.Scalar(1.0) - characteristic)
            * self.damage_convolution_integral(
                self.damage_length,
                self.damage_evolution_coefficient,
                subdomains=subdomains,
            )
            - self.damage_history(subdomains)
            + characteristic * self.damage_history(subdomains).previous_timestep(1)
        )
        eq.set_name(self.damage_equation_name)
        return eq


class IsotropicFractureDamageLength(pp.PorePyModel):
    """Isotropic damage equations for both friction and dilation.

    When combined with both :class:`FrictionDamageEquation` and
    :class:`DilationDamageEquation`, the use of a single damage length method
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
            subdomains: List of subdomains where the damage is defined.
            time_step_index: Index of the time step.

        Returns:
            Tuple containing the contribution to the equation and the norm of the
            displacement increment at the specified time step. If the increment norm is
            zero, the full contribution is also zero.
        """
        nd_vec_to_tangential = self.tangential_component(subdomains)
        u_t = nd_vec_to_tangential @ self.plastic_displacement_jump(subdomains)
        u_t_increment = u_t.previous_timestep(time_step_index) - u_t.previous_timestep(
            time_step_index + 1
        )

        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # The norm vanishes if and only if the increment does, which is what the caller
        # needs in order to discard a history term. Summing the tangential components
        # instead would report an increment such as (a, -a) as zero.
        increment_norm = f_norm(u_t_increment)

        return increment_norm, increment_norm


class AnisotropicFractureDamageLength(pp.PorePyModel):
    """Anisotropic damage equations for both friction and dilation.

    When combined with both :class:`FrictionDamageEquation` and
    :class:`DilationDamageEquation`, the use of a single damage length method implies a
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

        The damage length is the *absolute* difference between the positive parts of the
        tangential plastic jump projected on the slip direction :math:`m` at steps
        :math:`n` and :math:`n-1`:

        .. math::

            \ell_n = \left| \max(0, m \cdot u_{t,n}^p)
                          - \max(0, m \cdot u_{t,n-1}^p) \right|

        The absolute value is essential and easy to lose: without it, slip that reduces
        the projection onto :math:`m` would subtract from the accumulated history, i.e.
        reverse shear would heal the fracture. It is the discrete counterpart of the
        :math:`\left| m \cdot \dot{u}_t^p \right|` in the continuous length function.

        The :math:`\max(0, \cdot)` pair implements the one-sided nature of asperity
        contact, the Heaviside factor of the continuous form: slip oblique to :math:`m`
        contributes with reduced magnitude, and slip beyond 90 degrees not at all.

        Note that :math:`m` is evaluated at the *current* time in both terms, see
        :meth:`normalized_tangential_plastic_jump`.

        Parameters:
            subdomains: List of subdomains where the damage is defined.
            time_step_index: Index of the time step.

        Returns:
            Tuple containing the contribution to the equation and the norm of the
            displacement increment at the specified time step. If the increment norm is
            zero, the full contribution is also zero.
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
        # displacement increment is zero. Return the increment norm for checking before
        # adding the contribution; the norm vanishes if and only if the increment does,
        # whereas summing the tangential components would report an increment such as
        # (a, -a) as zero.
        increment = u_t_0 - u_t_1
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        return contribution, f_norm(increment)

    def normalized_tangential_plastic_jump(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        r"""Normalized tangential plastic jump [-].

        This is the direction of the *accumulated* plastic slip at the current time,
        :math:`m(t) = u_t^p(t) / \|u_t^p(t)\|` -- not the direction of the current
        increment. It is evaluated at the current time in both of its occurrences in
        the anisotropic length function, cf.
        :meth:`AnisotropicFractureDamageLength.damage_length`.

        The consequence is that the damage history cannot be accumulated incrementally:
        when :math:`m` rotates, past contributions change, so the slip history must be
        retained and re-projected. This is why
        :meth:`FractureDamageVariables.variables_stored_all_time_steps` keeps all time
        steps rather than a fixed window.

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
