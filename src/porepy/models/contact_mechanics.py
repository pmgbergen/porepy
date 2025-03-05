"""Model for contact mechanics.

Implements the contact mechanics equations for fracture deformation. The model consists
of classes for the equations, constitutive laws, variables, initial conditions, boundary
conditions, and solution strategy. The model is primarily intended to be used in
combination with the momentum balance model, but can be used as a standalone model.

"""

from functools import partial
from typing import Callable, Optional, cast

import numpy as np

import porepy as pp
from porepy.models import constitutive_laws
from porepy.models.abstract_equations import VariableMixin


class ContactMechanicsEquations(pp.BalanceEquation):
    """Class for contact mechanics equations governing fracture deformation."""

    nd: int
    """Ambient dimension of the problem."""
    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator giving the displacement jump on fracture grids. Normally defined in a
    mixin instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    plastic_displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator giving the plastic displacement jump on fracture grids. Normally defined
    in a mixin instance of
    :class:`~porepy.models.constitutive_laws.DisplacementJump`.
    """
    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Contact traction variable. Normally defined in a mixin instance of
    :class:`~porepy.models.contact_mechanics.ContactTractionVariable`.

    """
    fracture_gap: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Gap of a fracture. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FractureGap`.

    """
    friction_bound: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction bound of a fracture. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.CoulombFrictionBound`.

    """
    contact_mechanics_numerical_constant: Callable[[list[pp.Grid]], pp.ad.Scalar]
    """Numerical constant for contact mechanics. Normally provided by a mixin instance
    of :class:`~porepy.models.momuntum_balance.SolutionStrategyMomentumBalance`.

    """
    contact_mechanics_open_state_characteristic: Callable[
        [list[pp.Grid]], pp.ad.Operator
    ]
    """Characteristic function used in the tangential contact mechanics relation.
    Can be interpreted as an indicator of the fracture cells in the open state.
    Normally provided by a mixin instance of
    :class:`~porepy.models.contact_mechanics.SolutionStrategyMomentumBalance`.

    """

    def set_equations(self) -> None:
        """Set the contact mechanics equations to the equation system."""
        super().set_equations()
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        # We split the contact mechanics equations into two parts, for the normal and
        # tangential components for convenience.
        fracture_eq_normal = self.normal_fracture_deformation_equation(
            fracture_subdomains
        )
        fracture_eq_tangential = self.tangential_fracture_deformation_equation(
            fracture_subdomains
        )
        self.equation_system.set_equation(
            fracture_eq_normal, fracture_subdomains, {"cells": 1}
        )
        self.equation_system.set_equation(
            fracture_eq_tangential, fracture_subdomains, {"cells": self.nd - 1}
        )

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Equation for the normal component of the contact mechanics.

        This constraint equation enforces non-penetration of opposing fracture
        interfaces. The equation is dimensionless, as we use nondimensionalized
        contact traction.

        Parameters:
            subdomains: List of subdomains where the contact mechanics equation is
            defined.

        Returns:
            Operator for the normal deformation equation.

        """
        # The lines below is an implementation of equations (24) and (26) in the paper
        #
        # Berge et al. (2020): Finite volume discretization for poroelastic media with
        #   fractures modeled by contact mechanics (IJNME, DOI: 10.1002/nme.6238). The
        #
        # Note that:
        #  - We do not directly implement the matrix elements of the contact traction
        #    as are derived by Berge in their equations (28)-(32). Instead, we directly
        #    implement the complimentarity function, and let the AD framework take care
        #    of the derivatives.
        #  - Related to the previous point, we do not implement the regularization that
        #    is discussed in Section 3.2.1 of the paper.

        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump.
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells), "zeros_frac")

        # The complimentarity condition
        equation: pp.ad.Operator = t_n + max_function(
            pp.ad.Scalar(-1.0) * t_n
            - self.contact_mechanics_numerical_constant(subdomains)
            * (u_n - self.fracture_gap(subdomains)),
            zeros_frac,
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the tangential constraints.

        The equation is dimensionless, as we use nondimensionalized contact traction.
        The function reads
        .. math::
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)

        with `u` being displacement jump increments, `t` denoting tangential component
        and `b_p` the friction bound.

        For `b_p = 0`, the equation `C_t = 0` does not in itself imply `T_t = 0`, which
        is what the contact conditions require. The case is handled through the use of a
        characteristic function.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            complementary_eq: Contact mechanics equation for the tangential constraints.

        """
        # The lines below is an implementation of equations (25) and (27) in the paper
        #
        # Berge et al. (2020): Finite volume discretization for poroelastic media with
        #   fractures modeled by contact mechanics (IJNME, DOI: 10.1002/nme.6238). The
        #
        # Note that:
        #  - We do not directly implement the matrix elements of the contact traction
        #    as are derived by Berge in their equations (28)-(32). Instead, we directly
        #    implement the complimentarity function, and let the AD framework take care
        #    of the derivatives.
        #  - Related to the previous point, we do not implement the regularization that
        #    is discussed in Section 3.2.1 of the paper.

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        # Basis vectors for the tangential components. This is a list of Ad matrices,
        # each of which represents a cell-wise basis vector which is non-zero in one
        # dimension (and this is known to be in the tangential plane of the subdomains).
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        # To map a scalar to the tangential plane, we need to sum the basis vectors. The
        # individual basis vectors can be represented as projection matrices of shape
        # (Nc * (self.nd - 1), Nc), where Nc is the total number of cells in the
        # subdomain. The matrix representation of the sum has the same shape, but the
        # row corresponding to each cell will be non-zero in all rows corresponding to
        # the tangential basis vectors of this cell.
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        # Variables: The tangential component of the contact traction and the plastic
        # displacement jump.
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.plastic_displacement_jump(
            subdomains
        )
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.DenseArray(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # The numerical constant is used to loosen the sensitivity in the transition
        # between sticking and sliding. Expanding using only left multiplication to with
        # scalar_to_tangential does not work for an array, unlike the operators below.
        # Arrays need right multiplication as well.
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        # The numerical parameter is a cell-wise scalar, or a single scalar common for
        # all cells. In both cases, it must be extended to a vector quantity to be used
        # in the equation (multiplied from the right). Do this by multiplying with the
        # sum of the tangential basis vectors. Then take a Hadamard product with the
        # tangential displacement jump and add to the tangential component of the
        # contact traction to arrive at the expression that enters the equation.
        basis_sum = pp.ad.sum_projection_list(tangential_basis)
        tangential_sum = t_t + (basis_sum @ c_num_as_scalar) * u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        maxbp_abs = scalar_to_tangential @ f_max(b_p, norm_tangential_sum)

        # The characteristic function below reads "1 if (abs(b_p) < tol) else 0".
        # With the active set method, the performance of the Newton solver is sensitive
        # to changes in state between sticking and sliding. To reduce the sensitivity to
        # round-off errors, we use a tolerance to allow for slight inaccuracies before
        # switching between the two cases. The tolerance is a numerical method parameter
        # and can be tailored.
        characteristic = self.contact_mechanics_open_state_characteristic(subdomains)

        # Compose the equation itself. The last term handles the case bound=0, in which
        # case t_t = 0 cannot be deduced from the standard version of the complementary
        # function (i.e. without the characteristic function). Filter out the other
        # terms in this case to improve convergence
        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation


class ConstitutiveLawsContactMechanics(
    constitutive_laws.FractureGap,
    constitutive_laws.CoulombFrictionBound,
    constitutive_laws.DisplacementJump,
    constitutive_laws.DimensionReduction,
    constitutive_laws.ElasticModuli,
    constitutive_laws.ElasticTangentialFractureDeformation,
):
    """Class for constitutive equations for contact mechanics."""


class InterfaceDisplacementArray(pp.PorePyModel):
    """Displacement on interfaces as a TimeDependentDenseArray.

    Intended usage is to define the displacement on the interfaces as a parameter, not a
    primary variable.

    """

    interface_displacement_parameter_key: str
    """Key for the interface displacement parameter."""
    nd: int
    """Ambient dimension of the problem."""

    def interface_displacement(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Displacement on interfaces [m].

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the displacement on the interfaces.

        """
        return pp.ad.TimeDependentDenseArray(
            self.interface_displacement_parameter_key, interfaces
        )

    def interface_displacement_parameter_values(
        self, interface: pp.MortarGrid
    ) -> np.ndarray:
        """Displacement on interfaces [m].

        Parameters:
            interface: Single interface grid.

        Returns:
            Array representing the displacement on the interface of shape (nd, num_cells).

        """
        return np.zeros((self.nd, interface.num_cells))

    def update_time_dependent_ad_arrays(self) -> None:
        """Update values of external sources and boundary conditions."""
        super().update_time_dependent_ad_arrays()  # type: ignore[misc]

        name = self.interface_displacement_parameter_key
        for intf, data in self.mdg.interfaces(return_data=True):
            if pp.ITERATE_SOLUTIONS in data and name in data[pp.ITERATE_SOLUTIONS]:
                # Use the values at the unknown time step from the previous time step.
                vals = pp.get_solution_values(name=name, data=data, iterate_index=0)
            else:
                # No current value stored. The method was called during the
                # initialization.
                vals = self.interface_displacement_parameter_values(intf).ravel(
                    order="F"
                )

            # Before setting the new, most recent time step, shift the stored values
            # backwards in time.
            pp.shift_solution_values(
                name=name,
                data=data,
                location=pp.TIME_STEP_SOLUTIONS,
                max_index=len(self.time_step_indices),
            )
            # Set the values of current time to most recent previous time.
            pp.set_solution_values(name=name, values=vals, data=data, time_step_index=0)

            # Set the unknown time step values.
            vals = self.interface_displacement_parameter_values(intf).ravel(order="F")
            pp.set_solution_values(name=name, values=vals, data=data, iterate_index=0)


class ContactTractionVariable(VariableMixin):
    """Contact traction variable for contact mechanics."""

    contact_traction_variable: str
    """Name of the primary variable representing the contact traction on a fracture
    subdomain. Normally defined in a mixin of instance
    :class:`~porepy.models.contact_mechanics.SolutionStrategyContactMechanics`.

    """

    def create_variables(self) -> None:
        """Introduces the contact traction variable to the equation system."""
        super().create_variables()

        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.contact_traction_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
            tags={"si_units": "-"},
        )

    def contact_traction(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fracture contact traction [-].

        Parameters:
            subdomains: List of subdomains where the contact traction is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture contact traction.

        """
        # Check that the subdomains are fractures
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Contact traction only defined on fractures")

        return self.equation_system.md_variable(
            self.contact_traction_variable, subdomains
        )


class InitialConditionsContactTraction(pp.InitialConditionMixin):
    """Mixin for providing initial values for contact traction."""

    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """See :class:`VariablesMomentumBalance`."""

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values contact traction at iterate index 0 after the
        super-call.
        """
        # Super call for compatibility with multi-physics.
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains(dim=self.nd - 1):
            # Contact traction is only defined on fractures
            self.equation_system.set_variable_values(
                self.ic_values_contact_traction(sd),
                [cast(pp.ad.Variable, self.contact_traction([sd]))],
                iterate_index=0,
            )

    def ic_values_contact_traction(self, sd: pp.Grid) -> np.ndarray:
        """Initial values for the contact traction variable.

        Override this method to customize the initialization.

        Note:
            This method will only be called for grids with dimension ``nd-1``.

        Important:
            By default, this initialization does not return trivial values.
            The contact traction is set to zero in the tangential direction, and -1
            (that is, in contact) in the normal direction.

            This initialization is consistent with the zero displacement on matrix and
            interfaces.

        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial displacement values on the matrix with
            ``shape=(sd.num_cells * nd,)``. Defaults to zero array.

        """
        # Contact as initial guess. Ensure traction is consistent with zero jump, which
        # follows from the default zeros set for all variables, specifically interface
        # displacement.
        num_frac_cells = sd.num_cells
        traction_vals = np.zeros((self.nd, num_frac_cells))
        traction_vals[-1] = -1  # Unitary nondimensional traction.
        return traction_vals.ravel("F")


class BoundaryConditionsContactMechanics(pp.BoundaryConditionMixin):
    """No boundary values for contact mechanics.

    The class is nevertheless required for compatibility with contracts
    between model classes and their run methods.

    """


class SolutionStrategyContactMechanics(pp.SolutionStrategy):
    """Solution strategy for contact mechanics.

    Parameters:
        params: Dictionary of model parameters.

    """

    characteristic_displacement: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Characteristic displacement of the problem. Normally defined in a mixin
    instance of :class:`~porepy.models.constitutive_laws.ElasticModuli`.

    """
    friction_bound: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction bound of a fracture. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.CoulombFrictionBound`.

    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        self.contact_traction_variable: str = "contact_traction"
        """Name of the contact traction variable."""

        self.interface_displacement_parameter_key: str = (
            "interface_displacement_parameter"
        )
        """Key for the interface displacement parameter."""

    def contact_mechanics_numerical_constant(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Numerical constant for the contact problem [m^-1].

        A physical interpretation of this constant is a characteristic length of
        the fracture, as it appears as a scaling of displacement jumps when
        comparing to nondimensionalized contact tractions.

        Parameters:
            subdomains: List of subdomains. Only the first is used.

        Returns:
            c_num: Numerical constant.

        """
        # Interpretation (EK):
        # The scaling factor should not be too large, otherwise the contact problem
        # may be discretized wrongly. I therefore introduce a safety factor here; its
        # value is somewhat arbitrary.
        softening_factor = pp.ad.Scalar(self.numerical.contact_mechanics_scaling)

        constant = softening_factor / self.characteristic_displacement(subdomains)
        constant.set_name("Contact_mechanics_numerical_constant")
        return constant

    def contact_mechanics_open_state_characteristic(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        r"""Characteristic function used in the tangential contact mechanics relation.
        Can be interpreted as an indicator of the fracture cells in the open state.
        Used to make the problem well-posed in the case b_p is zero.

        The function reads
        .. math::
            \begin{equation}
            \text{characteristic} =
            \begin{cases}
                1 & \\text{if }~~ |b_p| < tol  \\
                0 & \\text{otherwise.}
            \end{cases}
            \end{equation}
        or simply `1 if (abs(b_p) < tol) else 0`

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            characteristic: Characteristic function.

        """

        # Basis vectors for the tangential components. This is a list of Ad matrices,
        # each of which represents a cell-wise basis vector which is non-zero in one
        # dimension (and this is known to be in the tangential plane of the subdomains).
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        # To map a scalar to the tangential plane, we need to sum the basis vectors. The
        # individual basis functions can be represented as projection matrices of size
        # (Nc * (self.nd - 1), Nc), where Nc is # the total number of cells in the
        # subdomain. The sum of basis vectors can likewise be represented as a matrix of
        # the same shape, but the row corresponding to each cell will be non-zero in all
        # rows corresponding to the tangential basis vectors of this cell.
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        # With the active set method, the performance of the Newton solver is sensitive
        # to changes in state between sticking and sliding. To reduce the sensitivity to
        # round-off errors, we use a tolerance to allow for slight inaccuracies before
        # switching between the two cases.
        tol = self.numerical.open_state_tolerance
        # The characteristic function will evaluate to 1 if the argument is less than
        # the tolerance, and 0 otherwise.
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        # Composing b_p = max(friction_bound, 0).
        num_cells = sum([sd.num_cells for sd in subdomains])
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        characteristic: pp.ad.Operator = scalar_to_tangential @ f_characteristic(b_p)
        characteristic.set_name("characteristic_function_of_b_p")
        return characteristic

    def _is_nonlinear_problem(self) -> bool:
        """The contact mechanics problem is nonlinear."""
        return True


class ContactMechanics(
    ContactMechanicsEquations,
    # Keep interface displacement array separate from other constituive laws
    # to avoid conflicts with other mixins, e.g. in momentum balance.
    InterfaceDisplacementArray,
    ConstitutiveLawsContactMechanics,
    ContactTractionVariable,
    InitialConditionsContactTraction,
    BoundaryConditionsContactMechanics,
    SolutionStrategyContactMechanics,
    # For clarity, the functionality of the FluidMixin is not really used in the pure
    # contact mechanics model, but for unity of implementation (and to avoid some
    # technical programming related to the FluidMixin not always being present) it is
    # convenient to mix it in here.
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    pass
