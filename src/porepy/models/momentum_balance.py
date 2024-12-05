"""
Class types:
    - MomentumBalanceEquations defines subdomain and interface equations through the
    terms entering. Momentum balance between opposing fracture interfaces is imposed.

Notes:
    - The class MomentumBalanceEquations is a mixin class, and should be inherited by a
      class that defines the variables and discretization.

    - Refactoring needed for constitutive equations. Modularisation and moving to the
      library.

"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp
from porepy.models.abstract_equations import VariableMixin

from . import constitutive_laws

logger = logging.getLogger(__name__)


class MomentumBalanceEquations(pp.BalanceEquation):
    """Class for momentum balance equations and fracture deformation equations."""

    stress: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Stress on the grid faces. Provided by a suitable mixin class that specifies the
    physical laws governing the stress.

    """
    fracture_stress: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Stress on the fracture faces. Provided by a suitable mixin class that specifies
    the physical laws governing the stress, see for instance
    :class:`~porepy.models.constitutive_laws.LinearElasticMechanicalStress` or
    :class:`~porepy.models.constitutive_laws.PressureStress`.

    """
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
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

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
    :class:`~porepy.models.momuntum_balance.SolutionStrategyMomentumBalance`.

    """
    gravity_force: Callable[[list[pp.Grid] | list[pp.MortarGrid], str], pp.ad.Operator]
    """Gravity force. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.GravityForce` or
    :class:`~porepy.models.constitutive_laws.ZeroGravityForce`.

    """

    def set_equations(self) -> None:
        """Set equations for the subdomains and interfaces.

        The following equations are set:
            - Momentum balance in the matrix.
            - Force balance between fracture interfaces.
            - Deformation constraints for fractures, split into normal and tangential
              part.

        See individual equation methods for details.

        """
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        interfaces = self.mdg.interfaces(dim=self.nd - 1, codim=1)
        matrix_eq = self.momentum_balance_equation(matrix_subdomains)
        # We split the fracture deformation equations into two parts, for the normal and
        # tangential components for convenience.
        fracture_eq_normal = self.normal_fracture_deformation_equation(
            fracture_subdomains
        )
        fracture_eq_tangential = self.tangential_fracture_deformation_equation(
            fracture_subdomains
        )
        intf_eq = self.interface_force_balance_equation(interfaces)
        self.equation_system.set_equation(
            matrix_eq, matrix_subdomains, {"cells": self.nd}
        )
        self.equation_system.set_equation(
            fracture_eq_normal, fracture_subdomains, {"cells": 1}
        )
        self.equation_system.set_equation(
            fracture_eq_tangential, fracture_subdomains, {"cells": self.nd - 1}
        )
        self.equation_system.set_equation(intf_eq, interfaces, {"cells": self.nd})

    def momentum_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Momentum balance equation in the matrix.

        Inertial term is not included.

        Parameters:
            subdomains: List of subdomains where the force balance is defined. Only
                known usage is for the matrix domain(s).

        Returns:
            Operator for the force balance equation in the matrix.

        """
        accumulation = self.inertia(subdomains)
        # By the convention of positive tensile stress, the balance equation is
        # acceleration - stress = body_force. The balance_equation method will *add* the
        # surface term (stress), so we need to multiply by -1.
        stress = pp.ad.Scalar(-1) * self.stress(subdomains)
        body_force = self.body_force(subdomains)

        equation = self.balance_equation(
            subdomains, accumulation, stress, body_force, dim=self.nd
        )
        equation.set_name("momentum_balance_equation")
        return equation

    def inertia(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Inertial term [m^2/s].

        Added here for completeness, but not used in the current implementation. Be
        aware that the elasticity discretization has employed herein has, as far as we
        know, never been used to solve a problem with inertia. Thus, if inertia is added
        in a submodel, proceed with caution. In addition to overriding this method, it
        would also be necessary to add the inertial term to the balance equation
        :meth:`momentum_balance_equation`.

        Parameters:
            subdomains: List of subdomains where the inertial term is defined.

        Returns:
            Operator for the inertial term.

        """
        return pp.ad.Scalar(0)

    def interface_force_balance_equation(
        self,
        interfaces: list[pp.MortarGrid],
    ) -> pp.ad.Operator:
        """Momentum balance equation at matrix-fracture interfaces.

        Parameters:
            interfaces: Fracture-matrix interfaces.

        Returns:
            Operator representing the force balance equation.

        Raises:
            ValueError: If an interface is not a fracture-matrix interface.

        """
        # Check that the interface is a fracture-matrix interface.
        for interface in interfaces:
            if interface.dim != self.nd - 1:
                raise ValueError("Interface must be a fracture-matrix interface.")

        subdomains = self.interfaces_to_subdomains(interfaces)
        # Split into matrix and fractures. Sort on dimension to allow for multiple
        # matrix domains. Otherwise, we could have picked the first element.
        matrix_subdomains = [sd for sd in subdomains if sd.dim == self.nd]

        # Geometry related
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        proj = pp.ad.SubdomainProjections(subdomains, self.nd)

        # Contact traction from primary grid and mortar displacements (via primary grid).
        # Spelled out for clarity:
        #   1) The sign of the stress on the matrix subdomain is corrected so that all
        #      stress components point outwards from the matrix (or inwards, EK is not
        #      completely sure, but the point is the consistency).
        #   2) The stress is prolonged from the matrix subdomains to all subdomains seen
        #      by the mortar grid (that is, the matrix and the fracture).
        #   3) The stress is projected to the mortar grid.
        contact_from_primary_mortar = (
            mortar_projection.primary_to_mortar_int
            @ proj.face_prolongation(matrix_subdomains)
            @ self.internal_boundary_normal_to_outwards(matrix_subdomains, dim=self.nd)
            @ self.stress(matrix_subdomains)
        )
        # Traction from the actual contact force.
        traction_from_secondary = self.fracture_stress(interfaces)
        # The force balance equation. Note that the force from the fracture is a
        # traction, not a stress, and must be scaled with the area of the interface.
        # This is not the case for the force from the matrix, which is a stress.
        force_balance_eq: pp.ad.Operator = (
            contact_from_primary_mortar
            + self.volume_integral(traction_from_secondary, interfaces, dim=self.nd)
        )
        force_balance_eq.set_name("interface_force_balance_equation")
        return force_balance_eq

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Equation for the normal component of the fracture deformation.

        This constraint equation enforces non-penetration of opposing fracture
        interfaces. The equation is dimensionless, as we use nondimensionalized
        contact traction.

        Parameters:
            subdomains: List of subdomains where the normal deformation equation is
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
        # individual basis functions have shape (Nc * (self.nd - 1), Nc), where Nc is
        # the total number of cells in the subdomain. The sum will have the same shape,
        # but the row corresponding to each cell will be non-zero in all rows
        # corresponding to the tangential basis vectors of this cell. EK: mypy insists
        # that the argument to sum should be a list of booleans. Ignore this error.
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )

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
        # between sticking and sliding.
        # Expanding using only left multiplication to with scalar_to_tangential does not
        # work for an array, unlike the operators below. Arrays need right
        # multiplication as well.
        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        # The numerical parameter is a cell-wise scalar which must be extended to a
        # vector quantity to be used in the equation (multiplied from the right).
        # Spelled out, from the right: Restrict the vector quantity to one dimension in
        # the tangential plane (e_i.T), multiply with the numerical parameter, prolong
        # to the full vector quantity (e_i), and sum over all all directions in the
        # tangential plane. EK: mypy insists that the argument to sum should be a list
        # of booleans. Ignore this error.
        c_num = pp.ad.sum_operator_list(
            [e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis]
        )

        # Combine the above into expressions that enter the equation. c_num will
        # effectively be a sum of SparseArrays, thus we use a matrix-vector product @
        tangential_sum = t_t + c_num @ u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        # Remove parentheses to make the equation more readable if possible. The product
        # between (the SparseArray) scalar_to_tangential and b_p is of matrix-vector
        # type (thus @), and the result is then multiplied elementwise with
        # tangential_sum.
        bp_tang = (scalar_to_tangential @ b_p) * tangential_sum

        # For the use of @, see previous comment.
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

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force integrated over the subdomain cells.

        Parameters:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force [kg*m*s^-2].

        """
        return self.volume_integral(
            self.gravity_force(subdomains, "solid"), subdomains, dim=self.nd
        )


class ConstitutiveLawsMomentumBalance(
    constitutive_laws.ZeroGravityForce,
    constitutive_laws.ElasticModuli,
    constitutive_laws.ElasticTangentialFractureDeformation,
    constitutive_laws.LinearElasticMechanicalStress,
    constitutive_laws.ConstantSolidDensity,
    constitutive_laws.FractureGap,
    constitutive_laws.CoulombFrictionBound,
    constitutive_laws.DisplacementJump,
    constitutive_laws.DimensionReduction,
):
    """Class for constitutive equations for momentum balance equations."""

    def stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(domains)


class VariablesMomentumBalance(VariableMixin):
    """Variables for mixed-dimensional deformation.

    The variables are:
        - Displacement in matrix
        - Displacement on fracture-matrix interfaces
        - Fracture contact traction.

    """

    displacement_variable: str
    """Name of the primary variable representing the displacement in subdomains.
    Normally defined in a mixin of instance
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    interface_displacement_variable: str
    """Name of the primary variable representing the displacement on an interface.
    Normally defined in a mixin of instance
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """
    contact_traction_variable: str
    """Name of the primary variable representing the contact traction on a fracture
    subdomain. Normally defined in a mixin of instance
    :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.

    """

    def create_variables(self) -> None:
        """Set variables for the subdomains and interfaces.

        The following variables are set:
            - Displacement in the matrix.
            - Displacement on fracture-matrix interfaces.
            - Fracture contact traction.
        See individual variable methods for details.
        """

        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.displacement_variable,
            subdomains=self.mdg.subdomains(dim=self.nd),
            tags={"si_units": "m"},
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.interface_displacement_variable,
            interfaces=self.mdg.interfaces(dim=self.nd - 1, codim=1),
            tags={"si_units": "m"},
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.contact_traction_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
            tags={"si_units": "-"},
        )

    def displacement(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Displacement in the matrix.

        Parameters:
            domains: List of subdomains or interface grids where the displacement is
                defined. Should be the matrix subdomains.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the subdomains is not equal to the ambient
                dimension of the problem.
            ValueError: If the method is called on a mixture of grids and boundary
                grids

        """
        if len(domains) == 0 or all(
            isinstance(grid, pp.BoundaryGrid) for grid in domains
        ):
            domains = cast(Sequence[pp.BoundaryGrid], domains)
            return self.create_boundary_operator(
                name=self.displacement_variable, domains=domains
            )
        # Check that the subdomains are grids
        if not all(isinstance(grid, pp.Grid) for grid in domains):
            raise ValueError(
                "Method called on a mixture of subdomain and boundary grids."
            )
        # Now we can cast to Grid
        domains = cast(list[pp.Grid], domains)

        if not all([grid.dim == self.nd for grid in domains]):
            raise ValueError(
                "Displacement is only defined in subdomains of dimension nd."
            )

        return self.equation_system.md_variable(self.displacement_variable, domains)

    def interface_displacement(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Variable:
        """Displacement on fracture-matrix interfaces.

        Parameters:
            interfaces: List of interface grids where the displacement is defined.
                Should be between the matrix and fractures.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the interfaces is not equal to the ambient
                dimension of the problem minus one.

        """
        if not all([intf.dim == self.nd - 1 for intf in interfaces]):
            raise ValueError(
                "Interface displacement is only defined on interfaces of dimension "
                "nd - 1."
            )

        return self.equation_system.md_variable(
            self.interface_displacement_variable, interfaces
        )

    def contact_traction(self, subdomains: list[pp.Grid]) -> pp.ad.Variable:
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


class SolutionStrategyMomentumBalance(pp.SolutionStrategy):
    """Solution strategy for the momentum balance.

    At some point, this will be refined to be a more sophisticated (modularised)
    solution strategy class. More refactoring may be beneficial.

    Parameters:
        params: Parameters for the solution strategy.

    """

    stiffness_tensor: Callable[[pp.Grid], pp.FourthOrderTensor]
    """Function that returns the stiffness tensor of a subdomain. Normally provided by a
    mixin of instance :class:`~porepy.models.constitutive_laws.ElasticModuli`.

    """
    bc_type_mechanics: Callable[[pp.Grid], pp.BoundaryConditionVectorial]
    """Function that returns the boundary condition type for the momentum problem.
    Normally provided by a mixin instance of
    :class:`~porepy.models.momentum_balance.BoundaryConditionsMomentumBalance`.

    """
    friction_bound: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction bound of a fracture. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.CoulombFrictionBound`.

    """
    characteristic_displacement: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Characteristic displacement of the problem. Normally defined in a mixin
    instance of :class:`~porepy.models.constitutive_laws.ElasticModuli`.

    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        # Variables
        self.displacement_variable: str = "u"
        """Name of the displacement variable."""

        self.interface_displacement_variable: str = "u_interface"
        """Name of the displacement variable on fracture-matrix interfaces."""

        self.contact_traction_variable: str = "t"
        """Name of the contact traction variable."""

        # Discretization
        self.stress_keyword: str = "mechanics"
        """Keyword for stress term.

        Used to access discretization parameters and store discretization matrices.

        """

    def initial_condition(self) -> None:
        """Set initial guess for the variables.

        The displacement is set to zero in the Nd-domain, and at the fracture interfaces
        The displacement jump is thereby also zero.

        The contact pressure is set to zero in the tangential direction, and -1 (that
        is, in contact) in the normal direction.

        """
        # Zero for displacement and initial bc values.
        super().initial_condition()

        # Contact as initial guess. Ensure traction is consistent with zero jump, which
        # follows from the default zeros set for all variables, specifically interface
        # displacement, by super method.
        num_frac_cells = sum(
            sd.num_cells for sd in self.mdg.subdomains(dim=self.nd - 1)
        )
        traction_vals = np.zeros((self.nd, num_frac_cells))
        traction_vals[-1] = -1  # Unitary nondimensional traction.
        self.equation_system.set_variable_values(
            traction_vals.ravel("F"),
            [self.contact_traction_variable],
            time_step_index=0,
            iterate_index=0,
        )

    def set_discretization_parameters(self) -> None:
        """Set discretization parameters for the simulation."""

        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                pp.initialize_data(
                    sd,
                    data,
                    self.stress_keyword,
                    {
                        "bc": self.bc_type_mechanics(sd),
                        "fourth_order_tensor": self.stiffness_tensor(sd),
                    },
                )

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
        # individual basis functions have shape (Nc * (self.nd - 1), Nc), where Nc is
        # the total number of cells in the subdomain. The sum will have the same shape,
        # but the row corresponding to each cell will be non-zero in all rows
        # corresponding to the tangential basis vectors of this cell.
        scalar_to_tangential = pp.ad.sum_operator_list(
            [e_i for e_i in tangential_basis]
        )

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
        """
        If there is no fracture, the problem is usually linear. Overwrite this function
        if e.g. parameter nonlinearities are included.
        """
        return self.mdg.dim_min() < self.nd


class BoundaryConditionsMomentumBalance(pp.BoundaryConditionMixin):
    """Boundary conditions for the momentum balance."""

    displacement_variable: str

    stress_keyword: str

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Boundary condition representation. Dirichlet on all global boundaries,
            Dirichlet also on fracture faces.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the fracture
        # faces.
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values for the Dirichlet boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the displacement
            values on the provided boundary grid.

        """
        return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Stress values for the Nirichlet boundary condition.

        Parameters:
            boundary_grid: Boundary grid to evaluate values on.

        Returns:
            An array with shape (boundary_grid.num_cells,) containing the stress values
            on the provided boundary grid.

        """
        return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

    def update_all_boundary_conditions(self) -> None:
        """Set values for the displacement and the stress on boundaries."""
        super().update_all_boundary_conditions()
        self.update_boundary_condition(
            self.displacement_variable, self.bc_values_displacement
        )
        self.update_boundary_condition(self.stress_keyword, self.bc_values_stress)


# Note that we ignore a mypy error here. There are some inconsistencies in the method
# definitions of the mixins, related to the enforcement of keyword-only arguments. The
# type Callable is poorly supported, except if protocols are used and we really do not
# want to go there. Specifically, method definitions that contains a *, for instance,
#   def method(a: int, *, b: int) -> None: pass
# which should be types as Callable[[int, int], None], cannot be parsed by mypy.
# For this reason, we ignore the error here, and rely on the tests to catch any
# inconsistencies.
class MomentumBalance(  # type: ignore[misc]
    MomentumBalanceEquations,
    VariablesMomentumBalance,
    ConstitutiveLawsMomentumBalance,
    BoundaryConditionsMomentumBalance,
    SolutionStrategyMomentumBalance,
    # For clarity, the functionality of the FluidMixin is not really used in the pure
    # momentum balance model, but for unity of implementation (and to avoid some
    # technical programing related to the FluidMixin not always being present) it is
    # convenient to mix it in here.
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for mixed-dimensional momentum balance with contact mechanics."""
