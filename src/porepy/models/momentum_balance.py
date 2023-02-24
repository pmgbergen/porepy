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
from typing import Callable, Optional, Sequence

import numpy as np

import porepy as pp

from . import constitutive_laws

logger = logging.getLogger(__name__)


class MomentumBalanceEquations(pp.BalanceEquation):
    """Class for momentum balance equations and fracture deformation equations."""

    stress: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Stress on the grid faces. Provided by a suitable mixin class that specifies the
    physical laws governing the stress.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    internal_boundary_normal_to_outwards: Callable[[list[pp.Grid], int], pp.ad.Matrix]
    """Switch interface normal vectors to point outwards from the subdomain. Normally
    set by a mixin instance of :class:`porepy.models.geometry.ModelGeometry`.

    """
    fracture_stress: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Stress on the fracture faces. Provided by a suitable mixin class that specifies
    the physical laws governing the stress, see for instance
    :class:`~porepy.models.constitutive_laws.LinearElasticMechanicalStress` or
    :class:`~porepy.models.constitutive_laws.PressureStress`.

    """
    basis: Callable[[Sequence[pp.GridLike], int], list[pp.ad.Matrix]]
    """Basis for the local coordinate system. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    normal_component: Callable[[list[pp.Grid]], pp.ad.Matrix]
    """Operator giving the normal component of vectors. Normally defined in a mixin
    instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    tangential_component: Callable[[list[pp.Grid]], pp.ad.Matrix]
    """Operator giving the tangential component of vectors. Normally defined in a mixin
    instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Operator giving the displacement jump on fracture grids. Normally defined in a
    mixin instance of :class:`~porepy.models.models.ModelGeometry`.

    """
    contact_traction: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Contact traction variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    gap: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Gap of a fracture. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FracturedSolid`.

    """
    friction_bound: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction bound of a fracture. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FrictionBound`.

    """
    contact_mechanics_numerical_constant: Callable[[list[pp.Grid]], pp.ad.Scalar]
    """Numerical constant for contact mechanics. Normally provided by a mixin instance
    of :class:`~porepy.models.momuntum_balance.SolutionStrategyMomentumBalance`.

    """

    def set_equations(self):
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
        interfaces = self.mdg.interfaces(dim=self.nd - 1)
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

    def momentum_balance_equation(self, subdomains: list[pp.Grid]):
        """Momentum balance equation in the matrix.

        Inertial term is not included.

        Parameters:
            subdomains: List of subdomains where the force balance is defined. Only
            known usage
                is for the matrix domain(s).

        Returns:
            Operator for the force balance equation in the matrix.

        """
        accumulation = self.inertia(subdomains)
        stress = self.stress(subdomains)
        body_force = self.body_force(subdomains)
        equation = self.balance_equation(
            subdomains, accumulation, stress, body_force, dim=self.nd
        )
        equation.set_name("momentum_balance_equation")
        return equation

    def inertia(self, subdomains: list[pp.Grid]):
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
            * proj.face_prolongation(matrix_subdomains)
            * self.internal_boundary_normal_to_outwards(
                matrix_subdomains, dim=self.nd  # type: ignore[call-arg]
            )
            * self.stress(matrix_subdomains)
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

    def normal_fracture_deformation_equation(self, subdomains: list[pp.Grid]):
        """Equation for the normal component of the fracture deformation.

        This constraint equation enforces non-penetration of opposing fracture
        interfaces.

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
        # The normal component of the contact traction and the displacement jump
        t_n: pp.ad.Operator = nd_vec_to_normal * self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal * self.displacement_jump(subdomains)

        # Maximum function
        num_cells: int = sum([sd.num_cells for sd in subdomains])
        max_function = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.Array(np.zeros(num_cells), "zeros_frac")

        # The complimentarity condition
        equation: pp.ad.Operator = t_n + max_function(
            (-1) * t_n
            # EK: I will take care of typing of this term when we have a better name for
            # the method.
            - self.contact_mechanics_numerical_constant(subdomains)
            * (u_n - self.gap(subdomains)),
            zeros_frac,
        )
        equation.set_name("normal_fracture_deformation_equation")
        return equation

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Contact mechanics equation for the tangential constraints.

        The function reads
        .. math::
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)

        with `u` being displacement jump increments, `t` denoting tangential component
        and `b_p` the friction bound.

        For `b_p = 0`, the equation `C_t = 0` does not in itself imply `T_t = 0`, which
        is what the contact conditions require. The case is handled through the use of a
        characteristic function.

        Parameters:
            fracture_subdomains: List of fracture subdomains.

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
        # Ignore mypy complaint on unknown keyword argument
        tangential_basis = self.basis(
            subdomains, dim=self.nd - 1  # type: ignore[call-arg]
        )

        # To map a scalar to the tangential plane, we need to sum the basis vectors.
        # The individual basis functions have shape (Nc * (self.nd - 1), Nc), where
        # Nc is the total number of cells in the subdomain. The sum will have the same
        # shape, but the row corresponding to each cell will be non-zero in all rows
        # corresponding to the tangential basis vectors of this cell.
        scalar_to_tangential = sum([e_i for e_i in tangential_basis])

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential * self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential * self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Vectors needed to express the governing equations
        ones_frac = pp.ad.Array(np.ones(num_cells * (self.nd - 1)))
        zeros_frac = pp.ad.Array(np.zeros(num_cells))

        # Functions EK: Should we try to agree on a name convention for ad functions?
        # EK: Yes. Suggestions?
        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # With the active set method, the performance of the Newton solver is sensitive
        # to changes in state between sticking and sliding. To reduce the sensitivity to
        # round-off errors, we use a tolerance to allow for slight inaccuracies before
        # switching between the two cases.
        tol = 1e-5  # FIXME: Revisit this tolerance!
        # The characteristic function will evaluate to 1 if the argument is less than
        # the tolerance, and 0 otherwise.
        f_characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

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
        # tangential plane.
        c_num = sum([e_i * c_num_as_scalar * e_i.T for e_i in tangential_basis])

        # Combine the above into expressions that enter the equation
        tangential_sum = t_t + c_num * u_t_increment

        norm_tangential_sum = f_norm(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = f_max(self.friction_bound(subdomains), zeros_frac)
        b_p.set_name("bp")

        # Remove parentheses to make the equation more readable if possible
        bp_tang = (scalar_to_tangential * b_p) * tangential_sum

        maxbp_abs = scalar_to_tangential * f_max(b_p, norm_tangential_sum)
        characteristic: pp.ad.Operator = scalar_to_tangential * f_characteristic(b_p)
        characteristic.set_name("characteristic_function_of_b_p")

        # Compose the equation itself. The last term handles the case bound=0, in which
        # case t_t = 0 cannot be deduced from the standard version of the complementary
        # function (i.e. without the characteristic function). Filter out the other
        # terms in this case to improve convergence
        equation: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * t_t
        ) + characteristic * t_t
        equation.set_name("tangential_fracture_deformation_equation")
        return equation

    def body_force(self, subdomains: list[pp.Grid]):
        """Body force integrated over the subdomain cells.

        FIXME: See FluidMassBalanceEquations.fluid_source.

        Parameters:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force.

        """
        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros(num_cells * self.nd)
        source = pp.ad.Array(vals, "body_force")
        return source


class ConstitutiveLawsMomentumBalance(
    constitutive_laws.LinearElasticSolid,
    constitutive_laws.FracturedSolid,
    constitutive_laws.FrictionBound,
):
    """Class for constitutive equations for momentum balance equations."""

    def stress(self, subdomains: list[pp.Grid]):
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(subdomains)


class VariablesMomentumBalance:
    """
    Variables for mixed-dimensional deformation:
        Displacement in matrix and on fracture-matrix interfaces. Fracture contact
        traction.

    .. note::
        Implementation postponed till Veljko's more convenient SystemManager is available.

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
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    local_coordinates: Callable[[list[pp.Grid]], pp.ad.Matrix]
    """Mapping to local coordinates. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def create_variables(self):
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
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.interface_displacement_variable,
            interfaces=self.mdg.interfaces(dim=self.nd - 1),
        )
        self.equation_system.create_variables(
            dof_info={"cells": self.nd},
            name=self.contact_traction_variable,
            subdomains=self.mdg.subdomains(dim=self.nd - 1),
        )

    def displacement(self, subdomains: list[pp.Grid]):
        """Displacement in the matrix.

        Parameters:
            grids: List of subdomains or interface grids where the displacement is
                defined. Should be the matrix subdomains.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the subdomains is not equal to the ambient
                dimension of the problem.

        """
        if not all([sd.dim == self.nd for sd in subdomains]):
            raise ValueError(
                "Displacement is only defined in subdomains of dimension nd."
            )

        return self.equation_system.md_variable(self.displacement_variable, subdomains)

    def interface_displacement(self, interfaces: list[pp.MortarGrid]):
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

    def contact_traction(self, subdomains: list[pp.Grid]):
        """Fracture contact traction.

        Parameters:
            subdomains: List of subdomains where the contact traction is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Variable for fracture contact traction.

        """
        # Check that the subdomains are fractures
        for sd in subdomains:
            if sd.dim != self.nd - 1:
                raise ValueError("Contact traction only defined on fractures")

        return self.equation_system.md_variable(
            self.contact_traction_variable, subdomains
        )

    def displacement_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Displacement jump on fracture-matrix interfaces.

        Parameters:
            subdomains: List of subdomains where the displacement jump is defined.
                Should be a fracture subdomain.

        Returns:
            Operator for the displacement jump.

        Raises:
             AssertionError: If the subdomains are not fractures, i.e. have dimension
                `nd - 1`.

        """
        if not all([sd.dim == self.nd - 1 for sd in subdomains]):
            raise ValueError("Displacement jump only defined on fractures")

        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        # Only use matrix-fracture interfaces
        interfaces = [intf for intf in interfaces if intf.dim == self.nd - 1]
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, self.nd
        )
        # The displacement jmup is expressed in the local coordinates of the fracture.
        # First use the sign of the mortar sides to get a difference, then map first
        # from the interface to the fracture, and finally to the local coordinates.
        rotated_jumps: pp.ad.Operator = (
            self.local_coordinates(subdomains)
            * mortar_projection.mortar_to_secondary_avg
            * mortar_projection.sign_of_mortar_sides
            * self.interface_displacement(interfaces)
        )
        rotated_jumps.set_name("Rotated_displacement_jump")
        return rotated_jumps


class SolutionStrategyMomentumBalance(pp.SolutionStrategy):
    """Solution strategy for the momentum balance.

    At some point, this will be refined to be a more sophisticated (modularised)
    solution strategy class. More refactoring may be beneficial.

    Parameters:
        params: Parameters for the solution strategy.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    stiffness_tensor: Callable[[pp.Grid], pp.FourthOrderTensor]
    """Function that returns the stiffness tensor of a subdomain. Normally provided by a
    mixin of instance :class:`~porepy.models.constitutive_laws.LinearElasticSolid`.

    """
    bc_type_mechanics: Callable[[pp.Grid], pp.BoundaryConditionVectorial]
    """Function that returns the boundary condition type for the momentum problem.
    Normally provided by a mixin instance of
    :class:`~porepy.models.momentum_balance.BoundaryConditionsMomentumBalance`.

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
        # Zero for displacement and initial bc values for Biot
        super().initial_condition()
        # Contact as initial guess. Ensure traction is consistent with zero jump, which
        # follows from the default zeros set for all variables, specifically interface
        # displacement, by super method.
        num_frac_cells = sum(
            sd.num_cells for sd in self.mdg.subdomains(dim=self.nd - 1)
        )
        traction_vals = np.zeros((self.nd, num_frac_cells))
        traction_vals[-1] = self.solid.convert_units(-1, "Pa")
        self.equation_system.set_variable_values(
            traction_vals.ravel("F"),
            [self.contact_traction_variable],
            to_state=True,
            to_iterate=True,
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
    ) -> pp.ad.Scalar:
        """Numerical constant for the contact problem.

        The numerical constant is a cell-wise scalar, but we return a matrix to allow
        for automatic differentiation and left multiplication.

        Not sure about method location, but it is a property of the contact problem, and
        more solution strategy than material property or constitutive law.

        TODO: We need a more descritive name for this method.

        Parameters:
            subdomains: List of subdomains. Only the first is used.

        Returns:
            c_num: Numerical constant, as scalar.

        """
        # Conversion unnecessary for dimensionless parameters, but included as good
        # practice.
        val = self.solid.convert_units(1, "-")
        return pp.ad.Scalar(val, name="c_num")

    def _is_nonlinear_problem(self) -> bool:
        """
        If there is no fracture, the problem is usually linear. Overwrite this function
        if e.g. parameter nonlinearities are included.
        """
        return self.mdg.dim_min() < self.nd


class BoundaryConditionsMomentumBalance:
    """Boundary conditions for the momentum balance."""

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    domain_boundary_sides: Callable[[pp.Grid], pp.domain.DomainSides]

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions.


        Parameters:
            sd: Subdomain grid.

        Returns:
            bc: Boundary condition representation. Dirichlet on all global boundaries,
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

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the momentum balance.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            bc_values: Array of boundary condition values, zero by default. If combined
            with transient problems in e.g. Biot, this should be a
            :class:`pp.ad.TimeDependentArray` (or a variable following BoundaryGrid
            extension).

        """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return pp.wrap_as_ad_array(0, num_faces * self.nd, "bc_vals_mechanics")


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
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for mixed-dimensional momentum balance with contact mechanics."""

    pass
