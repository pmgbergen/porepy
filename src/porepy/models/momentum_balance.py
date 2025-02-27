"""This module contains the momentum balance model, including force balance at the
fractures.

It contains classes for equations, constitutive laws, variables, boundary conditions,
solution strategy, and initial conditions. The complete, runnable model is also based on
the contact mechanics model.

"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp
from porepy.models import contact_mechanics
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
        super().set_equations()
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        interfaces = self.mdg.interfaces(dim=self.nd - 1, codim=1)
        matrix_eq = self.momentum_balance_equation(matrix_subdomains)
        intf_eq = self.interface_force_balance_equation(interfaces)
        self.equation_system.set_equation(
            matrix_eq, matrix_subdomains, {"cells": self.nd}
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
            mortar_projection.primary_to_mortar_int()
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
    constitutive_laws.LinearElasticMechanicalStress,
    constitutive_laws.ConstantSolidDensity,
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

    def create_variables(self) -> None:
        """Introduces the following variables into the system:

        1. Displacement variable on all subdomains.
        2. Displacement variable on all interfaces with codimension 1.
        3. Contact traction variable on all fracture subdomains.

        """
        super().create_variables()

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

        # Discretization
        self.stress_keyword: str = "mechanics"
        """Keyword for stress term.

        Used to access discretization parameters and store discretization matrices.

        """

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
        self.update_boundary_condition(self.stress_keyword, self.bc_values_stress)

    def update_boundary_values_primary_variables(self) -> None:
        """Updates the displacement on the boundary, as the primary variable for
        mechanics."""
        super().update_boundary_values_primary_variables()
        self.update_boundary_condition(
            self.displacement_variable, self.bc_values_displacement
        )


class InitialConditionsMomentumBalance(pp.InitialConditionMixin):
    """Mixin for providing initial values for displacement."""

    displacement: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`VariablesMomentumBalance`."""

    interface_displacement: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """See :class:`VariablesMomentumBalance`."""

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for displacement, contact traction and interface
        displacement at iterate index 0 after the super-call.

        See also:

            - :meth:`ic_values_displacement`
            - :meth:`ic_values_interface_displacement`
            - :meth:`ic_values_contact_traction`

        """
        # Super call for compatibility with multi-physics.
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains():
            # Displacement is only defined on grids with ambient dimension.
            if sd.dim == self.nd:
                # Need to cast the return value to variable, because it is typed as
                # operator.
                self.equation_system.set_variable_values(
                    self.ic_values_displacement(sd),
                    [cast(pp.ad.Variable, self.displacement([sd]))],
                    iterate_index=0,
                )

        # interface dispacement is only defined on fractures with codimension 1
        for intf in self.mdg.interfaces(dim=self.nd - 1, codim=1):
            self.equation_system.set_variable_values(
                self.ic_values_interface_displacement(intf),
                [cast(pp.ad.Variable, self.interface_displacement([intf]))],
                iterate_index=0,
            )

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        """Initial values for displacement on the matrix grid.

        Override this method to customize the initialization.

        Note:
            This method will only be called with the matrix grid (ambient dimension),
            since displacement is only defined there.

        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial displacement values on the matrix with
            ``shape=(sd.num_cells * nd,)``. Defaults to zero array.

        """
        return np.zeros(sd.num_cells * self.nd)

    def ic_values_interface_displacement(self, intf: pp.MortarGrid) -> np.ndarray:
        """Initial values for interface displacement.

        Override this method to customize the initialization.

        Note:
            This method will only be called for interfaces with dimension ``nd-1`` and
            codimension 1.

        Parameters:
            intf: A mortar grid in the md-grid.

        Returns:
            The initial displacement values on the matrix with
            ``shape=(intf.num_cells * nd,)``. Defaults to zero array.

        """
        return np.zeros(intf.num_cells * self.nd)


# Note that we ignore a mypy error here. There are some inconsistencies in the method
# definitions of the mixins, related to the enforcement of keyword-only arguments. The
# type Callable is poorly supported, except if protocols are used and we really do not
# want to go there. Specifically, method definitions that contains a *, for instance,
#   def method(a: int, *, b: int) -> None: pass
# which should be types as Callable[[int, int], None], cannot be parsed by mypy.
# For this reason, we ignore the error here, and rely on the tests to catch any
# inconsistencies.
class MomentumBalance(  # type: ignore[misc]
    contact_mechanics.ContactMechanicsEquations,
    MomentumBalanceEquations,
    contact_mechanics.ContactTractionVariable,
    VariablesMomentumBalance,
    contact_mechanics.ConstitutiveLawsContactMechanics,
    ConstitutiveLawsMomentumBalance,
    BoundaryConditionsMomentumBalance,
    contact_mechanics.InitialConditionsContactTraction,
    InitialConditionsMomentumBalance,
    contact_mechanics.SolutionStrategyContactMechanics,
    SolutionStrategyMomentumBalance,
    # For clarity, the functionality of the FluidMixin is not really used in the pure
    # momentum balance model, but for unity of implementation (and to avoid some
    # technical programming related to the FluidMixin not always being present) it is
    # convenient to mix it in here.
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for mixed-dimensional momentum balance with contact mechanics."""
