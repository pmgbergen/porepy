"""Coupling of mass and momentum balance to obtain poromechanics equations.

The module only contains what is needed for the coupling, the two individual subproblems
are defined elsewhere.

The main changes to the equations are achieved by changing the constitutive laws for
porosity and stress. The former aquires a pressure dependency and an additional
:math:`\alpha`\nabla\cdot\mathbf{u} term, while the latter is modified to include a
isotropic pressure term :math:`\alpha p \mathbf{I}`.

Suggested references (TODO: add more, e.g. Inga's in prep):
    - Coussy, 2004, https://doi.org/10.1002/0470092718.
    - Garipov and Hui, 2019, https://doi.org/10.1016/j.ijrmms.2019.104075.

"""
from __future__ import annotations

from typing import Callable

import numpy as np

import porepy as pp
import porepy.models.fluid_mass_balance as mass
import porepy.models.momentum_balance as momentum
from porepy.numerics.ad.equation_system import set_solution_values


class ConstitutiveLawsPoromechanics(
    # Combined effects
    pp.constitutive_laws.DisplacementJumpAperture,
    pp.constitutive_laws.BiotCoefficient,
    pp.constitutive_laws.PressureStress,
    pp.constitutive_laws.PoroMechanicsPorosity,
    # Fluid mass balance subproblem
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.FluidDensityFromPressure,
    pp.constitutive_laws.ConstantViscosity,
    # Mechanical subproblem
    pp.constitutive_laws.LinearElasticSolid,
    pp.constitutive_laws.FracturedSolid,
    pp.constitutive_laws.FrictionBound,
):
    """Class for the coupling of mass and momentum balance to obtain poromechanics
    equations.

    """

    def stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        return self.mechanical_stress(subdomains) + self.pressure_stress(subdomains)


class EquationsPoromechanics(
    mass.MassBalanceEquations,
    momentum.MomentumBalanceEquations,
):
    """Combines mass and momentum balance equations."""

    def set_equations(self):
        """Set the equations for the poromechanics problem.

        Call both parent classes' set_equations methods.

        """
        mass.MassBalanceEquations.set_equations(self)
        momentum.MomentumBalanceEquations.set_equations(self)


class VariablesPoromechanics(
    mass.VariablesSinglePhaseFlow,
    momentum.VariablesMomentumBalance,
):
    """Combines mass and momentum balance variables."""

    def create_variables(self):
        """Set the variables for the poromechanics problem.

        Call both parent classes' set_variables methods.

        """
        mass.VariablesSinglePhaseFlow.create_variables(self)
        momentum.VariablesMomentumBalance.create_variables(self)


class BoundaryConditionsMechanicsTimeDependent(
    momentum.BoundaryConditionsMomentumBalance,
):
    bc_values_mechanics_key: str
    """Key for mechanical boundary conditions in the solution and iterate dictionaries.

    """

    def bc_values_mechanics(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.TimeDependentDenseArray:
        """Boundary values for mechanics.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        if not all([sd.dim == self.nd for sd in subdomains]):
            raise ValueError("Subdomains must be of dimension nd.")
        # Use time dependent array to allow for time dependent boundary conditions in
        # the div(u) term.
        return pp.ad.TimeDependentDenseArray(self.bc_values_mechanics_key, subdomains)

    def time_dependent_bc_values_mechanics(
        self, subdomains: list[pp.Grid]
    ) -> np.ndarray:
        """Boundary values for mechanics.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        if not all([sd.dim == self.nd for sd in subdomains]):
            raise ValueError("Subdomains must be of dimension nd.")

        # Default is zero.
        num_faces = np.sum([sd.num_faces for sd in subdomains])
        vals = np.zeros((self.nd, num_faces))
        return vals.ravel("F")


class BoundaryConditionsPoromechanics(
    mass.BoundaryConditionsSinglePhaseFlow,
    BoundaryConditionsMechanicsTimeDependent,
):
    """Combines mass and momentum balance boundary conditions.

    Note:
        The mechanical boundary conditions are differentiated wrt time in the div_u
        term. Thus, time dependent values must be defined using
        :class:`~porepy.numerics.ad.operators.TimeDependentArray`.

        To modify the values of the mechanical boundary conditions, the user must
        redefine the method
        :meth:`~porepy.models.poromechanics.BoundaryConditionsPoromechanics.
        time_dependent_bc_values_mechanics`, which is called by the methods
        :meth:`~porepy.models.poromechanics.SolutionStrategyPoromechanics.
        initial_condition` and :meth:`~porepy.models.poromechanics.
        SolutionStrategyPoromechanics.before_nonlinear_loop` to update the boundary
        conditions in `data['stored_solutions']` and `data['stored_iterates']`.

    """


class SolutionStrategyTimeDependentBCs(pp.SolutionStrategy):

    time_dependent_bc_values_mechanics: Callable[[list[pp.Grid]], np.ndarray]
    """Method for time dependent boundary conditions for mechanics."""

    @property
    def bc_values_mechanics_key(self) -> str:
        """Key for mechanical boundary conditions in the solution and iterate
        dictionaries.

        """
        return "bc_values_mechanics"

    def initial_condition(self) -> None:
        """Set initial condition for the coupled problem.

        The initial condition for the coupled problem is the initial condition for the
        subproblems.

        """
        # Set initial condition for the subproblems.
        super().initial_condition()

        self.update_time_dependent_ad_arrays(initial=True)

    def before_nonlinear_loop(self) -> None:
        super().before_nonlinear_loop()
        # Update the mechanical boundary conditions to both the solution and iterate.
        self.update_time_dependent_ad_arrays(initial=False)

    def update_time_dependent_ad_arrays(self, initial: bool) -> None:
        """Update the time dependent arrays for the mechanics boundary conditions.

        Parameters:
            initial: If True, the array generating method is called for both solution
                and iterate. If False, the array generating method is called only for
                the iterate, and the solution is updated by copying the iterate.

        """
        # Call super in case class is combined with other classes implementing this
        # method.
        super().update_time_dependent_ad_arrays(initial)
        # Update the mechanical boundary conditions to both the solutions and iterates.
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            if initial:
                vals = self.time_dependent_bc_values_mechanics([sd])
                set_solution_values(
                    name=self.bc_values_mechanics_key,
                    values=vals,
                    data=data,
                    solution_index=0,
                )
            else:
                # Copy old values from iterate to the solution.
                vals = data["stored_iterates"][self.bc_values_mechanics_key][0]
                set_solution_values(
                    name=self.bc_values_mechanics_key,
                    values=vals,
                    data=data,
                    solution_index=0,
                )

            vals = self.time_dependent_bc_values_mechanics([sd])
            set_solution_values(
                name=self.bc_values_mechanics_key,
                values=vals,
                data=data,
                iterate_index=0,
            )


class SolutionStrategyPoromechanics(
    SolutionStrategyTimeDependentBCs,
    mass.SolutionStrategySinglePhaseFlow,
    momentum.SolutionStrategyMomentumBalance,
):
    """Combines mass and momentum balance solution strategies.

    This class has a diamond structure inheritance. The user should be aware of this
    and take method resolution order into account when defining new methods.

    TODO: More targeted (re-)discretization.

    """

    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid."""

    time_dependent_bc_values_mechanics: Callable[[list[pp.Grid]], np.ndarray]
    """Method for time dependent boundary values for mechanics."""

    def set_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem."""
        # Set parameters for the subproblems.
        super().set_discretization_parameters()

        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):

            pp.initialize_data(
                sd,
                data,
                self.stress_keyword,
                {
                    "biot_alpha": self.solid.biot_coefficient(),  # TODO: Rename in Biot
                },
            )

    def _is_nonlinear_problem(self) -> bool:
        """The coupled problem is nonlinear."""
        return True


# Note that we ignore a mypy error here. There are some inconsistencies in the method
# definitions of the mixins, related to the enforcement of keyword-only arguments. The
# type Callable is poorly supported, except if protocols are used and we really do not
# want to go there. Specifically, method definitions that contains a *, for instance,
#   def method(a: int, *, b: int) -> None: pass
# which should be types as Callable[[int, int], None], cannot be parsed by mypy.
# For this reason, we ignore the error here, and rely on the tests to catch any
# inconsistencies.
class Poromechanics(  # type: ignore[misc]
    EquationsPoromechanics,
    VariablesPoromechanics,
    ConstitutiveLawsPoromechanics,
    BoundaryConditionsPoromechanics,
    SolutionStrategyPoromechanics,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for the coupling of mass and momentum balance in a mixed-dimensional porous
    medium.

    """

    pass
