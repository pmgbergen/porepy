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
import porepy as pp
import porepy.models.fluid_mass_balance as mass
import porepy.models.momentum_balance as momentum


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


class BoundaryConditionsPoromechanics(
    mass.BoundaryConditionsSinglePhaseFlow,
    momentum.BoundaryConditionsMomentumBalance,
):
    """Combines mass and momentum balance boundary conditions.

    Note:
        The mechanical boundary conditions are differentiated wrt time in the div_u
        term. Thus, time dependent values must be defined using
        :class:~porepy.numerics.ad.operators.TimeDependentArray. This is as of yet
        untested.

    """

    pass


class SolutionStrategyPoromechanics(
    mass.SolutionStrategySinglePhaseFlow,
    momentum.SolutionStrategyMomentumBalance,
):
    """Combines mass and momentum balance solution strategies.

    This class has a diamond structure inheritance. The user should be aware of this
    and take method resolution order into account when defining new methods.

    TODO: More targeted (re-)discretization.

    """

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
