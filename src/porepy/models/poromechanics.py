"""Coupling of mass and momentum balance to obtain poromechanics equations.

The module only contains what is needed for the coupling, the two individual subproblems
are defined elsewhere.

"""
from typing import Optional

import porepy as pp
import porepy.models.fluid_mass_balance as mass
from porepy.models import momentum


class ConstitutiveLawsPoromechanicsCoupling(
    pp.constitutive_laws.BiotWillisCoefficient,
    pp.constitutive_laws.PressureStress,
):
    """Class for the coupling of mass and momentum balance to obtain poromechanics
    equations.

    """

    def stress(self, subdomains: list[pp.Grid]):
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


class VariablesPoromechanics(
    mass.VariablesSinglePhaseFlow,
    momentum.VariablesMomentumBalance,
):
    """Combines mass and momentum balance variables."""

    pass


class BoundaryConditionPoromechanics(
    mass.BoundaryConditionSinglePhaseFlow,
    momentum.BoundaryConditionMomentumBalance,
):
    """Combines mass and momentum balance boundary conditions.

    Note:
        The mechanical boundary conditions are differentiated wrt time in the div_u term.
        Thus, time dependent values must be defined using
        :class:pp.ad.TimeDependentArray. This is as of yet untested.
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

    def __init__(self, params: Optional[dict] = None) -> None:
        # Call the constructor of the parent classes. Note that this calls the
        # constructor of the parent of the parent class (:class:`pp.SolutionStrategy`),
        # twice. TODO: Decide if this is a good idea.
        super().__init__(params)

    def initial_condition(self) -> None:
        """Set initial condition for both subproblems."""
        super().initial_condition()

    def set_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem."""
        # Set parameters for the subproblems
        super().set_discretization_parameters()

        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):

            pp.initialize_data(
                sd,
                data,
                self.stress_keyword,
                {
                    "biot_alpha": self.solid.biot_alpha(),
                },
            )

    def _is_nonlinear_problem(self) -> bool:
        """The coupled problem is nonlinear."""
        return True


class ConstitutiveLawsPoromechanics(
    ConstitutiveLawsPoromechanicsCoupling,
    mass.ConstitutiveLawsMassBalance,
    momentum.ConstitutiveLawsMomentumBalance,
):
    pass
