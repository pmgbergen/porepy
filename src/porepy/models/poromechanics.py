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
from typing import Optional

import porepy as pp
import porepy.models.fluid_mass_balance as mass
import porepy.models.momentum_balance as momentum


class ConstitutiveLawsPoromechanicsCoupling(
    pp.constitutive_laws.BiotCoefficient,
    pp.constitutive_laws.PressureStress,
    pp.constitutive_laws.PoroMechanicsPorosity,
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

    def set_equations(self):
        """Set the equations for the poromechanics problem.

        Call both parent classes' set_equations methods.

        """
        super().set_equations()
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
        super().create_variables()
        momentum.VariablesMomentumBalance.create_variables(self)


class BoundaryConditionPoromechanics(
    mass.BoundaryConditionsSinglePhaseFlow,
    momentum.BoundaryConditionsMomentumBalance,
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
        # Set initial condition for the subproblems. Mass balance calls super, so this
        # call also sets the initial condition for the momentum balance.
        super().initial_condition()

    def set_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem."""
        # Set parameters for the subproblems.
        super().set_discretization_parameters()
        # Mass balance method does not call super, so we need to call it explicitly.
        momentum.SolutionStrategyMomentumBalance.set_discretization_parameters(self)

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


class ConstitutiveLawsPoromechanics(
    ConstitutiveLawsPoromechanicsCoupling,
    mass.ConstitutiveLawsSinglePhaseFlow,
    momentum.ConstitutiveLawsMomentumBalance,
):
    pass
