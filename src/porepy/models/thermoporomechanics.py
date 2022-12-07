"""Coupling of energy, mass and momentum balance to obtain thermoporomechanics equations.

The module only contains what is needed for the coupling, the three individual subproblems
are defined elsewhere.

The main changes to the equations are achieved by changing the constitutive laws for
porosity and stress. The former aquires a pressure and temperature dependency and an
additional :math:`\alpha\nabla\cdot\mathbf{u}` term, while the stress is modified to
include isotropic pressure and temperature terms :math:`\alpha p \mathbf{I}+ \beta T
\mathbf{I}`.

Suggested references (TODO: add more, e.g. Inga's in prep, ppV2):
    - Coussy, 2004, https://doi.org/10.1002/0470092718.
    - Garipov and Hui, 2019, https://doi.org/10.1016/j.ijrmms.2019.104075.

"""

import porepy as pp

from . import energy_balance as energy
from . import fluid_mass_balance as mass
from . import momentum_balance as momentum


class ConstitutiveLawsThermoporomechanics(
    # Combined effects
    pp.constitutive_laws.DisplacementJumpAperture,
    pp.constitutive_laws.BiotCoefficient,
    pp.constitutive_laws.ThermalExpansion,
    pp.constitutive_laws.ThermoPressureStress,
    pp.constitutive_laws.ThermoPoroMechanicsPorosity,
    pp.constitutive_laws.FluidDensityFromPressureAndTemperature,
    # Individual subproblems
    energy.ConstitutiveLawsEnergyBalance,
    mass.ConstitutiveLawsSinglePhaseFlow,
    momentum.ConstitutiveLawsMomentumBalance,
):
    """Class for the coupling of energy, mass and momentum balance to obtain
    thermoporomechanics equations.

    """

    def stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Method from constitutive library's LinearElasticRock.
        traction = (
            self.mechanical_stress(subdomains)
            + self.pressure_stress(subdomains)
            + self.thermal_stress(subdomains)
        )
        traction.set_name("thermo_poro_mechnical_stress")
        return traction


class EquationsThermoporomechanics(
    energy.EnergyBalanceEquations,
    mass.MassBalanceEquations,
    momentum.MomentumBalanceEquations,
):
    """Combines energy, mass and momentum balance equations."""

    def set_equations(self):
        """Set the equations for the poromechanics problem.

        Call all parent classes' set_equations methods.

        """
        # Energy balance
        super().set_equations()
        mass.MassBalanceEquations.set_equations(self)
        momentum.MomentumBalanceEquations.set_equations(self)


class VariablesThermoporomechanics(
    energy.VariablesEnergyBalance,
    mass.VariablesSinglePhaseFlow,
    momentum.VariablesMomentumBalance,
):
    """Combines mass and momentum balance variables."""

    def create_variables(self):
        """Set the variables for the poromechanics problem.

        Call all parent classes' set_variables methods.

        """
        # Energy balance and its parent mass balance
        super().create_variables()
        mass.VariablesSinglePhaseFlow.create_variables(self)
        momentum.VariablesMomentumBalance.create_variables(self)


class BoundaryConditionsThermoporomechanics(
    energy.BoundaryConditionsEnergyBalance,
    mass.BoundaryConditionsSinglePhaseFlow,
    momentum.BoundaryConditionsMomentumBalance,
):
    """Combines energy, mass and momentum balance boundary conditions.

    Note:
        The mechanical boundary conditions are differentiated wrt time in the div_u term.
        Thus, time dependent values must be defined using
        :class:pp.ad.TimeDependentArray. This is as of yet untested.

    """

    pass


class SolutionStrategyThermoporomechanics(
    energy.SolutionStrategyEnergyBalance,
    mass.SolutionStrategySinglePhaseFlow,
    momentum.SolutionStrategyMomentumBalance,
):
    """Combines mass and momentum balance solution strategies.

    This class has an extended diamond structure inheritance, i.e., all parent classes
    inherit from :class:`~porepy.models.solution_strategy.SolutionStrategy`. The user
    should be aware of this and take method resolution order into account when defining
    new methods.

    TODO: More targeted (re-)discretization. See parent classes and other combined
    models.

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


class Thermoporomechanics(
    SolutionStrategyThermoporomechanics,
    EquationsThermoporomechanics,
    VariablesThermoporomechanics,
    BoundaryConditionsThermoporomechanics,
    ConstitutiveLawsThermoporomechanics,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for the coupling of energy, mass and momentum balance in a
    mixed-dimensional porous medium.

    """
