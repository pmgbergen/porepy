"""Combine single-physics models into coupled mass and energy balance equations. """
import porepy as pp

from . import energy_balance as energy
from . import fluid_mass_balance as mass


class EquationsFluidMassAndEnergy(
    energy.EnergyBalanceEquations,
    mass.MassBalanceEquations,
):
    """Combine fluid mass and energy balance equations."""

    def set_equations(self):
        """Set the equations for the fluid mass and energy balance problem.

        Call both parent classes' set_equations methods.

        """
        # Energy balance
        super().set_equations()
        mass.MassBalanceEquations.set_equations(self)


class VariablesFluidMassAndEnergy(
    energy.VariablesEnergyBalance,
    mass.VariablesSinglePhaseFlow,
):
    """Combine fluid mass and energy balance variables."""

    def create_variables(self) -> None:
        """Set the variables for the fluid mass and energy balance problem.

        Call both parent classes' set_variables methods.

        """
        # Energy balance
        super().create_variables()
        mass.VariablesSinglePhaseFlow.create_variables(self)


class ConstitutiveLawFluidMassAndEnergy(
    pp.constitutive_laws.FluidDensityFromPressureAndTemperature,
    energy.ConstitutiveLawsEnergyBalance,
    mass.ConstitutiveLawsSinglePhaseFlow,
):
    """Combine fluid mass and energy balance constitutive laws.

    Fluid density dependends on pressure and temperature in the mass and energy class,
    respectively. Here, both dependencies are included.

    """

    pass


class BoundaryConditionsFluidMassAndEnergy(
    energy.BoundaryConditionsEnergyBalance,
    mass.BoundaryConditionsSinglePhaseFlow,
):
    """Combine fluid mass and energy balance boundary conditions."""

    pass


class SolutionStrategyFluidMassAndEnergy(
    energy.SolutionStrategyEnergyBalance,
    mass.SolutionStrategySinglePhaseFlow,
):
    """Combine fluid mass and energy balance solution strategies.

    Solution strategies are proper classes (not mixins) and inherit from
    :class:`~porepy.models.solution_strategies.SolutionStrategy`. Thus, overridden
    methods call super() by default and explicitly calling both parent classes'
    methods is not necessary.

    """

    pass


class MassAndEnergyBalance(
    EquationsFluidMassAndEnergy,
    VariablesFluidMassAndEnergy,
    ConstitutiveLawFluidMassAndEnergy,
    BoundaryConditionsFluidMassAndEnergy,
    SolutionStrategyFluidMassAndEnergy,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Combine fluid mass and energy balance models into a single class.

    The equations assume single-phase flow and local thermal equilibrium.

    """

    pass
