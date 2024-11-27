"""Combine single-physics models into coupled mass and energy balance equations. """

from __future__ import annotations

import porepy as pp

from . import energy_balance as energy
from . import fluid_mass_balance as mass


class EquationsFluidMassAndEnergy(
    energy.EnergyBalanceEquations,
    mass.MassBalanceEquations,
):
    """Combine fluid mass and energy balance equations."""


class VariablesFluidMassAndEnergy(
    energy.VariablesEnergyBalance,
    mass.VariablesSinglePhaseFlow,
):
    """Combine fluid mass and energy balance variables."""


class ConstitutiveLawFluidMassAndEnergy(
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.FluidDensityFromPressureAndTemperature,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.EnthalpyFromTemperature,
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.ThermalConductivityLTE,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.ConstantViscosity,
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


class InitialConditionsMassAndEnergy(
    energy.InitialConditionsEnergy,
    mass.InitialConditionsSinglePhaseFlow,
):
    """Combining initial conditions for fluid mass and energy balance."""

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


class MassAndEnergyBalance(  # type: ignore
    EquationsFluidMassAndEnergy,
    VariablesFluidMassAndEnergy,
    ConstitutiveLawFluidMassAndEnergy,
    BoundaryConditionsFluidMassAndEnergy,
    InitialConditionsMassAndEnergy,
    SolutionStrategyFluidMassAndEnergy,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Combine fluid mass and energy balance models into a single class.

    The equations assume single-phase flow and local thermal equilibrium.

    """

    pass
