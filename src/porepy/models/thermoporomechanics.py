r"""Coupling of energy, mass and momentum balance to obtain thermoporomechanics equations.

The module only contains what is needed for the coupling, the three individual subproblems
are defined elsewhere.

The main changes to the equations are achieved by changing the constitutive laws for
porosity and stress. The former aquires a pressure and temperature dependency and an
additional :math:`\alpha\nabla\cdot\mathbf{u}` term, while the stress is modified to
include isotropic pressure and temperature terms :math:`\alpha p \mathbf{I}+ \beta T
\mathbf{I}`.

References:

    - Coussy, 2004, https://doi.org/10.1002/0470092718.
    - Garipov and Hui, 2019, https://doi.org/10.1016/j.ijrmms.2019.104075.
    - Stefansson et al., 2021, https://doi.org/10.1016/j.cma.2021.114122.
    - Stefansson et al., 2024, https://doi.org/10.1016/j.rinam.2023.100428.

"""

from __future__ import annotations

from typing import Callable, Union

import porepy as pp


class ConstitutiveLawsThermoporomechanics(
    # Combined effects
    pp.constitutive_laws.DisplacementJumpAperture,
    pp.constitutive_laws.BiotCoefficient,
    pp.constitutive_laws.ThermalExpansion,
    pp.constitutive_laws.ThermoPressureStress,
    pp.constitutive_laws.ThermoPoroMechanicsPorosity,
    pp.constitutive_laws.FluidDensityFromPressureAndTemperature,
    # Energy subproblem
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.EnthalpyFromTemperature,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.ThermalConductivityLTE,
    # Fluid mass balance subproblem
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.ConstantViscosity,
    # Mechanical subproblem
    pp.constitutive_laws.ElasticModuli,
    pp.constitutive_laws.ElasticTangentialFractureDeformation,
    pp.constitutive_laws.LinearElasticMechanicalStress,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.FractureGap,
    pp.constitutive_laws.CoulombFrictionBound,
    pp.constitutive_laws.DisplacementJump,
):
    """Class for the coupling of energy, mass and momentum balance to obtain
    thermoporomechanics equations.

    """

    def stress(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Thermo-poromechanical stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """
        # Simply add the pressure and temperature terms to the mechanical stress
        traction = (
            self.mechanical_stress(subdomains)
            + self.pressure_stress(subdomains)
            + self.thermal_stress(subdomains)
        )
        traction.set_name("thermo_poro_mechnical_stress")
        return traction


class EquationsThermoporomechanics(
    pp.energy_balance.TotalEnergyBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquations,
    pp.momentum_balance.MomentumBalanceEquations,
):
    """Combines energy, mass and momentum balance equations."""


class VariablesThermoporomechanics(
    pp.energy_balance.VariablesEnergyBalance,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    pp.momentum_balance.VariablesMomentumBalance,
):
    """Combines mass and momentum balance variables."""


class BoundaryConditionsThermoporomechanics(
    pp.energy_balance.BoundaryConditionsEnergyBalance,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    pp.momentum_balance.BoundaryConditionsMomentumBalance,
):
    """Combines energy, mass and momentum balance boundary conditions.

    Note:
        The mechanical boundary conditions are differentiated wrt time in the
        displacement_divergence term. Thus, time dependent values must be defined using
        :class:pp.ad.TimeDependentArray. This is as of yet untested.

    """


class InitialConditionsThermoporomechanics(
    pp.energy_balance.InitialConditionsEnergy,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    pp.momentum_balance.BoundaryConditionsMomentumBalance,
):
    """Combines initial conditions for energy, mass and momentum balance and associated
    primary variables."""


class SolutionStrategyThermoporomechanics(
    pp.energy_balance.SolutionStrategyEnergyBalance,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.momentum_balance.SolutionStrategyMomentumBalance,
):
    """Combines mass and momentum balance solution strategies.

    This class has an extended diamond structure inheritance, i.e., all parent classes
    inherit from :class:`~porepy.models.solution_strategy.SolutionStrategy`. The user
    should be aware of this and take method resolution order into account when defining
    new methods.

    """

    darcy_flux_discretization: Callable[
        [list[pp.Grid]], Union[pp.ad.TpfaAd, pp.ad.MpfaAd]
    ]
    """Discretization of the Darcy flux. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.DarcysLaw`.

    """
    fourier_flux_discretization: Callable[
        [list[pp.Grid]], Union[pp.ad.TpfaAd, pp.ad.MpfaAd]
    ]
    """Discretization of the Fourier flux. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FouriersLaw`.

    """
    temperature_variable: str
    """Name of the temperature variable. Normally set by a mixin instance of
    :class:`~porepy.models.energy_balance.SolutionStrategyEnergyBalance`.
    """
    biot_tensor: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that defines the Biot tensor. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.BiotCoefficient`.
    """
    solid_thermal_expansion_tensor: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Thermal expansion coefficient. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.ThermalExpansion`.
    """

    def set_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem."""
        # Set parameters for the subproblems.
        super().set_discretization_parameters()

        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):
            scalar_vector_mappings = data[pp.PARAMETERS][self.darcy_keyword].get(
                "scalar_vector_mappings", {}
            )
            scalar_vector_mappings[self.enthalpy_keyword] = (
                self.solid_thermal_expansion_tensor([sd])
            )
            scalar_vector_mappings[self.darcy_keyword] = self.biot_tensor([sd])
            data[pp.PARAMETERS][self.stress_keyword][
                "scalar_vector_mappings"
            ] = scalar_vector_mappings

    def set_nonlinear_discretizations(self) -> None:
        """Collect discretizations for nonlinear terms."""
        # Super calls method in mass and energy balance. Momentum balance has no
        # nonlinear discretizations.
        super().set_nonlinear_discretizations()
        # Aperture changes render permeability variable. This requires a re-discretization
        # of the diffusive flux in subdomains where the aperture changes.
        subdomains = [sd for sd in self.mdg.subdomains() if sd.dim < self.nd]
        self.add_nonlinear_discretization(
            self.darcy_flux_discretization(subdomains).flux(),
        )
        # Aperture and porosity changes render thermal conductivity variable. This
        # requires a re-discretization of the diffusive flux.
        self.add_nonlinear_discretization(
            self.fourier_flux_discretization(self.mdg.subdomains()).flux(),
        )


# Note that we ignore a mypy error here. There are some inconsistencies in the method
# definitions of the mixins, related to the enforcement of keyword-only arguments. The
# type Callable is poorly supported, except if protocols are used and we really do not
# want to go there. Specifically, method definitions that contains a *, for instance,
#   def method(a: int, *, b: int) -> None: pass
# which should be types as Callable[[int, int], None], cannot be parsed by mypy.
# For this reason, we ignore the error here, and rely on the tests to catch any
# inconsistencies.
class Thermoporomechanics(  # type: ignore[misc]
    SolutionStrategyThermoporomechanics,
    EquationsThermoporomechanics,
    VariablesThermoporomechanics,
    BoundaryConditionsThermoporomechanics,
    InitialConditionsThermoporomechanics,
    ConstitutiveLawsThermoporomechanics,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for the coupling of energy, mass and momentum balance in a
    mixed-dimensional porous medium.

    """
