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
    - Stefansson et al, 2021, https://doi.org/10.1016/j.cma.2021.114122

"""
from __future__ import annotations

import porepy as pp

from . import energy_balance as energy
from . import fluid_mass_balance as mass
from . import momentum_balance as momentum
from . import poromechanics


class ConstitutiveLawsThermoporomechanics(
    # Combined effects
    pp.constitutive_laws.DisplacementJumpAperture,
    pp.constitutive_laws.BiotCoefficient,
    pp.constitutive_laws.ThermalExpansion,
    pp.constitutive_laws.ThermoPressureStress,
    pp.constitutive_laws.ThermoPoroMechanicsPorosity,
    pp.constitutive_laws.FluidDensityFromPressureAndTemperature,
    # Energy subproblem
    pp.constitutive_laws.EnthalpyFromTemperature,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.ThermalConductivityLTE,
    # Fluid mass balance subproblem
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.ConstantViscosity,
    # Mechanical subproblem
    pp.constitutive_laws.LinearElasticSolid,
    pp.constitutive_laws.FracturedSolid,
    pp.constitutive_laws.FrictionBound,
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
    energy.EnergyBalanceEquations,
    mass.MassBalanceEquations,
    momentum.MomentumBalanceEquations,
):
    """Combines energy, mass and momentum balance equations."""

    def set_equations(self):
        """Set the equations for the poromechanics problem.

        Call all parent classes' set_equations methods.

        """
        # Call all super classes' set_equations methods. Do this explicitly (calling the
        # methods of the super classes directly) instead of using super() since this is
        # more transparent.
        energy.EnergyBalanceEquations.set_equations(self)
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
        energy.VariablesEnergyBalance.create_variables(self)
        mass.VariablesSinglePhaseFlow.create_variables(self)
        momentum.VariablesMomentumBalance.create_variables(self)


class BoundaryConditionsThermoporomechanics(
    energy.BoundaryConditionsEnergyBalance,
    mass.BoundaryConditionsSinglePhaseFlow,
    poromechanics.BoundaryConditionsMechanicsTimeDependent,
):
    """Combines energy, mass and momentum balance boundary conditions.

    Note:
        The mechanical boundary conditions are differentiated wrt time in the div_u term.
        Thus, time dependent values must be defined using
        :class:pp.ad.TimeDependentArray. This is as of yet untested.

    """


class SolutionStrategyThermoporomechanics(
    poromechanics.SolutionStrategyTimeDependentBCs,
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
    ConstitutiveLawsThermoporomechanics,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for the coupling of energy, mass and momentum balance in a
    mixed-dimensional porous medium.

    """
