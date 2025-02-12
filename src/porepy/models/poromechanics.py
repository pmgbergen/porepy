r"""Coupling of mass and momentum balance to obtain poromechanics equations.

The module only contains what is needed for the coupling, the two individual subproblems
are defined elsewhere.

The main changes to the equations are achieved by changing the constitutive laws for
porosity and stress. The former aquires a pressure dependency and an additional
:math:`\alpha`\nabla\cdot\mathbf{u} term, while the latter is modified to include a
isotropic pressure term :math:`\alpha p \mathbf{I}`.

Suggested references:
    - Coussy, 2004, https://doi.org/10.1002/0470092718.
    - Garipov and Hui, 2019, https://doi.org/10.1016/j.ijrmms.2019.104075.
    - Stefansson et al, 2024 https://doi.org/10.1016/j.rinam.2023.100428.

"""

from __future__ import annotations

from typing import Callable, Union

import porepy as pp


class ConstitutiveLawsPoromechanics(
    # Combined effects
    pp.constitutive_laws.DisplacementJumpAperture,
    pp.constitutive_laws.BiotCoefficient,
    pp.constitutive_laws.PressureStress,
    pp.constitutive_laws.PoroMechanicsPorosity,
    # Fluid mass balance subproblem
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.FluidDensityFromPressure,
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
    pp.momentum_balance.MomentumBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquations,
):
    """Combines mass and momentum balance equations."""


class VariablesPoromechanics(
    pp.momentum_balance.VariablesMomentumBalance,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
):
    """Combines mass and momentum balance variables."""


class BoundaryConditionsPoromechanics(
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    pp.momentum_balance.BoundaryConditionsMomentumBalance,
):
    """Combines mass and momentum balance boundary conditions.

    Note:
        The mechanical boundary conditions are differentiated wrt time in the
        displacement_divergence term.

        To modify the values of the mechanical boundary conditions, the user must
        redefine the method
        :meth:`~momentum.BoundaryConditionsMomentumBalance.
        boundary_displacement_values`, which is triggered by the method
        :meth:`~porepy.BoundaryConditionMixin.update_all_boundary_conditions`
        to update the boundary condition values in `data[pp.TIME_STEP_SOLUTIONS]` and
        `data[pp.ITERATE_SOLUTIONS]`.

    """


class InitialConditionsPoromechanics(
    pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow,
    pp.momentum_balance.InitialConditionsMomentumBalance,
):
    """Combines initial conditions for mass and momentum balance, and associated
    primary variables."""


class SolutionStrategyPoromechanics(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.momentum_balance.SolutionStrategyMomentumBalance,
):
    """Combines mass and momentum balance solution strategies.

    This class has a diamond structure inheritance. The user should be aware of this
    and take method resolution order into account when defining new methods.

    """

    darcy_flux_discretization: Callable[
        [list[pp.Grid]], Union[pp.ad.TpfaAd, pp.ad.MpfaAd]
    ]
    """Discretization of the Darcy flux. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.DarcysLaw`.

    """

    biot_tensor: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that defines the Biot tensor. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.BiotCoefficient`.
    """

    def set_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem."""
        # Set parameters for the subproblems.
        super().set_discretization_parameters()

        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):
            # Set the Biot coefficient.
            scalar_vector_mappings = data[pp.PARAMETERS][self.stress_keyword].get(
                "scalar_vector_mappings", {}
            )
            scalar_vector_mappings[self.darcy_keyword] = self.biot_tensor([sd])
            data[pp.PARAMETERS][self.stress_keyword]["scalar_vector_mappings"] = (
                scalar_vector_mappings
            )

    def _is_nonlinear_problem(self) -> bool:
        """The coupled problem is nonlinear."""
        return True

    def set_nonlinear_discretizations(self) -> None:
        """Collect discretizations for nonlinear terms."""
        # Nonlinear discretizations for the fluid mass balance subproblem. The momentum
        # balance does not have any.
        super().set_nonlinear_discretizations()
        # Aperture changes render permeability variable. This requires a re-discretization
        # of the diffusive flux in subdomains where the aperture changes.
        subdomains = [sd for sd in self.mdg.subdomains() if sd.dim < self.nd]
        self.add_nonlinear_discretization(
            self.darcy_flux_discretization(subdomains).flux(),
        )


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
    InitialConditionsPoromechanics,
    SolutionStrategyPoromechanics,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for the coupling of mass and momentum balance in a mixed-dimensional porous
    medium.

    """
