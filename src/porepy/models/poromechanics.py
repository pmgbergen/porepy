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
    pp.constitutive_laws.CharacteristicTractionFromDisplacement,
    pp.constitutive_laws.ElasticTangentialFractureDeformation,
    pp.constitutive_laws.LinearElasticMechanicalStress,
    pp.constitutive_laws.ConstantSolidDensity,
    pp.constitutive_laws.FractureGap,
    pp.constitutive_laws.CoulombFrictionBound,
    pp.constitutive_laws.DisplacementJump,
):
    """Class for combined constitutive laws for poromechanics."""

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
    pp.contact_mechanics.ContactMechanicsEquations,
):
    """Combines mass and momentum balance and contact mechanics equations.
    Adaptation is made to the body force taking into account solid and
    fluid."""

    def body_force(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Body force integrated over the subdomain cells.

        Parameters:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force [kg*m*s^-2].

        """
        return self.volume_integral(
            self.gravity_force(subdomains, "bulk"), subdomains, dim=self.nd
        )


class _SolidMassEquation(momentum.SolidMassEquation):
    """Solid mass equation for poromechanics.

    This is an extension of the solid mass equation in the three-field formulation of
    the mechanics problem. The extension is the addition of the fluid pressure term.

    """

    biot_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Biot coefficient. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws.BiotCoefficient`.
    """
    inv_lambda: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Inverse of the second Lame parameter. Normally defined in a mixin instance of
    :class:`~porepy.models.constitutive_laws._ThreeFieldLinearElasticMechanicalStress`.
    """
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """

    def solid_mass_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Extension of the solid mass equation to the poromechanics problem [-].

        For details on the the solid mass equation, and the extension to poromechanical
        systems, see https://arxiv.org/pdf/2405.10390 Section 2.1.

        Parameters:
            subdomains: List of subdomains where the solid mass equation is defined.

        Returns:
            Operator for the solid mass equation.

        """
        # The mechanics part of the solid mass equation is the same as in the momentum
        # balance model.
        momentum_term = super().solid_mass_equation(subdomains)

        # Add the term related to the fluid pressure.
        iLambda = self.inv_lambda(subdomains)
        # Biot coefficient.
        biot = self.biot_coefficient(subdomains)

        pressure_term = self.volume_integral(
            iLambda * biot * self.pressure(subdomains),
            subdomains,
            dim=1,
        )
        full_eq = momentum_term - pressure_term

        full_eq.set_name("Solid_mass_equation_poromechanics")

        return full_eq


class VariablesPoromechanics(
    pp.momentum_balance.VariablesMomentumBalance,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    pp.contact_mechanics.ContactTractionVariable,
):
    """Combines mass and momentum balance and contact mechanics variables."""


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
    pp.contact_mechanics.InitialConditionsContactTraction,
):
    """Combines initial conditions for mass and momentum balance and contact mechanics,
    and associated primary variables."""


class TpsaPoromechanicsMixin(
    pp.constitutive_laws._ConstitutiveLawsTpsaPoromechanics,
    _SolidMassEquation,
    momentum.TpsaMomentumBalanceMixin,
):
    """Mixin for the TPSA poromechanics model. This can be mixed into a Poromechanics
    class to get four-field (displacement, rotation stress, total pressure and fluid
    pressure) formulation for poromechanics. The resulting model will be discretized
    using Tpsa.

    Can also be used to define a THM model with Tpsa.

    """

    pass


class SolutionStrategyPoromechanics(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.momentum_balance.SolutionStrategyMomentumBalance,
    pp.contact_mechanics.SolutionStrategyContactMechanics,
):
    """Combines mass and momentum balance and contact mechanics solution strategies.

    This class has a diamond structure inheritance. The user should be aware of this
    and take method resolution order into account when defining new methods.

    """

    biot_tensor: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that defines the Biot tensor. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.BiotCoefficient`.
    """

    def update_discretization_parameters(self) -> None:
        """Set parameters for the subproblems and the combined problem."""
        # Set parameters for the subproblems.
        super().update_discretization_parameters()

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

    def add_nonlinear_darcy_flux_discretization(self) -> None:
        """Poromechanics rely by default on Darcy flux re-discretization.

        The re-discretization is performed only on subdomains with
        ``dim < nd`` due to changes in aperture!
        The default behavior defined here concerns only those domains.

        """

        self.add_nonlinear_diffusive_flux_discretization(
            self.darcy_flux_discretization(
                [sd for sd in self.mdg.subdomains() if sd.dim < self.nd]
            ).flux(),
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
