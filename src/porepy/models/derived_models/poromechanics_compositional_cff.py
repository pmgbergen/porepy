"""
Poromechanics coupled with compositional multiphase flow.

This module combines:
- Biot thermo-poromechanics (momentum balance with thermo-pressure-stress 
    (mostly know as Thermo-poroelastic coupling) coupling)
- Fraction form of Compositional flow (mass balance for H2O and NaCl components)
- Energy balance (non-isothermal)

The coupling is achieved through:
1. Effective (total) stress: \sigma = \sigma_mechanical - \alpha·p·I - \beta·K·(T-T_ref)·I
2. Porosity: \phi = \phi_ref + (\alpha-\phi_ref)(1-\phi)/K·\delta p + \alpha·\div u + thermal_contribution
3. Fracture aperture from displacement jump (when fractures present),
where \alpha is the Biot constant given as 0<=\alpha<=1

Author: Michael Oguntola
Based on PorePy framework
"""

from __future__  import annotations
from typing import Callable, Union

import porepy as pp
from porepy.models import compositional_flow as cf


# ===============================================================================
# EQUATIONS
# ===============================================================================

class EquationsPoromechanicsCompositional(
    # Momentum balance (provides the displacement equation)
    pp.momentum_balance.MomentumBalanceEquations,
    # Compositional flow: pressure, component mass, and energy balance equations
    cf.PrimaryEquationsCFF,
    # Contact mechanics (for fractures, if present)
    pp.contact_mechanics.ContactMechanicsEquations,
):
    """Coupled equations for poromechanics with compositional multiphase flow.
    Equations: 
        - Momentum balance (displacement): \div σ + \rho_s.g = 0
        - Total mass balance equation (pressure equation)
        - Component mass balance equations (for NaCl)
        - Energy balance equation (enthalpy equation)
        - Contact mechanics equations (for fractures, if present)
    """
    pass


# ==============================================================================
# VARIABLES
# ==============================================================================

class VariablesPoromechanicsCompositional(
    # Displacement variables for poromechanics
    pp.momentum_balance.VariablesMomentumBalance,
    # Contact mechanics variables (for fractures, if present)
    pp.contact_mechanics.ContactTractionVariable,
    # Pressure and component mass variables for compositional flow
    cf.VariablesCF,
):
    """Combined Variables for poromechanics + compositional flow + contact mechanics (if fractures present).

    Primary variables include:
        - Displacement (vector, nd components)
        - Pressure (scalar)
        - enthalpy (scalar)
        - Overall fractions (per independent component, e.g., NaCl)
        - contact tractions (vector, for fracture surfaces)
    
    Secondary variables include:
        - temperature
        - saturations (per phase)
        - phase fractions and compositions (per phase)
    """
    pass

# =======================================================================
# CONSTITUTIVE LAWS
# =======================================================================

class ConstitutiveLawsPoromechanicsCompositional(
    # ----- Compositional flow constitutive laws -----
    # This includes EOS to eliminate quantities: FluidMobility, FouriersLaw, etc.
    cf.ConstitutiveLawsCF,

    # ----- Biot coupling (thermo-poro-mechanics)------
    pp.constitutive_laws.BiotCoefficient,   # \alpha = 1 - K_s/K,
    pp.constitutive_laws.ThermoPressureStress,  # \sigma_p = -\alpha·p·I, \sigma_T = -\beta·K·\delta T·I
    pp.constitutive_laws.ThermalExpansion,
    pp.constitutive_laws.ThermoPoroMechanicsPorosity,   # \phi(p, \div u, T)

    # ------ Fracture aperture from displacement jump (if fractures present) ------
    pp.constitutive_laws.DisplacementJumpAperture,

    # ----- Mechanical Constitutive laws -----
    pp.constitutive_laws.ElasticModuli,
    pp.constitutive_laws.CharacteristicTractionFromDisplacement,
    pp.constitutive_laws.LinearElasticMechanicalStress,
    pp.constitutive_laws.ConstantSolidDensity,

    # ----- Fracture contact mechanics (if fractures present) -----
    pp.constitutive_laws.ElasticTangentialFractureDeformation,
    pp.constitutive_laws.FractureGap,
    pp.constitutive_laws.CoulombFrictionBound,
    pp.constitutive_laws.DisplacementJump,
):
    """Combined constitutive laws for thermo-poromechanics with compositional flow

    Key couplings:
        - stress(): mechanical + pressure + thermal stress contribution
        - porosity(): reference porosity + pressure + deformation + thermal contribution
        - aperture(): from displacement jump (if fractures present)
    """
    def stress(self, subdomains:list[pp.Grid]) -> pp.ad.Operator:
        """Total stress = mechanical + pressure + thermal contributions
            \sigma = \sigma_mechanical - \alpha·p·I - \beta·K·\delta T·I

        Parameters:
            subdomains: List of nd-dimensional subdomains.
        Returns:
            Total stress operator
        """

        stress = (
            self.mechanical_stress(subdomains)
            + self.pressure_stress(subdomains)
            + self.thermal_stress(subdomains)
        )
        stress.set_name("total_stress")
        return stress
    

# =======================================================================
# BOUNDARY CONDITIONS
# =======================================================================

class BoundaryConditionsPoromechanicsCompositional(
    pp.momentum_balance.BoundaryConditionsMomentumBalance,
    pp.contact_mechanics.BoundaryConditionsContactMechanics,
    cf.BoundaryConditionsCFF,
):
    """Combined boundary conditions for thermo-poromechanics with compositional flow

    Mechanical BCs:
        - Displacement BCs (Dirichlet) or Traction BCs (Neumann)
        - contact traction (if fractures present)
    Flow BCs:
        - Pressure BCs (Dirichlet) or Flux BCs (Neumann)
        - Component fractions on inflow boundaries
        - Temperature or heat flux BCs for energy balance
    """
    pass


# =======================================================================
# INITIAL CONDITIONS
# =======================================================================

class InitialConditionsPoromechanicsCompositional(
    pp.momentum_balance.InitialConditionsMomentumBalance,
    pp.contact_mechanics.InitialConditionsContactTraction,
    cf.InitialConditionsCF,
):
    """Combined initial conditions for thermo-poromechanics with compositional flow

    Initial conditions include:
        - Initial displacement field (for poromechanics)
        - Initial pressure field (for flow)
        - Initial temperature field (for energy balance)
        - Initial component mass fractions (for compositional flow)
        - contact traction (if fractures present)
    """
    pass


# =======================================================================
# SOLUTION STRATEGY
# =======================================================================
class SolutionStrategyPoromechanicsCompositional(
    cf.SolutionStrategyCF,  # Phase property updates, Schur complement, etc.
    pp.momentum_balance.SolutionStrategyMomentumBalance,
    pp.contact_mechanics.SolutionStrategyContactMechanics,
):
    """Solution strategy for coupled thermo-poromechanics with compositional flow.

    Handles:
        - Phase property evaluation (EOS calls)
        - Nonlinear discretization updates
        - Biot tensor setup for MPSA
        - Newton iteration settings
    
    Note that, we need a scalar_vector_mappings to handle the coupling 
    between scalar (pressure, temperature) and vector (displacement) variables 
    affect the stress in the MPSA discretization.

    scalar_vector_mappings — a dictionary that tells the BiotAd discretization 
    which scalar fields (pressure, temperature) couple to the vector field 
    (displacement).
    """
    darcy_flux_discretization: Callable[
        [list[pp.Grid]], Union[pp.ad.TpfaAd, pp.ad.MpfaAd]
    ]
    """Discretization of the Darcy flux."""

    biot_tensor: Callable[[list[pp.Grid]], pp.SecondOrderTensor]
    """"Method that defines biot tensor."""

    def update_discretization_parameters(self) -> None:
        """"Set parameters for the coupled problem.

        This includes setting the Biot tensor for the scalar-vector coupling 
        in the MPSA discretization, and ensuring that the Darcy flux discretization 
        is consistent with the flow equations.
        """
        super().update_discretization_parameters()

        # Set Biot coefficient for pressure-displacement coupling in MPSA
        for sd, data in self.mdg.subdomains(dim=self.nd, return_data=True):
            scalar_vector_mappings = data[pp.PARAMETERS][self.stress_keyword].get(
                    "scalar_vector_mappings", {}
            )

            # Pressure coupling
            scalar_vector_mappings[self.darcy_keyword] = self.biot_tensor([sd])

            # Temperature coupling (if using ThermoPressureStress)
            # The thermal expansion tensor maps temperature to stress.
            if hasattr(self, "enthalpy_keyword"):
                scalar_vector_mappings[self.enthalpy_keyword] = (
                    self.solid_thermal_expansion_tensor([sd])
                )

            data[pp.PARAMETERS][self.stress_keyword]["scalar_vector_mappings"] = (
                scalar_vector_mappings
            )
    
    def _is_nonlinear_problem(self) -> bool:
        return True
    
    def set_nonlinear_discretizations(self) -> None:
        """"Collect discretizations that need updating each Newton iteration.
        
        For poromechanics with compositional flow:
            - Darcy flux in fractures (aperture changes)
            - Phase properties (from EOS)
            - Upwind discretizations
        Note: The re-discretization is performed only on subdomains with
            ``dim < nd`` due to changes in aperture!
            The default behavior defined here concerns only those domains.
        """

        super().set_nonlinear_discretizations()

        # Rediscretize Darcy flux in fractures due to aperture changes
        # Only rediscretize Darcy flux if we have actual fractures (not wells)
        # Tags that identify wells (not fracture intersections)
        well_tags = {"injection_well", "production_well"}

        # Lower-dimensional subdomains EXCLUDING wells
        # This includes fractures (dim = nd-1) and intersections (dim < nd-1)
        fractures_and_intersections = [
            sd for sd in self.mdg.subdomains() 
            if sd.dim < self.nd and not well_tags.intersection(sd.tags)
        ]

        if len(fractures_and_intersections) > 0:
            self.add_nonlinear_discretization(
                self.darcy_flux_discretization(fractures_and_intersections).flux(),
            )


# =======================================================================
# Thermo+Poromechanical+Compositional TEMPLATE MODEL
# =======================================================================

class PoromechanicsCompositionalTemplate( # type: ignore[misc]
    EquationsPoromechanicsCompositional,
    VariablesPoromechanicsCompositional,
    ConstitutiveLawsPoromechanicsCompositional,
    BoundaryConditionsPoromechanicsCompositional,
    InitialConditionsPoromechanicsCompositional,
    SolutionStrategyPoromechanicsCompositional,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """"Template for poromechanics coupled with compositional multiphase flow.
    
    This template combines :
       - Biot poromechanics (momentum balance with thermo-pressure-stress coupling)
       - Compositional flow (multicomponent system)
       - Energy balance (non-isothermal)
       - Contact mechanics (for fractures, if present)

    To create a runable model, users need to:
        1. Define fluid with phases and components
        2. Provide local equilibrium closure (flash / OBL)
        3. Set appropriate material parameters

    Example usage:
        class MyBrinePoromechanicsModel(PoromechanicsCompositionalTemplate):
            def set_fluid(self):
                # Define your H2O-NaCl fluid system
            def set_materials(self):
                # Set solid mechanics parameters
                self.solid = pp.SolidConstants(
                    lame_lambda=1e9,
                    shear_modulus=1e9,
                    biot_coefficient=0.8,
                    porosity=0.1,
                    permeability=1e-15,
                    thermal_expansion_coefficient=1e-5,
                )
    
    Note: 
        The contact mechanics components will be inactive (zero equations/ variable) 
        if no fractures are present in the model geometry, so users can ignore them if not modeling fractures. 
        
    """
    pass