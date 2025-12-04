"""Type stub for mixin classes.

This stub file declares:
1. The methods defined by each mixin class
2. The attributes from PorePyModel that the mixin expects via duck typing

This allows type checkers to understand the mixin interface without
requiring runtime inheritance from PorePyModel.
"""

from typing import Any

class DisplacementJump:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def displacement_jump(self, subdomains: ...) -> None: ...
    def elastic_displacement_jump(self, subdomains: ...) -> None: ...
    def plastic_displacement_jump(self, subdomains: ...) -> None: ...

class DimensionReduction:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def grid_aperture(self, grid: ...) -> None: ...
    def aperture(self, subdomains: ...) -> None: ...
    def specific_volume(self, grids: ...) -> None: ...

class SecondOrderTensorUtils:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def isotropic_second_order_tensor(self, subdomains: ..., permeability: ...) -> None: ...
    def operator_to_SecondOrderTensor(self, subdomains: ..., operator: ..., fallback_value: ...) -> None: ...

class ConstantPermeability:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def permeability(self, subdomains: ...) -> None: ...
    def normal_permeability(self, interfaces: ...) -> None: ...

class DarcysLaw:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def pressure_trace(self, subdomains: ...) -> None: ...
    def darcy_flux(self, domains: ...) -> None: ...
    def combine_boundary_operators_darcy_flux(self, subdomains: ...) -> None: ...
    def interface_darcy_flux_equation(self, interfaces: ...) -> None: ...
    def darcy_flux_discretization(self, subdomains: ...) -> None: ...
    def vector_source_darcy_flux(self, grids: ...) -> None: ...
    def interface_vector_source_darcy_flux(self, interfaces: ...) -> None: ...

class AdTpfaFlux:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def diffusive_flux(self, domains: ..., potential: ..., diffusivity_tensor: ..., boundary_operator: ..., flux_name: ...) -> None: ...
    def potential_trace(self, subdomains: ..., potential: ..., diffusivity_tensor: ..., boundary_operator: ..., flux_name: ...) -> None: ...
    def __transmissibility_matrix(self, subdomains: ..., diffusivity_tensor: ...) -> None: ...
    def __mpfa_flux_discretization(self, base_discr: ..., T_f: ..., p_diff: ..., p: ...) -> None: ...
    def __mpfa_vector_source_discretization(self, base_discr: ..., T_f: ..., vs_diff: ..., vs: ...) -> None: ...
    def __mpfa_bound_pressure_discretization(self, base_discr: ..., bound_pressure_face: ..., internal_flux: ..., external_bc: ...) -> None: ...

class PeacemanWellFlux:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def well_flux_equation(self, interfaces: ...) -> None: ...
    def equivalent_well_radius(self, subdomains: ...) -> None: ...
    def skin_factor(self, interfaces: ...) -> None: ...
    def well_radius(self, subdomains: ...) -> None: ...

class ThermalExpansion:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def solid_thermal_expansion_coefficient(self, subdomains: ...) -> None: ...
    def solid_thermal_expansion_tensor(self, subdomains: ...) -> None: ...

class FouriersLaw:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def temperature_trace(self, subdomains: ...) -> None: ...
    def fourier_flux(self, subdomains: ...) -> None: ...
    def combine_boundary_operators_fourier_flux(self, subdomains: ...) -> None: ...
    def interface_fourier_flux_equation(self, interfaces: ...) -> None: ...
    def vector_source_fourier_flux(self, grids: ...) -> None: ...
    def interface_vector_source_fourier_flux(self, interfaces: ...) -> None: ...
    def fourier_flux_discretization(self, subdomains: ...) -> None: ...

class AdvectiveFlux:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def advective_flux(self, subdomains: ..., advected_entity: ..., discr: ..., bc_values: ..., interface_flux: ...) -> None: ...
    def interface_advective_flux(self, interfaces: ..., advected_entity: ..., discr: ...) -> None: ...
    def well_advective_flux(self, interfaces: ..., advected_entity: ..., discr: ...) -> None: ...

class GravityForce:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def gravity_force(self, grids: ..., material: ...) -> None: ...

class ZeroGravityForce:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def gravity_force(self, grids: ..., material: ...) -> None: ...

class LinearElasticMechanicalStress:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def mechanical_stress(self, domains: ...) -> None: ...
    def combine_boundary_operators_mechanical_stress(self, subdomains: ...) -> None: ...
    def fracture_stress(self, interfaces: ...) -> None: ...
    def stress_discretization(self, subdomains: ...) -> None: ...

class ThreeFieldLinearElasticMechanicalStress:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def mechanical_stress(self, domains: ...) -> None: ...
    def stress_discretization(self, subdomains: ...) -> None: ...
    def total_rotation(self, domains: ...) -> None: ...
    def solid_mass_flux(self, domains: ...) -> None: ...
    def first_lame_parameter(self, subdomains: ...) -> None: ...
    def second_lame_parameter(self, subdomains: ...) -> None: ...

class ConstitutiveLawsTpsaPoromechanics:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def stress(self, subdomains: ...) -> None: ...
    def porosity_change_from_displacement(self, subdomains: ...) -> None: ...

class ConstantSolidDensity:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def solid_density(self, subdomains: ...) -> None: ...

class ElasticModuli:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def shear_modulus(self, subdomains: ...) -> None: ...
    def lame_lambda(self, subdomains: ...) -> None: ...
    def youngs_modulus(self, subdomains: ...) -> None: ...
    def bulk_modulus(self, subdomains: ...) -> None: ...
    def stiffness_tensor(self, subdomain: ...) -> None: ...

class CharacteristicTractionFromDisplacement:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def characteristic_contact_traction(self, subdomains: ...) -> None: ...
    def characteristic_displacement(self, subdomains: ...) -> None: ...

class CharacteristicDisplacementFromTraction:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def characteristic_contact_traction(self, subdomains: ...) -> None: ...
    def characteristic_displacement(self, subdomains: ...) -> None: ...

class CoulombFrictionBound:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def friction_bound(self, subdomains: ...) -> None: ...
    def friction_coefficient(self, subdomains: ...) -> None: ...

class ShearDilation:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def shear_dilation_gap(self, subdomains: ...) -> None: ...
    def dilation_angle(self, subdomains: ...) -> None: ...

class BartonBandis:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def elastic_normal_fracture_deformation(self, subdomains: ...) -> None: ...
    def maximum_elastic_fracture_opening(self, subdomains: ...) -> None: ...
    def fracture_normal_stiffness(self, subdomains: ...) -> None: ...

class ElasticTangentialFractureDeformation:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def fracture_tangential_stiffness(self, subdomains: ...) -> None: ...
    def elastic_tangential_fracture_deformation(self, subdomains: ...) -> None: ...

class FrictionDamage:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def friction_damage(self, subdomains: ...) -> None: ...
    def initial_friction_damage(self, subdomains: ...) -> None: ...
    def friction_damage_decay(self, subdomains: ...) -> None: ...
    def friction_bound(self, subdomains: ...) -> None: ...

class DilationDamage:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def dilation_damage(self, subdomains: ...) -> None: ...
    def initial_dilation_damage(self, subdomains: ...) -> None: ...
    def dilation_damage_decay(self, subdomains: ...) -> None: ...
    def shear_dilation_gap(self, subdomains: ...) -> None: ...

class BiotCoefficient:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def biot_coefficient(self, subdomains: ...) -> None: ...
    def biot_tensor(self, subdomains: ...) -> None: ...

class SpecificStorage:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def specific_storage(self, subdomains: ...) -> None: ...

class ConstantPorosity:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def porosity(self, subdomains: ...) -> None: ...

class PoroMechanicsPorosity:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def porosity(self, subdomains: ...) -> None: ...
    def matrix_porosity(self, subdomains: ...) -> None: ...
    def reference_porosity(self, subdomains: ...) -> None: ...
    def porosity_change_from_pressure(self, subdomains: ...) -> None: ...
    def porosity_change_from_displacement(self, subdomains: ...) -> None: ...
    def displacement_divergence(self, subdomains: ...) -> None: ...
    def _mpsa_consistency(self, subdomains: ..., physics_name: ..., variable_name: ...) -> None: ...

class BiotPoroMechanicsPorosity:
    # Attributes expected from PorePyModel (via duck typing)
    domain: Any
    equation_system: Any
    fluid: Any
    mdg: Any
    nd: Any
    numerical: Any
    solid: Any
    units: Any

    def porosity_change_from_pressure(self, subdomains: ...) -> None: ...
