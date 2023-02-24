"""Module for printing docstrings of variables and methods from model mixin classes.

Mixins have introduced the need to declare types of methods and attributes that will be
supplied by other mixin classes. These types are normally declared at the very top of
the class, with their associated docstring.

This potentially introduces a great deal of non-uniformity, in addition to the tedious
process of writing the docstrings. The purpose of this module is to provide a simple
way of accessing _suggested_ docstrings for commonly used methods and variables
defined in the model's framework.

These are stored as entries (see :class:`Entry`) in the :class:`Glossary` data class.
Each entry contains three string attributes: the name, the type, and the docstring.
The intended usage is to print an entry via :func:`print_glossary_entry()` (e.g.,
in an iPython session or similar) and copy-paste the suggested docstring into a file.

Notes:
    Depending on the usage, entries might NOT be completely accurate. It is therefore
    the responsibility of the user/developer to make sure that the docstrings are
    correct.

Examples:

    .. code: python3

        import porepy as pp

        # Print a single entry
        pp.print_glossary_entry(pp.Glossary.mdg)

        # Print all entries currently stored in the glossary
        pp.Glossary().print_all_entries()

"""
from __future__ import annotations

from dataclasses import dataclass, fields
from textwrap import wrap
from typing import NamedTuple


class Entry(NamedTuple):
    """Named tuple class to store attributes of glossary entries."""

    type: str
    """Name of the class associated with the entry. For variables, the actual class. For
    functions and methods, a variant of ``Callable[[arg_type], return_type]``.

    """

    docstring: str
    """Suggested docstring for the entry."""

    name: str
    """Suggested variable/method name."""


@dataclass
class Glossary:
    """Data class to store entries as named tuples.

    Notes:
          Please insert new entries respecting the alphabetical order.

    """

    advective_flux: Entry = Entry(
        type="Callable[[list[pp.Grid], pp.ad.Operator, pp.ad.UpwindAd,"
        " pp.ad.Operator, Callable[[list[pp.MortarGrid]], pp.ad.Operator]],"
        " pp.ad.Operator]",
        docstring="Ad operator representing the advective flux. Normally provided by a"
        " mixin instance of :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.",
        name="advective_flux",
    )

    aperture: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Function that returns the aperture of a subdomain. Normally provided"
        " by a mixin of instance"
        " :class:`~porepy.models.constitutive_laws.DimensionReduction`.",
        name="aperture",
    )

    balance_equation: Entry = Entry(
        type="Callable[[list[pp.Grid], pp.ad.Operator, pp.ad.Operator, pp.ad.Operator,"
        " int], pp.ad.Operator]",
        docstring="Make a balance equation on subdomains. Normally defined in a mixin"
        " instance of :class:`~porepy.models.abstract_equations.BalanceEquation`.",
        name="balance_equation",
    )

    basis: Entry = Entry(
        type="Callable[[Sequence[pp.GridLike], int], list[pp.ad.Matrix]]",
        docstring="Basis for the local coordinate system. Normally set by a mixin"
        " instance of :class:`porepy.models.geometry.ModelGeometry`.",
        name="basis",
    )

    bc_type_darcy: Entry = Entry(
        type="Callable[[pp.Grid], pp.BoundaryCondition]",
        docstring="Function that returns the boundary condition type for the Darcy"
        " flux. Normally provided by a mixin instance of :class:`~porepy.models."
        "fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.",
        name="bc_type_darcy",
    )

    bc_type_enthalpy: Entry = Entry(
        type="Callable[[pp.Grid], pp.ad.Array]",
        docstring="Function that returns the boundary condition type for the enthalpy"
        " flux. Normally defined in a mixin instance of :class:`~porepy.models."
        "fluid_mass_balance.BoundaryConditionsEnergyBalance`.",
        name="bc_type_enthalpy",
    )

    bc_type_fourier: Entry = Entry(
        type="Callable[[pp.Grid], pp.ad.Array]",
        docstring="Function that returns the boundary condition type for the Fourier"
        " flux. Normally defined in a mixin instance of :class:`~porepy.models."
        "fluid_mass_balance.BoundaryConditionsEnergyBalance`.",
        name="bc_type_fourier",
    )

    bc_type_mobrho: Entry = Entry(
        type="Callable[[pp.Grid], pp.BoundaryCondition]",
        docstring="Function that returns the boundary condition type for the advective"
        " flux. Normally provided by a mixin instance of :class:`~porepy.models."
        "fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.",
        name="bc_type_mobrho",
    )

    bc_type_mechanics: Entry = Entry(
        type="Callable[[pp.Grid], pp.BoundaryConditionVectorial]",
        docstring="Function that returns the boundary condition type for the momentum"
        " problem. Normally provided by a mixin instance of :class:`~porepy.models."
        "momentum_balance.BoundaryConditionsMomentumBalance`.",
        name="bc_type_mechanics",
    )

    bc_values_darcy: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Array]",
        docstring="Darcy flux boundary conditions. Normally defined in a mixin instance"
        " of :class:`~porepy.models.fluid_mass_balance"
        ".BoundaryConditionsSinglePhaseFlow`.",
        name="bc_values_darcy",
    )

    bc_values_enthalpy_flux: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Array]",
        docstring="Boundary condition for enthalpy flux. Normally defined in a mixin"
        " instance of"
        " :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsEnergyBalance`",
        name="bc_values_enthalpy_flux",
    )

    bc_values_fourier: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Array]",
        docstring="Fourier flux boundary conditions. Normally defined in a mixin"
        " instance of :class:`~porepy.models.fluid_mass_balance.BoundaryCondition.",
        name="bc_values_fourier",
    )

    bc_values_mechanics: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Array]",
        docstring="Mechanics boundary conditions. Normally defined in a mixin instance"
        " of :class:`~porepy.models.fluid_mass_balance"
        ".BoundaryConditionsMomentumBalance`.",
        name="bc_values_mechanics",
    )

    bc_values_mobrho: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Array]",
        docstring="Mobility times density boundary conditions. Normally defined in a"
        " mixin instance of :class:`~porepy.models.fluid_mass_balance"
        ".BoundaryConditionsSinglePhaseFlow`.",
        name="bc_values_mobrho",
    )

    biot_coefficient: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Biot coefficient. Normally defined in a mixin instance of "
        " :class:`~porepy.models.constitutive_laws.BiotCoefficient`.",
        name="biot_coefficient",
    )

    bulk_modulus: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Bulk modulus. Normally defined in a mixin instance of"
        " :class:`~porepy.models.constitutive_laws.LinearElasticSolid`.",
        name="bulk_modulus",
    )

    contact_mechanics_numerical_constant: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Scalar]",
        docstring="Numerical constant for contact mechanics. Normally provided by an"
        " instance of"
        " :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.",
        name="contact_mechanics_numerical_constant",
    )

    contact_traction: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]",
        docstring="Contact traction variable. Normally defined in a mixin instance of"
        " :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.",
        name="contact_traction",
    )

    contact_traction_variable: Entry = Entry(
        type="str",
        docstring="Name of the primary variable representing the contact traction on a"
        " fracture subdomain. Normally defined by an instance of"
        " :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.",
        name="contact_traction_variable",
    )

    darcy_keyword: Entry = Entry(
        type="str",
        docstring="Keyword used to identify the Darcy flux discretization. Normally"
        " set by an instance of"
        " :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.",
        name="darcy_keyword",
    )

    displacement: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]",
        docstring="Displacement variable. Normally defined in a mixin instance of"
        " :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.",
        name="displacement",
    )

    displacement_jump: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Operator giving the displacement jump on fracture grids. Normally"
        " defined in a mixin instance of"
        " :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.",
        name="displacement_jump",
    )

    displacement_variable: Entry = Entry(
        type="str",
        docstring="Name of the primary variable representing the displacement in"
        " subdomains. Normally defined by an instance of"
        " :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.",
        name="displacement_variable",
    )

    domain_boundary_sides: Entry = Entry(
        type="Callable[[pp.Grid], pp.domain.DomainSides]",
        docstring="Boundary sides of the domain. Normally defined in a mixin instance"
        " of :class:`~porepy.models.geometry.ModelGeometry`.",
        name="domain_boundary_sides",
    )

    enthalpy_keyword: Entry = Entry(
        type="str",
        docstring="Keyword used to identify the enthalpy flux discretization. Normally"
        " set by an instance of"
        " :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.",
        name="enthalpy_keyword",
    )

    equation_system: Entry = Entry(
        type="pp.ad.EquationSystem",
        docstring="EquationSystem object for the current model. Normally defined in a"
        " mixin class defining the solution strategy.",
        name="equation_system",
    )

    e_i: Entry = Entry(
        type="Callable[[Sequence[pp.GridLike], int, int], pp.ad.Matrix]",
        docstring="A unit vector in a local coordinate system. Normally set by a mixin"
        " instance of :class:`porepy.models.geometry.ModelGeometry`.",
        name="e_i",
    )

    finalize_data_saving: Entry = Entry(
        type="Callable[[], None]",
        docstring="Finalize data saving. Normally provided by a mixin instance of"
        " :class:`~porepy.viz.data_saving_model_mixin.DataSavingMixin`.",
        name="finalize_data_saving",
    )

    fluid: Entry = Entry(
        type="pp.FluidConstants",
        docstring="Fluid constant object that takes care of storing and scaling"
        " numerical values representing fluid-related quantities. Normally, this is"
        " set by an instance of"
        " :class:`~porepy.models.solution_strategy.SolutionStrategy`.",
        name="fluid",
    )

    fluid_density: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Fluid density. Normally defined in a mixin class with a suitable"
        " constitutive relation, e.g.,"
        " :class:`~porepy.models.constitutive_laws.FluidDensityFromPressure`.",
        name="fluid_density",
    )

    fluid_enthalpy: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Variable for interface enthalpy flux. Normally provided by a mixin"
        " instance of :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.",
        name="fluid_enthalpy",
    )

    fluid_internal_energy: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Fluid internal energy. Normally defined in a mixin class with a"
        " suitable constitutive relation.",
        name="fluid_internal_energy",
    )

    fourier_flux: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Fourier flux. Normally provided by a mixin instance"
        " of :class:`~porepy.models.constitutive_laws.FouriersLaw`.",
        name="fourier_flux",
    )

    fourier_keyword: Entry = Entry(
        type="str",
        docstring="Keyword used to identify the Fourier flux discretization. Normally"
        " set by an instance of"
        " :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.",
        name="fourier_keyword",
    )

    fracture_stress: Entry = Entry(
        type="Callable[[list[pp.MortarGrid]], pp.ad.Operator]",
        docstring="Stress on the fracture faces. Provided by a suitable mixin class"
        " that specifies the physical laws governing the stress, see for instance"
        " :class:`~porepy.models.constitutive_laws.LinearElasticMechanicalStress` or"
        " :class:`~porepy.models.constitutive_laws.PressureStress`.",
        name="fracture_stress",
    )

    friction_bound: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Friction bound of a fracture. Normally provided by a mixin instance"
        " of :class:`~porepy.models.constitutive_laws.FrictionBound`.",
        name="friction_bound",
    )

    friction_coefficient: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Friction coefficient. Normally defined in a mixin instance of"
        " :class:`~porepy.models.constitutive_laws.FracturedSolid`.",
        name="friction_coefficient",
    )

    gap: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Gap of a fracture. Normally provided by a mixin instance of"
        " :class:`~porepy.models.constitutive_laws.FracturedSolid`.",
        name="gap",
    )

    local_coordinates: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Matrix]",
        docstring="Mapping to local coordinates. Normally defined in a mixin instance"
        " of :class:`~porepy.models.geometry.ModelGeometry`.",
        name="local_coordinates",
    )

    initialize_data_saving: Entry = Entry(
        type="Callable[[], None]",
        docstring="Initialize data saving. Normally provided by a mixin instance of"
        " :class:`~porepy.viz.data_saving_model_mixin.DataSavingMixin`.",
        name="initialize_data_saving",
    )

    interfaces_to_subdomains: Entry = Entry(
        type="Callable[[list[pp.MortarGrid]], list[pp.Grid]]",
        docstring="Map from interfaces to the adjacent subdomains. Normally defined in"
        " a mixin instance of :class:`~porepy.models.geometry.ModelGeometry`.",
        name="interfaces_to_subdomains",
    )

    interface_advective_flux: Entry = Entry(
        type="Callable[[list[pp.MortarGrid], pp.ad.Operator, pp.ad.UpwindCouplingAd],"
        " pp.ad.Operator]",
        docstring="Ad operator representing the advective flux on internal"
        " boundaries. Normally provided by a mixin instance of"
        " :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.",
        name="interface_advective_flux",
    )

    interface_darcy_flux: Entry = Entry(
        type="Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]",
        docstring="Darcy flux variable on interfaces. Normally defined in a mixin"
        " instance of"
        " :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.",
        name="interface_darcy_flux",
    )

    interface_darcy_flux_equation: Entry = Entry(
        type="Callable[[list[pp.MortarGrid]], pp.ad.Operator]",
        docstring="Interface Darcy flux equation. Normally provided by a mixin"
        " instance of :class:`~porepy.models.constitutive_laws.DarcysLaw`.",
        name="interface_darcy_flux_equation",
    )

    interface_darcy_flux_variable: Entry = Entry(
        type="str",
        docstring="Name of the primary variable representing the Darcy flux across an"
        " interface. Normally defined by an instance of"
        " :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.",
        name="interface_darcy_flux_variable",
    )

    interface_displacement: Entry = Entry(
        type="str",
        docstring="Displacement variable on interfaces. Normally defined in a mixin"
        " instance of"
        " :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.",
        name="interface_displacement",
    )

    interface_displacement_variable: Entry = Entry(
        type="str",
        docstring="Name of the primary variable representing the displacement on an"
        " interface.Normally defined by an instance of"
        " :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.",
        name="interface_displacement_variable",
    )

    interface_enthalpy_flux: Entry = Entry(
        type="Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]",
        docstring="Variable for interface enthalpy flux. Normally provided by a mixin"
        " instance of :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.",
        name="interface_enthalpy_flux",
    )

    interface_enthalpy_flux_variable: Entry = Entry(
        type="str",
        docstring="Name of the primary variable representing the enthalpy flux across"
        " an interface. Normally defined by an instance of"
        " :class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.",
        name="interface_enthalpy_flux_variable",
    )

    interface_fourier_flux: Entry = Entry(
        type="Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]",
        docstring="Fourier flux variable on interfaces. Normally defined in a mixin"
        " instance of :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.",
        name="interface_fourier_flux",
    )

    interface_fourier_flux_variable: Entry = Entry(
        type="str",
        docstring="Name of the primary variable representing the Fourier flux across"
        " an interface. Normally defined by an instance of"
        ":class:`~porepy.models.fluid_mass_balance.SolutionStrategyEnergyBalance`.",
        name="interface_fourier_flux_variable",
    )

    interface_mobility_discretization: Entry = Entry(
        type="Callable[[list[pp.MortarGrid]], pp.ad.UpwindCouplingAd]",
        docstring="Discretization of the fluid mobility on internal boundaries."
        " Normally providedby a mixin instance of"
        " :class:`~porepy.models.constitutive_laws.FluidMobility`.",
        name="interface_mobility_discretization",
    )

    internal_boundary_normal_to_outwards: Entry = Entry(
        type="Callable[[list[pp.Grid], int], pp.ad.Matrix]",
        docstring="Switch interface normal vectors to point outwards from the"
        " subdomain. Normally set by a mixin instance of "
        ":class:`porepy.models.geometry.ModelGeometry`.",
        name="internal_boundary_normal_to_outwards",
    )

    mdg: Entry = Entry(
        type="pp.MixedDimensionalGrid",
        docstring="Mixed-dimensional grid for the current model. Normally defined in"
        " a mixin instance of :class:`~porepy.models.geometry.ModelGeometry`.",
        name="mdg",
    )

    mobility: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Fluid mobility. Normally provided by a mixin instance of"
        " :class:`~porepy.models.constitutive_laws.FluidMobility`.",
        name="mobility",
    )

    mobility_discretization: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.UpwindAd]",
        docstring="Discretization of the fluid mobility. Normally provided by a mixin"
        " instance of :class:`~porepy.models.constitutive_laws.FluidMobility`.",
        name="mobility_discretization",
    )

    nd: Entry = Entry(
        type="int",
        docstring="Ambient dimension of the problem. Normally set by a mixin instance"
        " of :class:`porepy.models.geometry.ModelGeometry`.",
        name="nd",
    )

    normal_thermal_conductivity: Entry = Entry(
        type="Callable[[list[pp.MortarGrid]], pp.ad.Operator]",
        docstring="Conductivity on a mortar grid. Normally defined in a mixin instance"
        " of :class:`~porepy.models.constitutive_laws.ThermalConductivityLTE` or a"
        " subclass.",
        name="normal_thermal_conductivity",
    )

    outwards_internal_boundary_normals: Entry = Entry(
        type="Callable[[list[pp.MortarGrid], bool], pp.ad.Operator]",
        docstring="Outwards normal vectors on internal boundaries. Normally defined in"
        " a mixin instance of :class:`~porepy.models.geometry.ModelGeometry`.",
        name="outwards_internal_boundary_normals",
    )

    permeability: Entry = Entry(
        type="Callable[[pp.Grid], Union[pp.ad.Operator, np.ndarray]]",
        docstring="Function that returns the permeability of a subdomain. Normally"
        " provided by a mixin class with a suitable permeability definition.",
        name="permeability",
    )

    perturbation_from_reference: Entry = Entry(
        type="Callable[[str, list[pp.Grid]], pp.ad.Operator]",
        docstring="Function that returns a perturbation from reference state. Normally"
        " provided by a mixin of instance :class:`~porepy.models.VariableMixin`.",
        name="perturbation_from_reference",
    )

    porosity: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Porosity of the rock. Normally provided by a mixin instance of"
        " :class:`~porepy.models.constitutive_laws.ConstantPorosity`"
        " or a subclass thereof.",
        name="porosity",
    )

    pressure: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]",
        docstring="Pressure variable. Normally defined in a mixin instance of"
        " :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.",
        name="pressure",
    )

    pressure_variable: Entry = Entry(
        type="str",
        docstring="Name of the primary variable representing the pressure. Normally"
        " defined by an instance of"
        " :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.",
        name="pressure_variable",
    )

    reference_porosity: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Reference porosity of the rock. Normally provided by a mixin"
        " instance of :class:`~porepy.models.constitutive_laws.PoroMechanicsPorosity`.",
        name="reference_porosity",
    )

    reference_pressure: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Reference pressure. Normally defined in a mixin instance of"
        " :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.",
        name="reference_pressure",
    )

    reference_temperature: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Reference temperature. Normally defined in a mixin instance of"
        " :class:`~porepy.models.energy_balance.VariablesEnergyBalance`.",
        name="reference_temperature",
    )

    save_data_time_step: Entry = Entry(
        type="Callable[[], None]",
        docstring="Save data at the end of a time step. Normally provided by a mixin"
        " instance of"
        " :class:`~porepy.viz.data_saving_model_mixin.DataSavingMixin`.",
        name="save_data_time_step",
    )

    set_equations: Entry = Entry(
        type="Callable[[], None]",
        docstring="Set the governing equations of the model. Normally provided by the"
        " solution strategy of a specific model (i.e., a subclass of this class).",
        name="set_equations",
    )

    set_geometry: Entry = Entry(
        type="Callable[[], None]",
        docstring="Set the geometry of the model. Normally provided by a mixin instance"
        " of :class:`~porepy.models.geometry.ModelGeometry`.",
        name="set_geometry",
    )

    solid: Entry = Entry(
        type="pp.SolidConstants",
        docstring="Solid constant object that takes care of storing and scaling"
        " numerical values representing solid-related"
        " quantities. Normally, this is set by an instance of"
        " :class:`~porepy.models.solution_strategy.SolutionStrategy`.",
        name="solid",
    )

    solid_density: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Solid density. Defined in a mixin class with a suitable"
        " constitutive relation,"
        " e.g., :class:`~porepy.models.constitutive_laws.ConstantSolidDensity`.",
        name="solid_density",
    )

    solid_thermal_expansion: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Thermal expansion coefficient. Normally defined in a mixin instance"
        " of :class:`~porepy.models.constitutive_laws.ThermalExpansion`.",
        name="solid_thermal_expansion",
    )

    subdomain_projections: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Projections between subdomains. Normally defined in a mixin"
        " instance of :class:`~porepy.models.geometry.ModelGeometry`.",
        name="subdomain_projections",
    )

    specific_volume: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Function that returns the specific volume of a subdomain. Normally"
        " provided by a mixin of instance"
        " :class:`~porepy.models.constitutive_laws.DimensionReduction`.",
        name="specific_volume",
    )

    stiffness_tensor: Entry = Entry(
        type="Callable[[pp.Grid], pp.FourthOrderTensor]",
        docstring="Function that returns the stiffness tensor of a subdomain. Normally"
        " provided by a mixin of instance"
        " :class:`~porepy.models.constitutive_laws.LinearElasticSolid`.",
        name="stiffness_tensor",
    )

    stress: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Stress on the grid faces. Provided by a suitable mixin class that"
        " specifies the physical laws governing the stress, e.g.,"
        " :class:`~porepy.models.constitutive_laws.LinearElasticMechanicalStress.`",
        name="stress",
    )

    stress_keyword: Entry = Entry(
        type="str",
        docstring="Keyword used to identify the stress discretization. Normally set by"
        " an instance of"
        " :class:`~porepy.models.momentum_balance.SolutionStrategyMomentumBalance`.",
        name="stress_keyword",
    )

    tangential_component: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Matrix]",
        docstring="Operator giving the tangential component of vectors. Normally"
        " defined in a mixin instance of "
        " :class:`~porepy.models.models.ModelGeometry`.",
        name="tangential_component",
    )

    thermal_conductivity: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Thermal conductivity. Normally defined in a mixin instance of"
        " :class:`~porepy.models.constitutive_laws.ThermalConductivityLTE` or a"
        " subclass.",
        name="thermal_conductivity",
    )

    thermal_expansion: Entry = Entry(
        type="Callable[[list[pp.Grid]], pp.ad.Operator]",
        docstring="Thermal expansion coefficient. Normally defined in a mixin class"
        " with a suitable thermal expansion definition.",
        name="thermal_expansion",
    )

    time_dependent_bc_values_mechanics: Entry = Entry(
        type="Callable[[list[pp.Grid]], np.ndarray]",
        docstring="Values of the mechanical boundary conditions for a time-dependent"
        " problem. Normally set by a mixin instance of :class:`~porepy.models."
        "poromechanics.BoundaryConditionsMechanicsTimeDependent`.",
        name="time_dependent_bc_values_mechanics",
    )

    time_manager: Entry = Entry(
        type="pp.TimeManager",
        docstring="Time manager. Normally set by an instance of a subclass of"
        " :class:`porepy.models.solution_strategy.SolutionStrategy`.",
        name="time_manager",
    )

    volume_integral: Entry = Entry(
        type="Callable[[Union[list[pp.Grid], list[pp.MortarGrid]],"
        " pp.ad.Operator, int], pp.ad.Operator]",
        docstring="Operator giving the tangential component of vectors. Normally"
        " defined in a mixin instance of"
        " :class:`~porepy.models.abstract_equations.BalanceEquation`.",
        name="volume_integral",
    )

    wrap_grid_attribute: Entry = Entry(
        type="Callable[[Sequence[pp.GridLike], str, int, bool], pp.ad.Matrix]",
        docstring="Wrap grid attributes as Ad operators. Normally set by a mixin"
        " instance of :class:`porepy.models.geometry.ModelGeometry`.",
        name="wrap_grid_attribute",
    )

    def print_all_entries(self) -> None:
        """Prints all stored entries."""
        for field in fields(Glossary):
            entry = field.default
            print_glossary_entry(entry)  # type: ignore[arg-type]

    def num_entries(self) -> int:
        """Number of entries stored in the glossary.

        Returns:
            Number of entries that are stored in the glossary.

        """
        return len([field.name for field in fields(Glossary)])


def print_glossary_entry(entry: Entry, wrap_at: int = 88, offset: int = 4) -> None:
    """Print copy-and-paste-ready docstring of a glossary entry.

    Parameters:
        entry: An instance of the :class:`~Glossary` data class.
        wrap_at: Maximum number of admissible characters in one line. Default is 88.
        offset: Number of spaces accounted for indentation. Default is 4.

    """
    # Retrieve raw docstring
    raw: str = entry.docstring

    # Pad string with """ """
    padded: str = '"""' + raw + '"""'

    # Wrap string if it is too long
    wrapped: list[str] = wrap(padded, wrap_at - offset)

    # Print header, e.g., entry name and its type
    print(f"\n{entry.name}: {entry.type}")

    # Print docstring
    if len(wrapped) > 1:  # the multiline docstring scenario
        # Remove `"""` from last line
        last_line = wrapped[-1].strip('"')
        wrapped[-1] = last_line
        for line in wrapped:
            print(line)
        print("\n" + '"""')
    else:  # the single line docstring scenario
        print(wrapped[0])
