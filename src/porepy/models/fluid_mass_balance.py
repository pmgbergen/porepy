"""
Class types:
    MassBalanceEquations defines subdomain and interface equations through the
        terms entering. Darcy type interface relation is assumed.
    Specific ConstitutiveLaws and specific SolutionStrategy for both incompressible
    and compressible case.

Notes:
    Apertures and specific volumes are not included.

    Refactoring needed for constitutive equations. Modularisation and moving to the
    library.

"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence, Union, cast

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class MassBalanceEquations(pp.BalanceEquation):
    """Mixed-dimensional mass balance equation.

    Balance equation for all subdomains and Darcy-type flux relation on all interfaces
    of codimension one and Peaceman flux relation on interfaces of codimension two
    (well-fracture intersections).

    """

    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Darcy flux variable on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    fluid_density: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Fluid density. Defined in a mixin class with a suitable constitutive relation.
    """
    porosity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """
    mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Fluid mobility. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FluidMobility`.

    """
    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """Discretization of the fluid mobility. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FluidMobility`.

    """
    interface_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Discretization of the fluid mobility on internal boundaries. Normally provided
    by a mixin instance of :class:`~porepy.models.constitutive_laws.FluidMobility`.

    """
    advective_flux: Callable[
        [
            list[pp.Grid],
            pp.ad.Operator,
            pp.ad.UpwindAd,
            pp.ad.Operator,
            Callable[[list[pp.MortarGrid]], pp.ad.Operator],
        ],
        pp.ad.Operator,
    ]
    """Ad operator representing the advective flux. Normally provided by a mixin
    instance of :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.

    """
    interface_advective_flux: Callable[
        [list[pp.MortarGrid], pp.ad.Operator, pp.ad.UpwindCouplingAd], pp.ad.Operator
    ]
    """Ad operator representing the advective flux on internal boundaries. Normally
    provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.

    """
    interface_darcy_flux_equation: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Interface Darcy flux equation. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.DarcysLaw`.

    """
    subdomains_to_interfaces: Callable[[list[pp.Grid], list[int]], list[pp.MortarGrid]]
    """Map from subdomains to the adjacent interfaces. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    interfaces_to_subdomains: Callable[[list[pp.MortarGrid]], list[pp.Grid]]
    """Map from interfaces to the adjacent subdomains. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    well_flux_equation: Callable[[list[pp.MortarGrid]], pp.ad.Operator]
    """Well flux equation. Provided e.g. by a mixin instance of
    :class:`~porepy.models.constitutive_laws.PiecmannWellFlux`.

    """
    well_advective_flux: Callable[
        [list[pp.MortarGrid], pp.ad.Operator, pp.ad.UpwindCouplingAd], pp.ad.Operator
    ]
    """Ad operator representing the advective flux on well interfaces. Normally
    provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.AdvectiveFlux`.

    """
    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]

    bc_data_fluid_flux_key: str

    bc_type_fluid_flux: Callable[[pp.Grid], pp.BoundaryCondition]

    _combine_boundary_operators: Callable[
        [
            Sequence[pp.Grid],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            Callable[[pp.Grid], pp.BoundaryCondition],
            str,
            int,
        ],
        pp.ad.Operator,
    ]

    def set_equations(self):
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all subdomains and a Darcy-type flux relation
        is set for all interfaces of codimension one.

        """
        subdomains = self.mdg.subdomains()
        codim_1_interfaces = self.mdg.interfaces(codim=1)
        # TODO: If wells are integrated for nd=2 models, consider refactoring sorting of
        # interfaces into method returning either "normal" or well interfaces.
        codim_2_interfaces = self.mdg.interfaces(codim=2)
        sd_eq = self.mass_balance_equation(subdomains)
        intf_eq = self.interface_darcy_flux_equation(codim_1_interfaces)
        well_eq = self.well_flux_equation(codim_2_interfaces)
        self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})
        self.equation_system.set_equation(intf_eq, codim_1_interfaces, {"cells": 1})
        self.equation_system.set_equation(well_eq, codim_2_interfaces, {"cells": 1})

    def mass_balance_equation(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Mass balance equation for subdomains.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the mass balance equation.

        """
        # Assemble the terms of the mass balance equation.
        accumulation = self.fluid_mass(subdomains)
        flux = self.fluid_flux(subdomains)
        source = self.fluid_source(subdomains)

        # Feed the terms to the general balance equation method.
        eq = self.balance_equation(subdomains, accumulation, flux, source, dim=1)
        eq.set_name("mass_balance_equation")
        return eq

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """The full measure of cell-wise fluid mass.

        The product of fluid density and porosity is assumed constant cell-wise, and
        integrated over the cell volume.

        Note:
            This implementation assumes constant porosity and must be overridden for
            variable porosity. This has to do with wrapping of scalars as vectors or
            matrices and will hopefully be improved in the future. Extension to variable
            density is straightforward.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the cell-wise fluid mass.

        """
        mass_density = self.fluid_density(subdomains) * self.porosity(subdomains)
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name("fluid_mass")
        return mass

    def mobility_rho(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Fluid density times mobility.

        Parameters:
            domains: List of grids to define the operator on.
                Returns a variable-independent representation of boundary conditions if
                called using a list of boundary grids.

        Returns:
            Operator representing the fluid density times mobility.

        """
        return self.fluid_density(domains) * self.mobility(domains)

    def fluid_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Fluid flux as Darcy flux times density and mobility.

        Note:
            The advected entity in the fluid flux is given by :meth:`mobility_rho`.
            When using upwinding, Dirichlet-type data for pressure and temperature
            must also be provided on the Neumann-boundary when there is
            an in-flux into the domain.
            The advected entity must provide values on the boundary in this case, since
            the upstream value of it is on the boundary.

        Parameters:
            domains: List of subdomains or boundary grids.

        Raises:
            ValueError: If the domains are not all grids or all boundary grids.

        Returns:
            Operator representing the fluid flux.

        """
        if len(domains) == 0 or all(isinstance(d, pp.BoundaryGrid) for d in domains):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_fluid_flux_key, domains=domains
            )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)

        discr = self.mobility_discretization(domains)
        mob_rho = self.mobility_rho(domains)

        # )
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.mobility_rho,
            neumann_operator=self.fluid_flux,
            bc_type=self.bc_type_fluid_flux,
            name="bc_values_fluid_flux",
        )
        flux = self.advective_flux(
            domains, mob_rho, discr, boundary_operator, self.interface_fluid_flux
        )
        flux.set_name("fluid_flux")
        return flux

    def interface_flux_equation(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """Interface flux equation.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface flux equation.

        """
        return self.interface_darcy_flux_equation(interfaces)

    def interface_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Interface fluid flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface fluid flux.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        mob_rho = self.mobility(subdomains) * self.fluid_density(subdomains)
        # Call to constitutive law for advective fluxes.
        flux: pp.ad.Operator = self.interface_advective_flux(interfaces, mob_rho, discr)
        flux.set_name("interface_fluid_flux")
        return flux

    def well_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Interface fluid flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface fluid flux.

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        mob_rho = self.mobility(subdomains) * self.fluid_density(subdomains)
        # Call to constitutive law for advective fluxes.
        flux: pp.ad.Operator = self.well_advective_flux(interfaces, mob_rho, discr)
        flux.set_name("well_fluid_flux")
        return flux

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid source term integrated over the subdomain cells.

        Includes:
            - external sources
            - interface flow from neighboring subdomains of higher dimension.
            - well flow from neighboring subdomains of lower and higher dimension.

        .. note::
            When overriding this method to assign internal fluid sources, one is advised
            to call the base class method and add the new contribution, thus ensuring
            that the source term includes the contribution from the interface fluxes.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the source term [kg/s].

        """
        # Interdimensional fluxes manifest as source terms in lower-dimensional
        # subdomains.
        interfaces = self.subdomains_to_interfaces(subdomains, [1])
        well_interfaces = self.subdomains_to_interfaces(subdomains, [2])
        well_subdomains = self.interfaces_to_subdomains(well_interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
        well_projection = pp.ad.MortarProjections(
            self.mdg, well_subdomains, well_interfaces
        )
        subdomain_projection = pp.ad.SubdomainProjections(self.mdg.subdomains())
        source = projection.mortar_to_secondary_int @ self.interface_fluid_flux(
            interfaces
        )
        source.set_name("interface_fluid_flux_source")
        well_fluxes = well_projection.mortar_to_secondary_int @ self.well_fluid_flux(
            well_interfaces
        ) - well_projection.mortar_to_primary_int @ self.well_fluid_flux(
            well_interfaces
        )
        well_fluxes.set_name("well_fluid_flux_source")
        source += subdomain_projection.cell_restriction(subdomains) @ (
            subdomain_projection.cell_prolongation(well_subdomains) @ well_fluxes
        )
        return source


class ConstitutiveLawsSinglePhaseFlow(
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantPermeability,
    pp.constitutive_laws.FluidDensityFromPressure,
    pp.constitutive_laws.ConstantViscosity,
    pp.constitutive_laws.FluidMobility,
):
    """Constitutive equations for single-phase flow.

    The combined laws access the following material constants:

    solid:
        permeability
        normal_permeability
        porosity

    fluid:
        viscosity
        density
        compressibility

    """


class BoundaryConditionsSinglePhaseFlow(pp.BoundaryConditionMixin):
    """Boundary conditions for single-phase flow."""

    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    bc_data_fluid_flux_key: str = "fluid_flux"
    bc_data_darcy_flux_key: str = "darcy_flux"
    """Name of the boundary data for the Neuman boundary condition."""
    pressure_variable: str
    """Name of the pressure variable."""
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned,
            requiring pressure values on the bonudary.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure values for the Dirichlet boundary condition.

        These values are used for quantities relying on Dirichlet data for pressure on
        the boundary, such as mobility, density or Darcy flux.

        Important:
            Override this method to provide custom Dirichlet boundary data for pressure,
            per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape(boundary_grid.num_cells,)`` containing the pressure
            values on the provided boundary grid.

        """
        return self.fluid.pressure() * np.ones(boundary_grid.num_cells)

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """**Volumetric** Darcy flux values for the Neumann boundary condition.

        These values are used on the boundaries where Neumann data for the
        volumetric Darcy :math:`\\mathbf{K}\\nabla p` flux are required.

        Important:
            Override this method to provide custom Neumann data for the flux,
            per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(boundary_grid.num_cells,)`` containing the volumetric
            Darcy flux values on the provided boundary grid.

        """
        return np.zeros(boundary_grid.num_cells)

    def bc_values_fluid_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        r"""**Mass** flux values on the Neumann boundary.

        These values are used on the boundaries where `self.bc_type_fluid_flux` is
        Neumann.

        These values are used on the boundary for
        :math:`\frac{\rho}{\mu} \mathbf{K} \nabla p` where Neumann data is required for
        the whole expression.

        Important:
            Override this method to provide custom Neumann data for the flux,
            per boundary grid as a numpy array with numerical values.

        Parameters:
            boundary_grid: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(boundary_grid.num_cells,)`` containing the mass
            fluid flux values on the provided boundary grid.

        """
        return np.zeros(boundary_grid.num_cells)

    def update_all_boundary_conditions(self) -> None:
        """Set values for the pressure and the darcy flux on boundaries."""
        super().update_all_boundary_conditions()

        self.update_boundary_condition(
            name=self.pressure_variable, function=self.bc_values_pressure
        )
        self.update_boundary_condition(
            name=self.bc_data_darcy_flux_key, function=self.bc_values_darcy_flux
        )
        self.update_boundary_condition(
            name=self.bc_data_fluid_flux_key, function=self.bc_values_fluid_flux
        )


class VariablesSinglePhaseFlow(pp.VariableMixin):
    """
    Creates necessary variables (pressure, interface flux) and provides getter methods
    for these and their reference values. Getters construct mixed-dimensional variables
    on the fly, and can be called on any subset of the grids where the variable is
    defined. Setter method (assig_variables), however, must create on all grids where
    the variable is to be used.

    Note:
        Wrapping in class methods and not calling equation_system directly allows for
        easier changes of primary variables. As long as all calls to fluid_flux() accept
        Operators as return values, we can in theory add it as a primary variable and
        solved mixed form. Similarly for different formulations of the pressure (e.g.
        pressure head) or enthalpy/ temperature for the energy equation.

    """

    equation_system: pp.ad.EquationSystem
    """EquationSystem object for the current model. Normally defined in a mixin class
    defining the solution strategy.

    """
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """
    fluid: pp.FluidConstants
    """Fluid constant object that takes care of scaling of fluid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """
    pressure_variable: str
    """Name of the primary variable representing the pressure. Normally defined in a
    mixin of instance
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.

    """
    interface_darcy_flux_variable: str
    """Name of the primary variable representing the Darcy flux across an interface.
    Normally defined in a mixin of instance
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.

    """
    well_flux_variable: str
    """Name of the primary variable representing the flux across a well interface.
    Normally defined in a mixin of instance
    :class:`~porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow`.

    """
    create_boundary_operator: Callable[
        [str, Sequence[pp.BoundaryGrid]], pp.ad.TimeDependentDenseArray
    ]
    """Boundary conditions wrapped as an operator. Defined in
    :class:`~porepy.models.boundary_condition.BoundaryConditionMixin`.

    """
    nd: int
    """Number of spatial dimensions. Normally defined in a mixin of instance
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    def create_variables(self) -> None:
        """Assign primary variables to subdomains and interfaces of the
        mixed-dimensional grid.

        """
        self.equation_system.create_variables(
            self.pressure_variable,
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "Pa"},
        )
        # Note that `interface_darcy_flux_variable` is not multiplied by rho * mu^-1.
        # However, after multiplication, whe know that the resulting flux should be a
        # mass flux with units  `kg * s^-1`. The units of `interface_darcy_flux` can
        # then be inferred by solving the below equation for `int_flux_units`:
        # kg * s^-1 = [kg * (m^nd)^-1] * [Pa * s]^-1 * intf_flux_units
        self.equation_system.create_variables(
            self.interface_darcy_flux_variable,
            interfaces=self.mdg.interfaces(codim=1),
            tags={"si_units": f"m^{self.nd} * Pa"},
        )
        self.equation_system.create_variables(
            self.well_flux_variable,
            interfaces=self.mdg.interfaces(codim=2),
            tags={"si_units": f"m^{self.nd} * Pa"},
        )

    def pressure(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Pressure term. Either a primary variable if subdomains are provided a
        boundary condition operator if boundary grids are provided.

        Parameters:
            domains: List of subdomains or boundary grids.

        Raises:
            ValueError: If the grids are not all subdomains or all boundary grids.

        Returns:
            Operator representing the pressure.

        """
        if len(domains) > 0 and isinstance(domains[0], pp.BoundaryGrid):
            return self.create_boundary_operator(
                name=self.pressure_variable, domains=domains  # type: ignore[call-arg]
            )
        # Check that all domains are subdomains.
        if not all(isinstance(g, pp.Grid) for g in domains):
            raise ValueError("grids must consist entirely of subdomains.")
        # Now we can cast the grids
        domains = cast(list[pp.Grid], domains)

        return self.equation_system.md_variable(self.pressure_variable, domains)

    def interface_darcy_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Interface Darcy flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the interface Darcy flux.

        """
        flux = self.equation_system.md_variable(
            self.interface_darcy_flux_variable, interfaces
        )
        return flux

    def well_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Well flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the well flux.

        """
        flux = self.equation_system.md_variable(self.well_flux_variable, interfaces)
        return flux

    def reference_pressure(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference pressure.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the reference pressure.

        """
        # TODO: Confirm that this is the right place for this method. # IS: Definitely
        # not a Material. Most closely related to the constitutive laws. # Perhaps
        # create a reference values class that is a mixin to the constitutive laws? #
        # Could have values in the init and methods returning operators just as # this
        # method.
        p_ref = self.fluid.pressure()
        size = sum([sd.num_cells for sd in subdomains])
        return pp.wrap_as_dense_ad_array(p_ref, size, name="reference_pressure")


class SolutionStrategySinglePhaseFlow(pp.SolutionStrategy):
    """Setup and numerics-related methods for a single-phase flow problem.

    At some point, this will be refined to be a more sophisticated (modularised)
    solution strategy class. More refactoring may be beneficial.

    This is *not* a full-scale model (in the old sense), but must be mixed with balance
    equations, constitutive laws etc. See user_examples.

    Parameters:
        params: Parameters for the solution strategy.

    """

    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]

    """Function that returns the specific volume of a subdomain or interface.

    Normally provided by a mixin of instance
    :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """

    permeability: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the permeability of a subdomain. Normally provided by a
    mixin class with a suitable permeability definition.

    """
    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    bc_type_darcy_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Function that returns the boundary condition type for the Darcy flux. Normally
    provided by a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.

    """
    bc_type_fluid_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """Function that returns the boundary condition type for the advective flux.
    Normally provided by a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.

    """
    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """Discretization of the fluid mobility. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.FluidMobility`.

    """
    interface_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Discretization of the fluid mobility on internal boundaries. Normally provided
    by a mixin instance of :class:`~porepy.models.constitutive_laws.FluidMobility`.

    """
    operator_to_SecondOrderTensor: Callable[
        [pp.Grid, pp.ad.Operator, pp.number], pp.SecondOrderTensor
    ]
    """Function that returns a SecondOrderTensor provided a method returning
    permeability as a Operator. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.SecondOrderTensorUtils`.

    """

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)

        # Variables
        self.pressure_variable: str = "pressure"
        """Name of the pressure variable."""

        self.interface_darcy_flux_variable: str = "interface_darcy_flux"
        """Name of the primary variable representing the Darcy flux on interfaces of
        codimension one."""

        self.well_flux_variable: str = "well_flux"
        """Name of the primary variable representing the well flux on interfaces of
        codimension two."""

        # Discretization
        self.darcy_keyword: str = "flow"
        """Keyword for Darcy flux term.

        Used to access discretization parameters and store discretization matrices.

        """
        self.mobility_keyword: str = "mobility"
        """Keyword for mobility factor.

        Used to access discretization parameters and store discretization matrices.

        """

    def initial_condition(self) -> None:
        """New formulation requires darcy flux (the flux is "advective" with mobilities
        included).

        """
        super().initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.mobility_keyword,
                {"darcy_flux": np.zeros(sd.num_faces)},
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.mobility_keyword,
                {"darcy_flux": np.zeros(intf.num_cells)},
            )

    def set_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameter fields of the data dictionaries are updated for all subdomains and
        interfaces. The data to be set is related to:
            * The fluid diffusion, e.g., the permeability and boundary conditions for
              the pressure. This applies to both subdomains and interfaces.
            * Boundary conditions for the advective flux. This applies to subdomains
              only.

        """
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.darcy_keyword,
                {
                    "bc": self.bc_type_darcy_flux(sd),
                    "second_order_tensor": self.operator_to_SecondOrderTensor(
                        sd, self.permeability([sd]), self.solid.permeability()
                    ),
                    "ambient_dimension": self.nd,
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.mobility_keyword,
                {
                    "bc": self.bc_type_fluid_flux(sd),
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for intf, intf_data in self.mdg.interfaces(return_data=True, codim=1):
            pp.initialize_data(
                intf,
                intf_data,
                self.darcy_keyword,
                {
                    "ambient_dimension": self.nd,
                },
            )

    def before_nonlinear_iteration(self):
        """
        Evaluate Darcy flux for each subdomain and interface and store in the data
        dictionary for use in upstream weighting.

        """
        # Update parameters *before* the discretization matrices are re-computed.
        for sd, data in self.mdg.subdomains(return_data=True):
            vals = self.darcy_flux([sd]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = self.interface_darcy_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            vals = self.well_flux([intf]).value(self.equation_system)
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        super().before_nonlinear_iteration()

    def set_nonlinear_discretizations(self) -> None:
        """Collect discretizations for nonlinear terms."""
        super().set_nonlinear_discretizations()
        self.add_nonlinear_discretization(
            self.mobility_discretization(self.mdg.subdomains()).upwind,
        )
        self.add_nonlinear_discretization(
            self.interface_mobility_discretization(self.mdg.interfaces()).flux,
        )


# Note that we ignore a mypy error here. There are some inconsistencies in the method
# definitions of the mixins, related to the enforcement of keyword-only arguments. The
# type Callable is poorly supported, except if protocols are used and we really do not
# want to go there. Specifically, method definitions that contains a *, for instance,
#   def method(a: int, *, b: int) -> None: pass
# which should be types as Callable[[int, int], None], cannot be parsed by mypy.
# For this reason, we ignore the error here, and rely on the tests to catch any
# inconsistencies.
class SinglePhaseFlow(  # type: ignore[misc]
    MassBalanceEquations,
    VariablesSinglePhaseFlow,
    ConstitutiveLawsSinglePhaseFlow,
    BoundaryConditionsSinglePhaseFlow,
    SolutionStrategySinglePhaseFlow,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for single-phase flow in mixed-dimensional porous media."""
