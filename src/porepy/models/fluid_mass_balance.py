"""
Class types:
    FluidMassBalanceEquations defines subdomain and interface equations through the
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
from typing import Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class FluidMassBalanceEquations(pp.BalanceEquation):
    """Mixed-dimensional balance equation for total mass (pressure equation).

    Balance equation for all subdomains and Darcy-type flux relation on all interfaces
    of codimension one and Peaceman flux relation on interfaces of codimension two
    (well-fracture intersections).

    """

    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """Darcy flux variable on interfaces. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """
    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """
    total_mass_mobility: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Total fluid mobility. Normally provided by a mixin instance of
    :class:`~porepy.models.fluid_property_library.FluidMobility`.

    """
    mobility_discretization: Callable[[list[pp.Grid]], pp.ad.UpwindAd]
    """Discretization of the fluid mobility. Normally provided by a mixin instance of
    :class:`~porepy.models.fluid_property_library.FluidMobility`.

    """
    interface_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Discretization of the fluid mobility on internal boundaries. Normally provided
    by a mixin instance of :class:`~porepy.models.fluid_property_library.FluidMobility`.

    """
    advective_flux: Callable[
        [
            list[pp.Grid],
            pp.ad.Operator,
            pp.ad.UpwindAd,
            pp.ad.Operator,
            Optional[Callable[[list[pp.MortarGrid]], pp.ad.Operator]],
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
    bc_data_fluid_flux_key: str
    """See :class:`BoundaryConditionsSinglePhaseFlow`.
    """
    bc_type_fluid_flux: Callable[[pp.Grid], pp.BoundaryCondition]
    """See :class:`BoundaryConditionsSinglePhaseFlow`.
    """

    @staticmethod
    def primary_equation_name() -> str:
        """Returns the string which is used to name the pressure equation on all
        subdomains, which is the primary PDE set by this class.

        Important:
            When using this class in a mixin-setting, do not use
            ``self.primary_equation_name()``, as other equations might have this method
            implemented as well.

            This is a static method, utilize it. I.e., use
            ``FluidMassBalanceEquations.primary_equation_name()``.

        """
        return "mass_balance_equation"

    def set_equations(self) -> None:
        """Set the equations for the mass balance problem.

        A mass balance equation is set for all subdomains and a Darcy-type flux relation
        is set for all interfaces of codimension one.

        """
        super().set_equations()
        subdomains = self.mdg.subdomains()
        codim_1_interfaces = self.mdg.interfaces(codim=1)
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
        eq.set_name(FluidMassBalanceEquations.primary_equation_name())
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
        mass_density = self.fluid.density(subdomains) * self.porosity(subdomains)
        mass = self.volume_integral(mass_density, subdomains, dim=1)
        mass.set_name("fluid_mass")
        return mass

    def advection_weight_mass_balance(
        self, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Fluid density divided by viscosity [kg * m^(-3) * Pa^(-1) * s^(-1)].

        Note:
            Depending on the literature, this may also be called total mobility (CF).

        Parameters:
            domains: List of grids to define the operator on.
                Returns a variable-independent representation of boundary conditions if
                called using a list of boundary grids.

        Returns:
            Operator representing the fluid density times mobility [s * m^-2].

        """
        return self.total_mass_mobility(domains)

    def fluid_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Fluid flux as Darcy flux times density and mobility.

        Note:
            The advected entity in the fluid flux is given by
            :meth:`advection_weight_mass_balance`. When using upwinding, Dirichlet-type
            data for pressure and temperature must also be provided on the
            Neumann-boundary when there is an in-flux into the domain. The advected
            entity must provide values on the boundary in this case, since the upstream
            value of it is on the boundary.

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
            return self.create_boundary_operator(
                name=self.bc_data_fluid_flux_key,
                domains=cast(Sequence[pp.BoundaryGrid], domains),
            )

        # Verify that the domains are subdomains.
        if not all(isinstance(d, pp.Grid) for d in domains):
            raise ValueError("domains must consist entirely of subdomains.")
        # Now we can cast the domains
        domains = cast(list[pp.Grid], domains)

        flux = self.advective_flux(
            domains,
            self.advection_weight_mass_balance(domains),
            self.mobility_discretization(domains),
            self.boundary_fluid_flux(domains),
            self.interface_fluid_flux,
        )
        flux.set_name("fluid_flux")
        return flux

    def boundary_fluid_flux(self, subdomains: Sequence[pp.Grid]) -> pp.ad.Operator:
        """Combined representation of the fluid flux on the boundaries of
        ``subdomains``.

        This base uses the :meth:`fluid_flux` as the Neumann-type operator, and the
        :meth:`advection_weight_mass_balance` as the Dirichlet-type operator.
        The former assumes that the total fluid flux on the Neumann-type boundary is
        explicitly given by the user.

        The boundary fluid flux is used in Upwinding (see :meth:`advective_flux`), i.e.
        it is a massic flux. Note however, that this is a numerical approximation of the
        otherwise diffusive total fluid flux.

        Note:
            This operator does not necessarily contain flux values per se. If
            ``bc_type_fluid_flux`` indicates that the in/out-flux is given using
            Dirichlet-type data, the values correspond to the values of the non-linear
            weight in the massic flux.

        Parameters:
            subdomains: A sequence of grids on whose boundaries the fluid flux is
                accessed.

        Returns:
            The massic fluid flux on the boundary to be used for the Upwinding scheme.

        """
        return self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.advection_weight_mass_balance,
            neumann_operator=self.fluid_flux,
            # Robin operator is not relevant for advective fluxes.
            robin_operator=None,
            bc_type=self.bc_type_fluid_flux,
            name="bc_values_fluid_flux",
        )

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
        mob_rho = self.advection_weight_mass_balance(subdomains)
        # Call to constitutive law for advective fluxes.
        flux: pp.ad.Operator = self.interface_advective_flux(interfaces, mob_rho, discr)
        flux.set_name("interface_fluid_flux")
        return flux

    def well_fluid_flux(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Interface fluid flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the interface fluid flux [kg * s^-1].

        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        discr = self.interface_mobility_discretization(interfaces)
        mob_rho = self.advection_weight_mass_balance(subdomains)
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

        Note:
            When overriding this method to assign internal fluid sources, one is advised
            to call the base class method and add the new contribution, thus ensuring
            that the source term includes the contribution from the interface fluxes.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the source term [kg * s^-1].

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
        source = projection.mortar_to_secondary_int() @ self.interface_fluid_flux(
            interfaces
        )
        source.set_name("interface_fluid_flux_source")
        well_fluxes = well_projection.mortar_to_secondary_int() @ self.well_fluid_flux(
            well_interfaces
        ) - well_projection.mortar_to_primary_int() @ self.well_fluid_flux(
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
    pp.constitutive_laws.SecondOrderTensorUtils,
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

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Pressure values for the Dirichlet boundary condition.

        These values are used for quantities relying on Dirichlet data for pressure on
        the boundary, such as mobility, density or Darcy flux.

        Important:
            Override this method to provide custom Dirichlet boundary data for pressure,
            per boundary grid as a numpy array with numerical values.

        Parameters:
            bg: Boundary grid to provide values for.

        Returns:
            An array with ``shape(bg.num_cells,)`` containing the pressure values on the
            provided boundary grid.

        """
        return self.reference_variable_values.pressure * np.ones(bg.num_cells)

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """**Volumetric** Darcy flux values for the Neumann boundary condition.

        These values are used on the boundaries where Neumann data for the
        volumetric Darcy :math:`\\mathbf{K}\\nabla p` flux are required.

        Important:
            Override this method to provide custom Neumann data for the flux,
            per boundary grid as a numpy array with numerical values.

        Parameters:
            bg: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the volumetric Darcy flux
            values on the provided boundary grid.

        """
        return np.zeros(bg.num_cells)

    def bc_values_fluid_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
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
            bg: Boundary grid to provide values for.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the mass fluid flux
            values on the provided boundary grid.

        """
        return np.zeros(bg.num_cells)

    def update_all_boundary_conditions(self) -> None:
        """Set values for the pressure and the darcy flux on boundaries."""
        super().update_all_boundary_conditions()

        self.update_boundary_condition(
            name=self.bc_data_darcy_flux_key, function=self.bc_values_darcy_flux
        )
        self.update_boundary_condition(
            name=self.bc_data_fluid_flux_key, function=self.bc_values_fluid_flux
        )

    def update_boundary_values_primary_variables(self) -> None:
        """Updates the pressure on the boundary, as the primary variable for flow."""
        super().update_boundary_values_primary_variables()
        self.update_boundary_condition(
            name=self.pressure_variable, function=self.bc_values_pressure
        )


class InitialConditionsSinglePhaseFlow(pp.InitialConditionMixin):
    """Mixin for providing initial values for pressure."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`VariablesSinglePhaseFlow`."""

    interface_darcy_flux: Callable[
        [list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable
    ]
    """See :class:`VariablesSinglePhaseFlow`."""

    well_flux: Callable[[list[pp.MortarGrid]], pp.ad.MixedDimensionalVariable]
    """See :class:`VariablesSinglePhaseFlow`."""

    def initial_condition(self):
        """After the super-call, it sets initial values for the interface darcy flux
        and well flux.

        Note:
            Pressure is considered primary and the interface fluxes can in theory be
            constructed from it in simple cases. This makes them secondary variables
            indirectly but in practice they are treated as primary variables.

            Uses cases requiring a consistent initialization will benefit from the
            order here in the initialization routine.

        See also:

            - :meth:`ic_values_interface_darcy_flux`
            - :meth:`ic_values_well_flux`

        """
        # NOTE IMPORTANT: Super-call placed on top to ensure that variables considered
        # primary in the initialization (like pressure) are available before various
        # fluxes are initialized (see also set_initial_values_primary_variables, whose
        # super-call must be resolved first by the IC base mixin).
        super().initial_condition()

        for intf in self.mdg.interfaces():
            if intf.codim == 1:
                self.equation_system.set_variable_values(
                    self.ic_values_interface_darcy_flux(intf),
                    [cast(pp.ad.Variable, self.interface_darcy_flux([intf]))],
                    iterate_index=0,
                )

            if intf.codim == 2:
                self.equation_system.set_variable_values(
                    self.ic_values_well_flux(intf),
                    [cast(pp.ad.Variable, self.well_flux([intf]))],
                    iterate_index=0,
                )

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

    def set_initial_values_primary_variables(self) -> None:
        """Method to set initial values for pressure at iterate index 0.

        See also:

            - :meth:`ic_values_pressure`

        """
        # Super call for compatibility with multi-physics.
        super().set_initial_values_primary_variables()

        for sd in self.mdg.subdomains():
            # Need to cast the return value to variable, because it is typed as
            # operator.
            self.equation_system.set_variable_values(
                self.ic_values_pressure(sd),
                [cast(pp.ad.Variable, self.pressure([sd]))],
                iterate_index=0,
            )

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        """Method returning the initial pressure values for a given grid.

        Override this method to provide different initial conditions.

        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            The initial pressure values on that subdomain with
            ``shape=(sd.num_calles,)``. Defaults to zero array.

        """
        return np.zeros(sd.num_cells)

    def ic_values_interface_darcy_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        """Method returning the initial interface Darcy flux values on a given
        interface.

        Override this method to customize the initialization.

        Note:
            This method is only called for interfaces with codimension 1.

        Parameters:
            intf: A mortar grid in the md-grid.

        Returns:
            The initial interface Darcy flux values with
            ``shape=(interface.num_cells,)``. Defaults to zero array.

        """
        return np.zeros(intf.num_cells)

    def ic_values_well_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        """Method returning the initial well flux values on a given interface.

        Override this method to customize the initialization.

        Note:
            This method is only called for interfaces with codimension 2.

        Parameters:
            intf: A mortar grid in the md-grid.

        Returns:
            The initial interface well flux values with
            ``shape=(interface.num_cells,)``. Defaults to zero array.

        """
        return np.zeros(intf.num_cells)


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

    def create_variables(self) -> None:
        """Introduces the following variables into the system:

        1. Pressure variable on all subdomains.
        2. Darcy flux variable on all interfaces with codimension 1.
        3. Well flux variable on all interfaces with codimension 2.

        """
        super().create_variables()

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
            Operator representing the pressure [Pa].

        """
        if len(domains) > 0 and isinstance(domains[0], pp.BoundaryGrid):
            return self.create_boundary_operator(
                name=self.pressure_variable,
                domains=cast(Sequence[pp.BoundaryGrid], domains),
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

        Integrated over faces in the mortar grid.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the interface Darcy flux [kg * m^2 * s^-2].

        """
        flux = self.equation_system.md_variable(
            self.interface_darcy_flux_variable, interfaces
        )
        return flux

    def well_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Variable for the volumetric well flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the Darcy-like well flux [kg * m^2 * s^-2].

        """
        flux = self.equation_system.md_variable(self.well_flux_variable, interfaces)
        return flux


class SolutionStrategySinglePhaseFlow(pp.SolutionStrategy):
    """Solution strategy and numerics-related methods for a single-phase flow problem.

    At some point, this will be refined to be a more sophisticated (modularised)
    solution strategy class. More refactoring may be beneficial.

    This is *not* a full-scale model (in the old sense), but must be mixed with balance
    equations, constitutive laws etc. See user_examples.

    Parameters:
        params: Parameters for the solution strategy.

    """

    permeability: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the permeability of a subdomain. Normally provided by a
    mixin class with a suitable permeability definition.

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
    :class:`~porepy.models.fluid_property_library.FluidMobility`.

    """
    interface_mobility_discretization: Callable[
        [list[pp.MortarGrid]], pp.ad.UpwindCouplingAd
    ]
    """Discretization of the fluid mobility on internal boundaries. Normally provided
    by a mixin instance of :class:`~porepy.models.fluid_property_library.FluidMobility`.

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
                        sd, self.permeability([sd]), self.solid.permeability
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
            vals = self.equation_system.evaluate(self.darcy_flux([sd]))
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = self.equation_system.evaluate(self.interface_darcy_flux([intf]))
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            vals = self.equation_system.evaluate(self.well_flux([intf]))
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        super().before_nonlinear_iteration()

    def set_nonlinear_discretizations(self) -> None:
        """Collect discretizations for nonlinear terms."""
        super().set_nonlinear_discretizations()
        self.add_nonlinear_discretization(
            self.mobility_discretization(self.mdg.subdomains()).upwind(),
        )
        self.add_nonlinear_discretization(
            self.interface_mobility_discretization(self.mdg.interfaces()).flux(),
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
    FluidMassBalanceEquations,
    VariablesSinglePhaseFlow,
    ConstitutiveLawsSinglePhaseFlow,
    BoundaryConditionsSinglePhaseFlow,
    InitialConditionsSinglePhaseFlow,
    SolutionStrategySinglePhaseFlow,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Class for single-phase flow in mixed-dimensional porous media."""
