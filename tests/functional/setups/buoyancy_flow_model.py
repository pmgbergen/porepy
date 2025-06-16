"""
Module implementing compositional flow models for multi-phase, multi-component systems
with buoyancy effects.

Supports both:
- 2-phase, 2-component systems (e.g., water and methane)
- 3-phase, 3-component systems (e.g., water, oil, and methane)

The module defines geometry setup, equations of state, initial and boundary conditions,
and solution procedures for compositional fluid flow problems with gravitational effects.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, cast, Any
import numpy as np
import porepy as pp
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import CompositionalFractionalFlowTemplate as FlowTemplate
from abc import abstractmethod

# Constants for fluid phase densities (kg/m^3)
rho_w = 1000.0  #: Density of water (H2O)
rho_o = 700.0   #: Density of oil (C5H12)
rho_g = 200.0   #: Density of gas (CH4)

# Constants for fluid phase viscosities (Pa * second)
mu_w = 1.0e-3  #: Viscosity of water (H2O)
mu_o = 1.0e-4   #: Viscosity of oil (C5H12)
mu_g = 1.0e-5   #: Viscosity of gas (CH4)

# Conversion factor to Mega (1e-6)
to_Mega = 1.0e-6  #: Unit conversion factor to Mega units

class Geometry(pp.PorePyModel):
    """
    Abstract base class defining geometry.

    Subclasses must implement:
        - dirichlet_facets: Identifies Dirichlet boundary facets on the domain.
    """

    @abstractmethod
    def dirichlet_facets(
            self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray]:
        """
        Abstract method to select Dirichlet boundary facets.

        Args:
            sd (pp.Grid or pp.BoundaryGrid): Grid or boundary grid object
                on which to identify Dirichlet facets.

        Returns:
            tuple[np.ndarray]: Indices of facets where Dirichlet conditions apply.
        """
        pass

    @staticmethod
    def harvest_sphere_members(
            xc: np.ndarray,
            rc: float,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Select points inside a sphere defined by center and radius.

        Args:
            xc (np.ndarray): Coordinates of the sphere center.
            rc (float): Radius of the sphere.
            x (np.ndarray): Array of points to test.

        Returns:
            np.ndarray: Boolean mask array indicating points inside the sphere.
        """
        dx = x - xc
        r = np.linalg.norm(dx, axis=1)
        return np.where(r < rc, True, False)


class ModelGeometry2D(Geometry):
    """Concrete geometry class for 2D Cartesian grid domain."""

    _sphere_radius: float = 1.0
    _sphere_centre: np.ndarray = np.array([2.5, 5.0, 0.0])

    def set_domain(self) -> None:
        """
        Define a 2D squared domain.
        """
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")

        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._sphere_radius, face_centers[bf_indices]
            )
            return bf_indices[logical]

        return find_facets(self._sphere_centre)


class ModelGeometry3D(Geometry):
    """Concrete geometry class for 3D Cartesian grid domain."""

    _sphere_radius: float = 1.0
    _sphere_centre: np.ndarray = np.array([2.5, 2.5, 5.0])

    def set_domain(self) -> None:
        """
        Define a 3D cubic domain.
        """
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        z_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length, "zmax": z_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")

        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._sphere_radius, face_centers[bf_indices]
            )
            return bf_indices[logical]

        return find_facets(self._sphere_centre)


class BaseEOS(pp.compositional.EquationOfState):
    """
    Base class for simplified equations of state for fluid phases.

    Provides placeholder implementations of phase property functions
    including dynamic viscosity, enthalpy, thermal conductivity, and density,
    with methods to return values and derivatives for compositional flow.

    Subclasses should override:
        -rho_func to provide phase-specific density.
        -mu_func to provide phase-viscosity.
    """

    def h(
            self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Specific enthalpy function.

        Args:
            thermodynamic_dependencies: Variable number of arrays representing
                thermodynamic inputs.

        Returns:
            Tuple of enthalpy values and their derivatives w.r.t. inputs.
        """
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def kappa(
            self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Thermal conductivity function.

        Args:
            thermodynamic_dependencies: Variable number of arrays representing
                thermodynamic inputs.

        Returns:
            Tuple of conductivity values and their derivatives w.r.t. inputs.
        """
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def compute_phase_properties(
            self,
            phase_state: pp.compositional.PhysicalState,
            *thermodynamic_input: np.ndarray,
            params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> pp.compositional.PhaseProperties:
        """
        Compute phase properties given the current thermodynamic state.

        Args:
            phase_state (pp.compositional.PhysicalState): Physical state of the phase.
            thermodynamic_input: Arrays of thermodynamic variables.
            params: Optional parameters for computation.

        Returns:
            pp.compositional.PhaseProperties: Container with computed phase properties.
        """
        nc = len(thermodynamic_input[0])
        rho, drho = self.rho_func(*thermodynamic_input)  # mass density and derivatives
        h, dh = self.h(*thermodynamic_input)             # specific enthalpy and derivatives
        mu, dmu = self.mu_func(*thermodynamic_input)     # viscosity and derivatives
        kappa, dkappa = self.kappa(*thermodynamic_input) # thermal conductivity and derivatives

        return pp.compositional.PhaseProperties(
            state=phase_state,
            rho=rho,
            drho=drho,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=np.empty((2, nc)),
            dphis=np.empty((2, 3, nc)),
        )


class WaterEOS(BaseEOS):
    """
    Equation of State for the water phase.

    Implements constant density specific to water and inherits other properties
    from BaseEOS.
    """

    def rho_func(
            self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = rho_w * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def mu_func(
            self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = mu_w * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class OilEOS(BaseEOS):
    """
    Equation of State for the oil phase.

    Implements constant density specific to oil and inherits other properties
    from BaseEOS.
    """

    def rho_func(
            self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = rho_o * np.ones(nc)
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
            self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = mu_o * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

class GasEOS(BaseEOS):
    """
    Equation of State for the gas phase.

    Implements constant density specific to gas and inherits other properties
    from BaseEOS.
    """

    def rho_func(
            self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = rho_g * np.ones(nc)
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
            self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = mu_g * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

class BaseFlowModel(
    FlowTemplate,
):

    def __init__(self, params: dict):
        """Initializes the flow model."""
        super().__init__(params)
        self.expected_order_mass_loss = params.get("expected_order_mass_loss", 10)

    def relative_permeability(
            self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)**2

    def set_equations(self):
        super().set_equations()
        self.set_buoyancy_discretization_parameters()

    def set_nonlinear_discretizations(self) -> None:
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def before_nonlinear_iteration(self) -> None:
        self.update_buoyancy_driven_fluxes()
        self.rediscretize()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field

    def check_convergence(
            self,
            nonlinear_increment: np.ndarray,
            residual: Optional[np.ndarray],
            reference_residual: np.ndarray,
            nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:

        if self._is_nonlinear_problem():

            self.equation_system
            # nonlinear_increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )

            residual_norm = np.linalg.norm(residual)
            # Check convergence requiring both the increment and residual to be small.
            converged_inc = (
                    nl_params["nl_convergence_tol"] is np.inf
                    or nonlinear_increment_norm < nl_params["nl_convergence_tol"]
            )
            converged_res = (
                    nl_params["nl_convergence_tol_res"] is np.inf
                    or residual_norm < nl_params["nl_convergence_tol_res"]
            )
            converged = converged_inc and converged_res
            diverged = False
        else:
            raise ValueError("Gravitational segregation is nonlinear in its simpler form.")
        return converged, diverged

# constitutive description for N=2
def temperature_2N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])

    factor = 250.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs

def gas_saturation_2N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CH4 * rho_w) / (z_CH4 * rho_w + rho_g - z_CH4 * rho_g)
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = (rho_w * rho_g) / ((z_CH4 * (rho_w - rho_g) + rho_g) * (z_CH4 * (rho_w - rho_g) + rho_g))
    return vals, diffs

def CH4_water_2N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_gas_2N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


chi_functions_map_2N = {
    "CH4_water": CH4_water_2N,
    "CH4_gas": CH4_gas_2N,
}


# Two phases Two components case
class FluidMixture2N(pp.PorePyModel):

    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(["H2O", "CH4"], "chemicals")

    def get_phase_configuration(
            self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_W = WaterEOS(components)
        eos_G = GasEOS(components)
        configuration_W = (pp.compositional.PhysicalState.liquid, "water", eos_W)
        configuration_G = (pp.compositional.PhysicalState.gas, "gas", eos_G)
        return [configuration_W, configuration_G]

    def dependencies_of_phase_properties(
            self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z  # type:ignore[return-value]


class SecondaryEquations2N(LocalElimination):
    dependencies_of_phase_properties: Callable[
        ..., Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]
    ]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        # liquid phase is dependent
        rphase = self.fluid.reference_phase
        # gas phase is independent
        independent_phases = [p for p in self.fluid.phases if p != rphase]

        for phase in independent_phases:
            self.eliminate_locally(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdomains
                gas_saturation_2N,  # numerical function implementing correlation
                subdomains_and_matrix,  # all subdomains on which to eliminate s_gas
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid.phases:
            for comp in phase:
                check = self.has_independent_partial_fraction(comp, phase)
                if check:
                    self.eliminate_locally(
                        phase.partial_fraction_of[comp],
                        self.dependencies_of_phase_properties(phase),
                        chi_functions_map_2N[comp.name + "_" + phase.name],
                        subdomains_and_matrix,
                    )

        ### Provide constitutive law for temperature
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            temperature_2N,
            subdomains_and_matrix,
        )


# model description
class BoundaryConditions2N(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_top = 10.0e6 * to_Mega
        p = p_top * np.ones(boundary_grid.num_cells)
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h_inlet = 1.0
        h = h_inlet * np.ones(boundary_grid.num_cells)
        return h

    def bc_values_overall_fraction(
            self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        z_CH4 = np.zeros(boundary_grid.num_cells)
        return z_CH4


class InitialConditions2N(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_condition(self) -> None:
        super().initial_condition()

        # set the values to be the custom functions
        liq, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_gas_val = self.ic_values_staturation(sd)
            x_CH4_liq_v = np.zeros_like(s_gas_val)
            x_CH4_gas_v = np.ones_like(s_gas_val)

            x_CH4_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
            x_CH4_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            s_gas = gas.saturation([sd])
            self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
            self.equation_system.set_variable_values(x_CH4_liq_v, [x_CH4_liq], 0, 0)
            self.equation_system.set_variable_values(x_CH4_gas_v, [x_CH4_gas], 0, 0)

    def ic_values_staturation(self, sd: pp.Grid) -> np.ndarray:
        z_v = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        return (z_v * rho_w) / (z_v * rho_w + rho_g - z_v * rho_g)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 10.0e6 * to_Mega
        return np.ones(sd.num_cells) * p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 1.0
        return np.ones(sd.num_cells) * h

    def ic_values_overall_fraction(
            self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        xc = sd.cell_centers.T
        z = (np.where((xc[:, 1] >= 1.0) & (xc[:, 1] <= 2.0), 0.5, 0.0) +
             np.where((xc[:, 1] >= 3.0) & (xc[:, 1] <= 4.0), 0.5, 0.0) +
             np.where((xc[:, 0] >= 1.0) & (xc[:, 0] <= 2.0), 0.5, 0.0) +
             np.where((xc[:, 0] >= 3.0) & (xc[:, 0] <= 4.0), 0.5, 0.0))
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class FlowModel2N(
    BaseFlowModel,
):

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        sd = self.mdg.subdomains()[0]
        phases = list(self.fluid.phases)
        components = list(self.fluid.components)

        flux_buoyancy_c0 = self.component_buoyancy(components[0], self.mdg.subdomains())
        flux_buoyancy_c1 = self.component_buoyancy(components[1], self.mdg.subdomains())

        b_c0 = self.equation_system.evaluate(flux_buoyancy_c0)
        b_c1 = self.equation_system.evaluate(flux_buoyancy_c1)
        buoyancy_fluxes_are_reciprocal_Q = np.all(np.isclose(b_c0 + b_c1, 0.0))
        assert buoyancy_fluxes_are_reciprocal_Q

        ic_sg_val = self.ic_values_staturation(sd)
        ref_sg_integral = np.sum(sd.cell_volumes * ic_sg_val)

        s_gas = phases[1].saturation([sd])
        sg_val = self.equation_system.evaluate(s_gas)
        num_sg_integral = np.sum(sd.cell_volumes * sg_val)
        mass_loss = np.abs(ref_sg_integral - num_sg_integral)
        order_mass_loss = np.abs(np.floor(np.log10(mass_loss)))
        mass_conservative_Q = order_mass_loss >= self.expected_order_mass_loss
        assert mass_conservative_Q


class BuoyancyFlowModel2N(
    FluidMixture2N,
    InitialConditions2N,
    BoundaryConditions2N,
    SecondaryEquations2N,
    FlowModel2N,
):
    """A compositional flow model with buoyancy effects."""

    pass

# constitutive description for N=3
def temperature_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])

    factor = 250.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs

def oil_saturation_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_C5H12*rho_g*rho_w)/(-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -((z_C5H12*rho_g*rho_w*(-(rho_g*rho_o) + rho_g*rho_w))/
       (-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)**2) + (rho_g*rho_w)/(-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)
    diffs[3, :] = -((z_C5H12*rho_g*rho_w*(-(rho_g*rho_o) + rho_o*rho_w))/
     (-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)**2)
    return vals, diffs

def gas_saturation_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CH4*rho_o*rho_w)/(-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -((z_CH4*rho_o*rho_w*(-(rho_g*rho_o) + rho_g*rho_w))/
      (-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)**2)
    diffs[3, :] = -((z_CH4*rho_o*rho_w*(-(rho_g*rho_o) + rho_o*rho_w))/
       (-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)**2) + (rho_o*rho_w)/(-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)
    return vals, diffs

saturation_functions_map_3N = {
    "oil": oil_saturation_3N,
    "gas": gas_saturation_3N,
}

def C5H12_water_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))

def C5H12_oil_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))

def C5H12_gas_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))

def CH4_water_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))

def CH4_oil_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))

def CH4_gas_3N(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


chi_functions_map_3N = {
    "C5H12_water": C5H12_water_3N,
    "C5H12_oil": C5H12_oil_3N,
    "C5H12_gas": C5H12_gas_3N,
    "CH4_water": CH4_water_3N,
    "CH4_oil": CH4_oil_3N,
    "CH4_gas": CH4_gas_3N,
}

class FluidMixture3N(pp.PorePyModel):

    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(["H2O", "C5H12", "CH4"], "chemicals")

    def get_phase_configuration(
            self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_L = WaterEOS(components)
        eos_O = OilEOS(components)
        eos_G = GasEOS(components)
        configuration_W = (pp.compositional.PhysicalState.liquid, "water", eos_L)
        configuration_O = (pp.compositional.PhysicalState.liquid, "oil", eos_O)
        configuration_G = (pp.compositional.PhysicalState.gas, "gas", eos_G)
        return [configuration_W, configuration_O, configuration_G]

    def dependencies_of_phase_properties(
            self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z  # type:ignore[return-value]


class SecondaryEquations3N(LocalElimination):
    dependencies_of_phase_properties: Callable[
        ..., Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]
    ]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def set_equations(self) -> None:
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        # liquid phase is dependent
        rphase = self.fluid.reference_phase
        # gas phase is independent
        independent_phases = [p for p in self.fluid.phases if p != rphase]

        for phase in independent_phases:
            self.eliminate_locally(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdomains
                saturation_functions_map_3N[phase.name],  # numerical function implementing correlation
                subdomains_and_matrix,  # all subdomains on which to eliminate s_gas
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid.phases:
            for comp in phase:
                check = self.has_independent_partial_fraction(comp, phase)
                if check:
                    self.eliminate_locally(
                        phase.partial_fraction_of[comp],
                        self.dependencies_of_phase_properties(phase),
                        chi_functions_map_3N[comp.name + "_" + phase.name],
                        subdomains_and_matrix,
                    )

        ### Provide constitutive law for temperature
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            temperature_3N,
            subdomains_and_matrix,
        )


# model description
class BoundaryConditions3N(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    get_inlet_outlet_sides: Callable[
        [pp.Grid | pp.BoundaryGrid], tuple[np.ndarray, np.ndarray]
    ]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dirichlet_facets(sd), "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        p_top = 10.0e6 * to_Mega
        p = p_top * np.ones(boundary_grid.num_cells)
        return p

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h_inlet = 1.0
        h = h_inlet * np.ones(boundary_grid.num_cells)
        return h

    def bc_values_overall_fraction(
            self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        return np.zeros(boundary_grid.num_cells)


class InitialConditions3N(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_condition(self) -> None:
        super().initial_condition()

        # set the values to be the custom functions
        water, oil, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_oil_val = self.ic_values_staturation_oil(sd)
            s_gas_val = self.ic_values_staturation_gas(sd)
            self.equation_system.set_variable_values(s_oil_val, [oil.saturation([sd])], 0, 0)
            self.equation_system.set_variable_values(s_gas_val, [gas.saturation([sd])], 0, 0)

            x_inactive_v = np.zeros_like(s_oil_val)
            x_active_v = np.ones_like(s_gas_val)

            x_C5H12_water = water.partial_fraction_of[self.fluid.components[1]]([sd])
            x_C5H12_oil = oil.partial_fraction_of[self.fluid.components[1]]([sd])
            x_C5H12_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            x_CH4_water = water.partial_fraction_of[self.fluid.components[2]]([sd])
            x_CH4_oil = oil.partial_fraction_of[self.fluid.components[2]]([sd])
            x_CH4_gas = gas.partial_fraction_of[self.fluid.components[2]]([sd])

            self.equation_system.set_variable_values(x_inactive_v, [x_C5H12_water], 0, 0)
            self.equation_system.set_variable_values(x_active_v, [x_C5H12_oil], 0, 0)
            self.equation_system.set_variable_values(x_inactive_v, [x_C5H12_gas], 0, 0)

            self.equation_system.set_variable_values(x_inactive_v, [x_CH4_water], 0, 0)
            self.equation_system.set_variable_values(x_inactive_v, [x_CH4_oil], 0, 0)
            self.equation_system.set_variable_values(x_active_v, [x_CH4_gas], 0, 0)


    def ic_values_staturation_oil(self, sd: pp.Grid) -> np.ndarray:
        z_C5H12 = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        z_CH4 = self.ic_values_overall_fraction(self.fluid.components[2], sd)
        so_val = (z_C5H12*rho_g*rho_w)/(-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)
        return so_val

    def ic_values_staturation_gas(self, sd: pp.Grid) -> np.ndarray:
        z_C5H12 = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        z_CH4 = self.ic_values_overall_fraction(self.fluid.components[2], sd)
        sg_val = (z_CH4*rho_o*rho_w)/(-((-1 + z_C5H12 + z_CH4)*rho_g*rho_o) + z_C5H12*rho_g*rho_w + z_CH4*rho_o*rho_w)
        return sg_val

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 10.0e6 * to_Mega
        return np.ones(sd.num_cells) * p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 1.0
        return np.ones(sd.num_cells) * h

    def ic_values_overall_fraction(
            self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        xc = sd.cell_centers.T
        z = (np.where((xc[:, 1] >= 1.0) & (xc[:, 1] <= 2.0), 1/6.0, 0.0) +
             np.where((xc[:, 1] >= 3.0) & (xc[:, 1] <= 4.0), 1/6.0, 0.0) +
             np.where((xc[:, 0] >= 1.0) & (xc[:, 0] <= 2.0), 1/6.0, 0.0) +
             np.where((xc[:, 0] >= 3.0) & (xc[:, 0] <= 4.0), 1/6.0, 0.0))
        return z * np.ones(sd.num_cells)



class FlowModel3N(
    BaseFlowModel,
):

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        sd = self.mdg.subdomains()[0]
        phases = list(self.fluid.phases)
        components = list(self.fluid.components)

        flux_buoyancy_c0 = self.component_buoyancy(components[0], self.mdg.subdomains())
        flux_buoyancy_c1 = self.component_buoyancy(components[1], self.mdg.subdomains())
        flux_buoyancy_c2 = self.component_buoyancy(components[2], self.mdg.subdomains())

        b_c0 = self.equation_system.evaluate(flux_buoyancy_c0)
        b_c1 = self.equation_system.evaluate(flux_buoyancy_c1)
        b_c2 = self.equation_system.evaluate(flux_buoyancy_c2)
        buoyancy_fluxes_are_reciprocal_Q = np.all(np.isclose(b_c0 + b_c1 + b_c2, 0.0))
        assert buoyancy_fluxes_are_reciprocal_Q

        ic_so_val = self.ic_values_staturation_oil(sd)
        ic_sg_val = self.ic_values_staturation_gas(sd)
        ref_so_integral = np.sum(sd.cell_volumes * ic_so_val)
        ref_sg_integral = np.sum(sd.cell_volumes * ic_sg_val)

        s_oil = phases[1].saturation([sd])
        so_val = self.equation_system.evaluate(s_oil)
        num_so_integral = np.sum(sd.cell_volumes * so_val)
        oil_mass_loss = np.abs(ref_so_integral - num_so_integral)
        order_oil_mass_loss = np.abs(np.floor(np.log10(oil_mass_loss)))

        s_gas = phases[2].saturation([sd])
        sg_val = self.equation_system.evaluate(s_gas)
        num_sg_integral = np.sum(sd.cell_volumes * sg_val)
        gas_mass_loss = np.abs(ref_sg_integral - num_sg_integral)
        order_gas_mass_loss = np.abs(np.floor(np.log10(gas_mass_loss)))

        oil_mass_conservative_Q = order_oil_mass_loss >= self.expected_order_mass_loss
        gas_mass_conservative_Q = order_gas_mass_loss >= self.expected_order_mass_loss
        mass_conservative_Q = oil_mass_conservative_Q and gas_mass_conservative_Q
        assert mass_conservative_Q


class BuoyancyFlowModel3N(
    FluidMixture3N,
    InitialConditions3N,
    BoundaryConditions3N,
    SecondaryEquations3N,
    FlowModel3N,
):
    """A compositional flow model with buoyancy effects."""

    pass