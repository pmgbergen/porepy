"""
Module implementing compositional flow models for multi-phase, multi-component systems
with buoyancy effects.

Supports both:
- 2-phase, 2-component systems (e.g., water and methane)
- 3-phase, 3-component systems (e.g., water, oil, and methane)

The module defines 2D and 3D geometry setup, equations of state, initial and boundary
conditions, and solution procedures for compositional fluid flow problems with
gravitational effects.

Fixed- and mixed-dimensional meshes are defined on cartesian grids.

"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, cast, Any
import numpy as np
import porepy as pp
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import (
    CompositionalFractionalFlowTemplate as FlowTemplate,
)
from abc import abstractmethod

# Constants for fluid phase densities (kg/m^3)
rho_w = 1000.0  #: Density of water (H2O)
rho_o = 700.0  #: Density of oil (C5H12)
rho_g = 200.0  #: Density of gas (CH4)

# Constants for fluid phase viscosities (Pa * second)
mu_w = 1.0e-3  #: Viscosity of water (H2O)
mu_o = 1.0e-4  #: Viscosity of oil (C5H12)
mu_g = 1.0e-5  #: Viscosity of gas (CH4)

# Specific enthalpies (physical units MJ/kg)
h_w = 1.0  # Water
h_o = 1.5  # Oil
h_g = 2.0  # Gas

# Conversion factor to Mega (1e-6)
to_Mega = 1.0e-6  #: Unit conversion factor to Mega units


class Geometry(pp.PorePyModel):
    """
    Abstract base class defining geometry.

    Subclasses must implement:
        - dirichlet_facets: Identifies Dirichlet boundary facets on the domain.
    """

    @abstractmethod
    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> tuple[np.ndarray]:
        """Return Dirichlet facet indices."""
        pass

    @staticmethod
    def harvest_sphere_members(xc: np.ndarray, rc: float, x: np.ndarray) -> np.ndarray:
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
    """2D Cartesian domain."""

    _sphere_radius: float = 1.0
    _sphere_center: np.ndarray = np.array(
        [2.5, 5.0, 0.0]
    )  # renamed from _sphere_centre

    def set_domain(self) -> None:
        """Set square domain."""
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

        return find_facets(self._sphere_center)


class ModelMDGeometry2D(ModelGeometry2D):
    """2D mixed-dimensional domain."""

    def set_fractures(self) -> None:
        points = np.array(
            [
                [1.0, 2.0],
                [4.0, 2.0],
                [1.0, 2.0],
                [1.0, 4.0],
                [4.0, 2.0],
                [4.0, 4.0],
                [2.0, 1.0],
                [2.0, 4.0],
                [3.0, 1.0],
                [3.0, 4.0],
            ]
        ).T
        fracs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]).T
        self._fractures = pp.frac_utils.pts_edges_to_linefractures(points, fracs)


class ModelGeometry3D(Geometry):
    """3D Cartesian domain."""

    _sphere_radius: float = 1.0
    _sphere_center: np.ndarray = np.array(
        [2.5, 2.5, 5.0]
    )  # renamed from _sphere_centre

    def set_domain(self) -> None:
        """Set a 3D cubic domain."""
        x_length = self.units.convert_units(5.0, "m")
        y_length = self.units.convert_units(5.0, "m")
        z_length = self.units.convert_units(5.0, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length,
            "ymax": y_length,
            "zmax": z_length,
        }
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

        return find_facets(self._sphere_center)


class ModelMDGeometry3D(ModelGeometry3D):
    """3D mixed-dimensional domain."""

    def set_fractures(self) -> None:
        kind_1_square_u = np.array([1.0, 1.0, 4.0, 4.0])
        kind_1_square_v = np.array([1.0, 4.0, 4.0, 1.0])

        kind_2_square_u = np.array([2.0, 2.0, 4.0, 4.0])
        kind_2_square_v = np.array([2.0, 4.0, 4.0, 2.0])

        # normal along z from z = 2.0
        f1 = np.vstack([kind_1_square_u, kind_1_square_v, np.full(4, 2.0)])

        # normal along y from y = 1.0
        f2 = np.vstack([kind_1_square_u, np.full(4, 1.0), kind_1_square_v])

        # normal along y from y = 4.0
        f3 = np.vstack([kind_1_square_u, np.full(4, 4.0), kind_1_square_v])

        # normal along y from y = 3.0
        f4 = np.vstack([kind_1_square_u, np.full(4, 3.0), kind_1_square_v])

        # normal along x from x = 2.0
        f5 = np.vstack([np.full(4, 2.0), kind_2_square_u, kind_2_square_v])

        disjoint_set = [f1, f2, f3, f4, f5]
        self._fractures = [pp.PlaneFracture(p) for p in disjoint_set]


class BaseEOS(pp.compositional.EquationOfState):
    """Simple constant-property EOS base.

    Provides placeholder implementations of phase property functions
    including dynamic viscosity, enthalpy, thermal conductivity, and density,
    with methods to return values and derivatives for compositional flow.

    Subclasses should override:
        -h_func to provide phase-enthalpy.
        -rho_func to provide phase-specific density.
        -mu_func to provide phase-viscosity.
    """

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Thermal conductivity function.

        Args:
            thermodynamic_dependencies: Variable number of arrays representing
                thermodynamic inputs.
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
        """Compile phase properties as a pp.compositional.PhaseProperties."""

        nc = len(thermodynamic_input[0])
        rho, drho = self.rho_func(*thermodynamic_input)  # mass density and derivatives
        h, dh = self.h(*thermodynamic_input)  # specific enthalpy and derivatives
        mu, dmu = self.mu_func(*thermodynamic_input)  # viscosity and derivatives
        kappa, dkappa = self.kappa(
            *thermodynamic_input
        )  # thermal conductivity and derivatives

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
    """Water EOS (constant props)."""

    def h(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_w * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def rho_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_w * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def mu_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = mu_w * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class OilEOS(BaseEOS):
    """Oil EOS (constant props)."""

    def h(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_o * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def rho_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_o * np.ones(nc)
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = mu_o * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class GasEOS(BaseEOS):
    """Gas EOS (constant props)."""

    def h(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = h_g * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def rho_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_g * np.ones(nc)
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def mu_func(
        self, *thermodynamic_dependencies: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = mu_g * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class BoundaryConditions(pp.PorePyModel):
    """Boundary conditions."""

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


class SecondaryEquations(LocalElimination):
    """Base class for Secondary relations (2N or 3N)."""

    dependencies_of_phase_properties: Callable[
        ..., Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]
    ]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def __init__(
        self,
        *args,
        saturation_functions_map: dict[str, Callable],
        chi_functions_map: dict[str, Callable],
        temperature_function: Callable,
        **kwargs,
    ):
        # Pass positional + keyword args upstream
        super().__init__(*args, **kwargs)

        self._saturation_functions_map = saturation_functions_map
        self._chi_functions_map = chi_functions_map
        self._temperature_function = temperature_function

    def set_equations(self) -> None:
        """Register eliminations."""
        super().set_equations()
        subdomains = self.mdg.subdomains()

        matrix = self.mdg.subdomains(dim=self.mdg.dim_max())[0]
        matrix_boundary = cast(
            pp.BoundaryGrid, self.mdg.subdomain_to_boundary_grid(matrix)
        )
        subdomains_and_matrix = subdomains + [matrix_boundary]

        # liquid phase is dependent
        rphase = self.fluid.reference_phase
        # other phases independent
        independent_phases = [p for p in self.fluid.phases if p != rphase]

        # Saturation eliminations
        for phase in independent_phases:
            if phase.name in self._saturation_functions_map:
                self.eliminate_locally(
                    phase.saturation,
                    self.dependencies_of_phase_properties(phase),
                    self._saturation_functions_map[phase.name],
                    subdomains_and_matrix,
                )

        # Partial fractions eliminations
        for phase in self.fluid.phases:
            for comp in phase:
                if self.has_independent_partial_fraction(comp, phase):
                    key = f"{comp.name}_{phase.name}"
                    if key in self._chi_functions_map:
                        self.eliminate_locally(
                            phase.partial_fraction_of[comp],
                            self.dependencies_of_phase_properties(phase),
                            self._chi_functions_map[key],
                            subdomains_and_matrix,
                        )

        # Temperature elimination
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),
            self._temperature_function,
            subdomains_and_matrix,
        )


class BaseFlowModel(
    FlowTemplate,
):
    def __init__(self, params: dict):
        """Initialize flow model."""
        super().__init__(params)
        self.expected_order_loss = params.get("expected_order_loss", 10)

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """kr = saturation."""
        return phase.saturation(domains)

    def set_equations(self):
        """Set equations + buoyancy params."""
        super().set_equations()
        self.set_buoyancy_discretization_parameters()

    def set_nonlinear_discretizations(self) -> None:
        """Register nonlinear discretizations."""
        super().set_nonlinear_discretizations()
        self.set_nonlinear_buoyancy_discretization()

    def before_nonlinear_iteration(self) -> None:
        """Update buoyancy fluxes."""
        self.update_buoyancy_driven_fluxes()
        self.rediscretize()

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Gravity magnitude field."""
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
        """Check nonlinear convergence."""

        if self._is_nonlinear_problem():
            total_volume = 0.0
            for sd in self.mdg.subdomains():
                total_volume += np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
                    )
                )

            # nonlinear_increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )

            residual_norm = np.linalg.norm(residual) * total_volume
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
            raise ValueError("Gravitational segregation is always nonlinear.")
        return converged, diverged


# constitutive description for N=2
def temperature_2N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Temperature correlation (zeroed)."""
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])

    # Set temperature to zero to isolate
    # the effect of energy convection driven by buoyancy.
    factor = 0.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def gas_saturation_2N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gas saturation correlation."""
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CH4 * rho_w) / (z_CH4 * rho_w + rho_g - z_CH4 * rho_g)
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = (rho_w * rho_g) / (
        (z_CH4 * (rho_w - rho_g) + rho_g) * (z_CH4 * (rho_w - rho_g) + rho_g)
    )
    return vals, diffs


def CH4_water_2N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in water fraction."""
    p, h, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_gas_2N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in gas fraction."""
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
    """2-phase (water-gas), 2-component mixture."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        component_1 = pp.FluidComponent(name="H2O")
        component_2 = pp.FluidComponent(name="CH4")
        return [component_1, component_2]

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


class SecondaryEquations2N(SecondaryEquations):
    """Secondary (eliminated) relations 2N."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            saturation_functions_map={"gas": gas_saturation_2N},
            chi_functions_map=chi_functions_map_2N,
            temperature_function=temperature_2N,
            *args,
            **kwargs,
        )


class InitialConditions2N(pp.PorePyModel):
    """Initial conditions 2N."""

    def initial_condition(self) -> None:
        """Set initial fields."""
        super().initial_condition()

        # set the values to be the custom functions
        liq, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_gas_val = self.ic_values_saturation(sd)
            x_CH4_liq_v = np.zeros_like(s_gas_val)
            x_CH4_gas_v = np.ones_like(s_gas_val)

            x_CH4_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
            x_CH4_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            s_gas = gas.saturation([sd])
            self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
            self.equation_system.set_variable_values(x_CH4_liq_v, [x_CH4_liq], 0, 0)
            self.equation_system.set_variable_values(x_CH4_gas_v, [x_CH4_gas], 0, 0)

    def ic_values_saturation(self, sd: pp.Grid) -> np.ndarray:
        z_v = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        return (z_v * rho_w) / (z_v * rho_w + rho_g - z_v * rho_g)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 10.0e6 * to_Mega
        return np.ones(sd.num_cells) * p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        ic_s = self.ic_values_saturation(sd)
        ic_rho = rho_g * ic_s + rho_w * (1.0 - ic_s)
        h = (ic_s * h_g * rho_g + (1.0 - ic_s) * h_w * rho_w) / ic_rho
        return np.ones(sd.num_cells) * h

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        xc = sd.cell_centers.T
        z = (
            np.where((xc[:, 1] >= 1.0) & (xc[:, 1] <= 2.0), 0.5, 0.0)
            + np.where((xc[:, 1] >= 3.0) & (xc[:, 1] <= 4.0), 0.5, 0.0)
            + np.where((xc[:, 0] >= 1.0) & (xc[:, 0] <= 2.0), 0.5, 0.0)
            + np.where((xc[:, 0] >= 3.0) & (xc[:, 0] <= 4.0), 0.5, 0.0)
        )
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class FlowModel2N(
    BaseFlowModel,
):
    def after_nonlinear_convergence(self) -> None:
        """Post-convergence diagnostics."""
        super().after_nonlinear_convergence()

        subdomains = self.mdg.subdomains()
        phases = list(self.fluid.phases)
        components = list(self.fluid.components)

        # Buoyancy flux reciprocity
        buoy_ops = [
            self.component_buoyancy(comp, subdomains) for comp in components[:2]
        ]
        buoy_vals = [self.equation_system.evaluate(op) for op in buoy_ops]
        assert np.all(np.isclose(sum(buoy_vals), 0.0))

        # Total volume
        total_volume = sum(
            np.sum(
                self.equation_system.evaluate(
                    self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
                )
            )
            for sd in subdomains
        )

        def norm_vol_int(op: pp.ad.Operator, sd: pp.Grid) -> float:
            return (
                np.sum(
                    self.equation_system.evaluate(self.volume_integral(op, [sd], dim=1))
                )
                / total_volume
            )

        # Reference and numerical accumulators
        ref_rho = ref_rho_z = ref_energy = 0.0
        num_rho = num_rho_z = num_energy = 0.0

        for sd in subdomains:
            ic_sg = self.ic_values_saturation(sd)
            rho_l = phases[0].density([sd])
            rho_g = phases[1].density([sd])

            ic_rho = (
                pp.wrap_as_dense_ad_array(1.0 - ic_sg) * rho_l
                + pp.wrap_as_dense_ad_array(ic_sg) * rho_g
            )
            ref_rho += norm_vol_int(ic_rho, sd)

            ic_z = self.ic_values_overall_fraction(components[1], sd)
            ic_rho_z = ic_rho * pp.wrap_as_dense_ad_array(ic_z)
            ref_rho_z += norm_vol_int(ic_rho_z, sd)

            ic_p = self.ic_values_pressure(sd)
            ic_h = self.ic_values_enthalpy(sd)
            ic_energy = ic_rho * pp.wrap_as_dense_ad_array(
                ic_h
            ) - pp.wrap_as_dense_ad_array(ic_p)
            ref_energy += norm_vol_int(ic_energy, sd)

            cur_rho = self.fluid.density([sd])
            num_rho += norm_vol_int(cur_rho, sd)

            cur_rho_z = cur_rho * components[1].fraction([sd])
            num_rho_z += norm_vol_int(cur_rho_z, sd)

            cur_energy = cur_rho * self.enthalpy([sd]) - self.pressure([sd])
            num_energy += norm_vol_int(cur_energy, sd)

        # Loss metrics
        def order(loss: float) -> float:
            return np.inf if loss <= 0.0 else abs(np.floor(np.log10(loss)))

        mass_loss = abs(ref_rho - num_rho)
        z_mass_loss = abs(ref_rho_z - num_rho_z)
        energy_loss = abs(ref_energy - num_energy)

        assert order(mass_loss) >= self.expected_order_loss
        assert order(z_mass_loss) >= self.expected_order_loss
        assert order(energy_loss) >= self.expected_order_loss


class BuoyancyFlowModel2N(
    FluidMixture2N,
    InitialConditions2N,
    BoundaryConditions,
    SecondaryEquations2N,
    FlowModel2N,
):
    """Complete 2N buoyancy model."""

    pass


# constitutive description for N=3
def temperature_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Temperature correlation (zeroed)."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])

    # Set temperature to zero to isolate
    # the effect of energy convection driven by buoyancy.
    factor = 0.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def oil_saturation_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Oil saturation correlation."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_C5H12 * rho_g * rho_w) / (
        -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
        + z_C5H12 * rho_g * rho_w
        + z_CH4 * rho_o * rho_w
    )
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -(
        (z_C5H12 * rho_g * rho_w * (-(rho_g * rho_o) + rho_g * rho_w))
        / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        ** 2
    ) + (rho_g * rho_w) / (
        -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
        + z_C5H12 * rho_g * rho_w
        + z_CH4 * rho_o * rho_w
    )
    diffs[3, :] = -(
        (z_C5H12 * rho_g * rho_w * (-(rho_g * rho_o) + rho_o * rho_w))
        / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        ** 2
    )
    return vals, diffs


def gas_saturation_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gas saturation correlation."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CH4 * rho_o * rho_w) / (
        -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
        + z_C5H12 * rho_g * rho_w
        + z_CH4 * rho_o * rho_w
    )
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -(
        (z_CH4 * rho_o * rho_w * (-(rho_g * rho_o) + rho_g * rho_w))
        / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        ** 2
    )
    diffs[3, :] = -(
        (z_CH4 * rho_o * rho_w * (-(rho_g * rho_o) + rho_o * rho_w))
        / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        ** 2
    ) + (rho_o * rho_w) / (
        -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
        + z_C5H12 * rho_g * rho_w
        + z_CH4 * rho_o * rho_w
    )
    return vals, diffs


saturation_functions_map_3N = {
    "oil": oil_saturation_3N,
    "gas": gas_saturation_3N,
}


def C5H12_water_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """C5H12 in water."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def C5H12_oil_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """C5H12 in oil."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def C5H12_gas_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """C5H12 in gas."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_water_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in water."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_oil_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in oil."""
    p, h, z_C5H12, z_CH4 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CH4)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CH4)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CH4_gas_3N(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CH4 in gas."""
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
    """3-phase (water-oil-gas), 3-component mixture."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        component_1 = pp.FluidComponent(name="H2O")
        component_2 = pp.FluidComponent(name="C5H12")
        component_3 = pp.FluidComponent(name="CH4")
        return [component_1, component_2, component_3]

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


class SecondaryEquations3N(SecondaryEquations):
    """Secondary relations 3N."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            saturation_functions_map=saturation_functions_map_3N,
            chi_functions_map=chi_functions_map_3N,
            temperature_function=temperature_3N,
            *args,
            **kwargs,
        )


class InitialConditions3N(pp.PorePyModel):
    """Initial conditions 3N."""

    def initial_condition(self) -> None:
        """Set initial fields."""
        super().initial_condition()

        # set the values to be the custom functions
        water, oil, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_oil_val = self.ic_values_saturation_oil(sd)
            s_gas_val = self.ic_values_saturation_gas(sd)
            self.equation_system.set_variable_values(
                s_oil_val, [oil.saturation([sd])], 0, 0
            )
            self.equation_system.set_variable_values(
                s_gas_val, [gas.saturation([sd])], 0, 0
            )

            x_inactive_v = np.zeros_like(s_oil_val)
            x_active_v = np.ones_like(s_gas_val)

            x_C5H12_water = water.partial_fraction_of[self.fluid.components[1]]([sd])
            x_C5H12_oil = oil.partial_fraction_of[self.fluid.components[1]]([sd])
            x_C5H12_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            x_CH4_water = water.partial_fraction_of[self.fluid.components[2]]([sd])
            x_CH4_oil = oil.partial_fraction_of[self.fluid.components[2]]([sd])
            x_CH4_gas = gas.partial_fraction_of[self.fluid.components[2]]([sd])

            self.equation_system.set_variable_values(
                x_inactive_v, [x_C5H12_water], 0, 0
            )
            self.equation_system.set_variable_values(x_active_v, [x_C5H12_oil], 0, 0)
            self.equation_system.set_variable_values(x_inactive_v, [x_C5H12_gas], 0, 0)

            self.equation_system.set_variable_values(x_inactive_v, [x_CH4_water], 0, 0)
            self.equation_system.set_variable_values(x_inactive_v, [x_CH4_oil], 0, 0)
            self.equation_system.set_variable_values(x_active_v, [x_CH4_gas], 0, 0)

    def ic_values_saturation_oil(self, sd: pp.Grid) -> np.ndarray:
        z_C5H12 = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        z_CH4 = self.ic_values_overall_fraction(self.fluid.components[2], sd)
        so_val = (z_C5H12 * rho_g * rho_w) / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
        return so_val

    def ic_values_saturation_gas(self, sd: pp.Grid) -> np.ndarray:
        z_C5H12 = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        z_CH4 = self.ic_values_overall_fraction(self.fluid.components[2], sd)
        sg_val = (z_CH4 * rho_o * rho_w) / (
            -((-1 + z_C5H12 + z_CH4) * rho_g * rho_o)
            + z_C5H12 * rho_g * rho_w
            + z_CH4 * rho_o * rho_w
        )
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
        z = (
            np.where((xc[:, 1] >= 1.0) & (xc[:, 1] <= 2.0), 1 / 6.0, 0.0)
            + np.where((xc[:, 1] >= 3.0) & (xc[:, 1] <= 4.0), 1 / 6.0, 0.0)
            + np.where((xc[:, 0] >= 1.0) & (xc[:, 0] <= 2.0), 1 / 6.0, 0.0)
            + np.where((xc[:, 0] >= 3.0) & (xc[:, 0] <= 4.0), 1 / 6.0, 0.0)
        )
        return z * np.ones(sd.num_cells)


class FlowModel3N(
    BaseFlowModel,
):
    def after_nonlinear_convergence(self) -> None:
        """Post-convergence diagnostics."""
        super().after_nonlinear_convergence()

        phases = list(self.fluid.phases)  # water, oil, gas
        components = list(self.fluid.components)  # H2O (ref), C5H12, CH4

        # Buoyancy flux reciprocity (sum over components zero)
        flux_buoyancy_c0 = self.component_buoyancy(components[0], self.mdg.subdomains())
        flux_buoyancy_c1 = self.component_buoyancy(components[1], self.mdg.subdomains())
        flux_buoyancy_c2 = self.component_buoyancy(components[2], self.mdg.subdomains())

        b_c0 = self.equation_system.evaluate(flux_buoyancy_c0)
        b_c1 = self.equation_system.evaluate(flux_buoyancy_c1)
        b_c2 = self.equation_system.evaluate(flux_buoyancy_c2)
        buoyancy_fluxes_are_reciprocal_Q = np.all(np.isclose(b_c0 + b_c1 + b_c2, 0.0))
        assert buoyancy_fluxes_are_reciprocal_Q

        # Total volume for normalization
        total_volume = 0.0
        for sd in self.mdg.subdomains():
            vol_op = self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)
            total_volume += np.sum(self.equation_system.evaluate(vol_op))

        # Reference (initial) and numerical integrals
        ref_rho_integral = 0.0
        num_rho_integral = 0.0

        ref_rho_c1_integral = 0.0
        ref_rho_c2_integral = 0.0
        num_rho_c1_integral = 0.0
        num_rho_c2_integral = 0.0

        ref_energy_integral = 0.0
        num_energy_integral = 0.0

        # Loop subdomains
        for sd in self.mdg.subdomains():
            # Initial saturations
            ic_so = self.ic_values_saturation_oil(sd)
            ic_sg = self.ic_values_saturation_gas(sd)
            ic_sw = 1.0 - ic_so - ic_sg

            # Phase densities (AD operators)
            rho_w = phases[0].density([sd])
            rho_o = phases[1].density([sd])
            rho_g = phases[2].density([sd])

            # Initial mixture density (AD)
            ic_rho = (
                pp.wrap_as_dense_ad_array(ic_sw) * rho_w
                + pp.wrap_as_dense_ad_array(ic_so) * rho_o
                + pp.wrap_as_dense_ad_array(ic_sg) * rho_g
            )

            # Initial overall fractions for non-reference components
            ic_z_c1 = self.ic_values_overall_fraction(components[1], sd)  # C5H12
            ic_z_c2 = self.ic_values_overall_fraction(components[2], sd)  # CH4

            # Reference mass integrals (normalized)
            ref_rho_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(ic_rho, [sd], dim=1)
                    )
                )
                / total_volume
            )

            ref_rho_c1_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(
                            ic_rho * pp.wrap_as_dense_ad_array(ic_z_c1), [sd], dim=1
                        )
                    )
                )
                / total_volume
            )
            ref_rho_c2_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(
                            ic_rho * pp.wrap_as_dense_ad_array(ic_z_c2), [sd], dim=1
                        )
                    )
                )
                / total_volume
            )

            # Initial energy (rho*h - p)
            ic_p = self.ic_values_pressure(sd)
            ic_h = self.ic_values_enthalpy(sd)
            ic_energy = ic_rho * pp.wrap_as_dense_ad_array(
                ic_h
            ) - pp.wrap_as_dense_ad_array(ic_p)
            ref_energy_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(ic_energy, [sd], dim=1)
                    )
                )
                / total_volume
            )

            # Current mixture density and integrals
            num_rho = self.fluid.density([sd])
            num_rho_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(num_rho, [sd], dim=1)
                    )
                )
                / total_volume
            )

            num_rho_c1 = num_rho * components[1].fraction([sd])
            num_rho_c2 = num_rho * components[2].fraction([sd])
            num_rho_c1_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(num_rho_c1, [sd], dim=1)
                    )
                )
                / total_volume
            )
            num_rho_c2_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(num_rho_c2, [sd], dim=1)
                    )
                )
                / total_volume
            )

            num_energy = num_rho * self.enthalpy([sd]) - self.pressure([sd])
            num_energy_integral += (
                np.sum(
                    self.equation_system.evaluate(
                        self.volume_integral(num_energy, [sd], dim=1)
                    )
                )
                / total_volume
            )

        # Loss metrics (orders)
        total_mass_loss = abs(ref_rho_integral - num_rho_integral)
        c1_mass_loss = abs(ref_rho_c1_integral - num_rho_c1_integral)
        c2_mass_loss = abs(ref_rho_c2_integral - num_rho_c2_integral)
        energy_loss = abs(ref_energy_integral - num_energy_integral)

        order_total_mass = abs(np.floor(np.log10(total_mass_loss)))
        order_c1_mass = abs(np.floor(np.log10(c1_mass_loss)))
        order_c2_mass = abs(np.floor(np.log10(c2_mass_loss)))
        order_energy = abs(np.floor(np.log10(energy_loss)))

        # Assertions
        assert order_total_mass >= self.expected_order_loss
        assert order_c1_mass >= self.expected_order_loss
        assert order_c2_mass >= self.expected_order_loss
        assert order_energy >= self.expected_order_loss


class BuoyancyFlowModel3N(
    FluidMixture3N,
    InitialConditions3N,
    BoundaryConditions,
    SecondaryEquations3N,
    FlowModel3N,
):
    """Complete 3N buoyancy model."""

    pass
