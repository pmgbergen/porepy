from __future__ import annotations

from typing import Callable, Optional, Sequence, cast, Any
import numpy as np
import porepy as pp
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import (
    CompositionalFractionalFlowTemplate as FlowTemplate,
)
from abc import abstractmethod

mesh_2d_Q = True

# define constant phase densities
rho_l = 1000.0
rho_g = 700.0
to_Mega = 1.0e-6


# geometry description
class Geometry(pp.PorePyModel):
    @abstractmethod
    def dirichlet_facets(self, sd: pp.Grid | pp.BoundaryGrid) -> tuple[np.ndarray]:
        pass

    @staticmethod
    def harvest_sphere_members(xc, rc, x):
        dx = x - xc
        r = np.linalg.norm(dx, axis=1)
        return np.where(r < rc, True, False)


class ModelGeometry(Geometry):
    _sphere_radius: float = 0.0078125
    _sphere_centre: np.ndarray = np.array([0.00390625, 10.0, 0.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(0.0078125, "m")
        y_length = self.units.convert_units(10.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.0078125, "m")
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
    _sphere_radius: float = 0.25
    _sphere_centre: np.ndarray = np.array([0.125, 0.125, 10.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(0.25, "m")
        y_length = self.units.convert_units(0.25, "m")
        z_length = self.units.convert_units(10.0, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length,
            "ymax": y_length,
            "zmax": z_length,
        }
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.25, "m")
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


# constitutive description
def gas_saturation_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = (z_CO2 * rho_l) / (z_CO2 * rho_l + rho_g - z_CO2 * rho_g)
    vals = np.clip(vals, 1.0e-16, 1.0)

    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = (rho_l * rho_g) / (
        (z_CO2 * (rho_l - rho_g) + rho_g) * (z_CO2 * (rho_l - rho_g) + rho_g)
    )
    return vals, diffs


def temperature_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])

    factor = 0.0
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0 * factor
    return vals, diffs


def CO2_liq_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


def CO2_gas_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_CO2 = thermodynamic_dependencies
    assert len(p) == len(h) == len(z_CO2)

    nc = len(thermodynamic_dependencies[0])
    vals = np.ones_like(z_CO2)
    vals = np.clip(vals, 1.0e-16, 1.0)
    return vals, np.zeros((len(thermodynamic_dependencies), nc))


chi_functions_map = {
    "CO2_liq": CO2_liq_func,
    "CO2_gas": CO2_gas_func,
}


class LiquidEOS(pp.compositional.EquationOfState):
    def rho_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_l * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def mu_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (1.0e-3) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (1.0) * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def kappa(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc) * to_Mega
        return vals, np.zeros((len(thermodynamic_dependencies), nc))

    def compute_phase_properties(
        self,
        phase_state: pp.compositional.PhysicalState,
        *thermodynamic_input: np.ndarray,
        params: Optional[Sequence[np.ndarray | float]] = None,
    ) -> pp.compositional.PhaseProperties:
        nc = len(thermodynamic_input[0])
        # mass density of phase
        rho, drho = self.rho_func(*thermodynamic_input)  # (n,), (3, n) array
        # specific enthalpy of phase
        h, dh = self.h(*thermodynamic_input)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = self.mu_func(*thermodynamic_input)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

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


class GasEOS(LiquidEOS):
    def rho_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = rho_g * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nc = len(thermodynamic_dependencies[0])
        vals = (2.0) * np.ones(nc)
        return vals, np.zeros((len(thermodynamic_dependencies), nc))


class FluidMixture(pp.PorePyModel):
    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(["H2O", "CO2"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.Component]
    ) -> Sequence[
        tuple[pp.compositional.EquationOfState, pp.compositional.PhysicalState, str]
    ]:
        eos_L = LiquidEOS(components)
        eos_G = GasEOS(components)
        configuration_L = (pp.compositional.PhysicalState.liquid, "liq", eos_L)
        configuration_G = (pp.compositional.PhysicalState.gas, "gas", eos_G)
        return [configuration_L, configuration_G]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z_CO2 = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]
        return [self.pressure, self.enthalpy] + z_CO2  # type:ignore[return-value]


class SecondaryEquations(LocalElimination):
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
                ),  # callables giving primary variables on subdoains
                gas_saturation_func,  # numerical function implementing correlation
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
                        chi_functions_map[comp.name + "_" + phase.name],
                        subdomains_and_matrix,
                    )

        ### Provide constitutive law for temperature
        self.eliminate_locally(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            temperature_func,
            subdomains_and_matrix,
        )


# model description
class BoundaryConditions(pp.PorePyModel):
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
        h = np.ones(boundary_grid.num_cells)
        return h

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        z_CO2 = np.zeros(boundary_grid.num_cells)
        return z_CO2

    def bc_values_fractional_flow_energy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return self.bc_values_enthalpy(bg)

class InitialConditions(pp.PorePyModel):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def initial_condition(self) -> None:
        super().initial_condition()

        # set the values to be the custom functions
        liq, gas = self.fluid.phases
        for sd in self.mdg.subdomains():
            s_gas_val = self.ic_values_staturation(sd)
            x_CO2_liq_v = np.zeros_like(s_gas_val)
            x_CO2_gas_v = np.ones_like(s_gas_val)

            x_CO2_liq = liq.partial_fraction_of[self.fluid.components[1]]([sd])
            x_CO2_gas = gas.partial_fraction_of[self.fluid.components[1]]([sd])

            s_gas = gas.saturation([sd])
            self.equation_system.set_variable_values(s_gas_val, [s_gas], 0, 0)
            self.equation_system.set_variable_values(x_CO2_liq_v, [x_CO2_liq], 0, 0)
            self.equation_system.set_variable_values(x_CO2_gas_v, [x_CO2_gas], 0, 0)

    def ic_values_staturation(self, sd: pp.Grid) -> np.ndarray:
        z_v = self.ic_values_overall_fraction(self.fluid.components[1], sd)
        return (z_v * rho_l) / (z_v * rho_l + rho_g - z_v * rho_g)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        p_init = 10.0e6 * to_Mega
        return np.ones(sd.num_cells) * p_init

    def ic_values_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 1.0 + self.ic_values_overall_fraction(self.fluid.components[1], sd)
        return h

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        s_init_val = 0.8
        z_init_val = (rho_g*s_init_val)/(rho_l*(1 - s_init_val) + rho_g*(s_init_val))
        xc = sd.cell_centers.T
        z = np.where((xc[:, 1] >= 2.0) & (xc[:, 1] <= 5.0), z_init_val, 0.0)
        if component.name == "H2O":
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


class FlowModel(
    ModelGeometry if mesh_2d_Q else ModelGeometry3D,
    FluidMixture,
    InitialConditions,
    BoundaryConditions,
    SecondaryEquations,
    FlowTemplate,
):
    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        return phase.saturation(domains)**2

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value: ", self.time_manager.time)
        print("Time index: ", self.time_manager.time_index)
        print("")

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

            total_volume = 0.0
            for sd in model.mdg.subdomains():
                total_volume += np.sum(
                    self.equation_system.evaluate(self.volume_integral(pp.ad.Scalar(1), [sd], dim=1)))

            self.equation_system
            # nonlinear_increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )

            # Residual per subsystem
            res_idx = self.equation_system.assembled_equation_indices
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
            raise ValueError(
                "Gravitational segregation is nonlinear in its simpler form."
            )
        return converged, diverged


day = 86400
t_scale = 1.0
tf = 0.1 * day
dt = 0.001 * day
time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    permeability=3.0e-12,
    porosity=0.35,
    thermal_conductivity=2.0 * to_Mega,
    density=2500.0,
    specific_heat_capacity=1000.0 * to_Mega,
)
material_constants = {"solid": solid_constants}
params = {
    "fractional_flow": True,
    "buoyancy_on": True,
    "material_constants": material_constants,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-8,
    "max_iterations": 100,
}

model = FlowModel(params)
model.prepare_simulation()
pp.run_time_dependent_model(model, params)
