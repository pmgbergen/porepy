from typing import Callable, Sequence
import numpy as np
import porepy as pp
from porepy.models.abstract_equations import LocalElimination
from porepy.models.compositional_flow import CompositionalFlowTemplate

from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_closure import NumericClosure
from porepy.examples.geothermal_flow.cf_convergence_analysis.mms_halite_data import HaliteMMSDataSaving
from porepy.examples.geothermal_flow.cf_convergence_analysis import mms_halite_2d_frac as exact
from .poro_perm_aperture_model import (
    ApertureMixin,
    PermeabilityMixin,
    PorosityMixin,
)


class SurrogatePhaseEOS(pp.compositional.EquationOfState):
    """Constant phase properties with phase enthalpy equal to primary enthalpy."""

    def __init__(self, components, constants, is_solid_phase):
        super().__init__(components)
        self.constants = constants
        self.is_solid_phase = is_solid_phase

    def compute_phase_properties(self, phase_state, *dependencies, params=None):
        pressure, enthalpy, composition = dependencies
        size = pressure.size
        constants = self.constants
        zeros = np.zeros((3, size))
        derivative = zeros.copy()
        derivative[1] = 1
        density = constants.rho_hal if self.is_solid_phase else constants.rho_liq
        conductivity = constants.kappa_hal if self.is_solid_phase else constants.kappa_liq
        return pp.compositional.PhaseProperties(
            state=phase_state,
            rho=np.full(size, density), 
            drho=zeros.copy(),
            h=enthalpy.copy(), 
            dh=derivative,
            mu=np.full(size, constants.mu_liq), 
            dmu=zeros.copy(),
            kappa=np.full(size, conductivity), 
            dkappa=zeros.copy(),
            phis=np.ones((2, size)), 
            dphis=np.zeros((2, 3, size)),
        )


class HaliteThermodynamics(LocalElimination):
    """Analytic closure registered through local elimination."""

    def get_components(self):
        return pp.compositional.load_fluid_constants(["H2O", "NaCl"], "chemicals")

    def get_phase_configuration(self, components):
        return [
            (pp.compositional.PhysicalState.liquid, "liq",
             SurrogatePhaseEOS(components, self.constants, False)),
            (pp.compositional.PhysicalState.halite, "halite",
             SurrogatePhaseEOS(components, self.constants, True)),
        ]

    def dependencies_of_phase_properties(
        self, phase:pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        z_NaCl = [
            comp.fraction
            for comp in self.fluid.components
            if comp != self.fluid.reference_component
        ]

        return [self.pressure, self.enthalpy] + z_NaCl

    def set_equations(self):
        
        super().set_equations()

        subdomains = self.mdg.subdomains()
        domains = subdomains + self.mdg.boundaries()
        dependencies = self.dependencies_of_phase_properties(self.fluid.reference_phase)

        for phase in self.fluid.phases:
            if phase.name == "halite":
                self.eliminate_locally(
                    phase.saturation, 
                    dependencies,
                    self.closure.halite_saturation_func, 
                    domains,
                )
            for component in phase:
                if self.has_independent_partial_fraction(component, phase):
                    function = getattr(self.closure, f"{component.name}_{phase.name}_func")
                    self.eliminate_locally(
                        phase.partial_fraction_of[component], dependencies,
                        function, 
                        domains,
                    )
        self.eliminate_locally(
            self.temperature, 
            dependencies, 
            self.closure.temperature_func, 
            subdomains,
        )


class HaliteMMSPhysics(
    ApertureMixin,
    PorosityMixin,
    PermeabilityMixin,
):
    """Constitutive laws of the manufactured-solution test, with every non-smooth law replaced by a smooth one."""

    def fracture_or_intersection_aperture(self, subdomains):
        """Aperture ``a0 (1 - s_hal)**theta`` without the ``a_min`` floor.

        Parameters:
            subdomains: List of fracture and intersection subdomains.

        Returns:
            Cell-wise aperture operator.

        """
        saturation = self.fluid.phases[1].saturation(subdomains)

        return pp.ad.Scalar(self.constants.a0) * (
            pp.ad.Scalar(1) - saturation
        ) ** pp.ad.Scalar(self.constants.theta)

    def porosity_fracture_and_intersection(self, subdomains):
        """Unit fracture porosity.

        Parameters:
            subdomains: List of fracture and intersection subdomains.

        Returns:
            Cell-wise porosity operator.

        """
        return pp.wrap_as_dense_ad_array(1.0, sum(sd.num_cells for sd in subdomains))

    def porosity_matrix(self, subdomains):
        """Matrix porosity ``phi0 (1 - s_hal)``.

        Parameters:
            subdomains: List of matrix subdomains.

        Returns:
            Cell-wise porosity operator.

        """
        saturation = self.fluid.phases[1].saturation(subdomains)
        porosity = pp.ad.Scalar(self.constants.phi0) * (
            pp.ad.Scalar(1) - saturation
        )
        porosity.set_name("mms_matrix_porosity")
        return porosity

    def porosity_matrix_const(self, subdomains):
        """Matrix porosity ``phi0 (1 - s_hal)``.

        Parameters:
            subdomains: List of matrix subdomains.

        Returns:
            Cell-wise porosity operator.

        """
        size = sum(sd.num_cells for sd in subdomains)
        porosity = pp.wrap_as_dense_ad_array(
            self.constants.phi0,
            size,
            name="porosity",
        )
        porosity.set_name("mms_matrix_porosity")
        return porosity

    def matrix_permeability(self, subdomains):
        """Matrix permeability ``K0 (1 - s_hal)**2``.

        Parameters:
            subdomains: List of matrix subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        saturation = self.fluid.phases[1].saturation(subdomains)
        permeability = pp.ad.Scalar(self.constants.K0) * (
            pp.ad.Scalar(1) - saturation
        ) ** pp.ad.Scalar(2)
        return self.isotropic_second_order_tensor(subdomains, permeability)
    
    def matrix_permeability_const(self, subdomains):
        """Matrix permeability ``K0 (1 - s_hal)**2``.

        Parameters:
            subdomains: List of matrix subdomains.

        Returns:
            Cell-wise permeability operator.

        """

        size = sum(sd.num_cells for sd in subdomains)
        base_perm = pp.wrap_as_dense_ad_array(
            self.constants.K0,
            size,
            name="permeability",
        )
        return self.isotropic_second_order_tensor(subdomains, base_perm)

    def normal_permeability(self, interfaces):
        """Constant normal permeability ``K_k``.

        Parameters:
            interfaces: List of interfaces.

        Returns:
            Cell-wise normal permeability operator.

        """
        return pp.ad.Scalar(self.constants.K_k)

    def relative_permeability(self, phase, domains):
        """Relative permeability ``s**2`` of the phase saturation; zero for halite.

        Parameters:
            phase: Phase whose relative permeability is returned.
            domains: Subdomains or boundary grids.

        Returns:
            Cell-wise relative permeability operator.

        """
        saturation = phase.saturation(domains)
        if phase.name == "halite":
            return pp.ad.Scalar(0) * saturation
        return saturation ** pp.ad.Scalar(2)

    def add_nonlinear_darcy_flux_discretization(self):
        self.add_nonlinear_diffusive_flux_discretization(
            self.darcy_flux_discretization(self.mdg.subdomains()).flux()
        )


class HaliteMMSGeometry:
    """Unit square with one immersed vertical segment and no well grids."""

    def set_domain(self):
        self._domain = pp.Domain({"xmax": 1.0, "ymax": 1.0})

    def set_fractures_old(self):
        self._fractures = [pp.LineFracture(np.array([[0.5, 0.5], [0.25, 0.75]]))]

    def set_fractures(self):
        self._fractures = [
            pp.LineFracture(
                np.array([[0.5, 0.5], [0.25, 0.75]])
            )
        ]

        if self.grid_type() == "simplex":
            self._fractures.extend([
                pp.LineFracture(
                    np.array([[0.0, 1.0], [0.25, 0.25]])
                ),
                pp.LineFracture(
                    np.array([[0.0, 1.0], [0.75, 0.75]])
                ),
            ])

    def meshing_kwargs(self):
        kwargs = dict(super().meshing_kwargs())
        if self.grid_type() == "simplex":
            kwargs["constraints"] = np.array([1, 2], dtype=int)
        return kwargs

    def grid_type(self):
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self):
        return self.params.get("meshing_arguments", {"cell_size": 0.125})


class HaliteMMSConditions:
    """Exact initial values and external Dirichlet data; no flow at the fracture tips."""

    def exact_values(self, grid, name, time):
        centers = grid.cell_centers
        dimension = grid.parent.dim if isinstance(grid, pp.BoundaryGrid) else grid.dim
        if dimension == 2:
            return self.matrix_exact(name, centers[0], centers[1], time)
        return self.fracture_exact[name](centers[1], time)

    def bc_type_darcy_flux(self, sd):
        faces = self.domain_boundary_sides(sd).all_bf
        return pp.BoundaryCondition(sd, faces, "dir" if sd.dim == 2 else "neu")

    bc_type_fluid_flux = bc_type_darcy_flux
    bc_type_fourier_flux = bc_type_darcy_flux
    bc_type_enthalpy_flux = bc_type_darcy_flux

    def bc_values_pressure(self, bg):
        return self.exact_values(bg, "pressure", self.time_manager.time)

    def bc_values_enthalpy(self, bg):
        return self.exact_values(bg, "enthalpy", self.time_manager.time)

    def bc_values_temperature(self, bg):
        return self.exact_values(bg, "temperature", self.time_manager.time)

    def bc_values_overall_fraction(self, component, bg):
        values = self.exact_values(bg, "composition", self.time_manager.time)
        return values if component.name == "NaCl" else 1 - values

    def ic_values_pressure(self, sd):
        return self.exact_values(sd, "pressure", 0.0)

    def ic_values_enthalpy(self, sd):
        return self.exact_values(sd, "enthalpy", 0.0)

    def ic_values_temperature(self, sd):
        return self.exact_values(sd, "temperature", 0.0)

    def ic_values_overall_fraction(self, component, sd):
        values = self.exact_values(sd, "composition", 0.0)
        return values if component.name == "NaCl" else 1 - values


class HaliteMMSSources:
    """Exact source terms integrated over cells, using the exact specific volume."""

    def update_mms_sources(self):
        time = self.time_manager.time
        for sd, data in self.mdg.subdomains(return_data=True):
            volume = sd.cell_volumes.copy()
            if sd.dim == 1:
                volume *= self.exact_values(sd, "aperture", time)
            for name in ("mass", "NaCl", "energy"):
                values = volume * self.exact_values(sd, f"source_{name}", time)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Non-finite MMS source {name} on dimension {sd.dim}")
                pp.set_solution_values(
                    name=f"mms_{name}", values=values, data=data, time_step_index=0,
                )

    def before_nonlinear_loop(self):
        self.update_mms_sources()
        super().before_nonlinear_loop()

    def mms_source(self, name, subdomains):
        return pp.ad.TimeDependentDenseArray(
            name=f"mms_{name}", domains=subdomains
        ).previous_timestep()

    def fluid_source(self, subdomains):
        return super().fluid_source(subdomains) + self.mms_source("mass", subdomains)

    def component_source(self, component, subdomains):
        source = super().component_source(component, subdomains)
        if component.name == "NaCl":
            return source + self.mms_source("NaCl", subdomains)
        return source + self.mms_source("mass", subdomains) - self.mms_source("NaCl", subdomains)

    def energy_source(self, subdomains):
        return super().energy_source(subdomains) + self.mms_source("energy", subdomains)


class HaliteMMSModel(
    HaliteMMSGeometry, 
    HaliteMMSConditions, 
    HaliteMMSSources, 
    HaliteMMSDataSaving,
    HaliteMMSPhysics, 
    HaliteThermodynamics, 
    CompositionalFlowTemplate,
):
    """PorePy model of the manufactured-solution convergence test."""

    def prepare_simulation(self):
        super().prepare_simulation()
        # eliminate_locally sets eliminated variables at iterate_index=0 only, after the
        # initial state was copied to the time-step slot; without this they start at 0.
        system = self.equation_system
        system.set_variable_values(system.get_variable_values(iterate_index=0), time_step_index=0)
        assert [grid.dim for grid in self.mdg.subdomains()] == [2, 1], (
            "Expected exactly one matrix and one physical fracture."
        )
        assert len(self.mdg.interfaces()) == 1, (
            "Expected exactly one two-sided matrix-fracture interface."
        )

        # check that fracture tips snap to the correct locations.
        fracture = self.mdg.subdomains(dim=1)[0]
        x, y, c = fracture.nodes[0], fracture.nodes[1], self.constants
        assert (np.allclose(x, c.x_frac) and np.isclose(y.min(), c.y_tip_lo)
                and np.isclose(y.max(), c.y_tip_hi)), (
            f"The mesh moved the fracture to x={x.mean():.3f}, "
            f"y=[{y.min():.3f}, {y.max():.3f}]; use a cell size 0.25/k so that "
            "x=1/2 and y=1/4, 3/4 are grid lines."
        )


    def __init__(self, params=None):
        params = {} if params is None else params.copy()
        if params.get("fractional_flow", False):
            raise ValueError("The MMS requires mobility-free Darcy flux: fractional_flow=False")
        self.constants = params.get("mms_constants")
        self.closure = NumericClosure(self.constants)
        self.matrix_exact = exact.MatrixCallables(self.constants)
        self.fracture_exact = exact.make_fracture_callables(self.constants)
        super().__init__(params)

    def _is_nonlinear_problem(self):
        return True

    def after_nonlinear_iteration(self, nonlinear_increment):
        super().after_nonlinear_iteration(nonlinear_increment)
        if self.params.get("debug_mms_residual", False):
            if self.nonlinear_solver_statistics.num_iteration == 1:
                print(f"\nTime step {self.time_manager.time_index}, "
                      f"t = {self.time_manager.time:.4g}")
            _, res = self.equation_system.assemble()
            print(f"  it {self.nonlinear_solver_statistics.num_iteration}  "
                f"|res| = {np.linalg.norm(res):.3e}")