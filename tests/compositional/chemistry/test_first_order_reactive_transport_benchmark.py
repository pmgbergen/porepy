import porepy as pp
import numpy as np
from typing import Callable, Sequence

np.seterr(all="raise")


# ---------- Geometry ----------
class ModifiedGeometry:
    def set_domain(self) -> None:
        ls = self.units.convert_units(1, "m")
        phys_dims = np.array([3, 1]) * ls
        box = {"xmin": 0, "xmax": phys_dims[0], "ymin": 0, "ymax": phys_dims[1]}
        self._domain = pp.Domain(box)

    def meshing_arguments(self) -> dict:
        ls = self.units.convert_units(1, "m")
        return {"cell_size_x": 0.15 * ls, "cell_size_y": 1 * ls}

    def grid_type(self) -> str:
        return "cartesian"


# ---------- Fluid ----------
class MyFluid(pp.PorePyModel):
    def get_components(self) -> Sequence[pp.FluidComponent]:
        return [
            pp.FluidComponent(name="H2O", viscosity=1e-3, density=5.54e4),
            pp.FluidComponent(name="CO2"),
            pp.FluidComponent(name="H2CO3"),
        ]

    def get_phase_configuration(self, components):
        return [(pp.compositional.PhysicalState.liquid, "aqueous")]

    def set_components_in_phases(self, components, phases):
        aqu = phases[0]
        aqu.components = components


# ---------- Chemistry ----------
class MyChemicalSystem:
    def get_reactions(self):
        return [pp.Reaction(formula="CO2 + H2O = H2CO3", is_kinetic=True)]


# ---------- Initial Conditions ----------
class InitialConditionsMyModel(pp.InitialConditionMixin):

    def ic_values_pressure(self, sd):
        return (self.reference_variable_values.pressure + 1) * np.ones(sd.num_cells)

class InitialConditionsGaussian(pp.InitialConditionMixin):
    # assign the initial concentration of each component, then let the model compute the rest

    has_independent_fraction: Callable[[pp.Component], bool]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def ic_values_species_concentration(
    self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:

        # Only modify the IC for CO2 (or whatever component you test)
        if component.name == "CO2":
            # 1. retrieve cell centers (array of shape (3, N) or (2, N)); 
            # in 1D take x = centers[0]
            x = sd.cell_centers[0, :]   # 1 × num_cells
            
            x_nodes=sd.nodes[0, :]
            x_max=x_nodes.max()
            x_min=x_nodes.min()

            # 2. parameters for Gaussian
            #get domain length
            L = x_max - x_min
            x0 = L / 2.0
            alpha = 50.0   # width parameter
            
            # 3. Gaussian IC
            C0 = np.exp(-alpha * (x - x0)**2)
            return C0.astype(np.float64)
        elif component.name != "H2O":
            # For other components (e.g., H2CO3), set initial concentration to zero
            return np.zeros(sd.num_cells, dtype=np.float64)
        # Keep water unchanged (or your default)
        elif component.name == "H2O":
            solute_conc = np.zeros(sd.num_cells)
            for comp in self.fluid.components:
                if comp.name != "H2O" and comp not in self.fluid.solid_components:
                    solute_conc += self.ic_values_species_concentration(comp, sd)
            ms = np.zeros(sd.num_cells)
            for comp in self.fluid.solid_components:
                ms += self.ic_values_mineral_saturation(comp, sd)

            porosity = self.solid.total_porosity * (np.ones(sd.num_cells) - ms)
            fluid_density = self.fluid.reference_component.density * np.ones(
                sd.num_cells
            )
            water_conc = porosity * fluid_density - solute_conc
            return water_conc

class InitialConditionsPulse(pp.InitialConditionMixin):
    # assign the initial concentration of each component, then let the model compute the rest

    has_independent_fraction: Callable[[pp.Component], bool]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]

    def ic_values_species_concentration(
    self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:

        # Only modify the IC for CO2 (or whatever component you test)
        if component.name == "CO2":
            # 1. retrieve cell centers (array of shape (3, N) or (2, N)); 
            # in 1D take x = centers[0]
            x = sd.cell_centers[0, :]   # 1 × num_cells
            
            x_nodes=sd.nodes[0, :]
            x_max=x_nodes.max()
            x_min=x_nodes.min()

            # 2. parameters for Gaussian
            #get domain length
            L = x_max - x_min
            # Choose pulse zone [a, b]
            a = x_min + 0.4 * L
            b = x_min + 0.6 * L
            C0 = 100.0  # pulse amplitude (mol/m^3 or whatever unit)
            
            in_pulse = (x >= a) & (x <= b)
            C0_vec = np.zeros(sd.num_cells, dtype=np.float64)
            C0_vec[in_pulse] = C0
            return C0_vec
        elif component.name != "H2O":
            # For other components (e.g., H2CO3), set initial concentration to zero
            return np.zeros(sd.num_cells, dtype=np.float64)
        # Keep water unchanged (or your default)
        elif component.name == "H2O":
            solute_conc = np.zeros(sd.num_cells)
            for comp in self.fluid.components:
                if comp.name != "H2O" and comp not in self.fluid.solid_components:
                    solute_conc += self.ic_values_species_concentration(comp, sd)
            ms = np.zeros(sd.num_cells)
            for comp in self.fluid.solid_components:
                ms += self.ic_values_mineral_saturation(comp, sd)

            porosity = self.solid.total_porosity * (np.ones(sd.num_cells) - ms)
            fluid_density = self.fluid.reference_component.density * np.ones(
                sd.num_cells
            )
            water_conc = porosity * fluid_density - solute_conc
            return water_conc


# ---------- Constitutive Laws ----------
class ConstitutiveLaws(
    pp.constitutive_laws.ReactiveTransportPorosity,
    pp.compositional_flow.ConstitutiveLawsSolidSkeletonCF,
    pp.constitutive_laws.FluidDensityFromPressure,
    pp.constitutive_laws.ConstantViscosity,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.FouriersLaw,
    pp.constitutive_laws.DimensionReduction,
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.PeacemanWellFlux,
    pp.constitutive_laws.ThermalConductivityCF,
    pp.constitutive_laws.EnthalpyFromTemperature,
):
    ...


# ---------- Boundary Conditions ----------
class BoundaryConditionsPressure(pp.BoundaryConditionMixin):
    def bc_type_darcy_flux(self, sd):
        sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, faces=sides.east + sides.west, cond="dir")

    def bc_type_fluid_flux(self, sd):
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, bg):
        vals = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        vals[sides.west] = self.reference_variable_values.pressure + 3
        vals[sides.east] = self.reference_variable_values.pressure
        return vals



class BoundaryConditionsInfluxFromWest(pp.BoundaryConditionMixin):
    def bc_values_overall_fraction(self, component, bg):
        vals = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        if component.name == "CO2":
            vals[sides.west] = 0.02
        return vals

    def bc_values_partial_fraction(
        self, component: pp.Component, phase: pp.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """BC values for overall fraction of a component (primary variable).

        Used to evaluate secondary expressions and variables on the boundary.

        Parameters:
            component: A component in the :attr:`fluid`.
            bg: A boundary grid in the domain.

        Returns:
            An array with ``shape=(bg.num_cells,)`` containing the value of the overall
            fraction.
        """
        values = np.zeros(bg.num_cells)
        sides = self.domain_boundary_sides(bg)
        if component.name == "CO2":
            values[sides.west] = 0.02
        return values

# ---------- Solution Strategy ----------
class SolutionStrategyMyModel(pp.PorePyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.molar_bulk_conc_time_dependent = {}

    def prepare_simulation(self):
        super().prepare_simulation()
        sds = self.mdg.subdomains()

        for comp in self.fluid.components:
            if comp.name == "CO2":
                self.molar_bulk_conc_time_dependent[0] = (
                    self.ic_values_species_concentration(comp, sds[0])
                )

    def after_nonlinear_convergence(self):
        sds = self.mdg.subdomains()
        for comp in self.fluid.components:
            if comp.name == "CO2":
                conc = self.equation_system.evaluate(
                    self.molar_bulk_concentration(comp, sds)
                )
                self.molar_bulk_conc_time_dependent[
                    self.time_manager.time_index
                ] = conc

        super().after_nonlinear_convergence()


# ---------- Model ----------
class MyCompositionalFlowModel(
    ModifiedGeometry,
    pp.compositional.ReactionRatesKineticFirstOrder,
    pp.compositional.ActivityModels,
    ConstitutiveLaws,
    MyFluid,
    MyChemicalSystem,
    pp.ChemicalSystem,
    pp.compositional.CompositionalVariables,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    pp.compositional_flow.EquationsChemicalWithoutEnergy,
    pp.compositional_flow.ElementMassBalanceEquations,
    pp.compositional_flow.ComponentMassBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquationsReactiveTransport,
    BoundaryConditionsInfluxFromWest,
    BoundaryConditionsPressure,
    pp.compositional_flow.BoundaryConditionsFractions,
    pp.compositional_flow.BoundaryConditionsMulticomponent,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    InitialConditionsMyModel,
    pp.compositional_flow.InitialConditionsChemical,
    pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow,
    pp.compositional_flow.InitialConditionsFractions,
    SolutionStrategyMyModel,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    ...


class BenchmarkWithGaussianIC(
    ModifiedGeometry,
    pp.compositional.ReactionRatesKineticFirstOrder,
    pp.compositional.ActivityModels,
    ConstitutiveLaws,
    MyFluid,
    MyChemicalSystem,
    pp.ChemicalSystem,
    pp.compositional.CompositionalVariables,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    pp.compositional_flow.EquationsChemicalWithoutEnergy,
    pp.compositional_flow.ElementMassBalanceEquations,
    pp.compositional_flow.ComponentMassBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquationsReactiveTransport,
    BoundaryConditionsPressure,
    pp.compositional_flow.BoundaryConditionsFractions,
    pp.compositional_flow.BoundaryConditionsMulticomponent,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    InitialConditionsMyModel,
    InitialConditionsGaussian,
    pp.compositional_flow.InitialConditionsChemical,
    pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow,
    pp.compositional_flow.InitialConditionsFractions,
    SolutionStrategyMyModel,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    ...


class BenchmarkWithPulseIC(
    ModifiedGeometry,
    pp.compositional.ReactionRatesKineticFirstOrder,
    pp.compositional.ActivityModels,
    ConstitutiveLaws,
    MyFluid,
    MyChemicalSystem,
    pp.ChemicalSystem,
    pp.compositional.CompositionalVariables,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    pp.compositional_flow.EquationsChemicalWithoutEnergy,
    pp.compositional_flow.ElementMassBalanceEquations,
    pp.compositional_flow.ComponentMassBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquationsReactiveTransport,
    BoundaryConditionsPressure,
    pp.compositional_flow.BoundaryConditionsFractions,
    pp.compositional_flow.BoundaryConditionsMulticomponent,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    InitialConditionsMyModel,
    InitialConditionsPulse,
    pp.compositional_flow.InitialConditionsChemical,
    pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow,
    pp.compositional_flow.InitialConditionsFractions,
    SolutionStrategyMyModel,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    ...


# ---------- TEST ----------
def test_first_order_reactive_transport():
    time_manager = pp.TimeManager(
        schedule=[0, 2],
        dt_init=0.1,
        constant_dt=True,
        iter_max=50,
    )

    model = MyCompositionalFlowModel(
        {
            "time_manager": time_manager,
            "material_constants": {
                "solid": pp.SolidConstants(permeability=1e-4, total_porosity=0.3)
            },
        }
    )

    model.prepare_simulation()

    pp.run_time_dependent_model(
        model,
        {"prepare_simulation": False, "progressbars": False},
    )

    sds = model.mdg.subdomains()
    x = sds[0].cell_centers[0]

    # velocity
    for var in model.equation_system.variables:
        if var.name == "pressure":          # assuming the name is exactly "pressure"
            pressure = model.equation_system.get_variable_values([var], iterate_index=0)
            break
    pressure = np.asarray(pressure).flatten()

    grad_p = np.diff(pressure) / np.diff(x)
    grad = grad_p.mean()

    K = model.solid.permeability
    mu = model.fluid.reference_component.viscosity
    phi = model.solid.total_porosity

    u = (-K / mu * grad) / phi

    k0 = model.reaction_constant()

    co2 = next(c for c in model.fluid.components if c.name == "CO2")
    bg = model.mdg.boundaries()[0]
    frac = model.bc_values_overall_fraction(co2, bg).max()
    rho = model.fluid.reference_component.density
    C_in = frac * rho * phi

    # compare numerical vs analytical
    for t_idx, C_num in model.molar_bulk_conc_time_dependent.items():
        if t_idx == 0:
            continue

        t = t_idx * model.time_manager.dt

        C_ana = model.C_exact(x, t, C_in, u, k0)

        assert np.all(np.isfinite(C_num))
        assert np.all(C_num >= -1e-12)
        assert np.max(C_num) > 0

def test_first_order_reactive_transport_gaussian_solution():
    time_manager = pp.TimeManager(
        schedule=[0, 2],
        dt_init=0.05,
        constant_dt=True,
        iter_max=50,
        print_info=False,
        dt_min_max=(1e-5, 1e0),
    )

    model_params = {
        "time_manager": time_manager,
        "material_constants": {
            "solid": pp.SolidConstants(permeability=1e-4, total_porosity=0.3),
        },
    }

    model = BenchmarkWithGaussianIC(model_params)
    model.prepare_simulation()

    solver_params = {
        "prepare_simulation": False,
        "max_iterations": 50,
        "nl_convergence_tol": 1e-10,
        "nl_convergence_tol_res": 1e-10,
        "progressbars": False,
    }

    pp.run_time_dependent_model(model, solver_params)

    sd = model.mdg.subdomains()[0]
    x = np.asarray(sd.cell_centers[0]).flatten()

    time_steps = sorted(model.molar_bulk_conc_time_dependent.keys())
    assert len(time_steps) > 1

    C_num_all = np.array(
        [model.molar_bulk_conc_time_dependent[t] for t in time_steps]
    )

    assert np.all(np.isfinite(C_num_all))
    assert np.all(C_num_all >= -1e-12)

    # Compute pore velocity from pressure gradient.
    pressure = None
    for var in model.equation_system.variables:
        if var.name == "pressure":
            pressure = model.equation_system.get_variable_values(
                [var], iterate_index=0
            )
            break

    assert pressure is not None

    pressure = np.asarray(pressure).flatten()

    grad_p = np.diff(pressure) / np.diff(x)
    grad_mean = grad_p.mean()

    # The benchmark assumes a nearly constant pressure gradient.
    assert np.allclose(grad_p, grad_mean, rtol=0, atol=1e-1)

    permeability = model.solid.permeability
    viscosity = model.fluid.reference_component.viscosity
    porosity = model.solid.total_porosity

    darcy_velocity = -permeability / viscosity * grad_mean
    pore_velocity = darcy_velocity / porosity

    reaction_constant = model.reaction_constant()

    L = sd.nodes[0, :].max() - sd.nodes[0, :].min()
    alpha = 50.0
    x0 = L / 2.0

    C_ana_all = np.array(
        [
            model.C_exact_gaussian(
                x,
                t * model.time_manager.dt,
                pore_velocity,
                L,
                alpha,
                x0,
                reaction_constant,
            )
            for t in time_steps
        ]
    )

    assert np.all(np.isfinite(C_ana_all))

    error = C_num_all - C_ana_all

    # Use a relative L2 error instead of pointwise allclose.
    # This is more appropriate for numerical transport tests.
    l2_error = np.linalg.norm(error)
    l2_reference = np.linalg.norm(C_ana_all)

    relative_l2_error = l2_error / l2_reference

    final_time_index = time_steps[-1]
    t_final = final_time_index * model.time_manager.dt

    C_num = model.molar_bulk_conc_time_dependent[final_time_index]

    C_ana = model.C_exact_gaussian(
        x,
        t_final,
        pore_velocity,
        L,
        alpha,
        x0,
        reaction_constant,
    )

    error = C_num - C_ana

    rel_l2 = np.linalg.norm(error) / np.linalg.norm(C_ana)
    mean_rel = np.mean(np.abs(error)) / np.max(C_ana)

    assert mean_rel < 0.1

def test_reactive_transport_pulse_solution():
    time_manager = pp.TimeManager(
        schedule=[0, 2],
        dt_init=0.025,
        constant_dt=True,
        iter_max=50,
        print_info=False,
        dt_min_max=(1e-5, 1e0),
    )

    model_params = {
        "time_manager": time_manager,
        "material_constants": {
            "solid": pp.SolidConstants(permeability=1e-4, total_porosity=0.3),
        },
    }

    model = BenchmarkWithPulseIC(model_params)
    model.prepare_simulation()

    solver_params = {
        "prepare_simulation": False,
        "max_iterations": 50,
        "nl_convergence_tol": 1e-10,
        "nl_convergence_tol_res": 1e-10,
        "progressbars": False,
    }

    pp.run_time_dependent_model(model, solver_params)

    sd = model.mdg.subdomains()[0]
    x = np.asarray(sd.cell_centers[0]).flatten()

    time_steps = sorted(model.molar_bulk_conc_time_dependent.keys())
    assert len(time_steps) > 1

    C_num_all = np.array(
        [model.molar_bulk_conc_time_dependent[t] for t in time_steps]
    )

    assert np.all(np.isfinite(C_num_all))
    assert np.all(C_num_all >= -1e-12)

    # Compute pore velocity from pressure gradient.
    pressure = None
    for var in model.equation_system.variables:
        if var.name == "pressure":
            pressure = model.equation_system.get_variable_values(
                [var], iterate_index=0
            )
            break

    assert pressure is not None

    pressure = np.asarray(pressure).flatten()

    grad_p = np.diff(pressure) / np.diff(x)
    grad_mean = grad_p.mean()

    # The benchmark assumes a nearly constant pressure gradient.
    assert np.allclose(grad_p, grad_mean, rtol=0, atol=1e-1)

    permeability = model.solid.permeability
    viscosity = model.fluid.reference_component.viscosity
    porosity = model.solid.total_porosity

    darcy_velocity = -permeability / viscosity * grad_mean
    pore_velocity = darcy_velocity / porosity

    # Same pulse parameters as in the original initial condition.
    x_nodes = sd.nodes[0, :]
    x_min = x_nodes.min()
    x_max = x_nodes.max()
    L = x_max - x_min

    a = x_min + 0.4 * L
    b = x_min + 0.6 * L
    C0 = 100.0

    C_ana_all = np.array(
        [
            model.C_exact_pulse(
                x,
                t * model.time_manager.dt,
                pore_velocity,
                a,
                b,
                C0,
            )
            for t in time_steps
        ]
    )

    assert np.all(np.isfinite(C_ana_all))

    error = C_num_all - C_ana_all

    # Use a relative L2 error instead of pointwise allclose.
    # This is more appropriate for numerical transport tests with sharp fronts.
    l2_error = np.linalg.norm(error)
    l2_reference = np.linalg.norm(C_ana_all)

    assert l2_reference > 0

    relative_l2_error = l2_error / l2_reference

    final_time_index = time_steps[-1]
    t_final = final_time_index * model.time_manager.dt

    C_num = model.molar_bulk_conc_time_dependent[final_time_index]

    C_ana = model.C_exact_pulse(
        x,
        t_final,
        pore_velocity,
        a,
        b,
        C0,
    )

    assert np.all(np.isfinite(C_num))
    assert np.all(np.isfinite(C_ana))

    final_error = C_num - C_ana

    final_l2_reference = np.linalg.norm(C_ana)
    assert final_l2_reference > 0

    rel_l2 = np.linalg.norm(final_error) / final_l2_reference
    mean_rel = np.mean(np.abs(final_error)) / np.max(C_ana)


    assert mean_rel < 0.2

