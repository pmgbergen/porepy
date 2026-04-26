import numpy as np
import porepy as pp
from typing import Sequence
np.seterr(all="raise")


class ModifiedGeometry:
    def set_domain(self) -> None:
        ls = self.units.convert_units(1, "m")
        phys_dims = np.array([3, 1]) * ls
        box = {"xmin": 0, "xmax": phys_dims[0], "ymin": 0, "ymax": phys_dims[1]}
        self._domain = pp.Domain(box)

    def meshing_arguments(self) -> dict:
        ls = self.units.convert_units(1, "m")
        return {"cell_size_x": 1 * ls, "cell_size_y": 1 * ls}

    def grid_type(self) -> str:
        return "cartesian"


class MyFluid(pp.PorePyModel):
    def get_components(self):
        return [
            pp.FluidComponent(name="H2O", viscosity=1e-3, density=5.54e4),
            pp.FluidComponent(name="Li+"),
            pp.FluidComponent(name="CO3-2"),
            pp.FluidComponent(name="Li2CO3", molar_volume=3.5e-5),
        ]

    def get_phase_configuration(self, components):
        return [
            (pp.compositional.PhysicalState.liquid, "aqueous"),
            (pp.compositional.PhysicalState.solid, "solid"),
        ]

    def set_components_in_phases(self, components, phases):
        c1, c2, c3, c4 = components
        aqu, sol = phases
        aqu.components = [c1, c2, c3]
        sol.components = [c4]


class MyCompressibleFluid(pp.PorePyModel):
    def get_components(self) -> Sequence[pp.FluidComponent]:
        return [
            pp.FluidComponent(
                name="H2O", viscosity=1e-3, density=5.54e4, compressibility=4.5e-10
            ),
            pp.FluidComponent(name="Li+"),
            pp.FluidComponent(name="CO3-2"),
            pp.FluidComponent(name="Li2CO3", molar_volume=3.5e-5),
        ]


class MyChemicalSystem:
    def get_reactions(self):
        return [pp.Reaction(formula="Li2CO3 = 2Li+ + CO3-2", is_kinetic=True)]


class ReactionRatesKineticZero:
    def set_kinetic_reaction_rates(self, reactions):
        def rr(domains):
            return pp.ad.Scalar(0.0)

        for reaction in reactions:
            reaction.reaction_rate = rr

        return reactions


class ReactionRatesKineticConstant:
    def set_kinetic_reaction_rates(
        self, reactions: Sequence[pp.Reaction]
    ) -> Sequence[pp.Reaction]:
        """Sets the reaction rates for kinetic reactions.

        Parameters:
            reactions: A list of Reaction objects defining the chemical reactions.
        This needs to be overridden to provide actual reaction rates.
        """

        def rr(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            return pp.ad.Scalar(2, "synthetic_kinetic_reaction_rate")

        for reaction in reactions:
            if reaction.is_kinetic:
                reaction.reaction_rate = rr
            else:

                def rr_eq(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
                    return pp.ad.Scalar(0.0, "equilibrium_reaction_rate")

                reaction.reaction_rate = rr_eq

        return reactions


class InitialConditionsMyModel(pp.InitialConditionMixin):
    def ic_values_species_concentration(self, component, sd):
        if component.name == "Li+":
            return np.zeros(sd.num_cells, dtype=np.float64)
        elif component.name == "CO3-2":
            return np.zeros(sd.num_cells, dtype=np.float64)
        elif component.name == "Li2CO3":
            return 200 * np.ones(sd.num_cells, dtype=np.float64)
        elif component.name == "H2O":
            solute_conc = np.zeros(sd.num_cells, dtype=np.float64)

            for comp in self.fluid.components:
                if comp.name != "H2O" and comp not in self.fluid.solid_components:
                    solute_conc += self.ic_values_species_concentration(comp, sd)

            mineral_saturation = np.zeros(sd.num_cells, dtype=np.float64)
            for comp in self.fluid.solid_components:
                mineral_saturation += self.ic_values_mineral_saturation(comp, sd)

            porosity = self.solid.total_porosity * (
                np.ones(sd.num_cells, dtype=np.float64) - mineral_saturation
            )
            fluid_density = self.fluid.reference_component.density * np.ones(
                sd.num_cells, dtype=np.float64
            )
            return porosity * fluid_density - solute_conc

        raise ValueError(f"Unknown component: {component.name}")

    def ic_values_overall_fraction(self, component, sd):
        total_conc = np.zeros(sd.num_cells, dtype=np.float64)
        for comp in self.fluid.components:
            total_conc += self.ic_values_species_concentration(comp, sd)

        initial_conc = self.ic_values_species_concentration(component, sd)
        return initial_conc / total_conc

    def ic_values_mineral_saturation(self, component, sd):
        if component.name == "Li2CO3":
            initial_conc = self.ic_values_species_concentration(component, sd)
            return component.molar_volume * initial_conc / self.solid.total_porosity

        return np.zeros(sd.num_cells, dtype=np.float64)

    def ic_values_partial_fraction(self, component, phase, sd):
        phase_conc = np.zeros(sd.num_cells, dtype=np.float64)
        for comp in phase.components:
            phase_conc += self.ic_values_species_concentration(comp, sd)

        if self.has_independent_partial_fraction(component, phase):
            initial_conc = self.ic_values_species_concentration(component, sd)
            return initial_conc / phase_conc

        return np.zeros(sd.num_cells, dtype=np.float64)

    def ic_values_pressure(self, sd):
        return (self.reference_variable_values.pressure + 1) * np.ones(
            sd.num_cells, dtype=np.float64
        )


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


class BoundaryConditionsMyModel(pp.BoundaryConditionMixin):
    def bc_type_darcy_flux(self, sd):
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(
            sd, faces=domain_sides.east + domain_sides.west, cond="dir"
        )

    def bc_type_fluid_flux(self, sd):
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, bg):
        pressure_vals = np.zeros(bg.num_cells, dtype=np.float64)
        domain_sides = self.domain_boundary_sides(bg)

        pressure_vals[domain_sides.west] = self.reference_variable_values.pressure + 2
        pressure_vals[domain_sides.east] = self.reference_variable_values.pressure

        return pressure_vals

    def bc_values_overall_fraction(self, component, bg):
        values = np.zeros(bg.num_cells, dtype=np.float64)
        sides = self.domain_boundary_sides(bg)

        if component.name == "Li+":
            values[sides.west] = 0.2
        elif component.name == "CO3-2":
            values[sides.west] = 0.1

        return values

    def bc_values_partial_fraction(self, component, phase, bg):
        values = np.zeros(bg.num_cells, dtype=np.float64)
        sides = self.domain_boundary_sides(bg)

        if component.name == "Li+":
            values[sides.west] = 0.2
        elif component.name == "CO3-2":
            values[sides.west] = 0.1

        return values



class BoundaryConditionsNeumannEverywhere(pp.BoundaryConditionMixin):


    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Setting the Dirichlet type on the east boundary, Neumann elsewhere."""
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(
            sd,
            faces=domain_sides.all_bf,
            cond="neu",
        )

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    # # This method did not change.
    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Everything is the same as in the previous example."""
        pressure_vals = np.zeros(bg.num_cells)
        domain_sides = self.domain_boundary_sides(bg)

        pressure_vals[domain_sides.all_bf] = self.reference_variable_values.pressure
        return pressure_vals



class SolutionStrategyMyModel(pp.PorePyModel):
    def after_nonlinear_convergence(self):
        self.comp_flux_after_solve = {}
        self.div_fluxes = {}

        sds = self.mdg.subdomains()
        div = pp.ad.Divergence(sds, dim=1)

        for comp in self.fluid.components:
            flux = self.equation_system.evaluate(self.component_flux(comp, sds))
            self.comp_flux_after_solve[comp.name] = flux
            self.div_fluxes[comp.name] = self.equation_system.evaluate(div @ flux)

        super().after_nonlinear_convergence()


class MyCompositionalFlowModel(
    ModifiedGeometry,
    ReactionRatesKineticZero,
    pp.compositional.ActivityModels,
    ConstitutiveLaws,
    MyFluid,
    MyChemicalSystem,
    pp.ChemicalSystem,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    pp.compositional.CompositionalVariables,
    pp.compositional_flow.EquationsChemicalWithoutEnergy,
    pp.compositional_flow.ElementMassBalanceEquations,
    pp.compositional_flow.ComponentMassBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquationsReactiveTransport,
    BoundaryConditionsMyModel,
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
    pass

class KineticReactionsWithoutFlowModel(
    MyCompressibleFluid,
    ReactionRatesKineticConstant,
    BoundaryConditionsNeumannEverywhere,
    MyCompositionalFlowModel,
):
    pass




def test_inactive_mineral_does_not_affect_mobile_component_balance():
    time_manager = pp.TimeManager(
        schedule=[0, 5e-4],
        dt_init=5e-4,
        constant_dt=True,
    )

    model = MyCompositionalFlowModel(
        {
            "time_manager": time_manager,
            "material_constants": {
                "solid": pp.SolidConstants(permeability=1e-5),
            },
        }
    )

    model.prepare_simulation()
    sds = model.mdg.subdomains()

    initial = {}
    for comp in model.fluid.components:
        initial[comp.name] = model.ic_values_species_concentration(comp, sds[0]).copy()

    pp.run_time_dependent_model(
        model,
        {
            "prepare_simulation": False,
            "progressbars": False,
    "nl_max_iterations": 20,  # Max iterations of a nonlinear solver (Newton)
    "nl_convergence_inc_atol": 1e-9,  # Increment norm
    "nl_convergence_res_atol": 1e-9,  # Residual norm
        },
    )

    for comp in model.fluid.components:
        if comp.name == "Li2CO3":
            continue

        molar_bulk_conc = model.equation_system.evaluate(
            model.molar_bulk_concentration(comp, sds)
        )

        comp_flux_div = model.div_fluxes[comp.name]

        assert np.allclose(
            molar_bulk_conc,
            initial[comp.name] - comp_flux_div * time_manager.dt,
        )

def test_kinetic_mineral_reaction_consumes_li2co3_without_negative_concentration():
    time_manager = pp.TimeManager(
        schedule=[0, 0.2],
        dt_init=0.2,
        constant_dt=True,
        iter_max=50,
        print_info=False,
        dt_min_max=(1e-5, 1e0),
    )

    model_params = {
        "time_manager": time_manager,
        "material_constants": {
            "solid": pp.SolidConstants(permeability=1e-5),
        },
    }

    model = KineticReactionsWithoutFlowModel(model_params)
    model.prepare_simulation()

    sds = model.mdg.subdomains()

    initial_concentration = {
        comp.name: model.ic_values_species_concentration(comp, sds[0]).copy()
        for comp in model.fluid.components
    }

    solver_params = {
        "prepare_simulation": False,
        "max_iterations": 50,
        "nl_convergence_tol": 1e-11,
        "nl_convergence_tol_res": 1e-11,
        "progressbars": False,
    }

    pp.run_time_dependent_model(model, solver_params)

    S = model.fluid.stoichiometric_matrix
    species_names = model.species_names
    # For this test, there is one kinetic reaction.
    assert len(model.reactions) == 1

    reaction = model.reactions[0]
    reaction_rate = reaction.reaction_rate(sds)
    reaction_rate_value = model.equation_system.evaluate(reaction_rate)

    for comp in model.fluid.components:
        species_index = species_names.index(comp.name)

        actual = model.equation_system.evaluate(
            model.molar_bulk_concentration(comp, sds)
        )

        expected = (
            initial_concentration[comp.name]
            + S[0, species_index] * reaction_rate_value * time_manager.dt
        )

        assert np.allclose(actual, expected)