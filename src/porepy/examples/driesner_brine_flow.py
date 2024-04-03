"""Example implementing a multi-phase multi component flow of H2O-NaCl using
the Driesner correlations as constitutive laws.

This model uses pressure, specific fluid mixture enthalpy and NaCl overall fraction as
primary variables.

No equilibrium calculations included.

Ergo, the user must close the model to provide expressions for saturations, partial
fractions and temperature, depending on primary variables.

Note:
    With some additional work, it is straight forward to implement a model without
    h as the primary variable, but T.

    What needs to change is:

    1. Overwrite
       porepy.models.compositional_flow.VariablesCF
       mixin s.t. it does not create a h variable.
    2. Modify accumulation term in
       porepy.models.compositional_flow.TotalEnergyBalanceEquation_h
       to use T, not h.
    3. H20_NaCl_brine.dependencies_of_phase_properties: Use T instead of h.

"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

import porepy as pp
import porepy.composite as ppc
from FluidDescription import LiquidLikeLinearTracerCorrelations
from FluidDescription import GasLikeLinearTracerCorrelations
from porepy.applications.md_grids.domains import nd_cube_domain

# CompositionalFlow has data savings mixin, composite variables mixin,
# Solution strategy eliminating local equations with Schur complement and no flash.
# It also hase the ConstitutiveLaws for CF, which use the FluidMixture.
# For changing constitutive laws, import ConstitutiveLawsCF and overwrite mixins
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
    SecondaryEquationsMixin,
)

class H20_NaCl_brine(ppc.FluidMixtureMixin):
    """Mixture mixin creating the brine mixture with two components."""

    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.compositional_flow.VariablesEnergyBalance`."""

    def get_components(self) -> Sequence[ppc.Component]:
        """Setting H20 as first component in Sequence makes it the reference component.
        z_H20 will be eliminated."""
        species = ppc.load_species(["H2O", "NaCl"])
        components = [ppc.Component.from_species(s) for s in species]
        return components

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.AbstractEoS, int, str]]:
        eos_L = LiquidLikeLinearTracerCorrelations(components)
        eos_G = GasLikeLinearTracerCorrelations(components)
        return [(eos_L, 0, "liq"), (eos_G, 1, "gas")]

    def dependencies_of_phase_properties(
        self, phase: ppc.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]:
        z_NaCl = [
            comp.fraction
            for comp in self.fluid_mixture.components
            if comp != self.fluid_mixture.reference_component
        ]
        return [self.pressure, self.enthalpy] + z_NaCl

    def set_components_in_phases(
        self, components: Sequence[ppc.Component], phases: Sequence[ppc.Phase]
    ) -> None:
        """By default, the unified assumption is applied: all components are present
        in all phases."""
        super().set_components_in_phases(components, phases)


class ModelGeometry:
    def set_domain(self) -> None:
        dimension = 2
        size_x = self.solid.convert_units(10, "m")
        size_y = self.solid.convert_units(1, "m")
        size_z = self.solid.convert_units(1, "m")

        box: dict[str, pp.number] = {"xmax": size_x}

        if dimension > 1:
            box.update({"ymax": size_y})

        if dimension > 2:
            box.update({"zmax": size_z})

        self._domain = pp.Domain(box)

    # def set_fractures(self) -> None:
    #     frac_1_points = self.solid.convert_units(
    #         np.array([[0.2, 0.8], [0.2, 0.8]]), "m"
    #     )
    #     frac_1 = pp.LineFracture(frac_1_points)
    #
    #     frac_2_points = self.solid.convert_units(
    #         np.array([[0.2, 0.8], [0.8, 0.2]]), "m"
    #     )
    #     frac_2 = pp.LineFracture(frac_2_points)
    #     self._fractures = [frac_1, frac_2]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.1, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class BoundaryConditions(BoundaryConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if self.mdg.dim_max() == 2:
            all, east, west, north, south, top, bottom = self.domain_boundary_sides(sd)
            return pp.BoundaryCondition(sd, all, "dir")
        elif self.mdg.dim_max() == 3:
            all, east, west, north, south, top, bottom = self.domain_boundary_sides(sd)
            return pp.BoundaryCondition(sd, all, "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Returns the BC type of the darcy flux for consistency reasons."""
        return self.bc_type_darcy_flux(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if self.mdg.dim_max() == 2:
            all, east, west, north, south, top, bottom = self.domain_boundary_sides(sd)
            return pp.BoundaryCondition(sd, all, "dir")
        elif self.mdg.dim_max() == 3:
            all, east, west, north, south, top, bottom = self.domain_boundary_sides(sd)
            return pp.BoundaryCondition(sd, all, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        if self.mdg.dim_max() == 2:
            all, east, west, north, south, top, bottom = self.domain_boundary_sides(boundary_grid)
        elif self.mdg.dim_max() == 3:
            all, east, west, north, south, top, bottom = self.domain_boundary_sides(boundary_grid)
        p_inlet = 10.0e6
        xcs = boundary_grid.cell_centers.T
        l = 10.0
        def p_D(xc):
            x, y, z = xc
            return p_inlet * (1 - x/l) + 5
        p_D_iter = map(p_D, xcs)
        vals = np.fromiter(p_D_iter,dtype=float)
        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        T = 500.0
        return np.ones(boundary_grid.num_cells) * T

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h = 1.0e5
        return np.ones(boundary_grid.num_cells) * h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        z = 0.2
        if component.name == 'H2O':
            return (1 - z) * np.ones(boundary_grid.num_cells)
        else:
            return z * np.ones(boundary_grid.num_cells)

    def bc_values_saturation(
        self, phase: ppc.Phase, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        if phase.name == 'liq':
            return np.ones(boundary_grid.num_cells)
        else:
            return np.zeros(boundary_grid.num_cells)

    def bc_values_relative_fraction(
        self, component: ppc.Component, phase: ppc.Phase, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:

        z = 0.2
        if component.name == 'H2O':
            return (1 - z) * np.ones(boundary_grid.num_cells)
        else:
            return z * np.ones(boundary_grid.num_cells)

class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def intial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p = 10.0e6
        return np.ones(sd.num_cells) * p

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        T = 500.0
        return np.ones(sd.num_cells) * T

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 1.0e5
        return np.ones(sd.num_cells)  * h

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.1
        if component.name == 'H2O':
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)


def gas_saturation_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros(nc)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))

    return vals, diffs


def temperature_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(h)/200.0
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.ones((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0/200.0
    return vals, diffs

def H2O_liq_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(1-z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -1.0
    return vals, diffs

def NaCl_liq_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(z_NaCl)
    # row-wise storage of derivatives, (4, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = +1.0
    return vals, diffs

def H2O_gas_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(1-z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -1.0
    return vals, diffs

def NaCl_gas_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = +1.0
    return vals, diffs

chi_functions_map = {'H2O_liq': H2O_liq_func,'NaCl_liq': NaCl_liq_func,'H2O_gas': H2O_gas_func,'NaCl_gas': NaCl_gas_func}

class SecondaryEquations(SecondaryEquationsMixin):
    """Mixin to provide expressions for dangling variables.

    The CF framework has the following quantities always as independent variables:

    - independent phase saturations
    - partial fractions (independent since no equilibrium)
    - temperature (needs to be expressed through primary variables in this model, since
      no p-h equilibrium)

    """

    dependencies_of_phase_properties: Sequence[
        Callable[[pp.GridLikeSequence], pp.ad.Operator]
    ]
    """Defined in the Brine mixture mixin."""

    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    def set_equations(self) -> None:
        subdomains = self.mdg.subdomains()

        ### Providing constitutive law for gas saturation based on correlation
        rphase = self.fluid_mixture.reference_phase  # liquid phase
        # gas phase is independent
        independent_phases = [p for p in self.fluid_mixture.phases if p != rphase]

        for phase in independent_phases:
            self.eliminate_by_constitutive_law(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdoains
                gas_saturation_func,  # numerical function implementing correlation
                subdomains,  # all subdomains on which to eliminate s_gas
                # dofs = {'cells': 1},  # default value
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid_mixture.phases:
            for comp in phase:
                self.eliminate_by_constitutive_law(
                    phase.partial_fraction_of[comp],
                    self.dependencies_of_phase_properties(phase),
                    chi_functions_map[comp.name +  "_" + phase.name],
                    subdomains,
                )

        ### Provide constitutive law for temperature
        self.eliminate_by_constitutive_law(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            temperature_func,
            subdomains,
        )


class ModelEquations(
    PrimaryEquationsCF,
    SecondaryEquations,
):
    """Collecting primary flow and transport equations, and secondary equations
    which provide substitutions for independent saturations and partial fractions.
    """

    def set_equations(self):
        """Call to the equation. Parent classes don't use super(). User must provide
        proper order resultion.

        I don't know why, but the other models are doing it this way was well.
        Maybe it has something to do with the sparsity pattern.

        """
        # Flow and transport in MD setting
        PrimaryEquationsCF.set_equations(self)
        # local elimination of dangling secondary variables
        SecondaryEquations.set_equations(self)

class DriesnerBrineFlowModel(
    ModelGeometry,
    H20_NaCl_brine,
    InitialConditions,
    BoundaryConditions,
    ModelEquations,
    CFModelMixin,
):
    """Model assembly. For more details see class CompositionalFlow."""

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu(self.primary_variable_names(), time_dependent=True)

    def after_simulation(self):
        self.exporter.write_pvd()

    # def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:


day = 86400
time_manager = pp.TimeManager(
    schedule=[0, 0.1 * day],
    dt_init=0.01 * day,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

# Model setup:
# eliminate reference phase fractions  and reference component.\
# self.solid.thermal_conductivity(), "solid_thermal_conductivity"
solid_constants = pp.SolidConstants({"permeability": 1.0e-12, "porosity": 0.25, "thermal_conductivity": 100000.0})
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation":  False,
    "reduced_system_q": False,
    'nl_convergence_tol': 1.0e-3,
}

model = DriesnerBrineFlowModel(params)

model.prepare_simulation()
# print geometry
model.exporter.write_vtu()

pp.run_time_dependent_model(model, params)
# pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)
# sd, data = model.mdg.subdomains(True,2)[0]
# print(data['time_step_solutions']['pressure'])
# print(data['time_step_solutions']['enthalpy'])
# # print(data['time_step_solutions']['temperature'])
# # print(data['time_step_solutions']['z_NaCl'])
# res_at_final_time = model.equation_system.assemble(evaluate_jacobian=False)
# print('residual norm: ', np.linalg.norm(res_at_final_time))


