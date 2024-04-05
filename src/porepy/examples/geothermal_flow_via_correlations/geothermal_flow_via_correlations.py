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

import numpy as np
import time
import porepy as pp
import porepy.composite as ppc
import TracerConstitutiveDescription
import BrineConstitutiveDescription

# CompositionalFlow has data savings mixin, composite variables mixin,
# Solution strategy eliminating local equations with Schur complement and no flash.
# It also hase the ConstitutiveLaws for CF, which use the FluidMixture.
# For changing constitutive laws, import ConstitutiveLawsCF and overwrite mixins
from porepy.models.compositional_flow import (
    BoundaryConditionsCF,
    CFModelMixin,
    InitialConditionsCF,
    PrimaryEquationsCF,
)

from DriesnerBrineOBL import DriesnerBrineOBL



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

    def set_fractures(self) -> None:
        frac_1_points = self.solid.convert_units(
            np.array([[0.2, 0.8], [0.2, 0.8]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)

        frac_2_points = self.solid.convert_units(
            np.array([[0.2, 0.8], [0.8, 0.2]]), "m"
        )
        frac_2 = pp.LineFracture(frac_2_points)
        self._fractures = [frac_1, frac_2]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.5, "m")
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
        p_inlet = 15.0e6
        p_outlet = 15.0e6
        xcs = boundary_grid.cell_centers.T
        l = 10.0
        def p_D(xc):
            x, y, z = xc
            return p_inlet * (1 - x/l) + p_outlet * (x/l)
        p_D_iter = map(p_D, xcs)
        vals = np.fromiter(p_D_iter,dtype=float)
        return vals

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        h_inlet = 3.0e6 #(3.0e6 / 573.5)*373.15
        h_outlet = 3.0e6
        xcs = boundary_grid.cell_centers.T
        l = 10.0
        def h_D(xc):
            x, y, z = xc
            return h_inlet * (1 - x/l) + h_outlet * (x/l)
        h_D_iter = map(h_D, xcs)
        vals = np.fromiter(h_D_iter,dtype=float)
        return vals

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        z = 0.1
        if component.name == 'H2O':
            return (1 - z) * np.ones(boundary_grid.num_cells)
        else:
            return z * np.ones(boundary_grid.num_cells)

    def bc_values_saturation(
        self, phase: ppc.Phase, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        # TODO: refactor with functional programming
        # adhoc functional programming for BC consistency
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p = self.bc_values_pressure(boundary_grid)
        h = self.bc_values_enthalpy(boundary_grid)
        z_NaCl = 0.1 * np.ones_like(p)
        par_points = np.array((z_NaCl,h*h_scale,p*p_scale)).T
        self.obl.sample_at(par_points)
        if phase.name == 'liq':
            s_l = self.obl.sampled_could.point_data['S_l']
            return s_l
        else:
            s_v = self.obl.sampled_could.point_data['S_v']
            return s_v

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # adhoc functional programming for BC consistency
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p = self.bc_values_pressure(boundary_grid)
        h = self.bc_values_enthalpy(boundary_grid)
        z_NaCl = 0.1 * np.ones_like(p)
        par_points = np.array((z_NaCl,h*h_scale,p*p_scale)).T
        self.obl.sample_at(par_points)
        T = self.obl.sampled_could.point_data['Temperature']
        return T

    def bc_values_relative_fraction(
        self, component: ppc.Component, phase: ppc.Phase, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        # adhoc functional programming for BC consistency
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p = self.bc_values_pressure(boundary_grid)
        h = self.bc_values_enthalpy(boundary_grid)
        z_NaCl = 0.1 * np.ones_like(p)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)
        if phase.name == 'liq':
            x_l = self.obl.sampled_could.point_data['Xl']
            if component.name == 'H2O':
                return (1 - x_l)
            else:
                return x_l
        else:
            x_v = self.obl.sampled_could.point_data['Xv']
            if component.name == 'H2O':
                return (1 - x_v)
            else:
                return x_v

class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def intial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p = 15.0e6
        return np.ones(sd.num_cells) * p

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        # adhoc functional programming for IC consistency
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p = self.intial_pressure(sd)
        h = self.initial_enthalpy(sd)
        z_NaCl = 0.5e-2 * np.ones_like(p)
        par_points = np.array((z_NaCl,h*h_scale,p*p_scale)).T
        self.obl.sample_at(par_points)
        T = self.obl.sampled_could.point_data['Temperature']
        return T

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 3.0e6
        return np.ones(sd.num_cells)  * h

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.5e-2
        if component.name == 'H2O':
            return (1 - z) * np.ones(sd.num_cells)
        else:
            return z * np.ones(sd.num_cells)

# class SecondaryEquations(TracerConstitutiveDescription.SecondaryEquations):
#     pass

class SecondaryEquations(BrineConstitutiveDescription.SecondaryEquations):
    pass


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
    BrineConstitutiveDescription.FluidMixture,
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

    @property
    def obl(self):
        return self._obl

    @obl.setter
    def obl(self, obl):
        self._obl = obl



day = 86400
time_manager = pp.TimeManager(
    schedule=[0, 0.0001 * day],
    dt_init=0.00001 * day,
    constant_dt=True,
    iter_max=10,
    print_info=True,
)

# Model setup:
# eliminate reference phase fractions  and reference component.\
# self.solid.thermal_conductivity(), "solid_thermal_conductivity"
solid_constants = pp.SolidConstants({"permeability": 1.0e-12, "porosity": 0.25, "thermal_conductivity": 3.0})
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation":  False,
    "reduce_linear_system_q": False,
    'nl_convergence_tol': 1.0e0,
}

model = DriesnerBrineFlowModel(params)
file_name = 'binary_files/PHX_l0_with_gradients.vtk'
brine_obl = DriesnerBrineOBL(file_name)
model.obl = brine_obl

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

# print geometry
model.exporter.write_vtu()

pp.run_time_dependent_model(model, params)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)
# pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)
# sd, data = model.mdg.subdomains(True,2)[0]
# print(data['time_step_solutions']['pressure'])
# print(data['time_step_solutions']['enthalpy'])
# # print(data['time_step_solutions']['temperature'])
# # print(data['time_step_solutions']['z_NaCl'])
# res_at_final_time = model.equation_system.assemble(evaluate_jacobian=False)
# print('residual norm: ', np.linalg.norm(res_at_final_time))


