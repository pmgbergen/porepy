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

    # def set_fractures(self) -> None:
    #
    #     cross_fractures = np.array([[[0.2, 0.8], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]])
    #     disjoint_set = []
    #     dx = 1.0
    #     for i in range(10):
    #         chunk = cross_fractures.copy()
    #         chunk[:, 0, :] = chunk[:, 0, :] + dx * (i)
    #         disjoint_set.append(chunk[0])
    #         disjoint_set.append(chunk[1])
    #
    #     disjoint_fractures = [
    #         pp.LineFracture(self.solid.convert_units(fracture_pts, "m"))
    #         for fracture_pts in disjoint_set
    #     ]
    #     self._fractures = disjoint_fractures

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.5, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class BoundaryConditions(BoundaryConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        all, east, west, north, south, top, bottom = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, east + west, "dir")

    def bc_type_advective_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        all, east, west, north, south, top, bottom = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, east + west, "neu")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        all, east, west, north, south, top, bottom = self.domain_boundary_sides(
            boundary_grid
        )
        p_inlet = 20.0e6
        p_outlet = 15.0e6
        xcs = boundary_grid.cell_centers.T
        l = 10.0

        def p_D(xc):
            x, y, z = xc
            return p_inlet * (1 - x / l) + p_outlet * (x / l)

        p_D_iter = map(p_D, xcs)
        p_D_vals = np.fromiter(p_D_iter, dtype=float)
        return p_D_vals

    def bc_values_enthalpy(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        all, east, west, north, south, top, bottom = self.domain_boundary_sides(
            boundary_grid
        )
        h_init = 2.5e6
        h_inlet = 2.5e6
        h = h_init * np.ones(boundary_grid.num_cells)
        h[west] = h_inlet
        return h

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        all, east, west, north, south, top, bottom = self.domain_boundary_sides(
            boundary_grid
        )
        z_init = 0.01
        z_inlet = 0.2  # 0.5e-2
        if component.name == "H2O":
            z_H2O = (1 - z_init) * np.ones(boundary_grid.num_cells)
            z_H2O[west] = 1 - z_inlet
            return z_H2O
        else:
            z_NaCl = z_init * np.ones(boundary_grid.num_cells)
            z_NaCl[west] = z_inlet
            return z_NaCl

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # # adhoc functional programming for BC consistency
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p = self.bc_values_pressure(boundary_grid)
        h = self.bc_values_enthalpy(boundary_grid)
        z_NaCl = 0.1 * np.ones_like(p)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)
        T = self.obl.sampled_could.point_data["Temperature"]
        return T


class InitialConditions(InitialConditionsCF):
    """See parent class how to set up BC. Default is all zero and Dirichlet."""

    def intial_pressure(self, sd: pp.Grid) -> np.ndarray:
        p = 20.0e6
        return np.ones(sd.num_cells) * p

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        # adhoc functional programming for IC consistency
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p = self.intial_pressure(sd)
        h = self.initial_enthalpy(sd)
        z_NaCl = 0.1 * np.ones_like(p)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)
        T = self.obl.sampled_could.point_data["Temperature"]
        return T

    def initial_enthalpy(self, sd: pp.Grid) -> np.ndarray:
        h = 2.5e6
        return np.ones(sd.num_cells) * h

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        z = 0.1
        if component.name == "H2O":
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

    def relative_permeability(self, saturation: pp.ad.Operator) -> pp.ad.Operator:
        # return saturation**2
        return saturation

    @property
    def obl(self):
        return self._obl

    @obl.setter
    def obl(self, obl):
        self._obl = obl


day = 86400
t_scale = 0.000001
time_manager = pp.TimeManager(
    schedule=[0, 1.0 * day * t_scale],
    dt_init=1.0 * day * t_scale,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

# Model setup:
# eliminate reference phase fractions  and reference component.\
# self.solid.thermal_conductivity(), "solid_thermal_conductivity"
solid_constants = pp.SolidConstants(
    {"permeability": 9.869233e-14, "porosity": 0.2, "thermal_conductivity": 1000.0*1.92}
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,  # s_liq eliminated, default is True
    "eliminate_reference_component": True,  # z_H2O eliminated, default is True
    "time_manager": time_manager,
    "prepare_simulation": False,
    "reduce_linear_system_q": False,
    "nl_convergence_tol": 1.0e-4,
    "max_iterations": 50,
}

model = DriesnerBrineFlowModel(params)
file_name = "binary_files/PHX_l0_with_gradients.vtk"
brine_obl = DriesnerBrineOBL(file_name)
model.obl = brine_obl

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)

# # print geometry
# model.exporter.write_vtu()
tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)

