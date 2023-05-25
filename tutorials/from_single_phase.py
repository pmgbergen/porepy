import porepy as pp
import numpy as np
from porepy.models.fluid_mass_balance import SinglePhaseFlow

from porepy.applications.md_grids.domains import nd_cube_domain

from porepy.models.fluid_mass_balance import *

import os
import sys
import pdb

os.system("clear")


class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = 2 / self.units.m
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1 = pp.LineFracture(np.array([[0.2, 1.8], [0.2, 1.8]]) / self.units.m)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        """Choosing the grid type for our domain.

        As we have a diagonal fracture we cannot use a cartesian grid.
        Cartesian grid is the default grid type, and we therefore override this method to assign simplex instead.

        """
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation.

        Here we determine the cell size.

        """
        mesh_args: dict[str, float] = {"cell_size": 10 / self.units.m}
        return mesh_args


class ModifiedBC:
    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.west + bounds.east, "dir")
        return bc

    def bc_values_darcy(self, subdomains: list[pp.Grid]):
        """Zero bc value on top and bottom, 5 on west side, 2 on east side."""
        # Define boundary regions
        values = []
        for sd in subdomains:
            bounds = self.domain_boundary_sides(sd)
            val_loc = np.zeros(sd.num_faces)
            # See section on scaling for explanation of the conversion.
            val_loc[bounds.west] = self.fluid.convert_units(5, "Pa")
            val_loc[bounds.east] = self.fluid.convert_units(2, "Pa")
            values.append(val_loc)
        return pp.wrap_as_ad_array(np.hstack(values), name="bc_values_darcy")


class MySolutionStrategySinglePhaseFlow(SolutionStrategySinglePhaseFlow):
    """ """

    def prepare_simulation(self) -> None:
        """ """

        self.set_geometry()

        self.initialize_data_saving()

        self.set_equation_system_manager()
        self.set_materials()

        self.create_variables()
        self.initial_condition()
        self.reset_state_from_file()
        self.set_equations()

        # if you want to add some keys in the data dict associated to sd, you can do it here

        self.set_discretization_parameters()
        self.discretize()

        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        self.save_data_time_step()

    def before_nonlinear_iteration(self):
        """
        Evaluate Darcy flux for each subdomain and interface and store in the data
        dictionary for use in upstream weighting.

        """

        # Update parameters *before* the discretization matrices are re-computed.
        for sd, data in self.mdg.subdomains(return_data=True):
            vals = self.darcy_flux([sd]).evaluate(self.equation_system).val
            data[pp.PARAMETERS][self.mobility_keyword].update(
                {"darcy_flux": vals}
            )  # EB: dictionary["key"] = val?

        for sd, data in self.mdg.subdomains(return_data=True):
            pressure_array = self.pressure([sd]).evaluate(self.equation_system).val
            data["TMP_pressure"] = pressure_array

            # hu(pressure, saturation, sd) # it will save the data in the dictionary
            # use a modified version of wrapper and MergedOperator

        for intf, data in self.mdg.interfaces(return_data=True, codim=1):
            vals = self.interface_darcy_flux([intf]).evaluate(self.equation_system).val
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})
        for intf, data in self.mdg.interfaces(return_data=True, codim=2):
            vals = self.well_flux([intf]).evaluate(self.equation_system).val
            data[pp.PARAMETERS][self.mobility_keyword].update({"darcy_flux": vals})

        super().before_nonlinear_iteration()


class MySinglePhaseFlow(  # type: ignore[misc
    MassBalanceEquations,
    VariablesSinglePhaseFlow,
    ConstitutiveLawsSinglePhaseFlow,
    BoundaryConditionsSinglePhaseFlow,
    MySolutionStrategySinglePhaseFlow,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """ """


class SinglePhaseFlowGeometry(ModifiedGeometry, ModifiedBC, MySinglePhaseFlow):
    """ """

    ...


np.set_printoptions(precision=3, threshold=sys.maxsize, linewidth=200)
params = {"PROVA_KEY": "PROVA_VAL"}
model = SinglePhaseFlowGeometry(params)

# for sd, data in model.mdg.subdomains(return_data=True):
#     print("sd = ", sd)
#     print("data = ", data)
# theere aren't any data, there is not even a mdg... so data are created and added in run_time_...
# pdb.set_trace()


pp.run_time_dependent_model(model, params)


print("\n\n model.mdg = ", model.mdg)
pdb.set_trace()


# print("\n\n model.fluid = ", model.fluid)
# print("\n\n type(model.fluid) = ", type(model.fluid))

# pdb.set_trace()


for sd, data in model.mdg.subdomains(return_data=True):
    print("sd = ", sd)
    print("data = ", data)

for i in dir(model):
    print(i)

pp.plot_grid(model.mdg, "pressure")

pdb.set_trace()

print("\n\nDone!")
