"""
    Example of single-phase flow with fracture network
    Copyright (C) 2023  Fabio Durastante

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


This module contains an implementation of a 2D single-phase flow problem with a network of 10 fractures,
different normal/tangential permeabilities and material properties.

Run an example by doing:

python3 singlephase_flow.py -c 0.05

"""
import porepy as pp
import numpy as np
import matplotlib.pyplot as plt
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.fluid_mass_balance import BoundaryConditionsSinglePhaseFlow
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--cell-size", type=float, help="Set the maximum size of the cell for simplex grid")
args = argParser.parse_args()

class ModifiedGeometry:
    def set_domain(self) -> None:
        """Defining a two-dimensional square domain with sidelength 2."""
        size = self.solid.convert_units(1, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Setting a network of fractures"""
        # Define each individual fracture, collect into a list. This is a network made of 10 fractures, some
        # of them will be highly permeable, some other will not.
        frac1 = pp.LineFracture(self.solid.convert_units(np.array([[0.0500, 0.2200],[0.4160, 0.0624]]),"m"))
        frac2 = pp.LineFracture(self.solid.convert_units(np.array([[0.0500, 0.2500],[0.2750, 0.1350]]),"m"))
        frac3 = pp.LineFracture(self.solid.convert_units(np.array([[0.1500, 0.6300],[0.4500, 0.0900]]),"m"))
        frac4 = pp.LineFracture(self.solid.convert_units(np.array([[0.1500, 0.4000],[0.9167, 0.5000]]),"m"))
        frac5 = pp.LineFracture(self.solid.convert_units(np.array([[0.6500, 0.849723],[0.8333, 0.167625]]),"m"))
        frac6 = pp.LineFracture(self.solid.convert_units(np.array([[0.7000, 0.849723],[0.2350, 0.167625]]),"m"))
        frac7 = pp.LineFracture(self.solid.convert_units(np.array([[0.6000, 0.8500],[0.3800, 0.2675]]),"m"))
        frac8 = pp.LineFracture(self.solid.convert_units(np.array([[0.3500, 0.8000],[0.9714, 0.7143]]),"m"))
        frac9 = pp.LineFracture(self.solid.convert_units(np.array([[0.7500, 0.9500],[0.9574, 0.8155]]),"m"))
        frac10 = pp.LineFracture(self.solid.convert_units(np.array([[0.1500, 0.4000],[0.8363, 0.9727]]),"m"))
        fractures = [frac1, frac2, frac3, frac4, frac5, frac6, frac7, frac8, frac9, frac10]
        # Define a fracture network in 2d
        self._fractures = fractures
        
    def grid_type(self) -> str:
        """Choosing the grid type for our domain.

        As we have a set of non-oriented fracture we cannot use a cartesian grid. We override this method 
        to assign a simplex grid instead.

        """
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        """Meshing arguments for md-grid creation.

        Here we determine the cell size. This is passed to the script on launch.

        """
        cell_size = self.solid.convert_units(args.cell_size, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args
        
class ModifiedBC(BoundaryConditionsSinglePhaseFlow):
    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Assign dirichlet to the west and east boundaries. The rest are Neumann by default."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.west + bounds.east, "dir")
        return bc

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Zero bc value on top and bottom, 4 on west side, 1 on east side."""
        bounds = self.domain_boundary_sides(boundary_grid)
        values = np.zeros(boundary_grid.num_cells)
        # See section on scaling for explanation of the conversion.
        values[bounds.west] = self.fluid.convert_units(4, "Pa")
        values[bounds.east] = self.fluid.convert_units(1, "Pa")
        return values

class ConstantPermeability:
    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        permvec = self.params['permeability_vector']
        size = sum(sd.num_cells for sd in subdomains)
        if len(subdomains) > 0:
            for sd in subdomains:
                if sd.frac_num >= 0:
                    # print("On fracture "+str(sd.frac_num)+" given permeability "+str(permvec[sd.frac_num]))
                    permeability = pp.wrap_as_dense_ad_array(
                        permvec[sd.frac_num], size, name="permeability"
                    )
                else:
                    # print("On fracture "+str(sd.frac_num)+" given permeability "+str(self.solid.permeability()))
                    permeability = pp.wrap_as_dense_ad_array(
                        self.solid.permeability(), size, name="permeability")
        else:
            permeability = pp.wrap_as_dense_ad_array(
               self.solid.permeability(), size, name="permeability"
           )
        return permeability
    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        # temporary storage for the values
        tmp_array = []
        HIGH_PERM_VALUE = self.params['HIGH_PERM_VALUE']
        LOW_PERM_VALUE = self.params['LOW_PERM_VALUE']
        list_of_fracture_indexes_with_high_permeability = self.params['highperm_fractures']
        for intf in interfaces:
        # Make a numpy array of ones, one element per cell in the interface grid
            e = np.ones(intf.num_cells) 
        # Get hold of the fracture subdomain
            sd_high, sd_low = self.mdg.interface_to_subdomain_pair(intf)
            if intf.dim == 1:
                if sd_low.frac_num in list_of_fracture_indexes_with_high_permeability:
                    # Scale the array of ones with the right permeability, append it to the list
                    tmp_array.append(HIGH_PERM_VALUE[0] * e)
                else:
                    # print("On high permeability fracture "+str(sd_low.frac_num)+" given value "+str(LOW_PERM_VALUE[0]))
                    tmp_array.append(LOW_PERM_VALUE[0] * e)
            else:  # 0d-1d coupling
                if sd_high in list_of_fracture_indexes_with_high_permeability:
                    tmp_array.append(HIGH_PERM_VALUE[1] * e)
                else:
                    tmp_array.append(LOW_PERM_VALUE[1] * e)
        # tmp_array is a list of numpy arrays, but we need to convert it to a single array of type pp.ad.Operator:
        # First convert to a single numpy array, then wrap it into the right format.
        single_array = pp.wrap_as_dense_ad_array(np.hstack(tmp_array), name='normal_permeability_array')
  
        return single_array

class SinglePhaseFlowManyFractures(
    ModifiedGeometry,
    ConstantPermeability,
    ModifiedBC,
    SinglePhaseFlow):
    """Including the fracture network."""
    ...      


fluid_constants = pp.FluidConstants({"viscosity": 1.0, "permeability" : 1.0}) 
solid_constants = pp.SolidConstants({"normal_permeability" : 1.0})
material_constants = {"fluid": fluid_constants, "solid" : solid_constants}
params = {"material_constants": material_constants, 
          "permeability_vector": 2.0*np.array([1,1,1,1e-8,1e-8,1,1,1,1,1]), # multiplication by two to counteract interface law
          "HIGH_PERM_VALUE": np.array([2.0e8,2e4]), 
          "LOW_PERM_VALUE" : np.array([2.0,4/(1e4+1e-4)]),
          "highperm_fractures" : [0, 1, 2, 5, 6, 7, 8, 9] }
model = SinglePhaseFlowManyFractures(params)
pp.run_stationary_model(model,params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), linewidth=0.25, title="Pressure distribution", plot_2d=True)
