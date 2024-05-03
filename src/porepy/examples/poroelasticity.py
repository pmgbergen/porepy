import os
import resource
import time

import numpy as np

import porepy as pp
from porepy.examples.mandel_biot import (
    MandelSetup,
    mandel_fluid_constants,
    mandel_solid_constants,
)

N_THREADS = "1"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

material_constants = {
    "solid": pp.SolidConstants(mandel_solid_constants),
    "fluid": pp.FluidConstants(mandel_fluid_constants),
}

scaling = {"m": 1e-3}
units = pp.Units(**scaling)
time_manager = pp.TimeManager(
    schedule=[0, 2e1, 1e2, 1e3, 5e3, 1e4],  # [s]
    dt_init=20,  # [s]
    constant_dt=True,  # [s]
)

dimension = 3
cell_size = 2.125 * 0.3125

ls = 1 / units.m  # length scaling

if dimension == 2:
    mesh_arguments = {"cell_size": cell_size * ls}
elif dimension == 3:
    mesh_arguments = {"cell_size": cell_size * ls}
else:
    raise ValueError("Case not implemented.")

params = {
    "material_constants": material_constants,
    "meshing_arguments": mesh_arguments,
    "time_manager": time_manager,
    "plot_results": True,
    "prepare_simulation": False,
    "units": units,
}


class MandelSetupCartesian(MandelSetup):

    def set_domain(self) -> None:
        """Set the domain."""
        ls = self.solid.convert_units(1, "m")  # length scaling
        if dimension == 2:
            a, b = self.params.get("domain_size", (10, 10))  # [m]
            domain = pp.Domain(
                {"xmin": 0.0, "xmax": a * ls, "ymin": 0.0, "ymax": b * ls}
            )
        elif dimension == 3:
            a, b, c = self.params.get("domain_size", (10, 10, 10))  # [m]
            domain = pp.Domain(
                {
                    "xmin": 0.0,
                    "xmax": a * ls,
                    "ymin": 0.0,
                    "ymax": b * ls,
                    "zmin": 0.0,
                    "zmax": c * ls,
                }
            )
        else:
            raise ValueError("Case not implemented.")
        self._domain = domain

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid):
        return np.zeros((self.nd, boundary_grid.num_cells)).ravel("F")

    def bc_type_mechanics(self, sd: pp.Grid):
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        return bc


memory_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
setup = MandelSetupCartesian(params=params)
tb = time.time()
setup.prepare_simulation()
te = time.time()
print("Time prepare_simulation(): ", te - tb, "[seconds]")
data = setup.mdg.subdomains(return_data=True, dim=dimension)
# def count_nnz(data):
#     n_sparse_arrays = 0
#     total_nnz = 0
#     for obj in data[0][1]["discretization_matrices"].values():
#         if isinstance(obj,dict):
#             for key, mat in obj.items():
#                 # print("key: ", key)
#                 n_sparse_arrays += 1
#                 total_nnz += mat.nnz
#     return total_nnz, n_sparse_arrays
# total_nnz, n_sparse_arrays = count_nnz(data)
# print(
#     "Memory used:",
#     resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - memory_start, " [bytes]"
# )
print("Number of DoF: ", setup.equation_system.num_dofs())
# print("Number of discretization matrices: ", n_sparse_arrays)
# print("Number of nonzeros (nnz) of discretization matrices: ", total_nnz)
