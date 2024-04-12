import porepy as pp
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, cast
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.applications.md_grids.mdg_library import benchmark_3d_case_3


# class Geometry(ABC):
#
#     @property
#     def file_name(self):
#         return self._file_name
#
#     @file_name.setter
#     def file_name(self, file_name):
#         self._file_name = file_name

class Benchmark3DC3:


    def set_geometry(self) -> None:
        """Create mixed-dimensional grid and fracture network."""

        # Create mixed-dimensional grid and fracture network.
        self.mdg, self.fracture_network = benchmark_3d_case_3(
            refinement_level=self.params.get("refinement_level", 0)
        )
        self.nd: int = self.mdg.dim_max()

        # Obtain domain and fracture list directly from the fracture network.
        self._domain = cast(pp.Domain, self.fracture_network.domain)
        self._fractures = self.fracture_network.fractures

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections.
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)

class SimpleGeometry:



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