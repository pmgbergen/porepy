from abc import abstractmethod
from typing import cast

import numpy as np

import porepy as pp
from porepy.applications.md_grids.mdg_library import benchmark_3d_case_3
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.models.geometry import ModelGeometry


class Geometry(ModelGeometry):

    @abstractmethod
    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        pass

    @staticmethod
    def harvest_sphere_members(xc, rc, x):
        dx = x - xc
        r = np.linalg.norm(dx, axis=1)
        return np.where(r < rc, True, False)


class Benchmark2DC1(Geometry):

    def set_fractures(self) -> None:
        self._fractures = pp.applications.md_grids.fracture_sets.benchmark_2d_case_1()

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.1, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:

        if isinstance(sd, pp.Grid):
            x = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            x = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        sides = self.domain_boundary_sides(sd)
        idx = sides.all_bf

        rc = 0.1
        xc = np.array([0.0, 0.0, 0.0])
        logical = Geometry.harvest_sphere_members(xc, rc, x[idx])
        inlet_facets = idx[logical]

        rc = 0.1
        xc = np.array([1.0, 1.0, 0.0])
        logical = Geometry.harvest_sphere_members(xc, rc, x[idx])
        outlet_facets = idx[logical]

        return inlet_facets, outlet_facets


class Benchmark2DC3(Geometry):

    def set_fractures(self) -> None:
        self._fractures = pp.applications.md_grids.fracture_sets.benchmark_2d_case_3()

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.1, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:

        if isinstance(sd, pp.Grid):
            x = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            x = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        sides = self.domain_boundary_sides(sd)
        idx = sides.all_bf

        rc = 0.1
        xc = np.array([0.0, 0.0, 0.0])
        logical = Geometry.harvest_sphere_members(xc, rc, x[idx])
        inlet_facets = idx[logical]

        rc = 0.1
        xc = np.array([1.0, 1.0, 0.0])
        logical = Geometry.harvest_sphere_members(xc, rc, x[idx])
        outlet_facets = idx[logical]

        return inlet_facets, outlet_facets


class Benchmark3DC3(Geometry):

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

    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:

        if isinstance(sd, pp.Grid):
            x = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            x = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        sides = self.domain_boundary_sides(sd)
        idx = sides.all_bf

        rc = 0.1
        xc = np.array([0.0, 0.0, 0.0])
        logical = Geometry.harvest_sphere_members(xc, rc, x[idx])
        inlet_facets = idx[logical]

        rc = 0.1
        xc = np.array([1.0, 2.25, 1.0])
        logical = Geometry.harvest_sphere_members(xc, rc, x[idx])
        outlet_facets = idx[logical]

        return inlet_facets, outlet_facets


class SimpleGeometry(Geometry):

    def set_domain(self) -> None:

        dimension = 2
        size_x = self.solid.convert_units(10.0, "m")
        size_y = self.solid.convert_units(10.0, "m")
        size_z = self.solid.convert_units(1.0, "m")
        box: dict[str, pp.number] = {"xmax": size_x}
        if dimension > 1:
            box.update({"ymax": size_y})
        if dimension > 2:
            box.update({"zmax": size_z})
        self._domain = pp.Domain(box)

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:

        if isinstance(sd, pp.Grid):
            x = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            x = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        sides = self.domain_boundary_sides(sd)
        idx = sides.all_bf

        z_level = 0.0
        if self._domain.dim == 3:
            z_level = 0.5

        rc = 0.25
        xc = np.array([0.0, 0.5, z_level])
        logical = Geometry.harvest_sphere_members(xc, rc, x[idx])
        inlet_facets = idx[logical]

        rc = 0.25
        xc = np.array([10.0, 9.5, z_level])
        logical = Geometry.harvest_sphere_members(xc, rc, x[idx])
        outlet_facets = idx[logical]

        return inlet_facets, outlet_facets
