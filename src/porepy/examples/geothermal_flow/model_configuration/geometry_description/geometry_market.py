from abc import abstractmethod
from typing import Literal, cast, Sequence

import numpy as np

import porepy as pp
from porepy.applications.md_grids.mdg_library import benchmark_3d_case_3
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.wells_3d import _add_interface
import scipy.sparse as sps


class Geometry(pp.PorePyModel):
    @abstractmethod
    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def harvest_sphere_members(xc, rc, x):
        dx = x - xc
        r = np.linalg.norm(dx, axis=1)
        return np.where(r < rc, True, False)


class Benchmark2DC1(Geometry):
    def set_fractures(self) -> None:
        self._fractures = pp.applications.md_grids.fracture_sets.benchmark_2d_case_1()

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.1, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(0.1, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
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
        size_x = self.units.convert_units(10.0, "m")
        size_y = self.units.convert_units(10.0, "m")
        size_z = self.units.convert_units(1.0, "m")
        box: dict[str, pp.number] = {"xmax": size_x}
        if dimension > 1:
            box.update({"ymax": size_y})
        if dimension > 2:
            box.update({"zmax": size_z})
        self._domain = pp.Domain(box)

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
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


class SimpleGeometryHorizontal(Geometry):
    """A class to represent a simple 1D geometry for a simulation domain.
    The start of domain serve as inlet and end of domain serves as the outlet
    """

    _dist_from_ref_point: float = 5.0
    _inlet_centre: np.ndarray = np.array([0.0, 5.0, 0.0])
    _outlet_centre: np.ndarray = np.array([2000.0, 5.0, 0.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(2000.0, "m")
        y_length = self.units.convert_units(10.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(10.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._dist_from_ref_point, face_centers[bf_indices]
            )
            return bf_indices[logical]

        inlet_facets = find_facets(self._inlet_centre)
        outlet_facets = find_facets(self._outlet_centre)
        return inlet_facets, outlet_facets


class SimpleGeometryVertical(Geometry):
    """A class to represent a simple 1D geometry for a simulation domain.
    The start of domain serve as inlet and end of domain serves as the outlet
    """

    _dist_from_ref_point: float = 5.0
    _inlet_centre: np.ndarray = np.array([5.0, 0.0, 0.0])
    _outlet_centre: np.ndarray = np.array([5.0, 2000.0, 0.0])

    def set_domain(self) -> None:
        x_length = self.units.convert_units(10.0, "m")
        y_length = self.units.convert_units(2000.0, "m")
        box: dict[str, pp.number] = {"xmax": x_length, "ymax": y_length}
        self._domain = pp.Domain(box)

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(10.0, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args

    def get_inlet_outlet_sides(
        self, sd: pp.Grid | pp.BoundaryGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def find_facets(center: np.ndarray) -> np.ndarray:
            logical = Geometry.harvest_sphere_members(
                center, self._dist_from_ref_point, face_centers[bf_indices]
            )
            return bf_indices[logical]

        inlet_facets = find_facets(self._inlet_centre)
        outlet_facets = find_facets(self._outlet_centre)
        return inlet_facets, outlet_facets


class SimpleGeometry2D(Geometry):
    """A class to represent a simple 1D geometry for a simulation domain. 
       The start of domain serve as inlet and end of domain serves as the outlet
    """
    _dist_from_ref_point: float = 5.0                           
    _inlet_coverage: tuple[np.ndarray, np.ndarray] = (
        np.array([4000.0, 0.0, 0.0]),
        np.array([5000.0, 0.0, 0.0])
    )
    _outlet_coverage: tuple[np.ndarray, np.ndarray] = (
        np.array([0.0, 3000.0, 0.0]),
        np.array([9000.0, 3000.0, 0.0])
    )
    _domain_x_length: float = 9000.0
    _domain_y_length: float = 3000.0

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length_in_m,
            "ymax": y_length_in_m
        }
        self._domain = pp.Domain(box)
 
    def set_fractures(self) -> None:
        """Define two diagonal fractures crossing at the center.
        """
        frac1 = pp.LineFracture(np.array([
            [4500.0, self._domain_x_length-4500.0],
            [50.0, self._domain_y_length-100.0]
        ]))
        self._fractures = [frac1]
        # frac1 = pp.LineFracture(np.array([
        #     [5.0, self._domain_x_length-5.0],
        #     [5.0, self._domain_y_length-5.0]
        # ]))

        # frac2 = pp.LineFracture(np.array([
        #     [5.0, self._domain_x_length-5],
        #     [self._domain_y_length-5, 5.0]
        # ]))
        # self._fractures = [frac1, frac2]
        
    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")
    
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(100.0, "m")
        frac_cell_size = self.units.convert_units(100.0, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": frac_cell_size
        }
        return mesh_args

    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf
    
        def find_facets_within_coverage(coverage: tuple[np.ndarray, np.ndarray])-> np.ndarray:
            start, end = coverage
            logical_x = (face_centers[bf_indices][:, 0] >= start[0]) & (face_centers[bf_indices][:, 0] <= end[0])
            logical_y = (face_centers[bf_indices][:, 1] >= start[1]) & (face_centers[bf_indices][:, 1] <= end[1])
            return bf_indices[logical_x & logical_y]
        
        inlet_facets = find_facets_within_coverage(self._inlet_coverage)
        outlet_facets = find_facets_within_coverage(self._outlet_coverage)
        return inlet_facets, outlet_facets
    

class SimpleGeometryWellsFracture2D_Old(Geometry):
    """A class to represent a simple 1D geometry for a simulation domain. 
       The start of domain serve as inlet and end of domain serves as the outlet
    """
    _dist_from_ref_point: float = 5.0                           
    _inlet_coverage: tuple[np.ndarray, np.ndarray] = (
        np.array([4000.0, 0.0, 0.0]),
        np.array([5000.0, 0.0, 0.0])
    )
    _outlet_coverage: tuple[np.ndarray, np.ndarray] = (
        np.array([0.0, 1000.0, 0.0]),
        np.array([9000.0, 1000.0, 0.0])
    )
    _domain_x_length: float = 9000.0
    _domain_y_length: float = 1000.0

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length_in_m,
            "ymax": y_length_in_m
        }
        self._domain = pp.Domain(box)
 
    def set_fractures(self) -> None:
        """Define two diagonal fractures crossing at the center.
        """
        frac1 = pp.LineFracture(np.array([
            [4500.0, self._domain_x_length-4500.0],
            [50.0, self._domain_y_length-100.0]
        ]))
        self._fractures = [frac1]
        
    def grid_type(self) -> str:
        return self.params.get("grid_type", "cartesian")
    
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(100.0, "m")
        frac_cell_size = self.units.convert_units(100.0, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": frac_cell_size
        }
        return mesh_args

    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf
    
        def find_facets_within_coverage(coverage: tuple[np.ndarray, np.ndarray])-> np.ndarray:
            start, end = coverage
            logical_x = (face_centers[bf_indices][:, 0] >= start[0]) & (face_centers[bf_indices][:, 0] <= end[0])
            logical_y = (face_centers[bf_indices][:, 1] >= start[1]) & (face_centers[bf_indices][:, 1] <= end[1])
            return bf_indices[logical_x & logical_y]
        
        inlet_facets = find_facets_within_coverage(self._inlet_coverage)
        outlet_facets = find_facets_within_coverage(self._outlet_coverage)
        return inlet_facets, outlet_facets


class DomainPointGridWellsFracture2D(Geometry):
    """A class to represent a simple 2D geometry for a simulation domain. 

    """
    _dist_from_ref_point: float = 5.0                           
    _inlet_coverage: tuple[np.ndarray, np.ndarray] = (
        np.array([4000.0, 0.0, 0.0]),
        np.array([5000.0, 0.0, 0.0])
    )
    _outlet_coverage: tuple[np.ndarray, np.ndarray] = (
        np.array([0.0, 1000.0, 0.0]),
        np.array([3000.0, 1000.0, 0.0])
    )
    _injection_points: list[np.ndarray] = [np.array([15.0, 15.0])]
    _production_points: list[np.ndarray] = [np.array([85.0, 15.0])]

    _domain_x_length: float = 100.0
    _domain_y_length: float = 30.0

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length_in_m,
            "ymax": y_length_in_m
        }
        self._domain = pp.Domain(box)
 
    def set_fractures(self) -> None:
        """Define two diagonal fractures crossing at the center.
        """
        x_start = self._domain_x_length / 2.0
        x_end = self._domain_x_length - x_start
        y_margin = 0.1 * self._domain_y_length
        frac1 = pp.LineFracture(np.array([
            [x_start, x_end],
            [y_margin, self._domain_y_length - y_margin]
        ]))
        self._fractures = [frac1]
        
    def grid_type(self) -> str:
        # return self.params.get("grid_type", "simplex")
        return self.params.get("grid_type", "cartesian")
    
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(1.0, "m")
        frac_cell_size = self.units.convert_units(1.0, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": frac_cell_size
        }
        return mesh_args
    
    def set_geometry(self):
        """Create the injection and production wells."""

        super().set_geometry()
        for i, injection_point in enumerate(self._injection_points):
            self._add_well(injection_point, i, "injection")

        for i, production_point in enumerate(self._production_points):
            self._add_well(production_point, i, "production")

    def _add_well(
        self,
        point: np.ndarray,
        well_index: int,
        well_type: Literal["injection", "production"],
    ) -> None:
        """Insert single well as point grid and connect to matrix."""

        # Convert to 3D coordinates (for porepy PointGrid)
        p = np.zeros(3)
        p[:2] = point

        # Create 0D grid
        sd_0d = pp.PointGrid(p)
        sd_0d.tags[f"{well_type}_well"] = well_index
        sd_0d.compute_geometry()

        # This object must have been passed or prepared by a mixin
        self.mdg.add_subdomains(sd_0d)

        # Couple well to the matrix
        matrix = self.mdg.subdomains(dim=self.domain.dim)[0]
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )
        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)

    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        if isinstance(sd, pp.Grid):
            face_centers = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            face_centers = sd.cell_centers.T
        else:
            raise ValueError("Type not expected.")
        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf
    
        def find_facets_within_coverage(coverage: tuple[np.ndarray, np.ndarray])-> np.ndarray:
            start, end = coverage
            logical_x = (face_centers[bf_indices][:, 0] >= start[0]) & (face_centers[bf_indices][:, 0] <= end[0])
            logical_y = (face_centers[bf_indices][:, 1] >= start[1]) & (face_centers[bf_indices][:, 1] <= end[1])
            return bf_indices[logical_x & logical_y]
        
        inlet_facets = find_facets_within_coverage(self._inlet_coverage)
        outlet_facets = find_facets_within_coverage(self._outlet_coverage)
        return inlet_facets, outlet_facets
    
    def get_well_cells(self, well_type: str) -> list[pp.Grid]:
        return [sd for sd in self.mdg.subdomains(dim=0) if sd.tags.get(f"{well_type}_well") is not None]
    
    def filter_wells(
        self,
        subdomains: Sequence[pp.Grid],
        well_type: Literal["production", "injection"],
    ) -> tuple[list[pp.Grid], list[pp.Grid]]:
        """Helper method to return the partitioning of subdomains into wells of defined
        ``well_type`` and other grids.

        Parameters:
            subdomains: A list of subdomains.
            well_type: Well type to filter out (injector or producer).

        Returns:
            A  2-tuple containing
            1. All 0D grids tagged as wells of type ``well_type``.
            2. All other grids found in ``subdomains``.

        """
        tag = f"{well_type}_well"
        wells = [sd for sd in subdomains if sd.dim == 0 and tag in sd.tags]
        other_sds = [sd for sd in subdomains if sd not in wells]
        return wells, other_sds

