from abc import abstractmethod
from typing import Literal, cast, Sequence

import numpy as np

import porepy as pp
from porepy.applications.md_grids.mdg_library import benchmark_3d_case_3
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.wells_3d import _add_interface
import scipy.sparse as sps

from typing import Callable
import gmsh
from porepy.fracs import gmsh_interface as GI


def inner_radius_for_one_cell(coarse: float, f: float = 1.0) -> float:
    return f * coarse * ((3**0.5 / 4) / 3.141592653589793)**0.5  # ≈ 0.371*coarse


import math
import gmsh
from typing import Callable

def install_triangle_bg(
    x0: float,
    y0: float,
    coarse: float,
    fine: float,
    side: float,
    band: float,
    k: float = 25.0,
) -> Callable[[], None]:
    """
    Build a no-arg callback that installs a Gmsh MathEval background size field
    producing triangular refinement around (x0, y0).

    - The centroid triangle has side length = `side` (fixed size region).
    - Cells inside the triangle get mesh size = `side`.
    - Cells in a band of thickness `band` around the triangle get mesh size = `fine`.
    - Cells outside get mesh size = `coarse`.
    """

    def _cb() -> None:
        # Same setup as your annulus version
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # --- 1. Compute vertices of equilateral triangle centered at (x0,y0) ---
        h = math.sqrt(3.0) / 3.0 * side  # centroid → vertex distance
        angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]
        vertices = [(x0 + h*math.cos(a), y0 + h*math.sin(a)) for a in angles]

        # --- 2. Build signed distances to edges ---
        edge_exprs = []
        for i in range(3):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i+1)%3]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            norm = math.sqrt(A*A + B*B)
            expr = f"(({A:.16g})*x + ({B:.16g})*y + ({C:.16g}))/{norm:.16g}"
            edge_exprs.append(expr)

        # --- 3. Inside-triangle selector (≈1 inside, 0 outside) ---
        inside_exprs = [f"1/(1+exp(-{k:.16g}*({e})))" for e in edge_exprs]
        inside_selector = "*".join(inside_exprs)

        # --- 4. Band around triangle edges ---
        band_exprs = [
            f"1/(1+exp(-{k:.16g}*(({band:.16g})-sqrt(({e})*({e})))))"
            for e in edge_exprs
        ]
        band_selector = "*".join(band_exprs)

        # --- 5. Combine selectors ---
        # Priority: fixed size (side) inside triangle, fine in band, coarse outside
        selector = f"{inside_selector} + (1-{inside_selector})*{band_selector}"

        # --- 6. Final target mesh size ---
        expr = (
            f"({coarse:.16g})"
            f"-(({coarse:.16g})-({fine:.16g}))*({selector})"
            f"-(({fine:.16g})-({side:.16g}))*({inside_selector})"
        )

        # --- 7. Install field ---
        gmsh.model.mesh.field.add("MathEval", 303)
        gmsh.model.mesh.field.setString(303, "F", expr)
        gmsh.model.mesh.field.setAsBackgroundMesh(303)

    return _cb


def install_boundary_band_bg(
    boundary_tags: list[int],
    coarse: float,
    fine: float,
    d_in: float,
    d_out: float,
) -> Callable[[], None]:
    """
    Build a callback that installs a Gmsh background size field refining
    a band `d_in < dist < d_out` away from given boundary curves.

    Parameters
    ----------
    boundary_tags : list[int]
        Tags of the boundary curves (e.g. [1,2,3,4] for a square).
    coarse : float
        Element size outside the band.
    fine : float
        Element size inside the band.
    d_in : float
        Inner offset from boundary [m].
    d_out : float
        Outer offset from boundary [m].
    """
    def _cb() -> None:
        # Disable automatic sizing
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Distance field to boundary curves
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", boundary_tags)

        # Threshold field for band refinement
        f_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMin", fine)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMax", coarse)
        gmsh.model.mesh.field.setNumber(f_th, "DistMin", d_in)
        gmsh.model.mesh.field.setNumber(f_th, "DistMax", d_out)

        # Activate as background mesh
        gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

    return _cb


def install_annulus_bg(
    x0: float,
    y0: float,
    coarse: float,
    fine: float,
    radius_in: float,
    radius_out: float,
    k: float = 25.0,
) -> Callable[[], None]:
    """
    Build a no-arg callback that installs a Gmsh MathEval background size field
    producing an annular refinement around (x0, y0): mesh size = `coarse` for
    r ≤ radius_in and r ≥ radius_out, and `fine` for radius_in < r < radius_out.
    Assumes Gmsh is initialized and a model is defined.
    """
    def _cb() -> None:
        # Make the background field authoritative (optional but recommended)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Distance r(x,y) from the well center
        distance_expr = f"sqrt((x-{x0})*(x-{x0}) + (y-{y0})*(y-{y0}))"

        # Smooth steps (logistics) at inner/outer radii
        inner_sigmoid = f"1/(1+exp(-{k}*({distance_expr} - {radius_in})))"
        outer_sigmoid = f"1/(1+exp(-{k}*({distance_expr} - {radius_out})))"

        # Annular selector: ≈1 in (radius_in, radius_out), ≈0 elsewhere
        annular_selector = f"({inner_sigmoid}) * (1 - ({outer_sigmoid}))"

        # Final target size field
        expr = f"{coarse} - ({coarse} - {fine}) * {annular_selector}"

        # Install as background field (use a tag that doesn't collide with yours)
        gmsh.model.mesh.field.add("MathEval", 101)
        gmsh.model.mesh.field.setString(101, "F", " ".join(expr.split()))
        gmsh.model.mesh.field.setAsBackgroundMesh(101)

    return _cb


def install_multi_annuli_bg(
    centers: list[tuple[float, float]],
    coarse: float,
    fine: float,
    radius_in: float,
    radius_out: float,
    k: float = 25.0,          # sharpness of inner/outer ring transitions
    k_union: float = 10.0,    # sharpness for smooth "max" when unioning rings
):
    """
    Build a no-arg callback that installs ONE Gmsh MathEval background size field
    producing annular refinement around all centers:
      - mesh size = `coarse` for r <= radius_in and r >= radius_out (for every center)
      - mesh size = `fine`  for radius_in < r < radius_out (for any center)
    Reuses the same logistic-ring logic you wrote.

    Notes:
      - Uses a smooth max (log-sum-exp) to union multiple ring selectors:
          S_total ≈ max_i S_i = (1/k_union) * log(Σ_i exp(k_union * S_i))
      - Increase k_union for a crisper union; decrease if you want gentler blending.
    """
    def _cb() -> None:
        import gmsh

        # Make the background field authoritative
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        if not centers:
            # Nothing to refine: constant coarse field
            expr = f"{coarse}"
        else:
            ring_exprs = []
            for (x0, y0) in centers:
                r = f"sqrt((x-{x0})*(x-{x0}) + (y-{y0})*(y-{y0}))"
                inner = f"1/(1+exp(-{k}*({r} - {radius_in})))"
                outer = f"1/(1+exp(-{k}*({r} - {radius_out})))"
                # Your annulus selector: ≈1 in (radius_in, radius_out), ≈0 elsewhere
                S = f"({inner})*(1-({outer}))"
                ring_exprs.append(S)

            if len(ring_exprs) == 1:
                S_total = ring_exprs[0]
            else:
                # Smooth max via log-sum-exp: (1/k_union) * log(Σ exp(k_union * S_i))
                sum_exp = " + ".join([f"exp({k_union}*({s}))" for s in ring_exprs])
                S_total = f"(1/{k_union})*log({sum_exp})"

            # Final target size field (same blend as your single-point version)
            expr = f"{coarse} - ({coarse} - {fine})*({S_total})"

        gmsh.model.mesh.field.add("MathEval", 101)
        gmsh.model.mesh.field.setString(101, "F", " ".join(expr.split()))
        gmsh.model.mesh.field.setAsBackgroundMesh(101)

    return _cb



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
    _number_of_fractures: int = 11

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length_in_m,
            "ymax": y_length_in_m
        }
        self._domain = pp.Domain(box)
 
    def set_fractures_one(self) -> None:
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
    
    def set_fractures_near(self) -> None:
        """Define two diagonal fractures crossing at the center.
        """
        x_start = 16.0  # self._domain_x_length / 2.0
        x_end = 16.0  # self._domain_x_length - x_start
        y_margin = 0.1 * self._domain_y_length
        frac1 = pp.LineFracture(np.array([
            [x_start, x_end],
            [y_margin, self._domain_y_length - y_margin]
        ]))
        self._fractures = [frac1]
    
    def set_fractures(self) -> None:
        """Define one diagonal fracture and add randomly generated ones."""
        np.random.seed(42)  # Optional for reproducibility
        x_min, y_min = 0.0, 0.0
        x_max = self._domain_x_length
        y_max = self._domain_y_length
        domain_width = x_max - x_min
        domain_height = y_max - y_min
        min_length = 3.0
        max_length = 15.0

        fractures = []
        for _ in range(self._number_of_fractures):
            # Random center within bounds
            x_center = np.random.uniform(x_min + 0.1 * domain_width, x_max - 0.1 * domain_width)
            y_center = np.random.uniform(y_min + 0.1 * domain_height, y_max - 0.1 * domain_height)

            # Random angle and length
            theta = np.random.uniform(0, np.pi)
            length = np.random.uniform(min_length, max_length)

            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)

            x0, x1 = x_center - dx, x_center + dx
            y0, y1 = y_center - dy, y_center + dy

            coords = np.array([[x0, x1], [y0, y1]])
            fractures.append(pp.LineFracture(coords))
        self._fractures = fractures
        
    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")
        # return self.params.get("grid_type", "cartesian")
    
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(2.0, "m")
        frac_cell_size = self.units.convert_units(2.0, "m")
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


class DomainConnectedFracture2D(Geometry):
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

    _domain_x_length: float = 100.0
    _domain_y_length: float = 30.0
    _number_of_fractures: int = 15
    prod_x_point: float = _domain_x_length - 15.0
    prod_y_point: float = _domain_y_length*0.5

    _injection_points: list[np.ndarray] = [
        np.array([_domain_x_length-prod_x_point, prod_y_point])
    ]
    _production_points: list[np.ndarray] = [
        np.array([prod_x_point, prod_y_point])
    ]

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length_in_m,
            "ymax": y_length_in_m
        }
        self._domain = pp.Domain(box)
    
    def set_fractures(self) -> None:
        # """Define fractures that are connected!"""
        # x_start, x_end = 15.0, self._domain_x_length - 15.0
        # y_start, y_end = 10.0, self._domain_y_length - 15.0
        
        f1 = pp.LineFracture(np.array([
            [self._production_points[0][0]-4, 90],
            [11.0, 11.0]
        ]))
        f2 = pp.LineFracture(np.array([
            [self._production_points[0][0]-40, 90-30],
            [14.0, 14.0]
        ]))
        f3 = pp.LineFracture(np.array([
            [self._production_points[0][0]-35, 90-30],
            [14.5, 8.0]
        ]))
        frac1 = pp.LineFracture(np.array([
            [self._production_points[0][0], self._production_points[0][0]+7.0],
            [self._production_points[0][1], self._production_points[0][1]- 5.0]
        ]))
        
        self._fractures = [
            frac1,f2,f3
            # frac1
            # frac1, frac2, frac3,
            # frac4, frac5
            # frac7
            # f2, f3, f4, f5, f6
        ]
    
    def set_fractures_many(self) -> None:
        """Define one diagonal fracture and add randomly generated ones."""
        np.random.seed(42)  # Optional for reproducibility
        x_min, y_min = 0.0, 0.0
        x_max = self._domain_x_length
        y_max = self._domain_y_length
        domain_width = x_max - x_min
        domain_height = y_max - y_min
        min_length = 3.0
        max_length = 15.0

        fractures = []
        for _ in range(self._number_of_fractures):
            # Random center within bounds
            x_center = np.random.uniform(x_min + 0.1 * domain_width, x_max - 0.1 * domain_width)
            y_center = np.random.uniform(y_min + 0.1 * domain_height, y_max - 0.1 * domain_height)

            # Random angle and length
            theta = np.random.uniform(0, np.pi)
            length = np.random.uniform(min_length, max_length)

            dx = 0.5 * length * np.cos(theta)
            dy = 0.5 * length * np.sin(theta)

            x0, x1 = x_center - dx, x_center + dx
            y0, y1 = y_center - dy, y_center + dy

            coords = np.array([[x0, x1], [y0, y1]])
            fractures.append(pp.LineFracture(coords))
        self._fractures = fractures
      
    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")
        # return self.params.get("grid_type", "cartesian")
 
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(2.0, "m")
        frac_cell_size = self.units.convert_units(2.0, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": frac_cell_size
        }
        # set to trigger adaptive meshing arround production well.
        GI.BACKGROUND_FIELD_CB = install_annulus_bg(
            x0=self._production_points[0][0],
            y0=self._production_points[0][1],
            coarse=cell_size,
            fine=1.5,  # .5 without fracture
            radius_in=0.0,  # with r =0.0, refinement is is uniform!
            radius_out=6.0,
            k=100.0
        )
        # GI.BACKGROUND_FIELD_CB_I = install_triangle_bg(
        #     x0=self._injection_points[0][0],
        #     y0=self._injection_points[0][1],
        #     coarse=cell_size,
        #     fine=0.1, # .5 without fracture
        #     side=2.0,
        #     band=3.0,
        #     k=13.0
        # )
        return mesh_args
    
    def set_geometry(self):
        """Create the injection and production wells."""

        super().set_geometry()
        for i, injection_point in enumerate(self._injection_points):
            self._add_well(injection_point, i, "injection", is_multiple_cells=False)

        for i, production_point in enumerate(self._production_points):
            self._add_well(production_point, i, "production", is_multiple_cells=False)
    
    def closest_face(self, grid: pp.Grid, point: np.ndarray) -> int:
        """Return index of closest face center to given point."""
        dists = np.linalg.norm(grid.face_centers - point.reshape(-1, 1), axis=0)
        return int(np.argmin(dists))
    
    def _add_well(
        self, point,
        well_index,
        well_type,
        radius=1.1,
        is_multiple_cells: bool = False
    ):
        """Insert well as multiple 0D grids, each coupled to one matrix cell."""

        matrix = self.mdg.subdomains(dim=self.domain.dim)[0]

        # Find candidate cells within radius
        dist = np.linalg.norm(matrix.cell_centers[:2, :] - point[:, None], axis=0)
        candidate_cells = np.where(dist < radius)[0]
        if is_multiple_cells:
            for c in candidate_cells:
                # Create 0D grid for each connection
                p = np.zeros(3)
                p[:2] = point
                sd_0d = pp.PointGrid(p)
                sd_0d.tags[f"{well_type}_well"] = well_index
                sd_0d.compute_geometry()
                self.mdg.add_subdomains(sd_0d)

                # One-to-one mapping: single well cell → single matrix cell
                cell_well = np.array([0], dtype=int)
                cell_cell_map = sps.coo_matrix(
                    (np.ones(1, dtype=bool), (cell_well, np.array([c]))),
                    shape=(sd_0d.num_cells, matrix.num_cells),
                )

                # Add interface
                _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)
        else:
            self._add_well_inj(point, well_index, well_type)

    def _add_well_inj(
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

        # Couple well to the matrix (0D <--> 2D coupling is allowed with wells in PorePy)
        matrix = self.mdg.subdomains(dim=self.domain.dim)[0]
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )
        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)

        # Couple well to the fracture cells for producer wells and Couple well to the matrix for injector wells: 
        # matrix = self.mdg.subdomains(dim=self.domain.dim)[0]  # pp.Grid type
        # cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        # if is_production:
        #     fracture_subdomains = self.mdg.subdomains(dim=1)  # list[pp.Grid]
        #     # Find the fracture grid + cell that is closest to the well
        #     min_dist = np.inf
        #     chosen_frac = None
        #     chosen_face = None
        #     for frac in fracture_subdomains:
        #         face_indx = self.closest_face(frac, sd_0d.cell_centers)  # candidate cell index
        #         dist = np.linalg.norm(frac.face_centers[:, face_indx].reshape(3,) - sd_0d.cell_centers[:, 0])
        #         if dist < min_dist:
        #             min_dist = dist
        #             chosen_frac = frac
        #             chosen_face = face_indx
        #     target_grid = chosen_frac
        #     target_cell_or_face = np.array([chosen_face])
        #     cell_or_face_num = target_grid.num_faces
        # else:
        #     target_grid = matrix
        #     target_cell_or_face = cell_matrix
        #     cell_or_face_num = target_grid.num_cells

        # cell_well = np.array([0], dtype=int)
        # cell_cell_or_face_map = sps.coo_matrix(
        #     (np.ones(1, dtype=bool), (cell_well, target_cell_or_face)),
        #     shape=(sd_0d.num_cells, cell_or_face_num),
        # )
        # _add_interface(0, target_grid, sd_0d, self.mdg, cell_cell_or_face_map)

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


class SmallBoxPointGridWellsFracture2D(Geometry):
    """A class to represent a simple 2D geometry for a simulation domain. 

    """
    # _dist_from_ref_point: float = 5.0                         
    # _outlet_coverage: tuple[np.ndarray, np.ndarray] = (
    #     np.array([100.0, 0.0, 0.0]),
    #     np.array([100.0, 0.0, 0.0])
    # )
    # _inlet_coverage: list[tuple[np.ndarray, np.ndarray]] = [
    #     (np.array([0.0, 0.0, 0.0]), np.array([0.0, 100.0, 0.0])),
    #     (np.array([0.0, 100.0, 0.0]), np.array([100.0, 100.0, 0.0])),
    #     (np.array([100.0, 100.0, 0.0]), np.array([100.0, 0.0, 0.0])),
    #     (np.array([100.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    # ]
    # _injection_points: list[np.ndarray] = [np.array([25.0, 25.0])]
    # _production_points: list[np.ndarray] = [np.array([25.0, 25.0])]

    # _domain_x_length: float = 100.0
    # _domain_y_length: float = 30.0

    # Domain size
    _domain_x_length: float = 100.0
    _domain_y_length: float = 30.0
    _dist_from_ref_point: float = 5.0

    # Corners of the rectangle (in CCW order)
    _bottom_left = np.array([0.0, 0.0, 0.0])
    _bottom_right = np.array([_domain_x_length, 0.0, 0.0])
    _top_right = np.array([_domain_x_length, _domain_y_length, 0.0])
    _top_left = np.array([0.0, _domain_y_length, 0.0])

    # Outlet coverage (example: right boundary)
    _outlet_coverage: tuple[np.ndarray, np.ndarray] = (
        _top_right,
        _bottom_right,
    )

    # Inlet coverage (all 4 sides)
    _inlet_coverage: list[tuple[np.ndarray, np.ndarray]] = [
        (_bottom_left,  _bottom_right),  # bottom
        (_bottom_right, _top_right),     # right
        (_top_right,    _top_left),      # top
        (_top_left,     _bottom_left),   # left
    ]

    # Injection / production points (automatic placement)
    _injection_points: list[np.ndarray] = [
        np.array([_domain_x_length * 0.25, _domain_y_length * 0.5])  # 1/4 along x, mid y
    ]

    # Production point (center of the domain)
    _production_points: list[np.ndarray] = [
        np.array([_domain_x_length * 0.5, _domain_y_length * 0.5])
    ]

    def set_domain(self) -> None:
        x_length_in_m = self.units.convert_units(self._domain_x_length, "m")
        y_length_in_m = self.units.convert_units(self._domain_y_length, "m")
        box: dict[str, pp.number] = {
            "xmax": x_length_in_m,
            "ymax": y_length_in_m
        }
        self._domain = pp.Domain(box)
    
    def set_fractures_old(self) -> None:
        # """Define fractures that are connected!"""

        frac1 = pp.LineFracture(np.array([
            [25.0, 50.0],
            [25.0, 25.0]
        ]))
        
        self._fractures = [
            frac1
        ]
      
    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")
        # return self.params.get("grid_type", "cartesian")
 
    def meshing_arguments(self) -> dict:
        cell_size = self.units.convert_units(2.0, "m")
        frac_cell_size = self.units.convert_units(2.0, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": frac_cell_size
        }
        # set to trigger adaptive meshing arround production well.
        # GI.BACKGROUND_FIELD_CB = install_boundary_band_bg(
        #     boundary_tags=[1,2,3,4],
        #     coarse=cell_size,
        #     fine=1.0,
        #     d_in=1.0,
        #     d_out=10.0,
        # )
        # GI.BACKGROUND_FIELD_CB = install_annulus_bg(
        #     x0=self._production_points[0][0],
        #     y0=self._production_points[0][1],
        #     coarse=cell_size*1.5,
        #     fine=cell_size,  # .5 without fracture
        #     radius_in=5.0, 
        #     radius_out=10.0, k=13.0
        # )
        # GI.BACKGROUND_FIELD_CB = install_triangle_bg(
        #     x0=self._production_points[0][0],
        #     y0=self._production_points[0][1],
        #     coarse=cell_size,
        #     fine=0.1, # .5 without fracture
        #     side=2.0,
        #     band=3.0,
        #     k=13.0
        # )
        return mesh_args
    
    def set_geometry(self):
        """Create the injection and production wells."""

        super().set_geometry()
        # for i, injection_point in enumerate(self._injection_points):
        #     self._add_well(injection_point, i, "injection", is_multiple_cells=False)

        for i, production_point in enumerate(self._production_points):
            self._add_well(production_point, i, "production", is_multiple_cells=False)
    
    def closest_face(self, grid: pp.Grid, point: np.ndarray) -> int:
        """Return index of closest face center to given point."""
        dists = np.linalg.norm(grid.face_centers - point.reshape(-1, 1), axis=0)
        return int(np.argmin(dists))
    
    def _add_well(
        self, point,
        well_index,
        well_type,
        radius=1.1,
        is_multiple_cells: bool = False
    ):
        """Insert well as multiple 0D grids, each coupled to one matrix cell."""

        matrix = self.mdg.subdomains(dim=self.domain.dim)[0]

        # Find candidate cells within radius
        dist = np.linalg.norm(matrix.cell_centers[:2, :] - point[:, None], axis=0)
        candidate_cells = np.where(dist < radius)[0]
        if is_multiple_cells:
            for c in candidate_cells:
                # Create 0D grid for each connection
                p = np.zeros(3)
                p[:2] = point
                sd_0d = pp.PointGrid(p)
                sd_0d.tags[f"{well_type}_well"] = well_index
                sd_0d.compute_geometry()
                self.mdg.add_subdomains(sd_0d)

                # One-to-one mapping: single well cell → single matrix cell
                cell_well = np.array([0], dtype=int)
                cell_cell_map = sps.coo_matrix(
                    (np.ones(1, dtype=bool), (cell_well, np.array([c]))),
                    shape=(sd_0d.num_cells, matrix.num_cells),
                )

                # Add interface
                _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)
        else:
            self._add_well_inj(point, well_index, well_type)

    def _add_well_inj(
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

        # Couple well to the matrix (0D <--> 2D coupling is allowed with wells in PorePy)
        matrix = self.mdg.subdomains(dim=self.domain.dim)[0]
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )
        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)

    def point_line_distance(self, point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        """Shortest distance between a point and a line segment in 2D or 3D."""
        line_vec = end - start
        p_vec = point - start
        line_len2 = np.dot(line_vec, line_vec)
        if line_len2 == 0:
            return np.linalg.norm(p_vec)  # start == end (degenerate segment)
        t = np.clip(np.dot(p_vec, line_vec) / line_len2, 0, 1)
        projection = start + t * line_vec
        return np.linalg.norm(point - projection)

    def get_inlet_outlet_sides(self, sd: pp.Grid | pp.BoundaryGrid) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(sd, pp.Grid):
            coords = sd.face_centers.T
        elif isinstance(sd, pp.BoundaryGrid):
            coords = sd.cell_centers.T
        else:
            raise ValueError(f"Unsupported type {type(sd)}. Expected Grid or BoundaryGrid.")

        boundary_faces = self.domain_boundary_sides(sd)
        bf_indices = boundary_faces.all_bf

        def filter_facets_by_lines(coverages: list[tuple[np.ndarray, np.ndarray]], tol: float = 1e-8) -> np.ndarray:
            selected = []
            for i, pt in enumerate(coords[bf_indices]):
                for start, end in coverages:
                    if self.point_line_distance(pt, start, end) < tol:
                        selected.append(bf_indices[i])
                        break
            return np.array(selected, dtype=int)

        outlet_facets = np.array([], dtype=int)
        # inlet_facets = filter_facets_by_box(self._inlet_coverage)
        inlet_facets = filter_facets_by_lines(self._inlet_coverage)
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