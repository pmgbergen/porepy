import numpy as np
import porepy as pp
import gmsh
from pathlib import Path
from typing import Optional, NamedTuple

from .wells_3d import Well
from .gmsh_interface import PhysicalNames, GmshEntity, GmshLine
from scipy import sparse as sps


class WellFractureIntersection(NamedTuple):
    """Container class to store representation between a well and a fracture."""

    coord: np.ndarray
    index: int
    well_index: int
    fracture_index: list[int]
    gmsh_index: int


class WellNetwork3d:
    """Collection of :class:`~Well` classes with geometrical information.

    Facilitates meshing of all wells in the network and their addition to a
    mixed-dimensional grid, see e.g., :meth:`~mesh` method.

    Parameters:
        domain: Domain specification.
        wells: ``default=None``

            List of wells in the network. If not empty, the constructor assigns indices
            to the wells corresponding to the order in this list.
        tol: ``default=1e-8``

            Geometric tolerance used in computations.
        parameters: ``default=None``

            Dictionary of parameters, e.g., for the meshing process.

    """

    def __init__(
        self,
        wells: list[Well],
        domain: pp.Domain,
        tol: float = 1e-8,
        parameters: Optional[dict] = None,
    ) -> None:
        self.domain: pp.Domain = domain
        """Domain specification."""

        self.well_dim: int = 1
        """All polyline wells have dimension 1."""

        self.wells: list[Well] = wells if wells is not None else []
        """List of wells in the network."""

        for i, w in enumerate(self.wells):
            w.index = i

        self.parameters: dict = {}
        """Dictionary of parameters, e.g. for the meshing process passed at
        instantiation. """
        if parameters is not None:
            self.parameters = parameters

        self.tol: float = tol
        """Geometric tolerance used in computations."""

        # Assign an empty tag dictionary
        self.tags: dict[str, list[bool]] = dict()

    def mesh(
        self,
        fracture_network: FractureNetwork3d,
        mdg: pp.MixedDimensionalGrid,
    ) -> None:
        """Mesh the well network and add to the mixed-dimensional grid.

        Parameters:
            well_network: Network of wells. Dimension 2 or 3 must match that of the
                fracture network.
            fracture_network: Three-dimensional fracture network.
            mdg: Mixed-dimensional grid to which the well grids will be added.

        """
        # Export to gmsh.
        intersections, wells, fractures = self.intersect_well_fractures(
            fracture_network.fractures, fracture_network.nd
        )
        _set_physical_names(intersections, wells)
        _set_mesh_size(wells, 1.0)
        gmsh.model.mesh.generate(1)
        file_name = Path("well_mesh.msh")
        gmsh.write(file_name.as_posix())

        subdomains = pp.fracs.simplex.line_grid_from_gmsh(
            file_name,
            physical_name_stem_1d=PhysicalNames.WELL.value,
            physical_name_stem_0d=PhysicalNames.WELL_FRACTURE_INTERSECTION_POINT.value,
        )
        well_mdg = pp.meshing.subdomains_to_mdg(subdomains)
        well_mdg.compute_geometry()

        for wg in well_mdg.subdomains(dim=1):
            _update_well_grid_tags(wg, self.domain)

        mdg.add_subdomains(well_mdg.subdomains())

        for isect in intersections:
            g_0d = well_mdg.subdomains(dim=0)[isect.index]
            assert np.allclose(g_0d.cell_centers[:, 0], isect.coord)

            frac_inds = isect.fracture_index

            # Intersection at a fracture intersection. This is in principle possible,
            #  but it will create a non-conforming coupling of codimension 1, which the
            # constitutive laws are probably not ready for. On the other hand, this
            # should also be equivalent to a 1d fracture for nd=2, so perhaps it will
            # not be an issue.
            assert len(frac_inds) == 1, (
                """Multiple fractures intersecting at the same point is not implemented."""
            )
            g_frac = mdg.subdomains(dim=mdg.dim_max() - 1)[frac_inds[0]]
            assert g_frac.frac_num == frac_inds[0]

            embedded_cell = g_frac.closest_cell(g_0d.cell_centers)

            proj = sps.coo_matrix(
                (np.array([1], dtype=bool), (np.array([0]), embedded_cell)),
                shape=(1, g_frac.num_cells),
            ).tocsr()
            _add_interface(0, g_frac, g_0d, mdg, proj)

        return mdg

    def intersect_well_fractures(self, fractures, nd):
        wells = self.wells
        if len(fractures) == 0 or len(wells) == 0:
            return {}
        if not gmsh.is_initialized():
            gmsh.initialize()

        fracture_tags = _export_fractures_to_gmsh(fractures)
        gmsh.model.occ.synchronize()

        segment_inds, segment_to_wells = self._to_gmsh()

        gmsh.model.occ.synchronize()
        fracture_entities, well_entities = _fragment_wells_fractures(
            segment_inds, fracture_tags, nd, segment_to_wells
        )

        point_finder = _PointDetector()

        all_well_points, well_inds = point_finder.points_on_wells(well_entities)
        all_fracture_points, fracture_inds = point_finder.points_on_fractures(
            fracture_entities, nd
        )

        return (
            _find_intersections(
                well_inds, all_well_points, all_fracture_points, fracture_inds
            ),
            well_entities,
            fracture_entities,
        )

    def _to_gmsh(self) -> tuple[list[int], list[int]]:
        segment_inds = [well.to_gmsh() for well in self.wells]
        gmsh.model.occ.synchronize()

        unified_segments = []
        segment_to_wells = []
        for i, segments in enumerate(segment_inds):
            for segment in segments:
                unified_segments.append(segment)
                segment_to_wells.append(i)

        return unified_segments, segment_to_wells

    def _set_physical_names(self, intersections, wells):
        for isect in intersections:
            gmsh.model.addPhysicalGroup(
                0,
                [isect.gmsh_index],
                -1,
                f"{PhysicalNames.WELL_FRACTURE_INTERSECTION_POINT.value}{isect.index}",
            )

        for well in wells:
            gmsh.model.addPhysicalGroup(
                1, well.tags, -1, f"{PhysicalNames.WELL.value}{well.index}"
            )

    def _mesh_size(
        self, well: Well, segment_ind: Optional[tuple[int, int]] = None
    ) -> float:
        """Return the mesh size for a well or one of its segments.

        Parameters:
            well: Well for which to access mesh size.
            segment_ind: ``default=None``

                Indices defining the segment, i.e., indices of the endpoints of the
                segment. If ``None``, the mesh size for the entire well is returned.

        Returns:
            Mesh size for the :attr:`well` or one of its segments.

        """
        size = well._mesh_size(segment_ind)
        if size is None:
            size = self.parameters["mesh_size"]
        return size

    def __repr__(self) -> str:
        """Return a string representation of the well network.

        Returns:
            A string representation of the well network.

        """
        # At the moment, it is unclear what more information should be included in
        # the string representation. We therefore implement only __repr__ (calls to
        # __str__ will be forwarded to __repr__).
        s = f"Well network consisting of {len(self.wells)} wells.\n"
        return s


def _export_fractures_to_gmsh(fractures: list[pp.Fracture]) -> list[int]:
    return [fracture.fracture_to_gmsh() for fracture in fractures]


def _merge_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if len(arrays) > 0:
        return np.hstack(arrays)
    else:
        return np.array([], dtype=int)


class _PointDetector:
    """Helper class to detect intersection points between wells and fractures.

    The class provides the logic for, given a set of wells and fractures represented
    as GmshEntities, to find

    Mainly used as a namespace.
    """

    def points_on_wells(self, wells):
        points, inds = [], []
        for well in wells:
            loc_points = well.embedded_points()
            inds += [well.index] * len(loc_points)
            points.append(loc_points)
        return _merge_arrays(points), inds

    def points_on_fractures(self, fractures, nd):
        if nd == 2:
            return self._points_on_fractures_2d(fractures)
        else:
            return self._points_on_fractures_3d(fractures)

    def _points_on_fractures_2d(self, fractures):
        # Identify points on fractures in 2d, that is, line fractures.
        points, inds = [], []
        for frac in fractures:
            loc_points = frac.embedded_points()
            points.extend(loc_points)
            inds += [frac.index] * len(loc_points)
        return _merge_arrays(points), inds

    def _points_on_fractures_3d(self, fractures):
        # Identify points on the fracture, both embedded and on the boundary.
        embedded_points, fracture_inds_embedded = self._find_embedded_points(fractures)
        boundary_points, fracture_inds_boundary = self._find_boundary_points(fractures)

        return _merge_arrays(
            embedded_points + boundary_points
        ), fracture_inds_embedded + fracture_inds_boundary

    def _find_embedded_points(self, fractures):
        # Find points that are embedded in the fracture - that is, not on the boundary.
        points, inds = [], []
        for fracture in fractures:
            for tag in fracture.tags:
                for point in gmsh.model.mesh.get_embedded(fracture.dim, tag):
                    if point[0] == 0:
                        points.extend([point[1]])
                        inds += [fracture.index]
        return points, inds

    def _find_boundary_points(self, fractures):
        # Find intersections on the fracture boundary
        points, inds = [], []
        for fracture in fractures:
            for tag in fracture.tags:
                boundary_lines = gmsh.model.get_boundary(
                    [(fracture.dim, tag)], oriented=False
                )
                for line in boundary_lines:
                    loc_points = gmsh.model.get_boundary([line], oriented=False)
                    points.extend([p[1] for p in loc_points])
                    inds += [fracture.index] * len(loc_points)
        return points, inds


def _match_well_and_fracture_points(
    well_inds: list[int],
    all_well_points: np.ndarray,
    all_fracture_points: np.ndarray,
    fracture_inds: list[int],
) -> dict[tuple[int, int], set[int]]:
    # Find the points that are shared between wells and fractures. These correspond
    # to intersections.

    # Dictionary that maps (point index, well index) to a set of fracture indices.
    intersections: dict[tuple[int, int], set[int]] = {}
    # Only register each point-well-fracture combination once.
    visited_point_fracture_combo = set()

    for wi, pi in zip(well_inds, all_well_points):
        if pi in all_fracture_points:
            # Find all fractures that contain this point, loop over a unique set of
            # these.
            in_fracture_inds = np.where(all_fracture_points == pi)[0]
            for fi in list(set([fracture_inds[i] for i in in_fracture_inds])):
                if (pi, wi, fi) in visited_point_fracture_combo:
                    continue
                visited_point_fracture_combo.add((pi, wi, fi))
                val = intersections.get((pi, wi), set())
                val.add(fi)
                intersections[(pi, wi)] = val

    return intersections


def _find_intersections(
    well_inds: list[int],
    all_well_points: np.ndarray,
    all_fracture_points: np.ndarray,
    fracture_inds: list[int],
) -> list[WellFractureIntersection]:
    # Combine intersections with the same point and well indices - these will correspond
    # to intersections between the well and a fracture intersection line or point.

    common_points = _match_well_and_fracture_points(
        well_inds, all_well_points, all_fracture_points, fracture_inds
    )

    merged_intersections: list[WellFractureIntersection] = []
    for ind, ((pi, wi), fi_set) in enumerate(common_points.items()):
        coord = gmsh.model.get_bounding_box(0, pi)[:3]
        merged_intersections.append(
            WellFractureIntersection(
                coord=coord,
                index=ind,
                well_index=wi,
                fracture_index=list(fi_set),
                gmsh_index=pi,
            )
        )

    return merged_intersections


def _fragment_wells_fractures(well_tags, fracture_tags, nd, segment_to_wells):
    _, split_objects = gmsh.model.occ.fragment(
        [(nd - 1, t) for t in fracture_tags],
        [(1, t) for t in well_tags],
        removeObject=True,
        removeTool=True,
    )
    gmsh.model.occ.synchronize()
    fractures = []
    for fi, fracture in enumerate(split_objects[: len(fracture_tags)]):
        gmsh_inds = [t[1] for t in fracture]
        if nd == 3:
            fractures.append(GmshEntity(index=fi, dim=nd - 1, tags=gmsh_inds))
        else:
            fractures.append(GmshLine(index=fi, tags=gmsh_inds))

    wells = []
    for wi in np.unique(segment_to_wells):
        ind_in_object = (
            len(fracture_tags) + np.where(np.array(segment_to_wells) == wi)[0]
        )
        gmsh_tags = []
        for si in ind_in_object:
            gmsh_tags.extend([t[1] for t in split_objects[si]])
        wells.append(GmshLine(index=wi, tags=gmsh_tags))

    return fractures, wells


def _update_well_grid_tags(g, domain):
    # Update the tags for the well grid, to identify boundary faces and tips.
    bounding_planes = domain.polytope_from_bounding_box()
    on_boundary = np.zeros(g.num_faces, dtype=bool)
    for plane in bounding_planes:
        if domain.dim == 2:
            plane = np.vstack((plane, np.zeros(plane.shape[1])))
            dist, *_ = pp.geometry.distances.points_segments(
                g.face_centers, plane[:, 0], plane[:, 1]
            )
        else:
            dist, _, _ = pp.geometry.distances.points_polygon(g.face_centers, plane)

        on_boundary = np.logical_or(on_boundary, np.isclose(dist, 0))
    g.tags["tip_faces"] = g.tags["domain_boundary_faces"] & np.logical_not(on_boundary)
    g.tags["domain_boundary_faces"] = on_boundary


def _set_mesh_size(wells, cell_size):
    gmsh.model.mesh.set_size([(w.dim, t) for w in wells for t in w.tags], cell_size)


def _add_interface(
    dim: int,
    sd_primary: pp.Grid,
    sd_secondary: pp.Grid,
    mdg: pp.MixedDimensionalGrid,
    primary_secondary_map: sps.coo_matrix,
) -> None:
    """Utility method to add an interface to the mdg.

    Both grids should already be present in the mixed-dimensional grid.

    Parameters:
        sd_primary: Primary subdomain grid. In the context of this module, it represents
            a fracture or well.
        sd_secondary: Secondary subdomain grid. In the context of this module, it
            typically represents an intersection point.
        mdg: MixedDimensionalGrid to which the interface will be added.
        primary_secondary_map: Map between ``cells_l`` and either ``faces_h`` (codim=1)
            or ``cells_h`` (codim=2).

    """
    codim = sd_primary.dim - sd_secondary.dim
    subdomain_pair = (sd_primary, sd_secondary)
    side_g = {pp.grids.mortar_grid.MortarSides.LEFT_SIDE: sd_secondary.copy()}
    mg = pp.MortarGrid(dim, side_g, primary_secondary_map, codim=codim)
    mg._primary_to_mortar_int = primary_secondary_map
    mg.compute_geometry()
    mdg.add_interface(mg, subdomain_pair, primary_secondary_map)
