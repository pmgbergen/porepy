import numpy as np
import porepy as pp
import gmsh
from pathlib import Path
from typing import Optional, NamedTuple
from dataclasses import dataclass
from .wells_3d import Well


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
        domain: pp.Domain,
        wells: Optional[list[Well]] = None,
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

    def to_gmsh(self) -> list[list[int]]:
        inds = []
        for w in self.wells:
            inds.append(w.to_gmsh())

        return inds

    def mesh(self, mdg: pp.MixedDimensionalGrid) -> None:
        """Produce grids for the network's wells and add to existing ``mdg``.

        One grid is constructed for each sub-line extending between two fracture
        intersections. In the simplest case, the well is a (poly)-line with two end
        points, and a single grid is produced.

        Intersection grids are added for all intersection points between wells and
        fractures. Finally, edges are added between intersection points and both
        fractures and well segments.

        Example:
            Topology for well intersecting two fractures, terminating at the lowermost

            .. code:: python3

                            |
                sd_well_0   |
                            |
                            * e(sd_isec_0, sd_well_0)
                sd_isec_0    . * e(sd_isec_0, sd_frac_0)  ___________ sd_frac_0 (2d)
                            * e(sd_isec_0, sd_well_1)
                            |
                sd_well_1   |
                            |
                            * e(sd_isec_1, sd_well_1)
                sd_isec_1    . * e(sd_isec_1, sd_frac_1)  ___________ sd_frac_1 (2d)

            Note that all edge grids ``*`` are zero-dimensional, and that those
            connected with the fracture have co-dimension 2.

        Each point defining the well polyline is assumed to have a tag list stored in
        ``well.tags["intersecting_fractures"]``. An empty tag means the point does
        not correspond to a fracture intersection. An integer entry ``i`` identifies
        an intersection with the fracture with ``g.frac_num = i``. If the list
        contains multiple tags, the interpretation would be an intersection between
        the well and a fracture intersection line or point. This is not implemented.
        Points not corresponding to a fracture intersection, but merely representing
        a kink in the polyline, will not be represented by a zero-dimensional grid.
        Instead, the two neighboring segments are joined and a single *piecewise*
        linear grid is produced.

        This function may be split/restructured in the future. One possibility is to
        let gmsh do the actual meshing as done in the ``FractureNetwork`` classes.
        For now, this simplified approach is deemed sufficient.

        Parameters:
            mdg: Mixed-dimensional grid.

        """
        # Bounding planes for the domain, used to identify boundary faces of the well.
        bounding_planes = self.domain.polytope_from_bounding_box()

        # Will be added as g.well_num for the well grids.
        well_num = 0
        for w in self.wells:
            tags_w = w.tags.get(
                "intersecting_fractures", [np.empty(0)] * w.pts.shape[1]
            )
            for t in tags_w:
                if t.size > 1:
                    raise NotImplementedError(
                        """Meshing of wells intersecting multiple fractures at
                        the same point is not implemented."""
                    )

            subline_endpoint_inds = [0]
            points_subline = np.empty((3, 0))
            # Tags for the endpoint faces of the well grid.
            endp_tip_tags = np.zeros(2, dtype=bool)
            endp_frac_tags = np.zeros(2, dtype=bool)
            # Special treatment of first end point. We need to check whether it
            # corresponds to an intersection. If so, we make an intersection grid and
            # add an edge between that grid and the fracture in question. Note that the
            # edge with the first well segment is added below.
            if tags_w[0].size > 0:
                sd_isec = _intersection_subdomain(w.pts[:, 0], mdg)
                _add_fracture_2_intersection_interface(sd_isec, tags_w[0], mdg)
                endp_frac_tags[0] = True

            for inds_seg, seg in w.segments():
                tags_seg = [tags_w[i] for i in inds_seg]
                length = pp.geometry.distances.point_pointset(seg[:, 0], seg[:, 1])[0]
                num_pts = int(length / self._mesh_size(w, inds_seg))
                num_pts = max(num_pts, 2)
                points_loc = np.linspace(seg[:, 0], seg[:, 1], num_pts).T
                points_subline = np.hstack((points_subline, points_loc))

                # Flag to tell if this segment ends on the global boundary. This is
                # needed to identify whether the end point is a tip or on the boundary.
                segment_ends_on_boundary = False

                # Check if the second end point is a fracture intersection. If not,
                # proceed to next segment unless we're at the well's second endpoint.
                if tags_seg[1].size == 0:
                    if inds_seg[1] == w.num_points() - 1:
                        # We're at an end at the well. Depending on which direction the
                        # well was traversed, this is either a tip or on the global
                        # boundary.
                        for plane in bounding_planes:
                            dist, _, _ = pp.geometry.distances.points_polygon(
                                seg[:, -1].reshape(3, 1), plane
                            )
                            segment_ends_on_boundary = np.logical_or(
                                segment_ends_on_boundary, np.isclose(dist, 0)[0]
                            )

                        endp_tip_tags[1] = not segment_ends_on_boundary
                        # This is definitely not a fracture intersection.
                        endp_frac_tags[1] = False
                    else:
                        # Remove last point, since it is included in next iteration.
                        points_subline = np.reshape(points_subline[:, :-1], (3, -1))
                        continue

                # The end point is an intersection. Thus, we make a grid for the
                # subdomain consisting of the segments from endpoint_inds_loc[0] to
                # this end point and for the intersection point.
                subline_endpoint_inds.append(inds_seg[1])
                sd_w = pp.TensorGrid(np.arange(points_subline.shape[1]))
                sd_w.nodes = points_subline.copy()
                sd_w.compute_geometry()
                mdg.add_subdomains(sd_w)

                sd_w.well_num = well_num
                sd_w.name += " well " + str(well_num)
                sd_w.tags["parent_well_index"] = w.index
                well_num += 1

                # Add intersection grid and interfaces if the second segment point is
                # not a tip and not on the global boundary.
                if not endp_tip_tags[1] and not segment_ends_on_boundary:
                    endp_frac_tags[1] = True
                    sd_isec = _intersection_subdomain(seg[:, 1], mdg)
                    sd_isec.tags["parent_well_index"] = w.index

                    # Add interfaces between intersection grid and both fracture and
                    # well grid.
                    _add_well_2_intersection_interface(sd_w, sd_isec, mdg)
                    _add_fracture_2_intersection_interface(
                        sd_isec, tags_w[inds_seg[1]], mdg
                    )

                # Further, if the new segment's first end point corresponds to an
                # intersection (as opposed to a global boundary or internal tip),
                # add the interface between this segment and that intersection.
                if endp_frac_tags[0]:
                    # Index for the intersection grid corresponding to the first
                    # endpoint of this subline. Last one if we have not added for the
                    # second endpoint, in which case it's the penultimate 0d grid
                    previous_ind = -1 - endp_frac_tags[1]
                    previous_g_isec = mdg.subdomains(dim=self.well_dim - 1)[
                        previous_ind
                    ]  # EK, is there a preferred method?
                    _add_well_2_intersection_interface(sd_w, previous_g_isec, mdg)

                # Finally, update tags for the well's faces (boundary, tip, fracture).
                endp_inds = [0, -1]
                endpts = sd_w.face_centers[:, endp_inds]
                # Strictly speaking, we already know if the segment ends (index [1]) on
                # the boundary. However, for code simplicity, we recompute this here.
                boundary = np.zeros(2, dtype=bool)
                for plane in bounding_planes:
                    dist, _, _ = pp.geometry.distances.points_polygon(endpts, plane)
                    boundary = np.logical_or(boundary, np.isclose(dist, 0))

                # It was determined earlier whether the second endpoint (index [1]) is a
                # tip. Set the value for the first endpoint [0] here.
                endp_tip_tags[0] = np.logical_not(
                    np.logical_or(boundary[0], endp_frac_tags[0])
                )
                sd_w.tags["domain_boundary_faces"][endp_inds] = boundary
                sd_w.tags["tip_faces"][endp_inds] = endp_tip_tags
                sd_w.tags["fracture_faces"][endp_inds] = endp_frac_tags

                # Properly initalize the newly generated boundary grid.
                if (bg_w := mdg.subdomain_to_boundary_grid(sd_w)) is not None:
                    # Overwrite number of cells. This was initialized wrongly before
                    # sd_w.tags["domain_boundary_faces"] was set.
                    bg_w.num_cells = np.sum(boundary)
                    bg_w.set_projections()
                    bg_w.compute_geometry()
                # Reset the points for next iteration/subline.
                points_subline = np.empty((3, 0))
                subline_endpoint_inds = [inds_seg[1]]
                endp_tip_tags = np.zeros(2, dtype=bool)
                endp_frac_tags = np.array([1, 0], dtype=bool)
        for t in ["domain_boundary", "tip", "fracture"]:
            pp.utils.tags.add_node_tags_from_face_tags(mdg, t)

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


def _export_wells_to_gmsh(wells: list[Well]) -> tuple[list[int], list[int]]:
    segment_inds = [well.to_gmsh() for well in wells]
    gmsh.model.occ.synchronize()

    unified_segments = []
    segment_to_wells = []
    for i, segments in enumerate(segment_inds):
        for segment in segments:
            unified_segments.append(segment)
            segment_to_wells.append(i)

    return unified_segments, segment_to_wells


def _export_fractures_to_gmsh(fractures: list[pp.Fracture]) -> list[int]:
    return [fracture.fracture_to_gmsh() for fracture in fractures]


def _merge_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if len(arrays) > 0:
        return np.hstack(arrays)
    else:
        return np.array([], dtype=int)


def _points_on_wells(wells):
    inds = []
    points = []
    for well in wells:
        for tag in well.tags:
            _, adjacent_points = gmsh.model.get_adjacencies(well.dim, tag)
            points.extend(adjacent_points)
            inds += [well.index] * len(adjacent_points)
    points = _merge_arrays(points)
    return points, inds


def _points_on_fractures(split_fractures, nd):
    if nd == 2:
        return _points_on_fractures_2d(split_fractures)
    else:
        return _points_on_fractures_3d(split_fractures)


def _points_on_fractures_2d(fractures):
    # Identify points on fractures in 2d, that is, line fractures.
    points, inds = [], []
    for frac in fractures:
        for tag in frac.tags:
            _, adjacent_points = gmsh.model.get_adjacencies(1, tag)
            points.extend(adjacent_points)
            inds += [frac.index] * len(adjacent_points)
    return _merge_arrays(points), inds


def _points_on_fractures_3d(fractures):
    # Identify points on the fracture, both embedded and on the boundary.
    def _find_embedded_points():
        # Find points that are embedded in the fracture - that is, not on the boundary.
        points, inds = [], []
        for fracture in fractures:
            for tag in fracture.tags:
                for point in gmsh.model.mesh.get_embedded(fracture.dim, tag):
                    if point[0] == 0:
                        points.extend([point[1]])
                        inds += [fracture.index]
        return points, inds

    def _find_boundary_points():
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

    embedded_points, fracture_inds_embedded = _find_embedded_points()
    boundary_points, fracture_inds_boundary = _find_boundary_points()

    return _merge_arrays(
        embedded_points + boundary_points
    ), fracture_inds_embedded + fracture_inds_boundary


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
) -> list[IntersectionInfo]:
    # Combine intersections with the same point and well indices - these will correspond
    # to intersections between the well and a fracture intersection line or point.

    common_points = _match_well_and_fracture_points(
        well_inds, all_well_points, all_fracture_points, fracture_inds
    )

    merged_intersections: list[IntersectionInfo] = []
    for ind, ((pi, wi), fi_set) in enumerate(common_points.items()):
        coord = gmsh.model.get_bounding_box(0, pi)[:3]
        merged_intersections.append(
            IntersectionInfo(
                coord=coord,
                index=ind,
                well_index=wi,
                fracture_index=list(fi_set),
                gmsh_index=pi,
            )
        )

    return merged_intersections


class IntersectionInfo(NamedTuple):
    coord: np.ndarray
    index: int
    well_index: int
    fracture_index: list[int]
    gmsh_index: int


@dataclass
class Entity:
    """Representation of a single geometric entity, in terms of gmsh tags.

    The object may have been fragmented into multiple sub-entities.
    """

    index: int
    dim: int
    tags: list[int]


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
        fractures.append(Entity(index=fi, dim=nd - 1, tags=gmsh_inds))

    wells = []
    for wi in np.unique(segment_to_wells):
        ind_in_object = (
            len(fracture_tags) + np.where(np.array(segment_to_wells) == wi)[0]
        )
        gmsh_tags = []
        for si in ind_in_object:
            gmsh_tags.extend([t[1] for t in split_objects[si]])
        wells.append(Entity(index=wi, dim=1, tags=gmsh_tags))

    return fractures, wells


def intersect_well_fractures(wells, fractures, nd):
    if len(fractures) == 0 or len(wells) == 0:
        return {}
    if not gmsh.is_initialized():
        gmsh.initialize()

    fracture_tags = _export_fractures_to_gmsh(fractures)
    gmsh.model.occ.synchronize()

    segment_inds, segment_to_wells = _export_wells_to_gmsh(wells)

    gmsh.model.occ.synchronize()
    fracture_entities, well_entities = _fragment_wells_fractures(
        segment_inds, fracture_tags, nd, segment_to_wells
    )
    all_well_points, well_inds = _points_on_wells(well_entities)
    all_fracture_points, fracture_inds = _points_on_fractures(fracture_entities, nd)

    return (
        _find_intersections(
            well_inds, all_well_points, all_fracture_points, fracture_inds
        ),
        well_entities,
        fracture_entities,
    )


def mesh(
    well_network: WellNetwork3d,
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
    intersections, wells, fractures = intersect_well_fractures(
        well_network.wells, fracture_network.fractures, fracture_network.nd
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
        _update_well_grid_tags(wg, domain)

    mdg.add_subdomains(well_mdg.subdomains())

    for isect in intersections:
        g_0d = well_mdg.subdomains(dim=0)[isect.index]
        assert np.allclose(g_0d.cell_centers[:, 0], isect.coord)

        frac_inds = isect.fracture_index

        # Intersection at a fracture intersection. This is in principle possible, but it
        #  will create a non-conforming coupling of
        # codimension 1, which the constitutive laws are probably not ready for.
        # On the other hand, this should also be equivalent to a 1d fracture for nd=2,
        # so perhaps it will not be an issue.
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


def _update_well_grid_tags(g, domain):
    # Update the tags for the well grid, to identify boundary faces and tips.
    bounding_planes = domain.polytope_from_bounding_box()
    on_boundary = np.zeros(g.num_faces, dtype=bool)
    for plane in bounding_planes:
        dist, _, _ = pp.geometry.distances.points_polygon(g.face_centers, plane)
        on_boundary = np.logical_or(on_boundary, np.isclose(dist, 0))
    g.tags["tip_faces"] = g.tags["domain_boundary_faces"] & np.logical_not(on_boundary)
    g.tags["domain_boundary_faces"] = on_boundary


def _set_physical_names(intersections, wells):
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


def _set_mesh_size(wells, cell_size):
    gmsh.model.mesh.set_size([(w.dim, t) for w in wells for t in w.tags], cell_size)
