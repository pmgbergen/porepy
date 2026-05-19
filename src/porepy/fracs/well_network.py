import numpy as np
import porepy as pp
import gmsh
from pathlib import Path
from typing import Optional, NamedTuple

from .wells_3d import Well
from .gmsh_interface import PhysicalNames, GmshEntity, GmshLine, GmshSurface, fragment
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
        mesh_args: dict,
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
        self._set_physical_names(intersections, wells)
        self._set_mesh_size(wells, mesh_args.get("cell_size"))
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
            self._update_well_grid_tags(wg, self.domain)

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
            self._add_interface(0, g_frac, g_0d, mdg, proj)

        return mdg

    def intersect_well_fractures(self, fractures, nd):
        wells = self.wells
        if len(fractures) == 0 or len(wells) == 0:
            return {}
        if not gmsh.is_initialized():
            gmsh.initialize()

        fractures = _export_fractures_to_gmsh(fractures)
        gmsh.model.occ.synchronize()

        segments = self._to_gmsh()

        gmsh.model.occ.synchronize()
        fracture_entities, well_entities = fragment(fractures, segments)

        return (
            self._intersections_from_points(
                _PointsOnEntities(well_entities), _PointsOnEntities(fracture_entities)
            ),
            well_entities,
            fracture_entities,
        )

    def _intersections_from_points(
        self, well_points: _PointsOnEntities, fracture_points: _PointsOnEntities
    ) -> list[WellFractureIntersection]:
        # Combine intersections with the same point and well indices - these will
        # correspond to intersections between the well and a fracture intersection line
        # or point.

        common_points = self._match_well_and_fracture_points(
            well_points, fracture_points
        )
        well_points = self._well_kink_points(well_points, common_points)

        all_points = common_points | well_points

        merged_intersections: list[WellFractureIntersection] = []
        for ind, ((pi, wi), fi_set) in enumerate(all_points.items()):
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

    def _well_kink_points(
        self,
        well_points: _PointsOnEntities,
        well_fracture_comment: dict[tuple[int, int], set[int]],
    ) -> dict[tuple[int, int], set[int]]:
        # Find points that are shared between wells. These correspond to kinks in the
        # well geometry.

        # Dictionary that maps (point index, well index) to a set of fracture indices.
        kinks: dict[tuple[int, int], set[int]] = {}

        for wi in np.unique(well_points.inds):
            ind_in_well = np.where(well_points.inds == wi)[0]
            loc_points = well_points.points[ind_in_well]
            duplicate_indices = np.where(np.bincount(loc_points) > 1)[0]
            for p in duplicate_indices:
                if (p, wi) in well_fracture_comment:
                    # This is an intersection point, so we do not want to register it as
                    # a kink.
                    continue
                kinks[(p, wi)] = set()

        return kinks

    def _match_well_and_fracture_points(
        self, well_points: _PointsOnEntities, fracture_points: _PointsOnEntities
    ) -> dict[tuple[int, int], set[int]]:
        # Find the points that are shared between wells and fractures. These correspond
        # to intersections.

        # Dictionary that maps (point index, well index) to a set of fracture indices.
        intersections: dict[tuple[int, int], set[int]] = {}
        # Only register each point-well-fracture combination once.
        visited_point_fracture_combo = set()

        for wi, pi in zip(well_points.inds, well_points.points):
            if pi in fracture_points.points:
                # Find all fractures that contain this point, loop over a unique set of
                # these.
                in_fracture_inds = np.where(fracture_points.points == pi)[0]
                for fi in list(
                    set([fracture_points.inds[i] for i in in_fracture_inds])
                ):
                    if (pi, wi, fi) in visited_point_fracture_combo:
                        continue
                    visited_point_fracture_combo.add((pi, wi, fi))
                    val = intersections.get((pi, wi), set())
                    val.add(fi)
                    intersections[(pi, wi)] = val

        return intersections

    def _to_gmsh(self) -> tuple[list[int], list[int]]:
        segment_inds = [well.to_gmsh() for well in self.wells]
        gmsh.model.occ.synchronize()

        entities = []

        for i, segments in enumerate(segment_inds):
            indices = [s for s in segments]
            entities += [GmshLine(index=i, tags=indices)]

        return entities

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

    def _update_well_grid_tags(self, g, domain):
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
        g.tags["tip_faces"] = g.tags["domain_boundary_faces"] & np.logical_not(
            on_boundary
        )
        g.tags["domain_boundary_faces"] = on_boundary

    def _add_interface(
        self,
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

    def _set_mesh_size(self, wells, cell_size):
        gmsh.model.mesh.set_size([(w.dim, t) for w in wells for t in w.tags], cell_size)

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
    entities = []
    dim = 1 if isinstance(fractures[0], pp.LineFracture) else 2
    for fracture in fractures:
        tag = fracture.fracture_to_gmsh()
        entities += [GmshEntity(index=fracture.index, tags=[tag], dim=dim)]
    return entities


def _merge_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if len(arrays) > 0:
        return np.hstack(arrays)
    else:
        return np.array([], dtype=int)


class _PointsOnEntities:
    def __init__(self, entities: list[GmshEntity]):
        points, inds = [], []
        for entity in entities:
            loc_points, loc_inds = entity.embedded_points()
            points.extend(loc_points)
            inds.extend(loc_inds)
        self.points = _merge_arrays(points)
        self.inds = inds
