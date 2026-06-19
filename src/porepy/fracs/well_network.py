from __future__ import annotations

import csv
import numpy as np
import porepy as pp
import gmsh
from pathlib import Path
from typing import Optional, NamedTuple, TYPE_CHECKING

from .wells_3d import Well
from .gmsh_interface import PhysicalNames, GmshEntity, GmshLine, fragment
from scipy import sparse as sps


if TYPE_CHECKING:
    from porepy.fracs.fracture_network import FractureNetwork


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
            w._index = i

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
        fracture_network: FractureNetwork,
        mdg: pp.MixedDimensionalGrid,
        mesh_args: dict,
    ) -> pp.MixedDimensionalGrid:
        """Mesh the well network and add to the mixed-dimensional grid.

        Parameters:
            well_network: Network of wells. Dimension 2 or 3 must match that of the
                fracture network.
            fracture_network: Three-dimensional fracture network.
            mdg: Mixed-dimensional grid to which the well grids will be added.

        """
        if len(self.wells) == 0:
            return mdg

        if not gmsh.is_initialized():
            gmsh.initialize()

        # Export to gmsh.
        intersections, wells, fractures = self.intersect_well_fractures(
            fracture_network.fractures, fracture_network.nd
        )
        well_mdg = _generate_well_mesh(intersections, wells, mesh_args)

        orig_0d_domain_id = _add_well_subdomains(mdg, well_mdg, self.tol, self.domain)

        if len(intersections) == 0:
            return mdg

        _add_well_fracture_interfaces(
            mdg, well_mdg, intersections, orig_0d_domain_id, self.tol
        )

        return mdg

    def intersect_well_fractures(
        self,
        fractures: list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture],
        nd: int,
    ) -> tuple[list[WellFractureIntersection], list[GmshEntity], list[GmshEntity]]:
        wells = self.wells
        if not gmsh.is_initialized():
            gmsh.initialize()

        fracture_tags = _export_fractures_to_gmsh(fractures)
        gmsh.model.occ.synchronize()

        segments = _to_gmsh(self.wells)

        gmsh.model.occ.synchronize()
        fracture_entities, well_entities = fragment(fracture_tags, segments)

        return (
            _intersections_from_points(
                _PointsOnEntities(well_entities), _PointsOnEntities(fracture_entities)
            ),
            well_entities,
            fracture_entities,
        )

    def to_csv(self, file_name: Path, write_header: bool = True) -> None:
        """Export the well network to a csv file.

        Parameters:
            file_name: Path to the csv file to which to export the well network.
            write_header: Whether to write a header row to the csv file.

        """
        file_name = file_name.with_suffix(".csv")

        # Delete the file 'csv_file' if it exists. This seems to be necessary to run
        # tests on GH actions.
        if file_name.exists():
            file_name.unlink()

        with open(file_name, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            if write_header:
                csv_writer.writerow("# Well network exported from porepy.")
                csv_writer.writerow(
                    "# The first line may contain a 6-item bounding box for the domain"
                    " in the format X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX."
                )
                csv_writer.writerow(
                    "# Each row contains the coordinates of the endpoints of each well "
                    "segment, ordered as (x1, y1, z1, x2, y2, z2, ...)."
                )

            # Write the domain bounding box.
            if self.domain is not None:
                if self.domain.dim == 2:
                    order = ["xmin", "ymin", "xmax", "ymax"]
                else:
                    order = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
                csv_writer.writerow([self.domain.bounding_box[o] for o in order])

            # write all the wells
            for w in self.wells:
                csv_writer.writerow(w.pts.ravel(order="F"))

    @classmethod
    def from_csv(cls, file_name: Path) -> WellNetwork3d:
        """Import a well network from a csv file.

        Parameters:
            file_name: Path to the csv file from which to import the well network. The
                csv file should have the same format as the one exported by the
                ``to_csv`` method.

        Returns:
            A new instance of the WellNetwork3d class initialized with the wells from
            the csv file.

        """
        wells = []
        with open(file_name, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            domain = None
            for row in csv_reader:
                if row[0].startswith("#"):
                    continue
                if domain is None:
                    if len(row) == 6:
                        domain = pp.Domain(
                            bounding_box={
                                "xmin": float(row[0]),
                                "ymin": float(row[1]),
                                "zmin": float(row[2]),
                                "xmax": float(row[3]),
                                "ymax": float(row[4]),
                                "zmax": float(row[5]),
                            }
                        )
                    elif len(row) == 4:
                        domain = pp.Domain(
                            bounding_box={
                                "xmin": float(row[0]),
                                "ymin": float(row[1]),
                                "xmax": float(row[2]),
                                "ymax": float(row[3]),
                            }
                        )
                    else:
                        raise ValueError(
                            "The first non-comment line in the csv file should contain "
                            "the domain bounding box with 6 entries in 3d and 4 "
                            "entries in 2d."
                        )
                else:
                    pts = np.array(row, dtype=float).reshape(-1, 3, order="F")
                    wells.append(Well(pts))
        assert domain is not None, (
            "The csv file should contain a line with the domain bounding box."
        )
        return cls(wells, domain)

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


def _export_fractures_to_gmsh(
    fractures: list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture],
) -> list[GmshEntity]:
    entities = []
    if len(fractures) == 0:
        return entities

    dim = 1 if isinstance(fractures[0], pp.LineFracture) else 2
    for fracture in fractures:
        tag = fracture.fracture_to_gmsh()
        entities += [GmshEntity(index=fracture.index, tags=[tag], dim=dim)]
    return entities


def _intersections_from_points(
    well_points: _PointsOnEntities, fracture_points: _PointsOnEntities
) -> list[WellFractureIntersection]:
    # Combine intersections with the same point and well indices - these will
    # correspond to intersections between the well and a fracture intersection line
    # or point.

    common_points = _match_well_and_fracture_points(well_points, fracture_points)
    kink_points = _well_kink_points(well_points, common_points)

    all_points = common_points | kink_points

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


def _generate_well_mesh(
    intersections: list[WellFractureIntersection],
    wells: list[GmshEntity],
    mesh_args: dict,
) -> pp.MixedDimensionalGrid:
    _set_physical_names(intersections, wells)
    _set_mesh_size(wells, mesh_args.get("cell_size"))
    gmsh.model.mesh.generate(1)
    file_name = Path("well_mesh.msh")
    gmsh.write(file_name.as_posix())

    subdomains = pp.fracs.simplex.line_grid_from_gmsh(
        file_name,
        physical_name_stem_1d=PhysicalNames.WELL.value,
        physical_name_stem_0d=PhysicalNames.WELL_FRACTURE_INTERSECTION_POINT.value,
        sort_1d_nodes=False,
    )
    well_mdg = pp.meshing.subdomains_to_mdg(subdomains)
    well_mdg.compute_geometry()
    for wi, wg in enumerate(well_mdg.subdomains(dim=1)):
        wg.well_num = wi
        wg.frac_num = -1

    gmsh.finalize()
    return well_mdg


def _add_well_subdomains(
    mdg: pp.MixedDimensionalGrid,
    well_mdg: pp.MixedDimensionalGrid,
    tol: float,
    domain: pp.Domain,
) -> list[int]:
    _check_overlapping_point_grids(mdg, well_mdg, tol)

    orig_0d_domain_id = [sd.id for sd in mdg.subdomains(dim=0)]

    mdg.add_subdomains(well_mdg.subdomains())
    for intf, data in well_mdg.interfaces(return_data=True):
        sd_primary, sd_secondary = well_mdg.interface_to_subdomain_pair(intf)
        mdg.add_interface(intf, (sd_primary, sd_secondary), data["face_cells"])

    for wg in well_mdg.subdomains(dim=1):
        _update_well_grid_tags(wg, domain, mdg)

    return orig_0d_domain_id


def _add_well_fracture_interfaces(
    mdg: pp.MixedDimensionalGrid,
    well_mdg: pp.MixedDimensionalGrid,
    intersections: list[WellFractureIntersection],
    orig_0d_domain_id: list[int],
    tol: float,
) -> None:

    def match_point_grid(isect) -> pp.Grid:
        # The intersection should be on a 0d intersection point, or else
        # we have either overlapping fractures or a faulty interpretation
        # of the geometry.
        found = False
        g_frac = None
        for sd in mdg.subdomains(dim=0):
            if sd.id in orig_0d_domain_id and np.isclose(
                np.linalg.norm(sd.cell_centers - isect.coord), 0, atol=tol
            ):
                g_frac = sd
                break
        return g_frac, found

    def match_line_grid(isect) -> pp.Grid:
        dist_min = np.inf
        for sd in common_fracture_intersections(frac_inds):
            dist = np.min(
                np.linalg.norm(
                    sd.cell_centers - np.array(isect.coord).reshape((-1, 1)), axis=0
                )
            )
            if dist < dist_min:
                dist_min = dist
                g_frac = sd
        return g_frac

    def common_fracture_intersections(frac_inds):
        sds_1d = set(mdg.subdomains(dim=mdg.dim_max() - 2))
        for fi in frac_inds:
            g_frac = mdg.subdomains(dim=mdg.dim_max() - 1)[fi]
            sds_1d = sds_1d.intersection(
                mdg.neighboring_subdomains(g_frac, only_lower=True)
            )
        return sds_1d

    point_grid_coord = np.vstack(
        [g.cell_centers[:, 0] for g in well_mdg.subdomains(dim=0)]
    ).T

    for isect in intersections:
        if len(isect.fracture_index) == 0:
            # This is a kink in the well. Continue.
            continue

        ind_0d = np.argmin(
            np.linalg.norm(point_grid_coord - np.reshape(isect.coord, (-1, 1)), axis=0)
        )

        g_low = well_mdg.subdomains(dim=0)[ind_0d]

        frac_inds = isect.fracture_index

        # Intersection at a fracture intersection. This is in principle possible,
        #  but it will create a non-conforming coupling of codimension 1, which the
        # constitutive laws are probably not ready for. On the other hand, this
        # should also be equivalent to a 1d fracture for nd=2, so perhaps it will
        # not be an issue.
        if len(frac_inds) == 1:
            g_high = mdg.subdomains(dim=mdg.dim_max() - 1)[frac_inds[0]]
            assert g_high.frac_num == frac_inds[0]
        else:
            # TODO: Document this carefully, and check the logic. It may be that some
            # cases could be ruled out.
            if mdg.dim_max() == 2:
                g_high, found = match_point_grid(isect)
                assert found, "Intersection point not found in fracture mesh."
            else:  # mdg.dim_max() == 3
                g_high, found = match_point_grid(isect)
                if found:
                    assert len(frac_inds) > 2
                else:
                    g_high = match_line_grid(isect)

        embedded_cell = g_high.closest_cell(g_low.cell_centers)

        proj = sps.coo_matrix(
            (np.array([1], dtype=bool), (np.array([0]), embedded_cell)),
            shape=(1, g_high.num_cells),
        ).tocsr()
        _add_interface(0, g_high, g_low, mdg, proj)


def _match_well_and_fracture_points(
    well_points: _PointsOnEntities, fracture_points: _PointsOnEntities
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
            for fi in list(set([fracture_points.inds[i] for i in in_fracture_inds])):
                if (pi, wi, fi) in visited_point_fracture_combo:
                    continue
                visited_point_fracture_combo.add((pi, wi, fi))
                val = intersections.get((pi, wi), set())
                val.add(fi)
                intersections[(pi, wi)] = val

    return intersections


def _well_kink_points(
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


def _set_physical_names(
    intersections: list[WellFractureIntersection], wells: list[GmshEntity]
) -> None:
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


def _set_mesh_size(wells: list[GmshEntity], cell_size: Optional[float]) -> None:
    gmsh.model.mesh.set_size([(w.dim, t) for w in wells for t in w.tags], cell_size)


def _to_gmsh(wells: list[Well]) -> list[GmshLine]:
    segment_inds = [well.to_gmsh() for well in wells]
    gmsh.model.occ.synchronize()

    entities = []

    for i, segments in enumerate(segment_inds):
        indices = [s for s in segments]
        entities += [GmshLine(index=i, tags=indices)]

    return entities


def _update_well_grid_tags(
    g: pp.Grid, domain: pp.Domain, mdg: pp.MixedDimensionalGrid
) -> None:
    # Update the tags for the well grid, to identify boundary faces and tips.
    bounding_planes = domain.polytope_from_bounding_box()
    on_domain_boundary = np.zeros(g.num_faces, dtype=bool)
    for plane in bounding_planes:
        if domain.dim == 2:
            plane = np.vstack((plane, np.zeros(plane.shape[1])))
            dist, *_ = pp.geometry.distances.points_segments(
                g.face_centers, plane[:, 0], plane[:, 1]
            )
        else:
            dist, _, _ = pp.geometry.distances.points_polygon(g.face_centers, plane)

        on_domain_boundary = np.logical_or(
            on_domain_boundary, np.isclose(dist.ravel(), 0)
        )

    on_some_boundary = (
        np.bincount(g.cell_faces.tocsc().indices, minlength=g.num_faces) == 1
    )
    g.tags["tip_faces"] = on_some_boundary & np.logical_not(
        on_domain_boundary | g.tags["fracture_faces"]
    )

    g.tags["domain_boundary_faces"] = on_domain_boundary
    if (bg_w := mdg.subdomain_to_boundary_grid(g)) is not None:
        # Overwrite number of cells. This was initialized wrongly before
        # sd_w.tags["domain_boundary_faces"] was set.
        bg_w.num_cells = np.sum(on_domain_boundary)
        bg_w.set_projections()
        bg_w.compute_geometry()


def _check_overlapping_point_grids(
    mdg: pp.MixedDimensionalGrid, well_mdg: pp.MixedDimensionalGrid, tol: float
) -> None:
    """Check that there are no overlapping point grids in the well and fracture
    meshes.

    It should be possible to cover this case with a minor effort at the level of
    geometry (no idea on the constitutive laws), but this has not been prioritized.
    """

    for sd_w in well_mdg.subdomains(dim=0):
        for sd_f in mdg.subdomains(dim=0):
            if np.allclose(sd_w.cell_centers, sd_f.cell_centers, atol=tol):
                raise NotImplementedError(
                    "Coinciding point grids in fracture and well meshes."
                )


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
        sd_primary: Primary subdomain grid. In the context of this module, it
            represents a fracture or well.
        sd_secondary: Secondary subdomain grid. In the context of this module, it
            typically represents an intersection point.
        mdg: MixedDimensionalGrid to which the interface will be added.
        primary_secondary_map: Map between ``cells_l`` and either ``faces_h``
            (codim=1) or ``cells_h`` (codim=2).

    """
    codim = sd_primary.dim - sd_secondary.dim
    subdomain_pair = (sd_primary, sd_secondary)
    side_g = {pp.grids.mortar_grid.MortarSides.LEFT_SIDE: sd_secondary.copy()}
    mg = pp.MortarGrid(dim, side_g, primary_secondary_map, codim=codim)
    mg._primary_to_mortar_int = primary_secondary_map
    mg.compute_geometry()
    mdg.add_interface(mg, subdomain_pair, primary_secondary_map)


def _merge_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if len(arrays) > 0:
        return np.hstack(arrays)
    else:
        return np.array([], dtype=int)


class _PointsOnEntities:
    def __init__(self, entities: list[GmshEntity]) -> None:
        points, inds = [], []
        for entity in entities:
            loc_points, loc_inds = entity.points_on_entity()
            points.extend(loc_points)
            inds.extend(loc_inds)
        self.points = _merge_arrays(points)
        self.inds = inds
