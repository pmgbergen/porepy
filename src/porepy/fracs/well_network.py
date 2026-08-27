"""Collection of classes for the representation of a network of wells in 3d.

The main classes are:
  - WellNetwork3d: Used to represent a collection of wells. Currently mainly used for
    meshing of the wells and their addition to a mixed-dimensional grid.
  - WellFractureIntersection: Used to store information about the intersection between a
    well and a fracture. Primarily intended for internal use during meshing.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Optional, Sequence

import gmsh
import numpy as np
from scipy import sparse as sps

import porepy as pp

from .gmsh_interface import (
    GmshEntity,
    GmshLine,
    PhysicalNames,
    PointsOnGmshEntities,
    fragment,
)
from .wells_3d import Well

if TYPE_CHECKING:
    from porepy.fracs.fracture_network import FractureNetwork


class WellFractureIntersection(NamedTuple):
    """Container class to store intersection between a well and a fracture."""

    coord: np.ndarray
    """Coordinates of the intersection point."""
    point_index: int
    """Index of the intersection point. Assigned in the order in which the intersection
    points are found, used to assign physical names in the gmsh mesh."""
    well_index: int
    """PorePy index of the well to which the intersection point belongs."""
    fracture_index: list[int]
    """List of PorePy indices of the fractures to which the intersection point belongs.
    """
    gmsh_index: int
    """Gmsh index of the intersection point."""


class WellNetwork3d:
    """Collection of :class:`~Well` classes with geometrical information.

    Facilitates meshing of all wells in the network and their addition to a
    mixed-dimensional grid, see e.g. :meth:`~mesh` method.

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
    ) -> None:
        self.domain: pp.Domain = domain
        """Domain specification."""

        self.well_dim: int = 1
        """All polyline wells have dimension 1."""

        self.wells: list[Well] = wells if wells is not None else []
        """List of wells in the network."""

        for i, w in enumerate(self.wells):
            w._index = i

        self.tol: float = tol
        """Geometric tolerance used in computations."""

        # Assign an empty tag dictionary.
        self.tags: dict[str, list[bool]] = dict()

    def mesh(
        self,
        fracture_network: FractureNetwork,
        mdg: pp.MixedDimensionalGrid,
        mesh_args: dict,
    ) -> pp.MixedDimensionalGrid:
        """Mesh the well network and add to the mixed-dimensional grid.

        Parameters:
            fracture_network: Three-dimensional fracture network.
            mdg: Mixed-dimensional grid to which the well grids will be added.
            mesh_args: Dictionary of arguments for the meshing process. Should contain
                a key ``cell_size`` with the mesh size for the well grids.

        """
        if len(self.wells) == 0:
            return mdg

        if not gmsh.is_initialized():
            gmsh.initialize()

        # Process well-fracture intersections. This will also generate a gmsh
        # representation of fractures and wells. Implementation note: Pass the full
        # fracture list, including meshing constraints: gmsh_interface.fragment() relies
        # on each fracture's position in the list matching its ``index`` attribute, so
        # the list cannot be filtered ahead of this call.
        intersections, wells, _ = self.compute_well_fracture_intersections(
            fracture_network.fractures, fracture_network.nd
        )
        intersections = _remove_constraint_only_intersection_points(intersections, mdg)

        # Generate a mesh for the well network and transfer the generated subdomains and
        # interfaces to the mixed-dimensional grid.
        well_mdg = _generate_well_mesh(intersections, wells, mesh_args)
        orig_0d_domain_ids = _add_well_subdomains(mdg, well_mdg, self.tol, self.domain)

        if len(intersections) == 0:
            return mdg

        # Add new interfaces between the well subdomains and the fracture subdomains.
        _add_well_fracture_interfaces(
            mdg, well_mdg, intersections, orig_0d_domain_ids, self.tol
        )

        return mdg

    def compute_well_fracture_intersections(
        self,
        fractures: list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture],
        nd: int,
    ) -> tuple[
        list[WellFractureIntersection], Sequence[GmshEntity], Sequence[GmshEntity]
    ]:
        """Find the intersections between the wells and the fractures.

        Parameters:
            fractures: List of fractures in the fracture network.
            nd: Ambient dimension of the problem.

        Returns:
            intersections: List of well-fracture intersections, grouped in a tailored
                class.
            wells: List of GmshEntity objects corresponding to the wells.
            fractures: List of GmshEntity objects corresponding to the fractures.

        """
        if not gmsh.is_initialized():
            gmsh.initialize()

        # Export fractures and wells to gmsh.
        fracture_tags = _fractures_to_gmsh(fractures)
        segments = _wells_to_gmsh(self.wells)
        # Use Gmsh to find the intersection points between wells and fractures.
        fracture_entities, well_entities = fragment(fracture_tags, segments)

        return (
            _intersections_from_points(
                PointsOnGmshEntities(well_entities),
                PointsOnGmshEntities(fracture_entities),
            ),
            well_entities,
            fracture_entities,
        )

    def to_csv(self, file_name: Path, write_header: bool = True) -> None:
        """Export the well network to a csv file.

        Parameters:
            file_name: Path to the csv file to which to export the well network.
            write_header: Whether to write a header row to the csv file.

        Raises:
            ValueError: If the domain is not boxed.

        """
        file_name = file_name.with_suffix(".csv")

        # Delete the file if it exists. This seems to be necessary to run tests on GH
        # actions.
        if file_name.exists():
            file_name.unlink()

        with open(file_name, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            if write_header:
                csv_writer.writerow("# Well network exported from PorePy.")
                # Note: non-box domains will raise an error below.
                csv_writer.writerow(
                    "# The first line may contain a 6-item bounding box for the domain"
                    " in the format X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX."
                )
                csv_writer.writerow(
                    "# Each row contains the coordinates of the endpoints of all well "
                    "segments of one well, ordered as (x1, y1, z1, x2, y2, z2, ...)."
                )

            # Write the domain bounding box.
            if self.domain is not None:
                if not self.domain.is_boxed:
                    # This should be possible to relax if needed.
                    raise ValueError(
                        "The domain must be boxed to export the well network to csv."
                    )
                self.domain.to_csv(csv_writer)

            # Write all the wells.
            for w in self.wells:
                csv_writer.writerow(w.pts.ravel(order="F"))

    @classmethod
    def from_csv(
        cls, file_name: Path, has_domain: bool = True, domain: pp.Domain | None = None
    ) -> WellNetwork3d:
        """Import a well network from a csv file.

        Parameters:
            file_name: Path to the csv file from which to import the well network. The
                csv file should have the same format as the one exported by the
                ``to_csv`` method.
            has_domain: Whether the csv file contains a domain bounding box.
            domain: The domain to which the well network belongs. If not provided, the
                domain will be inferred from the csv file.

            The domain must either be provided as an argument (domain is not None) or be
            inferred from the csv file (has_domain is True).

        Raises:
            ValueError: If the csv file does not contain a domain bounding box and the
                domain is not provided as an argument, or if the csv file contains a
                domain bounding box and the domain is provided as an argument.

        Returns:
            A new instance of the WellNetwork3d class initialized with the wells from
            the csv file.

        """
        if has_domain and domain is not None:
            raise ValueError(
                "If has_domain is True, the domain must be inferred from the csv file."
            )
        if not has_domain and domain is None:
            raise ValueError(
                "If has_domain is False, the domain must be provided as an argument."
            )

        wells = []
        with open(file_name, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            domain = None
            for row in csv_reader:
                if row[0].startswith("#"):
                    continue
                if domain is None:
                    if len(row) in [4, 6]:
                        domain = pp.Domain.from_numpy_array(row)
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
        s = f"Well network consisting of {len(self.wells)} wells.\n"
        return s


# MARK: Export to Gmsh


def _fractures_to_gmsh(
    fractures: list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture],
) -> list[GmshEntity]:
    """Export a list of fractures to gmsh.

    Parameters:
        fractures: List of fractures to export to gmsh.

    Returns:
        List of GmshEntity objects corresponding to the fractures.
    """
    if len(fractures) == 0:
        return []

    dim = 1 if isinstance(fractures[0], pp.LineFracture) else 2
    entities = []
    for fracture in fractures:
        assert fracture.index is not None, "Fracture index is not set."
        entities.append(
            GmshEntity(
                index=fracture.index, tags=[fracture.fracture_to_gmsh()], dim=dim
            )
        )
    gmsh.model.occ.synchronize()
    return entities


def _wells_to_gmsh(wells: list[Well]) -> list[GmshLine]:
    """Export a list of wells to gmsh.

    Parameters:
        wells: List of wells to export to gmsh.

    Returns:
        List of GmshLine objects corresponding to the wells.
    """
    if len(wells) == 0:
        return []

    segment_inds = [well.to_gmsh() for well in wells]
    gmsh.model.occ.synchronize()

    # The loop in the tags assignment joins all segments that together form a well.
    entities = [
        GmshLine(index=i, tags=[s for s in segments])
        for i, segments in enumerate(segment_inds)
    ]

    return entities


# MARK: Processing of fracture-well intersection information ---


def _remove_constraint_only_intersection_points(
    intersections: list[WellFractureIntersection], mdg: pp.MixedDimensionalGrid
) -> list[WellFractureIntersection]:
    """Meshing constraints are not represented as subdomains in the mdg, and a
    well should be unaffected by crossing one. Drop constraint indices from the
    computed intersections, and discard intersection points that touched only
    constraints (as opposed to kink points, which have an empty fracture_index
    from the start and must be kept).

    Parameters:
        intersections: List of well-fracture intersections.
        mdg: Mixed-dimensional grid containing the fracture subdomains.

    Returns:
        List of well-fracture intersections with constraint indices removed.
    """
    real_frac_nums = {sd.frac_num for sd in mdg.subdomains(dim=mdg.dim_max() - 1)}
    filtered_intersections = []
    for isect in intersections:
        real_frac_inds = [fi for fi in isect.fracture_index if fi in real_frac_nums]
        if len(isect.fracture_index) > 0 and len(real_frac_inds) == 0:
            continue
        if real_frac_inds != isect.fracture_index:
            isect = isect._replace(fracture_index=real_frac_inds)
        filtered_intersections.append(isect)
    return filtered_intersections


def _intersections_from_points(
    well_points: PointsOnGmshEntities, fracture_points: PointsOnGmshEntities
) -> list[WellFractureIntersection]:
    """From a set of points on wells and fractures, find the intersections between wells
    and fractures.

    Parameters:
        well_points: Points on wells.
        fracture_points: Points on fractures.

    Returns:
        List of WellFractureIntersection objects corresponding to the intersections
        between wells and fractures.
    """
    # Find points that are shared between wells and fractures, and points that are
    # shared between well segments.
    common_points = _match_well_and_fracture_points(well_points, fracture_points)
    kink_points = _well_kink_points(well_points, common_points)

    # Loop over the union of the two sets of points, create a WellFractureIntersection
    # object that stores the relevant information for each point.
    merged_intersections: list[WellFractureIntersection] = []
    for ind, ((pi, wi), fi_set) in enumerate((common_points | kink_points).items()):
        coord = gmsh.model.get_bounding_box(0, pi)[:3]
        merged_intersections.append(
            WellFractureIntersection(
                coord=coord,
                point_index=ind,
                well_index=wi,
                fracture_index=list(fi_set),
                gmsh_index=pi,
            )
        )

    return merged_intersections


def _match_well_and_fracture_points(
    well_points: PointsOnGmshEntities, fracture_points: PointsOnGmshEntities
) -> dict[tuple[int, int], set[int]]:
    """Find the points that are shared between wells and fractures. These correspond
    to intersections.

    Parameters:
        well_points: Points on wells.
        fracture_points: Points on fractures.

    Returns:
        Dictionary mapping (point index, well index) to a set of fracture indices.
    """
    # Dictionary that maps a tuple (point index, well index) to a set of fracture
    # indices.
    intersections: dict[tuple[int, int], set[int]] = {}
    # Only register each point-well-fracture combination once.
    visited_point_fracture_combo = set()

    for wi, pi in zip(well_points.inds, well_points.points):
        if pi in fracture_points.points:
            # Find all fractures that contain this point, loop over a unique set of
            # these.
            in_fracture_inds = np.where(fracture_points.points == pi)[0]
            for fi in list(set([fracture_points.inds[i] for i in in_fracture_inds])):
                # Only register each point-well-fracture combination once.
                if (pi, wi, fi) in visited_point_fracture_combo:
                    continue
                visited_point_fracture_combo.add((pi, wi, fi))

                # Add this fracture to the (potentially empty) set of fractures for this
                # point-well combination.
                val = intersections.get((pi, wi), set())
                val.add(fi)
                intersections[(pi, wi)] = val

    return intersections


def _well_kink_points(
    well_points: PointsOnGmshEntities,
    well_fracture_common: dict[tuple[int, int], set[int]],
) -> dict[tuple[int, int], set[int]]:
    """Find points that are shared between wells. These correspond to kinks in the
    well geometry.

    Parameters:
        well_points: Points on wells.
        well_fracture_common: Dictionary mapping (point index, well index) to a set of
            fracture indices.

    Returns:
        Dictionary mapping (point index, well index) to a set of fracture indices. The
        set will be empty, since these are well kink points, not well-fracture
        intersection points.
    """
    # Dictionary that maps (point index, well index) to a set of fracture indices.
    kinks: dict[tuple[int, int], set[int]] = {}

    for wi in np.unique(well_points.inds):
        ind_in_wells = np.where(well_points.inds == wi)[0]
        loc_points = well_points.points[ind_in_wells]
        duplicate_indices = np.where(np.bincount(loc_points) > 1)[0].astype(int)
        for p in duplicate_indices:
            if (p, wi) in well_fracture_common:
                # This is an intersection point, so we do not want to register it as a
                # kink.
                continue
            kinks[(p, wi)] = set()

    return kinks


# MARK: Gmsh mesh generation ---


def _set_physical_names(
    intersections: Sequence[WellFractureIntersection], wells: Sequence[GmshEntity]
) -> None:
    """Set Gmsh physical names for the well-fracture intersection points and the wells.

    Parameters:
        intersections: List of well-fracture intersection points.
        wells: List of GmshEntity objects corresponding to the wells.
    """
    for isect in intersections:
        gmsh.model.addPhysicalGroup(
            0,
            [isect.gmsh_index],
            -1,
            f"{PhysicalNames.WELL_FRACTURE_INTERSECTION_POINT.value}{isect.point_index}",
        )

    for well in wells:
        gmsh.model.addPhysicalGroup(
            1, well.tags, -1, f"{PhysicalNames.WELL.value}{well.index}"
        )


def _set_mesh_size(wells: Sequence[GmshEntity], cell_size: float) -> None:
    """Set the mesh size for the well entities.

    For now, we only allow for a single mesh size for all wells. Improvements can be
    introduced on demand later.

    Parameters:
        wells: List of GmshEntity objects corresponding to the wells.
        cell_size: Mesh size to set for the well entities.
    """
    gmsh.model.mesh.set_size([(w.dim, t) for w in wells for t in w.tags], cell_size)


def _generate_well_mesh(
    intersections: list[WellFractureIntersection],
    wells: Sequence[GmshEntity],
    mesh_args: dict,
) -> pp.MixedDimensionalGrid:
    """Generate a mesh for the wells and the well-fracture intersection points.

    Parameters:
        intersections: List of well-fracture intersection points.
        wells: List of GmshEntity objects corresponding to the wells.
        mesh_args: Dictionary of arguments for the meshing process. Should contain a
            key ``cell_size`` with the mesh size for the well grids.

    Returns:
        Mixed-dimensional grid containing the well meshes and the well-fracture
        intersection points.
    """
    # NOTE: For now, the level of flexibility in meshing of wells is limited compared to
    # that for meshing of fractures. Improvements may be introduced later if this turns
    # out to be desirable.

    # Set physical names for later identification of objects in the mesh. Then set the
    # mesh size for the wells, and generate the mesh.
    _set_physical_names(intersections, wells)
    cell_size = mesh_args.get("cell_size")
    assert cell_size is not None, "Mesh size for wells must be specified."
    _set_mesh_size(wells, cell_size)
    gmsh.model.mesh.generate(1)
    file_name = Path("well_mesh.msh")
    gmsh.write(file_name.as_posix())

    # From the generated mesh, create subdomains for the wells and the intersection
    # points (these are either fracture-well intersections or kinks in the well
    # geometry).
    subdomains = pp.fracs.simplex.line_grid_from_gmsh(
        file_name,
        physical_name_stem_1d=PhysicalNames.WELL.value,
        physical_name_stem_0d=PhysicalNames.WELL_FRACTURE_INTERSECTION_POINT.value,
        sort_1d_nodes=False,
    )
    # Generate a mixed-dimensional grid representing the well network.
    well_mdg = pp.meshing.subdomains_to_mdg(subdomains)
    well_mdg.compute_geometry()

    # The function for generating subdomains will by default interpret the well grids as
    # 1d fractures and assign them a fracture number. Rewrite this to -1 (meaning this
    # is no fracture) and assign a well number instead.
    for wi, wg in enumerate(well_mdg.subdomains(dim=1)):
        wg.well_num = wi
        wg.frac_num = -1

    gmsh.finalize()
    return well_mdg


# MARK: Adding well subdomains and interfaces to the mixed-dimensional grid ---


def _add_well_fracture_interfaces(
    mdg: pp.MixedDimensionalGrid,
    well_mdg: pp.MixedDimensionalGrid,
    intersections: Sequence[WellFractureIntersection],
    orig_0d_domain_ids: list[int],
    tol: float,
) -> None:
    """Add interfaces between the well subdomains and the fracture subdomains.

    Parameters:
        mdg: Mixed-dimensional grid to which the well subdomains will be added.
        well_mdg: Mixed-dimensional grid containing the well subdomains.
        intersections: List of well-fracture intersection points.
        orig_0d_domain_ids: List of original (before the introduction of well
            subdomains) 0-dimensional domain IDs.
        tol: Geometric tolerance used in computations.

    Raises:
        NotImplementedError: If a well-fracture intersection point is located at a
            fracture intersection point.

    """

    def match_point_grid(isect) -> bool:
        """Check if the intersection point is on a 0d subdomain in the mdg.

        For now, only return a boolean indicating if a matching 0d subdomain was found.
        The function may be expanded in the future.

        Parameters:
            isect: Well-fracture intersection point.

        Returns:
            True if a matching 0d subdomain was found, False otherwise.
        """
        found = False
        for sd in mdg.subdomains(dim=0):
            if sd.id in orig_0d_domain_ids and np.isclose(
                np.linalg.norm(sd.cell_centers - isect.coord), 0, atol=tol
            ):
                found = True
                # Implementation note: At this stage we have also identified the 0d
                # subdomain in the mdg that is closest to the intersection point. This
                # can be returned when we need to expand the function to also add
                # well-fracture intersections in fracture intersection points.

        return found

    def match_line_grid(isect, frac_inds) -> pp.Grid:
        """Identify the fracture intersection (1d) subdomain that is closest to the
        intersection point."""

        dist_min = np.inf
        # Loop over the 1d intersection subdomains that are common to all fractures,
        # find the one with a cell center closest to the intersection point. While this
        # is, strictly speaking, not a test of whether the intersection point is on the
        # line, it should be sufficient for our purposes.
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
        """For a set of fractures, find the 1d intersection subdomains that are common
        to all of them."""
        assert mdg.dim_max() == 3, (
            "This function is only valid for 3d fracture networks."
        )
        sds_1d = set(mdg.subdomains(dim=1))
        for fi in frac_inds:
            g_frac = frac_num_to_subdomain[fi]
            sds_1d = sds_1d.intersection(
                mdg.neighboring_subdomains(g_frac, only_lower=True)
            )
        return sds_1d

    def _add_intersection(g_high, g_low):
        """Add an interface between the fracture and well network."""
        embedded_cell = g_high.closest_cell(g_low.cell_centers)

        proj = sps.coo_matrix(
            (np.array([1], dtype=bool), (np.array([0]), embedded_cell)),
            shape=(1, g_high.num_cells),
        ).tocsr()
        mg = pp.MortarGrid(
            0,
            side_grids={pp.grids.mortar_grid.MortarSides.LEFT_SIDE: g_low.copy()},
            primary_secondary=proj,
            codim=g_high.dim - g_low.dim,
        )
        mg.compute_geometry()
        mdg.add_interface(mg, (g_high, g_low), proj)

    point_grid_coord = np.vstack(
        [g.cell_centers[:, 0] for g in well_mdg.subdomains(dim=0)]
    ).T

    # Map from the PorePy fracture index (i.e., the position of the fracture in the
    # full fracture list passed to the fracture network) to the corresponding
    # subdomain in the mdg. This is needed since the two need not be enumerated in
    # the same order.
    frac_num_to_subdomain = {
        sd.frac_num: sd for sd in mdg.subdomains(dim=mdg.dim_max() - 1)
    }

    for isect in intersections:
        frac_inds = isect.fracture_index
        if len(frac_inds) == 0:
            # This is a kink in the well. Continue.
            continue

        # Find the 0d subdomain in the well mdg that is closest to the intersection
        # point.
        ind_0d = np.argmin(
            np.linalg.norm(point_grid_coord - np.reshape(isect.coord, (-1, 1)), axis=0)
        )
        g_low = well_mdg.subdomains(dim=0)[ind_0d]

        # Identify the subdomain from the fracture mdg that is closest to the
        # intersection point. This is either a fracture subdomain or an intersection.
        if len(frac_inds) == 1:
            g_high = frac_num_to_subdomain[frac_inds[0]]
        else:
            # We do not yet support well-fracture intersections at fracture intersection
            # points. If we get to this point and the domain is 2d, we know that this is
            # the case and can raise an error.
            if mdg.dim_max() == 2:
                raise NotImplementedError(
                    "Multiple fractures intersecting at a point is not implemented."
                )
            else:  # mdg.dim_max() == 3
                # We need to check if the fractures intersect at a point or along a
                # line. If the former, we raise an error.
                if match_point_grid(isect):
                    raise NotImplementedError(
                        "Multiple fractures intersecting at a point is not implemented."
                    )
                else:
                    g_high = match_line_grid(isect, frac_inds)

        _add_intersection(g_high, g_low)


def _add_well_subdomains(
    mdg: pp.MixedDimensionalGrid,
    well_mdg: pp.MixedDimensionalGrid,
    tol: float,
    domain: pp.Domain,
) -> list[int]:
    """Add the well subdomains to the mixed-dimensional grid.

    Parameters:
        mdg: Mixed-dimensional grid to which the well subdomains will be added.
        well_mdg: Mixed-dimensional grid containing the well subdomains.
        tol: Geometric tolerance used in computations.
        domain: Domain specification.

    Returns:
        List of original (before the introduction of well subdomains) 0-dimensional
        domain IDs.

    """
    # Check that there are no point grids stemming from the well mesh that are already
    # present as a fracture intersection point. Fixing this will require some
    # bookkeeping, but it should not be a major problem. Still, we don't prioritize this
    # at the moment.
    _check_overlapping_point_grids(mdg, well_mdg, tol)

    orig_0d_domain_ids = [sd.id for sd in mdg.subdomains(dim=0)]

    # Transfer the subdomains and interfaces from the well mdg to the main mdg. This
    # leaves out the well-fracture interfaces, which are added later.
    mdg.add_subdomains(well_mdg.subdomains())
    for intf, data in well_mdg.interfaces(return_data=True):
        sd_primary, sd_secondary = well_mdg.interface_to_subdomain_pair(intf)
        mdg.add_interface(intf, (sd_primary, sd_secondary), data["face_cells"])

    # Also update the tags for the well grids, to identify boundary faces and tips.
    for wg in well_mdg.subdomains(dim=1):
        _update_well_grid_tags_and_boundary_grid(wg, domain, mdg)

    return orig_0d_domain_ids


def _check_overlapping_point_grids(
    mdg: pp.MixedDimensionalGrid, well_mdg: pp.MixedDimensionalGrid, tol: float
) -> None:
    """Check that there are no overlapping point grids in the well and fracture
    meshes.

    It should be possible to cover this case with a minor effort at the level of
    geometry (no idea on the constitutive laws), but this has not been prioritized.

    Parameters:
        mdg: Mixed-dimensional grid containing the fracture mesh.
        well_mdg: Mixed-dimensional grid containing the well mesh.
        tol: Geometric tolerance used in computations.

    Raises:
        NotImplementedError: If there are overlapping point grids in the well and
        fracture meshes.

    """
    for sd_w in well_mdg.subdomains(dim=0):
        for sd_f in mdg.subdomains(dim=0):
            if np.allclose(sd_w.cell_centers, sd_f.cell_centers, atol=tol):
                raise NotImplementedError(
                    "Coinciding point grids in fracture and well meshes."
                )


def _update_well_grid_tags_and_boundary_grid(
    g: pp.Grid, domain: pp.Domain, mdg: pp.MixedDimensionalGrid
) -> None:
    """Update the tags for the well grid, to identify boundary faces and tips.

    Also update the boundary grid for the well grid.

    Parameters:
        g: Well grid.
        domain: Domain specification.
        mdg: Mixed-dimensional grid to which the well grid belongs.

    """
    # Loop over the domain boundary planes, keep track of whether any of the well faces
    # are on it.
    on_domain_boundary = np.zeros(g.num_faces, dtype=bool)
    for plane in domain.polytope_from_bounding_box():
        # Use PorePy's distance functions. These happen to be different in 2d and 3d.
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

    g.tags["domain_boundary_faces"] = on_domain_boundary

    # Use the face-cell connectivity to find faces that are only connected with one
    # cell. These are eiter on the domain boundary, on a fracture face, or they are at
    # the tip of the well.
    on_some_boundary = (
        np.bincount(g.cell_faces.tocsc().indices, minlength=g.num_faces) == 1
    )
    # Fracture faces were marked as part of the well mdg construction. The tip faces
    # are found by excluding the domain boundary and fracture faces.
    g.tags["tip_faces"] = on_some_boundary & np.logical_not(
        on_domain_boundary | g.tags["fracture_faces"]
    )

    if (bg_w := mdg.subdomain_to_boundary_grid(g)) is not None:
        # Overwrite number of cells. This was initialized wrongly before
        # sd_w.tags["domain_boundary_faces"] was set.
        bg_w.num_cells = np.sum(on_domain_boundary)
        bg_w.set_projections()
        bg_w.compute_geometry()
