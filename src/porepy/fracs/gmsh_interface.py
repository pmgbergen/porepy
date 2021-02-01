"""
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List

import porepy as pp
import numpy as np

import gmsh

__all__ = ["GmshData1d", "GmshData2d", "GmshData3d", "Tags", "GmshWriter"]


class Tags(Enum):
    """Numerical tags used to identify special objects in a mixed-dimensional
    geometry. These may be mapped to the string-based tag system used in Gmsh
    (see PhysicalNames)
    """

    # A neutral tag - objects with this tag will not be accounted for in any praticular
    # way in grid and geometry processing.
    NEUTRAL = 0

    # Objects used to define the domain boundary
    DOMAIN_BOUNDARY_POINT = 1
    DOMAIN_BOUNDARY_LINE = 2
    DOMAIN_BOUNDARY_SURFACE = 3

    FRACTURE = 10
    AUXILIARY = 11

    # A fracture tip - can be point or line
    FRACTURE_TIP = 20

    # Intersection line between (at least) two fractures. Ambient dimension is 3.
    FRACTURE_INTERSECTION_LINE = 21

    # Line on both fracture and a domain boundary
    FRACTURE_BOUNDARY_LINE = 22

    # Point on the intersection between fractures.
    # Will at least two fractures in 2d, at least three fractures in 3d.
    FRACTURE_INTERSECTION_POINT = 30

    FRACTURE_CONSTRAINT_INTERSECTION_POINT = 31

    # Intersection point between fracture and domain boundary
    FRACTURE_BOUNDARY_POINT = 12


class PhysicalNames(Enum):
    """String-based tags used to assign physical names (Gmsh jargon) to classes
    of objects in a mixed-dimensional geometry.
    """

    # Note that neutral tags have no physical names - they should not receive special treatment.

    # The assumption (for now) is that there is a single boundary
    # This may change if we implement DD - but then we need to revisit these concepts to
    # include (metis?) partitioning
    DOMAIN = "DOMAIN"

    FRACTURE = "FRACTURE_"
    AUXILIARY = "AUXILIARY_"

    # Objects used to define the domain boundary
    DOMAIN_BOUNDARY_POINT = "DOMAIN_BOUNDARY_POINT_"

    DOMAIN_BOUNDARY_LINE = "DOMAIN_BOUNDARY_LINE_"

    DOMAIN_BOUNDARY_SURFACE = "DOMAIN_BOUNDARY_SURFACE_"

    # A fracture tip - can be point or line
    FRACTURE_TIP = "FRACTURE_TIP_"

    # Line on both fracture and a domain boundary
    FRACTURE_BOUNDARY_LINE = "FRACTURE_BOUNDARY_LINE_"

    # Intersection line between (at least) two fractures. Ambient dimension is 3.
    FRACTURE_INTERSECTION_LINE = "FRACTURE_INTERSECTION_LINE_"

    # Point on the intersection between fractures.
    # Will at least two fractures in 2d, at least three fractures in 3d.
    FRACTURE_INTERSECTION_POINT = "FRACTURE_INTERSECTION_POINT_"

    # Intersection point between fracture and domain boundary
    FRACTURE_BOUNDARY_POINT = "FRACTURE_BOUNDARY_POINT_"

    # Point defined in the intersection between a fracture and constraints / auxiliary planes.
    # Ambient dimension is 3.
    FRACTURE_CONSTRAINT_INTERSECTION_POINT = "FRACTURE_CONSTRAINT_INTERSECTION_POINT_"


def tag_to_physical_name(tag: int) -> str:
    t = Tags(tag)
    for pn in PhysicalNames:
        if pn.name == t.name:
            return pn.value

    raise KeyError(f"Found no physical name corresponding to tag {tag}")


@dataclass
class _GmshData:
    pts: np.ndarray
    mesh_size: np.ndarray
    lines: np.ndarray
    physical_points: Dict[int, Tags]


@dataclass
class GmshData1d(_GmshData):
    dim: int = 1


@dataclass
class GmshData2d(_GmshData):

    physical_lines: Dict[int, Tags]
    #    physical_boundary: bool = True
    dim: int = 2


@dataclass
class GmshData3d(_GmshData):

    polygons: List[np.ndarray]
    polygon_tags: np.ndarray
    physical_surfaces: Dict[int, Tags]
    lines_in_surface: List[List[int]]
    physical_lines: Dict[int, Tags]

    dim: int = 3


class GmshWriter:
    def __init__(self, data: _GmshData):
        self._dim = data.dim
        self._data = data

        self.define_geometry()

    def set_gmsh_options(self, options=None):
        if options is None:
            options = {}

        for key, val in options:
            if isinstance(val, int) or isinstance(val, float):
                try:
                    gmsh.option.setNumber(key, val)
                except Exception:
                    raise ValueError(
                        f"Could not set numeric gmsh option with key {key}"
                    )
            elif isinstance(val, str):
                try:
                    gmsh.option.setString(key, val)
                except Exception:
                    raise ValueError(f"Could not set string gmsh option with key {key}")

    def define_geometry(self):
        gmsh.initialize()

        # All geometries need their points
        self._point_tags = self._add_points()

        # The boundary, and the domain, must be specified prior to the
        # co-dimension 1 objects (fractures) so that the latter can be
        # embedded in the domain. This requires some specialized treatment
        # in 2d and 3d, since the boundary and fractures are made of up
        # different geometric objects in the two:
        if self._dim == 1:
            self._add_fractures_1d()
        elif self._dim == 2:
            # First add the domain. This will add the lines that form the domain
            # boundary, but not the fracture lines.
            self._domain_tag = self._add_domain_2d()

            # Next add the fractures (and constraints). This adds additional lines,
            # and embeds them in the domain.
            self._add_fractures_2d()

        else:
            # In 3d, we can first add lines to the description, they are only used
            # to define surfaces later.

            # Make a index list covering all the lines
            inds = np.arange(self._data.lines.shape[1])

            # The function to write lines assumes that the lines are formed by
            # point indices (first two rows), tags (third) and line index (fourth)
            # Add the latter, essentially saying that each line is a separate line.
            if self._data.lines.shape[0] < 4:
                self._data.lines = np.vstack((self._data.lines, inds))

            # Now we can write the lines
            # The lines are not directly embedded in the domain.
            # NOTE: If we ever get around to do a 1d-3d model (say a well), this may
            # be a place to do that.
            self._line_tags = self._add_lines(inds, embed_in_domain=False)

            # Next add the domain. This adds the boundary surfaces, embeds lines in
            # the relevant boundary surfaces, and constructs a 3d volume from the
            # surfaces
            self._domain_tag = self._add_domain_3d()

            # Finally, we can add the fractures and constraints and embed them in
            # the domain
            self._add_fractures_3d()

        gmsh.model.geo.synchronize()

    def generate(self, file_name: str, ndim: int = -1, write_geo=False):
        if ndim == -1:
            ndim = self._dim
        if write_geo:
            fn = file_name[:-4] + ".geo_unrolled"
            gmsh.write(fn)
        gmsh.model.mesh.generate(dim=ndim)
        gmsh.write(file_name)
        gmsh.finalize()

    def _add_points(self):
        point_tags = []
        for pi, sz in enumerate(self._data.mesh_size):

            if self._dim == 2:
                x, y = self._data.pts[:, pi]
                z = 0
            else:
                x, y, z = self._data.pts[:, pi]
            point_tags.append(gmsh.model.geo.addPoint(x, y, z, sz))

            if pi in self._data.physical_points:
                point_type = self._data.physical_points[pi]
                phys_name = tag_to_physical_name(point_type) + str(pi)
                gmsh.model.geo.synchronize()
                ps = gmsh.model.addPhysicalGroup(0, [point_tags[-1]])
                gmsh.model.setPhysicalName(0, ps, phys_name)
        return point_tags

    # Write fractures

    def _add_fractures_1d(self):
        assert False

    def _add_fractures_2d(self):
        # Only add the lines that correspond to fractures or auxiliary lines.
        # Boundary lines are added elsewhere.
        inds = np.argwhere(
            np.logical_or.reduce(
                (
                    self._data.lines[2] == Tags.FRACTURE.value,
                    self._data.lines[2] == Tags.AUXILIARY.value,
                )
            )
        ).ravel()
        self._add_lines(inds, embed_in_domain=True)

        # We consider fractures and boundary tag

    def _add_fractures_3d(self):
        inds = np.argwhere(
            np.logical_or.reduce(
                (
                    self._data.polygon_tags == Tags.FRACTURE.value,
                    self._data.polygon_tags == Tags.AUXILIARY.value,
                )
            )
        ).ravel()
        self._add_polygons_3d(inds, embed_in_domain=True)

    def _add_polygons_3d(self, inds, embed_in_domain):

        phys_surf = self._data.physical_surfaces

        gmsh.model.geo.synchronize()

        line_dim = 1
        surf_dim = 2

        surf_tag = []
        for pi in inds:
            line_tags = []
            for line in self._data.polygons[0][pi]:
                line_tags.append(self._line_tags[line])

            loop_tag = gmsh.model.geo.addCurveLoop(line_tags, reorient=True)
            surf_tag.append(gmsh.model.geo.addPlaneSurface([loop_tag]))

            # Register the surface as physical if relevant. This will make gmsh export
            # the cells on the surface.
            if pi in phys_surf:
                gmsh.model.geo.synchronize()
                physical_name = tag_to_physical_name(phys_surf[pi])
                ps = gmsh.model.addPhysicalGroup(surf_dim, [surf_tag[-1]])
                gmsh.model.setPhysicalName(surf_dim, ps, physical_name + str(pi))

            gmsh.model.geo.synchronize()

            # Embed the surface in the domain
            if embed_in_domain:
                gmsh.model.mesh.embed(
                    surf_dim, [surf_tag[-1]], self._dim, self._domain_tag
                )
            gmsh.model.geo.synchronize()

            # Emded lines in this surface
            for li in self._data.lines_in_surface[pi]:
                gmsh.model.mesh.embed(
                    line_dim, [self._line_tags[li]], surf_dim, surf_tag[-1]
                )
        gmsh.model.geo.synchronize()
        return surf_tag

    def _add_lines(self, ind, embed_in_domain):
        # Helper function to write lines. Used for both
        line_tags = []

        if ind.size == 0:
            return line_tags

        lines = self._data.lines[:, ind]

        tag = lines[2]

        lines_id = lines[3, :]

        range_id = np.arange(np.amin(lines_id), np.amax(lines_id) + 1)

        line_dim = 1

        has_physical_lines = hasattr(self._data, "physical_lines")

        for i in range_id:
            loc_tags = []
            for mask in np.flatnonzero(lines_id == i):
                p0 = self._point_tags[lines[0, mask]]
                p1 = self._point_tags[lines[1, mask]]
                loc_tags.append(gmsh.model.geo.addLine(p0, p1))

                physical_name = tag_to_physical_name(tag[mask])

            # Assign physical name to the line if specified. This will make gmsh
            # represent the line as a separate physical object in the .msh file
            if has_physical_lines and i in self._data.physical_lines:
                gmsh.model.geo.synchronize()
                phys_group = gmsh.model.addPhysicalGroup(line_dim, loc_tags)
                gmsh.model.setPhysicalName(line_dim, phys_group, f"{physical_name}{i}")

            line_tags += loc_tags

            gmsh.model.geo.synchronize()
            if embed_in_domain:
                gmsh.model.mesh.embed(line_dim, loc_tags, self._dim, self._domain_tag)

        return line_tags

    # Write domains

    def _add_domain_2d(self):
        bound_line_ind = np.argwhere(
            self._data.lines[2] == Tags.DOMAIN_BOUNDARY_LINE.value
        ).ravel()

        bound_line_tags = self._add_lines(bound_line_ind, embed_in_domain=False)
        gmsh.model.geo.synchronize()
        #        line_group = gmsh.model.addPhysicalGroup(1, bound_line_tags)
        #        gmsh.model.setPhysicalName(1, line_group, PhysicalNames.DOMAIN_BOUNDARY.value)
        #       breakpoint()
        loop_tag = gmsh.model.geo.addCurveLoop(bound_line_tags, reorient=True)
        domain_tag = gmsh.model.geo.addPlaneSurface([loop_tag])
        gmsh.model.geo.synchronize()
        phys_group = gmsh.model.addPhysicalGroup(2, [domain_tag])
        gmsh.model.setPhysicalName(2, phys_group, PhysicalNames.DOMAIN.value)
        return domain_tag

    def _add_domain_3d(self):

        inds = np.argwhere(
            self._data.polygon_tags == Tags.DOMAIN_BOUNDARY_SURFACE.value
        ).ravel()

        bound_surf_tags = self._add_polygons_3d(inds, embed_in_domain=False)
        gmsh.model.geo.synchronize()
        bound_group = gmsh.model.addPhysicalGroup(2, bound_surf_tags)
        gmsh.model.setPhysicalName(
            2, bound_group, PhysicalNames.DOMAIN_BOUNDARY_SURFACE.value
        )

        gmsh.model.geo.synchronize()
        loop_tag = gmsh.model.geo.addSurfaceLoop(bound_surf_tags)
        domain_tag = gmsh.model.geo.addVolume([loop_tag])
        gmsh.model.geo.synchronize()

        phys_group = gmsh.model.addPhysicalGroup(3, [domain_tag])
        gmsh.model.setPhysicalName(3, phys_group, PhysicalNames.DOMAIN.value)
        return domain_tag
