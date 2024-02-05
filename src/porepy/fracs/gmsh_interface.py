"""Functionality to interface with gmsh, mainly by translating information from a
PorePy format to that used by the gmsh Python API.

For examples on how that can be used, see
:class:`~porepy.fracs.fracture_network_2d.FractureNetwork2d` and
:class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`.

Content

    - :class:`Tags`: Enum with fixed numerical representation of geometric objects.
      Used for tagging geometric objects internally in PorePy.

    - :class:`PhysicalNames`: Enum with fixed string representation of geometric
      objects. The mesh generated by gmsh will use these to tag cells on geometric
      objects.

    - :class:`GmshData1d`, :class:`GmshData2d`, and :class:`GmshData3d`: Dataclasses
      for the specification of geometries in the various dimensions.

    - :class:`GmshWriter`: Interface to Gmsh. Takes a GmshData*d object and translates
       it to a gmsh model. Can also mesh.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import gmsh
import numpy as np

__all__ = [
    "GmshData",
    "GmshData1d",
    "GmshData2d",
    "GmshData3d",
    "Tags",
    "PhysicalNames",
    "GmshWriter",
]


class Tags(Enum):
    """Numerical tags used to identify special objects in a mixed-dimensional geometry.

    These may be mapped to the string-based tag system used in Gmsh (see
    :class:`PhysicalNames`).

    Used internally in porepy.

    """

    NEUTRAL = 0
    """A neutral tag. Objects with this tag will not be accounted for in any
    particular way in grid and geometry processing. """

    DOMAIN_BOUNDARY_POINT = 1
    """Tag to identify points on the domain boundary."""
    DOMAIN_BOUNDARY_LINE = 2
    """Tag to identify lines on the domain boundary."""
    DOMAIN_BOUNDARY_SURFACE = 3
    """Tag to identify surfaces on the domain boundary."""

    FRACTURE = 10
    """Tag for fractures (grids with dimensions lower than the ambient dimension)."""

    AUXILIARY_LINE = 11
    """Tag for auxiliary lines which represent constraints."""
    AUXILIARY_PLANE = 12
    """Tag for auxiliary panes which represent constraints."""

    FRACTURE_TIP = 20
    """Tag for fracture tips. Can be a point or a line."""

    FRACTURE_INTERSECTION_LINE = 21
    """Tag to identify a line resulting from at least two intersecting fractures.

    Ambient dimension must be 3 in this case.

    """

    FRACTURE_BOUNDARY_LINE = 22
    """Tag to identify lines which are both in a fracture and on the domain boundary."""

    FRACTURE_INTERSECTION_POINT = 30
    """Tag to identify points resulting from intersecting fractures.

    Requires at least 2 fractures in 2D, and at least 3 fractures in 3D.

    """

    FRACTURE_CONSTRAINT_INTERSECTION_POINT = 31
    """Tag to identify points at the intersection between fractures and constraints or
    auxiliary lines/planes.

    Ambient dimension must be 3 in this case.

    """

    FRACTURE_BOUNDARY_POINT = 32
    """Tag for points lying on the intersection between a fracture and the domain
    boundary."""


class PhysicalNames(Enum):
    """String-based tags used to assign physical names (Gmsh jargon) to classes of
    objects in a mixed-dimensional geometry.

    The mesh generated by gmsh will use these to tag cells on geometric objects.

    This enumeration contains stringified analoga to the integer tags stored in
    :class:`Tags`.

    Notes:
        1. The neutral tag has no physical name.
           It should not receive special treatment.
        2. Every tag, except for ``DOMAIN`` is usually combined with some other
           identifier. The tags defined here represent a part of the physical name.

    """

    DOMAIN = "DOMAIN"
    """Name tag for the domain, the grid of ambient dimension.

    As of now, only one such domain per mixed-dimensional grid is assumed.

    This may change if domain decomposition is implemented and will be revisited when
    partitioning is included.

    """

    FRACTURE = "FRACTURE_"
    """Name tag for fractures."""

    AUXILIARY_LINE = "AUXILIARY_LINE_"
    """Name tag for auxiliary lines."""
    AUXILIARY_PLANE = "AUXILIARY_PLANE_"
    """Name tag for auxiliary planes."""

    DOMAIN_BOUNDARY_POINT = "DOMAIN_BOUNDARY_POINT_"
    """Name tag for points on the domain boundary."""
    DOMAIN_BOUNDARY_LINE = "DOMAIN_BOUNDARY_LINE_"
    """Name tag for lines on the domain boundary."""
    DOMAIN_BOUNDARY_SURFACE = "DOMAIN_BOUNDARY_SURFACE_"
    """Name tag for surfaces on the domain boundary."""

    FRACTURE_TIP = "FRACTURE_TIP_"
    """Name tag for fracture tips, which can be points or lines."""

    FRACTURE_BOUNDARY_LINE = "FRACTURE_BOUNDARY_LINE_"
    """Name tag for lines which are both in a fracture and on the domain boundary."""

    FRACTURE_INTERSECTION_LINE = "FRACTURE_INTERSECTION_LINE_"
    """Name tag for lines resulting from at least two intersecting fractures.

    Ambient dimension must be 3 in this case.

    """

    FRACTURE_INTERSECTION_POINT = "FRACTURE_INTERSECTION_POINT_"
    """Name tag for points resulting from intersecting fractures.

    Requires at least 2 fractures in 2D, and at least 3 fractures in 3D.

    """

    FRACTURE_BOUNDARY_POINT = "FRACTURE_BOUNDARY_POINT_"
    """Name tag for points lying on the intersection between a fracture and the domain
    boundary."""

    FRACTURE_CONSTRAINT_INTERSECTION_POINT = "FRACTURE_CONSTRAINT_INTERSECTION_POINT_"
    """Name tag to identify points at the intersection between fractures and constraints
    or auxiliary lines/planes.

    Ambient dimension must be 3 in this case.

    """


def _tag_to_physical_name(tag: int | Tags) -> str:
    """Maps the numerical enumeration to the physical name of
    enumerated categories of elements in a mixed-dimensional grid.

    Parameters:
        tag: Number of enumeration to be mapped.

    Raises:
        KeyError: If ``tag`` not in :class:`Tags`

    Returns:
        The string-value of corresponding enumeration in :class:`PhysicalNames`

    """
    # Convenience function to map from numerical to string representation of a geometric object
    #    if isinstance(tag, Tags):
    t = Tags(tag)
    #    else:
    #        t = Tags(tag)
    for pn in PhysicalNames:
        if pn.name == t.name:
            return pn.value

    raise KeyError(f"Found no physical name corresponding to tag {tag}")


@dataclass
class GmshData:
    """Data class to store information for a mesh coming from ``gmsh``, which are
    common in every dimension.

    This class is not supposed to be used, but its derivatives for respective
    dimension.

    """

    pts: np.ndarray
    """Points in the geometry."""

    mesh_size: np.ndarray
    """Mesh size for each point in ``pts``."""

    lines: np.ndarray
    """1D lines needed for the description of the geometry.

    The third row should contain tags for the lines (note that these are needed for
    all lines, whereas the field physical lines is only used for lines that should be
    represented in the output mesh file).

    """

    physical_points: dict[int, Tags]
    """A mapping from point indices (ordered as in ``pts``) to numerical tags for
    respective points. """

    physical_lines: dict[int, Tags]
    """A mapping from indices of lines (ordered as in ``lines``) to numerical tags
    for respective lines. """


@dataclass
class GmshData1d(GmshData):
    """A derived data class containing additional information for 1D meshes.

    Note:
        1D data is not fully supported yet.

    """

    dim: int = 1
    """The dimension is 1."""


@dataclass
class GmshData2d(GmshData):
    """A derived data class containing additional information for 2D meshes."""

    dim: int = 2
    """The dimension is 2."""


@dataclass
class GmshData3d(GmshData):
    """A derived data class containing additional information for 3D meshes."""

    polygons: tuple[list[np.ndarray], list[np.ndarray]]
    """Tuple with 2 lists:

    Each item of the first  list is a numpy array of integers with the indices of the
    edges need to form a polygon (fracture, boundary) in the decomposition.

    Each item of the list is boolean numpy array, specifying whether the edge must be
    reversed (relative to its description).

    See :class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d` for more.

    """

    polygon_tags: dict[int, Tags]
    """A map containing tags to identify the type of a polygon, per index of a
    polygon in ``polygons. """

    physical_surfaces: dict[int, Tags]
    """A mapping containing the tags per index of a polygon in ``polygons``.

    This is used to set gmsh tags for objects to be presented in the output mesh.

    """

    lines_in_surface: list[list[int]]
    """A list of indices of lines in ``lines`` embedded in surfaces.

    The outer list has one item per polygon, the inner list has indices of embedded
    lines.

    """

    dim: int = 3
    """The dimension 3."""


class GmshWriter:
    """Interface to the gmsh python API.

    The geometry specification (in the form of a GmshData object) is converted to a
    model (in the Gmsh sense) upon initiation of a GmshWriter. A mesh can then be
    constructed either by the method :meth:`generate`, or by directly working with
    the gmsh api in a client script.

    Parameters:
        data: Geometry specifications

    """

    gmsh_initialized: bool = False
    """A flag keeping track of whether gmsh has been initialized.

    Important:
        This does not account for calls to ``gmsh.initialize`` and
        ``gmsh.finalize`` outside the class.

    """

    def __init__(self, data: GmshData2d | GmshData3d) -> None:
        self._dim: int = data.dim
        """The ambient dimension of the mesh."""
        self._data: Union[GmshData2d, GmshData3d] = data
        """The mesh data passed at instantiation."""

        self._point_tags: list[int]
        """List of (numerical) point tags in Gmsh.

        Set during a call to :meth:`define_geometry`, which is performed during
        instantiation.

        """

        self._domain_tag: int
        """The (numerical) domain tag for the geometry.

        Set during a call to :meth:`define_geometry`, which is performed during
        instantiation.

        """

        self._line_tags: list[int]
        """A list of (numerical) line tags for lines in the geometry.

        Set during a call to :meth:`define_geometry`, which is performed during
        instantiation.

        """

        self._frac_tags: list[int]
        """A list of (numerical) fracture tags for fractures in the geometry.

        Set during a call to :meth:`define_geometry`, which is performed during
        instantiation.

        """

        self.define_geometry()

    def set_gmsh_options(self, options: Optional[dict] = None) -> None:
        """Set gmsh options. See gmsh documentation for choices available.

        Parameters:
            options: Options to set. Keys should be recognizable for gmsh.

        """
        if options is None:
            options = {}

        for key, val in options:
            if isinstance(val, (int, float)):
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

    def define_geometry(self) -> None:
        """Feed the geometry passed at instantiation to gmsh.

        Sets the tags for points, lines and domain and stores them in this instance.

        """

        # Initialize gmsh if this is not done before
        if not GmshWriter.gmsh_initialized:
            gmsh.initialize()
            GmshWriter.gmsh_initialized = True

        gmsh.option.setNumber("General.Verbosity", 3)

        # All geometries need their points
        self._point_tags = self._add_points()

        # The boundary and the domain, must be specified prior to the co-dimension 1
        # objects (fractures) so that the latter can be embedded in the domain. This
        # requires some specialized treatment in 2d and 3d, since the boundary and
        # fractures are made of up different geometric objects in the two:
        if self._dim == 1:
            # This should not be difficult, but has not been prioritized.
            raise NotImplementedError("1d geometries are not implemented")

        elif self._dim == 2:
            # First add the domain. This will add the lines that form the domain
            # boundary, but not the fracture lines.
            self._domain_tag = self._add_domain_2d()

            # Next add the fractures (and constraints). This adds additional lines,
            # and embeds them in the domain.
            self._add_fractures_2d()

        else:
            # In 3d, we can first add lines to the description, they are only used to
            # define surfaces later.

            # Make a index list covering all the lines
            inds = np.arange(self._data.lines.shape[1])

            # The function to write lines assumes that the lines are formed by point
            # indices (first two rows), tags (third) and line index (fourth) Add the
            # latter, essentially saying that each line is a separate line.
            if self._data.lines.shape[0] < 4:
                self._data.lines = np.vstack((self._data.lines, inds))

            # Now we can write the lines. The lines are not directly embedded in the
            # domain. NOTE: If we ever get around to do a 1d-3d model (say a well),
            # this may be a place to do that.
            self._line_tags = self._add_lines(inds, embed_in_domain=False)

            # Next add the domain. This adds the boundary surfaces, embeds lines in
            # the relevant boundary surfaces, and constructs a 3d volume from the
            # surfaces
            self._domain_tag = self._add_domain_3d()

            # Finally, we can add the fractures and constraints and embed them in the
            # domain
            self._add_fractures_3d()

        gmsh.model.geo.synchronize()

    def generate(
        self,
        file_name: str,
        ndim: int = -1,
        write_geo: bool = False,
        clear_gmsh: bool = True,
        finalize: bool = True,
    ) -> None:
        """Make gmsh generate a mesh and write it to specified mesh size.

        The mesh is generated from the geometry specified in gmsh.model.

        Note:
            We have experienced issues relating to memory leakages in gmsh which
            manifest when gmsh is initialized and finalized several times in a session.
            In these situation, best practice seems to be not to finalize gmsh after
            mesh generation (set ``finalize=False``, but rather clear the gmsh model by
            setting ``clear_gmsh=True``), and finalize gmsh from the outside.

        Parameters:
            file_name: Name of the file. The suffix ``.msh`` is added if necessary.
            ndim: ``default=-1``

                Dimension of the grid to be made. If not specified, the
                dimension of the data used to initialize this GmshWriter will be used.
            write_geo: ``default=False``

                If True, the geometry will be written before meshing.
                The name of the file will be ``file_name.geo_unrolled``.
            clear_gmsh: ``default=True``

                If True, the function ``gmsh.clear`` is called after
                mesh generation. This will delete the geometry from gmsh.
            finalize: ``default=True``

                If True, the finalize method of the gmsh module is called after mesh
                generation.

                If set to False, gmsh should be finalized by a direct call to
                ``gmsh.finalize``.
                Note however that if this is done, gmsh cannot be accessed either from
                the outside or by an instance of the GmshWriter class before
                ``gmsh.initialize`` is called.

        """
        if ndim == -1:
            ndim = self._dim
        if file_name[-4:] != ".msh":
            file_name = file_name + ".msh"
        if write_geo:
            fn = file_name[:-4] + ".geo_unrolled"
            gmsh.write(fn)

        for dim in range(1, ndim + 1):
            try:
                gmsh.model.mesh.generate(dim=dim)
            except Exception as exc:
                s = f"Gmsh meshing failed in dimension {dim}. Error message:\n"
                s += str(exc)
                print(s)
                raise exc
        gmsh.write(file_name)

        if clear_gmsh:
            gmsh.clear()
        if finalize:
            gmsh.finalize()
            GmshWriter.gmsh_initialized = False

    def _add_points(self) -> list[int]:
        """Auxiliary function to add points to gmsh.

        Also sets physical names for points if required.

        Returns:
            A list of numerical tags (enumeration) for each point.

        """
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
                phys_name = _tag_to_physical_name(point_type) + str(pi)
                gmsh.model.geo.synchronize()
                ps = gmsh.model.addPhysicalGroup(0, [point_tags[-1]])
                gmsh.model.setPhysicalName(0, ps, phys_name)
        return point_tags

    # Write fractures

    def _add_fractures_1d(self) -> None:
        """
        Raises:
            NotImplementedError: 1D domains are not supported yet.

        """
        raise NotImplementedError("1d domains are not supported yet")

    def _add_fractures_2d(self) -> None:
        """Auxiliary function to add all fracture lines to gmsh.

        Only lines that correspond to fractures or auxiliary lines are added.
        Boundary lines are added elsewhere.

        """
        # NOTE: Here we operate on the numerical tagging of lines (the attribute edges
        # in FractureNetwork2d), thus we must compare with the values of the Tags.
        inds = np.argwhere(
            np.logical_or.reduce(
                (
                    self._data.lines[2] == Tags.FRACTURE.value,
                    self._data.lines[2] == Tags.AUXILIARY_LINE.value,
                )
            )
        ).ravel()
        self._frac_tags = self._add_lines(inds, embed_in_domain=True)

    def _add_fractures_3d(self) -> None:
        """Helper function to add fracture polygons to gmsh.

        Polygons tagged as fractures or auxiliary planes are added.

        """

        if not isinstance(self._data, GmshData3d):
            raise ValueError("Need 3d geometry to write fractures")

        # Loop over the polygons, find those tagged as fractures or auxiliary planes
        # (which could be constraints in the meshing). Add these to the information
        # to be written.
        indices_to_write = []
        for ind, tag in self._data.polygon_tags.items():
            if tag in (Tags.FRACTURE, Tags.AUXILIARY_PLANE):
                indices_to_write.append(ind)

        self._frac_tags = self._add_polygons_3d(indices_to_write, embed_in_domain=True)

    def _add_polygons_3d(self, inds: list[int], embed_in_domain: bool) -> list[int]:
        """Generic method to add polygons to a 3d geometry.

        Parameters:
            inds: A list of point indices for points spanning the polygon.
            embed_in_domain: A flag to embed the polygonial surface into the domain
                using gmsh.

        Returns:
            A list of numerical tags (enumeration) for the added polygons.

        """

        if not isinstance(self._data, GmshData3d):
            raise ValueError("Need 3d geometry to write polygons")

        if self._data.physical_surfaces is not None:
            phys_surf = self._data.physical_surfaces
        else:
            phys_surf = {}

        gmsh.model.geo.synchronize()

        line_dim = 1
        surf_dim = 2

        surf_tag = []

        to_phys_tags: list[tuple[int, str, list[int]]] = []

        for pi in inds:
            line_tags = []
            for line in self._data.polygons[0][pi]:
                line_tags.append(self._line_tags[line])

            loop_tag = gmsh.model.geo.addCurveLoop(line_tags, reorient=True)
            surf_tag.append(gmsh.model.geo.addPlaneSurface([loop_tag]))

            # Register the surface as physical if relevant. This will make gmsh export
            # the cells on the surface.
            if pi in phys_surf:
                physical_name = _tag_to_physical_name(phys_surf[pi])
                to_phys_tags.append((pi, physical_name + str(pi), [surf_tag[-1]]))

        # Update the model with all added surfaces
        gmsh.model.geo.synchronize()

        # Add the surfaces as physical tags if so specified.
        for pi, phys_name, tag in to_phys_tags:
            ps = gmsh.model.addPhysicalGroup(surf_dim, tag)
            gmsh.model.setPhysicalName(surf_dim, ps, phys_name)

        # Embed the surface in the domain
        if embed_in_domain:
            for tag in surf_tag:
                gmsh.model.mesh.embed(surf_dim, [tag], self._dim, self._domain_tag)

        gmsh.model.geo.synchronize()

        # For all surfaces, embed lines in ... Do this after all surfaces have been
        # added to get away with a single synchronization
        for tag_ind, pi in enumerate(inds):
            for li in self._data.lines_in_surface[pi]:
                # note the use of indices here, different for lines, polygons and the
                # ordering of surface tags
                gmsh.model.mesh.embed(
                    line_dim, [self._line_tags[li]], surf_dim, surf_tag[tag_ind]
                )

        return surf_tag

    def _add_lines(self, ind: np.ndarray, embed_in_domain: bool) -> list[int]:
        """Helper function to add lines in the geometry to gmsh.

        Parameters:
            ind: An array of indices for lines to be added.
            embed_in_domain: A flag to embed the line into the domain using gmsh.

        Returns:
            A list of numerical tags (enumeration) for the added lines.

        """
        # Helper function to write lines.
        line_tags: list[int] = []

        if ind.size == 0:
            return line_tags

        lines = self._data.lines[:, ind]

        tag = lines[2]

        lines_id = lines[3, :]

        range_id = np.arange(np.amin(lines_id), np.amax(lines_id) + 1)

        line_dim = 1

        has_physical_lines = hasattr(self._data, "physical_lines")

        # Temporary storage of the lines that are to be assigned physical groups
        to_physical_group: list[tuple[int, str, list[int]]] = []

        for i in range_id:
            loc_tags = []
            for mask in np.flatnonzero(lines_id == i):
                p0 = self._point_tags[lines[0, mask]]
                p1 = self._point_tags[lines[1, mask]]
                loc_tags.append(gmsh.model.geo.addLine(p0, p1))
                # Get hold of physical_name, in case we need it
                physical_name = _tag_to_physical_name(tag[mask])

            # Store local tags
            line_tags += loc_tags

            # Add this line to the set of physical groups to be assigned. We do not
            # assign physical groupings inside this for-loop, as this would require
            # multiple costly synchronizations (gmsh style).
            if has_physical_lines and i in self._data.physical_lines:
                to_physical_group.append((i, physical_name, loc_tags))

        # Synchronize model with all new lines
        gmsh.model.geo.synchronize()

        # Assign physical name to the line if specified. This will make gmsh
        # represent the line as a separate physical object in the .msh file
        for i, physical_name, loc_tags in to_physical_group:
            phys_group = gmsh.model.addPhysicalGroup(line_dim, loc_tags)
            gmsh.model.setPhysicalName(line_dim, phys_group, f"{physical_name}{i}")

        # Syncronize model, and embed all lines in the domain if specified
        if embed_in_domain:
            gmsh.model.mesh.embed(line_dim, line_tags, self._dim, self._domain_tag)

        return line_tags

    def _add_domain_2d(self) -> int:
        """Write boundary lines and tie them together to a closed loop to define the
        domain.

        This is a helper function for 2D domains.

        Returns:
            The numerical tag (enumeration) for the domain.

        """

        # Here we operate on ``FractureNetwork2d.edges``, thus we should use the
        # numerical values of the tag for comparison.
        bound_line_ind = np.argwhere(
            self._data.lines[2] == Tags.DOMAIN_BOUNDARY_LINE.value
        ).ravel()

        bound_line_tags = self._add_lines(bound_line_ind, embed_in_domain=False)

        self._bound_line_tags = bound_line_tags

        loop_tag = gmsh.model.geo.addCurveLoop(bound_line_tags, reorient=True)
        domain_tag = gmsh.model.geo.addPlaneSurface([loop_tag])

        gmsh.model.geo.synchronize()
        phys_group = gmsh.model.addPhysicalGroup(2, [domain_tag])
        gmsh.model.setPhysicalName(2, phys_group, PhysicalNames.DOMAIN.value)
        return domain_tag

    def _add_domain_3d(self) -> int:
        """Write boundary surfaces to define the domain.

        This is a helper function for 3D domains.

        Returns:
            The numerical tag (enumeration) for the domain.

        """
        if not isinstance(self._data, GmshData3d):
            raise ValueError("Need 3d geometry to specify 3d domain")

        inds = []
        for i, tag in self._data.polygon_tags.items():
            if tag == Tags.DOMAIN_BOUNDARY_SURFACE:
                inds.append(i)

        bound_surf_tags = self._add_polygons_3d(inds, embed_in_domain=False)

        self._bound_surf_tags = bound_surf_tags

        loop_tag = gmsh.model.geo.addSurfaceLoop(bound_surf_tags)
        domain_tag = gmsh.model.geo.addVolume([loop_tag])
        gmsh.model.geo.synchronize()

        phys_group = gmsh.model.addPhysicalGroup(3, [domain_tag])
        gmsh.model.setPhysicalName(3, phys_group, PhysicalNames.DOMAIN.value)
        return domain_tag
