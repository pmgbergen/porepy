"""Module with functionality for converting gmsh output file to PorePy's grid structure.

Todo:
    Elaborate more arguments ``cell_info`` and ``phys_names``
    (admissible keywords etc.)

"""

from __future__ import annotations

from typing import Optional

import numpy as np

import porepy as pp

from .gmsh_interface import PhysicalNames


def create_3d_grids(pts: np.ndarray, cells: dict[str, np.ndarray]) -> list[pp.Grid]:
    """Create a tetrahedral grid from a gmsh tesselation.

    Parameters:
        pts: ``shape=(num_points, 3)``

            Global point set from gmsh.
        cells: Should have a key ``'tetra'`` which maps to an :obj:`~numpy.ndarray` with
            indices of the points that form 3D grids.

    Returns:
        A list of 3D grids.

    """

    tet_cells = cells["tetra"]
    g_3d = pp.TetrahedralGrid(pts.transpose(), tet_cells.transpose())

    # Create mapping to global numbering (will be a unit mapping, but is
    # crucial for consistency with lower dimensions)
    g_3d.global_point_ind = np.arange(pts.shape[0])

    # Convert to list to be consistent with lower dimensions
    # This may also become useful in the future if we ever implement domain
    # decomposition approaches based on gmsh.
    return [g_3d]


def create_2d_grids(
    pts: np.ndarray,
    cells: dict[str, np.ndarray],
    phys_names: dict[int, str],
    cell_info: dict[str, np.ndarray],
    is_embedded: bool = False,
    surface_tag: Optional[str] = None,
    constraints: Optional[np.ndarray] = None,
) -> list[pp.Grid]:
    """Create 2D grids for lines of a specified type from a gmsh tesselation.

    Only surfaces that were defined as 'physical' in the gmsh sense may have a grid
    created, but then only if the physical name matches the specified ``surface_tag``.

    It is assumed that the mesh is read by meshio. See module
    :mod:`~porepy.fracs.simplex` for how to do this.

    Parameters:
        pts: ``shape=(num_points, 3)``

            Global point set from gmsh.
        cells: Should have a key ``'triangle'`` which maps to an :obj:`~numpy.ndarray`
            with indices of the points that form 2D grids.
        phys_names: mapping from the gmsh tags assigned to physical entities to the
            physical name of that tag.
        cell_info: Should have a key ``'triangle'`` that contains the physical names (in
            the gmsh sense) of the points.
        is_embedded: ``default=False``

            If ``True``, the triangle grids are embedded in 3D space.
            If ``False``, the grids are truly 2D.
        surface_tag: ``default=None``

            The target physical name.
            All surfaces that have this tag will be assigned a grid.

            The string is assumed to be on the from between the physical names and
            the line, up to the last underscore.

            If not provided, the physical names of fracture surfaces will be used as
            target.
        constraints: ``default=None``

            Array with lists of lines that should not become grids. The array
            items should match the ``INDEX`` in ``surface_tag``, see above.

    Returns:
        List of 2D grids for all physical surfaces that matched with the specified
        target tag.

    """

    if surface_tag is None:
        surface_tag = PhysicalNames.FRACTURE.value

    if constraints is None:
        constraints = np.array([], dtype=int)

    if is_embedded:
        # List of 2D grids, one for each surface
        g_list: list[pp.Grid] = []

        # Special treatment of the case with no fractures
        if "triangle" not in cells:
            return g_list
        # Recover cells on fracture surfaces, and create grids
        tri_cells = cells["triangle"]

        # Tags of all triangle grids
        tri_tags = cell_info["triangle"]

        # Loop over all gmsh tags associated with triangle grids
        for pn_ind in np.unique(tri_tags):
            # Split the physical name into a category and a number - which will
            # become the fracture number
            pn = phys_names[pn_ind]
            offset = pn.rfind("_")
            frac_num = int(pn[offset + 1 :])
            plane_type = pn[:offset]

            # Check if the surface is of the target type, or if the surface is tagged
            # as a constraint
            if plane_type != surface_tag[:-1] or int(pn[offset + 1 :]) in constraints:
                continue

            # Cells of this surface
            loc_cells = np.where(tri_tags == pn_ind)[0]
            loc_tri_cells = tri_cells[loc_cells, :].astype(int)

            # Find unique points, and a mapping from local to global points
            pind_loc, p_map = np.unique(loc_tri_cells, return_inverse=True)
            loc_tri_ind = p_map.reshape((-1, 3))
            g = pp.TriangleGrid(pts[pind_loc, :].transpose(), loc_tri_ind.transpose())
            # Add mapping to global point numbers
            g.global_point_ind = pind_loc

            # Associate a fracture id (corresponding to the ordering of the fracture
            # planes in the original fracture list provided by the user)
            g.frac_num = frac_num

            # Append to list of 2d grids
            g_list.append(g)

        # Done with all surfaces, return
        return g_list

    else:
        # Single grid

        triangles = cells["triangle"].transpose()
        # Construct grid
        g_2d: pp.Grid = pp.TriangleGrid(pts.transpose(), triangles)

        # we need to add the face tags from gmsh to the current mesh, however,
        # since there is not a cell-face relation from gmsh but only a cell-node
        # relation we need to recover the corresponding face-line map. First find the
        # nodes of each face
        faces = np.reshape(g_2d.face_nodes.indices, (2, -1), order="F")
        faces = np.sort(faces, axis=0)

        # Then we do a bunch of sorting to make sure faces and lines has the same
        # node ordering:
        idxf = np.lexsort(faces)

        line = np.sort(cells["line"].T, axis=0)
        idxl = np.lexsort(line)
        IC = np.empty(line.shape[1], dtype=int)
        IC[idxl] = np.arange(line.shape[1])

        # Next change the faces and line to string format ("node_idx0,node_idx1").
        # The reason to do so is because we want to compare faces and line columnwise,
        # i.e., is_line[i] should be true iff faces[:, i] == line[:, j] for ONE j. If
        # you can make numpy do this, you can remove the string formating.
        tmp = np.char.add(faces[0, idxf].astype(str), ",")
        facestr = np.char.add(tmp, faces[1, idxf].astype(str))
        tmp = np.char.add(line[0, idxl].astype(str), ",")
        linestr = np.char.add(tmp, line[1, idxl].astype(str))

        is_line = np.isin(facestr, linestr, assume_unique=True)

        # Now find the face index that correspond to each line. line2face is of length
        # line.shape[1] and we have: face[:, line2face] == line.
        line2face = idxf[is_line][IC]
        # Sanity check
        if not np.allclose(faces[:, line2face], line):
            raise RuntimeError(
                "Could not find mapping from gmsh lines to pp.Grid faces"
            )

        # Now we can assign the correct tags to the grid. First we add them as False
        # and after we change for the correct faces. The new tag name become the
        # lower version of what gmsh gives in the cell_info["line"]. The map
        # phys_names recover the literal name.
        for tag in np.unique(cell_info["line"]):
            tag_name = phys_names[tag].lower() + "_faces"
            g_2d.tags[tag_name] = np.zeros(g_2d.num_faces, dtype=bool)
            # Add correct tag
            faces = line2face[cell_info["line"] == tag]
            g_2d.tags[tag_name][faces] = True

        # Create mapping to global numbering (will be a unit mapping, but is crucial
        # for consistency with lower dimensions)
        g_2d.global_point_ind = np.arange(pts.shape[0])

        # Convert to list to be consistent with lower dimensions. This may also become
        # useful in the future if we ever implement domain decomposition approaches
        # based on gmsh.
        return [g_2d]


def create_1d_grids(
    pts: np.ndarray,
    cells: dict[str, np.ndarray],
    phys_names: dict[int, str],
    cell_info: dict[str, np.ndarray],
    line_tag: Optional[str] = None,
    tol: float = 1e-4,
    constraints: Optional[np.ndarray] = None,
    return_fracture_tips: bool = True,
):
    """Create 1D grids for lines of a specified type from a gmsh tesselation.

    Only lines that were defined as 'physical' in the gmsh sense may have a grid
    created, but then only if the physical name matches specified ``line_tag``.

    It is assumed that the mesh is read by meshio. See :mod:`~porepy.fracs.simplex` for
    how to do this.

    Parameters:
        pts: ``shape=(num_points, 3)``

            Global point set from gmsh.
        cells: Should have a key ``'line'``, which maps to a :obj:`~numpy.ndarray` with
            indices of the lines that form 1D grids.
        phys_names: Mapping from the gmsh tags assigned to physical entities to the
            physical name of that tag.
        cell_info: Should have a key ``'line'``, that contains the physical names (in
            the gmsh sense) of the points.
        line_tag: ``default=None``

            The target physical name. All lines that have this tag will be assigned a
            grid.

            The string is assumed to be on the from ``BASE_NAME_OF_TAG_{INDEX}``, where
            ``INDEX`` is a number. The comparison is made between the physical names and
            the line, up to the last underscore.

            If not provided, the physical names of fracture lines will be used as
            target.
        tol: ``default=1e-4``

            Tolerance used when comparing points in the creation of line grids.
        constraints: ``default=None``

            Array with lists of lines that should not become grids. The array items
            should match the ``INDEX`` in ``line_tag``, see above.
        return_fracture_tips: ``default=True``

            If ``True``, fracture tips will be found and returned.

    Returns:
        List of 1D grids for all physical lines that matched with the specified target
        tag.

        If ``return_fracture_tips=True``, returns additionally a :obj:`~numpy.ndarray`
        of indices of points on a fracture tip.

    """
    if line_tag is None:
        line_tag = PhysicalNames.FRACTURE_INTERSECTION_LINE.value

    if constraints is None:
        constraints = np.empty(0, dtype=int)
    # Recover lines.
    # There will be up to three types of physical lines: intersections (between
    # fractures), fracture tips, and auxiliary lines (to be disregarded).

    # Data structure for the point grids.
    g_1d: list[pp.Grid] = []

    # If there are no fracture intersections, we return empty lists.
    if "line" not in cells:
        return g_1d, np.empty(0)

    line_tags = cell_info["line"]
    line_cells = cells["line"]

    gmsh_tip_num = []
    tip_pts = np.empty(0)

    for i, pn_ind in enumerate(np.unique(line_tags)):
        # Index of the final underscore in the physical name. Chars before this will
        # identify the line type, the one after will give index.
        pn = phys_names[pn_ind]
        offset_index = pn.rfind("_")
        loc_line_cell_num = np.where(line_tags == pn_ind)[0]
        loc_line_pts = line_cells[loc_line_cell_num, :]

        assert loc_line_pts.size > 1

        line_type = pn[:offset_index]

        # Try to get the fracture number from the physical name of the object.

        # This will only work if everything after the '_' can be interpreted as a
        # number. Specifically, it should work for all meshes generated by the
        # standard PorePy procedure, but it may fail for externally generated
        # geo-files. If it fails, we simply set the frac_num to -1 in this case,
        # that is, assign the default value as defined in pp.Grid.__init__
        try:
            frac_num = int(pn[offset_index + 1 :])
        except ValueError:
            frac_num = -1

        # If this is a meshing constraint, but not a fracture, we don't need to do
        # anything.
        if frac_num in constraints:
            continue

        if line_type == PhysicalNames.FRACTURE_TIP.value[:-1]:
            gmsh_tip_num.append(i)

            # We need not know which fracture the line is on the tip of (do we?)
            tip_pts = np.append(tip_pts, np.unique(loc_line_pts))

        elif line_type == line_tag[:-1]:
            loc_pts_1d = np.unique(loc_line_pts)
            loc_coord = pts[loc_pts_1d, :].transpose()
            g = create_embedded_line_grid(loc_coord, loc_pts_1d, tol=tol)
            g.frac_num = int(frac_num)
            g_1d.append(g)

        else:  # Auxiliary line
            pass

    if return_fracture_tips:
        return g_1d, tip_pts
    else:
        return g_1d


def create_0d_grids(
    pts: np.ndarray,
    cells: dict[str, np.ndarray],
    phys_names: dict[int, str],
    cell_info: dict[str, np.ndarray],
    target_tag_stem: Optional[str] = None,
) -> list[pp.PointGrid]:
    """Create 0D grids for points of a specified type from a gmsh tesselation.

    Only points that were defined as 'physical' in the gmsh sense may have a grid
    created, but then only if the physical name matches specified ``target_tag_stem``.

    It is assumed that the mesh is read by meshio. See module
    :mod:`~porepy.fracs.simplex` for how to do this.

    Parameters:
        pts: ``shape=(num_points, 3)``

            Global point set from gmsh.
        cells: Should have a key ``'vertex'``, which maps to a :obj:`~numpy.ndarray`
            with indices of the points that form point grids.
        phys_names: mapping from the gmsh tags assigned to physical entities
            to the physical name of that tag.
        cell_info: Should have a key ``'vertex'``, that contains the
            physical names (in the gmsh sense) of the points.
        target_tag_stem: The target physical name.

            All points that have this tag will be assigned a grid. The string is assumed
            to be on from ``BASE_NAME_OF_TAG_{INDEX}``, where ``INDEX`` is a number.

            The comparison is made between the physical names and the
            ``target_tag_stem``, up to the last underscore.

            If not provided, the physical names of fracture points will be used as
            target.

    Returns:
        List of point grids for all physical points that matched with the specified
        target tag.

    """
    if target_tag_stem is None:
        target_tag_stem = PhysicalNames.FRACTURE_INTERSECTION_POINT.value

    g_0d = []

    if "vertex" in cells:
        # Index (in the array pts) of the points that are specified as physical in the
        # .geo-file.
        point_cells = cells["vertex"].ravel()

        # Keys to the physical names table of the points that have been declared as
        # physical.
        physical_name_indices = cell_info["vertex"]

        # Loop over all physical points.
        for pi, phys_names_ind in enumerate(physical_name_indices):
            pn = phys_names[phys_names_ind]
            offset_index = pn.rfind("_")

            phys_name_vertex = pn[:offset_index]

            # Check if this is the target. The -1 is needed to avoid the extra _ in the
            # defined constant.
            if phys_name_vertex == target_tag_stem[:-1]:
                # This should be a new grid.
                g = pp.PointGrid(pts[point_cells[pi]])
                g.global_point_ind = np.atleast_1d(np.asarray(point_cells[pi]))

                # Store the index of this physical name tag.
                g._physical_name_index = int(pn[offset_index + 1 :])

                g_0d.append(g)
            else:
                continue
    return g_0d


def create_embedded_line_grid(
    loc_coord: np.ndarray, glob_id: np.ndarray, tol: float = 1e-4
) -> pp.Grid:
    """Create a 1d grid embedded in a higher dimensional space.

    Parameters:
        loc_coord: Coordinates of points to be used in the grid.
        glob_id : Global indexes of the points. Typically refers to a global
            mesh, where the points of this grid is a subset.
        tol: Tolerance used for check of collinearity of the points.

    Returns:
        g: 1d grid.

    """
    loc_center = np.mean(loc_coord, axis=1).reshape((-1, 1))
    (
        sorted_coord,
        rot,
        active_dimension,
        sort_ind,
    ) = pp.map_geometry.project_points_to_line(loc_coord, tol)
    g = pp.TensorGrid(sorted_coord)

    # Project back to active dimension.
    nodes = np.zeros(g.nodes.shape)
    nodes[active_dimension] = g.nodes[0]
    g.nodes = nodes

    # Project back again to 3d coordinates.
    irot = rot.transpose()
    g.nodes = irot.dot(g.nodes)
    g.nodes += loc_center

    # Add mapping to global point numbers.
    g.global_point_ind = glob_id[sort_ind]
    return g
