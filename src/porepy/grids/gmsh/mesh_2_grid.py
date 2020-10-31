"""
Module for converting gmsh output file to our grid structure.
"""
from typing import Dict, List, Tuple, Union

import numpy as np

import porepy as pp
from porepy.grids import constants


def create_3d_grids(pts: np.ndarray, cells: Dict[str, np.ndarray]) -> List[pp.Grid]:
    """Create a tetrahedral grid from a gmsh tessalation.

    Parameters:
        pts (np.ndarray, npt x 3): Global point set from gmsh
        cells (dict): Should have a key 'tetra' which maps to a np.ndarray with indices
            of the points that form 3d grids.

    Returns:
        list of grids: List with a 3d grid.
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
    cells: Dict[str, np.ndarray],
    phys_names: Dict[str, str],
    cell_info: Dict,
    is_embedded: bool = False,
    surface_tag: str = None,
    constraints: np.ndarray = None,
) -> List[pp.Grid]:
    """Create 2d grids for lines of a specified type from a gmsh tessalation.

    Only surfaces that were defined as 'physical' in the gmsh sense may have a grid
    created, but then only if the physical name matches specified line_tag.

    It is assumed that the mesh is read by meshio. See porepy.fracs.simplex for how to
    do this.

    Parameters:
        pts (np.ndarray, npt x 3): Global point set from gmsh
        cells (dict): Should have a key 'triangle' which maps to a np.ndarray with
            indices of the points that form 2d grids.
        phys_names (dict): mapping from the gmsh tags assigned to physical entities
            to the physical name of that tag.
        cell_info (dictionary): Should have a key 'triangle' that contains the
            physical names (in the gmsh sense) of the points.
        is_embedded (boolean, optional): If True, the triangle grids are embedded in
            3d space. If False (default), the grids are truly 2d.
        surface_tag (str, optional): The target physical name, all surfaces that have
            this tag will be assigned a grid. The string is assumed to be on the from
            BASE_NAME_OF_TAG_{INDEX}, where _INDEX is a number. The comparison is made
            between the physical names and the line, up to the last
            underscore. If not provided, the physical names of fracture surfaces will be
            used as target.
        constraints (np.array, optional): Array with lists of lines that should not
            become grids. The array items should match the INDEX in line_tag, see above.

    Returns:
        list of grids: List of 2d grids for all physical surfaces that matched with the
            specified target tag.

    """

    gmsh_constants = constants.GmshConstants()
    if surface_tag is None:
        surface_tag = gmsh_constants.PHYSICAL_NAME_FRACTURES

    if constraints is None:
        constraints = np.array([], dtype=np.int)

    if is_embedded:
        # List of 2D grids, one for each surface
        g_list: List[pp.Grid] = []

        # Special treatment of the case with no fractures
        if "triangle" not in cells:
            return g_list
        # Recover cells on fracture surfaces, and create grids
        tri_cells = cells["triangle"]

        # Tags of all triangle grids
        tri_tags = cell_info["triangle"]

        # Loop over all gmsh tags associated with triangle grids
        for pn_ind in np.unique(tri_tags):

            # Split the physical name into a category and a number - which will become
            # the fracture number
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
            loc_tri_cells = tri_cells[loc_cells, :].astype(np.int)

            # Find unique points, and a mapping from local to global points
            pind_loc, p_map = np.unique(loc_tri_cells, return_inverse=True)
            loc_tri_ind = p_map.reshape((-1, 3))
            g = pp.TriangleGrid(pts[pind_loc, :].transpose(), loc_tri_ind.transpose())
            # Add mapping to global point numbers
            g.global_point_ind = pind_loc

            # Associate a fracture id (corresponding to the ordering of the
            # frature planes in the original fracture list provided by the
            # user)
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
        # relation we need to recover the corresponding face-line map.
        # First find the nodes of each face
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
        tmp = np.core.defchararray.add(faces[0, idxf].astype(str), ",")
        facestr = np.core.defchararray.add(tmp, faces[1, idxf].astype(str))
        tmp = np.core.defchararray.add(line[0, idxl].astype(str), ",")
        linestr = np.core.defchararray.add(tmp, line[1, idxl].astype(str))

        is_line = np.isin(facestr, linestr, assume_unique=True)

        # Now find the face index that correspond to each line. line2face is of length
        # line.shape[1] and we have: face[:, line2face] == line.
        line2face = idxf[is_line][IC]
        # Sanity check
        if not np.allclose(faces[:, line2face], line):
            raise RuntimeError(
                "Could not find mapping from gmsh lines to pp.Grid faces"
            )

        # Now we can assign the correct tags to the grid.
        # First we add them as False and after we change for the correct
        # faces. The new tag name become the lower version of what gmsh gives
        # in the cell_info["line"]. The map phys_names recover the literal name.
        for tag in np.unique(cell_info["line"]):
            tag_name = phys_names[tag].lower() + "_faces"
            g_2d.tags[tag_name] = np.zeros(g_2d.num_faces, dtype=np.bool)
            # Add correct tag
            faces = line2face[cell_info["line"] == tag]
            g_2d.tags[tag_name][faces] = True

        # Create mapping to global numbering (will be a unit mapping, but is
        # crucial for consistency with lower dimensions)
        g_2d.global_point_ind = np.arange(pts.shape[0])

        # Convert to list to be consistent with lower dimensions
        # This may also become useful in the future if we ever implement domain
        # decomposition approaches based on gmsh.
        return [g_2d]


def create_1d_grids(
    pts: np.ndarray,
    cells: Dict[str, np.ndarray],
    phys_names: Dict,
    cell_info: Dict,
    line_tag: str = None,
    tol: float = 1e-4,
    constraints: np.ndarray = None,
    return_fracture_tips: bool = True,
) -> Union[List[pp.Grid], Tuple[List[pp.Grid], np.ndarray]]:
    """Create 1d grids for lines of a specified type from a gmsh tessalation.

    Only lines that were defined as 'physical' in the gmsh sense may have a grid
    created, but then only if the physical name matches specified line_tag.

    It is assumed that the mesh is read by meshio. See porepy.fracs.simplex for how to
    do this.

    Parameters:
        pts (np.ndarray, npt x 3): Global point set from gmsh
        cells (dict): Should have a key 'line', which maps to a np.ndarray with indices
            of the lines that form 1d grids.
        phys_names (dict): mapping from the gmsh tags assigned to physical entities
            to the physical name of that tag.
        cell_info (dictionary): Should have a key 'line', that contains the
            physical names (in the gmsh sense) of the points.
        line_tag (str, optional): The target physical name, all lines that have
            this tag will be assigned a grid. The string is assumed to be on the from
            BASE_NAME_OF_TAG_{INDEX}, where _INDEX is a number. The comparison is made
            between the physical names and the line, up to the last
            underscore. If not provided, the physical names of fracture lines will be
            used as target.
        tol (double, optional): Tolerance used when comparing points in the creation of
            line grids. Defaults to 1e-4.
        constraints (np.array, optional): Array with lists of lines that should not
            become grids. The array items should match the INDEX in line_tag, see above.
        return_fracture_tips (boolean, optional): If True (default), fracture tips will
            be found and returned.

    Returns:
        list of grids: List of 1d grids for all physical lines that matched with the
            specified target tag.
        np.array, each item is an array of indices of points on a fracture tip. Only
            returned in return_fracture_tips is True.

    """
    gmsh_constants = constants.GmshConstants()
    if line_tag is None:
        line_tag = gmsh_constants.PHYSICAL_NAME_FRACTURE_LINE

    if constraints is None:
        constraints = np.empty(0, dtype=np.int)
    # Recover lines
    # There will be up to three types of physical lines: intersections (between
    # fractures), fracture tips, and auxiliary lines (to be disregarded)

    # Data structure for the point grids
    g_1d: List[pp.Grid] = []

    # If there are no fracture intersections, we return empty lists
    if "line" not in cells:
        return g_1d, np.empty(0)

    line_tags = cell_info["line"]
    line_cells = cells["line"]

    gmsh_tip_num = []
    tip_pts = np.empty(0)

    for i, pn_ind in enumerate(np.unique(line_tags)):
        # Index of the final underscore in the physical name. Chars before this
        # will identify the line type, the one after will give index
        pn = phys_names[pn_ind]
        offset_index = pn.rfind("_")
        loc_line_cell_num = np.where(line_tags == pn_ind)[0]
        loc_line_pts = line_cells[loc_line_cell_num, :]

        assert loc_line_pts.size > 1

        line_type = pn[:offset_index]

        # Try to get the fracture number from the physical name of the object
        # This will only work if everything after the '_' can be interpreted
        # as a number. Specifically, it should work for all meshes generated by
        # the standard PorePy procedure, but it may fail for externally generated
        # geo-files. If it fails, we simply set the frac_num to -1 in this case,
        # that is, assign the default value as defined in pp.Grid.__init__
        try:
            frac_num = int(pn[offset_index + 1 :])
        except ValueError:
            #
            frac_num = -1

        # If this is a meshing constraint, but not a fracture, we don't need to do anything
        if frac_num in constraints:
            continue

        if line_type == gmsh_constants.PHYSICAL_NAME_FRACTURE_TIP[:-1]:
            gmsh_tip_num.append(i)

            # We need not know which fracture the line is on the tip of (do
            # we?)
            tip_pts = np.append(tip_pts, np.unique(loc_line_pts))

        elif line_type == line_tag[:-1]:
            loc_pts_1d = np.unique(loc_line_pts)  # .flatten()
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
    cells: Dict[str, np.ndarray],
    phys_names: Dict[int, str],
    cell_info: Dict[str, np.ndarray],
    target_tag_stem: str = None,
) -> List[pp.PointGrid]:
    """Create 0d grids for points of a specified type from a gmsh tessalation.

    Only points that were defined as 'physical' in the gmsh sense may have a grid
    created, but then only if the physical name matches specified target_tag_stem.

    It is assumed that the mesh is read by meshio. See porepy.fracs.simplex for how to
    do this.

    Parameters:
        pts (np.ndarray, npt x 3): Global point set from gmsh
        cells (dict): Should have a key vertex, which maps to a np.ndarray if indices
            of the points that form point grids.
        phys_names (dict): mapping from the gmsh tags assigned to physical entities
            to the physical name of that tag.
        cell_info (dictionary): Should have a key 'vertex', that contains the
            physical names (in the gmsh sense) of the points.
        target_tag_stem (str, optional): The target physical name, all points that have
            this tag will be assigned a grid. The string is assumed to be on the from
            BASE_NAME_OF_TAG_{INDEX}, where _INDEX is a number. The comparison is made
            between the physical names and the target_tag_stem, up to the last
            underscore. If not provided, the physical names of fracture points will be
            used as target.

    Returns:
        list of grids: List of 0d grids for all physical points that matched with the
            specified target tag.

    """
    if target_tag_stem is None:
        target_tag_stem = constants.GmshConstants().PHYSICAL_NAME_FRACTURE_POINT

    g_0d = []

    if "vertex" in cells:
        # Index (in the array pts) of the points that are specified as physical in the
        # .geo-file
        point_cells = cells["vertex"].ravel()

        # Keys to the physical names table of the points that have been decleared as
        # physical
        physical_name_indices = cell_info["vertex"]

        # Loop over all physical points
        for pi, phys_names_ind in enumerate(physical_name_indices):
            pn = phys_names[phys_names_ind]
            offset_index = pn.rfind("_")

            phys_name_vertex = pn[:offset_index]

            # Check if this is the target. The -1 is needed to avoid the extra _ in
            # the defined constantnt
            if phys_name_vertex == target_tag_stem[:-1]:
                # This should be a new grid
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
    """
    Create a 1d grid embedded in a higher dimensional space.

    Args:
        loc_coord (np.ndarray): Coordinates of points to be used in the grid.
        glob_id (np.ndarray): Global indexes of the points. Typically refers to a global
            mesh, where the points of this grid is a subset.
        tol (float, optional): Tolerance used for check of collinearity of the points.
            Defaults to 1e-4.

    Returns:
        g (TYPE): DESCRIPTION.

    """
    loc_center = np.mean(loc_coord, axis=1).reshape((-1, 1))
    (
        sorted_coord,
        rot,
        active_dimension,
        sort_ind,
    ) = pp.map_geometry.project_points_to_line(loc_coord, tol)
    g = pp.TensorGrid(sorted_coord)

    # Project back to active dimension
    nodes = np.zeros(g.nodes.shape)
    nodes[active_dimension] = g.nodes[0]
    g.nodes = nodes

    # Project back again to 3d coordinates
    irot = rot.transpose()
    g.nodes = irot.dot(g.nodes)
    g.nodes += loc_center

    # Add mapping to global point numbers
    g.global_point_ind = glob_id[sort_ind]
    return g
