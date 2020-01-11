"""
Module for converting gmsh output file to our grid structure.
Maybe we will add the reverse mapping.
"""
import numpy as np

import porepy as pp
from porepy.grids import simplex, structured, point_grid
from porepy.grids import constants


def create_3d_grids(pts, cells):
    tet_cells = cells["tetra"]
    g_3d = simplex.TetrahedralGrid(pts.transpose(), tet_cells.transpose())

    # Create mapping to global numbering (will be a unit mapping, but is
    # crucial for consistency with lower dimensions)
    g_3d.global_point_ind = np.arange(pts.shape[0])

    # Convert to list to be consistent with lower dimensions
    # This may also become useful in the future if we ever implement domain
    # decomposition approaches based on gmsh.
    g_3d = [g_3d]
    return g_3d


def create_2d_grids(pts, cells, **kwargs):
    # List of 2D grids, one for each surface
    g_2d = []
    is_embedded = kwargs.get("is_embedded", False)
    phys_names = kwargs.get("phys_names", False)
    cell_info = kwargs.get("cell_info", False)
    if is_embedded:
        network = kwargs.get("network", False)

        # Check input
        if not phys_names:
            raise TypeError("Need to specify phys_names for embedded grids")
        if not cell_info:
            raise TypeError("Need to specify cell_info for embedded grids")
        if not network:
            raise TypeError("Need to specify network for embedded grids")

        # Special treatment of the case with no fractures
        if not "triangle" in cells:
            return g_2d

        # Recover cells on fracture surfaces, and create grids
        tri_cells = cells["triangle"]

        # Map from split polygons and fractures, as defined by the network
        # decomposition
        poly_2_frac = network.decomposition["polygon_frac"]

        phys_name_ind_tri = np.unique(cell_info["triangle"]["gmsh:physical"])

        # Index of the physical name tag assigned by gmsh to each fracture
        gmsh_num = np.zeros(phys_name_ind_tri.size, dtype="int")
        # Index of the corresponding name used in the input to gmsh (on the
        # from FRACTURE_{0, 1} etc.
        frac_num = np.zeros(phys_name_ind_tri.size, dtype="int")

        for i, pn_ind in enumerate(phys_name_ind_tri):
            pn = phys_names[pn_ind]
            offset = pn.rfind("_")
            frac_num[i] = poly_2_frac[int(pn[offset + 1 :])]
            gmsh_num[i] = pn_ind

        # Counter for boundary and auxiliary planes
        count_bound_and_aux = 0
        for fi in np.unique(frac_num):
            # This loop should only produce grids on surfaces that are actually fractures
            # Fractures are identified with physical names
            # a) Either give seperate physical name to non-fractures (e.g. AUX_POLYGON)
            # b) OR: Pass a list of which fractures (numbers) are really not fractures
            # b) is simpler, a) is better (long term)
            # If a) is chosen, you may need to be careful with

            pn = phys_names[phys_name_ind_tri[fi]]
            plane_type = pn[: pn.rfind("_")]

            if plane_type != "FRACTURE":
                count_bound_and_aux += 1
                continue

            loc_num = np.where(frac_num == fi - count_bound_and_aux)[0]
            loc_gmsh_num = gmsh_num[loc_num]

            loc_tri_glob_ind = np.empty((0, 3))
            for ti in loc_gmsh_num:
                # It seems the gmsh numbering corresponding to the physical tags
                # (as found in physnames) is stored in the first column of info
                gmsh_ind = np.where(cell_info["triangle"]["gmsh:physical"] == ti)[0]
                loc_tri_glob_ind = np.vstack((loc_tri_glob_ind, tri_cells[gmsh_ind, :]))

            loc_tri_glob_ind = loc_tri_glob_ind.astype("int")
            pind_loc, p_map = np.unique(loc_tri_glob_ind, return_inverse=True)
            loc_tri_ind = p_map.reshape((-1, 3))
            g = simplex.TriangleGrid(
                pts[pind_loc, :].transpose(), loc_tri_ind.transpose()
            )
            # Add mapping to global point numbers
            g.global_point_ind = pind_loc

            # Associate a fracture id (corresponding to the ordering of the
            # frature planes in the original fracture list provided by the
            # user)
            g.frac_num = fi - count_bound_and_aux

            # Append to list of 2d grids
            g_2d.append(g)

    else:

        triangles = cells["triangle"].transpose()
        # Construct grid
        g_2d = simplex.TriangleGrid(pts.transpose(), triangles)

        # we need to add the face tags from gmsh to the current mesh,
        # first we add them as False and after we change for the correct
        # faces. The new tag name become the lower version of what gmsh gives
        # in the cell_info["line"]. The map phys_names recover the literal name.

        # create all the extra tags for the grids, by default they're false
        for tag in np.unique(cell_info["line"]["gmsh:physical"]):
            tag_name = phys_names[tag].lower() + "_faces"
            g_2d.tags[tag_name] = np.zeros(g_2d.num_faces, dtype=np.bool)

        # since there is not a cell-face relation from gmsh but only a cell-node
        # relation we need to recover the corresponding face.
        for tag_id, tag in enumerate(cell_info["line"]["gmsh:physical"]):
            tag_name = phys_names[tag].lower() + "_faces"
            # check where is the first node indipendent on the position
            # in the triangle, being first, second or third node
            first = (triangles == cells["line"][tag_id, 0]).any(axis=0)
            # check where is the second node, same approach as before
            second = (triangles == cells["line"][tag_id, 1]).any(axis=0)
            # select which are the cells that have this edge
            tria = np.logical_and(first, second).astype(np.int)
            # with the cell_faces map we get the faces associated to
            # the selected triangles
            face = np.abs(g_2d.cell_faces).dot(tria)
            # we have two case, the face is internal or is at the boundary
            # we consider them separately
            if np.any(face > 1):
                # select the face if it is internal
                g_2d.tags[tag_name][face > 1] = True
            else:
                # the face is on a boundary
                face = np.logical_and(face, g_2d.tags["domain_boundary_faces"])
                if np.sum(face) == 2:
                    # the triangle has two faces at the boundary, check if it
                    # is the first otherwise it is the other.
                    face_id = np.where(face)[0]

                    first_face = np.zeros(face.size, dtype=np.bool)
                    first_face[face_id[0]] = True

                    nodes = g_2d.face_nodes.dot(first_face)
                    # check if the nodes of the first face are the same
                    if np.all(nodes[cells["line"][tag_id, :]]):
                        face[face_id[1]] = False
                    else:
                        face[face_id[0]] = False

                g_2d.tags[tag_name][face] = True

        # Create mapping to global numbering (will be a unit mapping, but is
        # crucial for consistency with lower dimensions)
        g_2d.global_point_ind = np.arange(pts.shape[0])

        # Convert to list to be consistent with lower dimensions
        # This may also become useful in the future if we ever implement domain
        # decomposition approaches based on gmsh.
        g_2d = [g_2d]
    return g_2d


def create_1d_grids(
    pts,
    cells,
    phys_names,
    cell_info,
    line_tag=constants.GmshConstants().PHYSICAL_NAME_FRACTURE_LINE,
    tol=1e-4,
    constraints=None,
    **kwargs
):

    if constraints is None:
        constraints = np.empty(0, dtype=np.int)
    # Recover lines
    # There will be up to three types of physical lines: intersections (between
    # fractures), fracture tips, and auxiliary lines (to be disregarded)

    # All intersection lines and points on boundaries are non-physical in 3d.
    # I.e., they are assigned boundary conditions, but are not gridded.

    g_1d = []

    # If there are no fracture intersections, we return empty lists
    if not "line" in cells:
        return g_1d, np.empty(0)

    gmsh_const = constants.GmshConstants()

    line_tags = cell_info["line"]["gmsh:physical"]
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
        # geo-files. If it fails, we simply set the frac_num to None in this case.
        try:
            frac_num = int(pn[offset_index + 1 :])
        except ValueError:
            frac_num = None

        # If this is a meshing constraint, but not a fracture, we don't need to do anything
        if frac_num in constraints:
            continue

        if line_type == gmsh_const.PHYSICAL_NAME_FRACTURE_TIP[:-1]:
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
    return g_1d, tip_pts


def create_0d_grids(pts, cells):
    # Find 0-d grids (points)
    # We know the points are 1d, so squeeze the superflous dimension

    # All intersection lines and points on boundaries are non-physical in 3d.
    # I.e., they are assigned boundary conditions, but are not gridded.
    g_0d = []
    if "vertex" in cells:
        point_cells = cells["vertex"].ravel()
        for pi in point_cells:
            g = point_grid.PointGrid(pts[pi])
            g.global_point_ind = np.atleast_1d(np.asarray(pi))
            g_0d.append(g)
    return g_0d


def create_embedded_line_grid(loc_coord, glob_id, tol=1e-4):
    loc_center = np.mean(loc_coord, axis=1).reshape((-1, 1))
    sorted_coord, rot, active_dimension, sort_ind = pp.map_geometry.project_points_to_line(
        loc_coord, tol
    )
    g = structured.TensorGrid(sorted_coord)

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
