"""
Module for converting gmsh output file to our grid structure.
Maybe we will add the reverse mapping. 
"""
import numpy as np

from core.grids import simplex, structured, point_grid
from gridding import constants
import compgeom.basics as cg


def create_3d_grids(pts, cells):
    tet_cells = cells['tetra']
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
    is_embedded = kwargs.get('is_embedded', False)
    if is_embedded:
        phys_names = kwargs.get('phys_names', False)
        cell_info = kwargs.get('cell_info', False)
        network = kwargs.get('network', False)

        # Check input
        if not phys_names:
            raise TypeError('Need to specify phys_names for embedded grids')
        if not cell_info:
            raise TypeError('Need to specify cell_info for embedded grids')
        if not network:
            raise TypeError('Need to specify network for embedded grids')

        # Special treatment of the case with no fractures
        if not 'triangle' in cells:
            return g_2d

        # Recover cells on fracture surfaces, and create grids
        tri_cells = cells['triangle']

        # Map from split polygons and fractures, as defined by the network
        # decomposition
        poly_2_frac = network.decomposition['polygon_frac']

        num_tri = len(phys_names['triangle'])
        gmsh_num = np.zeros(num_tri)
        frac_num = np.zeros(num_tri)

        for i, pn in enumerate(phys_names['triangle']):
            offset = pn[2].rfind('_')
            frac_num[i] = poly_2_frac[int(pn[2][offset + 1:])]
            gmsh_num[i] = pn[1]

        for fi in np.unique(frac_num):
            loc_num = np.where(frac_num == fi)[0]
            loc_gmsh_num = gmsh_num[loc_num]

            loc_tri_glob_ind = np.empty((0, 3))
            for ti in loc_gmsh_num:
                # It seems the gmsh numbering corresponding to the physical tags
                # (as found in physnames) is stored in the first column of info
                gmsh_ind = np.where(cell_info['triangle'][:, 0] == ti)[0]
                loc_tri_glob_ind = np.vstack((loc_tri_glob_ind,
                                              tri_cells[gmsh_ind, :]))

            loc_tri_glob_ind = loc_tri_glob_ind.astype('int')
            pind_loc, p_map = np.unique(loc_tri_glob_ind, return_inverse=True)
            loc_tri_ind = p_map.reshape((-1, 3))
            g = simplex.TriangleGrid(pts[pind_loc, :].transpose(),
                                     loc_tri_ind.transpose())
            # Add mapping to global point numbers
            g.global_point_ind = pind_loc

            # Append to list of 2d grids
            g_2d.append(g)

    else:
        triangles = cells['triangle'].transpose()
        # Construct grid
        g_2d = simplex.TriangleGrid(pts.transpose(), triangles)

        # Create mapping to global numbering (will be a unit mapping, but is
        # crucial for consistency with lower dimensions)
        g_2d.global_point_ind = np.arange(pts.shape[0])

        # Convert to list to be consistent with lower dimensions
        # This may also become useful in the future if we ever implement domain
        # decomposition approaches based on gmsh.
        g_2d = [g_2d]
    return g_2d


def create_1d_grids(pts, cells, phys_names, cell_info,
                    line_tag=constants.GmshConstants().PHYSICAL_NAME_FRACTURE_LINE):
    # Recover lines
    # There will be up to three types of physical lines: intersections (between
    # fractures), fracture tips, and auxiliary lines (to be disregarded)

    g_1d = []

    # If there are no fracture intersections, we return empty lists
    if not 'line' in cells:
        return g_1d, np.empty(0)

    gmsh_const = constants.GmshConstants()

    line_tags = cell_info['line'][:, 0]
    line_cells = cells['line']

    gmsh_tip_num = []
    tip_pts = np.empty(0)

    for i, pn in enumerate(phys_names['line']):
        # Index of the final underscore in the physical name. Chars before this
        # will identify the line type, the one after will give index
        offset_index = pn[2].rfind('_')
        loc_line_cell_num = np.where(line_tags == pn[1])[0]
        loc_line_pts = line_cells[loc_line_cell_num, :]

        assert loc_line_pts.size > 1

        line_type = pn[2][:offset_index]

        if line_type == gmsh_const.PHYSICAL_NAME_FRACTURE_TIP[:-1]:
            gmsh_tip_num.append(i)

            # We need not know which fracture the line is on the tip of (do
            # we?)
            tip_pts = np.append(tip_pts, np.unique(loc_line_pts))

        elif line_type == line_tag[:-1]:
            loc_pts_1d = np.unique(loc_line_pts)  # .flatten()
            loc_coord = pts[loc_pts_1d, :].transpose()
            loc_center = np.mean(loc_coord, axis=1).reshape((-1, 1))
            loc_coord -= loc_center
            # Check that the points indeed form a line
            assert cg.is_collinear(loc_coord)
            # Find the tangent of the line
            tangent = cg.compute_tangent(loc_coord)
            # Projection matrix
            rot = cg.project_plane_matrix(loc_coord, tangent)
            loc_coord_1d = rot.dot(loc_coord)
            # The points are now 1d along one of the coordinate axis, but we
            # don't know which yet. Find this.

            sum_coord = np.sum(np.abs(loc_coord_1d), axis=1)
            active_dimension = np.logical_not(np.isclose(sum_coord, 0))
            # Check that we are indeed in 1d
            assert np.sum(active_dimension) == 1
            # Sort nodes, and create grid
            coord_1d = loc_coord_1d[active_dimension]
            sort_ind = np.argsort(coord_1d)[0]
            sorted_coord = coord_1d[0, sort_ind]
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
            g.global_point_ind = loc_pts_1d[sort_ind]

            g_1d.append(g)

        else:  # Auxiliary line
            pass
    return g_1d, tip_pts


def create_0d_grids(pts, cells):
    # Find 0-d grids (points)
    # We know the points are 1d, so squeeze the superflous dimension
    g_0d = []
    if 'vertex' in cells:
        point_cells = cells['vertex'].ravel()
        for pi in point_cells:
            g = point_grid.PointGrid(pts[pi])
            g.global_point_ind = np.asarray(pi)
            g_0d.append(g)
    return g_0d
