import numpy as np
import scipy.sparse as sps
import time


from core.grids import simplex, structured, point_grid
from gridding import constants
from gridding.grid_bucket import GridBucket
from gridding.gmsh import gmsh_interface, mesh_io
from gridding.fractured import fractures
import compgeom.sort_points
import compgeom.basics as cg
from utils import setmembership


def create_grid(fracs, box, **kwargs):
    """
    Create grids for a domain with possibly intersecting fractures in 3d.

    Based on polygons describing the individual fractures, the method computes
    fracture intersections, creates a gmsh input file, runs gmsh and reads the
    result, and then constructs grids in 3d (the whole domain), 2d (one for
    each individual fracture), 1d (along fracture intersections), and 0d
    (meeting between intersections).

    TODO: The method finds the mapping between faces in one dimension and cells
        in a lower dimension, but the information is not used. Should be
        returned in a sensible format.

    Parameters:
        fracs (list of np.ndarray, each 3xn): Vertexes in the polygons for each
            fracture.
        box (dictionary). Domain specification. Should have keywords xmin,
            xmax, ymin, ymax, zmin, zmax.
        **kwargs: To be explored. Should contain the key 'gmsh_path'.

    Returns:
        list (length 4): For each dimension (3 -> 0), a list of all grids in
            that dimension.

    """

    # Verbosity level
    verbose = kwargs.get('verbose', 1)

    # File name for communication with gmsh
    file_name = kwargs.get('file_name', 'gmsh_frac_file')

    # Convert the fractures from numpy representation to our 3D fracture data
    # structure.
    frac_list = []
    for f in fracs:
        frac_list.append(fractures.Fracture(f))

    # Combine the fractures into a network
    network = fractures.FractureNetwork(frac_list)

    # Impose domain boundary. For the moment, the network should be immersed in
    # the domain, or else gmsh will complain.
    network.impose_external_boundary(box)

    # Find intersections and split them, preparing the way for dumping the
    # network to gmsh
    network.find_intersections()
    network.split_intersections()

    in_file = file_name + '.geo'
    out_file = file_name + '.msh'

    network.to_gmsh(in_file, **kwargs)
    gmsh_path = kwargs.get('gmsh_path')

    gmsh_verbose = kwargs.get('gmsh_verbose', verbose)
    gmsh_opts = {'-v': gmsh_verbose}
    gmsh_status = gmsh_interface.run_gmsh(gmsh_path, in_file, out_file, dims=3,
                                          **gmsh_opts)

    if verbose > 0:
        start_time = time.time()
        if gmsh_status == 0:
            print('Gmsh processed file successfully')
        else:
            print('Gmsh failed with status ' + str(gmsh_status))

    pts, cells, phys_names, cell_info = gmsh_interface.read_gmsh(out_file)

    # Call upon helper functions to create grids in various dimensions.
    # The constructors require somewhat different information, reflecting the
    # different nature of the grids.
    g_3d = _create_3d_grids(pts, cells)
    g_2d = _create_2d_grids(pts, cells, phys_names, cell_info, network)
    g_1d, tip_pts = _create_1d_grids(pts, cells, phys_names, cell_info)
    g_0d = _create_0d_grids(pts, cells)

    grids = [g_3d, g_2d, g_1d, g_0d]
    bucket = assemble_in_bucket(grids)

    if verbose > 0:
        print('\n')
        print('Grid creation completed. Elapsed time ' + str(time.time() -
                                                             start_time))
        print('\n')
        for g_set in grids:
            if len(g_set) > 0:
                s = 'Created ' + str(len(g_set)) + ' ' + str(g_set[0].dim) + \
                    '-d grids with '
                num = 0
                for g in g_set:
                    num += g.num_cells
                s += str(num) + ' cells'
                print(s)
        print('\n')

    # We should also return the result of interdim_mappings, and possibly
    # tip_pts?
    return bucket


def _create_3d_grids(pts, cells):
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


def _create_2d_grids(pts, cells, phys_names, cell_info, network):
    # List of 2D grids, one for each surface
    g_2d = []

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
    return g_2d


def _create_1d_grids(pts, cells, phys_names, cell_info):
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

    gmsh_intersection_num = []
    gmsh_tip_num = []
    tip_pts = np.empty(0)

    for i, pn in enumerate(phys_names['line']):

        # Index of the final underscore in the physical name. Chars before this
        # will identify the line type, the one after will give index
        offset_index = pn[2].rfind('_')
        line_num = int(pn[2][offset_index + 1:])
        loc_line_cell_num = np.where(line_tags == pn[1])[0]
        loc_line_pts = line_cells[loc_line_cell_num, :]

        assert loc_line_pts.size > 1

        line_type = pn[2][:offset_index]
        if line_type == gmsh_const.PHYSICAL_NAME_FRACTURE_TIP[:-1]:
            gmsh_tip_num.append(i)

            # We need not know which fracture the line is on the tip of (do
            # we?)
            tip_pts = np.append(tip_pts, np.unique(loc_line_pts))

        elif line_type == gmsh_const.PHYSICAL_NAME_FRACTURE_LINE[:-1]:
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
            coord_1d = loc_coord[active_dimension]
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


def _create_0d_grids(pts, cells):
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


def assemble_in_bucket(grids):
    bucket = GridBucket()
    [bucket.add_nodes(g_d) for g_d in grids]
    for dim in range(len(grids) - 1):
        for hg in grids[dim]:
            # Sort the face nodes for simple comparison. np.sort returns a copy
            # of the list,
            fn_loc = hg.face_nodes.indices.reshape((hg.dim, hg.num_faces),
                                                   order='F')
            # Convert to global numbering
            fn = hg.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            for lg in grids[dim + 1]:
                cell_2_face, cell = obtain_interdim_mappings(lg, fn)
                face_cells = sps.csc_matrix(
                    (np.array([True] * cell.size), (cell, cell_2_face)),
                    (lg.num_cells,hg.num_faces))
                bucket.add_edge([hg, lg], face_cells)
    return bucket


def obtain_interdim_mappings(lg, fn):
    # Next, find mappings between faces in one dimension and cells in the lower
    # dimension
    if lg.dim > 0:
        cn_loc = lg.cell_nodes().indices.reshape((lg.dim + 1,
                                                  lg.num_cells),
                                                 order='F')
        cn = lg.global_point_ind[cn_loc]
        cn = np.sort(cn, axis=0)
    else:
        cn = np.array([lg.global_point_ind])
        # We also know that the higher-dimensional grid has faces
        # of a single node. This sometimes fails, so enforce it.
        fn = fn.ravel()

    is_mem, cell_2_face = setmembership.ismember_rows(cn, fn)

    # An element in cell_2_face gives, for all cells in the
    # lower-dimensional grid, the index of the corresponding face
    # in the higher-dimensional structure.

    low_dim_cell = np.where(is_mem)[0]
    return cell_2_face, low_dim_cell
