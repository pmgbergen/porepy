import numpy as np

import gridding.constants
from core.grids import simplex, structured, point_grid
from gridding.gmsh import gmsh_interface, mesh_io
from gridding.fractured import fractures
import compgeom.sort_points
import compgeom.basics as cg


def create_grid(fracs, box, **kwargs):

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
    network = fractures.FractureSet(frac_list)

    # Impose domain boundary. For the moment, the network should be immersed in
    # the domain, or else gmsh will complain.
    network.impose_external_boundary(box)

    # Find intersections and split them, preparing the way for dumping the
    # network to gmsh
    network.find_intersections()
    network.split_intersections()

    in_file = file_name + '.geo'
    out_file = file_name + '.msh'

    network.to_gmsh(in_file)
    gmsh_path = kwargs.get('gmsh_path')


    gmsh_status = gmsh_interface.run_gmsh(gmsh_path, in_file, out_file, dims=3)
    
    if verbose > 0:
        if gmsh_status == 0:
            print('Gmsh processed file successfully')
        else:
            print('Gmsh failed with status ' + str(gmsh_status))

    pts, cells, phys_names, cell_info = gmsh_interface.read_gmsh(out_file)

    tet_cells = cells['tetra']
    g_3d = simplex.TetrahedralGrid(pts.transpose(), tet_cells.transpose())

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
        frac_num[i] = poly_2_frac[int(pn[2][offset+1:])]
        gmsh_num[i] = pn[1]

    # List of 2D grids, one for each surface
    g_2d = []

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
        loc_tri_loc_ind = pind_loc[p_map].reshape((-1, 3))
        g = simplex.TriangleGrid(pts[pind_loc, :].transpose(),
                                         loc_tri_loc_ind.transpose())
        g_2d.append(g)


    # Recover lines
    # There will be up to three types of physical lines: intersections (between
    # fractures), fracture tips, and auxiliary lines (to be disregarded)

    gmsh_const = gridding.constants.GmshConstants()

    line_tags = cell_info['line'][:, 0]
    line_cells = cells['line']

    gmsh_intersection_num = []
    gmsh_tip_num = []
    tip_pts = np.empty(0)

    g_1d = []
    for i, pn in enumerate(phys_names['line']):

        # Index of the final underscore in the physical name. Chars before this
        # will identify the line type, the one after will give index
        offset_index = pn[2].rfind('_')
        line_num = int(pn[2][offset_index+1:])
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



            loc_pts_1d = np.unique(loc_line_pts)#.flatten()

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
            sorted_coord = np.sort(loc_coord[active_dimension])
            g = structured.TensorGrid(sorted_coord)

            # Project back again to 3d coordinates
            irot = rot.transpose()
            g.nodes = irot.dot(g.nodes) + loc_coord
            g_1d.append(g)

        else:  # Auxiliary line
            pass


    # Find 0-d grids (points)
    # We know the points are 1d, so squeeze the superflous dimension
    point_cells = cells['vertex'].ravel()
    g_0d = []
    for pi in point_cells:
        g = point_grid.PointGrid(pts[pi])
        g_0d.append(g)




    import pdb
    pdb.set_trace()

    return g_3d, g_2d, g_1d, g_0d
