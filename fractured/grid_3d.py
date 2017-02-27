import numpy as np

import gridding.constants
from core.grids import simplex
from gridding.gmsh import gmsh_interface, mesh_io
from gridding.fractures import fractures

def create_grid(fracs, box, **kwargs):

    # Verbosity level
    verbose = kwargs.get('verbose', default=1)

    # File name for communication with gmsh
    file_name = kwargs.get('file_name', default='gmsh_frac_file')

    # Convert the fractures from numpy representation to our 3D fracture data
    # structure.
    frac_list = []
    for f in fracs:
        frac_list.add(fractures.Fracture(f))

    # Combine the fractures into a network
    network = fractures.FractureSet(frac_list)

    # Impose domain boundary. For the moment, the network should be immersed in
    # the domain, or else gmsh will complain.
    network.impose_external_boundary(box)

    # Find intersections and split them, preparing the way for dumping the
    # network to gmsh
    network.find_intersections()
    network.split_intersections()

    network.to_gmsh()
    gmsh_path = gmsh_opts['path']

    in_file = file_name + '.geo'
    out_file = file_name + '.msh'

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

