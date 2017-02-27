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

    tets = cells['tetra']
    g_3d = simplex.TetrahedralGrid(pts, tets)

    

