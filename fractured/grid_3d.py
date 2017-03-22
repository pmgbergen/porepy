import numpy as np
import scipy.sparse as sps
import time


from core.grids import simplex, structured, point_grid
from gridding import constants
from gridding.grid_bucket import GridBucket
from gridding.gmsh import gmsh_interface, mesh_io, mesh_2_grid
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
    g_3d = mesh_2_grid.create_3d_grids(pts, cells)
    g_2d = mesh_2_grid.create_2d_grids(
        pts, cells, is_embedded=True, phys_names=phys_names,
        cell_info=cell_info, network=network)
    g_1d, _ = mesh_2_grid.create_1d_grids(pts, cells, phys_names, cell_info)
    g_0d = mesh_2_grid.create_0d_grids(pts, cells)

    grids = [g_3d, g_2d, g_1d, g_0d]

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
    return grids
