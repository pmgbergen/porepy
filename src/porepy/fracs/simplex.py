import numpy as np
import time
from meshio import msh_io
import warnings

from porepy.grids import constants
from porepy.grids.gmsh import gmsh_interface, mesh_2_grid
from porepy.fracs import fractures, utils
import porepy.utils.comp_geom as cg



def tetrahedral_grid(fracs=None, box=None, network=None, **kwargs):
    """
    Create grids for a domain with possibly intersecting fractures in 3d.

    Based on the specified fractures, the method computes fracture
    intersections if necessary, creates a gmsh input file, runs gmsh and reads
    the result, and then constructs grids in 3d (the whole domain), 2d (one for
    each individual fracture), 1d (along fracture intersections), and 0d
    (meeting between intersections).

    The fractures can be specified is terms of the keyword 'fracs' (either as
    numpy arrays or Fractures, see below), or as a ready-made FractureNetwork
    by the keyword 'network'. For fracs, the boundary of the domain must be
    specified as well, by 'box'. For a ready network, the boundary will be
    imposed if provided. For a network will use pre-computed intersection and
    decomposition if these are available (attributes 'intersections' and
    'decomposition').

    TODO: The method finds the mapping between faces in one dimension and cells
        in a lower dimension, but the information is not used. Should be
        returned in a sensible format.

    Parameters:
        fracs (list, optional): List of either pre-defined fractures, or
            np.ndarrays, (each 3xn) of fracture vertexes.
        box (dictionary, optional). Domain specification. Should have keywords
            xmin, xmax, ymin, ymax, zmin, zmax.
        network (fractures.FractureNetwork, optional): A FractureNetwork
            containing fractures.

        The fractures should be specified either by a combination of fracs and
        box, or by network (possibly combined with box). See above.

        **kwargs: To be explored. Should contain the key 'gmsh_path'.

    Returns:
        list (length 4): For each dimension (3 -> 0), a list of all grids in
            that dimension.

    """

    # Verbosity level
    verbose = kwargs.get('verbose', 1)

    # File name for communication with gmsh
    file_name = kwargs.pop('file_name', 'gmsh_frac_file')

    if network is None:

        frac_list = []
        for f in fracs:
            if isinstance(f, fractures.Fracture):
                frac_list.append(f)
            else:
                # Convert the fractures from numpy representation to our 3D
                # fracture data structure..
                frac_list.append(fractures.Fracture(f))

        # Combine the fractures into a network
        network = fractures.FractureNetwork(frac_list, verbose=verbose,
                                            tol=kwargs.get('tol', 1e-4))

    # Impose domain boundary.
    if box is not None:
        network.impose_external_boundary(box)

    # Find intersections and split them, preparing the way for dumping the
    # network to gmsh
    if not network.has_checked_intersections:
        network.find_intersections()
    else:
        print('Use existing intersections')
    if not hasattr(network, 'decomposition'):
        network.split_intersections()
    else:
        print('Use existing decomposition')

    in_file = file_name + '.geo'
    out_file = file_name + '.msh'

    network.to_gmsh(in_file, **kwargs)
    gmsh_path = kwargs.get('gmsh_path')

    gmsh_opts = kwargs.get('gmsh_opts', {})
    gmsh_verbose = kwargs.get('gmsh_verbose', verbose)
    gmsh_opts['-v'] = gmsh_verbose
    gmsh_status = gmsh_interface.run_gmsh(gmsh_path, in_file, out_file, dims=3,
                                          **gmsh_opts)

    if verbose > 0:
        start_time = time.time()
        if gmsh_status == 0:
            print('Gmsh processed file successfully')
        else:
            print('Gmsh failed with status ' + str(gmsh_status))

    pts, cells, _, cell_info, phys_names = msh_io.read(out_file)

    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names
    phys_names = {v: k for k, v in phys_names.items()}

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

    return grids


def triangle_grid(fracs, domain, **kwargs):
    """
    Generate a gmsh grid in a 2D domain with fractures.

    The function uses modified versions of pygmsh and mesh_io,
    both downloaded from github.

    To be added:
    Functionality for tuning gmsh, including grid size, refinements, etc.

    Examples
    >>> p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
    >>> lines = np.array([[0, 2], [1, 3]])
    >>> char_h = 0.5 * np.ones(p.shape[1])
    >>> tags = np.array([1, 3])
    >>> fracs = {'points': p, 'edges': lines}
    >>> box = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2}
    >>> path_to_gmsh = '~/gmsh/bin/gmsh'
    >>> g = create_grid(fracs, box, gmsh_path=path_to_gmsh)
    >>> plot_grid.plot_grid(g)

    Parameters
    ----------
    fracs: (dictionary) Two fields: points (2 x num_points) np.ndarray,
        lines (2 x num_lines) connections between points, defines fractures.
    box: (dictionary) keys xmin, xmax, ymin, ymax, [together bounding box
        for the domain]
    **kwargs: To be explored. Must contain the key 'gmsh_path'
    Returns
    -------
    list (length 3): For each dimension (2 -> 0), a list of all grids in 
        that dimension.
    """
    # Verbosity level
    verbose = kwargs.get('verbose', 1)

    # File name for communication with gmsh
    file_name = kwargs.get('file_name', 'gmsh_frac_file')

    in_file = file_name + '.geo'
    out_file = file_name + '.msh'

    # Pick out fracture points, and their connections
    frac_pts = fracs['points']
    frac_con = fracs['edges']

    # Unified description of points and lines for domain, and fractures
    pts_all, lines = __merge_domain_fracs_2d(domain, frac_pts, frac_con)

    # We split all fracture intersections so that the new lines do not
    # intersect, except possible at the end points
    dx = np.array(
        [[domain['xmax'] - domain['xmin']], [domain['ymax'] - domain['ymin']]])
    pts_split, lines_split = cg.remove_edge_crossings(
        pts_all, lines, box=dx)
    # We find the end points that is shared by more than one intersection
    intersections = __find_intersection_points(lines_split)

    # Constants used in the gmsh.geo-file
    const = constants.GmshConstants()
    # Gridding size
    if 'mesh_size' in kwargs.keys():
        mesh_size, mesh_size_bound = \
            utils.determine_mesh_size(
                pts_split.shape[1], **kwargs['mesh_size'])
    else:
        mesh_size = None
        mesh_size_bound = None

    # gmsh options
    gmsh_path = kwargs.get('gmsh_path')

    gmsh_verbose = kwargs.get('gmsh_verbose', verbose)
    gmsh_opts = {'-v': gmsh_verbose}

    # Create a writer of gmsh .geo-files
    gw = gmsh_interface.GmshWriter(
        pts_split, lines_split, domain=domain, mesh_size=mesh_size,
        mesh_size_bound=mesh_size_bound, intersection_points=intersections)
    gw.write_geo(in_file)

    # Run gmsh
    gmsh_status = gmsh_interface.run_gmsh(gmsh_path, in_file, out_file, dims=2,
                                          **gmsh_opts)

    if verbose > 0:
        start_time = time.time()
        if gmsh_status == 0:
            print('Gmsh processed file successfully')
        else:
            print('Gmsh failed with status ' + str(gmsh_status))

    pts, cells, _, cell_info, phys_names = msh_io.read(out_file)
    warning.warn('The 2d gridder has not been validated for the new meshio'
                 + 'format. Use with care')

    # Invert phys_names dictionary to map from physical tags to corresponding
    # physical names
    phys_names = {v: k for k, v in phys_names.items()}

    # Create grids from gmsh mesh.
    g_2d = mesh_2_grid.create_2d_grids(pts, cells, is_embedded=False)
    g_1d, _ = mesh_2_grid.create_1d_grids(
        pts, cells, phys_names, cell_info, line_tag=const.PHYSICAL_NAME_FRACTURES)
    g_0d = mesh_2_grid.create_0d_grids(pts, cells)
    grids = [g_2d, g_1d, g_0d]

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

    return grids


def __merge_domain_fracs_2d(dom, frac_p, frac_l):
    """
    Merge fractures, domain boundaries and lines for compartments.
    The unified description is ready for feeding into meshing tools such as
    gmsh

    Parameters:
    dom: dictionary defining domain. fields xmin, xmax, ymin, ymax
    frac_p: np.ndarray. Points used in fracture definition. 2 x num_points.
    frac_l: np.ndarray. Connection between fracture points. 2 x num_fracs

    returns:
    p: np.ndarary. Merged list of points for fractures, compartments and domain
        boundaries.
    l: np.ndarray. Merged list of line connections (first two rows), tag
        identifying which type of line this is (third row), and a running index
        for all lines (fourth row)
    """

    # Use constants set outside. If we ever
    const = constants.GmshConstants()

    # First create lines that define the domain
    x_min = dom['xmin']
    x_max = dom['xmax']
    y_min = dom['ymin']
    y_max = dom['ymax']
    dom_p = np.array([[x_min, x_max, x_max, x_min],
                      [y_min, y_min, y_max, y_max]])
    dom_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).T

    num_dom_lines = dom_lines.shape[1]  # Should be 4

    # The  lines will have all fracture-related tags set to zero.
    # The plan is to ignore these tags for the boundary and compartments,
    # so it should not matter
    dom_tags = const.DOMAIN_BOUNDARY_TAG * np.ones((1, num_dom_lines))
    dom_l = np.vstack((dom_lines, dom_tags))

    # Also add a tag to the fractures, signifying that these are fractures
    frac_l = np.vstack((frac_l,
                        const.FRACTURE_TAG * np.ones(frac_l.shape[1])))

    # Merge the point arrays, compartment points first
    p = np.hstack((dom_p, frac_p))

    # Adjust index of fracture points to account for the compart points
    frac_l[:2] += dom_p.shape[1]

    l = np.hstack((dom_l, frac_l)).astype('int')

    # Add a second tag as an identifier of each line.
    l = np.vstack((l, np.arange(l.shape[1])))

    return p, l


def __find_intersection_points(lines):
    const = constants.GmshConstants()
    frac_id = np.ravel(lines[:2, lines[2] == const.FRACTURE_TAG])
    _, ia, count = np.unique(frac_id, True, False, True)
    return frac_id[ia[count > 1]]
