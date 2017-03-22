import numpy as np
import tempfile
import time

import compgeom.basics as cg
from ..gmsh import gmsh_interface, mesh_io, mesh_2_grid
from core.grids import simplex, structured
from utils import setmembership
from gridding import constants
from gridding.fractured import grid_3d


def create_grid(fracs, domain, **kwargs):
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
    >>> fracs = {'points': p, 'edges': lines, 'lchar': char_h, 'tags': tags}
    >>> box = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'lcar': 0.7}
    >>> g = generate_grid(fracs, box, 'gmsh_test')
    >>> plot_grid.plot_grid_fractures(g)

    Parameters
    ----------
    fracs: (dictionary) Two fields: points (2 x num_points) np.ndarray,
        lines (2 x num_lines) connections between points, defines fractures.
    box: (dictionary) keys xmin, xmax, ymin, ymax, [together bounding box
        for the domain], lcar - characteristic cell size for the domain (see
        gmsh documentation)
    compartments:
    file_name: Name for where to put the in and outfile for gmsh. Defaults to a
        temporary file

    Returns
    -------
    g - triangular grid that conforms to the fratures. In addition to the
        standard fields, g has a variable face_info (dictionary),
        with elements tagcols (an explanation of the tags), and tags,
        telling which faces lies on the fractures (numbering corresponding
        to the columns in fracs['edges'])
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
    dx = np.array(
        [[domain['xmax'] - domain['xmin']], [domain['ymax'] - domain['ymin']]])
    pts_split, lines_split = cg.remove_edge_crossings(
        pts_all, lines, box=dx)

    # Constants used in the gmsh.geo-file
    const = constants.GmshConstants()

    # Replace this at some point with a more
    lchar_dom = kwargs.get('lchar_dom', None)
    lchar_frac = kwargs.get('lchar_frac', lchar_dom)

    # gmsh options
    gmsh_path = kwargs.get('gmsh_path')

    gmsh_verbose = kwargs.get('gmsh_verbose', verbose)
    gmsh_opts = {'-v': gmsh_verbose}

    # Create a writer of gmsh .geo-files
    gw = gmsh_interface.GmshWriter(
        pts_split, lines_split, domain=domain, mesh_size=lchar_dom,
        mesh_size_bound=lchar_frac)
    gw.write_geo(in_file)
    gmsh_status = gmsh_interface.run_gmsh(gmsh_path, in_file, out_file, dims=2,
                                          **gmsh_opts)

    if verbose > 0:
        start_time = time.time()
        if gmsh_status == 0:
            print('Gmsh processed file successfully')
        else:
            print('Gmsh failed with status ' + str(gmsh_status))

    pts, cells, phys_names, cell_info = mesh_io.read(out_file)

    # gmsh works with 3D points, whereas we only need 2D

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


def _create_2d_grids(pts, cells):

    triangles = cells['triangle'].transpose()
    # Construct grid
    g_2d = simplex.TriangleGrid(pts, triangles)

    # Create mapping to global numbering (will be a unit mapping, but is
    # crucial for consistency with lower dimensions)
    g_2d.global_point_ind = np.arange(pts.shape[1])

    # Convert to list to be consistent with lower dimensions
    # This may also become useful in the future if we ever implement domain
    # decomposition approaches based on gmsh.
    g_2d = [g_2d]
    return g_2d


def _create_1d_grids(pts, cells, phys_names, cell_info):
    # Recover lines
    # There will be up to three types of physical lines: intersections (between
    # fractures), fracture tips, and auxiliary lines (to be disregarded)

    # Add third coordinate
    pts = np.vstack((pts, np.zeros(pts.shape[1])))
    g_1d = []

    # If there are no fracture intersections, we return empty lists
    if not 'line' in cells:
        return g_1d, np.empty(0)

    gmsh_const = constants.GmshConstants()

    line_tags = cell_info['line'][:, 0]
    line_cells = cells['line']

    for i, pn in enumerate(phys_names['line']):
        # Index of the final underscore in the physical name. Chars before this
        # will identify the line type, the one after will give index
        offset_index = pn[2].rfind('_')
        loc_line_cell_num = np.where(line_tags == pn[1])[0]
        loc_line_pts = line_cells[loc_line_cell_num, :]

        assert loc_line_pts.size > 1

        line_type = pn[2][:offset_index]
        if line_type == gmsh_const.PHYSICAL_NAME_FRACTURE_TIP[:-1]:
            assert False, 'We should not have fracture tips in 2D'

        elif line_type == gmsh_const.PHYSICAL_NAME_FRACTURES[:-1]:
            loc_pts_1d = np.unique(loc_line_pts)  # .flatten()
            loc_coord = pts[:, loc_pts_1d]
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

    return g_1d


def _create_0d_grids(pts, cells):
    """
    We just call 0d-grid generation for 3D gridding.
    """
    return grid_3d._create_0d_grids(pts, cells)
