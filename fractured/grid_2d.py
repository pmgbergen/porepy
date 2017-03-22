import numpy as np
import tempfile
import compgeom.basics as cg
from ..gmsh import gmsh_interface
from ..gmsh import mesh_io
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

    # If no tags are provided for the fractures, we assign increasing value
    # for each column
    # if frac_con.shape[0] == 2:
    #     frac_con = np.vstack((frac_con, np.arange(frac_con.shape[1])))

    # Compartments
    compartments = kwargs.get('compartments', np.zeros((2, 0)))

    # Unified description of points and lines for domain, fractures and
    # internal compartments
    pts, lines, line_tag_description = __merge_fracs_compartments_domain_2d(
        domain, frac_pts, frac_con, compartments)

    dx = np.array(
        [[domain['xmax'] - domain['xmin']], [domain['ymax'] - domain['ymin']]])
    pts_split, lines_split = cg.remove_edge_crossings(
        pts, lines, box=dx)

    # Constants used in the gmsh.geo-file
    const = constants.GmshConstants()

    # Replace this at some point with a more
    lchar_dom = kwargs.get('lchar_dom', None)
    lchar_frac = kwargs.get('lchar_frac', lchar_dom)
    frac_pts = np.ravel(lines_split[:2, np.argwhere(lines_split[2] ==
                                                    const.FRACTURE_TAG)])

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

    pts, cells, phys_names, cell_info = mesh_io.read(out_file)

    # gmsh works with 3D points, whereas we only need 2D
    pts = pts[:, :2].transpose()

    g_2d = _create_2d_grids(pts, cells)
    g_1d = _create_1d_grids(pts, cells, phys_names, cell_info)
    g_0d = _create_0d_grids(pts, cells)
    grids = [g_2d, g_1d, g_0d]
    bucket = grid_3d.assemble_in_bucket(grids)

    # Next, recover faces that coincides with fractures, and assign the
    # appropriate tags to the nodes

    # Nodes that define 1D physical entities, that is compartment lines
    # or fractures, as computed by gmsh.
    frac_face_nodes = cells['line'].transpose()
    # Nodes of faces in the grid
    g = bucket.grids_of_dimension(2)[0]
    face_nodes = g.face_nodes.indices.reshape((2, g.num_faces),
                                              order='F').astype('int')
    # Match faces in the grid with the 1D lines in gmsh. frac_ind gives a
    # mapping from faces laying on Physical Lines to all faces in the grid
    _, frac_ind = setmembership.ismember_rows(np.sort(frac_face_nodes, axis=0),
                                              np.sort(face_nodes, axis=0))
    frac_ind = np.array(frac_ind)

    # Find tags (in gmsh sense) of the line elements as specified as cell_info.
    # The 1-1 correspondence between line elements in cells and cell_info means
    # the tags will correspond to the indices in frac_ind
    frac_face_tags = __match_face_fractures(cell_info,
                                            phys_names).astype('int')

    # Having the pieces necessary for the connection between physical lines and
    # fracture faces, we also need to identify the correct 'unsplit' lines,
    # that is fractures as they were passed to this function

    # Lines that are not compartment or fracture were not passed to gmsh
    line_is_constraint = np.ravel(np.argwhere(
        np.logical_or(lines_split[2] == const.COMPARTMENT_BOUNDARY_TAG,
                      lines_split[2] == const.FRACTURE_TAG)))
    constraints = lines_split[:, line_is_constraint]

    # Among the constraints, we are only interested in the ones that are tagged
    # as fractures.
    # NOTE: If we ever need to identify faces laying on compartments, this is
    # the place to do it. However, a more reasonable approach probably is to
    # ignore compartments altogether, and instead pass the lines as extra
    # fractures.
    lines_types = constraints[2:, frac_face_tags]
    # Faces corresponding to real fractures - not compartments or boundaries
    real_frac_ind = np.ravel(np.argwhere(lines_types[0] ==
                                         const.FRACTURE_TAG))
    # Counting measure of the real fractures
    real_frac_num = lines_types[1, real_frac_ind]
    # Remove contribution from domain and compartments
    real_frac_num = real_frac_num - np.min(real_frac_num)

    # Assign tags according to fracture faces
    face_tags = np.zeros(g.num_faces)
    # The default value is set to nan. Not sure if this is the optimal option
    face_tags[:] = np.nan

    key = 'tags'
    if key in fracs:
        face_tags[frac_ind[real_frac_ind]] = fracs.g[real_frac_num]
    else:
        face_tags[frac_ind[real_frac_ind]] = real_frac_num

    g.face_info = {'tagcols': ['Fracture_faces'], 'tags': face_tags}

    return bucket


def __merge_fracs_compartments_domain_2d(dom, frac_p, frac_l, comp):
    """
    Merge fractures, domain boundaries and lines for compartments.
    The unified description is ready for feeding into meshing tools such as
    gmsh

    Parameters:
    dom: dictionary defining domain. fields xmin, xmax, ymin, ymax
    frac_p: np.ndarray. Points used in fracture definition. 2 x num_points.
    frac_l: np.ndarray. Connection between fracture points. 2 x num_fracs
    comp: Compartment lines

    returns:
    p: np.ndarary. Merged list of points for fractures, compartments and domain
        boundaries.
    l: np.ndarray. Merged list of line connections (first two rows), tag
        identifying which type of line this is (third row), and a running index
        for all lines (fourth row)
    tag_description: list describing the tags, that is, row 3 and 4 in l
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

    # Then the compartments
    # First ignore compartment lines that have no y-component
    comp = np.squeeze(comp[:, np.argwhere(np.abs(comp[1]) > 1e-5)])

    num_comps = comp.shape[1]
    # Assume that the compartments are not intersecting at the boundary.
    # This may need revisiting when we allow for general lines
    # (not only y=const)
    comp_p = np.zeros((2, num_comps * 2))

    for i in range(num_comps):
        # The line is defined as by=1, thus y = 1/b
        comp_p[:, 2 * i: 2 * (i + 1)] = np.array([[x_min, x_max],
                                                  [1. / comp[1, i],
                                                   1. / comp[1, i]]])

    # Define compartment lines
    comp_lines = np.arange(2 * num_comps, dtype='int').reshape((2, -1),
                                                               order='F')

    # The  lines will have all fracture-related tags set to zero.
    # The plan is to ignore these tags for the boundary and compartments,
    # so it should not matter
    dom_tags = const.DOMAIN_BOUNDARY_TAG * np.ones((1, num_dom_lines))
    dom_l = np.vstack((dom_lines, dom_tags))
    comp_tags = const.COMPARTMENT_BOUNDARY_TAG * np.ones((1, num_comps))
    comp_l = np.vstack((comp_lines, comp_tags))

    # Also add a tag to the fractures, signifying that these are fractures
    frac_l = np.vstack((frac_l,
                        const.FRACTURE_TAG * np.ones(frac_l.shape[1])))

    # Merge the point arrays, compartment points first
    p = np.hstack((dom_p, comp_p, frac_p))

    # Adjust index of compartment points to account for domain points coming
    # before them in the list
    comp_l[:2] += dom_p.shape[1]

    # Adjust index of fracture points to account for the compart points
    frac_l[:2] += dom_p.shape[1] + comp_p.shape[1]

    l = np.hstack((dom_l, comp_l, frac_l)).astype('int')

    # Add a second tag as an identifier of each line.
    l = np.vstack((l, np.arange(l.shape[1])))

    tag_description = ['Line type', 'Line number']

    return p, l, tag_description


def __match_face_fractures(cell_info, phys_names):
    # Match generated line cells (1D elements) with pre-defined fractures
    # via physical names.
    # Note that fracs and phys_names are both representation of the
    # fractures (after splitting due to intersections); fracs is the
    # information that goes into the gmsh algorithm (in the form of physical
    #  lines), while phys_names are those physical names generated in the
    # .msh file
    line_names = phys_names['line']

    # Tags of the cells, corresponding to the physical entities (gmsh
    # terminology) are known to be in the first column
    cell_tags = cell_info['line'][:, 0]

    # Prepare array for tags. Assign nan values for faces an a starting
    # point, these should be overwritten later
    cell_2_frac = np.zeros(cell_tags.size)
    cell_2_frac[:] = np.nan

    # Loop over all physical lines, compare the tag of the cell with the tag
    # of the physical line.
    for iter1, name in enumerate(line_names):
        line_tag = name[1]
        frac_name = name[2]
        ind = frac_name.find('_')
        cell_2_frac[np.argwhere(cell_tags == line_tag)] = frac_name[ind + 1:]

    # Sanity check, we should have assigned tags to all fracture faces by now
    assert np.all(np.isfinite(cell_2_frac))
    return cell_2_frac


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
