import numpy as np
import tempfile
from compgeom import basics as geom_basics
from ..gmsh import gmsh_interface
from ..gmsh import mesh_io
from core.grids import simplex
from utils import setmembership
import gridding.constants


def create_grid(fracs, domain, compartments=None, file_name='fractures_gmsh'):
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

    in_file = file_name + '.geo'
    out_file = file_name + '.msh'

    # Pick out fracture points, and their connections
    frac_pts = fracs['points']
    frac_con = fracs['edges']

    # If no tags are provided for the fractures, we assign increasing value
    # for each column
    # if frac_con.shape[0] == 2:
    #     frac_con = np.vstack((frac_con, np.arange(frac_con.shape[1])))

    # Unified description of points and lines for domain, fractures and
    # internal compartments
    pts, lines, line_tag_description = \
        __merge_fracs_compartments_domain_2d(domain, frac_pts, frac_con,
                                             compartments)
    dx = np.array(
        [[domain['xmax'] - domain['xmin']], [domain['ymax'] - domain['ymin']]])

    pts_split, lines_split = geom_basics.remove_edge_crossings(pts, lines, dx)

    gw = gmsh_interface.GmshWriter(pts_split, lines_split)

    # Replace this at some point with a more
    gw.lchar = np.mean(fracs['lchar'])

    gw.write_geo(in_file)

    gmsh_interface.run_gmsh(in_file, out_file, 2)

    points, cells, phys_names, cell_info = mesh_io.read(out_file)

    # gmsh works with 3D points, whereas we only need 2D
    points = points[:, :2].transpose()

    # Recover triangles
    triangles = cells['triangle'].transpose()
    # Construct grid
    g = simplex.TriangleGrid(points, triangles)

    # Next, recover faces that coincides with fractures

    # Fracture nodes
    frac_face_nodes = cells['line'].transpose()
    # Nodes of faces in the grid
    face_nodes = g.face_nodes.indices.reshape((2, g.num_faces),
                                              order='F').astype('int')

    _, frac_ind = setmembership.ismember_rows(np.sort(frac_face_nodes, axis=0),
                                              np.sort(face_nodes, axis=0))

    # Assign tags according to fracture faces
    face_tags = np.zeros(g.num_faces)
    face_tags[:] = -1
    frac_face_tags = __match_face_fractures(cell_info,
                                            phys_names).astype('int')
    lines_frac_face = lines[2:, frac_face_tags]
    # Faces corresponding to real fractures - not compartments or boundaries
    const = gridding.constants.GmshConstants()
    real_frac_ind = np.ravel(np.argwhere(lines_frac_face[0] ==
                                         const.FRACTURE_TAG))

    # Counting measure of the real fractures
    real_frac_num = lines_frac_face[1, real_frac_ind]
    # Remove contribution from domain and compartments
    real_frac_num = real_frac_num - np.min(real_frac_num)

    key = 'tags'
    if key in fracs:
        face_tags[real_frac_ind] = fracs[real_frac_num]
    else:
        face_tags[real_frac_ind] = real_frac_num

    g.face_info = {'tagcols': ['Fracture_faces'], 'tags': face_tags}

    return g


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
    constants = gridding.constants.GmshConstants()

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
    dom_tags = constants.DOMAIN_BOUNDARY_TAG * np.ones((1, num_dom_lines))
    dom_l = np.vstack((dom_lines, dom_tags))
    comp_tags = constants.COMPARTMENT_BOUNDARY_TAG * np.ones((1, num_comps))
    comp_l = np.vstack((comp_lines, comp_tags))

    # Also add a tag to the fractures, signifying that these are fractures
    frac_l = np.vstack((frac_l,
                        constants.FRACTURE_TAG * np.ones(frac_l.shape[1])))

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

    # Tags of the cells, corrensponding to the physical entities (gmsh
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
