import numpy as np
import pygmsh as pg
import tempfile
import sys
import subprocess

from compgeom import basics
from gridding.gmsh import mesh_io
from core.grids import simplex
from utils import setmembership
from viz import plot_grid


def generate_grid(fracs, box, filename=None):
    """
    Generate a gmsh grid in a 2D domain with fractures.

    The function uses modified versions of pygmsh and mesh_io,
    both downloaded from github.

    To be added:
    Functionality for tuning gmsh, including grid size, refinements, etc.

    Examples
    >>> p = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])
    >>> lines = np.array([[0, 2], [1, 3]])
    >>> fracs = {'points': p, 'edges': lines}
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
    filename: Name for where to put the in and outfile for gmsh. Defaults to a
        temporary file

    Returns
    -------
    g - triangular grid that conforms to the fratures. In addition to the
        standard fields, g has a variable face_info (dictionary),
        with elements tagcols (an explanation of the tags), and tags,
        telling which faces lies on the fractures (numbering corresponding
        to the columns in fracs['edges'])
    """

    # Filename for gmsh i/o, if none is specified
    if filename is None:
        filename = tempfile.mkstemp
    geofile = filename + '.geo'
    mshfile = filename + '.msh'

    # Initialize pygmsh geometry object, this is used for generating .geo file
    geom = pg.Geometry()

    # Pick out fracture points, and their connections
    pts = fracs['points']
    edges = fracs['edges']

    # If no tags are provided for the fractures, we assign increasing value
    # for each column
    if edges.shape[0] == 2:
        edges = np.vstack((edges, np.arange(edges.shape[1])))

    # Bounding box for the domain.
    domain = np.array([[box['xmax'] - box['xmin']],
                       [box['ymax'] - box['ymin']],
                       ])
    # Remove intersections before passing to gmsh.
    pts, edges = basics.remove_edge_crossings(pts, edges, domain)

    # Default values for characteristic point size. Need to give input here.
    lcar_points = 0.5 * np.ones(pts.shape[1])
    point_names = __add_points(geom, pts, lcar_points)

    # Add domain to geometry description
    domain_surface = geom.add_rectangle(box['xmin'], box['xmax'],
                                        box['ymin'], box['ymax'], 0,
                                        box['lcar'])
    # Defnie the domain as a physical surface
    __add_physical_surfaces(geom, domain_surface, "WholeDomain")

    # Add intersections between points (that is, fractures). Store their
    # names, we need this for defining physical lines and embeding them in
    # the domain later
    line_names = []
    for iter1 in range(edges.shape[1]):
        line_names.append(geom.add_line(point_names[edges[0, iter1]],
                                        point_names[edges[1, iter1]]))
    # Define fractures as physical lines, and embed them in the domain
    # surface. This will make gmsh conform to the line
    __add_physical_lines(geom, line_names)
    __embed_lines_in_surface(geom, line_names, domain_surface)

    # Print the domain to a geo-file, run gmsh, and read the result
    __print_geo_file(geom, geofile)
    __run_gmsh(geofile, mshfile, ' -2 ')
    # Points are nodes in the grid. Cells contain both triangles, and lines
    # along Physical lines (fractures). Physnames are the physical lines,
    # cell_info various tags
    point, cells, physnames, cell_info = mesh_io.read(mshfile)

    # gmsh works with 3D points, whereas we only need 2D
    point = point[:, :2].transpose()
    tri = cells['triangle'].transpose()
    # Construct grid
    g = simplex.TriangleGrid(point, tri)

    line_2_frac = __match_face_fractures(edges, cell_info, physnames)
    # Nodes of faces in the grid
    face_nodes = g.face_nodes.indices.reshape((2, -1), order='F')

    # Nodes of faces defined as laying on a physical line
    frac_face_nodes = cells['line'].transpose()
    # Mapping between the two sets of fracture nodes
    _, frac_ind = setmembership.ismember_rows(frac_face_nodes, face_nodes)

    # Assign tags according to fracture faces
    face_tags = np.zeros(g.num_faces)
    face_tags[:] = None
    face_tags[frac_ind] = line_2_frac
    g.face_info = {'tagcols': ['Fracture_faces'], 'tags': face_tags}

    return g


def __add_points(geom, pts, lcar):
    if pts.shape[0] == 2:
        pts = np.vstack((pts, np.zeros(pts.shape[1])))
    point_names = []
    for iter1 in range(pts.shape[1]):
        point_names.append(geom.add_point(pts[:, iter1], lcar[iter1]))
    return point_names


def __embed_lines_in_surface(geom, lines, surface):
    for l in lines:
        geom._GMSH_CODE.append('Line {%s} In Surface{%s};' % (l, surface))


def __add_physical_surfaces(geom, surface, label):
    """
    Workaround for adding physical lines; the standard version in pygmsh
    gave inexplicable errors, seemingly related to line shifts
    """
    # There is a bug in pygmsh's corresponding function, so make a
    # replacement here.
    geom._GMSH_CODE.append(
        'DUMMY_VARIABLE_%s = "%s";' % (label, label)
    )
    geom._GMSH_CODE.append(
        'Physical Surface (Str(DUMMY_VARIABLE_%s)) = {%s};' % (label, surface)
        # The old file looked like this
        # 'Physical Surface("%s") = %s;' % (label, surface)
    )
    return


def __add_physical_lines(geom, lines):
    """
    Workaround for adding physical lines; the standard version in pygmsh
    gave inexplicable errors, seemingly related to line shifts
    """
    for iter1, line in enumerate(lines):
        # There is a bug in pygmsh's corresponding function, so make a
        # replacement here.
        geom._GMSH_CODE.append(
            'DUMMY_VARIABLE_LINE_%s = "%s";' % (str(iter1),
                                                'Line' + '_' + str(iter1))
        )
        geom._GMSH_CODE.append(
            'Physical Line (Str(DUMMY_VARIABLE_LINE_%s)) = {%s};' % (str(iter1)
                                                                     , line)
            # The old file looked like this
            # 'Physical Surface("%s") = %s;' % (label, surface)
        )


def __match_face_fractures(fracs, cell_info, phys_names):
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
    #  of the physical line.
    # As tag, assign the value in fracs[2]; see what happens to this in the
    # calling method
    for iter1, name in enumerate(line_names):
        line_tag = name[1]
        cell_2_frac[np.argwhere(cell_tags == line_tag)] = fracs[2, iter1]

    # Sanity check, we should have asigned tags to all fracture faces by now
    assert np.all(np.isfinite(cell_2_frac))
    return cell_2_frac


def __run_gmsh(infile, outfile, opts):
    # Run GMSH.
    # Note that the path to gmsh is hardcoded here
    if sys.platform == 'linux':
        path = '/home/eke001/Dropbox/workspace/lib/gmsh/run/linux/gmsh'
    else:
        path = 'C:\\Users\\keile\\Dropbox\\workspace\\lib\\gmsh\\run' \
               '\\win\\gmsh.exe'
    cmd = path + ' ' + infile + opts + '-o ' + outfile
    subprocess.check_output(cmd, stderr=subprocess.STDOUT)


def __print_geo_file(geom, filename):
    with open(filename, 'w') as f:
        f.write(geom.get_code())
