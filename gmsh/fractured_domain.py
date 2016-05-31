import numpy as np
import pygmsh as pg
import tempfile
import sys
import subprocess

from compgeom import basics
from gridding.gmsh import mesh_io
from core.grids import simplex
from utils import setmembership


def add_points(geom, pts, lcar):
    if pts.shape[0] == 2:
        pts = np.vstack((pts, np.zeros(pts.shape[1])))
    point_names = []
    for iter1 in range(pts.shape[1]):
        point_names.append(geom.add_point(pts[:, iter1], lcar[iter1]))
    return point_names


def embed_lines_in_surface(geom, lines, surface):
    for l in lines:
        geom._GMSH_CODE.append('Line {%s} In Surface{%s};' % (l, surface))


def __add_physical_surfaces(geom, surface, label):
    # There is a bug in pygmsh's corresponding function, so make a
    # replacement here.
    geom._GMSH_CODE.append(
        'DUMMY_VARIABLE_%s = "%s";' % (label, label)
    )
    geom._GMSH_CODE.append(
        'Physical Surface (Str(DUMMY_VARIABLE_%s)) = {%s};' % (label,
                                                                 surface)
        # The old file looked like this
        # 'Physical Surface("%s") = %s;' % (label, surface)
    )
    return


def __add_physical_lines(geom, lines, label):
    for iter, line in enumerate(lines):
        # There is a bug in pygmsh's corresponding function, so make a
        # replacement here.
        geom._GMSH_CODE.append(
            'DUMMY_VARIABLE_LINE_%s = "%s";' % (str(iter),
                                                'Line' + '_' + str(iter))
        )
        geom._GMSH_CODE.append(
            'Physical Line (Str(DUMMY_VARIABLE_LINE_%s)) = {%s};' % (str(iter),
                                                                    line)
            # The old file looked like this
            # 'Physical Surface("%s") = %s;' % (label, surface)
        )


def print_2d_domain(fracs, box, filename=None):

    if filename is None:
        filename = tempfile.mkstemp
    geofile = filename + '.geo'
    mshfile = filename + '.msh'

    geom = pg.Geometry()

    pts = fracs['points']
    edges = fracs['edges']
    num_edges = edges.shape[1]
    if edges.shape[0] == 2:
        edges = np.vstack((edges, np.arange(edges.shape[1])))

    domain = np.array([[box['xmax'] - box['xmin']],
                       [box['ymax'] - box['ymin']],
                       ])
    pts, edges = basics.remove_edge_crossings(pts, edges, domain)

    edge_ind = []
    lcar_points = 0.5 * np.ones(pts.shape[1])
    point_names = add_points(geom, pts, lcar_points)
    line_names = []

    for iter1 in range(edges.shape[1]):
        line_names.append(geom.add_line(point_names[edges[0, iter1]],
                                        point_names[edges[1, iter1]]))
    domain_surface = geom.add_rectangle(box['xmin'], box['xmax'],
                                        box['ymin'], box['ymax'], 0,
                                        box['lcar'])
    __add_physical_surfaces(geom, domain_surface, "WholeDomain")
    __add_physical_lines(geom, line_names, edges[2])
    # geom.add_physical_surface(domain_surface, 'WholeDomain')
    # geom.add_physical_surface(dom, 'WholeDomain')
    embed_lines_in_surface(geom, line_names, domain_surface)

    # Print the domain to a geo-file, run gmsh, and read the result
    __print_geo_file(geom, geofile)
    __run_gmsh(geofile, mshfile, ' -2 ')
    point, cells, physnames, cell_info = mesh_io.read(mshfile)

    print('Map cell_info and physnames to fractures')
    # gmsh works with 3D points, whereas we only need 2D
    point = point[:, :2].transpose()
    tri = cells['triangle'].transpose()
    g = simplex.TriangleGrid(point, tri)

    line_2_frac = __match_face_fractures(edges, cell_info, physnames)
    face_nodes = g.face_nodes.indices.reshape((2, -1), order='F')

    frac_face_nodes = cells['line'].transpose()
    _, frac_ind = setmembership.ismember_rows(frac_face_nodes, face_nodes)

    face_tags = np.zeros(g.num_faces)
    face_tags[:] = None
    face_tags[frac_ind] = line_2_frac
    g.face_info = {'tagcols': ['Fracture_faces'], 'tags': face_tags}


    return g






def __match_face_fractures(fracs, cell_info, phys_names):
    # Match generated line cells (1D elements) with pre-defined fractures
    # via physical names
    line_names = phys_names['line']

    cell_tags = cell_info['line'][:, 0]

    cell_2_frac = np.zeros(cell_tags.size)
    cell_2_frac[:] = np.nan

    for iter1, name in enumerate(line_names):
        line_tag = name[1]
        cell_2_frac[np.argwhere(cell_tags == line_tag)] = fracs[2, iter1]
    assert np.all(np.isfinite(cell_2_frac))
    return cell_2_frac


def __run_gmsh(infile, outfile, opts):
    if sys.platform == 'linux':
        path = '/home/eke001/Dropbox/workspace/lib/gmsh/run/linux/gmsh'
    else:
        path = 'C:\\Users\\keile\\Dropbox\\workspace\\lib\\gmsh\\run' \
               '\\win\\gmsh.exe'
    print('Start gmsh')
    cmd = path + ' ' + infile + opts + '-o ' + outfile
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    print(out)


def __print_geo_file(geom, filename):
    with open(filename, 'w') as f:
        f.write(geom.get_code())


if __name__ == '__main__':
    p = np.array([[-1, 1, 0, 0],
                  [0, 0, -1, 1]])

    lines = np.array([[0, 2],
                      [1, 3]])

    fracs = {'points': p, 'edges': lines}

    box = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'lcar': 0.7}
    g = print_2d_domain(fracs, box, 'gmsh_test')

