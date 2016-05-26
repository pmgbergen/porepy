import numpy as np
import pygmsh as pg
import tempfile
import sys
import subprocess

from compgeom import basics


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
    # geom.add_physical_surface(domain_surface, 'WholeDomain')
    # geom.add_physical_surface(dom, 'WholeDomain')
    embed_lines_in_surface(geom, line_names, domain_surface)

    __print_geo_file(geom, geofile)
    __run_gmsh(geofile, mshfile, ' -2 ')
    # points, cells, _, _, _


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
    print_2d_domain(fracs, box, 'gmsh_test')

