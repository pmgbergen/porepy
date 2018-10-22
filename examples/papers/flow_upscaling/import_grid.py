import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def grid(file_geo, mesh_args, tol):

    p, e, _ = raw_from_csv(file_geo, mesh_args, tol)

    domain = {'xmin': p[0].min(), 'xmax': p[0].max(),
              'ymin': p[1].min(), 'ymax': p[1].max()}
    frac_dict = {'points': p, 'edges': e}
    print(frac_dict)
    gb = pp.meshing.simplex_grid(frac_dict, domain, tol=tol["geo"], **mesh_args)

    return gb, domain

# ------------------------------------------------------------------------------#

def square_grid(mesh_args):

    domain = {'xmin': 0, 'xmax': 1,
              'ymin': 0, 'ymax': 1}
    gb = pp.meshing.simplex_grid([], domain, **mesh_args)

    return gb, domain

# ------------------------------------------------------------------------------#

def raw_from_csv(file_geo, mesh_args, tol):

    p, e, frac = pp.importer.lines_from_csv(file_geo, mesh_args, polyline=True, skip_header=0, return_frac_id=True)
    p -= np.amin(p, axis=1).reshape((-1, 1))
    p, _ = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=tol["snap"])

    return p, e, frac

# ------------------------------------------------------------------------------#
