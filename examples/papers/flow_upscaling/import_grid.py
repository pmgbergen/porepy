import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def grid(file_geo, mesh_args, tol):

    p, e = raw_from_csv(file_geo, tol)

    domain = {'xmin': p[0].min(), 'xmax': p[0].max(),
              'ymin': p[1].min(), 'ymax': p[1].max()}
    frac_dict = {'points': p, 'edges': e}
    gb = pp.meshing.simplex_grid(frac_dict, domain, tol=tol["geo"], **mesh_args)

    return gb, domain

# ------------------------------------------------------------------------------#

def raw_from_csv(file_geo, mesh_args, tol):

    p, e = pp.importer.lines_from_csv(file_geo, mesh_args, polyline=True, skip_header=0)
    p, _ = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=tol["snap"])

    return p, e
