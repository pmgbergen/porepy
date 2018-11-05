import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def grid(file_geo, mesh_args, tol, enlarge=None, xi=0.1):

    p, e, _ = raw_from_csv(file_geo, mesh_args, tol=tol["snap"])

    domain = {'xmin': p[0].min(), 'xmax': p[0].max(),
              'ymin': p[1].min(), 'ymax': p[1].max()}
    subdom = []

    if enlarge is not None:
        delta_x = domain["xmax"] - domain["xmin"]
        delta_y = domain["ymax"] - domain["ymin"]
        subdom = {"edges": np.array([0, 1]).reshape((2, 1))}

        if enlarge == "left":
            subdom["points"] = np.array([[domain["xmin"], domain["xmin"]],
                                         [domain["ymin"], domain["ymax"]]])
            domain["xmin"] -= xi*delta_x
        elif enlarge == "right":
            subdom["points"] = np.array([[domain["xmax"], domain["xmax"]],
                                         [domain["ymin"], domain["ymax"]]])
            domain["xmax"] += xi*delta_x
        elif enlarge == "top":
            subdom["points"] = np.array([[domain["xmin"], domain["xmax"]],
                                         [domain["ymax"], domain["ymax"]]])
            domain["ymax"] += xi*delta_y
        elif enlarge == "bottom":
            subdom["points"] = np.array([[domain["xmin"], domain["xmax"]],
                                         [domain["ymin"], domain["ymin"]]])
            domain["ymin"] -= xi*delta_y
        else:
            raise ValueError

    frac_dict = {'points': p, 'edges': e}
    gb = pp.meshing.simplex_grid(frac_dict, domain, tol=tol["geo"], subdomains=subdom, **mesh_args)

    return gb, domain

# ------------------------------------------------------------------------------#

def square_grid(mesh_args):

    domain = {'xmin': 0, 'xmax': 1,
              'ymin': 0, 'ymax': 1}
    gb = pp.meshing.simplex_grid([], domain, **mesh_args)

    return gb, domain

# ------------------------------------------------------------------------------#

def raw_from_csv(file_geo, mesh_args, tol=None):

    p, e, frac = pp.importer.lines_from_csv(file_geo, mesh_args, polyline=True,
                                            skip_header=0, return_frac_id=True)
    p -= np.amin(p, axis=1).reshape((-1, 1))
    if tol is None:
        tol = mesh_args["tol"]
    p, _ = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=tol)

    return p, e, frac

# ------------------------------------------------------------------------------#
