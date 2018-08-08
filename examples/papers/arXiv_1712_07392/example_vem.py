import numpy as np
import scipy.sparse as sps
import logging

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.grids import coarsening

from porepy.numerics import elliptic
from example_data import VEMModelData

from example_advective import AdvectiveModel, AdvectiveModelData

# ------------------------------------------------------------------------------#


def add_data(gb, domain, tol):

    extra = {
        "domain": domain,
        "tol": tol,
        "special_fracture": 6,
        "aperture": 1e-2,
        "gb": gb,
        "km": 1,
        "kf_low": 1e-4,
        "kf_high": 1e4,
    }

    gb.add_node_props(["is_tangential", "frac_num"])
    for g, d in gb:
        d["problem"] = VEMModelData(g, d, **extra)

        d["is_tangential"] = True

        if g.dim == 3 or g.dim == 1:
            d["frac_num"] = -1 * np.ones(g.num_cells)
        else:
            d["frac_num"] = g.frac_num * np.ones(g.num_cells)

    # Assign coupling permeability
    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        g_l, g_h = gb.sorted_nodes_of_edge(e)

        if g_h.dim == gb.dim_max() and g_l.frac_num == extra["special_fracture"]:
            kxx = extra["kf_high"]

        elif g_h.dim < gb.dim_max() and g_h.frac_num == extra["special_fracture"]:
            neigh = gb.node_neighbors(g_l, only_higher=True)
            frac_num = np.array([gh.frac_num for gh in neigh])
            if np.any(frac_num == 1):
                kxx = extra["kf_high"]
            else:
                kxx = extra["kf_low"]
        else:
            kxx = extra["kf_low"]

        aperture = gb.node_prop(g_l, "param").get_aperture()
        d["kn"] = kxx / aperture


# ------------------------------------------------------------------------------#


def main(coarse):
    tol = 1e-6

    problem_kwargs = {}
    problem_kwargs["file_name"] = "solution"
    if coarse:
        problem_kwargs["folder_name"] = "vem_coarse"
    else:
        problem_kwargs["folder_name"] = "vem"

    h = 0.08
    grid_kwargs = {}
    grid_kwargs["mesh_size"] = {
        "mode": "constant",
        "value": h,
        "bound_value": h,
        "tol": tol,
    }

    file_dfm = "dfm.csv"
    gb, domain = importer.dfm_3d_from_csv(file_dfm, tol, **grid_kwargs)
    gb.compute_geometry()
    if coarse:
        coarsening.coarsen(gb, "by_volume")

    problem = elliptic.DualEllipticModel(gb, **problem_kwargs)

    # Assign parameters
    add_data(gb, domain, tol)

    problem.solve()
    problem.split()

    problem.pressure("pressure")
    problem.discharge("discharge")
    problem.project_discharge("P0u")
    problem.save(["pressure", "P0u", "frac_num"])

    problem_kwargs["file_name"] = "transport"

    for g, d in gb:
        d["problem"] = AdvectiveModelData(g, d, domain, tol)

    advective = AdvectiveModel(gb, **problem_kwargs)
    advective.solve()
    advective.save()


# ------------------------------------------------------------------------------#


if __name__ == "__main__":
    main(coarse=False)
#    main(coarse=True)

# ------------------------------------------------------------------------------#
