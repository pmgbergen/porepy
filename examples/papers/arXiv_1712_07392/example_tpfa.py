import numpy as np
import scipy.sparse as sps

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.numerics import elliptic
from example_data import TPFAModelData

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

    gb.add_node_props(["frac_num"])
    for g, d in gb:
        d["problem"] = TPFAModelData(g, d, **extra)

        if g.dim == 3 or g.dim == 1:
            d["frac_num"] = -1 * np.ones(g.num_cells)
        else:
            d["frac_num"] = g.frac_num * np.ones(g.num_cells)


# ------------------------------------------------------------------------------#


def main():
    tol = 1e-6

    problem_kwargs = {}
    problem_kwargs["file_name"] = "solution"
    problem_kwargs["folder_name"] = "tpfa"

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

    problem = elliptic.EllipticModel(gb, **problem_kwargs)

    # Assign parameters
    add_data(gb, domain, tol)

    problem.solve()
    problem.split()

    problem.pressure("pressure")
    problem.discharge("discharge")
    problem.save(["pressure", "frac_num"])

    problem_kwargs["file_name"] = "transport"

    for g, d in gb:
        d["problem"] = AdvectiveModelData(g, d, domain, tol)

    advective = AdvectiveModel(gb, **problem_kwargs)
    advective.solve()
    advective.save()


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------#
