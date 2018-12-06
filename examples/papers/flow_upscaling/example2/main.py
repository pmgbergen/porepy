import numpy as np
import porepy as pp

from examples.papers.flow_upscaling.upscaling import upscaling

if __name__ == "__main__":
    file_geo = "algeroyna_1to10.csv"
    folder = "solution"

    mesh_args = {'mesh_size_frac': 10}
    tol = {"geo": 1e-4, "snap": 1e-3}

    # define the physical data
    aperture = pp.MILLIMETER
    data = {
        "aperture": aperture,
        "kf": aperture**2 / 12,
        "km": 1e-14,
        "tol": tol["geo"]
    }

    dfn = False
    k = upscaling(file_geo, folder, dfn, data, mesh_args, tol)

    print(k.perm)
