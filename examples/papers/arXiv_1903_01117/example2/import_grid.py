import numpy as np
import porepy as pp


def import_grid(file_geo, tol):

    frac = pp.Fracture(np.array([[0, 10, 10, 0], [0, 0, 10, 10], [8, 2, 2, 8]]) * 10)
    network = pp.FractureNetwork3d([frac], tol=tol)

    domain = {"xmin": 0, "xmax": 100, "ymin": 0, "ymax": 100, "zmin": 0, "zmax": 100}
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network.to_gmsh("dummy.geo")

    gb = pp.fracture_importer.dfm_from_gmsh(file_geo, 3, network)
    gb.compute_geometry()

    return gb, domain
