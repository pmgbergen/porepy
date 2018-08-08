import numpy as np

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids.grid import FaceTag

from porepy.numerics import elliptic
from porepy.utils import comp_geom as cg

import create_porepy_grid

# ------------------------------------------------------------------------------#


def add_data(gb, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param"])
    tol = 1e-5
    a = 1e-3

    for g, d in gb:
        param = Parameters(g)

        # Permeability
        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        if g.dim == 2:
            perm = tensor.SecondOrder(3, kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = tensor.SecondOrder(3, kxx=kxx, kyy=1, kzz=1)
            if g.dim == 1:
                R = cg.project_line_matrix(g.nodes, reference=[1, 0, 0])
                perm.rotate(R)

        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        # Boundaries
        bound_faces = g.get_boundary_faces()
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            bottom_corner = np.logical_and(
                bound_face_centers[0, :] < 0.1, bound_face_centers[1, :] < 0.1
            )

            top_corner = np.logical_and(
                bound_face_centers[0, :] > 0.9, bound_face_centers[1, :] > 0.9
            )

            labels = np.array(["neu"] * bound_faces.size)
            dir_faces = np.logical_or(bottom_corner, top_corner)
            labels[dir_faces] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[bottom_corner]] = 1
            bc_val[bound_faces[top_corner]] = -1

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        g = gb.sorted_nodes_of_edge(e)[0]
        d["kn"] = kf / gb.node_prop(g, "param").get_aperture()


# ------------------------------------------------------------------------------#


def main():

    gb = create_porepy_grid.create(mesh_size=0.01)

    internal_flag = FaceTag.FRACTURE
    [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

    # Assign parameters
    domain = gb.bounding_box(as_dict=True)
    kf = 1e-4
    add_data(gb, domain, kf)

    # Choose and define the solvers and coupler
    problem = elliptic.DualElliptic(gb)
    problem.solve()

    problem.split()
    problem.pressure("pressure")
    problem.project_discharge("P0u")

    problem.save(["pressure", "P0u"])


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------#
