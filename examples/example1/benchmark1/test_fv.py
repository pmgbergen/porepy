"""
This example contains the set up and computation of the blocking version of the
benchmark1 with the fv discretizations. Note the choice between tpfa and mpfa in
line 79.
"""
import numpy as np
import scipy.sparse as sps
import porepy as pp

from examples.example1.benchmark1.test_vem import make_grid_bucket

# ------------------------------------------------------------------------------#


def add_data(gb, domain, kf, mesh_value):
    """
    Define the permeability, apertures, boundary conditions and sources
    """
    tol = 1e-5
    a = 1e-4

    for g, d in gb:
        # Assign apertures
        a_dim = np.power(a, gb.dim_max() - g.dim)
        aperture = np.ones(g.num_cells) * a_dim

        # Permeability
        # Use fracture value in the fractures, i.e., the lower dimensional grids
        k_frac = np.power(kf, g.dim < gb.dim_max())
        p = pp.SecondOrderTensor(np.ones(g.num_cells) * k_frac)

        # Source term
        specified_parameters = {
            "aperture": aperture,
            "source": np.zeros(g.num_cells),
            "second_order_tensor": p,
        }
        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[right] = "dir"

            bc_val = np.zeros(g.num_faces)

            if g.dim == 2:
                # Account for the double inflow on the matrix-fracture overlap
                left_mid = np.array(
                    np.absolute(g.face_centers[1, bound_faces[left]] - 0.5) < mesh_value
                )
                bc_val[bound_faces[left]] = (
                    -g.face_areas[bound_faces[left]] + left_mid * 0.5 * a
                )
            else:
                bc_val[bound_faces[left]] = -g.face_areas[bound_faces[left]] * a

            bc_val[bound_faces[right]] = np.ones(np.sum(right))

            bound = pp.BoundaryCondition(g, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})
        else:
            bound = pp.BoundaryCondition(g, np.empty(0), np.empty(0))

        pp.initialize_default_data(g, d, "flow", specified_parameters)

    # Assign coupling permeability
    for _, d in gb.edges():
        mg = d["mortar_grid"]
        kn = 2 * kf * np.ones(mg.num_cells) / a
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], [{"normal_diffusivity": kn}])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}


# ------------------------------------------------------------------------------#


def write_network(file_name):
    network = "FID,START_X,START_Y,END_X,END_Y\n"
    network += "0,0,0.5,1,0.5\n"
    network += "1,0.5,0,0.5,1\n"
    network += "2,0.5,0.75,1,0.75\n"
    network += "3,0.75,0.5,0.75,1\n"
    network += "4,0.5,0.625,0.75,0.625\n"
    network += "5,0.625,0.5,0.625,0.75\n"
    with open(file_name, "w") as text_file:
        text_file.write(network)


# ------------------------------------------------------------------------------#


def main(kf, description, multi_point, if_export=False):
    mesh_size = 0.045
    gb, domain = make_grid_bucket(mesh_size)
    # Assign parameters
    add_data(gb, domain, kf, mesh_size)

    key = "flow"
    if multi_point:
        method = pp.Mpfa(key)
    else:
        method = pp.Tpfa(key)

    coupler = pp.RobinCoupling(key, method)

    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
        d[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {"mortar_solution": {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            "lambda": {
                g1: ("pressure", "diffusive"),
                g2: ("pressure", "diffusive"),
                e: ("mortar_solution", coupler),
            }
        }
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    assembler = pp.Assembler(gb)

    # Discretize
    A, b = assembler.assemble_matrix_rhs()
    p = sps.linalg.spsolve(A, b)

    assembler.distribute_variable(p)

    if if_export:
        save = pp.Exporter(gb, "fv", folder="fv_" + description)
        save.write_vtk(["pressure"])


# ------------------------------------------------------------------------------#


def test_tpfa_blocking():
    kf = 1e-4
    if_export = True
    main(kf, "blocking_tpfa", multi_point=False, if_export=if_export)


# ------------------------------------------------------------------------------#


def test_tpfa_permeable():
    kf = 1e4
    if_export = True
    main(kf, "permeable_tpfa", multi_point=False, if_export=if_export)


# ------------------------------------------------------------------------------#


def test_mpfa_blocking():
    kf = 1e-4
    if_export = True
    main(kf, "blocking_mpfa", multi_point=True, if_export=if_export)


# ------------------------------------------------------------------------------#


def test_mpfa_permeable():
    kf = 1e4
    if_export = True
    main(kf, "permeable_mpfa", multi_point=True, if_export=if_export)


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    test_tpfa_blocking()
