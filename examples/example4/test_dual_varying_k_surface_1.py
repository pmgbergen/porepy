"""
3d convergence test for dual VEM for heterogeneous permeability.
"""
import numpy as np
import scipy.sparse as sps
import unittest
import porepy as pp

# ------------------------------------------------------------------------------#


def rhs(x, y, z):
    return 8. * z * (125. * x ** 2 + 200. * y ** 2 + 425. * z ** 2 + 2.)


# ------------------------------------------------------------------------------#


def solution(x, y, z):
    return x ** 2 * z + 4. * y ** 2 * np.sin(np.pi * y) - 3. * z ** 3


# ------------------------------------------------------------------------------#


def permeability(x, y, z):
    return 1. + 100. * (x ** 2 + y ** 2 + z ** 2.)


# ------------------------------------------------------------------------------#


def add_data(g):
    """
    Define the permeability, apertures, boundary conditions
    """
    param = pp.Parameters(g)

    # Permeability
    kxx = np.array([permeability(*pt) for pt in g.cell_centers.T])
    param.set_tensor("flow", pp.SecondOrderTensor(3, kxx))

    # Source term
    source = np.array([rhs(*pt) for pt in g.cell_centers.T])
    param.set_source("flow", g.cell_volumes * source)

    # Boundaries
    bound_faces = g.get_all_boundary_faces()
    bound_face_centers = g.face_centers[:, bound_faces]

    labels = np.array(["dir"] * bound_faces.size)

    bc_val = np.zeros(g.num_faces)
    bc_val[bound_faces] = np.array([solution(*pt) for pt in bound_face_centers.T])

    param.set_bc("flow", pp.BoundaryCondition(g, bound_faces, labels))
    param.set_bc_val("flow", bc_val)

    return {"param": param}


# ------------------------------------------------------------------------------#


def error_p(g, p):

    sol = np.array([solution(*pt) for pt in g.cell_centers.T])
    return np.sqrt(np.sum(np.power(np.abs(p - sol), 2) * g.cell_volumes))


# ------------------------------------------------------------------------------#


def main(N):
    Nx = Ny = N
    # g = structured.CartGrid([Nx, Ny], [1, 1])
    g = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
    R = pp.cg.rot(np.pi / 2., [1, 0, 0])
    g.nodes = np.dot(R, g.nodes)
    g.compute_geometry()
    # co.coarsen(g, 'by_volume')

    # Assign parameters
    data = add_data(g)

    # Choose and define the solvers
    solver_flow = pp.MVEM("flow")
    A_flow, b_flow = solver_flow.assemble_matrix_rhs(g, data)

    solver_source = pp.DualSource("flow")
    A_source, b_source = solver_source.assemble_matrix_rhs(g, data)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)

    u = solver_flow.extract_flux(g, up)
    p = solver_flow.extract_pressure(g, up)
    P0u = solver_flow.project_flux(g, u, data)

    diam = np.amax(g.cell_diameters())
    return diam, error_p(g, p)


# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):
    def test_vem_varing_k_surface(self):
        diam_10, error_10 = main(10)
        diam_20, error_20 = main(20)

        known_order = 1.9890160655
        order = np.log(error_10 / error_20) / np.log(diam_10 / diam_20)
        assert np.isclose(order, known_order)


# ------------------------------------------------------------------------------#
