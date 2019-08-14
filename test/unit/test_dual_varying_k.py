"""
3d convergence test for dual VEM for heterogeneous permeability.
"""
import numpy as np
import scipy.sparse as sps
import unittest
import porepy as pp


# ------------------------------------------------------------------------------#


def rhs(x, y, z):
    return (
        8.0
        * np.pi ** 2
        * np.sin(2.0 * np.pi * x)
        * np.sin(2.0 * np.pi * y)
        * permeability(x, y, z)
        - 400.0 * np.pi * y * np.cos(2.0 * np.pi * y) * np.sin(2.0 * np.pi * x)
        - 400.0 * np.pi * x * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    )


# ------------------------------------------------------------------------------#


def solution(x, y, z):
    return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


# ------------------------------------------------------------------------------#


def permeability(x, y, z):
    return 1 + 100.0 * x ** 2 + 100.0 * y ** 2


# ------------------------------------------------------------------------------#


def add_data(g):
    """
    Define the permeability, apertures, boundary conditions
    """
    # Permeability
    kxx = np.array([permeability(*pt) for pt in g.cell_centers.T])
    perm = pp.SecondOrderTensor(kxx)

    # Source term
    source = g.cell_volumes * np.array([rhs(*pt) for pt in g.cell_centers.T])

    # Boundaries
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bound_face_centers = g.face_centers[:, bound_faces]

    labels = np.array(["dir"] * bound_faces.size)

    bc_val = np.zeros(g.num_faces)
    bc_val[bound_faces] = np.array([solution(*pt) for pt in bound_face_centers.T])

    bound = pp.BoundaryCondition(g, bound_faces, labels)
    specified_parameters = {
        "second_order_tensor": perm,
        "source": source,
        "bc": bound,
        "bc_values": bc_val,
    }
    return pp.initialize_default_data(g, {}, "flow", specified_parameters)


# ------------------------------------------------------------------------------#


def error_p(g, p):

    sol = np.array([solution(*pt) for pt in g.cell_centers.T])
    return np.sqrt(np.sum(np.power(np.abs(p - sol), 2) * g.cell_volumes))


# ------------------------------------------------------------------------------#


def main(N):
    Nx = Ny = N

    # g = structured.CartGrid([Nx, Ny], [2, 2])
    g = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
    g.compute_geometry()
    # co.coarsen(g, 'by_volume')

    # Assign parameters
    data = add_data(g)

    # Choose and define the solvers
    solver_flow = pp.MVEM("flow")
    solver_flow.discretize(g, data)
    A_flow, b_flow = solver_flow.assemble_matrix_rhs(g, data)

    solver_source = pp.DualScalarSource("flow")
    A_source, b_source = solver_source.assemble_matrix_rhs(g, data)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)

    u = solver_flow.extract_flux(g, up, data)
    p = solver_flow.extract_pressure(g, up, data)
    #    P0u = solver_flow.project_flux(g, u, data, keyword="flow")

    diam = np.amax(g.cell_diameters())
    return diam, error_p(g, p)


# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):
    def test_vem_varying_k(self):
        diam_10, error_10 = main(10)
        diam_20, error_20 = main(20)

        known_order = 2.00266229752
        order = np.log(error_10 / error_20) / np.log(diam_10 / diam_20)
        assert np.isclose(order, known_order)


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
