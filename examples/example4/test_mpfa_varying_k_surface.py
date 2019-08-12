"""
2d convergence test for MPFA for heterogeneous permeability.
"""
import numpy as np
import scipy.sparse as sps
import unittest
import porepy as pp


# ------------------------------------------------------------------------------#


def rhs(x, y, z):
    return (
        7.0 * z * (x ** 2 + y ** 2 + 1.0)
        - y * (x ** 2 - 9.0 * z ** 2)
        - 4.0 * x ** 2 * z
        - (
            8 * np.sin(np.pi * y)
            - 4.0 * np.pi ** 2 * y ** 2 * np.sin(np.pi * y)
            + 16.0 * np.pi * y * np.cos(np.pi * y)
        )
        * (x ** 2 / 2.0 + y ** 2 / 2.0 + 1.0 / 2.0)
        - 4.0 * y ** 2 * (2.0 * np.sin(np.pi * y) + np.pi * y * np.cos(np.pi * y))
    )


# ------------------------------------------------------------------------------#


def solution(x, y, z):
    return x ** 2 * z + 4.0 * y ** 2 * np.sin(np.pi * y) - 3.0 * z ** 3


# ------------------------------------------------------------------------------#


def permeability(x, y, z):
    return 1.0 + x ** 2 + y ** 2


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

    # g = structured.CartGrid([Nx, Ny], [1, 1])
    g = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
    R = pp.map_geometry.rotation_matrix(np.pi / 4.0, [1, 0, 0])
    g.nodes = np.dot(R, g.nodes)
    g.compute_geometry()

    # Assign parameters
    data = add_data(g)

    # Choose and define the solvers
    solver = pp.Mpfa("flow")
    A, b_flux = solver.assemble_matrix_rhs(g, data)
    _, b_source = pp.ScalarSource("flow").assemble_matrix_rhs(g, data)
    p = sps.linalg.spsolve(A, b_flux + b_source)

    diam = np.amax(g.cell_diameters())
    return diam, error_p(g, p)


# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):
    def test_mpfa_varying_k_surface(self):
        diam_10, error_10 = main(10)
        diam_20, error_20 = main(20)

        known_order = 1.9956052512
        order = np.log(error_10 / error_20) / np.log(diam_10 / diam_20)
        assert np.isclose(order, known_order)


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
