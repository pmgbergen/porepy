import numpy as np
import scipy.sparse as sps
import unittest

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids import structured, simplex
from porepy.grids import coarsening as co

from porepy.numerics.vem import vem_dual, vem_source

# ------------------------------------------------------------------------------#


def rhs(x, y, z):
    return (
        8.
        * np.pi ** 2
        * np.sin(2. * np.pi * x)
        * np.sin(2. * np.pi * y)
        * permeability(x, y, z)
        - 400. * np.pi * y * np.cos(2. * np.pi * y) * np.sin(2. * np.pi * x)
        - 400. * np.pi * x * np.cos(2. * np.pi * x) * np.sin(2. * np.pi * y)
    )


# ------------------------------------------------------------------------------#


def solution(x, y, z):
    return np.sin(2. * np.pi * x) * np.sin(2. * np.pi * y)


# ------------------------------------------------------------------------------#


def permeability(x, y, z):
    return 1 + 100. * x ** 2 + 100. * y ** 2


# ------------------------------------------------------------------------------#


def add_data(g):
    """
    Define the permeability, apertures, boundary conditions
    """
    param = Parameters(g)

    # Permeability
    kxx = np.array([permeability(*pt) for pt in g.cell_centers.T])
    param.set_tensor("flow", tensor.SecondOrderTensor(3, kxx))

    # Source term
    source = np.array([rhs(*pt) for pt in g.cell_centers.T])
    param.set_source("flow", g.cell_volumes * source)

    # Boundaries
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bound_face_centers = g.face_centers[:, bound_faces]

    labels = np.array(["dir"] * bound_faces.size)

    bc_val = np.zeros(g.num_faces)
    bc_val[bound_faces] = np.array([solution(*pt) for pt in bound_face_centers.T])

    param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
    param.set_bc_val("flow", bc_val)

    return {"param": param}


# ------------------------------------------------------------------------------#


def error_p(g, p):

    sol = np.array([solution(*pt) for pt in g.cell_centers.T])
    return np.sqrt(np.sum(np.power(np.abs(p - sol), 2) * g.cell_volumes))


# ------------------------------------------------------------------------------#


def main(N):
    Nx = Ny = N

    # g = structured.CartGrid([Nx, Ny], [2, 2])
    g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
    g.compute_geometry()
    # co.coarsen(g, 'by_volume')

    # Assign parameters
    data = add_data(g)

    # Choose and define the solvers
    solver_flow = vem_dual.DualVEM("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(g, data)

    solver_source = vem_source.DualSource("flow")
    A_source, b_source = solver_source.matrix_rhs(g, data)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)

    u = solver_flow.extract_u(g, up)
    p = solver_flow.extract_p(g, up)
    P0u = solver_flow.project_u(g, u, data)

    diam = np.amax(g.cell_diameters())
    return diam, error_p(g, p)


# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):
    def test_vem_varing_k(self):
        diam_10, error_10 = main(10)
        diam_20, error_20 = main(20)

        known_order = 2.00266229752
        order = np.log(error_10 / error_20) / np.log(diam_10 / diam_20)
        assert np.isclose(order, known_order)


# ------------------------------------------------------------------------------#
