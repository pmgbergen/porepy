import numpy as np
import scipy.sparse as sps

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids import structured, simplex
from porepy.grids import coarsening as co

import porepy.utils.comp_geom as cg

from porepy.numerics.vem import dual

# ------------------------------------------------------------------------------#


def rhs(x, y, z):
    return (
        7 * z
        - 4 * np.sin(np.pi * y)
        + 2 * np.pi ** 2 * y ** 2 * np.sin(np.pi * y)
        - 8 * np.pi * y * np.cos(np.pi * y)
    )


# ------------------------------------------------------------------------------#


def solution(x, y, z):
    return x ** 2 * z + 4 * y ** 2 * np.sin(np.pi * y) - 3 * z ** 3


# ------------------------------------------------------------------------------#


def add_data(g):
    """
    Define the permeability, apertures, boundary conditions
    """
    param = Parameters(g)

    # Permeability
    param.set_tensor("flow", tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells)))

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


Nx = Ny = 20
# g = structured.CartGrid([Nx, Ny], [1, 1])
g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
R = cg.rot(np.pi / 4., [1, 0, 0])
g.nodes = np.dot(R, g.nodes)
g.compute_geometry()
# co.coarsen(g, 'by_volume')

# Assign parameters
data = add_data(g)

# Choose and define the solvers
solver = dual.DualVEM("flow")
A, b = solver.matrix_rhs(g, data)
up = sps.linalg.spsolve(A, b)

u = solver.extract_u(g, up)
p = solver.extract_p(g, up)
P0u = solver.project_u(g, u, data)

diam = np.amax(g.cell_diameters())
print("h=", diam, "- err(p)=", error_p(g, p))

exporter.export_vtk(g, "vem", {"p": p, "P0u": P0u}, folder="vem")

# ------------------------------------------------------------------------------#
