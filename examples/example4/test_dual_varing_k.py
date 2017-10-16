import numpy as np
import scipy.sparse as sps

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids.grid import FaceTag
from porepy.grids import structured, simplex
from porepy.grids import coarsening as co

from porepy.numerics.vem import dual

#------------------------------------------------------------------------------#

def rhs(x, y, z):
    return 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*(x**2 + y**2 + 1) -\
    4*np.pi*y*np.cos(2*np.pi*y)*np.sin(2*np.pi*x) -\
    4*np.pi*x*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)

#------------------------------------------------------------------------------#

def solution(x, y, z):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

#------------------------------------------------------------------------------#

def permeability(x, y, z):
    return 1+x**2+y**2

#------------------------------------------------------------------------------#

def add_data(g, tol):
    """
    Define the permeability, apertures, boundary conditions
    """
    param = Parameters(g)

    # Permeability
    kxx = np.array([permeability(*pt) for pt in g.cell_centers.T])
    param.set_tensor("flow", tensor.SecondOrder(g.dim, kxx))

    # Source term
    source = np.array([rhs(*pt) for pt in g.cell_centers.T])
    param.set_source("flow", g.cell_volumes*source)

    # Boundaries
    bound_faces = g.get_boundary_faces()
    bound_face_centers = g.face_centers[:, bound_faces]

    labels = np.array(['dir'] * bound_faces.size)

    bc_val = np.zeros(g.num_faces)
    bc_val[bound_faces] = np.array([solution(*pt) for pt in bound_face_centers.T])

    param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
    param.set_bc_val("flow", bc_val)

    return {'param': param}

#------------------------------------------------------------------------------#

def error_p(g, p):

    sol = np.array([solution(*pt) for pt in g.cell_centers.T])
    return np.sqrt(np.sum(np.power(np.abs(p - sol), 2)*g.cell_volumes))

#------------------------------------------------------------------------------#

tol = 1e-5

Nx = Ny = 320
#g = structured.CartGrid([Nx, Ny], [2, 2])
g = simplex.StructuredTriangleGrid([Nx, Ny], [2, 2])
g.compute_geometry()
#co.coarsen(g, 'by_volume')

# Assign parameters
data = add_data(g, tol)

# Choose and define the solvers
solver = dual.DualVEM('flow')
A, b = solver.matrix_rhs(g, data)
up = sps.linalg.spsolve(A, b)

u = solver.extract_u(g, up)
p = solver.extract_p(g, up)
P0u = solver.project_u(g, u, data)

diam = np.amax(g.cell_diameters())
print("h=", diam, "- err(p)=", error_p(g, p))

exporter.export_vtk(g, 'vem', {"p": p, "P0u": P0u}, folder='vem')

#------------------------------------------------------------------------------#
