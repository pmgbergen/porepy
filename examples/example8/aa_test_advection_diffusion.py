import numpy as np
import scipy.sparse as sps

from porepy.viz import plot_grid, exporter

from porepy.params import bc, second_order_tensor

from porepy.grids import structured

from porepy.numerics.fv.transport import upwind
from porepy.numerics.fv import tpfa

from porepy.utils.errors import error

# ------------------------------------------------------------------------------#


def add_data(g, advection):
    """
    Define the permeability, apertures, boundary conditions
    """

    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bnd = bc.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
    bnd_val = np.zeros(g.num_faces)

    beta_n = advection.beta_n(g, [1, 0, 0])

    kxx = 1e-2 * np.ones(g.num_cells)
    diffusion = second_order_tensor.SecondOrderTensorTensor(g.dim, kxx)

    f = np.ones(g.num_cells) * g.cell_volumes

    data = {"beta_n": beta_n, "bc": bnd, "bc_val": bnd_val, "k": diffusion, "f": f}

    return data


# ------------------------------------------------------------------------------#

# the f is considered twice, we guess that non-homogeneous Neumann as well.


Nx = Ny = 20
g = structured.CartGrid([Nx, Ny], [1, 1])
g.compute_geometry()

advection = upwind.Upwind()
diffusion = tpfa.Tpfa()

# Assign parameters
data = add_data(g, advection)

U, rhs_u = advection.matrix_rhs(g, data)
D, rhs_d = diffusion.matrix_rhs(g, data)

theta = sps.linalg.spsolve(D + U, rhs_u + rhs_d)

exporter.export_vtk(g, "advection_diffusion", {"theta": theta})
