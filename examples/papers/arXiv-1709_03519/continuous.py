import numpy as np
import scipy.sparse as sps

from porepy.viz import exporter
from porepy.fracs import meshing

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids import coarsening as co

from porepy.numerics.vem import dual

#------------------------------------------------------------------------------#

def define_data(g):
    domain = {'xmin': np.amin(g.nodes[0, :]),
              'xmax': np.amax(g.nodes[0, :]),
              'ymin': np.amin(g.nodes[1, :]),
              'ymax': np.amax(g.nodes[1, :])}
    tol = 1e-5

    param = Parameters(g)

    # Permeability
    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrder(g.dim, kxx)
    param.set_tensor("flow", perm)

    # Boundaries
    b_faces = g.get_boundary_faces()
    b_face_centers = g.face_centers[:, b_faces]

    top = b_face_centers[1, :] > domain['ymax'] - tol
    bot = b_face_centers[1, :] < domain['ymin'] + tol
    left = b_face_centers[0, :] < domain['xmin'] + tol
    right = b_face_centers[0, :] > domain['xmax'] - tol

    outer = np.logical_or.reduce((top, bot, left, right))
    inner = np.logical_not(outer)

    labels = np.array(['dir'] * b_faces.size)

    bc_val = np.zeros(g.num_faces)
    bc_val[b_faces[inner]] = 1

    param.set_bc("flow", BoundaryCondition(g, b_faces, labels))
    param.set_bc_val("flow", bc_val)

    return {'param': param}

#------------------------------------------------------------------------------#

def glue_tips(partition, gb, g):
    seeds = co.generate_seeds(gb)

    left = seeds[g.cell_centers[0, seeds] < 0.5]
    right = seeds[g.cell_centers[0, seeds] > 0.5]

    partition[partition == partition[left[1]]] = partition[left[0]]
    partition[partition == partition[right[1]]] = partition[right[0]]

#------------------------------------------------------------------------------#

folder = './'
file_name = 'geometry.geo'
if_coarse = True

gb = meshing.from_gmsh(folder + file_name, dim=2)
g = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]
g_fine = g.copy()

if if_coarse:
    partition = co.create_aggregations(g, weight=1)
    glue_tips(partition, gb, g)
    partition = co.reorder_partition(partition)
    co.generate_coarse_grid(g, partition)

# Choose and define the solvers and coupler
solver = dual.DualVEM("flow")
data = define_data(g)

A, b = solver.matrix_rhs(g, data)
up = sps.linalg.spsolve(A, b)

u = solver.extract_u(g, up)
p = solver.extract_p(g, up)
P0u = solver.project_u(g, u, data)

diams = g.cell_diameters()
print( "diam", np.amin(diams), np.average(diams), np.amax(diams) )

folder = 'continuous'
data = {"p": p, "P0u": P0u}
exporter.export_vtk(g, 'vem', data, binary=False, folder=folder)

if if_coarse:
    data = {'partition': partition, 'p': p[partition]}
    exporter.export_vtk(g_fine, 'sub_grid', data, binary=False, folder=folder)

#------------------------------------------------------------------------------#
