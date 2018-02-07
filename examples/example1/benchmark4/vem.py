import numpy as np
import scipy.sparse as sps

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids import coarsening as co

from porepy.numerics.vem import vem_dual, vem_source

#------------------------------------------------------------------------------#


def add_data(gb, domain):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(['param'])
    tol = 1e-3
    a = 1e-2

    for g, d in gb:
        param = Parameters(g)

        # Permeability
        if g.dim == 2:
            perm = tensor.SecondOrder(g.dim, 1e-14 * np.ones(g.num_cells))
        else:
            perm = tensor.SecondOrder(g.dim, 1e-8 * np.ones(g.num_cells))
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        # Boundaries
        bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain['xmin'] + tol
            right = bound_face_centers[0, :] > domain['xmax'] - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = 'dir'

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = 1013250

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(
                g, np.empty(0), np.empty(0)))

        d['param'] = param

    # Assign coupling permeability
    gb.add_edge_prop('kn')
    for e, d in gb.edges_props():
        gn = gb.sorted_nodes_of_edge(e)
        d['kn'] = 1e-8 * np.ones(gn[0].num_cells)

#------------------------------------------------------------------------------#


mesh_kwargs = {}
mesh_size = 10
mesh_kwargs = {'mesh_mode': 'constant',
               'h_ideal': mesh_size, 'h_min': 1 / 3 * mesh_size}


domain = {'xmin': 0, 'xmax': 700, 'ymin': 0, 'ymax': 600}
gb = importer.from_csv('network.csv', mesh_kwargs, domain)
gb.compute_geometry()
co.coarsen(gb, 'by_volume')
gb.assign_node_ordering()

exporter.export_vtk(gb, 'grid', folder='vem')

# Assign parameters
add_data(gb, domain)

# Choose and define the solvers and coupler
solver_flow = vem_dual.DualVEMMixDim('flow')
A_flow, b_flow = solver_flow.matrix_rhs(gb)

solver_source = vem_source.IntegralMixDim('flow')
A_source, b_source = solver_source.matrix_rhs(gb)

up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
solver.split(gb, "up", up)

gb.add_node_props(["discharge", 'pressure', "P0u"])
solver.extract_u(gb, "up", "discharge")
solver.extract_p(gb, "up", 'pressure')
solver.project_u(gb, "discharge", "P0u")

exporter.export_vtk(gb, 'vem', ['pressure', "P0u"], folder='vem')
