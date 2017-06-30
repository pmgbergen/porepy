import numpy as np
import scipy.sparse as sps

from porepy.viz import plot_grid, exporter
from porepy.fracs import importer

from porepy.params import bc, second_order_tensor

from porepy.grids.grid import FaceTag
from porepy.grids import coarsening

from porepy.numerics.mixed_dim import coupler
from porepy.numerics.vem import dual, dual_coupling

from porepy.utils.errors import error

#------------------------------------------------------------------------------#

def add_data(gb, domain):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(['k', 'f', 'bc', 'bc_val', 'a'])
    tol = 1e-3

    for g, d in gb:
        # Permeability
        kxx = np.ones(g.num_cells)
        if g.dim == 2:
            d['k'] = second_order_tensor.SecondOrderTensor(g.dim, 1e-14*kxx)
        else:
            d['k'] = second_order_tensor.SecondOrderTensor(g.dim, 1e-8*kxx)

        # Source term
        d['f'] = np.zeros(g.num_cells)

        # Assign apertures
        d['a'] = np.ones(g.num_cells) * np.power(1e-2, 2 - g.dim)

        # Boundaries
        bound_faces = g.get_boundary_faces()
        if bound_faces.size == 0:
            continue

        bound_face_centers = g.face_centers[:, bound_faces]

        left = bound_face_centers[0, :] < domain['xmin'] + tol
        right = bound_face_centers[0, :] > domain['xmax'] - tol

        labels = np.array(['neu'] * bound_faces.size)
        labels[np.logical_or(left, right)] = 'dir'

        bc_val = np.zeros(g.num_faces)
        bc_val[bound_faces[left]] = 1013250*np.ones(np.sum(left))

        d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
        d['bc_val'] = bc_val.ravel('F')

    # Assign coupling permeability
    gb.add_edge_prop('kn')
    for e, d in gb.edges_props():
        gn = gb.sorted_nodes_of_edge(e)
        d['kn'] = 1e-8*np.ones(gn[0].num_cells)

#------------------------------------------------------------------------------#

mesh_kwargs = {}
mesh_kwargs['mesh_size'] = {'mode': 'constant',
                            'value': 7, 'bound_value': 7}

domain = {'xmin': 0, 'xmax': 700, 'ymin': 0, 'ymax': 600}
gb = importer.from_csv('network.csv', mesh_kwargs, domain)
gb.compute_geometry()
gb.assign_node_ordering()

internal_flag = FaceTag.FRACTURE
[g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

# Assign parameters
add_data(gb, domain)

# Choose and define the solvers and coupler
solver = dual.DualVEM()
coupling_conditions = dual_coupling.DualCoupling(solver)
solver_coupler = coupler.Coupler(solver, coupling_conditions)
A, b = solver_coupler.matrix_rhs(gb)

up = sps.linalg.spsolve(A, b)
solver_coupler.split(gb, "up", up)

gb.add_node_props(["beta_n", "p", "P0u"])
for g, d in gb:
    d["beta_n"] = solver.extract_u(g, d["up"])
    d["p"] = solver.extract_p(g, d["up"])
    d["P0u"] = solver.project_u(g, d["beta_n"])

exporter.export_vtk(gb, 'vem', ["p", "P0u"], folder='vem')

print( np.sum(error.norm_L2(g, d['p']) for g, d in gb) )
print( np.sum(error.norm_L2(g, d['P0u']) for g, d in gb) )
# Consistency check
assert np.isclose(np.sum(error.norm_L2(g, d['p']) for g, d in gb), 1788853869.93)
assert np.isclose(np.sum(error.norm_L2(g, d['P0u']) for g, d in gb), 7.19640354325e-06)
