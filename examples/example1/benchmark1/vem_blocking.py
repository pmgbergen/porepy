import numpy as np
import scipy.sparse as sps

from porepy.viz import plot_grid, exporter
from porepy.fracs import importer

from porepy.params import bc, tensor

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
    gb.add_node_props(['perm', 'source', 'bc', 'bc_val', 'apertures'])
    kf = 1e-4
    tol = 1e-5
    a = 1e-4

    for g, d in gb:
        # Permeability
        kxx = np.ones(g.num_cells)
        if g.dim == 2:
            d['perm'] = tensor.SecondOrder(g.dim, kxx)
        else:
            d['perm'] = tensor.SecondOrder(g.dim, kf*kxx)

        # Source term
        d['source'] = np.zeros(g.num_cells)

        # Assign apertures
        d['apertures'] = np.ones(g.num_cells) * np.power(a, 2 - g.dim)

        # Boundaries
        bound_faces = g.get_boundary_faces()
        if bound_faces.size == 0:
            continue

        bound_face_centers = g.face_centers[:, bound_faces]

        top = bound_face_centers[1, :] > domain['ymax'] - tol
        bot = bound_face_centers[1, :] < domain['ymin'] + tol
        left = bound_face_centers[0, :] < domain['xmin'] + tol
        right = bound_face_centers[0, :] > domain['xmax'] - tol

        labels = np.array(['neu'] * bound_faces.size)
        labels[right] = 'dir'

        bc_val = np.zeros(g.num_faces)
        if g.dim == 2:
            bc_val[bound_faces[left]] = -np.ones(left.size)
        else:
            bc_val[bound_faces[left]] = -np.ones(left.size) * a

        bc_val[bound_faces[right]] = np.ones(right.size)

        d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
        d['bc_val'] = bc_val.ravel('F')

    # Assign coupling permeability
    gb.add_edge_prop('kn')
    for e, d in gb.edges_props():
        gn = gb.sorted_nodes_of_edge(e)
        d['kn'] = kf*np.ones(gn[0].num_cells)

#------------------------------------------------------------------------------#

mesh_kwargs = {}
mesh_kwargs['mesh_size'] = {'mode': 'constant',
                            'value': 0.045, 'bound_value': 0.045}
mesh_kwargs['gmsh_path'] = '~/gmsh/bin/gmsh'

domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
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

gb.add_node_props(["discharge", "p", "P0u"])
for g, d in gb:
    d["discharge"] = solver.extract_u(g, d["up"])
    d["p"] = solver.extract_p(g, d["up"])
    d["P0u"] = solver.project_u(g, d["discharge"])

exporter.export_vtk(gb, 'vem', ["p", "P0u"], folder='vem_blocking')

# Consistency check
assert np.isclose(np.sum(error.norm_L2(g, d['p']) for g, d in gb), 35.6444911616)
assert np.isclose(np.sum(error.norm_L2(g, d['P0u']) for g, d in gb), 1.03190700381)
