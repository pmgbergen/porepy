import numpy as np
import scipy.sparse as sps

from porepy.viz import plot_grid, exporter
from porepy.fracs import importer

from porepy.params import bc, second_order_tensor


from porepy.numerics.mixed_dim import coupler
from porepy.numerics.fv import tpfa, mpfa, tpfa_coupling

from porepy.utils.errors import error

#------------------------------------------------------------------------------#

def add_data(gb, domain):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(['k', 'f', 'bc', 'bc_val', 'a'])
    kf = 1e-4
    tol = 1e-5
    a = 1e-4

    for g, d in gb:
        # Permeability
        kxx = np.ones(g.num_cells)
        if g.dim == 2:
            d['k'] = second_order_tensor.SecondOrderTensor(g.dim, kxx)
        else:
            d['k'] = second_order_tensor.SecondOrderTensor(2, kf*kxx)

        # Source term
        d['f'] = np.zeros(g.num_cells)

        # Assign apertures
        d['a'] = np.ones(g.num_cells) * np.power(a, 2 - g.dim)

        # Boundaries
        bound_faces = g.get_boundary_faces()
        if bound_faces.size == 0:
            continue

        bound_face_centers = g.face_centers[:, bound_faces]

        
        left = bound_face_centers[0, :] < domain['xmin'] + tol
        right = bound_face_centers[0, :] > domain['xmax'] - tol

        labels = np.array(['neu'] * bound_faces.size)
        labels[right] = 'dir'

        bc_val = np.zeros(g.num_faces)
        if g.dim == 2:
            b_mesh_value = 0.045
            left_mid = np.array(np.absolute(g.face_centers[1, bound_faces[left]]
                                - 0.5) < b_mesh_value)
            bc_val[bound_faces[left]] = g.face_areas[bound_faces[left]] - left_mid * .5*a
        else:
            bc_val[bound_faces[left]] = g.face_areas[bound_faces[left]] * a

        bc_val[bound_faces[right]] = np.ones(np.sum(right))

        d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
        d['bc_val'] = bc_val.ravel('F')

    

#------------------------------------------------------------------------------#

mesh_kwargs = {}
mesh_kwargs['mesh_size'] = {'mode': 'constant',
                            'value': 0.045, 'bound_value': 0.045}

domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
gb = importer.from_csv('network.csv', mesh_kwargs, domain)
gb.compute_geometry()
gb.assign_node_ordering()

# Assign parameters
add_data(gb, domain)

# Choose and define the solvers and coupler
solver = tpfa.Tpfa()
#solver = mpfa.Mpfa()
coupling_conditions = tpfa_coupling.TpfaCoupling(solver)
solver_coupler = coupler.Coupler(solver, coupling_conditions)
A, b = solver_coupler.matrix_rhs(gb)

p = sps.linalg.spsolve(A, b)

gb.add_node_props(["p"])
solver_coupler.split(gb, "p", p)
exporter.export_vtk(gb, 'fv', ["p"], folder='fv_blocking')

# Consistency check
assert np.isclose(np.sum(error.norm_L2(g, d['p']) for g, d in gb), 36.1027322839)

