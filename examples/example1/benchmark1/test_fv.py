"""
This example contains the set up and computation of the blocking version of the
benchmark1 with the fv discretizations. Note the choice between tpfa and mpfa in
line 79.
"""
import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter
from porepy.fracs import importer
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.numerics.mixed_dim import coupler
from porepy.numerics.fv import tpfa, mpfa
from porepy.utils.errors import error

#------------------------------------------------------------------------------#

def add_data(gb, domain, kf, mesh_value):
    """
    Define the permeability, apertures, boundary conditions and sources
    """
    gb.add_node_props(['param'])
    tol = 1e-5
    a = 1e-4

    for g, d in gb:
        param = Parameters(g)

        # Assign apertures
        a_dim = np.power(a, gb.dim_max() - g.dim)
        aperture = np.ones(g.num_cells)*a_dim
        param.set_aperture(aperture)

        # Permeability
        # Use fracture value in the fractures, i.e., the lower dimensional grids
        k_frac = np.power(kf, g.dim<gb.dim_max())
        p = tensor.SecondOrder(3, np.ones(g.num_cells) * k_frac)
        param.set_tensor('flow', p)
        param.set_tensor('flow', p)

        # Source term
        param.set_source('flow', np.zeros(g.num_cells))

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
            # Account for the double inflow on the matrix-fracture overlap
            left_mid = np.array(np.absolute(g.face_centers[1, bound_faces[left]]
                                            - 0.5) < mesh_value)
            bc_val[bound_faces[left]] = -g.face_areas[bound_faces[left]] \
                                        + left_mid * .5*a
        else:
            bc_val[bound_faces[left]] = -g.face_areas[bound_faces[left]] * a

        bc_val[bound_faces[right]] = np.ones(np.sum(right))

        param.set_bc('flow', bc.BoundaryCondition(g, bound_faces, labels))
        param.set_bc_val('flow', bc_val)

        d['param'] = param

#------------------------------------------------------------------------------#

def write_network(file_name):
    network = "FID,START_X,START_Y,END_X,END_Y\n"
    network += "0,0,0.5,1,0.5\n"
    network += "1,0.5,0,0.5,1\n"
    network += "2,0.5,0.75,1,0.75\n"
    network += "3,0.75,0.5,0.75,1\n"
    network += "4,0.5,0.625,0.75,0.625\n"
    network += "5,0.625,0.5,0.625,0.75\n"
    with open(file_name, "w") as text_file:
        text_file.write(network)

#------------------------------------------------------------------------------#

def main(kf, description, multi_point, if_export=False):

    # Define the geometry and produce the meshes
    mesh_kwargs = {}
    mesh_size = 0.045
    mesh_kwargs['mesh_size'] = {'mode': 'constant',
                                'value': mesh_size, 'bound_value': mesh_size}
    domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

    file_name = 'network_geiger.csv'
    write_network(file_name)
    gb = importer.mesh_from_csv(file_name, mesh_kwargs, domain)
    gb.compute_geometry()
    gb.assign_node_ordering()

    # Assign parameters
    add_data(gb, domain, kf, mesh_size)

    # Choose discretization and define the solver
    if multi_point:
        solver = mpfa.MpfaMixDim('flow')
    else:
        solver = tpfa.TpfaMixDim('flow')

    # Discretize
    A, b = solver.matrix_rhs(gb)

    # Solve the linear system
    p = sps.linalg.spsolve(A, b)

    # Store the solution
    gb.add_node_props(["p"])
    solver.split(gb, "p", p)

    if if_export:
        save = Exporter(gb, "fv", folder="fv_"+description)
        save.write_vtk(["p"])

#------------------------------------------------------------------------------#

def test_fv_blocking():
    kf = 1e-4
    main(kf, "blocking", multi_point=False)

#------------------------------------------------------------------------------#

def test_fv_permeable():
    kf = 1e4
    main(kf, "permeable", multi_point=False)

#------------------------------------------------------------------------------#

def test_mpfa_blocking():
    kf = 1e-4
    main(kf, "blocking", multi_point=True)

#------------------------------------------------------------------------------#

def test_mpfa_permeable():
    kf = 1e4
    main(kf, "permeable", multi_point=True)

#------------------------------------------------------------------------------#
