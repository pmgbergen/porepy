import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids.grid import FaceTag
from porepy.grids import coarsening as co

from porepy.numerics.vem import vem_dual, vem_source

#------------------------------------------------------------------------------#


def add_data(gb, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(['param'])
    tol = 1e-5
    a = 1e-4

    for g, d in gb:
        param = Parameters(g)

        # Permeability
        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        if g.dim == 2:
            perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=1, kzz=1)
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        # Boundaries
        bound_faces = g.get_domain_boundary_faces()
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain['xmin'] + tol
            right = bound_face_centers[0, :] > domain['xmax'] - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[right] = 'dir'

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = -aperture \
                * g.face_areas[bound_faces[left]]
            bc_val[bound_faces[right]] = 1

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
        aperture = np.power(a, gb.dim_max() - gn[0].dim)
        d['kn'] = np.ones(gn[0].num_cells) * kf / aperture

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


def main(kf, description, is_coarse=False, if_export=False):
    mesh_kwargs = {}
    mesh_kwargs['mesh_size'] = {'mode': 'constant',
                                'value': 0.045, 'bound_value': 0.045}

    domain = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}

    file_name = 'network_geiger.csv'
    write_network(file_name)
    gb = importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain)
    gb.compute_geometry()
    if is_coarse:
        co.coarsen(gb, 'by_volume')
    gb.assign_node_ordering()

    internal_flag = FaceTag.FRACTURE
    [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

    # Assign parameters
    add_data(gb, domain, kf)

    # Choose and define the solvers and coupler
    solver_flow = vem_dual.DualVEMMixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = vem_source.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
    solver_flow.split(gb, "up", up)

    gb.add_node_props(["discharge", 'pressure', "P0u"])
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.extract_p(gb, "up", 'pressure')
    solver_flow.project_u(gb, "discharge", "P0u")

    if if_export:
        save = Exporter(gb, "vem", folder="vem_" + description)
        save.write_vtk(['pressure', "P0u"])

#------------------------------------------------------------------------------#


def test_vem_blocking():
    kf = 1e-4
    main(kf, "blocking")

#------------------------------------------------------------------------------#


def test_vem_permeable():
    kf = 1e4
    main(kf, "permeable")

#------------------------------------------------------------------------------#
