import numpy as np
import scipy.sparse as sps

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids import coarsening as co

from porepy.numerics.vem import dual
from porepy.utils import comp_geom as cg

# ------------------------------------------------------------------------------#


def add_data(gb, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param"])
    tol = 1e-5
    a = 1e-4

    for g, d in gb:
        param = Parameters(g)

        # Permeability
        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        if g.dim == 2:
            perm = tensor.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = tensor.SecondOrderTensor(3, kxx=kxx, kyy=1, kzz=1)
            if g.dim == 1:
                R = cg.project_line_matrix(g.nodes, reference=[1, 0, 0])
                perm.rotate(R)

        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[right] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = -aperture * g.face_areas[bound_faces[left]]
            bc_val[bound_faces[right]] = 1

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        gn = gb.sorted_nodes_of_edge(e)
        aperture = np.power(a, gb.dim_max() - gn[0].dim)
        d["kn"] = np.ones(gn[0].num_cells) * kf / aperture


# ------------------------------------------------------------------------------#


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


# ------------------------------------------------------------------------------#


def main(kf, description, mesh_size):
    mesh_kwargs = {}
    mesh_kwargs["mesh_size"] = {
        "mode": "constant",
        "value": mesh_size,
        "bound_value": mesh_size,
    }

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    if_coarse = True

    folder = "example_5_1_1_" + description

    file_name = "network_geiger.csv"
    write_network(file_name)
    gb = importer.from_csv(file_name, mesh_kwargs, domain)
    gb.compute_geometry()

    g_fine = gb.get_grids(lambda g: g.dim == gb.dim_max())[0].copy()

    if if_coarse:
        partition = co.create_aggregations(gb)
        partition = co.reorder_partition(partition)
        co.generate_coarse_grid(gb, partition)

    gb.assign_node_ordering()

    # Assign parameters
    add_data(gb, domain, kf)

    # Choose and define the solvers and coupler
    solver = dual.DualVEMMixDim("flow")
    A, b = solver.matrix_rhs(gb)

    up = sps.linalg.spsolve(A, b)
    solver.split(gb, "up", up)

    gb.add_node_props(["discharge", "pressure", "P0u"])
    solver.extract_u(gb, "up", "discharge")
    solver.extract_p(gb, "up", "pressure")
    solver.project_u(gb, "discharge", "P0u")

    exporter.export_vtk(gb, "vem", ["pressure", "P0u"], folder=folder, binary=False)

    if if_coarse:
        partition = partition[gb.grids_of_dimension(gb.dim_max())[0]]
        p = np.array([d["pressure"] for g, d in gb if g.dim == gb.dim_max()]).ravel()
        data = {"partition": partition, "pressure": p[partition]}
        exporter.export_vtk(g_fine, "sub_grid", data, binary=False, folder=folder)

    print("diam", gb.diameter(lambda g: g.dim == gb.dim_max()))
    print("num_cells 2d", gb.num_cells(lambda g: g.dim == 2))
    print("num_cells 1d", gb.num_cells(lambda g: g.dim == 1))


# ------------------------------------------------------------------------------#


def vem_blocking():
    kf = 1e-4
    mesh_size = 0.035 / np.array([1, 2, 4])

    for i in np.arange(mesh_size.size):
        main(kf, "blocking_" + str(i), mesh_size[i])


# ------------------------------------------------------------------------------#


def vem_permeable():
    kf = 1e4
    mesh_size = 0.035 / np.array([1, 2, 4])

    for i in np.arange(mesh_size.size):
        main(kf, "permeable_" + str(i), mesh_size[i])


# ------------------------------------------------------------------------------#


vem_blocking()
vem_permeable()
