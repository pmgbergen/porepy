import numpy as np
import scipy.sparse as sps

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids import coarsening as co

from porepy.numerics.vem import dual

# ------------------------------------------------------------------------------#


def source_f1(x, y, z):

    if z >= 0:
        val = 8 * y * (1 - y) - 8 * (z - 1) * (z - 1)
    else:
        val = 8 * y * (1 - y) - 8 * (z + 1) * (z + 1)
    return -val


# ------------------------------------------------------------------------------#


def source_f2(x, y, z):

    if x >= 0:
        val = 8 * y * (1 - y) - 8 * (x + 1) * (x + 1)
    else:
        val = 8 * y * (1 - y) - 8 * (x - 1) * (x - 1)
    return -val


# ------------------------------------------------------------------------------#


def sol_f1(x, y, z):

    if z >= 0:
        val = 4 * y * (1 - y) * (z - 1) * (z - 1)
    else:
        val = 4 * y * (1 - y) * (z + 1) * (z + 1)
    return val


# ------------------------------------------------------------------------------#


def sol_f2(x, y, z):

    if x >= 0:
        val = 4 * y * (1 - y) * (x + 1) * (x + 1)
    else:
        val = 4 * y * (1 - y) * (x - 1) * (x - 1)
    return val


# ------------------------------------------------------------------------------#


def perm(x, y, z):
    return 1


# ------------------------------------------------------------------------------#


def add_data(gb, tol):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param"])

    source_f = [source_f1, source_f2]
    sol_f = [sol_f1, sol_f2]

    for g, d in gb:
        if g.dim < 2:
            continue

        param = Parameters(g)

        # Permeability
        kxx = np.array([perm(*pt) for pt in g.cell_centers.T])
        param.set_tensor("flow", tensor.SecondOrderTensor(g.dim, kxx))

        # Source term
        frac_id = d["node_number"]
        source = np.array([source_f[frac_id](*pt) for pt in g.cell_centers.T])
        param.set_source("flow", g.cell_volumes * source)

        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            labels = np.array(["dir"] * bound_faces.size)

            bc_val = np.zeros(g.num_faces)
            bc = [sol_f[frac_id](*pt) for pt in bound_face_centers.T]
            bc_val[bound_faces] = np.array(bc)

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param


# ------------------------------------------------------------------------------#


def error_p(gb):

    error = 0
    sol_f = [sol_f1, sol_f2]
    for g, d in gb:
        if g.dim < 2:
            d["err"] = np.zeros(g.num_cells)
            continue
        frac_id = d["node_number"]
        sol = np.array([sol_f[frac_id](*pt) for pt in g.cell_centers.T])
        d["err"] = d["pressure"] - sol
        error += np.sum(np.power(d["err"], 2) * g.cell_volumes)

    return np.sqrt(error)


# ------------------------------------------------------------------------------#


tol = 1e-5

mesh_kwargs = {}
mesh_kwargs["mesh_size"] = {"mode": "constant", "value": 0.25, "bound_value": 1}

file_name = "dfn_square.fab"
file_intersections = "traces_square.dat"
gb = importer.dfn_3d_from_fab(file_name, file_intersections, tol=1e-5, **mesh_kwargs)
gb.remove_nodes(lambda g: g.dim == 0)
gb.compute_geometry()
# co.coarsen(gb, 'by_volume')
gb.assign_node_ordering()

exporter.export_vtk(gb, "grid", folder="vem")

# Assign parameters
add_data(gb, tol)

# Choose and define the solvers and coupler
solver = dual.DualVEMDFN(gb.dim_max(), "flow")
A, b = solver.matrix_rhs(gb)

up = sps.linalg.spsolve(A, b)
solver.split(gb, "up", up)

gb.add_node_props(["discharge", "pressure", "P0u", "err"])
solver.extract_u(gb, "up", "discharge")
solver.extract_p(gb, "up", "pressure")
solver.project_u(gb, "discharge", "P0u")

diam = gb.diameter(lambda g: g.dim == gb.dim_max())
print("h=", diam, "- err(p)=", error_p(gb))

exporter.export_vtk(gb, "vem", ["pressure", "err", "P0u"], folder="vem")

# ------------------------------------------------------------------------------#
