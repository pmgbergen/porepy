"""
This example contains the set up and computation of the blocking version of the
benchmark3 with the fv discretizations.
"""
import numpy as np
import scipy.sparse as sps

from porepy.fracs import importer
from porepy.params import bc, tensor
from porepy.numerics.fv import tpfa, mpfa
from porepy.numerics.elliptic import EllipticDataAssigner, EllipticModel
from porepy.numerics.mixed_dim import condensation as SC

# -----------------------------------------------------------------------------#


def assign_darcy_data(gb, domain, left_to_right):
    """
    Loop over grids to assign flow problem data.
    """
    gb.add_node_props(["problem"])
    for g, d in gb:
        d["problem"] = DarcyModelData(g, d, left_to_right)


def boundary_condition(g, left_to_right):
    if left_to_right:
        dirfaces_four = bc.face_on_side(g, "xmin")[0]
        dirfaces_one = bc.face_on_side(g, "xmax")[0]
    else:
        dirfaces_one = bc.face_on_side(g, "ymin")[0]
        dirfaces_four = bc.face_on_side(g, "ymax")[0]

    dirfaces = np.concatenate((dirfaces_four, dirfaces_one))
    labels = ["dir"] * dirfaces.size
    bc_val = np.zeros(g.num_faces)
    bc_val[dirfaces_four] = np.ones(dirfaces_four.size) * 4
    bc_val[dirfaces_one] = np.ones(dirfaces_one.size)
    return dirfaces, labels, bc_val


# -----------------------------------------------------------------------------#


class DarcyModelData(EllipticDataAssigner):
    def __init__(self, g, d, left_to_right):
        self.left_to_right = left_to_right
        self.data = d
        EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = np.power(1e-4, 2 - self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        if self.grid().dim == 2:
            k = 1
        elif np.in1d(self.data["node_number"], [4, 5]):
            k = 1e-4
        elif np.in1d(self.data["node_number"], [11, 13, 14, 15]):
            k = 2 / np.sum(1.0 / np.array([1e4, 1e-4]))
        else:
            k = 1e4
        return tensor.SecondOrderTensor(3, np.ones(self.grid().num_cells) * k)

    def bc(self):
        if self.grid().dim < 2:
            return bc.BoundaryCondition(self.grid())
        dirfaces, labels, _ = boundary_condition(self.grid(), self.left_to_right)
        return bc.BoundaryCondition(self.grid(), dirfaces, labels)

    def bc_val(self):
        if self.grid().dim < 2:
            return np.zeros(self.grid().num_faces)
        _, _, values = boundary_condition(self.grid(), self.left_to_right)
        return values


# -----------------------------------------------------------------------------#


def write_network(file_name):
    network = "FID,START_X,START_Y,END_X,END_Y\n"
    network += "0, 0.05, 0.4160, 0.22, 0.0624\n"
    network += "1, 0.05, 0.2750, 0.25, 0.1350\n"
    network += "2, 0.15, 0.6300, 0.45, 0.0900\n"
    network += "3, 0.15, 0.9167, 0.40, 0.5000\n"
    network += "4, 0.65, 0.8333, 0.849723, 0.167625\n"
    network += "5, 0.70, 0.2350, 0.849723, 0.167625\n"
    network += "6, 0.60, 0.3800, 0.85, 0.2675\n"
    network += "7, 0.35, 0.9714, 0.80, 0.7143\n"
    network += "8, 0.75, 0.9574, 0.95, 0.8155\n"
    network += "9, 0.15, 0.8363, 0.40, 0.9727\n"

    with open(file_name, "w") as text_file:
        text_file.write(network)


class DarcyModel(EllipticModel):
    def __init__(self, gb, mp=False, kw={}):
        self.multi_point = mp
        EllipticModel.__init__(self, gb, **kw)

    def flux_disc(self):
        if self.multi_point:
            return mpfa.MpfaMixedDim(physics=self.physics)
        else:
            return tpfa.TpfaMixedDim(physics=self.physics)


# -----------------------------------------------------------------------------#


def make_grid_bucket():
    """
    Define the geometry and produce the meshes
    """
    mesh_kwargs = {"tol": 1e-7}
    mesh_size = 0.05
    mesh_kwargs = {
        "mesh_size_frac": mesh_size,
        "mesh_size_min": mesh_size / 30,
        "tol": 1e-7,
    }

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

    file_name = "network_scotti.csv"
    write_network(file_name)
    gb = importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain)
    return gb, domain


def perform_condensation(full_problem, reduced_problem, dim):
    """
    Obtain reduced matrix and rhs.
    """
    A = full_problem.lhs
    rhs = full_problem.rhs
    to_be_eliminated = SC.dofs_of_dimension(full_problem.grid(), A, dim)
    a_reduced, rhs_reduced, _, _, _ = SC.eliminate_dofs(A, rhs, to_be_eliminated)

    reduced_problem.lhs = a_reduced
    reduced_problem.rhs = rhs_reduced


# -----------------------------------------------------------------------------#


def main(multi_point, if_export=False, left_to_right=True):

    gb, domain = make_grid_bucket()

    # Assign parameters
    assign_darcy_data(gb, domain, left_to_right)
    gb_el, _ = gb.duplicate_without_dimension(0)

    # Choose discretization and define the solver
    if multi_point:
        disc_type = "MPFA"
    else:
        disc_type = "TPFA"
    if left_to_right:
        direction = "left_to_right"
    else:
        direction = "top_to_bottom"
    FlowProblem = DarcyModel(gb, multi_point)

    # Discretize
    FlowProblem.reassemble()
    gb_el, _ = gb.duplicate_without_dimension(0)
    kw = {
        "file_name": disc_type + "_el",
        "folder_name": direction,
        "mesh_kw": {"binary": False},
    }
    FlowProblem_el = DarcyModel(gb_el, multi_point, kw)
    perform_condensation(FlowProblem, FlowProblem_el, 0)
    p_el = sps.linalg.spsolve(FlowProblem_el.lhs, FlowProblem_el.rhs)

    # Store the solution
    if if_export:
        FlowProblem_el.x = p_el
        FlowProblem_el.pressure()
        FlowProblem_el.permeability(["kxx"])
        FlowProblem_el.save(["pressure", "kxx"])


# -----------------------------------------------------------------------------#


def atest_fv_top_to_bottom():
    main(multi_point=False, if_export=False, left_to_right=False)
    main(multi_point=True, if_export=False, left_to_right=False)


# -----------------------------------------------------------------------------#


def atest_fv_left_to_right():
    main(multi_point=False, if_export=False, left_to_right=True)
    main(multi_point=True, if_export=False, left_to_right=True)


# -----------------------------------------------------------------------------#
