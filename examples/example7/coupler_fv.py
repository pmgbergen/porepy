"""
Example illustrating the FV coupling and the condensation elimination of the
intersection cells. The coupling is with TPFA, but for internal discretization of
the individual domains, one can choose between MPFA and TPFA.
"""
import numpy as np
import scipy.sparse as sps

from porepy.params import bc, tensor, data


# from porepy.viz.exporter import export_vtk
from porepy.fracs import meshing  # split_grid

from porepy.numerics.fv import mpfa, tpfa
from porepy.numerics.mixed_dim import condensation

# ------------------------------------------------------------------------------#


def add_data(gb):
    """
    Define the permeability, apertures, boundary conditions, source term
    """
    gb.add_node_props(["param"])
    for g, d in gb:
        # initialize Parameter class
        params = data.Parameters(g)
        # Permeability
        kxx = np.ones(g.num_cells)
        if all(g.cell_centers[0, :] < 0.0001):
            perm = tensor.SecondOrderTensor(3, kxx / 100, kxx, kxx)
        else:
            perm = tensor.SecondOrderTensor(3, kxx * np.power(100, g.dim < 3))

        params.set_tensor("flow", perm)

        # Source term
        params.set_source("flow", np.zeros(g.num_cells))

        # Boundaries
        top = np.argwhere(g.face_centers[1, :] > 1 - 1e-5)
        bot = np.argwhere(g.face_centers[1, :] < -1 + 1e-5)
        left = np.argwhere(g.face_centers[0, :] < -1 + 1e-5)

        dir_bound = np.concatenate((top, bot))

        bound = bc.BoundaryCondition(g, dir_bound.ravel(), ["dir"] * dir_bound.size)
        d_bound = np.zeros(g.num_faces)
        d_bound[dir_bound] = g.face_centers[1, dir_bound]

        d_bound[left] = -1 * g.face_centers[0, left]
        params.set_bc("flow", bound)
        params.set_bc_val("flow", d_bound.ravel("F"))

        # Assign apertures
        params.apertures = np.ones(g.num_cells) * np.power(1e-2, 3 - g.dim)
        d["param"] = params


# ------------------------------------------------------------------------------#


if __name__ == "__main__":

    np.set_printoptions(linewidth=999999, threshold=np.nan)
    np.set_printoptions(precision=3)

    # Set up and mesh the domain and collect meshes in a grid bucket
    f_1 = np.array([[-.8, .8, .8, -.8], [0, 0, 0, 0], [-.8, -.8, .8, .8]])
    f_2 = np.array([[0, 0, 0, 0], [-.8, .8, .8, -.8], [-.8, -.8, .8, .8]])
    f_3 = np.array([[-.8, .8, .8, -.8], [-.8, -.8, .8, .8], [0, 0, 0, 0]])
    f_1 = [[0, 1], [0, 1]]
    f_2 = [[0, .1], [0.2, .1]]
    f_set = [f_1, f_2, f_3]
    f_set = [f_1, f_2]
    domain = {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 1}  # , 'zmin': -1, 'zmax': 1}

    gb = meshing.simplex_grid(f_set, domain)
    gb.assign_node_ordering()

    # Assign parameters
    add_data(gb)

    # Choose and define the solvers and coupler
    solver = mpfa.MpfaMixedDim()
    A, b = solver.matrix_rhs(gb)
    p = sps.linalg.spsolve(A.copy(), b)

    # Solve the problem without 0d grids.
    eliminate_dim = 0
    p_full_condensation, p_reduced, _, _ = condensation.solve_static_condensation(
        A, b, gb, eliminate_dim
    )

    # The p_reduced only has pressures for the cells of grids of dim>0, so
    # should be plotted on a grid where the 0d has been removed:
    gb_r, _ = gb.duplicate_without_dimension(0)

    # Add the solutions to data fields in the grid buckets
    gb.add_node_props(["pressure", "p_condensation"])
    gb_r.add_node_props(["p_reduced"])

    solver.split(gb, "p_condensation", p_full_condensation)
    solver.split(gb_r, "p_reduced", p_reduced)
    solver.split(gb, "pressure", p)

    max_p, min_p, normalization, error_norm = np.zeros(1), np.zeros(1), 0, 0
    for g, d in gb:
        p1, p2 = d["pressure"], d["p_condensation"]
        error_norm += sum(np.power(p1 - p2, 2) * g.cell_volumes * d["param"].apertures)
        normalization += sum(g.cell_volumes)
        max_p = np.array(np.amax(np.concatenate((max_p, p1)), axis=0), ndmin=1)
        min_p = np.array(np.amin(np.concatenate((min_p, p1)), axis=0), ndmin=1)

    error_norm = np.power(error_norm / normalization, .5) / (max_p - min_p)

    print("The error of the condensation compared to the full method is ", error_norm)
    assert error_norm < 1e-3
    #
    #    export_vtk(gb, "grid_mpfa", ["p", "p_condensation"], folder="simu")
    #    export_vtk(gb_r, "grid_mpfa_r", ["p_reduced"], folder="simu")

    export_vtk(gb, "grid_mpfa", ["pressure", "p_condensation"], folder="simu")
    export_vtk(gb_r, "grid_mpfa_r", ["p_reduced"], folder="simu")
