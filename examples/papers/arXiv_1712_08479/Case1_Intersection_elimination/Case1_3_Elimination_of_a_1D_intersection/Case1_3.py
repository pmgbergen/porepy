"""
Case 1.3. Flow and transport with and without Schur complement elimination
on a case consisting of two intersecting 2D fractures.
"""
import numpy as np
import scipy.sparse as sps

from porepy.params import bc, tensor
from porepy.numerics.elliptic import EllipticModel, EllipticDataAssigner
from porepy.numerics.mixed_dim import condensation as SC
from porepy.fracs import meshing
from porepy.numerics.parabolic import ParabolicDataAssigner, ParabolicModel
from porepy.numerics.fv.fvutils import compute_discharges

from porepy.viz import exporter
import matplotlib.pyplot as plt
import matplotlib

from examples.papers.arXiv_1712_08479.utils import (
    perform_condensation,
    compute_errors,
    edge_params,
    assign_data,
)
from porepy.numerics.darcy_and_transport import static_flow_IE_solver


def define_grid():
    """
    Make cartesian grids and a bucket. One horizontal and one vertical 1d
    fracture in a 2d matrix domain.
    """

    f_1 = np.array([[.5, .5, .5, .5], [.25, .75, .75, .25], [.25, .25, .75, .75]])
    f_2 = np.array([[0.2, .8, .8, 0.2], [.5, .5, .5, .5], [.25, .25, .75, .75]])

    fracs = [f_1, f_2]
    mesh_kwargs = {"physdims": np.array([1, 1, 1])}
    nx = [20, 20, 20]
    gb = meshing.cart_grid(fracs, np.array(nx), **mesh_kwargs)
    gb.assign_node_ordering()
    return gb


def bc_object(g):
    if g.dim < 3:
        return bc.BoundaryCondition(g)
    # Neumann on all but two boundaries
    dirfaces = bc.face_on_side(g, ["xmin", "xmax"])
    dirfaces = np.concatenate(dirfaces)
    labels = np.array(["dir"] * dirfaces.size)
    return bc.BoundaryCondition(g, dirfaces, labels)


class FlowData(EllipticDataAssigner):
    """
    Assign flow problem data to a given grid.
    """

    def __init__(self, g, d):
        EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = np.power(1e-6, 3 - self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        if np.isclose(self.grid().cell_centers[0, 0], .5):
            k = 1e-6
        elif self.grid().dim == 2:
            k = 1e6
        else:
            k = 1

        return tensor.SecondOrderTensor(3, k * np.ones(self.grid().num_cells))

    def bc(self):
        return bc_object(self.grid())

    def bc_val(self):
        bc_values = np.zeros(self.grid().num_faces)
        # p_D = 1-x for highest dimension:
        if self.grid().dim > 2:
            bc_values[bc.face_on_side(self.grid(), "xmin")[0]] = 1
        return bc_values


class FlowModel(EllipticModel):
    def __init__(self, gb, el=""):
        # Initialize base class
        kw = {"folder_name": global_folder_name + el}
        EllipticModel.__init__(self, gb, **kw)


class BothProblems:
    def __init__(self, full, reduced):
        self.full = full
        self.el = reduced

    def solve(self, h, v):
        # Update and solve full problem
        p = self.full.solve()

        # Obtain the reduced solution matrix and solve. First discretize
        # the eliminated problem (without intersections) to obtain flux
        # discretizations for backcalculation. These could in principle have
        # been obtained from the full discretization.
        self.el.solve()
        # Then get the Schur complement eliminated lhs and rhs from the
        # existing full discretization
        perform_condensation(self.full, self.el, 1)
        # And solve
        self.el.x = sps.linalg.spsolve(self.el.lhs, self.el.rhs)
        # Evaluate condition numbers
        self.full.cond = SC.sparse_condition_number(self.full.lhs)
        self.el.cond = SC.sparse_condition_number(self.el.lhs)
        return p, self.el.x

    def save(self):
        """
        Save quantities to be compared for error-evaluation.
        """
        self.full.save(["pressure"])
        self.el.save(["pressure"])


#        np.savetxt(global_folder_name + '/pressures_full.csv',
#                   self.full.x, delimiter=",")
#        np.savetxt(global_folder_name + '/pressures_el.csv',
#                   self.el.x, delimiter=",")
#        np.savetxt(global_folder_name + '/condition_number.csv',
#                   [self.full.cond])
#        np.savetxt(global_folder_name + '/condition_number_el.csv',
#                   [self.el.cond])


class TransportData(ParabolicDataAssigner):
    """
    Assign transport problem data to given grid.
    """

    def __init__(self, g, d):
        ParabolicDataAssigner.__init__(self, g, d)

    def bc(self):
        return bc_object(self.grid())

    def initial_condition(self):
        return np.ones(self.grid().num_cells)

    def aperture(self):
        a = np.power(1e-6, 3 - self.grid().dim)
        return np.ones(self.grid().num_cells) * a


class TransportSolver(ParabolicModel):
    """
    Make a ParabolicModel for the transport problem with specified parameters.
    """

    def __init__(self, gb, el=""):
        self._g = gb
        kw = {"folder_name": global_folder_name + el}
        ParabolicModel.__init__(self, gb, **kw)
        self._solver.parameters["store_results"] = True

    def grid(self):
        return self._g

    def space_disc(self):
        return self.advective_disc()

    def time_step(self):
        return 0.025

    def end_time(self):
        return .5

    def solver(self):
        return static_flow_IE_solver(self)


def map_from_mrst(gb, fn):
    vols = []
    cc = np.array([[], [], []])
    for g, d in gb:
        c = g.cell_centers
        v = np.multiply(g.cell_volumes, d["param"].get_aperture())
        vols = np.append(vols, v)
        cc = np.append(cc, c, axis=1)

    cc_mrst = np.loadtxt(fn + "/cell_centers_mrst.csv", delimiter=",")
    nc = cc.shape[1]
    cc_map = np.zeros(nc)

    for i in range(nc):
        # Find porepy cell number i in mrst ordering (ismember 'rows')
        cc_map[i] = np.nonzero(
            np.all(np.isclose(cc_mrst, np.tile(cc[:, i], (nc, 1))), axis=1)
        )[0][0]
    return vols, cc_map.astype(int)


def mrst_variables(fn, cc_map):
    p = np.loadtxt(fn + "/pressures_mrst.csv", delimiter=",")
    t = np.loadtxt(fn + "/tracer_mrst.csv", delimiter=",")
    tvec = np.loadtxt(fn + "/tvec_mrst.csv", delimiter=",")
    return p[cc_map], t[cc_map], tvec


def save_mrst_to_paraview(gb, u, u_name, fn):
    Both.full.flux_disc().split(gb, u_name, u)
    exp = exporter.Exporter(gb, fn, folder=global_folder_name)
    exp.write_vtk([u_name])


def plot_monitored_tracer(t, t_SC, tv_SD, P):
    g3d = gb.grids_of_dimension(3)[0]
    a, b, c = .95, .45, .55
    tv_SD = np.mean(tv_SD, axis=1)

    cell = np.where(
        np.all(
            [
                g3d.cell_centers[0, :] > a,
                g3d.cell_centers[1, :] > b,
                g3d.cell_centers[1, :] < c,
                g3d.cell_centers[2, :] > b,
                g3d.cell_centers[2, :] < c,
            ],
            axis=0,
        )
    )[0]

    endtime = P.end_time()
    timesteps = len(t)
    tv = np.zeros(timesteps)
    tv_SC = np.zeros(timesteps)
    for i in range(timesteps):
        tv[i] = np.mean(t[i][cell])
        tv_SC[i] = np.mean(t_SC[i][cell])

    time = np.linspace(0, endtime, timesteps)
    plt.close("all")
    plt.figure(figsize=(172 / 25.4, 129 / 25.4))
    matplotlib.rc("font", **{"size": 14})
    matplotlib.rc("lines", linewidth=3)

    plt.plot(time, tv, label="No elimination")
    plt.plot(time, tv_SC, ls="--", label="Schur complement")
    plt.plot(time, tv_SD, label="Star-Delta")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.savefig(global_folder_name + "/monitored_tracer.png")
    plt.show()


if __name__ == "__main__":

    global_folder_name = "results"
    gb = define_grid()
    assign_data(gb, FlowData, "problem")

    edge_params(gb)
    gb_el, el_data = gb.duplicate_without_dimension(1)

    problem = FlowModel(gb)
    problem_el = FlowModel(gb_el, el="_el")

    Both = BothProblems(problem, problem_el)
    k_h = 10 ** 4
    k_v = 10 ** -4
    p, p_el = Both.solve(k_h, k_v)
    problem.flux_disc().split(gb, "pressure", p)
    problem_el.flux_disc().split(gb_el, "pressure", p_el)
    Both.save()

    SC.compute_elimination_fluxes(gb, gb_el, el_data)
    compute_discharges(gb_el)
    compute_discharges(gb)

    assign_data(gb, TransportData, "transport_data")
    transport_problem = TransportSolver(gb)
    sol = transport_problem.solve()

    ndof_el = problem_el.flux_disc().ndof(gb_el)
    assign_data(gb_el, TransportData, "transport_data")
    transport_problem_el = TransportSolver(gb_el, el="_el")
    sol_el = transport_problem_el.solve()

    t = sol["transport"]
    t_el = sol_el["transport"]
    transport_problem.split(x_name="solution")
    transport_problem.save(["solution"])
    transport_problem_el.split(x_name="solution")

    transport_problem_el.save(["solution"])

    mrst_fn = "MRST"
    cell_volumes, dof_map = map_from_mrst(gb_el, mrst_fn)
    p_mrst, t_mrst, tvec_mrst = mrst_variables(mrst_fn, dof_map)

    save_mrst_to_paraview(gb_el, p_mrst, "p_mrst", "pressures_mrst")
    save_mrst_to_paraview(gb_el, t_mrst, "t_mrst", "tracer_mrst")
    Ep_pp, Ep_mrst, Ep_glob_pp, Ep_glob_mrst = compute_errors(gb_el, p, p_el, p_mrst)
    Et_pp, Et_mrst, Et_glob_pp, Et_glob_mrst = compute_errors(
        gb_el, t[-1], t_el[-1], t_mrst
    )
    print(
        "\nPressure errors for each subdomain ",
        Ep_pp,
        Ep_mrst,
        "Global:",
        Ep_glob_pp,
        Ep_glob_mrst,
    )
    print(
        "\nPressure errors for each subdomain ",
        Et_pp,
        Et_mrst,
        "Global",
        Et_glob_pp,
        Et_glob_mrst,
    )
    plot_monitored_tracer(t, t_el, tvec_mrst, transport_problem)
    print(
        "\nCondition numbers. Eliminated ",
        Both.el.cond,
        ", non-eliminated ",
        Both.full.cond,
        " and ratio ",
        Both.full.cond / Both.el.cond,
    )
