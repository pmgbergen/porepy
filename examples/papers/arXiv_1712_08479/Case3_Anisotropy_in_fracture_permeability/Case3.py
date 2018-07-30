"""
Test case 3, investigating effect of fracture anisotropy.
Flow and transport problems solved. Three different flow solvers: MPFA, TPFA
and a hybrid with MPFA in the fracture and TPFA in the matrix.
"""
import matplotlib
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

from porepy.fracs import meshing
from porepy.numerics.elliptic import EllipticModel, EllipticDataAssigner
from porepy.numerics.darcy_and_transport import DarcyAndTransport, static_flow_IE_solver
from porepy.params import bc, tensor
from porepy.utils import comp_geom as cg
from porepy.numerics.fv import mpfa, tpfa
from porepy.numerics.parabolic import ParabolicModel, ParabolicDataAssigner
from porepy.numerics.mixed_dim.solver import SolverMixedDim
from porepy.numerics.mixed_dim.coupler import Coupler
from examples.papers.arXiv_1712_08479.utils import gb_error, edge_params, assign_data


def define_grid():
    """
    Make cartesian grids and a bucket. One horizontal and one vertical 1d
    fracture in a 2d matrix domain.
    """
    f_1 = np.array([[0, 0, 1, 1], [0, 1, 1, 0], [.5, .5, .5, .5]])

    fracs = [f_1]
    mesh_kwargs = {"physdims": np.array([1, 1, 1])}
    gb = meshing.cart_grid(fracs, np.array([nx, nx, nx]), **mesh_kwargs)
    gb.assign_node_ordering()
    return gb


def boundary_face_type(g):
    """
    Extract the faces where Dirichlet conditions are to be set.
    """
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bound_face_centers = g.face_centers[:, bound_faces]
    onev = np.ones(bound_face_centers.shape[1])
    a = 1 / nx
    tol = 1e-5
    dirface1 = np.where(
        np.array(bound_face_centers[0, :] < a * onev)
        & np.array(bound_face_centers[1, :] < a * onev)
        & np.array(bound_face_centers[2, :] < tol * onev)
    )
    dirface2 = np.where(
        np.array(bound_face_centers[0, :] > (1 - a) * onev)
        & np.array(bound_face_centers[1, :] > (1 - a) * onev)
        & np.array(bound_face_centers[2, :] > (1 - tol) * onev)
    )
    return bound_faces, dirface1, dirface2


def bc_values(g):
    bc_val = np.zeros(g.num_faces)
    if g.dim == 1:
        return bc_val
    bound_faces, dirface1, _ = boundary_face_type(g)
    bc_val[bound_faces[dirface1]] = 1
    return bc_val


class FlowData(EllipticDataAssigner):
    """
    Assign flow problem data to a given grid.
    """

    def __init__(self, g, d):
        EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = np.power(1e-3, 3 - self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        if self.grid().dim == 3:
            kxx = np.ones(self.grid().num_cells) * np.power(1e4, 3 > self.grid().dim)
            return tensor.SecondOrderTensor(3, kxx)
        else:
            return anisotropy(self.grid(), d, y)

    def bc(self):
        if self.grid().dim < 3:
            return bc.BoundaryCondition(self.grid())
        bound_faces, dirface1, dirface2 = boundary_face_type(self.grid())
        labels = np.array(["neu"] * bound_faces.size)
        dirfaces = np.concatenate((dirface1, dirface2))
        labels[dirfaces] = "dir"
        return bc.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self):
        return bc_values(self.grid())


def anisotropy(g, deg, yfactor):
    """
    Set anisotropic permeability in the 2d matrix.
    """
    # Get rotational tensor R
    k = 1e3
    perm_x = k
    perm_y = k / yfactor
    perm_z = k
    rad = deg * np.pi / 180
    v = np.array([0, 0, 1])
    R = cg.rot(rad, v)
    # Set up orthogonal permeability tensor and rotate it
    k_orth = np.array([[perm_x, 0, 0], [0, perm_y, 0], [0, 0, perm_z]])
    k = np.dot(np.dot(R, k_orth), R.T)

    kf = np.ones(g.num_cells)
    kxx = kf * k[0, 0]
    kyy = kf * k[1, 1]
    kxy = kf * k[0, 1]
    kxz = kf * k[0, 2]
    kyz = kf * k[1, 2]
    kzz = kf * k[2, 2]
    perm = tensor.SecondOrderTensor(
        3, kxx=kxx, kyy=kyy, kzz=kzz, kxy=kxy, kxz=kxz, kyz=kyz
    )
    return perm


class MixedDiscretization(SolverMixedDim):
    """
    Mixed-dimensional solver with flux approximation method dependent on
    subdomain dimension.
    """

    def __init__(self, physics="flow"):
        self.physics = physics
        self.discr = tpfa.Tpfa(self.physics)
        self.discr_ndof = self.discr.ndof
        self.coupling_conditions = tpfa.TpfaCoupling(self.discr)
        tp = tpfa.Tpfa(self.physics)
        mp = mpfa.Mpfa(self.physics)

        def discr_fct(g, d):
            return mp.matrix_rhs(g, d) if g.dim < 3 else tp.matrix_rhs(g, d)

        kwargs = {"discr_fct": discr_fct}
        self.solver = Coupler(self.discr, self.coupling_conditions, **kwargs)


class TransportData(ParabolicDataAssigner):
    """
    Assign transport problem data to given grid.
    """

    def __init__(self, g, d):
        ParabolicDataAssigner.__init__(self, g, d)

    def bc(self):
        bound_faces, dirface1, dirface2 = boundary_face_type(self.grid())
        labels = np.array(["neu"] * bound_faces.size)
        dirfaces = np.concatenate((dirface1, dirface2))
        labels[dirfaces] = "dir"
        return bc.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self, t):
        return np.zeros(self.grid().num_faces)

    def initial_condition(self):
        return np.ones(self.grid().num_cells)

    def aperture(self):
        a = np.power(1e-3, 3 - self.grid().dim)
        return np.ones(self.grid().num_cells) * a


class DarcySolver(EllipticModel):
    """
    Set up Darcy solver with MPFA.
    """

    def __init__(self, gb, mp, mix, kw):
        self.mp = mp
        self.mix = mix
        EllipticModel.__init__(self, gb, **kw)

    def flux_disc(self):
        if self.mp:
            return mpfa.MpfaMixedDim(physics=self.physics)
        elif self.mix:
            return MixedDiscretization(self.physics)
        else:
            return tpfa.TpfaMixedDim(physics=self.physics)


class TransportSolver(ParabolicModel):
    """
    Make a ParabolicModel for the transport problem with specified parameters.
    """

    def __init__(self, gb, kw):
        self._g = gb
        ParabolicModel.__init__(self, gb, **kw)
        self._solver.parameters["store_results"] = True

    def grid(self):
        return self._g

    def space_disc(self):
        return self.advective_disc()

    def time_step(self):
        return 1

    def end_time(self):
        return 30

    def solver(self):
        return static_flow_IE_solver(self)


class BothProblems(DarcyAndTransport):
    """
    Combine the two problems for convenience.
    """

    def __init__(self, gb, mp=False, mix=False):
        flow, transport = self.setup_subproblems(gb, mp, mix)
        DarcyAndTransport.__init__(self, flow, transport)

    def setup_subproblems(self, gb, mp, mix):
        appendix = ""
        if mp:
            appendix += "_mp"
        elif mix:
            appendix += "_mix"
        kw = {"folder_name": main_folder + appendix}

        darcy_problem = DarcySolver(gb, mp, mix, kw)
        assign_data(gb, TransportData, "transport_data")
        transport_problem = TransportSolver(gb, kw)
        return darcy_problem, transport_problem

    def solve_and_save_text(self):
        self.transport._solver.data["transport"] = []
        self.transport._solver.data["times"] = []
        p, t = self.solve()
        return p, t


def evaluate_errors(p1, p2, t, t_mp, gb):
    e = np.array([1, 1])
    tracer_errors_L2 = gb_error(gb, t[-1], t_mp[-1])
    pressure_errors_L2 = gb_error(gb, p1, p2)
    print(
        "L2 [matrix, fracture], tracer",
        tracer_errors_L2,
        " and pressure",
        pressure_errors_L2,
    )
    return e


def plot_monitored_tracer(t_tp, t_mp, t_mix, P):
    matplotlib.rc("font", **{"size": 14})
    matplotlib.rc("lines", linewidth=3)
    cell = gb.grids_of_dimension(3)[0].num_cells - 1
    endtime = P.transport.end_time()
    timesteps = len(t)
    tv_tp = np.zeros(timesteps)
    tv_mp = np.zeros(timesteps)
    tv_mix = np.zeros(timesteps)
    for i in range(timesteps):
        tv_tp[i] = t_tp[i][cell]
        tv_mp[i] = t_mp[i][cell]
        tv_mix[i] = t_mix[i][cell]
    time = np.linspace(0, endtime, timesteps)
    plt.figure(figsize=(172 / 25.4, 129 / 25.4))

    plt.plot(time, tv_mp)
    plt.plot(time, tv_mix, ls="--")
    plt.plot(time, tv_tp)
    plt.legend(["MPFA", "Hybrid", "TPFA"])
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.savefig("figures/monitored_tracer.png")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    nx = 40
    y = 3
    d = -45
    main_folder = "results"
    gb = define_grid()
    assign_data(gb, FlowData, "problem")
    edge_params(gb)
    gb_mp = gb.copy()
    gb_mix = gb.copy()
    Problems = BothProblems(gb)
    MP_Problems = BothProblems(gb_mp, mp=True)
    Mix_Problems = BothProblems(gb_mix, mix=True)
    p, t = Problems.solve_and_save_text()
    p_mix, t_mix = Mix_Problems.solve_and_save_text()
    p_mp, t_mp = MP_Problems.solve_and_save_text()
    plot_monitored_tracer(t, t_mp, t_mix, Problems)

    export_every = 2
    errors_tp = evaluate_errors(p, p_mp, t, t_mp, gb)
    errors_mix = evaluate_errors(p_mix, p_mp, t_mix, t_mp, gb)
    Problems.save(export_every)
    MP_Problems.save(export_every)
    Mix_Problems.save(export_every)
