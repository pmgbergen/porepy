"""
Test case 4. Several fractures, flow and transport with and without Schur
complement elimination.
"""
import numpy as np

from porepy.fracs import meshing
from porepy.numerics.elliptic import EllipticModel, EllipticDataAssigner
from porepy.numerics.darcy_and_transport import DarcyAndTransport, static_flow_IE_solver
from porepy.params import bc, tensor
from porepy.numerics.parabolic import ParabolicModel, ParabolicDataAssigner
from porepy.numerics.linalg.linsolve import Factory as LSFactory
from porepy.numerics.mixed_dim import condensation as SC

from porepy.fracs.fractures import EllipticFracture
import time
import logging
from examples.papers.arXiv_1712_08479.utils import (
    gb_error,
    global_error,
    edge_params,
    assign_data,
)

# Module-wide logger
logger = logging.getLogger(__name__)


def define_grid():
    """
    Make cartesian grids and a bucket. One horizontal and one vertical 1d
    fracture in a 2d matrix domain.
    """
    h_max = 0.018
    mesh_size_min = h_max / 3

    domain = {
        "xmin": 0.2,
        "xmax": .8,
        "ymin": 0.25,
        "ymax": .75,
        "zmin": 0.5,
        "zmax": .7,
    }

    f_1 = np.array([[0.2, .8, .8, 0.2], [.5, .5, .5, .5], [.5, .5, .7, .7]])
    f_2 = np.array([[.5, .5, .5, .5], [.3, .7, .7, .3], [.5, .5, .7, .7]])
    f_3 = np.array([[.7, .7, .7, .7], [.25, .75, .75, .25], [.5, .5, .7, .7]])
    f_4 = np.array([[.3, .3, .3, .3], [.25, .75, .75, .25], [.5, .5, .7, .7]])
    c_1 = np.array([.35, .6, .6])
    c_5 = np.array([.35, .4, .6])
    c_2 = np.array([.65, .38, .65])
    c_3 = np.array([.65, .62, .65])
    c_4 = np.array([.5, .5, .6])
    ma_1, mi_1, a_1, s_1, d_1 = 0.15, 0.08, -5 / 9, -5 / 9, np.pi / 2
    ma_5, mi_5, a_5, s_5, d_5 = 0.15, 0.08, 5 / 9, 5 / 9, np.pi / 2
    ma_2, mi_2, a_2, s_2, d_2 = 0.12, 0.07, -np.pi / 4, 0, 0
    ma_3, mi_3, a_3, s_3, d_3 = 0.12, 0.07, np.pi / 4, 0, 0
    ma_4, mi_4, a_4, s_4, d_4 = 0.08, 0.08, 0, 0, 0
    fracs = [
        f_1,
        f_2,
        f_3,
        f_4,
        EllipticFracture(c_1, ma_1, mi_1, a_1, s_1, d_1),
        EllipticFracture(c_2, ma_2, mi_2, a_2, s_2, d_2),
        EllipticFracture(c_3, ma_3, mi_3, a_3, s_3, d_3),
        EllipticFracture(c_4, ma_4, mi_4, a_4, s_4, d_4),
        EllipticFracture(c_5, ma_5, mi_5, a_5, s_5, d_5),
    ]
    gb = meshing.simplex_grid(
        fracs, domain, mesh_size_frac=h_max, mesh_size_min=mesh_size_min
    )

    gb.compute_geometry()
    gb.assign_node_ordering()
    return gb


def boundary_face_type(g):
    """
    Extract the faces where Dirichlet conditions are to be set.
    """
    if g.dim < 2:
        return np.array([])
    if g.dim == 2 and not np.isclose(g.cell_centers[0, 0], g.cell_centers[0, 1]):
        return np.array([])
    bound_faces = bc.face_on_side(g, ["ymin", "ymax"])
    return np.array(np.concatenate(bound_faces))


def bc_values(g):
    bc_val = np.zeros(g.num_faces)
    return bc_val


def perm(g):

    if np.isclose(g.cell_centers[2, 0], .6) or g.dim == 0:
        kxx = np.ones(g.num_cells) * 1e-5

    elif g.dim < 3:
        kxx = np.ones(g.num_cells) * 1e5

    else:
        kxx = np.ones(g.num_cells) * 1e-2
        kxx[g.cell_centers[1, :] < .5] = 1e-3
    return tensor.SecondOrderTensor(3, kxx)


def sources(g):
    s = np.zeros(g.num_cells)
    if (
        g.dim == 1
        and np.isclose(g.cell_centers[0, 0], .5)
        and g.cell_centers[2, 0] < .6
    ):
        s[-1] = 1
    return s


class FlowData(EllipticDataAssigner):
    """
    Assign flow problem data to a given grid.
    The parameters for which we want other values than
    the EllipticDataAssigner defaults are set by overwriting the
    corresponding function.
    """

    def __init__(self, g, d):
        EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = np.power(1e-6, 3 - self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        return perm(self.grid())

    def bc(self):
        dirfaces = boundary_face_type(self.grid())
        labels = np.array(["dir"] * dirfaces.size)

        return bc.BoundaryCondition(self.grid(), dirfaces, labels)

    def source(self):
        return sources(self.grid())

    def bc_val(self):
        return bc_values(self.grid())


class TransportData(ParabolicDataAssigner):
    """
    Assign transport problem data to given grid.
    """

    def __init__(self, g, d):
        ParabolicDataAssigner.__init__(self, g, d)

    def bc(self):
        dirfaces = boundary_face_type(self.grid())
        labels = np.array(["dir"] * dirfaces.size)

        return bc.BoundaryCondition(self.grid(), dirfaces, labels)

    def bc_val(self, t):
        return bc_values(self.grid()) * 0

    def source(self, t):
        return sources(self.grid())

    def aperture(self):
        a = np.power(1e-6, 3 - self.grid().dim)
        return np.ones(self.grid().num_cells) * a


class DarcySolver(EllipticModel):
    """
    Set up Darcy solver with MPFA.
    """

    def __init__(self, gb, mp, mix, el, kw):
        self.mp = mp
        self.mix = mix
        EllipticModel.__init__(self, gb, **kw)
        self.el = False
        if len(el) > 1:
            self.el = True
            self.full_grid = gb_full
            self.el_data = el_data

    def solve(self, max_direct=40000, callback=False, **kwargs):
        """
        This is an adaption of the elliptic solver to the Schur complement
        elimination. The only change can be found in the if self.el statement.
        """
        # Discretize
        tic = time.time()
        logger.info("Discretize")
        self.lhs, self.rhs = self.reassemble()
        if self.el:
            to_be_eliminated = SC.dofs_of_dimension(other.grid(), other.lhs, eldim)
            self.lhs, self.rhs, _, _, _ = SC.eliminate_dofs(
                other.lhs, other.rhs, to_be_eliminated
            )

        logger.info("Done. Elapsed time " + str(time.time() - tic))

        # Solve
        tic = time.time()
        ls = LSFactory()
        if self.rhs.size < max_direct:
            logger.info("Solve linear system using direct solver")
            self.x = ls.direct(self.lhs, self.rhs)
        else:
            logger.info("Solve linear system using GMRES")
            precond = self._setup_preconditioner()
            #            precond = ls.ilu(self.lhs)
            slv = ls.gmres(self.lhs)
            self.x, info = slv(
                self.rhs,
                M=precond,
                callback=callback,
                maxiter=10000,
                restart=1500,
                tol=1e-8,
            )
            if info == 0:
                logger.info("GMRES succeeded.")
            else:
                logger.error("GMRES failed with status " + str(info))

        logger.info("Done. Elapsed time " + str(time.time() - tic))
        return self.x


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
        return self.advective_disc(), self.source_disc()

    def time_step(self):
        return .5

    def end_time(self):
        return 2

    def solver(self):
        return static_flow_IE_solver(self)


class BothProblems(DarcyAndTransport):
    """
    Combine the two problems for convinience.
    """

    def __init__(self, gb, suffix="", mp=False, mix=False):
        flow, transport = self.setup_subproblems(gb, mp, mix, suffix)
        DarcyAndTransport.__init__(self, flow, transport)

    def setup_subproblems(self, gb, mp, mix, suffix):

        kw = {"folder_name": main_folder + suffix}

        darcy_problem = DarcySolver(gb, mp, mix, suffix, kw)
        assign_data(gb, TransportData, "transport_data")
        transport_problem = TransportSolver(gb, kw)
        return darcy_problem, transport_problem

    def solve_and_save(self):
        self.transport._solver.data["transport"] = []
        self.transport._solver.data["times"] = []
        p, t = self.solve()
        self.save(export_every=1)
        return p, t


def evaluate_errors(p1, p2, t, t_mp, gb):
    e = np.array([1, 1])
    tracer_errors_L2 = gb_error(gb, t[-1], t_mp[-1][: p1.size])
    pressure_errors_L2 = gb_error(gb, p1, p2)
    print(
        "L2 [matrix, fracture], tracer",
        tracer_errors_L2,
        " and pressure",
        pressure_errors_L2,
    )
    e = [
        global_error(gb, p1, p2[: p1.size]),
        global_error(gb, t[-1], t_mp[-1][: p1.size]),
    ]
    return e


if __name__ == "__main__":
    main_folder = "results"
    gb = define_grid()

    assign_data(gb, FlowData, "problem")
    edge_params(gb)
    gb_full = gb
    eldim = 0
    gb_el, el_data = gb.duplicate_without_dimension(eldim)

    Problems = BothProblems(gb)
    other = Problems.flow
    Problems_el = BothProblems(gb_el, "_el")
    p, t = Problems.solve_and_save()
    p_el, t_el = Problems_el.solve_and_save()
    cond_full = SC.sparse_condition_number(Problems.flow.lhs)
    cond_el = SC.sparse_condition_number(Problems_el.flow.lhs)
    print("Condition numbers", cond_full, cond_el, cond_full / cond_el)
    diff = t[-1][: p_el.size] - t_el[-1]
    print(np.amax(np.absolute(diff)), np.sum(np.absolute(diff)))
    errors = evaluate_errors(p_el, p, t_el, t, gb_el)
    print(errors)
