import logging, time
import scipy.sparse as sps
import numpy as np

import porepy as pp
from porepy.numerics.darcy_and_transport import static_flow_IE_solver as TransportSolver

# ------------------------------------------------------------------------------#

logger = logging.getLogger(__name__)

def export(gb, folder):

    props = ["cell_volumes", "cell_centers"]
    gb.add_node_props(props)
    for g, d in gb:
        d["cell_volumes"] = g.cell_volumes
        d["cell_centers"] = g.cell_centers

    # extra properties
    if all(gb.has_nodes_prop(gb.get_grids(), "pressure")):
        props.append("pressure")

    if all(gb.has_nodes_prop(gb.get_grids(), "P0u")):
        props.append("P0u")

    save = pp.Exporter(gb, "sol", folder=folder)
    save.write_vtk(props)

# ------------------------------------------------------------------------------#


def pressure(gb, folder):

    # Choose and define the solvers and coupler
    logger.info('VEM discretization')
    tic = time.time()
    solver_flow = pp.DualVEMMixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    logger.info('Done. Elapsed time: ' + str(time.time() - tic))
    logger.info('Linear solver')
    tic = time.time()
    up = sps.linalg.spsolve(A_flow, b_flow)
    logger.info('Done. Elapsed time ' + str(time.time() - tic))

    solver_flow.split(gb, "up", up)
    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.project_u(gb, "discharge", "P0u")

    export(gb, folder)

# ------------------------------------------------------------------------------#


def transport(gb, data, folder, adv_data_assigner, callback=None, save_every=1):

    physics = "transport"
    field_name = "tracer"
    for g, d in gb:
        d[physics + "_data"] = adv_data_assigner(gb, field_name, g, d, data)

    # Assign coupling diffusivity
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()

        gamma = check_P * gb.node_props(g_l, "param").get_aperture()
        k = check_P * gb.node_props(g_l, "param").get_tensor(physics).perm[0, 0, :]
        d["kn"] = k * np.ones(mg.num_cells) / gamma

    advective = AdvectiveProblem(
        gb,
        physics,
        time_step=data["dt"],
        end_time=data["t_max"],
        folder_name=folder,
        file_name=field_name,
        callback=callback,
    )
    advective.solve(field_name, save_every=save_every)
    return advective


class AdvectiveProblem(pp.ParabolicModel):
#    def space_disc(self):
#        return self.source_disc(), self.advective_disc()

    def solver(self):
        "Initiate solver"
        return Transport(self)


class Transport(TransportSolver):

    def __init__(self, problem):
        self.gb = problem.grid()
        self.outflow = np.empty(0)
        super().__init__(problem)

    def step(self, IE_solver):
        "Take one time step"
        self.p = IE_solver(self.lhs_time * self.p0 + self.static_rhs)
        self._compute_flow_rate()
        return self.p

    def _compute_flow_rate(self):
        # this function is only for the first benchmark case
        for g, d in self.gb:
            if g.dim < 2:
                continue
            faces, cells, sign = sps.find(g.cell_faces)
            index = np.argsort(cells)
            faces, sign = faces[index], sign[index]

            discharge = d["discharge"].copy()
            discharge[faces] *= sign
            discharge[g.get_internal_faces()] = 0
            discharge[discharge < 0] = 0
            val = np.dot(discharge, np.abs(g.cell_faces) * self.p[:g.num_cells])
            self.outflow = np.r_[self.outflow, val]
