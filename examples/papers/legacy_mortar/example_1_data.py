import numpy as np

from porepy.grids.grid import FaceTag
from porepy.fracs import importer, mortars
from porepy.grids import refinement

from porepy.params.data import Parameters
from porepy.params.bc import BoundaryCondition
from porepy.params import tensor

# ------------------------------------------------------------------------------#


def domain():
    return {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}


# ------------------------------------------------------------------------------#


def tol():
    return 1e-8


# ------------------------------------------------------------------------------#


def create_gb(cells_2d, alpha_1d=None, alpha_mortar=None):

    mesh_kwargs = {}
    mesh_size = np.sqrt(2 / cells_2d)
    mesh_kwargs["mesh_size"] = {
        "mode": "constant",
        "tol": tol(),
        "value": mesh_size,
        "bound_value": mesh_size,
    }

    file_name = "example_1_split.csv"
    gb = importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain())

    g_map = {}
    if alpha_1d is not None:
        for g, d in gb:
            if g.dim == 1:
                num_nodes = int(g.num_nodes * alpha_1d)
                g_map[g] = refinement.new_grid_1d(g, num_nodes=num_nodes)
                g_map[g].compute_geometry()

    mg_map = {}
    if alpha_mortar is not None:
        for e, d in gb.edges_props():
            mg = d["mortar_grid"]
            if mg.dim == 1:
                mg_map[mg] = {}
                for s, g in mg.side_grids.items():
                    num_nodes = int(g.num_nodes * alpha_mortar)
                    mg_map[mg][s] = refinement.new_grid_1d(g, num_nodes=num_nodes)

    gb = mortars.replace_grids_in_bucket(gb, g_map, mg_map, tol())
    gb.assign_node_ordering()

    return gb


# ------------------------------------------------------------------------------#


def add_data(gb, solver):

    data = {"km": 1, "kf_high": 1e2, "kf_low": 1e-2, "aperture": 1e-2}

    if_solver = solver == "vem" or solver == "rt0" or solver == "p1"

    # only when solving for the vem case
    if solver == "vem" or solver == "rt0":
        [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, FaceTag.FRACTURE) for g, _ in gb]

    gb.add_node_props(["param", "is_tangential"])
    for g, d in gb:
        param = Parameters(g)

        if g.dim == 2:
            kxx = np.ones(g.num_cells) * data["km"]
            if if_solver:
                perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=kxx, kzz=1)
            else:
                perm = tensor.SecondOrder(3, kxx=kxx)
        elif g.dim == 1:
            if g.cell_centers[1, 0] > 0.25 and g.cell_centers[1, 0] < 0.55:
                kxx = np.ones(g.num_cells) * data["kf_low"]
            else:
                kxx = np.ones(g.num_cells) * data["kf_high"]

            if if_solver:
                perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=1, kzz=1)
            else:
                perm = tensor.SecondOrder(3, kxx=kxx)

        param.set_tensor("flow", perm)

        aperture = np.power(data["aperture"], gb.dim_max() - g.dim)
        param.set_aperture(aperture * np.ones(g.num_cells))

        param.set_source("flow", np.zeros(g.num_cells))

        bound_faces = g.get_domain_boundary_faces()
        if bound_faces.size == 0:
            bc = BoundaryCondition(g, np.empty(0), np.empty(0))
            param.set_bc("flow", bc)
        else:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > domain()["ymax"] - tol()
            bottom = bound_face_centers[1, :] < domain()["ymin"] + tol()

            labels = np.array(["neu"] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[top]] = 1

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)

        d["is_tangential"] = True
        d["param"] = param

    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        g_l = gb.sorted_nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()

        if g_l.dim == 1:
            if g_l.cell_centers[1, 0] > 0.25 and g_l.cell_centers[1, 0] < 0.55:
                kxx = data["kf_low"]
            else:
                kxx = data["kf_high"]
        else:
            kxx = data["kf_high"]

        gamma = np.power(
            check_P * gb.node_prop(g_l, "param").get_aperture(),
            1. / (gb.dim_max() - g_l.dim),
        )
        d["kn"] = kxx * np.ones(mg.num_cells) / gamma


# ------------------------------------------------------------------------------#
