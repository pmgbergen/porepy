from porepy.fracs import importer
from porepy.grids import coarsening as co


def create(id_problem, is_coarse=False, mesh_size=0.05, tol=1e-5):

    folder = "/home/elle/Dropbox/Work/tipetut/test2/simple_geometry/"
    file_name = folder + "DFN_" + str(id_problem) + ".fab"
    file_intersections = folder + "TRACES_" + str(id_problem) + ".dat"

    mesh_kwargs = {}
    mesh_kwargs["mesh_size"] = {
        "mode": "constant",
        "value": mesh_size,  # 0.05 0.15 0.425
        "bound_value": 1,
    }

    gb = importer.dfn_3d_from_fab(file_name, file_intersections, tol=tol, **mesh_kwargs)
    gb.remove_nodes(lambda g: g.dim == 0)
    gb.compute_geometry()
    if is_coarse:
        co.coarsen(gb, "by_volume")
    gb.assign_node_ordering()

    return gb
