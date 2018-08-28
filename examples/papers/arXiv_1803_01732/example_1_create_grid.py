from porepy.fracs import importer


def create(mesh_size, tol):

    mesh_kwargs = {}
    mesh_kwargs["mesh_size"] = {
        "mode": "constant",
        "value": mesh_size,
        "bound_value": 1,
    }

    file_name = "analytical/DFN.fab"
    file_intersections = "analytical/TRACES.dat"

    gb = importer.dfn_3d_from_fab(file_name, file_intersections, tol=tol, **mesh_kwargs)
    gb.remove_nodes(lambda g: g.dim == 0)
    gb.compute_geometry()
    gb.assign_node_ordering()

    return gb
