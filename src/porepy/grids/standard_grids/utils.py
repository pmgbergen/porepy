import porepy as pp


def set_mesh_sizes(mesh_args):
    """
    Checks whether mesh_size_min and mesh_size_bound are present in the mesh_args dictionary.
    If not, they are set to 1/5 and 2 of the mesh_size_frac value, respectively.
    """
    if not "mesh_size_frac" in mesh_args:
        raise ValueError("Mesh size parameters should be specified via mesh_size_frac")
    if "mesh_size_bound" not in mesh_args:
        mesh_args["mesh_size_bound"] = 2 * mesh_args["mesh_size_frac"]
    if "mesh_size_min" not in mesh_args:
        mesh_args["mesh_size_min"] = 1 / 5 * mesh_args["mesh_size_frac"]


def make_gb_2d_simplex(mesh_args, points, edges, domain):
    """
    Construct a grid bucket with triangle grid in the 2d domain.

    Args:
        mesh_args (dict): Contains at least "mesh_size_frac". If the optional values of
            "mesh_size_bound" and "mesh_size_min" are not provided, these are set by
            set_mesh_sizes.
        points (): Fracture endpoints.
        fractures (): Connectivity list of the fractures, pointing to the provided
            endpoint list.
        domain (dict): Defines the size of the rectangular domain.

    Returns: Grid bucket with geometry computation and node ordering performed.
    """
    set_mesh_sizes(mesh_args)
    network = pp.FractureNetwork2d(points, edges, domain)
    gb = network.mesh(mesh_args)
    gb.compute_geometry()
    return gb


def make_gb_2d_cartesian(n_cells, fractures, domain):
    """
    Construct a grid bucket with Cartesian grid in the 2d domain.

    Args:
        n_cells list: Contains number of cells in x and y direction.
        fractures (list): Each element is an np.array([[x0, x1],[y0,y1]])
        domain (dict): Defines the size of the rectangular domain.

    Returns: Grid bucket with geometry computation and node ordering performed.
    """
    gb = pp.meshing.cart_grid(
        fractures, n_cells, physdims=[domain["xmax"], domain["ymax"]]
    )
