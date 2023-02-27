from __future__ import annotations

import numpy as np

import porepy as pp


def unit_domain(dimension: int) -> pp.Domain:
    """Return a domain of unitary size extending from 0 to 1 in all dimensions.

    Parameters:
        dimension: the dimension of the domain.

    Returns:
        dict: specification of max and min in all dimensions.

    """
    assert dimension in np.arange(1, 4)
    bbox: dict[str, pp.number] = {"xmin": 0, "xmax": 1}
    if dimension > 1:
        bbox.update({"ymin": 0, "ymax": 1})
    if dimension > 2:
        bbox.update({"zmin": 0, "zmax": 1})
    return pp.Domain(bbox)


def set_mesh_sizes(mesh_args: dict):
    """
    Checks whether mesh_size_min and mesh_size_bound are present in the mesh_args dictionary.
    If not, they are set to 1/5 and 2 of the mesh_size_frac value, respectively.

    """
    if "mesh_size_frac" not in mesh_args:
        raise ValueError("Mesh size parameters should be specified via mesh_size_frac")
    if "mesh_size_bound" not in mesh_args:
        mesh_args["mesh_size_bound"] = 2 * mesh_args["mesh_size_frac"]
    if "mesh_size_min" not in mesh_args:
        mesh_args["mesh_size_min"] = 1 / 5 * mesh_args["mesh_size_frac"]


def make_mdg_2d_simplex(
    mesh_args: dict, points: np.ndarray, fractures: np.ndarray, domain: pp.Domain
):
    """Construct a mixed-dimensional simplex grid in the 2d domain.

    Parameters:
        mesh_args: Contains at least "mesh_size_frac". If the optional values of
            "mesh_size_bound" and "mesh_size_min" are not provided, these are set by
            set_mesh_sizes.
        points: Fracture endpoints.
        fractures: Connectivity list of the fractures, pointing to the provided
            endpoint list.
        domain: Domain object.

    Returns:
        Mixed-dimensional grid with geometry computation and node ordering performed.

    """
    set_mesh_sizes(mesh_args)
    fracs = pp.frac_utils.pts_edges_to_linefractures(points, fractures)
    network = pp.FractureNetwork2d(fracs, domain)
    mdg = network.mesh(mesh_args)
    mdg.compute_geometry()
    return mdg


def make_mdg_2d_cartesian(
    n_cells: np.ndarray, fractures: list[np.ndarray], domain: pp.Domain
):
    """
    Construct a Cartesian mixed-dimensional grid in the 2d domain.

    Parameters:
        n_cells: Contains number of cells in x and y direction.
        fractures: Each element is an np.array([[x0, x1],[y0,y1]])
        domain: Domain object.

    Returns:
        Mixed-dimensional grid with geometry computation and node ordering performed.

    """
    mdg: pp.MixedDimensionalGrid = pp.meshing.cart_grid(
        fractures,
        n_cells,
        physdims=[domain.bounding_box["xmax"], domain.bounding_box["ymax"]],
    )
    return mdg
