"""Module for creating fractured cartesian grids in 2 and 3 dimensions.

The functions in this module can be accessed through the meshing wrapper module.

Todo:
    Since this module contains only private methods, consider making the whole
    module private.

"""
from __future__ import annotations

from typing import Optional

import numpy as np

import porepy as pp
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

from . import msh_2_grid
from .gmsh_interface import Tags


def _cart_grid_3d(
    fracs: list[np.ndarray], nx: np.ndarray, physdims: Optional[np.ndarray] = None
) -> list[list[pp.Grid]]:
    """Creates grids for a domain with possibly intersecting fractures in 3D.

    Based on rectangles describing the individual fractures, the method constructs
    grids in 3D (the whole domain), 2D (one for each individual fracture), 1D (along
    fracture intersections), and 0d (intersections of intersections).

    Parameters:
        fracs: A list of arrays with ``shape=(3, 4)``, representing vertices of the
            rectangle for each fracture. The vertices must be sorted and aligned to
            the axis. The fractures will snap to the closest grid faces.
        nx: ``shape=(3,)``

            Number of cells in each dimension.
        physdims: ``shape=(3,), default=None``

            Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.

    Returns:
        A nested list of length 4, where for each dimension 3 to 0 the respective
        sub-list contains all grids in that dimension.

    """

    nx = np.asarray(nx)
    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != nx.size:
        raise ValueError("Physical dimension must equal grid dimension")
    else:
        physdims = np.asarray(physdims)

    # We create a 3D cartesian grid. The global node mapping is trivial.
    g_3d = pp.CartGrid(nx, physdims=physdims)
    return _create_lower_dim_grids_3d(g_3d, fracs, nx, physdims)


def _tensor_grid_3d(
    fracs: list[np.ndarray], x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> list[list[pp.Grid]]:
    """
    Create a grids for a domain with possibly intersecting fractures in 3d.

    Based on lines describing the individual fractures, the method constructs grids
    in 3d (whole domain), 2d (individual fracture),  1d, and 0d (fracture
    intersections).

    Parameters:
        fracs: A list of arrays with ``shape=(3, 4)``, representing the vertices of the
            fractures for each fracture.
            The fracture lines must align to the coordinate axis.
            The fractures will snap to the closest grid nodes.
        x: Node coordinates in x-direction
        y: Node coordinates in y-direction.
        z: Node coordinates in z-direction.

    Returns:
        A nested list of length 4, where for each dimension 3 to 0 the respective
        sub-list contains all grids in that dimension.

    """

    nx = np.asarray((x.size - 1, y.size - 1, z.size - 1))
    g_3d = pp.TensorGrid(x, y, z)

    return _create_lower_dim_grids_3d(g_3d, fracs, nx)


def _cart_grid_2d(
    fracs: list[np.ndarray], nx: np.ndarray, physdims: Optional[np.ndarray] = None
) -> list[list[pp.Grid]]:
    """Creates grids for a domain with possibly intersecting fractures in 2D.

    Based on lines describing the individual fractures, the method constructs grids
    in 2D (whole domain), 1D (individual fracture), and 0D (fracture intersections).

    Parameters:
        fracs: A list of arrays with ``shape=(2, 2)``, representing the vertices of the
            line for each fracture.
            The fracture lines must align to the coordinate axis.
            The fractures will snap to the closest grid nodes.
        nx: ``shape=(2,)``

            Number of cells in each direction.
        physdims: ``shape=(2,), default=None``

            Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.

    Returns:
        A nested list of length 3, where for each dimension 2 to 0 the respective
        sub-list contains all grids in that dimension.

    """
    nx = np.asarray(nx)
    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != nx.size:
        raise ValueError("Physical dimension must equal grid dimension")
    else:
        physdims = np.asarray(physdims)

    g_2d = pp.CartGrid(nx, physdims=physdims)
    return _create_lower_dim_grids_2d(g_2d, fracs, nx)


def _tensor_grid_2d(
    fracs: list[np.ndarray], x: np.ndarray, y: np.ndarray
) -> list[list[pp.Grid]]:
    """Creates a grid for a domain with possibly intersecting fractures in 2D.

    Based on lines describing the individual fractures, the method constructs grids
    in 2D (whole domain), 1D (individual fracture), and 0D (fracture intersections).

    Parameters:
        fracs: A list of arrays with ``shape=(2, 2)``, representing the vertices of the
            line for each fracture.
            The fracture lines must align to the coordinate axis.
            The fractures will snap to the closest grid nodes.
        x: Node coordinates in x-direction
        y: Node coordinates in y-direction.

    Returns:
        A nested list of length 3, where for each dimension 2 to 0 the respective
        sub-list contains all grids in that dimension.

    """
    nx = np.asarray((x.size - 1, y.size - 1))
    g_2d = pp.TensorGrid(x, y)
    return _create_lower_dim_grids_2d(g_2d, fracs, nx)


def _create_lower_dim_grids_3d(
    g_3d: pp.Grid,
    fracs: list[np.ndarray],
    nx: np.ndarray,
    physdims: Optional[np.ndarray] = None,
) -> list[list[pp.Grid]]:
    """Auxiliary function to create a fractured domain in 3D.

    Creates nested lists of grids, where the first list contains the highest-dimensional
    grid, followed by list containing grids of lower dimension in descending order.

    Parameters:
        g_3d: The highest-dimensional grid.
        fracs: A list of arrays with ``shape=(3, 4)``, representing the vertices of the
            fractures for each fracture.
            The fracture lines must align to the coordinate axis.
            The fractures will snap to the closest grid nodes.
        nx: ``shape=(3,)``

            Number of cells in each dimension.
        physdims: ``shape=(3,), default=None``

            Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.

    Returns:
        A nested list of length 4, where for each dimension 3 to 0 the respective
        sub-list contains all grids in that dimension.

    """
    g_3d.global_point_ind = np.arange(g_3d.num_nodes)
    g_3d.compute_geometry()
    g_2d: list[pp.Grid] = []
    g_1d: list[pp.Grid] = []
    g_0d: list[pp.Grid] = []
    # We set the tolerance for finding points in a plane. This can be any small
    # number, that is smaller than .25 of the cell sizes.
    if physdims is None:
        tol = 1e-5 / nx
    else:
        tol = 0.1 * physdims / nx

    # Store a representation of the snapped fractures. This is needed for identifying
    # fracture intersections below.
    snapped_fractures = []

    # Create 2D grids
    for fi, f in enumerate(fracs):
        assert np.all(f.shape == (3, 4)), (
            "Fracture is set by an array of the edge points of shape (3, 4), "
            f"Passed array has shape: {f.shape}. Could it be because of trimming the "
            "part of the fracture outside the bounding box?"
        )
        is_xy_frac = np.allclose(f[2, 0], f[2])
        is_xz_frac = np.allclose(f[1, 0], f[1])
        is_yz_frac = np.allclose(f[0, 0], f[0])
        assert (
            is_xy_frac + is_xz_frac + is_yz_frac == 1
        ), "Fracture must align to x-, y- or z-axis"

        # snap to grid
        if physdims is None:
            f_s = g_3d.nodes[
                :,
                (
                    np.argmin(pp.distances.point_pointset(f[:, 0], g_3d.nodes)),
                    np.argmin(pp.distances.point_pointset(f[:, 1], g_3d.nodes)),
                    np.argmin(pp.distances.point_pointset(f[:, 2], g_3d.nodes)),
                    np.argmin(pp.distances.point_pointset(f[:, 3], g_3d.nodes)),
                ),
            ]
        else:
            f_s = (
                np.round(f * nx[:, np.newaxis] / physdims[:, np.newaxis])
                * physdims[:, np.newaxis]
                / nx[:, np.newaxis]
            )
        snapped_fractures.append(f_s)

        if is_xy_frac:
            flat_dim = [2]
            active_dim = [0, 1]
        elif is_xz_frac:
            flat_dim = [1]
            active_dim = [0, 2]
        else:
            flat_dim = [0]
            active_dim = [1, 2]
        # construct normal vectors. If the rectangle is ordered clockwise we need to
        # flip the normals so that they point outwards.
        sign = 2 * pp.geometry_property_checks.is_ccw_polygon(f_s[active_dim]) - 1
        tangent = f_s.take(np.arange(f_s.shape[1]) + 1, axis=1, mode="wrap") - f_s
        normal = tangent
        normal[active_dim] = tangent[active_dim[1::-1]]
        normal[active_dim[1]] = -normal[active_dim[1]]
        normal = sign * normal
        # We find all the faces inside the convex hull defined by the rectangle. To
        # find the faces on the fracture plane, we remove any faces that are further
        # than tol from the snapped fracture plane.
        in_hull = pp.half_space.point_inside_half_space_intersection(
            normal, f_s, g_3d.face_centers
        )
        f_tag = np.logical_and(
            in_hull,
            np.logical_and(
                f_s[flat_dim, 0] - tol[flat_dim] <= g_3d.face_centers[flat_dim],
                g_3d.face_centers[flat_dim] < f_s[flat_dim, 0] + tol[flat_dim],
            ),
        )
        f_tag = f_tag.ravel()
        nodes = sparse_array_to_row_col_data(g_3d.face_nodes[:, f_tag])[0]
        nodes = np.unique(nodes)
        loc_coord = g_3d.nodes[:, nodes]
        g = _create_embedded_2d_grid(loc_coord, nodes)

        g.frac_num = fi
        g_2d.append(g)

    # Create 1D grids.

    # Here we make use of the network class to find the intersection of fracture
    # planes. We could maybe avoid this by doing something similar as for the
    # 2D-case, and count the number of faces belonging to each edge, but we use the
    # FractureNetwork class for now.

    # We need to use the snapped fractures to be sure the identified intersections are
    # resolved in the grid.
    frac_list = []
    for f in snapped_fractures:
        frac_list.append(pp.PlaneFracture(f))

    # Combine the fractures into a network
    network = FractureNetwork3d(fractures=frac_list)
    # Impose domain boundary. For the moment, the network should be immersed in the
    # domain, or else gmsh will complain.
    if physdims is None:
        box = {
            "xmin": np.min(g_3d.nodes[0]),
            "ymin": np.min(g_3d.nodes[1]),
            "zmin": np.min(g_3d.nodes[2]),
            "xmax": np.max(g_3d.nodes[0]),
            "ymax": np.max(g_3d.nodes[1]),
            "zmax": np.max(g_3d.nodes[2]),
        }
    else:
        # Use default 0 for minima.
        box = {
            "xmax": physdims[0],
            "ymax": physdims[1],
            "zmax": physdims[2],
        }
    network.impose_external_boundary(pp.Domain(box))

    # Find intersections and split them.
    network.find_intersections()
    network.split_intersections()

    # Extract geometrical network information.
    pts = network.decomposition["points"]
    edges = network.decomposition["edges"]
    poly = network._poly_2_segment()
    # And tags identifying points and edges corresponding to normal fractures,
    # domain boundaries and subdomain boundaries. Only the entities corresponding to
    # normal fractures should actually be gridded.

    # TODO: Constraints have not been implemented for structured DFM grids.

    # Simply pass nothing for now, not sure how do deal with this, or if it at all is
    # meaningful.
    edge_tags, _, _ = network._classify_edges(poly, np.array([]))

    auxiliary_points, edge_tags = network._on_domain_boundary(edges, edge_tags)
    bound_and_aux = np.array(
        [Tags.DOMAIN_BOUNDARY_LINE.value, Tags.AUXILIARY_LINE.value]
    )

    # From information of which lines are internal, we can find intersection points.
    # This part will become more elaborate if we introduce constraints, see the
    # FractureNetwork3d class.

    # Find all points on fracture intersection lines
    isect_p = edges[:, edge_tags == Tags.FRACTURE_INTERSECTION_LINE.value].ravel()
    # Count the number of occurrences
    num_occ_pt = np.bincount(isect_p)
    # Intersection points if
    intersection_points = np.where(num_occ_pt > 1)[0]

    edges = np.vstack((edges, edge_tags))

    # Loop through the edges to make 1D grids. Omit the auxiliary edges.
    for e in np.ravel(np.where(edges[2] == Tags.FRACTURE_INTERSECTION_LINE.value)):
        # We find the start and end point of each fracture intersection (1D grid) and
        # then the corresponding global node index.
        if np.isin(edge_tags[e], bound_and_aux):
            continue
        s_pt = pts[:, edges[0, e]]
        e_pt = pts[:, edges[1, e]]
        nodes = _find_nodes_on_line(g_3d, nx, s_pt, e_pt)
        loc_coord = g_3d.nodes[:, nodes]
        assert (
            loc_coord.shape[1] > 1
        ), "1d grid in intersection should span\
            more than one node"
        g = msh_2_grid.create_embedded_line_grid(loc_coord, nodes)
        g_1d.append(g)

    # Create 0D grids

    # Here we also use the intersection information from the FractureNetwork class.
    # No grids for auxiliary points.
    for p in intersection_points:
        if auxiliary_points[p] == Tags.DOMAIN_BOUNDARY_POINT:
            continue
        node = np.argmin(pp.distances.point_pointset(pts[:, p], g_3d.nodes))
        assert np.allclose(g_3d.nodes[:, node], pts[:, p])
        g = pp.PointGrid(g_3d.nodes[:, node])
        g.global_point_ind = np.asarray(node)
        g_0d.append(g)

    grids: list[list[pp.Grid]] = [[g_3d], g_2d, g_1d, g_0d]
    return grids


def _create_lower_dim_grids_2d(
    g_2d: pp.Grid, fracs: list[np.ndarray], nx: np.ndarray
) -> list[list[pp.Grid]]:
    """Auxiliary function to create a fractured domain in 2D.

    Creates nested lists of grids, where the first list contains the highest-dimensional
    grid, followed by list containing grids of lower dimension in descending order.

    Parameters:
        g_2d: The highest-dimensional grid.
        fracs: A list of arrays with ``shape=(2, 2)``, representing the vertices of the
            line for each fracture.
            The fracture lines must align to the coordinate axis.
            The fractures will snap to the closest grid nodes.
        nx: ``shape=(2,)``

            Number of cells in each dimension.

    Returns:
        A nested list of length 3, where for each dimension 2 to 0 the respective
        sub-list contains all grids in that dimension.

    """
    g_2d.global_point_ind = np.arange(g_2d.num_nodes)
    g_2d.compute_geometry()
    g_1d: list[pp.Grid] = []
    g_0d: list[pp.Grid] = []

    # 1D grids:
    shared_nodes = np.zeros(g_2d.num_nodes)
    for fi, f in enumerate(fracs):
        is_x_frac = f[1, 0] == f[1, 1]
        is_y_frac = f[0, 0] == f[0, 1]
        assert is_x_frac != is_y_frac, "Fracture must align to x- or y-axis"
        if f.shape[0] == 2:
            f = np.vstack((f, np.zeros(f.shape[1])))
        nodes = _find_nodes_on_line(g_2d, nx, f[:, 0], f[:, 1])
        # nodes = np.unique(nodes)
        loc_coord = g_2d.nodes[:, nodes]
        g = msh_2_grid.create_embedded_line_grid(loc_coord, nodes)
        g.frac_num = fi
        g_1d.append(g)
        shared_nodes[nodes] += 1

    # Create 0-D grids
    if np.any(shared_nodes > 1):
        for global_node in np.argwhere(shared_nodes > 1).ravel():
            g = pp.PointGrid(g_2d.nodes[:, global_node])
            g.global_point_ind = np.asarray(global_node)
            g_0d.append(g)

    grids: list[list[pp.Grid]] = [[g_2d], g_1d, g_0d]
    return grids


def _create_embedded_2d_grid(loc_coord: np.ndarray, glob_id: np.ndarray) -> pp.Grid:
    """Creates a 2D grid that is embedded in a 3D grid.

    Parameters:
        loc_coords: ``shape=(2, np)``

            A cloud of 2D-points (column-wise) constituting the 2D Grid.
        glob_id: ``shape=(np,)``

            The global indexation of the passed ``np`` points, to be stored in the
            resulting grid.

    Returns:
        A 2D-grid based on above the specifications.

    """
    """Create a 2d grid that is embedded in a 3d grid."""

    # Center the points around the origin
    loc_center = np.mean(loc_coord, axis=1).reshape((-1, 1))
    loc_coord -= loc_center

    # Check that the points indeed form a line
    assert pp.geometry_property_checks.points_are_planar(loc_coord)

    # Map the points to a true 2d plane
    rot = pp.map_geometry.project_plane_matrix(loc_coord)
    loc_coord_2d = rot.dot(loc_coord)
    # The points are now 2d along two of the coordinate axis, but we don't know which
    # yet. Find this by identifying the coordinate axis which is numerical zero. This
    # requires a scaling with the fracture size to avoid numerical issues. Find the
    # size of the fracture by first finding the bounding box of the points.
    bounding_box: dict[str, float] = pp.domain.bounding_box_of_point_cloud(loc_coord)
    # The bounding box has keys 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'. The
    # list comprehension below loops over the min and max values of each coordinate
    # and finds the largest difference.
    fracture_size = np.max(
        [bounding_box[f"{s}max"] - bounding_box[f"{s}min"] for s in "xyz"]
    )
    # Compute the sum of the coordinates, scaled by the fracture size. Exactly two of
    # the coordinates should be non-zerozero.
    sum_coord = np.sum(np.abs(loc_coord_2d), axis=1) / fracture_size
    active_dimension = np.logical_not(np.isclose(sum_coord, 0))
    # Check that we are indeed in 2d
    assert np.sum(active_dimension) == 2

    # Sort nodes, and create grid
    coord_2d = loc_coord_2d[active_dimension]
    # Sort the indexes first by the first coordinate, then by the second
    sort_ind = np.lexsort((coord_2d[0], coord_2d[1]))
    sorted_coord = coord_2d[:, sort_ind]
    # EK: I have no idea what the next line does.
    sorted_coord = np.round(sorted_coord * 1e10) / 1e10
    unique_x = np.unique(sorted_coord[0])
    unique_y = np.unique(sorted_coord[1])
    g = pp.TensorGrid(unique_x, unique_y)
    assert np.all(g.nodes[0:2] - sorted_coord == 0)

    # Project back to active dimension
    nodes = np.zeros(g.nodes.shape)
    nodes[active_dimension] = g.nodes[0:2]
    g.nodes = nodes

    # Project back again to 3d coordinates
    irot = rot.transpose()
    g.nodes = irot.dot(g.nodes)
    g.nodes += loc_center

    # Add mapping to global point numbers
    g.global_point_ind = glob_id[sort_ind]
    return g


def _find_nodes_on_line(
    g: pp.Grid, nx: np.ndarray, s_pt: np.ndarray, e_pt: np.ndarray
) -> np.ndarray:
    """Find the nodes in a grid lying on a line specified by a start- and end-node.

    This function assumes a Cartesian structure on grid ``g``.

    Parameters:
        g: A Cartesian grid.
        nx: ``shape=(nd,)``

            An array containing the number of cells in each dimension, up to the
            dimension of the grid ``g`` (``nd``).
        s_pt: ``shape=(nd, 1)``

            The starting point of the line in dimension ``nd``.
        e_pt: ``shape=(nd, 1)``

            The end point of the line in dimension ``nd``.

    Returns:
        An array of points in ``g``, lying on the line defined by starting and
        end point.

    """
    s_node = np.argmin(pp.distances.point_pointset(s_pt, g.nodes))
    e_node = np.argmin(pp.distances.point_pointset(e_pt, g.nodes))

    # We make sure the nodes are ordered from low to high.
    if s_node > e_node:
        tmp = s_node
        s_node = e_node
        e_node = tmp
    # We now find the other grid nodes. We here use the node ordering of meshgrid (
    # which is used by the TensorGrid class).

    # We find the number of nodes along each dimension. From this we find the jump in
    # node number between two consecutive nodes.

    if np.all(np.isclose(s_pt[1:], e_pt[1:])):
        # x-line:
        nodes = np.arange(s_node, e_node + 1)
    elif np.all(np.isclose(s_pt[[0, 2]], e_pt[[0, 2]])):
        # y-line
        nodes = np.arange(s_node, e_node + 1, nx[0] + 1, dtype=int)
    elif nx.size == 3 and np.all(np.isclose(s_pt[0:2], e_pt[0:2])):
        # is z-line
        nodes = np.arange(s_node, e_node + 1, (nx[0] + 1) * (nx[1] + 1), dtype=int)
    else:
        raise RuntimeError("Something went wrong. Found a diagonal intersection")

    return nodes
