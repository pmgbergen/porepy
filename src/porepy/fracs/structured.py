"""
Module for creating fractured cartesian grids in 2- and 3-dimensions.

The functions in this module can be accesed through the meshing wrapper module.
"""
import numpy as np
import scipy.sparse as sps

from porepy.grids.gmsh import mesh_2_grid
from porepy.grids import constants
from porepy.fracs import fractures
from porepy.utils import half_space
from porepy.grids import structured, point_grid, constants
from porepy.utils import comp_geom as cg


def cart_grid_3d(fracs, nx, physdims=None):
    """
    Create grids for a domain with possibly intersecting fractures in 3d.

    Based on rectangles describing the individual fractures, the method
    constructs grids in 3d (the whole domain), 2d (one for each individual
    fracture), 1d (along fracture intersections), and 0d (meeting between
    intersections).

    Parameters
    ----------
    fracs (list of np.ndarray, each 3x4): Vertexes in the rectangle for each
        fracture. The vertices must be sorted and aligned to the axis.
        The fractures will snap to the closest grid faces.
    nx (np.ndarray): Number of cells in each direction. Should be 3D.
    physdims (np.ndarray): Physical dimensions in each direction.
        Defaults to same as nx, that is, cells of unit size.

    Returns
    -------
    list (length 4): For each dimension (3 -> 0), a list of all grids in
        that dimension.

    Examples
    --------
    frac1 = np.array([[1,1,4,4], [1,4,4,1], [2,2,2,2]])
    frac2 = np.array([[2,2,2,2], [1,1,4,4], [1,4,4,1]])
    fracs = [frac1, frac2]
    gb = cart_grid_3d(fracs, [5,5,5])
    """

    nx = np.asarray(nx)
    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != nx.size:
        raise ValueError("Physical dimension must equal grid dimension")
    else:
        physdims = np.asarray(physdims)

    # We create a 3D cartesian grid. The global node mapping is trivial.
    g_3d = structured.CartGrid(nx, physdims=physdims)
    g_3d.global_point_ind = np.arange(g_3d.num_nodes)
    g_3d.compute_geometry()
    g_2d = []
    g_1d = []
    g_0d = []
    # We set the tolerance for finding points in a plane. This can be any
    # small number, that is smaller than .25 of the cell sizes.
    tol = .1 * physdims / nx

    # Create 2D grids
    for fi, f in enumerate(fracs):
        assert np.all(f.shape == (3, 4)), "fractures must have shape [3,4]"
        is_xy_frac = np.allclose(f[2, 0], f[2])
        is_xz_frac = np.allclose(f[1, 0], f[1])
        is_yz_frac = np.allclose(f[0, 0], f[0])
        assert (
            is_xy_frac + is_xz_frac + is_yz_frac == 1
        ), "Fracture must align to x-, y- or z-axis"
        # snap to grid
        f_s = (
            np.round(f * nx[:, np.newaxis] / physdims[:, np.newaxis])
            * physdims[:, np.newaxis]
            / nx[:, np.newaxis]
        )
        if is_xy_frac:
            flat_dim = [2]
            active_dim = [0, 1]
        elif is_xz_frac:
            flat_dim = [1]
            active_dim = [0, 2]
        else:
            flat_dim = [0]
            active_dim = [1, 2]
        # construct normal vectors. If the rectangle is ordered
        # clockwise we need to flip the normals so they point
        # outwards.
        sign = 2 * cg.is_ccw_polygon(f_s[active_dim]) - 1
        tangent = f_s.take(np.arange(f_s.shape[1]) + 1, axis=1, mode="wrap") - f_s
        normal = tangent
        normal[active_dim] = tangent[active_dim[1::-1]]
        normal[active_dim[1]] = -normal[active_dim[1]]
        normal = sign * normal
        # We find all the faces inside the convex hull defined by the
        # rectangle. To find the faces on the fracture plane, we remove any
        # faces that are further than tol from the snapped fracture plane.
        in_hull = half_space.half_space_int(normal, f_s, g_3d.face_centers)
        f_tag = np.logical_and(
            in_hull,
            np.logical_and(
                f_s[flat_dim, 0] - tol[flat_dim] <= g_3d.face_centers[flat_dim],
                g_3d.face_centers[flat_dim] < f_s[flat_dim, 0] + tol[flat_dim],
            ),
        )
        f_tag = f_tag.ravel()
        nodes = sps.find(g_3d.face_nodes[:, f_tag])[0]
        nodes = np.unique(nodes)
        loc_coord = g_3d.nodes[:, nodes]
        g = _create_embedded_2d_grid(loc_coord, nodes)

        g.frac_num = fi
        g_2d.append(g)

    # Create 1D grids:
    # Here we make use of the network class to find the intersection of
    # fracture planes. We could maybe avoid this by doing something similar
    # as for the 2D-case, and count the number of faces belonging to each edge,
    # but we use the FractureNetwork class for now.
    frac_list = []
    for f in fracs:
        frac_list.append(fractures.Fracture(f))
    # Combine the fractures into a network
    network = fractures.FractureNetwork(frac_list)
    # Impose domain boundary. For the moment, the network should be immersed in
    # the domain, or else gmsh will complain.
    box = {
        "xmin": 0,
        "ymin": 0,
        "zmin": 0,
        "xmax": physdims[0],
        "ymax": physdims[1],
        "zmax": physdims[2],
    }
    network.impose_external_boundary(box)

    # Find intersections and split them.
    network.find_intersections()
    network.split_intersections()

    # Extract geometrical network information.
    pts = network.decomposition["points"]
    edges = network.decomposition["edges"]
    poly = network._poly_2_segment()
    # And tags identifying points and edges corresponding to normal
    # fractures, domain boundaries and subdomain boundaries. Only the
    # entities corresponding to normal fractures should actually be gridded.
    edge_tags, intersection_points = network._classify_edges(poly)
    const = constants.GmshConstants()
    auxiliary_points, edge_tags = network.on_domain_boundary(edges, edge_tags)
    bound_and_aux = np.array([const.DOMAIN_BOUNDARY_TAG, const.AUXILIARY_TAG])
    edges = np.vstack((edges, edge_tags))

    # Loop through the edges to make 1D grids. Ommit the auxiliary edges.
    for e in np.ravel(np.where(edges[2] == const.FRACTURE_INTERSECTION_LINE_TAG)):
        # We find the start and end point of each fracture intersection (1D
        # grid) and then the corresponding global node index.
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
        g = mesh_2_grid.create_embedded_line_grid(loc_coord, nodes)
        g_1d.append(g)

    # Create 0D grids
    # Here we also use the intersection information from the FractureNetwork
    # class. No grids for auxiliary points.
    for p in intersection_points:
        if auxiliary_points[p]:
            continue
        node = np.argmin(cg.dist_point_pointset(pts[:, p], g_3d.nodes))
        assert np.allclose(g_3d.nodes[:, node], pts[:, p])
        g = point_grid.PointGrid(g_3d.nodes[:, node])
        g.global_point_ind = np.asarray(node)
        g_0d.append(g)

    grids = [[g_3d], g_2d, g_1d, g_0d]
    return grids


def cart_grid_2d(fracs, nx, physdims=None):
    """
    Create grids for a domain with possibly intersecting fractures in 2d.

    Based on lines describing the individual fractures, the method
    constructs grids in 2d (whole domain), 1d (individual fracture), and 0d
    (fracture intersections).

    Parameters
    ----------
    fracs (list of np.ndarray, each 2x2): Vertexes of the line for each
        fracture. The fracture lines must align to the coordinat axis.
        The fractures will snap to the closest grid nodes.
    nx (np.ndarray): Number of cells in each direction. Should be 2D.
    physdims (np.ndarray): Physical dimensions in each direction.
        Defaults to same as nx, that is, cells of unit size.

    Returns
    -------
    list (length 3): For each dimension (2 -> 0), a list of all grids in
        that dimension.

    Examples
    --------
    frac1 = np.array([[1,4],[2,2]])
    frac2 = np.array([[2,2],[1,4]])
    fracs = [frac1,frac2]
    gb = cart_grid_2d(fracs, [5,5])
    """
    nx = np.asarray(nx)
    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != nx.size:
        raise ValueError("Physical dimension must equal grid dimension")
    else:
        physdims = np.asarray(physdims)

    g_2d = structured.CartGrid(nx, physdims=physdims)
    g_2d.global_point_ind = np.arange(g_2d.num_nodes)
    g_2d.compute_geometry()
    g_1d = []
    g_0d = []

    # 1D grids:
    shared_nodes = np.zeros(g_2d.num_nodes)
    for f in fracs:
        is_x_frac = f[1, 0] == f[1, 1]
        is_y_frac = f[0, 0] == f[0, 1]
        assert is_x_frac != is_y_frac, "Fracture must align to x- or y-axis"
        if f.shape[0] == 2:
            f = np.vstack((f, np.zeros(f.shape[1])))
        nodes = _find_nodes_on_line(g_2d, nx, f[:, 0], f[:, 1])
        # nodes = np.unique(nodes)
        loc_coord = g_2d.nodes[:, nodes]
        g = mesh_2_grid.create_embedded_line_grid(loc_coord, nodes)
        g_1d.append(g)
        shared_nodes[nodes] += 1

    # Create 0-D grids
    if np.any(shared_nodes > 1):
        for global_node in np.argwhere(shared_nodes > 1).ravel():
            g = point_grid.PointGrid(g_2d.nodes[:, global_node])
            g.global_point_ind = np.asarray(global_node)
            g_0d.append(g)

    grids = [[g_2d], g_1d, g_0d]
    return grids


def _create_embedded_2d_grid(loc_coord, glob_id):
    """
    Create a 2d grid that is embedded in a 3d grid.
    """
    loc_center = np.mean(loc_coord, axis=1).reshape((-1, 1))
    loc_coord -= loc_center
    # Check that the points indeed form a line
    assert cg.is_planar(loc_coord)
    # Find the tangent of the line
    # Projection matrix
    rot = cg.project_plane_matrix(loc_coord)
    loc_coord_2d = rot.dot(loc_coord)
    # The points are now 2d along two of the coordinate axis, but we
    # don't know which yet. Find this.
    sum_coord = np.sum(np.abs(loc_coord_2d), axis=1)
    active_dimension = np.logical_not(np.isclose(sum_coord, 0))
    # Check that we are indeed in 2d
    assert np.sum(active_dimension) == 2
    # Sort nodes, and create grid
    coord_2d = loc_coord_2d[active_dimension]
    sort_ind = np.lexsort((coord_2d[0], coord_2d[1]))
    sorted_coord = coord_2d[:, sort_ind]
    sorted_coord = np.round(sorted_coord * 1e10) / 1e10
    unique_x = np.unique(sorted_coord[0])
    unique_y = np.unique(sorted_coord[1])
    # assert unique_x.size == unique_y.size
    g = structured.TensorGrid(unique_x, unique_y)
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


def _find_nodes_on_line(g, nx, s_pt, e_pt):
    """
    We have the start and end point of the fracture. From this we find the 
    start and end node and use the structure of the cartesian grid to find
    the intermediate nodes.
    """
    s_node = np.argmin(cg.dist_point_pointset(s_pt, g.nodes))
    e_node = np.argmin(cg.dist_point_pointset(e_pt, g.nodes))

    # We make sure the nodes are ordered from low to high.
    if s_node > e_node:
        tmp = s_node
        s_node = e_node
        e_node = tmp
    # We now find the other grid nodes. We here use the node ordering of
    # meshgrid (which is used by the TensorGrid class).

    # We find the number of nodes along each dimension. From this we find the
    # jump in node number between two consecutive nodes.

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
