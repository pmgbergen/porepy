#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions for analysis of fracture networks. For the moment, 2d
only.
"""
import numpy as np
import networkx as nx

import porepy as pp
from examples.papers.flow_upscaling import segment_pixelation


def permeability_upscaling(network, data, mesh_args, directions, do_viz=True):

    gb = network.mesh(network.tol, mesh_args)
    directions = np.asarray(directions)
    upscaled_perm = np.zeros(directions.size)

    for di, direct in enumerate(directions):
        gb = _setup_simulation(gb, data, direct)

        key = "flow"
        discretization_key = key + "_" + pp.DISCRETIZATION

        mpfa = pp.Tpfa(key)
        for g, d in gb:
            d[pp.PRIMARY_VARIABLES] = {
                'pressure': {"cells": 1},
            }
            d[pp.DISCRETIZATION] = {
                'pressure': {"diffusive": pp.Tpfa('flow')},
            }
        coupler = pp.RobinCoupling('flow', pp.Tpfa('flow'))
        for e, d in gb.edges():
            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {'pressure': {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                'lambda': {
                    g1: ('pressure', "diffusive"),
                    g2: ('pressure', "diffusive"),
                    e: ('pressure', coupler),
                    }}

            d[discretization_key] = pp.RobinCoupling(key, mpfa)

        assembler = pp.Assembler()

        # Discretize
        A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
        p = np.linalg.solve(A.A, b)
        assembler.distribute_variable(gb, p, block_dof, full_dof)

        tot_inlet_flux = 0

        for g, d in gb:
            inlet = d.get('inlet_faces', None)
            if inlet is None or inlet.size == 0:
                continue
            flux = d[pp.DISCRETIZATION_MATRICES][key]['flux'] * d['pressure']
            flux += d[pp.DISCRETIZATION_MATRICES][key]['bound_flux'] * d['parameters'][key]['bc_values']
            tot_inlet_flux += flux[inlet].sum()
            if g.dim == gb.dim_max():
                dx = g.nodes[direct].max() - g.nodes[direct].min()
                area = (g.nodes[:g.dim].max(axis=1) - g.nodes[:g.dim].min(axis=1)).prod() / dx
        upscaled_perm[di] = tot_inlet_flux * dx / area

        if do_viz:
            exp = pp.Exporter(gb, 'direction_' + str(direct))
            exp.write_vtk('pressure')

    return upscaled_perm

def _setup_simulation(gb, data, direction):


    min_coord = gb.bounding_box()[0][direction]
    max_coord = gb.bounding_box()[1][direction]

    for g, d in gb:

        if g.dim == gb.dim_max():
            kxx = np.ones(g.num_cells)
        else:
            kxx = np.ones(g.num_cells) * data['fracture_perm']

        perm = pp.SecondOrderTensor(gb.dim_max(), kxx)
        a = data['aperture']
        a = np.power(a, gb.dim_max() - g.dim) * np.ones(g.num_cells)
        specified_parameters = {"aperture": a, "permeability": perm}

        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size > 0:
            hit_out = np.where(np.abs(g.face_centers[direction, bound_faces] - max_coord) < 1e-8)[0]
            hit_in = np.where(np.abs(g.face_centers[direction, bound_faces] - min_coord) < 1e-8)[0]
            bound_type = np.array(["neu"] * bound_faces.size)
            bound_type[hit_out] = 'dir'
            bound_type[hit_in] = 'dir'
            bound = pp.BoundaryCondition(
                g, bound_faces.ravel("F"), bound_type
            )
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = 0
            bc_val[bound_faces[hit_in]] = 1

            specified_parameters.update({"bc": bound, "bc_values": bc_val})

            d['inlet_faces'] = bound_faces[hit_in]

        pp.initialize_data(d, g, "flow", specified_parameters)

    for e, d in gb.edges():
        gl, _ = gb.nodes_of_edge(e)
        mg = d["mortar_grid"]
        kn = data['fracture_perm']
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], [{"normal_diffusivity": kn}])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    return gb


def connectivity_field(network, num_boxes):
    """ Compute the connectivity field associated with a fracture network.

    The method assumes that all points of the network are contained within the
    network's domain. This can be ensured by invoking network.constrain_to_domain().

    The method is motivated by the paper
        Connectivity field: A measure for characterizing fracture networks,
        by Alghalandis et al. Mathematical Geosciences 2015.

    Parameters:
        network (FractureSet): fracture network to be analyzed
        num_boxes (np.array, size 2): Number of bins to split the domain into in
            the x and y-direction, respectively.

    Returns:
        np.array, size num_boxes: For each cell in the Cartesian division of the
            domain, the number of other cells the cell is connected to

    """

    # Partition the domain
    _, _, dx, _ = cartesian_partition(network.domain, num_x=num_boxes[0], num_y=num_boxes[1])
    # Graph representation of the network
    graph, split_network = network.as_graph()

    num_clusters = len([sg for sg in nx.connected_components(graph)])

    # Field to store the presence of a network
    is_connected = np.zeros((num_clusters, num_boxes[0], num_boxes[1]))

    # Loop over all connected subgraphs of the network, identify connected
    # components
    for gi, sub_graph in enumerate(nx.connected_components(graph)):
        sg = graph.subgraph(sub_graph)
        loc_edges = np.array([[e[0], e[1]] for e in sg.edges()]).T
        # Use a pixelation algorithm to project fractures onto a Cartesian representation
        # of the domain
        pixelated = segment_pixelation.pixelate(split_network.pts, loc_edges, num_boxes, dx)
        is_connected[gi] = pixelated

    connectivity_field = np.zeros(num_boxes)
    for i in range(num_boxes[0]):
        for j in range(num_boxes[1]):
            hit = np.where(is_connected[:, i, j] > 0)[0]
            connectivity_field[i, j] = np.sum(is_connected[hit].sum(axis=0) > 0)

    # Binary division of connected and non-connected components
    return connectivity_field

def cartesian_partition(domain, num_x, num_y=None):
    """ Define a Cartesian partitioning of a domain.

    The domain could be 1d or 2d.

    Parameters:
        domain (dictionary): The domain in which the fracture set is defined.
            Should contain keys 'xmin', 'xmax', for 2d also 'ymin', 'ymax',
            each of which maps to a double giving the range of the domain.
        num_x (int): Number of bins in the x-direction
        num_y (int, optional): Number of bins in the y-direction. Only if the
            domain is 2d.

    Returns:
        double: Minimum x-coordinate of the domain
        double (optional): Minimum y-coordinate of the domain. Only if the domain
            is 2d.
        double: Spacing of the cells in the x-direction.
        double (optional): Spacing of the cells in the y-direction. Only if the
            domain is 2d.

    """
    x0 = domain['xmin']
    dx = (domain['xmax'] - domain['xmin']) / num_x

    if 'ymin' in domain.keys() and 'ymax' in domain.keys():
        y0 = domain['ymin']
        dy = (domain['ymax'] - domain['ymin']) / num_y
        return x0, y0, dx, dy
    else:
        return x0, dx

def analyze_intersections_of_sets(set_1, set_2=None, tol=1e-4):
    """ Count the number of node types (I, X, Y) per fracture in one or two
    fracture sets.

    The method finds, for each fracture, how many of the nodes are end-nodes,
    how many of the end-nodes abut to other fractures, and also how many other
    fractures crosses the main one in the form of an X or T intersection,
    respectively.

    Note that the fracture sets are treated as if they contain a single
    family, independent of any family tags in set_x.edges[2].

    To consider only intersections between fractures in different sets (e.g.
    disregard all intersections between fractures in the same family), run
    this function first with two input sets, then separately with a single set
    and take the difference.

    Parameters:
        set_1 (FractureSet): First set of fractures. Will be treated as a
            single family, independent of whether there are different family
            tags in set_1.edges[2].
        set_1 (FractureSet, optional): First set of fractures. Will be treated
            as a single family, independent of whether there are different
            family tags in set_1.edges[2]. If not provided,
        tol (double, optional): Tolerance used in computations to find
            intersections between fractures. Defaults to 1e-4.

    Returns:
        dictionary with keywords i_nodes, y_nodes, x_nodes, arrests. For each
            fracture in the set:
                i_nodes gives the number of the end-nodes of the fracture which
                    are i-nodes
                y_nodes gives the number of the end-nodes of the fracture which
                    terminate in another fracture
                x_nodes gives the number of X-intersections along the fracture
                arrests gives the number of fractures that terminates as a
                    Y-node in this fracture

        If two fracture sets are submitted, two such dictionaries will be
        returned, reporting on the fractures in the first and second set,
        respectively.

    """

    pts_1 = set_1.pts
    num_fracs_1 = set_1.edges.shape[1]

    num_pts_1 = pts_1.shape[1]

    # If a second set is present, also focus on the nodes in the intersections
    # between the two sets
    if set_2 is None:
        # The nodes are a sigle set
        pts = pts_1
        edges = np.vstack((set_1.edges[:2], np.arange(num_fracs_1, dtype=np.int)))

    else:
        # Assign famility based on the two sets, override whatever families
        # were assigned originally
        edges_1 = np.vstack((set_1.edges[:2], np.arange(num_fracs_1, dtype=np.int)))
        pts_2 = set_2.pts
        pts = np.hstack((pts_1, pts_2))

        num_fracs_2 = set_2.edges.shape[1]
        edges_2 = np.vstack((set_2.edges[:2], np.arange(num_fracs_2, dtype=np.int)))

        # The second set will have its points offset by the number of points
        # in the first set, and its edge numbering by the number of fractures
        # in the first set
        edges_2[:2] += num_pts_1
        edges_2[2] += num_fracs_1
        edges = np.hstack((edges_1, edges_2))

    num_fracs = edges.shape[1]

    _, e_split = pp.cg.remove_edge_crossings2(pts, edges, tol=tol, snap=False)

    # Find which of the split edges belong to family_1 and 2
    family_1 = np.isin(e_split[2], np.arange(num_fracs_1))
    if set_2 is not None:
        family_2 = np.isin(e_split[2], num_fracs_1 + np.arange(num_fracs_2))
    else:
        family_2 = np.logical_not(family_1)
    assert np.all(family_1 + family_2 == 1)

    # Story the fracture id of the split edges
    frac_id_split = e_split[2].copy()

    # Assign family identities to the split edges
    e_split[2, family_1] = 0
    e_split[2, family_2] = 1

    # For each fracture, identify its endpoints in terms of indices in the new
    # split nodes.
    end_pts = np.zeros((2, num_fracs))

    all_points_of_edge = np.empty(num_fracs, dtype=np.object)

    # Loop over all fractures
    for fi in range(num_fracs):
        # Find all split edges associated with the fracture, and its points
        loc_edges = frac_id_split == fi
        loc_pts = e_split[:2, loc_edges].ravel()

        # The endpoint ooccurs only once in this list
        loc_end_points = np.where(np.bincount(loc_pts) == 1)[0]
        assert loc_end_points.size == 2

        end_pts[0, fi] = loc_end_points[0]
        end_pts[1, fi] = loc_end_points[1]

        # Also store all nodes of this edge, including intersection points
        all_points_of_edge[fi] = np.unique(loc_pts)

    i_n, l_n, y_n_c, y_n_f, x_n = count_node_types_between_families(e_split)

    num_i_nodes = np.zeros(num_fracs)
    num_y_nodes = np.zeros(num_fracs)
    num_x_nodes = np.zeros(num_fracs)
    num_arrests = np.zeros(num_fracs)

    for fi in range(num_fracs):
        if set_2 is None:
            row = 0
            col = 0
        else:
            is_set_1 = fi < num_fracs_1
            if is_set_1:
                row = 0
                col = 1
            else:
                row = 1
                col = 0

        # Number of the endnodes that are y-nodes
        num_y_nodes[fi] = np.sum(np.isin(end_pts[:, fi], y_n_c[row, col]))

        # The number of I-nodes are 2 - the number of Y-nodes
        num_i_nodes[fi] = 2 - num_y_nodes[fi]

        # Number of nodes identified as x-nodes for this edge
        num_x_nodes[fi] = np.sum(np.isin(all_points_of_edge[fi], x_n[row, col]))

        # The number of fractures that have this edge as the constraint in the
        # T-node. This is are all nodes that are not end-nodes (2), and not
        # X-nodes
        num_arrests[fi] = all_points_of_edge[fi].size - num_x_nodes[fi] - 2

    if set_2 is None:
        results = {
            "i_nodes": num_i_nodes,
            "y_nodes": num_y_nodes,
            "x_nodes": num_x_nodes,
            "arrests": num_arrests,
        }
        return results
    else:
        results_set_1 = {
            "i_nodes": num_i_nodes[:num_fracs_1],
            "y_nodes": num_y_nodes[:num_fracs_1],
            "x_nodes": num_x_nodes[:num_fracs_1],
            "arrests": num_arrests[:num_fracs_1],
        }
        results_set_2 = {
            "i_nodes": num_i_nodes[num_fracs_1:],
            "y_nodes": num_y_nodes[num_fracs_1:],
            "x_nodes": num_x_nodes[num_fracs_1:],
            "arrests": num_arrests[num_fracs_1:],
        }
        return results_set_1, results_set_2


def count_node_types_between_families(e):
    """ Count the number of nodes (I, L, Y, X) between different fracture
    families.

    The fracutres are defined by their end-points (endpoints of branches should
    alse be fine).

    Parameters:
        e (np.array, 2 or 3 x n_frac): First two rows represent endpoints of
            fractures or branches. The third (optional) gives the family of
            each fracture. If this is not specified, the fractures are assumed
            to come from the same family.

    Returns:

        ** NB: For all returned matrices the family numbers are sorted, and
        rows and columns are defined accordingly.

        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all I-connections for the relevnant
            network. The main diagonal describes the i-nodes of the set
            considered by itself.
        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all L-connections for the relevnant
            networks. The main diagonal contains L-connection within the nework
            itself, off-diagonal elements represent the meeting between two
            different families. The elements [i, j] and [j, i] will be
            identical.
        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all Y-connections that were constrained
            by the other family. On the main diagonal, these are all the
            fractures. On the off-diagonal elments, element [i, j] contains
            all nodes where family i was constrained by family j.
        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all Y-connections that were not
            constrained by the other family. On the main diagonal, these are
            all the fractures. On the off-diagonal elments, element [i, j]
            contains all nodes where family j was constrained by family i.
        np.array (num_families x num_families): Each element contains a numpy
            array with the indexes of all X-connections for the relevnant
            networks. The main diagonal contains X-connection within the nework
            itself, off-diagonal elements represent the meeting between two
            different families. The elements [i, j] and [j, i] will be
            identical.

    """

    if e.shape[0] > 2:
        num_families = np.unique(e[2]).size
    else:
        num_families = 1
        e = np.vstack((e, np.zeros(e.shape[1], dtype=np.int)))

    # Nodes occuring only once. Hanging.
    i_nodes = np.empty((num_families, num_families), dtype=np.object)
    # Nodes occuring twice, defining an L-intersection, or equivalently the
    # meeting of two branches of a fracture
    l_nodes = np.empty_like(i_nodes)
    # Nodes in a Y-connection (or T-) that occurs twice. That is, the fracture
    # was not arrested by the crossing fracture.
    y_nodes_full = np.empty_like(i_nodes)
    # Nodes in a Y-connection (or T-) that occurs once. That is, the fracture
    # was arrested by the crossing fracture.
    y_nodes_constrained = np.empty_like(i_nodes)
    # Nodes in an X-connection.
    x_nodes = np.empty_like(i_nodes)

    max_p_ind = e[:2].max()

    # Ensure that all vectors are of the same size. Not sure if this is always
    # necessary, since we're doing an np.where later, but clearly this is useful
    def bincount(hit):
        tmp = np.bincount(e[:2, hit].ravel())
        num_occ = np.zeros(max_p_ind + 1, dtype=np.int)
        num_occ[: tmp.size] = tmp
        return num_occ

    # First do each family by itself
    families = np.sort(np.unique(e[2]))
    for i in families:
        hit = np.where(e[2] == i)[0]
        num_occ = bincount(hit)

        if np.any(num_occ > 4):
            raise ValueError("Not ready for more than two fractures meeting")

        i_nodes[i, i] = np.where(num_occ == 1)[0]
        l_nodes[i, i] = np.where(num_occ == 2)[0]
        y_nodes_full[i, i] = np.where(num_occ == 3)[0]
        y_nodes_constrained[i, i] = np.where(num_occ == 3)[0]
        x_nodes[i, i] = np.where(num_occ == 4)[0]

    # Next, compare two families

    for i in families:
        for j in families:
            if i == j:
                continue

            hit_i = np.where(e[2] == i)[0]
            num_occ_i = bincount(hit_i)
            hit_j = np.where(e[2] == j)[0]
            num_occ_j = bincount(hit_j)

            # I-nodes are not interesting in this setting (they will be
            # covered by the single-family case)

            hit_i_i = np.where(np.logical_and(num_occ_i == 1, num_occ_j == 0))[0]
            i_nodes[i, j] = hit_i_i
            hit_i_j = np.where(np.logical_and(num_occ_i == 0, num_occ_j == 1))[0]
            i_nodes[j, i] = hit_i_j

            # L-nodes between different families
            hit_l = np.where(np.logical_and(num_occ_i == 1, num_occ_j == 1))[0]
            l_nodes[i, j] = hit_l
            l_nodes[j, i] = hit_l

            # Two types of Y-nodes between different families
            hit_y = np.where(np.logical_and(num_occ_i == 1, num_occ_j == 2))[0]
            y_nodes_constrained[i, j] = hit_y
            y_nodes_full[j, i] = hit_y

            hit_y = np.where(np.logical_and(num_occ_i == 2, num_occ_j == 1))[0]
            y_nodes_constrained[j, i] = hit_y
            y_nodes_full[i, j] = hit_y

            hit_x = np.where(np.logical_and(num_occ_i == 2, num_occ_j == 2))[0]
            x_nodes[i, j] = hit_x
            x_nodes[j, i] = hit_x

    return i_nodes, l_nodes, y_nodes_constrained, y_nodes_full, x_nodes
