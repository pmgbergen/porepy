#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:54:36 2018

@author: Eirik Keilegavlens
"""
import numpy as np
import scipy.stats as stats

from examples.papers.flow_upscaling import frac_gen
import porepy as pp


class FractureSet(object):

    def __init__(self, pts, edges, domain):

        self.pts = pts
        self.edges = edges
        self.domain = domain

        self.num_frac = self.edges.shape[1]

    def fit_distributions(self, **kwargs):
        self.fit_length_distribution(**kwargs)
        self.fit_angle_distribution(**kwargs)
        self.fit_intensity_map(**kwargs)

    def fit_length_distribution(self, **kwargs):
        self.dist_length = frac_gen.fit_length_distribution(self.pts, self.edges,
                                                           **kwargs)

    def fit_angle_distribution(self, **kwargs):
        self.dist_angle = frac_gen.fit_angle_distribution(self.pts, self.edges,
                                                           **kwargs)

    def fit_intensity_map(self, **kwargs):
        self.intensity = frac_gen.count_center_point_densities(self.pts, self.edges, self.domain, **kwargs)

    def populate(self, domain=None, fit_distributions=True, **kwargs):
        if domain is None:
            domain = self.domain

        if fit_distributions:
            self.fit_distributions()

        # First define points
        p = frac_gen.define_centers_by_boxes(domain, self.intensity)
        # bookkeeping
        num_fracs = p.shape[1]

        # Then assign length and orientation
        angles = frac_gen.generate_from_distribution(num_fracs, self.dist_angle)
        lengths = frac_gen.generate_from_distribution(num_fracs, self.dist_angle)

        p, e = frac_gen.fracture_from_center_angle_length(p, angles, lengths)

        return FractureSet(p, e, domain)

    def plot(self, **kwargs):
        """ Plote the fracture set.

        The function passes this fracture set to PorePy plot_fractures

        Parameters:
            **kwargs: Keyword arguments to be passed on to matplotlib.

        """
        pp.plot_fractures(self.domain, self.pts, self.edges)


class ChildFractureSet(FractureSet):
    """ Fracture set that is defined based on its distance from a member of
    a parent family
    """

    def __init__(self, pts, edges, domain, parent):
        super(ChildFractureSet, self).__init__(pts, edges, domain)
        self.parent = parent

    def compute_statistics(self, **kwargs):

        # NOTE: Isolated nodes for the moment does not rule out that the child
        # intersects with a parent

        # Angle and length distribution as usual
        self.fit_angle_distribution(**kwargs)
        self.fit_length_distribution(**kwargs)

        node_types_self = analyze_intersections_of_sets(self, **kwargs)
        node_types_combined_self, node_types_combined_parent = analyze_intersections_of_sets(self, self.parent, **kwargs)

        # Find the number of Y-nodes that terminates in a parent node
        y_nodes_in_parent = node_types_combined_self['y_nodes'] - node_types_self['y_nodes']

        # Fractures that ends in a parent fracture on both sides. If this is
        # a high number (whatever that means) relative to the total number of
        # fractures in this set, we may be better off by describing the set as
        # constrained
        both_y = np.where(y_nodes_in_parent == 2)[0]

        one_y = np.where(y_nodes_in_parent == 1)[0]

        isolated = np.where(node_types_combined_self["i_nodes"] == 2)[0]

        self.isolated = isolated
        self.one_y = one_y
        self.both_y = both_y

        # Find the number of isolated fractures that cross a parent fracture.
        # Not sure how we will use this
        x_nodes_with_parent = node_types_combined_self['x_nodes'] - node_types_self['x_nodes']
        intersections_of_isolated_nodes = x_nodes_with_parent[isolated]

        # Start and end points of the parent fractures

        # Treat isolated nodes
        if isolated.size > 0:
            density, center_distance = self.compute_line_density_isolated_nodes(isolated)
            self.isolated_stats = {'density': density, 'center_distance': center_distance}

        ## fractures that have one Y-intersection with a parent
        # First, identify the parent-child relation
        if one_y.size > 0:
            density = self.compute_line_density_one_y_node(one_y)
            self.one_y_stats = {'density': density}


    def compute_line_density_one_y_node(self, one_y):
        num_one_y = one_y.size

        start_parent = self.parent.pts[:, self.parent.edges[0]]
        end_parent = self.parent.pts[:, self.parent.edges[1]]

        start_y = self.pts[:, self.edges[0, one_y]]
        end_y = self.pts[:, self.edges[1, one_y]]

        # Compute the distance from the start and end point of the children
        # to all parents
        # dist_start will here have dimensions num_children x num_parents
        # closest_pt_start has dimensions num_children x num_parents x dim (2)
        dist_start, closest_pt_start = pp.cg.dist_points_segments(start_y, start_parent, end_parent)
        dist_end, closest_pt_end = pp.cg.dist_points_segments(end_y, start_parent, end_parent)

        # For each child, identify which parent is the closest, and consider
        # only that distance and point
        closest_parent_start = np.argmin(dist_start, axis=1)
        dist_start = dist_start[np.arange(num_one_y), closest_parent_start]
        closest_pt_start = closest_pt_start[np.arange(num_one_y), closest_parent_start, :].T
        # Then the end points
        closest_parent_end = np.argmin(dist_end, axis=1)
        dist_end = dist_end[np.arange(num_one_y), closest_parent_end]
        closest_pt_end = closest_pt_end[np.arange(num_one_y), closest_parent_end, :].T


        # Exactly one of the children end point should be on a parent
        # The tolerance used here is arbitrary.
        assert np.all(np.logical_or(dist_start < 1e-4, dist_end < 1e-4))

        start_closest = dist_start < dist_end

        num_parent = self.parent.num_frac
        num_occ_all = np.empty(num_parent, dtype=np.object)

        for fi in range(num_parent):
            hit_start = np.logical_and(start_closest, closest_parent_start == fi)
            start_point_loc = closest_pt_start[:, hit_start]
            hit_end = np.logical_and(np.logical_not(start_closest), closest_parent_end == fi)
            end_point_loc = closest_pt_end[:, hit_end]
            p_loc = np.hstack((start_point_loc, end_point_loc))
            num_occ_all[fi] = self.compute_density_along_line(p_loc, start_parent[:, fi],
                       end_parent[:, fi], nx=1)

        return num_occ_all


    def compute_line_density_isolated_nodes(self, isolated):
        # To ultimately describe the isolated fractures as a marked point
        # process, with stochastic location in terms of its distribution along
        # the fracture and perpendicular to it, we describe the distance from
        # the child center to its parent line.

        # There may be some issues with accumulation close to the end points
        # of the parent fracture; in particular if the orientation of the
        # child is far from perpendicular (meaning that a lot of it is outside
        # the 'span' of the parent), or if multiple parents are located nearby,
        # and we end up taking the distance to one that is not the natural
        # parent, whatever that means.
        # This all seems to confirm that ideall, a unique parent should be
        # identified for all children.

                # Start and end points of the parent fractures
        start_parent = self.parent.pts[:, self.parent.edges[0]]
        end_parent = self.parent.pts[:, self.parent.edges[1]]

        center_of_isolated = 0.5 * (self.pts[:, self.edges[0, isolated]] +
                                        self.pts[:, self.edges[1, isolated]])
        dist_isolated, closest_pt_isolated = pp.cg.dist_points_segments(center_of_isolated, start_parent, end_parent)

        # Minimum distance from center to a fracture
        num_isolated = isolated.size
        min_dist = np.min(dist_isolated, axis=1)
        closest_parent_isolated = np.argmin(dist_isolated, axis=1)


        def dist_pt(a, b):
            return(np.sqrt(np.sum((a - b)**2, axis=0)))

        num_isolated = isolated.size

        # Distance from center of isolated node to the fracture (*not* its
        # prolongation). This will have some statistical distribution
        points_on_line = closest_pt_isolated[np.arange(num_isolated), closest_parent_isolated].T
        pert_dist_isolated = dist_pt(center_of_isolated, points_on_line)

        num_occ_all = np.empty(self.parent.edges.shape[1], dtype=np.object)

        # Loop over all parent fractures that are closest to some children.
        # Project the children onto the parent, compute a density map along
        # the parent.
        for counter, fi in enumerate(np.unique(closest_parent_isolated)):
            hit = np.where(closest_parent_isolated == fi)[0]
            p_loc = points_on_line[:, hit]
            num_occ_all[fi] = self.compute_density_along_line(p_loc, start_parent[:, fi],
                       end_parent[:, fi], nx=1)

        return num_occ_all, pert_dist_isolated

    def compute_density_along_line(self, p, start, end, **kwargs):

        if p.ndim == 1:
            p = p.reshape((-1, 1))
        if start.ndim == 1:
            start = start.reshape((-1, 1))
        if end.ndim == 1:
            end = end.reshape((-1, 1))

        def _to_3d(pt):
            return np.vstack((pt, np.zeros(pt.shape[1])))

        p -= start
        end -= start
        theta = np.arctan2(end[1], end[0])

        assert np.abs(end[0] * np.sin(theta) + end[1] * np.cos(theta)) < 1e-5

        start_x = 0
        p_x = p[0] * np.cos(theta) - p[1] * np.sin(theta)
        end_x = end[0] * np.cos(theta) - end[1] * np.sin(theta)

        if end_x < start_x:
            domain_loc = {'xmin': end_x, 'xmax': start_x}
        else:
            domain_loc = {'xmin': start_x, 'xmax': end_x}

        # The density calculation computes the center of each fracture,
        # based on an assumption that the fracture consist of two points.
        # Make a line out of the points, with identical start and end points
        loc_edge = np.tile(np.arange(p_x.size), (2, 1))

        # Count the point density along this fracture.
        return frac_gen.count_center_point_densities(p_x, loc_edge, domain_loc, **kwargs)


class ConstrainedFractureSet(FractureSet):
    """ Fracture set that is constrained on both sides by a parent. Will have
    an orientation and a position distribution, but not a length (although we
    may say that if two parent fractures are too far aparnt, they can have no
    children of this type)
    """

    def __init__(self, pts, edges, domain, constraining_set):
        pass


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

    _, e_split = pp.cg.remove_edge_crossings(pts, edges, tol=tol, snap=False)

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
        results = {'i_nodes': num_i_nodes, 'y_nodes': num_y_nodes,
                   'x_nodes': num_x_nodes, 'arrests': num_arrests}
        return results
    else:
        results_set_1 = {'i_nodes': num_i_nodes[:num_fracs_1],
                         'y_nodes': num_y_nodes[:num_fracs_1],
                         'x_nodes': num_x_nodes[:num_fracs_1],
                         'arrests': num_arrests[:num_fracs_1]}
        results_set_2 = {'i_nodes': num_i_nodes[num_fracs_1:],
                         'y_nodes': num_y_nodes[num_fracs_1:],
                         'x_nodes': num_x_nodes[num_fracs_1:],
                         'arrests': num_arrests[num_fracs_1:]}
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
        num_occ = np.zeros(max_p_ind+1, dtype=np.int)
        num_occ[:tmp.size] = tmp
        return num_occ

    # First do each family by itself
    families = np.sort(np.unique(e[2]))
    for i in families:
        hit = np.where(e[2] == i)[0]
        num_occ = bincount(hit)

        if np.any(num_occ > 4):
            raise ValueError('Not ready for more than two fractures meeting')

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
