#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:34:52 2018

@author: eke001
"""
import numpy as np
import unittest
import scipy.stats as stats

import porepy as pp
from examples.papers.flow_upscaling.fracture_sets import FractureSet, ChildFractureSet
from examples.papers.flow_upscaling import frac_gen

class TestSingleSet(unittest.TestCase):

    def compute_angle(self, network):
        p_start = network.pts[:, network.edges[0]]
        p_end = network.pts[:, network.edges[1]]

        vector = p_end - p_start
        return np.arctan2(vector[1], vector[0])

    def test_angle_length(self):
        pass

class TestParentChildrenRelations(unittest.TestCase):

    def test_only_isolated_one_parent(self):

        p_parent = np.array([[0, 5], [0, 0]])
        e_parent = np.array([[0], [1]])

        domain = {'xmin': -1, 'xmax': 6, 'ymin': -1, 'ymax': 2}

        parent = FractureSet(p_parent, e_parent, domain)

        p_children = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                               [0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1]], dtype=np.float)

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)


        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.compute_statistics()
        isolated = child.isolated_stats

        # There should be one parent fracture
        self.assertTrue(isolated['density'].size == 1)
        # All four isolated belongs to this parent
        self.assertTrue(isolated['density'][0] == 4)
        # All four children have a center
        self.assertTrue(isolated['center_distance'].size == 4)
        # The distance from parent to child center is the midpoint of the child
        self.assertTrue(np.all(isolated['center_distance'] == 0.5 * (0.1 + p_children[1, 4:])))

    def test_only_isolated_two_parents_one_far_away(self):
        # Isolated children. Two parents, but only one has children

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {'xmin': -1, 'xmax': 6, 'ymin': -1, 'ymax': 2}

        parent = FractureSet(p_parent, e_parent, domain)

        p_children = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                               [0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1]], dtype=np.float)

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)


        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.compute_statistics()
        isolated = child.isolated_stats

        # There should be one parent fracture
        self.assertTrue(isolated['density'].size == 2)
        # All four isolated belongs to the first parent
        self.assertTrue(isolated['density'][0] == 4)
        self.assertTrue(isolated['density'][1] is None)
        # All four children have a center
        self.assertTrue(isolated['center_distance'].size == 4)
        # The distance from parent to child center is the midpoint of the child
        self.assertTrue(np.all(isolated['center_distance'] == 0.5 * (0.1 + p_children[1, 4:])))

    def test_only_isolated_two_parents_both_active(self):
        # Isolated children. Two active parents both have children

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {'xmin': -1, 'xmax': 6, 'ymin': -5, 'ymax': 2}

        parent = FractureSet(p_parent, e_parent, domain)

        # Children
        p_children = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                               [0.1, 0.6, 0.1, 0.1, 1, 1, 1, 1]], dtype=np.float)

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        # Move points of the second child to the negative y-axis
        p_children[1, [1, 5]] *= -1

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.compute_statistics()
        isolated = child.isolated_stats

        # There should be one parent fracture
        self.assertTrue(isolated['density'].size == 2)
        # All four isolated belongs to the first parent
        self.assertTrue(isolated['density'][0] == 3)
        self.assertTrue(isolated['density'][1] == 1)
        # All four children have a center
        self.assertTrue(isolated['center_distance'].size == 4)
        # The distance from parent to child center is the midpoint of the child
        self.assertTrue(np.all(isolated['center_distance'][[0, 2, 3]] == 0.5 * (p_children[1, [0, 2, 3]] + p_children[1, [4, 6, 7]])))
        self.assertTrue(np.all(isolated['center_distance'][1] == np.abs(-0.5 - (0.5 * (p_children[1, 1] + p_children[1, 5])))))

    def test_only_y_one_parent(self):
        # Only one parent. All children are y-intersections for this parent

        p_parent = np.array([[0, 5], [0, 0]])
        e_parent = np.array([[0], [1]])

        domain = {'xmin': -1, 'xmax': 6, 'ymin': -1, 'ymax': 2}

        parent = FractureSet(p_parent, e_parent, domain)

        p_children = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                               [0., 0., 0., 0., 1, 1, 1, 1]], dtype=np.float)

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.compute_statistics()
        one_y = child.one_y_stats

        # There should be one parent fracture
        self.assertTrue(one_y['density'].size == 1)
        # All four isolated belongs to this parent
        self.assertTrue(one_y['density'][0] == 4)

    def test_only_y_two_parents_one_active(self):
        # Two parents, only one is active. All children are y-intersections for this parent

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {'xmin': -1, 'xmax': 6, 'ymin': -1, 'ymax': 2}

        parent = FractureSet(p_parent, e_parent, domain)

        p_children = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                               [0., 0., 0., 0., 1, 1, 1, 1]], dtype=np.float)

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.compute_statistics()
        one_y = child.one_y_stats

        # There should be one parent fracture
        self.assertTrue(one_y['density'].size == 2)
        # All four isolated belongs to this parent
        self.assertTrue(one_y['density'][0] == 4)
        self.assertTrue(one_y['density'][1] == 0)

    def test_only_y_two_parents_two_active(self):
        # Two parents, only one is active. All children are y-intersections for this parent

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {'xmin': -1, 'xmax': 6, 'ymin': -1, 'ymax': 2}

        parent = FractureSet(p_parent, e_parent, domain)

        # Define children points. The edge p1-p5 (second child) will be associated
        # with the second parent
        p_children = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                               [0., -0.5, 0., 0., 1, 1, 1, 1]], dtype=np.float)

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        # Flip the far end of the second child, the other end is already defined
        # to be on the second parent
        p_children[1, 5] *= -1

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.compute_statistics()
        one_y = child.one_y_stats

        # There should be one parent fracture
        self.assertTrue(one_y['density'].size == 2)
        # All four isolated belongs to this parent
        self.assertTrue(one_y['density'][0] == 3)
        self.assertTrue(one_y['density'][1] == 1)

    def test_isolated_and_only_y_two_parents_two_active(self):
        # Two parents, only one is active. All children are y-intersections for this parent

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {'xmin': -1, 'xmax': 6, 'ymin': -1, 'ymax': 2}

        parent = FractureSet(p_parent, e_parent, domain)

        # Define children points. The edge p1-p5 (second child) will be associated
        # with the second parent as a y-node. The edge p2-p6 will be associated
        # with the second parent as an isolated node
        # The edges p0-p4 and p3-p7 will be closest to first parent, as
        # I and Y-nodes, respectively
        p_children = np.array([[1, 2, 3, 4, 1, 2, 3, 4],
                               [0.1, -0.5, -0.6, 0., 1, 1, 1, 1]], dtype=np.float)

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        # Flip the far end of the second and third child, the other end is already defined
        # to be on the second parent
        p_children[1, [5, 6]] *= -1

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.compute_statistics()
        isolated = child.isolated_stats
        one_y = child.one_y_stats

        self.assertTrue(isolated['density'].size == 2)
        self.assertTrue(isolated['density'][0] == 1)
        self.assertTrue(isolated['density'][1] == 1)

        self.assertTrue(isolated['center_distance'].size == 2)
        self.assertTrue(np.allclose(isolated['center_distance'][0],
                                    np.array([0.5 * p_children[1, [0, 4]].sum()])))
        self.assertTrue(np.allclose(isolated['center_distance'][1],
                                    -0.5 - 0.5 * p_children[1, [2, 6]].sum()))

        self.assertTrue(np.all(child.isolated == np.array([0, 2])))

        # Y-nodes
        self.assertTrue(one_y['density'].size == 2)
        self.assertTrue(one_y['density'][0] == 1)
        self.assertTrue(one_y['density'][1] == 1)



class TestDensityCounting(unittest.TestCase):

    def test_1d_counting_single_box(self):
        domain = {'xmin': 0, 'xmax': 1}
        p = np.array([0.4])
        e = np.array([[0], [0]])

        num_occ = frac_gen.count_center_point_densities(p, e, domain, nx=1)
        self.assertTrue(num_occ.size == 1)
        self.assertTrue(num_occ[0] == 1)

    def test_1d_counting_no_points(self):
        domain = {'xmin': 0, 'xmax': 1}
        p = np.empty((1, 0))
        e = np.array([[0], [0]])

        num_occ = frac_gen.count_center_point_densities(p, e, domain, nx=1)
        self.assertTrue(num_occ.size == 1)
        self.assertTrue(num_occ[0] == 0)

    def test_1d_counting_two_boxes(self):
        domain = {'xmin': 0, 'xmax': 1}
        p = np.array([0.4])
        e = np.array([[0], [0]])

        num_occ = frac_gen.count_center_point_densities(p, e, domain, nx=2)
        self.assertTrue(num_occ.size == 2)
        self.assertTrue(np.all(num_occ == np.array([1, 0])))

#if __name__ == '__main__':
#    unittest.main()
#TestParentChildrenRelations().test_isolated_and_only_y_two_parents_two_active()
unittest.main()
#TestDensityCounting().test_1d_counting_two_boxes()