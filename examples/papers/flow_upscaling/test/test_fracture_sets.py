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


class TestFractureSetGeneration(unittest.TestCase):
    def atest_set_distributions_run_population_single_family(self):
        # Define a small fracture set, set distributions, and use this to
        # generate a set
        # The test is intended that

        p = np.array([[0, 5], [0, 0]])
        e = np.array([[0], [1]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        original = FractureSet(p, e, domain)

        # Set the random seed
        np.random.seed(0)

        # Lognormal distribution for length
        original.set_length_distribution(stats.lognorm, (0.5, -1, 2))
        original.set_angle_distribution(stats.uniform, (0, 1))

        original.set_intensity_map(np.array([[1]]))

        realiz = original.generate(fit_distributions=False)

        # Hard-coded points
        known_points = np.array(
            [
                [3.00394785, 2.62441671, 1.66490881, 2.26625838],
                [0.91037159, 0.96499309, 0.57025301, 0.05527026],
            ]
        )
        known_edges = np.array([[0, 2], [1, 3]])

        self.assertTrue(np.allclose(realiz.pts, known_points))
        self.assertTrue(np.allclose(realiz.edges, known_edges))

    def atest_draw_children_type(self):
        """ Check that the children type is drawn correctly
        """
        p = np.array([[0, 5], [0, 0]])
        e = np.array([[0], [1]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        frac_set = ChildFractureSet(p, e, domain, None)

        # First force only i-nodes
        frac_set.fraction_isolated = 1
        frac_set.fraction_one_y = 0
        frac_set.fraction_both_y = 0

        i, y, by = frac_set._draw_children_type(10)
        self.assertTrue(np.all(i))
        self.assertTrue(np.all(np.logical_not(y)))
        self.assertTrue(np.all(np.logical_not(by)))

        # Then only single y-nodes
        frac_set.fraction_isolated = 0
        frac_set.fraction_one_y = 1
        frac_set.fraction_both_y = 0

        i, y, by = frac_set._draw_children_type(10)
        self.assertTrue(np.all(np.logical_not(i)))
        self.assertTrue(np.all(y))
        self.assertTrue(np.all(np.logical_not(by)))

        # Then only double y-nodes
        frac_set.fraction_isolated = 0
        frac_set.fraction_one_y = 0
        frac_set.fraction_both_y = 1

        i, y, by = frac_set._draw_children_type(10)
        self.assertTrue(np.all(np.logical_not(i)))
        self.assertTrue(np.all(np.logical_not(y)))
        self.assertTrue(np.all(by))

    def atest_one_parent_all_i_children(self):
        # Define population methods and statistical distribution for a network
        # where all the children are isolated
        p = np.array([[0, 5], [0, 0]])
        e = np.array([[0], [1]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        original_parent = FractureSet(p, e, domain)

        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, original_parent)

        # Force all parents to have exactly three children
        child.fraction_of_parents_with_child = 1
        child.dist_num_children = make_dummy_distribution(3/5)

        # All children will be isolated
        child.fraction_isolated = 1
        child.fraction_one_y = 0

        # All fractures have distance from parents of 2
        child.dist_from_parents = {"dist": stats.randint(low=2, high=3), "param": {}}

        # All fractures have the same length
        known_length = np.random.rand(1)
        child.dist_length = make_dummy_distribution(known_length)

        # All fractures have the same angle
        angle = np.pi / 2 * np.random.rand(1)
        child.dist_angle = make_dummy_distribution(angle)

        realiz = child.generate(original_parent)

        p = realiz.pts
        e = realiz.edges
        self.assertTrue(p.shape[1] == 6)
        self.assertTrue(e.shape[1] == 3)

        # All generated fractures should have length 1
        dx = p[:, e[1]] - p[:, e[0]]
        length = np.sqrt(np.sum(dx ** 2, axis=0))
        self.assertTrue(np.allclose(length, known_length))

        # The end points of the fracture are shifted half the length * sin of the angle
        # compared with the known center point
        # We need to take the absolute value of the y-coordinate, since the child
        # make be put on either side of the center
        dy = 0.5 * known_length * np.sin(angle)
        self.assertTrue(np.sum(np.abs(np.abs(p[1]) - (2 + dy)) < 1e-3) == 3)
        self.assertTrue(np.sum(np.abs(np.abs(p[1]) - (2 - dy)) < 1e-3) == 3)

    def atest_one_parent_all_one_y_children(self):
        # Define population methods and statistical distribution for a network
        # where all the children have a single y-node
        p = np.array([[0, 5], [0, 0]])
        e = np.array([[0], [1]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        original_parent = FractureSet(p, e, domain)

        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0., 0., 0., 0., 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, original_parent)

        # Force all parents to have exactly three children
        child.fraction_of_parents_with_child = 1
        child.dist_num_children = make_dummy_distribution(3./5)

        # All children will be isolated
        child.fraction_isolated = 0
        child.fraction_one_y = 1

        # All fractures have distance from parents of 2
        child.dist_from_parents = {"dist": stats.randint(low=2, high=3), "param": {}}

        # All fractures have the same length
        known_length = np.random.rand(1)
        child.dist_length = make_dummy_distribution(known_length)

        # All fractures have the same angle
        angle = np.pi / 2 * np.random.rand(1)
        child.dist_angle = make_dummy_distribution(angle)

        realiz = child.generate(original_parent)

        p = realiz.pts
        e = realiz.edges
        self.assertTrue(p.shape[1] == 6)
        self.assertTrue(e.shape[1] == 3)

        # All generated fractures should have length 1
        dx = p[:, e[1]] - p[:, e[0]]
        length = np.sqrt(np.sum(dx ** 2, axis=0))
        self.assertTrue(np.allclose(length, known_length))

        # The end points of the fracture are shifted half the length * sin of the angle
        # compared with the known center point
        # We need to take the absolute value of the y-coordinate, since the child
        # make be put on either side of the center
        dy = known_length * np.sin(angle)
        self.assertTrue(np.sum(np.abs(np.abs(p[1]) - dy) < 1e-3) == 3)
        self.assertTrue(np.sum(np.abs(p[1])  < 1e-3) == 3)

    def atest_one_parent_all_both_y_children(self):
        # Only one constraint, no fractures should be generated
        p = np.array([[0, 5], [0, 0]])
        e = np.array([[0], [1]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        original_parent = FractureSet(p, e, domain)

        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0., 0., 0., 0., 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, original_parent)

        # Force all parents to have exactly three children
        child.fraction_of_parents_with_child = 1
        child.dist_num_children = make_dummy_distribution(3)

        # All children will be isolated
        child.fraction_isolated = 0
        child.fraction_one_y = 0

        # All fractures have distance from parents of 2
        child.dist_from_parents = {"dist": stats.randint(low=2, high=3), "param": {}}
        # Assign long fractures
        child.dist_max_constrained_length = make_dummy_distribution(5)

        # All fractures have the same length
        known_length = np.random.rand(1)
        child.dist_length = make_dummy_distribution(known_length)

        # All fractures have the same angle
        angle = np.pi / 2 * np.random.rand(1)
        child.dist_angle = make_dummy_distribution(angle)

        realiz = child.generate(original_parent)

        p = realiz.pts
        e = realiz.edges
        # There should be no points in the generated child
        self.assertTrue(p.size == 0)
        self.assertTrue(e.size == 0)

    def atest_two_parents_all_both_y_children(self):
        # Only one constraint, no fractures should be generated
        p = np.array([[0, 5, 0, 5], [0, 0, 1, 1]])
        e = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        original_parent = FractureSet(p, e, domain)

        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0., 0., 0., 0., 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, original_parent)

        # Force all parents to have exactly three children
        child.fraction_of_parents_with_child = 1
        child.dist_num_children = make_dummy_distribution(10)

        # All children will be isolated
        child.fraction_isolated = 0
        child.fraction_one_y = 0

        # All fractures have distance from parents of 2
        child.dist_from_parents = {"dist": stats.randint(low=2, high=3), "param": {}}
        # Assign long fractures
        child.dist_max_constrained_length = make_dummy_distribution(5)

        # All fractures have the same length
        known_length = np.random.rand(1)
        child.dist_length = make_dummy_distribution(known_length)

        # All fractures have the same angle
        angle = np.pi / 2 * np.random.rand(1)
        child.dist_angle = make_dummy_distribution(angle)

        realiz = child.generate(original_parent)

        p = realiz.pts
        e = realiz.edges
        # There should be no points in the generated child
        self.assertTrue(np.all(np.logical_or(np.abs(p[1]) < 1e-3, np.abs(p[1] - 1) < 1e-3)))

        self.assertTrue(np.allclose(realiz.length(), realiz.length().mean()))

    def atest_two_parents_one_far_away_all_both_y_children(self):
        # Only one constraint, no fractures should be generated
        p = np.array([[0, 5, 0, 5], [0, 0, 10, 10]])
        e = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        original_parent = FractureSet(p, e, domain)

        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0., 0., 0., 0., 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, original_parent)

        # Force all parents to have exactly three children
        child.fraction_of_parents_with_child = 1
        child.dist_num_children = make_dummy_distribution(10)
        child.points_along_fracture = 'random'
        child.dist_side = {'dist': stats.randint, 'param': {'low': 1, 'high': 2}}

        # All children will be isolated
        child.fraction_isolated = 0
        child.fraction_one_y = 0

        # All fractures have distance from parents of 2
        child.dist_from_parents = {"dist": stats.randint(low=2, high=3), "param": {}}
        # Assign fractures much shorter than the distance between the parents
        child.dist_max_constrained_length = make_dummy_distribution(5)

        # All fractures have the same length
        known_length = np.random.rand(1)
        child.dist_length = make_dummy_distribution(known_length)

        # All fractures have the same angle
        angle = np.pi / 2 * np.random.rand(1)
        child.dist_angle = make_dummy_distribution(angle)

        realiz = child.generate(original_parent, y_separately=True)

        p = realiz.pts
        e = realiz.edges
        # There should be no points in the generated child
        self.assertTrue(p.size == 0)
        self.assertTrue(e.size == 0)


class TestParentChildrenRelations(unittest.TestCase):
    def atest_only_isolated_one_parent(self):

        p_parent = np.array([[0, 5], [0, 0]])
        e_parent = np.array([[0], [1]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)

        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.fit_distributions()
        isolated = child.isolated_stats

        # There should be one parent fracture
        self.assertTrue(isolated["density"].size == 1)
        # All four isolated belongs to this parent
        self.assertTrue(isolated["density"][0] * parent.length()[0] == 4)
        # All four children have a center
        self.assertTrue(isolated["center_distance"].size == 4)
        # The distance from parent to child center is the midpoint of the child
        self.assertTrue(
            np.all(isolated["center_distance"] == 0.5 * (0.1 + p_children[1, 4:]))
        )

    def atest_only_isolated_two_parents_one_far_away(self):
        # Isolated children. Two parents, but only one has children

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)
        parent_length = parent.length()
        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.fit_distributions()
        isolated = child.isolated_stats

        # There should be one parent fracture
        self.assertTrue(isolated["density"].size == 2)
        # All four isolated belongs to the first parent
        self.assertTrue(isolated["density"][0] * parent_length[0] == 4)
        self.assertTrue(isolated["density"][1] * parent_length[1] == 0)
        # All four children have a center
        self.assertTrue(isolated["center_distance"].size == 4)
        # The distance from parent to child center is the midpoint of the child
        self.assertTrue(
            np.all(isolated["center_distance"] == 0.5 * (0.1 + p_children[1, 4:]))
        )

    def atest_only_isolated_two_parents_both_active(self):
        # Isolated children. Two active parents both have children

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -5, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)
        parent_length = parent.length()
        # Children
        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0.1, 0.6, 0.1, 0.1, 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        # Move points of the second child to the negative y-axis
        p_children[1, [1, 5]] *= -1

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.fit_distributions()
        isolated = child.isolated_stats

        # There should be one parent fracture
        self.assertTrue(isolated["density"].size == 2)
        # All four isolated belongs to the first parent
        self.assertTrue(isolated["density"][0] * parent_length[0] == 3)
        self.assertTrue(isolated["density"][1] * parent_length[1] == 1)
        # All four children have a center
        self.assertTrue(isolated["center_distance"].size == 4)
        # The distance from parent to child center is the midpoint of the child
        self.assertTrue(
            np.all(
                isolated["center_distance"][[0, 2, 3]]
                == 0.5 * (p_children[1, [0, 2, 3]] + p_children[1, [4, 6, 7]])
            )
        )
        self.assertTrue(
            np.all(
                isolated["center_distance"][1]
                == np.abs(-0.5 - (0.5 * (p_children[1, 1] + p_children[1, 5])))
            )
        )

    def atest_only_y_one_parent(self):
        # Only one parent. All children are y-intersections for this parent

        p_parent = np.array([[0, 5], [0, 0]])
        e_parent = np.array([[0], [1]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)

        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0., 0., 0., 0., 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.fit_distributions()
        one_y = child.one_y_stats

        # There should be one parent fracture
        self.assertTrue(one_y["density"].size == 1)
        # All four isolated belongs to this parent
        self.assertTrue(one_y["density"][0] * parent.length()[0] == 4)

    def atest_only_y_two_parents_one_active(self):
        # Two parents, only one is active. All children are y-intersections for this parent

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)

        parent_length = parent.length()[0]

        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0., 0., 0., 0., 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.fit_distributions()
        one_y = child.one_y_stats

        # There should be one parent fracture
        self.assertTrue(one_y["density"].size == 2)
        # All four isolated belongs to this parent
        self.assertTrue(one_y["density"][0] * parent_length == 4)
        self.assertTrue(one_y["density"][1] == 0)

    def atest_only_y_two_parents_two_active(self):
        # Two parents, only one is active. All children are y-intersections for this parent

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)
        parent_length = parent.length()
        # Define children points. The edge p1-p5 (second child) will be associated
        # with the second parent
        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0., -0.5, 0., 0., 1, 1, 1, 1]], dtype=np.float
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        # Flip the far end of the second child, the other end is already defined
        # to be on the second parent
        p_children[1, 5] *= -1

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.fit_distributions()
        one_y = child.one_y_stats

        # There should be one parent fracture
        self.assertTrue(one_y["density"].size == 2)
        # All four isolated belongs to this parent
        self.assertTrue(one_y["density"][0] * parent_length[0] == 3)
        self.assertTrue(one_y["density"][1] * parent_length[1] == 1)

    def atest_isolated_and_only_y_two_parents_two_active(self):
        # Two parents, only one is active. All children are y-intersections for this parent

        p_parent = np.array([[0, 5, 0, 5], [0, 0, -0.5, -0.5]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)
        parent_length = parent.length()
        # Define children points. The edge p1-p5 (second child) will be associated
        # with the second parent as a y-node. The edge p2-p6 will be associated
        # with the second parent as an isolated node
        # The edges p0-p4 and p3-p7 will be closest to first parent, as
        # I and Y-nodes, respectively
        p_children = np.array(
            [[1, 2, 3, 4, 1, 2, 3, 4], [0.1, -0.5, -0.6, 0., 1, 1, 1, 1]],
            dtype=np.float,
        )

        # Children length from a lognormal distribution
        p_children[1, [4, 5, 6, 7]] += stats.lognorm.rvs(s=1, size=4)

        # Flip the far end of the second and third child, the other end is already defined
        # to be on the second parent
        p_children[1, [5, 6]] *= -1

        e_children = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        child = ChildFractureSet(p_children, e_children, domain, parent)

        child.fit_distributions()
        isolated = child.isolated_stats
        one_y = child.one_y_stats

        self.assertTrue(isolated["density"].size == 2)
        self.assertTrue(isolated["density"][0] * parent_length[0] == 1)
        self.assertTrue(isolated["density"][1] * parent_length[1] == 1)

        self.assertTrue(isolated["center_distance"].size == 2)
        self.assertTrue(
            np.allclose(
                isolated["center_distance"][0],
                np.array([0.5 * p_children[1, [0, 4]].sum()]),
            )
        )
        self.assertTrue(
            np.allclose(
                isolated["center_distance"][1], -0.5 - 0.5 * p_children[1, [2, 6]].sum()
            )
        )

        self.assertTrue(np.all(child.isolated == np.array([0, 2])))

        # Y-nodes
        self.assertTrue(one_y["density"].size == 2)
        self.assertTrue(one_y["density"][0] * parent_length[0] == 1)
        self.assertTrue(one_y["density"][1] * parent_length[1] == 1)


class TestDensityCounting(unittest.TestCase):
    def test_1d_counting_single_box(self):
        domain = {"xmin": 0, "xmax": 1}
        p = np.array([0.4])
        e = np.array([[0], [0]])

        num_occ = frac_gen.count_center_point_densities(p, e, domain, nx=1)
        self.assertTrue(num_occ.size == 1)
        self.assertTrue(num_occ[0] == 1)

    def test_1d_counting_no_points(self):
        domain = {"xmin": 0, "xmax": 1}
        p = np.empty((1, 0))
        e = np.array([[0], [0]])

        num_occ = frac_gen.count_center_point_densities(p, e, domain, nx=1)
        self.assertTrue(num_occ.size == 1)
        self.assertTrue(num_occ[0] == 0)

    def test_1d_counting_two_boxes(self):
        domain = {"xmin": 0, "xmax": 1}
        p = np.array([0.4])
        e = np.array([[0], [0]])

        num_occ = frac_gen.count_center_point_densities(p, e, domain, nx=2)
        self.assertTrue(num_occ.size == 2)
        self.assertTrue(np.all(num_occ == np.array([1, 0])))

    def test_measure_computation_1d(self):
        n = np.random.rand(1)
        domain = {'xmin': 0, 'xmax': n}
        f = FractureSet(domain=domain)
        self.assertTrue(f.domain_measure() == n)

    def test_measure_computation_2d(self):
        n = np.random.rand(2)
        domain = {'xmin': 0, 'ymin': 0, 'xmax': n[0], 'ymax': n[1]}
        f = FractureSet(domain=domain)
        self.assertTrue(f.domain_measure() == n.prod())
        self.assertTrue(f.domain_measure(domain) == n.prod())

class TestFractureProlongationPruning(unittest.TestCase):

    def compare_points(self, a, b, tol=1e-4):
        sz = a.shape[1]
        ind = np.empty(sz)
        self.assertTrue(np.all(a.shape == b.shape))
        for i in range(sz):
            dist = np.sqrt(np.sum((a[:, i].reshape((-1, 1)) - b)**2, axis=0))
            mi = np.argmin(dist)
            self.assertTrue(dist[mi] < tol)
            ind[i] = mi
        return ind

    def test_prolong_fracture(self):
        p_parent = np.array([[0, 1, 0.5, 0.5], [0, 0, 0.1, 1]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)
        fracs = parent.snap(threshold=0.2)

        p_known = np.array([[0, 1, 0.5, 0.5], [0, 0, 1, 0]])
        ind = self.compare_points(p_known, fracs.pts)
        self.assertTrue(np.allclose(ind, np.array([0, 1, 3, 2])))

    def test_no_prolongation(self):
        p_parent = np.array([[0, 1, 0.5, 0.5], [0, 0, 0.1, 1]])
        e_parent = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        parent = FractureSet(p_parent, e_parent, domain)
        # The points are further away than the threshold
        fracs = parent.snap(threshold=0.05)
        ind = self.compare_points(p_parent, fracs.pts)
        self.assertTrue(np.allclose(ind, np.array([0, 1, 2, 3])))

    def test_branch_computation(self):
        p = np.array([[-1, 1, 0, 0, -1, 1], [0, 0, -1, 1, 2, 2]])
        e = np.array([[0, 2, 4], [1, 3, 5]])
        domain = {"xmin": -2, "xmax": 6, "ymin": -2, "ymax": 3}

        fracs = FractureSet(p, e, domain)
        p_split, e_split = fracs.branches()

        p_known = np.array([[-1, 1, 0, 0, -1, 1, 0], [0, 0, -1, 1, 2, 2, 0]])
        ind = self.compare_points(p_split, p_known)

        e_known = np.array([[0, 1, 2, 3, 4], [6, 6, 6, 6, 5]])
        self.assertTrue(np.allclose(np.sort(ind[e_split[:2]], axis=0),
                                    np.sort(e_known, axis=0)))

class TestDomainRestriction(unittest.TestCase):

    def test_no_change(self):
        p = np.array([[0, 1, 0.5, 0.5], [0, 0, 0.1, 1]])
        e = np.array([[0, 2], [1, 3]])

        domain = {"xmin": -1, "xmax": 6, "ymin": -1, "ymax": 2}

        fracs = FractureSet(p, e, domain)

        new_frac = fracs.constrain_to_domain(domain)

        self.assertTrue(np.allclose(new_frac.pts, fracs.pts))
        self.assertTrue(np.allclose(new_frac.edges, fracs.edges))

    def test_one_changed(self):
        p = np.array([[0, 1, 0.5, 0.5], [0, 0, 0.1, 1]])
        e = np.array([[0, 2], [1, 3]])

        # The domain is smaller than the extent of the fractures. That will
        # not be noted until we constrain the set
        domain = {"xmin": -1, "xmax": 0.7, "ymin": -1, "ymax": 2}

        fracs = FractureSet(p, e, domain)

        new_frac = fracs.constrain_to_domain(domain)

        known_p = np.array([[0, 0.7, 0.5, 0.5], [0, 0, 0.1, 1]])
        known_e = np.array([[0, 2], [1, 3]])

        self.assertTrue(np.allclose(new_frac.pts, known_p))
        self.assertTrue(np.allclose(new_frac.edges, known_e))

    def test_one_disapear(self):
        # One fracture is completely outside the domain
        p = np.array([[0, 1, 0.5, 0.5], [0, 0, 0.7, 1]])
        e = np.array([[0, 2], [1, 3]])

        # The domain is smaller than the extent of the fractures. That will
        # not be noted until we constrain the set
        domain = {"xmin": -1, "xmax": 1, "ymin": -1, "ymax": 0.5}

        fracs = FractureSet(p, e, domain)

        new_frac = fracs.constrain_to_domain(domain)

        known_p = np.array([[0, 1], [0, 0]])
        known_e = np.array([[0], [1]])

        self.assertTrue(np.allclose(new_frac.pts, known_p))
        self.assertTrue(np.allclose(new_frac.edges, known_e))

    def test_both_disapear(self):
        # One fracture is completely outside the domain
        p = np.array([[0, 1, 0.5, 0.5], [0, 0, 0.7, 1]])
        e = np.array([[0, 2], [1, 3]])

        # The domain is smaller than the extent of the fractures. That will
        # not be noted until we constrain the set
        domain = {"xmin": -1, "xmax": 1, "ymin": -5, "ymax": -0.5}

        fracs = FractureSet(p, e, domain)

        new_frac = fracs.constrain_to_domain(domain)

        self.assertTrue(new_frac.pts.size == 0)
        self.assertTrue(new_frac.edges.size == 0)


class DummyDistribution:
    def __init__(self, value):
        self.value = value

    def rvs(self, size, **kwagrs):
        return self.value * np.ones(size)


def make_dummy_distribution(value):
    return {"dist": DummyDistribution(value), "param": {}}


if __name__ == '__main__':
    unittest.main()
# TestParentChildrenRelations().test_only_isolated_two_parents_one_far_away()
#TestFractureSetGeneration().test_two_parents_one_far_away_all_both_y_children()
#TestDensityCounting().test_measure_computation_2d()
#unittest.main()
#TestDomainRestriction().test_both_disapear()
# TestDensityCounting().test_1d_counting_two_boxes()