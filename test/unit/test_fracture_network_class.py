#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 08:14:33 2018

@author: eke001
"""

import unittest
import numpy as np

import porepy as pp


class TestFractureNetworkBoundingBox(unittest.TestCase):
    def test_sinle_fracture(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        network = pp.FractureNetwork([f1])
        d = network.bounding_box()

        assert d["xmin"] == 0
        assert d["xmax"] == 1
        assert d["ymin"] == 0
        assert d["ymax"] == 1
        assert d["zmin"] == 0
        assert d["zmax"] == 1

    def test_sinle_fracture_aligned_with_axis(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
        )

        network = pp.FractureNetwork([f1])
        d = network.bounding_box()

        assert d["xmin"] == 0
        assert d["xmax"] == 1
        assert d["ymin"] == 0
        assert d["ymax"] == 1
        assert d["zmin"] == 0
        assert d["zmax"] == 0

    def test_two_fractures(self):
        # Test of method FractureNetwork.bounding_box() to inquire about
        # network extent
        f1 = pp.Fracture(
            np.array([[0, 2, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        f2 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [-1, -1, 1, 1]]),
            check_convexity=False,
        )

        network = pp.FractureNetwork([f1, f2])
        d = network.bounding_box()

        assert d["xmin"] == 0
        assert d["xmax"] == 2
        assert d["ymin"] == 0
        assert d["ymax"] == 1
        assert d["zmin"] == -1
        assert d["zmax"] == 1

    def test_external_boundary_added(self):
        # Test of method FractureNetwork.bounding_box() when an external
        # boundary is added. Thus double as test of this adding.
        f1 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), check_convexity=False
        )

        network = pp.FractureNetwork([f1])

        external_boundary = {
            "xmin": -1,
            "xmax": 2,
            "ymin": -1,
            "ymax": 2,
            "zmin": -1,
            "zmax": 2,
        }
        network.impose_external_boundary(external_boundary)
        d = network.bounding_box()

        assert d["xmin"] == external_boundary["xmin"]
        assert d["xmax"] == external_boundary["xmax"]
        assert d["ymin"] == external_boundary["ymin"]
        assert d["ymax"] == external_boundary["ymax"]
        assert d["zmin"] == external_boundary["zmin"]
        assert d["zmax"] == external_boundary["zmax"]


if __name__ == "__main__":
    unittest.main()
