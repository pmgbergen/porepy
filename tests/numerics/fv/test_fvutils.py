"""Unit tests for the fvutils module.

Currently tested are:
    * The subcell topology for 2d Cartesian and simplex grids.
    * The determination of the eta parameter.
    * The helper function computing a diagonal scaling matrix.

"""
from __future__ import division

import unittest

import numpy as np
import scipy.sparse as sps

from porepy.grids import simplex, structured
from porepy.numerics.fv import fvutils


class TestFvutils(unittest.TestCase):
    def test_subcell_topology_2d_cart(self):
        # Verify that the subcell topology is correct for a 2d Cartesian grid.
        x = np.ones(2, dtype=int)
        g = structured.CartGrid(x)

        subcell_topology = fvutils.SubcellTopology(g)

        self.assertTrue(np.all(subcell_topology.cno == 0))

        ncum = np.bincount(
            subcell_topology.nno, weights=np.ones(subcell_topology.nno.size)
        )
        self.assertTrue(np.all(ncum == 2))

        fcum = np.bincount(
            subcell_topology.fno, weights=np.ones(subcell_topology.fno.size)
        )
        self.assertTrue(np.all(fcum == 2))

        # There is only one cell, thus only unique subfno
        usubfno = np.unique(subcell_topology.subfno)
        self.assertTrue(usubfno.size == subcell_topology.subfno.size)

        self.assertTrue(
            np.all(np.in1d(subcell_topology.subfno, subcell_topology.subhfno))
        )

    def test_subcell_mapping_2d_simplex(self):
        # Verify that the subcell mapping is correct for a 2d simplex grid.
        p = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
        g = simplex.TriangleGrid(p)

        subcell_topology = fvutils.SubcellTopology(g)

        ccum = np.bincount(
            subcell_topology.cno, weights=np.ones(subcell_topology.cno.size)
        )
        self.assertTrue(np.all(ccum == 6))

        ncum = np.bincount(
            subcell_topology.nno, weights=np.ones(subcell_topology.nno.size)
        )
        self.assertTrue(ncum[0] == 2)
        self.assertTrue(ncum[1] == 4)
        self.assertTrue(ncum[2] == 2)
        self.assertTrue(ncum[3] == 4)

        fcum = np.bincount(
            subcell_topology.fno, weights=np.ones(subcell_topology.fno.size)
        )
        self.assertTrue(np.sum(fcum == 4) == 1)
        self.assertTrue(np.sum(fcum == 2) == 4)

        subfcum = np.bincount(
            subcell_topology.subfno, weights=np.ones(subcell_topology.subfno.size)
        )
        self.assertTrue(np.sum(subfcum == 2) == 2)
        self.assertTrue(np.sum(subfcum == 1) == 8)

    def test_determine_eta(self):
        # Test that the automatic computation of the pressure continuity point is
        # correct for a Cartesian and simplex grid.
        g = simplex.StructuredTriangleGrid([1, 1])
        self.assertTrue(fvutils.determine_eta(g) == 1 / 3)
        g = structured.CartGrid([1, 1])
        self.assertTrue(fvutils.determine_eta(g) == 0)

    def test_diagonal_scaling_matrix(self):
        # Generate a matrix with a known row sum, check that the target function
        # returns the correct diagonal.
        A = np.array([[1, 2, 3], [0, -5, 6], [-7, 8, 0]])
        A_sum = np.array([6, 11, 15])
        values = 1 / A_sum

        D = fvutils.diagonal_scaling_matrix(sps.csr_matrix(A))
        self.assertTrue(np.allclose(values, D.diagonal()))


if __name__ == "__main__":
    unittest.main()
