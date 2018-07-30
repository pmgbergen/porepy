from __future__ import division
import numpy as np
import scipy.sparse as sps

from porepy.numerics.fv import fvutils
from porepy.grids import structured, simplex


def test_subcell_topology_2d_cart_1():
    x = np.ones(2, dtype=np.int)
    g = structured.CartGrid(x)

    subcell_topology = fvutils.SubcellTopology(g)

    assert np.all(subcell_topology.cno == 0)

    ncum = np.bincount(subcell_topology.nno, weights=np.ones(subcell_topology.nno.size))
    assert np.all(ncum == 2)

    fcum = np.bincount(subcell_topology.fno, weights=np.ones(subcell_topology.fno.size))
    assert np.all(fcum == 2)

    # There is only one cell, thus only unique subfno
    usubfno = np.unique(subcell_topology.subfno)
    assert usubfno.size == subcell_topology.subfno.size

    assert np.all(np.in1d(subcell_topology.subfno, subcell_topology.subhfno))


def test_subcell_mapping_2d_simplex_1():
    p = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    g = simplex.TriangleGrid(p)

    subcell_topology = fvutils.SubcellTopology(g)

    ccum = np.bincount(subcell_topology.cno, weights=np.ones(subcell_topology.cno.size))
    assert np.all(ccum == 6)

    ncum = np.bincount(subcell_topology.nno, weights=np.ones(subcell_topology.nno.size))
    assert ncum[0] == 2
    assert ncum[1] == 4
    assert ncum[2] == 2
    assert ncum[3] == 4

    fcum = np.bincount(subcell_topology.fno, weights=np.ones(subcell_topology.fno.size))
    assert np.sum(fcum == 4) == 1
    assert np.sum(fcum == 2) == 4

    subfcum = np.bincount(
        subcell_topology.subfno, weights=np.ones(subcell_topology.subfno.size)
    )
    assert np.sum(subfcum == 2) == 2
    assert np.sum(subfcum == 1) == 8


def test_determine_eta():
    g = simplex.StructuredTriangleGrid([1, 1])
    assert fvutils.determine_eta(g) == 1 / 3
    g = structured.CartGrid([1, 1])
    assert fvutils.determine_eta(g) == 0
