import numpy as np
import unittest

from core.grids import structured

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_cell_diameters_2d(self):
        g = structured.CartGrid([3, 2], [1, 1])
        cell_diameters = g.cell_diameters()
        known = np.repeat( np.sqrt( 0.5**2 + 1./3.**2), g.num_cells )
        assert np.allclose( cell_diameters, known )

#------------------------------------------------------------------------------#

    def test_cell_diameters_3d(self):
        g = structured.CartGrid([3, 2, 1])
        cell_diameters = g.cell_diameters()
        known = np.repeat( np.sqrt(3), g.num_cells )
        assert np.allclose( cell_diameters, known )

#------------------------------------------------------------------------------#
