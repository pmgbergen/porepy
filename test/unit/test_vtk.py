import numpy as np
import unittest

from porepy.grids import structured, simplex
from porepy.fracs import meshing

from porepy.viz.exporter import Exporter

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_single_grid_1d(self):
        pass

#------------------------------------------------------------------------------#

    def test_single_grid_2d_simplex(self):
        pass

#------------------------------------------------------------------------------#

    def test_single_grid_2d_cart(self):
        pass

#------------------------------------------------------------------------------#

    def test_single_grid_2d_polytop(self):
        pass

#------------------------------------------------------------------------------#

    def test_single_grid_3d_simplex(self):
        pass

#------------------------------------------------------------------------------#

    def test_single_grid_3d_cart(self):
        pass

#------------------------------------------------------------------------------#

    def test_single_grid_3d_polytop(self):
        pass

#------------------------------------------------------------------------------#

    def test_gb_1(self):
        f1 = np.array([[0, 1], [.5, .5]])
        gb = meshing.cart_grid([f1], [4]*2, **{'physdims': [1, 1]})
        gb.compute_geometry()

        self.gb.add_node_props(extra_data)


        np.random.seed(0)
        np.random.rand(4)

        save = Exporter(gb, "grid", "test_vtk", binary=False)
        save.write_vtk()

#------------------------------------------------------------------------------#

    def test_gb_2(self):
        f1 = np.array([[0, 1], [.5, .5]])
        f2 = np.array([[.5, .5], [.25, .75]])
        gb = meshing.cart_grid([f1, f2], [4]*2, **{'physdims': [1, 1]})

#------------------------------------------------------------------------------#

BasicsTest().test_gb_1()
