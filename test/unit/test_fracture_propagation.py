"""
Tests of the propagation of fractures. This involves extension of the fracture
grids and splitting of faces in the matrix grid. The post-propagation buckets
are compared to equivalent buckets where the final fracture geometry is applied
at construction.


"""
import unittest

import numpy as np
import porepy as pp

from test.integration.fracture_propagation_utils import check_equivalent_buckets
from test.integration import setup_mixed_dimensional_grids as setup_gb
from test.test_utils import compare_arrays


class TestPropagateFractures(unittest.TestCase):
    """Tests of splitting of grids.

    The different tests have been written at different points in time, and could have
    benefited from a consolidation. On the other hand, the tests also cover different
    components of the propagation.

    """

    def test_equivalence_2d(self):
        """
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """

        # Generate grid buckets
        gb_1 = setup_gb.grid_2d_1d([4, 2], 0, 0.75)
        gb_2 = setup_gb.grid_2d_1d([4, 2], 0, 0.25)
        gb_3 = setup_gb.grid_2d_1d([4, 2], 0, 0.50)
        gb_4 = setup_gb.grid_2d_1d([4, 2], 0, 0.25)

        # Pick out the 1d grids in each bucket. These will be used to set which faces
        # in the 2d grid to split
        g2 = gb_2.grids_of_dimension(1)[0]
        g3 = gb_3.grids_of_dimension(1)[0]
        g4 = gb_4.grids_of_dimension(1)[0]

        # Do the splitting
        pp.propagate_fracture.propagate_fractures(gb_2, {g2: np.array([15, 16])})
        pp.propagate_fracture.propagate_fractures(gb_3, {g3: np.array([16])})
        # The fourth bucket is split twice
        pp.propagate_fracture.propagate_fractures(gb_4, {g4: np.array([15])})
        pp.propagate_fracture.propagate_fractures(gb_4, {g4: np.array([16])})

        # Check that the four grid buckets are equivalent
        check_equivalent_buckets([gb_1, gb_2, gb_3, gb_4])

    def test_equivalence_3d(self):
        """
        3d version of previous test.
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        # Make buckets
        gb_1 = setup_gb.grid_3d_2d([4, 2, 2], 0, 0.75)
        gb_2 = setup_gb.grid_3d_2d([4, 2, 2], 0, 0.25)
        gb_3 = setup_gb.grid_3d_2d([4, 2, 2], 0, 0.50)
        gb_4 = setup_gb.grid_3d_2d([4, 2, 2], 0, 0.25)

        # Pick out the 2d grids in each bucket. These will be used to set which faces
        # in the 3d grid to split
        g2 = gb_2.grids_of_dimension(2)[0]
        g3 = gb_3.grids_of_dimension(2)[0]
        g4 = gb_4.grids_of_dimension(2)[0]

        # Do the splitting
        pp.propagate_fracture.propagate_fractures(gb_2, {g2: np.array([53, 54])})
        pp.propagate_fracture.propagate_fractures(gb_3, {g3: np.array([54])})
        # The fourth bucket is split twice
        pp.propagate_fracture.propagate_fractures(gb_4, {g4: np.array([53])})
        pp.propagate_fracture.propagate_fractures(gb_4, {g4: np.array([54])})

        # Check that the four grid buckets are equivalent
        check_equivalent_buckets([gb_1, gb_2, gb_3, gb_4])

    def test_two_fractures_2d(self):
        """
        Two fractures growing towards each other, but not meeting.

        Tests simultanous growth of two fractures in multiple steps, and growth
        of one fracture in the presence of an inactive one.
        """
        f_1 = np.array([[0, 0.25], [0.5, 0.5]])
        f_2 = np.array([[1.75, 2], [0.5, 0.5]])
        gb = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 0.5], [0.5, 0.5]])
        f_2 = np.array([[1.5, 2], [0.5, 0.5]])
        gb_1 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 0.75], [0.5, 0.5]])
        f_2 = np.array([[1.25, 2], [0.5, 0.5]])
        gb_2 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 1.0], [0.5, 0.5]])
        gb_3 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        # Get the fracture grids
        g1 = gb.grids_of_dimension(1)[0]
        g2 = gb.grids_of_dimension(1)[1]

        # First propagation step
        faces = {g1: np.array([27]), g2: np.array([32])}
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_1])

        # Second step
        faces = {g1: np.array([28]), g2: np.array([31])}
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_2])

        # Final step - only one of the fractures grow
        faces = {g1: np.array([29]), g2: np.array([], dtype=int)}
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_3])



    def test_two_propagation_steps_3d(self):
        # Starting from a fracture one cell wide, three cells long, the first step
        # opens faces on the middle of the fracutre, so that it looks roughly like
        #  O O O   <- Initial fracture
        #    0     <- face opened in the first step
        #
        # The second step requests opening of the two flanking faces
        gb = self._make_grid()
        g_frac = gb.grids_of_dimension(2)[0]

        faces_to_split = [[43], [41, 45]]

        g = gb.grids_of_dimension(2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes

        gh = gb.grids_of_dimension(3)[0]

        new_fc = [
            np.array([[1.5, 2, 1.5], [1, 1.5, 2], [1.5, 2, 1.5]]),
            np.array([[1.5, 2, 2, 1.5], [0, 0.5, 2.5, 3], [1.5, 2, 2, 1.5]]),
        ]

        new_cell_volumes = [np.array([np.sqrt(2)]), np.sqrt(2) * np.ones(2)]

        # The first splitting will create no new nodes (only more tip nodes)
        # Second splitting generates 8 new nodes, since the domain is split in two
        num_nodes = gh.num_nodes + np.array([0, 8])

        for si, split in enumerate(faces_to_split):
            pp.propagate_fracture.propagate_fractures(gb, {g_frac: np.array(split)})
            cc, fc, cv = self._verify(
                gb, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    def test_three__from_bottom(self):
        # Starting from a fracture one cell wide, three cells long, the first step
        # opens one flanking face of the fracutre, so that it looks roughly like
        #  O O O   <- Initial fracture
        #  0       <- face opened in the first step
        #
        # The second step requests the opening of the middle face, the final step the
        # the second flaking face.
        gb = self._make_grid()
        g_frac = gb.grids_of_dimension(2)[0]

        faces_to_split = [[41], [43], [45]]

        g = gb.grids_of_dimension(2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes

        gh = gb.grids_of_dimension(3)[0]

        new_fc = [
            np.array([[1.5, 2, 1.5], [0, 0.5, 1], [1.5, 2, 1.5]]),
            np.array([[2, 1.5], [1.5, 2], [2, 1.5]]),
            np.array([[2, 1.5], [2.5, 3], [2, 1.5]]),
        ]

        new_cell_volumes = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]

        num_nodes = gh.num_nodes + np.array([2, 4, 8])

        for si, split in enumerate(faces_to_split):
            pp.propagate_fracture.propagate_fractures(gb, {g_frac: np.array(split)})
            cc, fc, cv = self._verify(
                gb, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    def test_open_face_from_two_sides_simultaneously(self):
        # Starting from a fracture one cell wide, three cells long, the first step
        # opens faces on the flanks of the fracture, so that it looks roughly like
        #  O O O   <- Initial fracture
        #  O   O   <- faces opened in the first step
        #
        # The second step requests opening of the middle face from both sides.
        gb = self._make_grid()
        g_frac = gb.grids_of_dimension(2)[0]

        faces_to_split = [[41, 45], [43, 43]]

        g = gb.grids_of_dimension(2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes

        gh = gb.grids_of_dimension(3)[0]

        new_fc = [
            np.array(
                [
                    [1.5, 2, 1.5, 1.5, 2, 1.5],
                    [0, 0.5, 1, 2, 2.5, 3],
                    [1.5, 2, 1.5, 1.5, 2, 1.5],
                ]
            ),
            np.array([[2], [1.5], [2]]),
        ]

        new_cell_volumes = [np.sqrt(2) * np.ones(2), np.sqrt(2)]

        num_nodes = gh.num_nodes + np.array([4, 8])

        for si, split in enumerate(faces_to_split):
            pp.propagate_fracture.propagate_fractures(gb, {g_frac: np.array(split)})
            cc, fc, cv = self._verify(
                gb, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    #### Helpers below

    def _verify(self, gb, split, cc, fc, cv, new_cell_volumes, new_fc, num_nodes):
        # Check that the geometry of a (propagated) GridBucket corresponds to a given
        # known geometry.

        gh = gb.grids_of_dimension(gb.dim_max())[0]
        g = gb.grids_of_dimension(gb.dim_max() - 1)[0]
        new_cc = gh.face_centers[:, split]
        if len(split) == 1:
            new_cc = new_cc.reshape((-1, 1))
        cc = np.append(cc, new_cc, axis=1)
        compare_arrays(g.cell_centers, cc)

        fc = np.append(fc, new_fc, axis=1)
        compare_arrays(g.face_centers, fc)

        cv = np.append(cv, new_cell_volumes)

        self.assertTrue(np.allclose(g.cell_volumes, cv))

        proj = gb.node_props(g)["tangential_normal_projection"]
        self.assertTrue(proj.normals.shape[1] == g.num_cells)
        self.assertTrue(
            np.logical_or(np.all(proj.normals[2] < 0), np.all(proj.normals[2] > 0))
        )

        hit = np.logical_and(g.face_centers[0] > 1.1, g.face_centers[0] < 1.9)
        self.assertTrue(np.allclose(g.face_areas[hit], np.sqrt(2)))
        self.assertTrue(np.allclose(g.face_areas[np.logical_not(hit)], 1))

        self.assertTrue(num_nodes == gh.num_nodes)

        # Return the new geometric fields; these will be needed to validate the next
        # propagation step.
        return cc, fc, cv

    def _make_grid(self):
        # Make a 2 x 3 x 2 Cartesian grid with a single fracture embedded.
        # The geometry of all grids is perturbed so that the grid, projected to the
        # xz-plane looks like this
        #
        #           x
        #        /  |
        # x - x  -  x
        # |   |  /  |
        # x - x  -  x
        # |   |  /
        # x - x
        # Not sure why the grid ended up like that, but there you go.


        frac_1 = np.array([[0, 1, 1, 0], [0, 0, 3, 3], [1, 1, 1, 1]])
        gb = pp.meshing.cart_grid([frac_1], [2, 3, 2])

        for g, _ in gb:
            hit = g.nodes[0] > 1.5
            g.nodes[2, hit] += 1

        gb.compute_geometry()

        pp.contact_conditions.set_projections(gb)

        return gb



if __name__ == "__main__":
    unittest.main()
