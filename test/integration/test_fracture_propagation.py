"""
Tests of the propagation of fractures. This involves extension of the fracture
grids and splitting of faces in the matrix grid. The post-propagation buckets
are compared to equivalent buckets where the final fracture geometry is applied
at construction.


"""
import unittest

import numpy as np
import porepy as pp
import scipy.sparse as sps

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


class MockPropagationModel(pp.ConformingFracturePropagation):
    # Helper class for variable mapping tests.
    # Define a propagation model with a tailored method to populate new fields in
    # variables. Should be easy to verify implementation.
    # Subclass ConformingFracturePropagation, which is the simplest known non-ABC
    # propagation class.

    def _initialize_new_variable_values(self, g, d, var, dofs):
        cell_dof: int = dofs.get("cells")
        # Number of new variables is given by the size of the cell map.
        cell_map = d["cell_index_map"]
        n_new = cell_map.shape[0] - cell_map.shape[1]

        # Fill the new values with 42, because of course
        vals = np.full(n_new * cell_dof, 42)
        return vals


class VariableMappingInitializationUnderPropagation(unittest.TestCase):
    """ Test of functionality to update variables during propagation.

    In reality, three features are tested: First, the mappings for cell and face variables,
    generated by fracture propagation, second the usage of these mappings to update
    variables, and third the functionality to initialize new variables.

    Three cases are considered (all 2d, 3d should raise no further complication):
        1. Update on a domain a single fracture.
        2. Update on a domain with two fractures, both propagating.
        3. Update on a domain with two fractures, only one propagating.

    """

    def setUp(self):
        self.cv2 = "cell_variable_2d"
        self.cv1 = "cell_variable_1d"
        self.mv = "mortar_variable"

    def test_single_fracture_propagate(self):
        # Domain with 5x2 cells. The fracture is initially one face long, and will be
        # propageted to be three faces long, in two steps.
        frac = [np.array([[1, 2], [1, 1]])]
        gb = pp.meshing.cart_grid(frac, [5, 2])

        g_frac = gb.grids_of_dimension(1)[0]
        # Two prolongation steps for the fracture.
        split_faces = [{g_frac: np.array([19])}, {g_frac: np.array([20])}]
        self._verify(gb, split_faces)

    def test_two_fractures_propagate_both(self):
        # Domain with 5x3 cells, with two fractures.
        # Both fractures areinitially one face long, and will be propagated in two steps

        frac = [np.array([[1, 2], [1, 1]]), np.array([[2, 3], [2, 2]])]
        gb = pp.meshing.cart_grid(frac, [5, 3])

        g_1, g_2 = gb.grids_of_dimension(1)

        # Two prolongation steps for the fracture.
        # In the first step, one fracutre grows, one is stuck.
        # In second step, the second fracture grows in both ends
        split_faces = [
            {g_1: np.array([25]), g_2: np.array([])},
            {g_1: np.array([26]), g_2: np.array([29, 31])},
        ]

        self._verify(gb, split_faces)

    def _verify(self, gb, split_faces):
        # Common function for all tests.
        # Propagate fracture, verify that variables are mapped correctly, and that
        # new variables are added with correct values

        # Individual grids
        g_2d = gb.grids_of_dimension(2)[0]
        g_1d = gb.grids_of_dimension(1)

        # Model used for fracture propagation
        model = MockPropagationModel({})
        model.gb = gb

        # Cell variable in 2d. Should stay constant throughout the process.
        cell_val_2d = np.random.rand(g_2d.num_cells)

        ## Initialize variables for all grids
        # Cell variable in 1d. Sholud be expanded, but initial size is num_vars_2d
        var_sz_1d = 2
        # 1d variables defined as a dict, indexed on the grid
        cell_val_1d = {g: np.random.rand(var_sz_1d) for g in g_1d}

        var_sz_mortar = 3
        cell_val_mortar = {g: np.random.rand(var_sz_mortar * 2) for g in g_1d}

        # Define variables on all grids.
        # Initialize the state by the known variable, and the iterate as twice that value
        # (mostly a why not)
        d = gb.node_props(g_2d)
        d[pp.PRIMARY_VARIABLES] = {self.cv2: {"cells": 1}}
        d[pp.STATE] = {
            self.cv2: cell_val_2d,
            pp.ITERATE: {self.cv2: np.array(2 * cell_val_2d)},
        }

        for g in g_1d:
            d = gb.node_props(g)
            d[pp.PRIMARY_VARIABLES] = {self.cv1: {"cells": var_sz_1d}}
            d[pp.STATE] = {
                self.cv1: cell_val_1d[g],
                pp.ITERATE: {self.cv1: np.array(2 * cell_val_1d[g])},
            }

            d = gb.edge_props((g_2d, g))
            d[pp.PRIMARY_VARIABLES] = {self.mv: {"cells": var_sz_mortar}}
            d[pp.STATE] = {
                self.mv: cell_val_mortar[g],
                pp.ITERATE: {self.mv: np.array(2 * cell_val_mortar[g])},
            }

        # Define assembler, thereby a dof ordering
        assembler = pp.Assembler(gb)
        model.assembler = assembler

        # Define and initialize a state vector
        x = np.zeros(assembler.full_dof.sum())
        x[assembler.dof_ind(g_2d, self.cv2)] = cell_val_2d
        for g in g_1d:
            x[assembler.dof_ind(g, self.cv1)] = cell_val_1d[g]
            x[assembler.dof_ind((g_2d, g), self.mv)] = cell_val_mortar[g]

        # Keep track of the previous values for each grid.
        # Not needed in 2d, where no updates are expected
        val_1d_prev = {g: cell_val_1d[g] for g in g_1d}
        val_mortar_prev = {g: cell_val_mortar[g] for g in g_1d}
        val_1d_iterate_prev = {g: 2 * cell_val_1d[g] for g in g_1d}
        val_mortar_iterate_prev = {g: 2 * cell_val_mortar[g] for g in g_1d}

        # Loop over all propagation steps
        for i, split in enumerate(split_faces):

            # Propagate the fracture. This will also generate mappings from old to new
            # cells
            pp.propagate_fracture.propagate_fractures(gb, split)

            # Update variables
            x_new = model._map_variables(x)

            # The values of the 2d cell should not change
            self.assertTrue(
                np.all(x_new[assembler.dof_ind(g_2d, self.cv2)] == cell_val_2d)
            )
            # Also check that pp.STATE and ITERATE has been correctly updated
            d = gb.node_props(g_2d)
            self.assertTrue(np.all(d[pp.STATE][self.cv2] == cell_val_2d))
            self.assertTrue(
                np.all(d[pp.STATE][pp.ITERATE][self.cv2] == 2 * cell_val_2d)
            )

            # Loop over all 1d grids, check both grid and the associated mortar grid
            for g in g_1d:

                num_new_cells = split[g].size

                # mapped variable
                x_1d = x_new[assembler.dof_ind(g, self.cv1)]

                # Extension of the 1d grid. All values should be 42 (see propagation class)
                extended_1d = np.full(var_sz_1d * num_new_cells, 42)

                # True values for 1d (both grid and iterate)
                truth_1d = np.r_[val_1d_prev[g], extended_1d]
                truth_iterate = np.r_[val_1d_iterate_prev[g], extended_1d]
                self.assertTrue(np.allclose(x_1d, truth_1d))

                # Also check that pp.STATE and ITERATE has been correctly updated
                d = gb.node_props(g)
                self.assertTrue(np.all(d[pp.STATE][self.cv1] == truth_1d))
                self.assertTrue(
                    np.all(d[pp.STATE][pp.ITERATE][self.cv1] == truth_iterate)
                )

                # The propagation model will assign the value 42 to new cells also in the
                # next step. To be sure values from the first propagation are mapped
                # correctly, we alter the true value (add 1), and update this both in
                # the solution vector, state and previous iterate
                x_new[assembler.dof_ind(g, self.cv1)] = np.r_[
                    val_1d_prev[g], extended_1d + 1
                ]
                val_1d_prev[g] = x_new[assembler.dof_ind(g, self.cv1)]
                d[pp.STATE][self.cv1] = x_new[assembler.dof_ind(g, self.cv1)]
                val_1d_iterate_prev[g] = np.r_[val_1d_iterate_prev[g], extended_1d + 1]
                d[pp.STATE][pp.ITERATE][self.cv1] = val_1d_iterate_prev[g]

                ## Check mortar grid - see 1d case above for comments
                x_mortar = x_new[assembler.dof_ind((g_2d, g), self.mv)]

                sz = int(np.round(val_mortar_prev[g].size / 2))

                # The new mortar cells are appended such that the cells are ordered
                # contiguously on the two sides.
                truth_mortar = np.r_[
                    val_mortar_prev[g][:sz],
                    np.full(var_sz_mortar * num_new_cells, 42),
                    val_mortar_prev[g][sz : 2 * sz],
                    np.full(var_sz_mortar * num_new_cells, 42),
                ]
                truth_iterate = np.r_[
                    val_mortar_iterate_prev[g][:sz],
                    np.full(var_sz_mortar * num_new_cells, 42),
                    val_mortar_iterate_prev[g][sz : 2 * sz],
                    np.full(var_sz_mortar * num_new_cells, 42),
                ]
                d = gb.edge_props((g_2d, g))

                self.assertTrue(np.all(x_mortar == truth_mortar))
                self.assertTrue(np.all(d[pp.STATE][self.mv] == truth_mortar))
                self.assertTrue(
                    np.all(d[pp.STATE][pp.ITERATE][self.mv] == truth_iterate)
                )

                x_new[assembler.dof_ind((g_2d, g), self.mv)] = np.r_[
                    val_mortar_prev[g][:sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                    val_mortar_prev[g][sz : 2 * sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                ]
                val_mortar_prev[g] = x_new[assembler.dof_ind((g_2d, g), self.mv)]
                val_mortar_iterate_prev[g] = np.r_[
                    val_mortar_iterate_prev[g][:sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                    val_mortar_iterate_prev[g][sz : 2 * sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                ]
                d[pp.STATE][self.mv] = val_mortar_prev[g]
                d[pp.STATE][pp.ITERATE][self.mv] = val_mortar_iterate_prev[g]

                x = x_new


if __name__ == "__main__":
    unittest.main()
