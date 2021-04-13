"""
Tests of the propagation of fractures.

Content:
  * test_pick_propagation_face_conforming_propagation: Given a critically stressed
    fracture tip in the lower-dimensional grid, find which face in the higher-dimensinoal
    grid should be split.
  * FaceSplittingHostGrid: Various consistency tests for splitting faces in the higher
    dimensional grid, resembling propagation.
  * PropagationCriteria: Tests i) SIF computation by displacement correlation, ii)
    tags for which tips to propagate iii) propagion angles (only asserts all angles are zero).
  * VariableMappingInitializationUpdate: Checks that mapping of variables and
    assignment of new variable values in the FracturePropagation classes are correct.

The tests follow in part the unittest framework, in part pytest.

"""
import unittest
import pytest

import numpy as np
import porepy as pp
import scipy.sparse as sps

from test.integration.fracture_propagation_utils import check_equivalent_buckets
from test.integration import setup_mixed_dimensional_grids as setup_gb
from test.test_utils import compare_arrays


## Below follows tests of the picking of high-dimensional faces to split, when using
#  the ConformingPropagation strategy.
#
# The first functions are helpers to set up geometries and propagation scenarios
# The test function (implemented in the pytest framework) is further down.


def _single_fracture_two_steps_2d():
    # Define a 2d medium with a single fracture.
    # Propagate first in both ends, and then only in one end.
    # This is (almost) the simplest possible case, so if the test fails, something is
    # either fundamentally wrong with the propagation, or a basic assumption is broken.
    fracs = [np.array([[1, 2], [1, 1]])]
    gb = pp.meshing.cart_grid(fracs, [4, 3])
    g1 = gb.grids_of_dimension(gb.dim_max() - 1)[0]

    # Target cells
    targets = [{g1: np.array([21, 19])}, {g1: np.array([22])}]
    # Always propagate in a straight line
    angles = [{g1: np.array([0, 0])}, {g1: np.array([0])}]

    return gb, targets, angles


def _single_fracture_multiple_steps_3d():
    # Simple 3d case with one fracture.
    # First step propagates in two directions.
    # Second step propagates in three directions, including both newly formed faces,
    # and the original fracture.
    fracs = [np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 1, 2, 2]])]
    gb = pp.meshing.cart_grid(fracs, [2, 3, 3])

    g1 = gb.grids_of_dimension(gb.dim_max() - 1)[0]
    targets = [{g1: np.array([10, 4])}, {g1: np.array([1, 7, 16])}]
    angles = [{g1: np.array([0, 0])}, {g1: np.array([0, 0, 0])}]

    return gb, targets, angles


def _two_fractures_multiple_steps_3d():
    # Simple 3d case with one fracture.
    # First step propagates in two directions.
    # Second step propagates in three directions, including both newly formed faces,
    # and the original fracture.

    # Note that the second fracture needs to be two faces away from the first one,
    # or else various assumptions in the code will be broken.
    fracs = [
        np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 1, 2, 2]]),
        np.array([[3, 3, 3, 3], [1, 2, 2, 1], [1, 1, 2, 2]]),
    ]
    gb = pp.meshing.cart_grid(fracs, [4, 3, 3])

    g1 = gb.grids_of_dimension(gb.dim_max() - 1)[0]
    g2 = gb.grids_of_dimension(gb.dim_max() - 1)[1]

    targets = [
        {g1: np.array([16, 6]), g2: np.array([], dtype=int)},
        {g1: np.array([1, 11, 26]), g2: np.array([8])},
    ]

    angles = [
        {g1: np.array([0, 0]), g2: np.array([], dtype=int)},
        {g1: np.array([0, 0, 0]), g2: np.array([0])},
    ]

    return gb, targets, angles


@pytest.mark.parametrize(
    "generate",
    [
        _single_fracture_two_steps_2d,
        _single_fracture_multiple_steps_3d,
        _two_fractures_multiple_steps_3d,
    ],
)
def test_pick_propagation_face_conforming_propagation(generate):
    """ Verify that the right high-dimensional face is picked, based on tagged faces
    on the tips of the lower-dimensional grid. Thus, the test probes the function
    _pick_propagation_faces() in ConformingFracturePropagation.

    The overall idea of the test is to give a GridBucket, and a sequence of target
    faces in g_h to be split, based on face indices. For each target face, the
    closest (by distance) of the tip faces in g_l is found, and tagged for propagation.
    We then call the function to be tested to identify the face in g_h which should be
    propagated, and verify that the identified face is indeed the target.

    A key assumption in the above reasoning is that the propagation angle is known, and
    fixed to zero; non-zero values can also be accomodated, but that case is outside
    the standard coverage of the current functionality for fracture propagation.

    """

    mech_key = "mechanics"

    # Get problem geometry and description
    gb, targets, angles = generate()
    pp.contact_conditions.set_projections(gb)

    # Bookkeeping
    g_h = gb.grids_of_dimension(gb.dim_max())[0]
    d_h = gb.node_props(g_h)
    d_h[pp.STATE] = {}

    # Propagation model; assign this some necessary fields.
    model = pp.ConformingFracturePropagation({})
    model.mechanics_parameter_key = mech_key
    model._Nd = gb.dim_max()

    # Loop over all propagation steps
    for step_target, step_angle in zip(targets, angles):

        # Map for faces to be split for all fracture grids
        propagation_map = {}

        # Loop over all fracture grids
        for g_l in gb.grids_of_dimension(gb.dim_max() - 1):

            # Data dictionaries
            d_l = gb.node_props(g_l)
            d_e = gb.edge_props((g_h, g_l))

            # Faces on the tip o the fracture
            tip_faces = np.where(g_l.tags["tip_faces"])[0]

            # Data structure to tag tip faces to propagate from, and angles
            prop_face = np.zeros(g_l.num_faces, dtype=np.bool)
            prop_angle = np.zeros(g_l.num_faces)

            # Loop over all targets for this grid, tag faces to propagate
            for ti, ang in zip(step_target[g_l], step_angle[g_l]):

                # Coordinate of the face center of the target face
                fch = g_h.face_centers[:, ti].reshape((-1, 1))
                # Find the face in g_l (among tip faces) that is closest to this face
                hit = np.argmin(
                    np.sum((g_l.face_centers[:, tip_faces] - fch) ** 2, axis=0)
                )
                # Tag this face for propagation.
                prop_face[tip_faces[hit]] = 1
                prop_angle[tip_faces[hit]] = ang

            # Store information on what to propagate for this fracture
            d_l[pp.PARAMETERS] = {
                mech_key: {
                    "propagation_angle_normal": prop_angle,
                    "propagate_faces": prop_face,
                }
            }

            # Use the functionality to find faces to split
            model._pick_propagation_faces(g_h, g_l, d_h, d_l, d_e)
            _, face, _ = sps.find(d_e["propagation_face_map"])

            # This is the test, the targets and the identified faces should be the same
            assert np.all(np.sort(face) == np.sort(step_target[g_l]))

            # Store propagation map for this face - this is necessary to split the faces
            # and do the test for multiple propagation steps
            propagation_map[g_l] = face

        # We have updated the propagation map for all fractures, and can split faces in
        # g_h
        pp.propagate_fracture.propagate_fractures(gb, propagation_map)
        # .. on to the next propagation step.


class FaceSplittingHostGrid(unittest.TestCase):
    """Tests of splitting of faces in the higher-dimensional grid.

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

    def test_three_from_bottom(self):
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


class PropagationCriteria(unittest.TestCase):
    """ Test of functionality to compute sifs and evaluate propagation onset and angle.

    Test logic:
        1. Set up 2d or 3d gb with single fracture.
        2. Assign displacement and minimal parameters. To help assigning to the right
        side of the interface, the former are specified according to cells in g_h
        and mapped using trace and projection.
        3. Compute the expected i) SIFs (purely tensile - i.e. K_II = K_III = 0),
        ii) propagation tips and iii) angles (all zero in tension).
        4. Call the ConformingFracturePropagation methods.
        5. Compare i-iii.


    We test
        * 2d + 3d
        * open+du_\tau=0, open+du_\tau!=0, closed, gliding
        * That propagation is predicted correctly for du_n = 0, du_n small
        (no propagation), du_n large (propagation)

    We don't really test
        * angles - ConformingFracturePropagation returns 0.
        * rotated fractures
        * parameter robustness. This is sort of helped by choosing varying, non-integer
        values for u and parameters.

    """

    def setUp(self):
        self.model = pp.ConformingFracturePropagation({})
        self.model.mortar_displacement_variable = "mortar_variable"
        self.model.mechanics_parameter_key = "dummy_key"
        self.mu = 1.1
        self.poisson = 0.29

    def _assign_data(self, u_bot, u_top):
        """
        u_bot and u_top are the u_h in the neighbouring cells on the two sides of the fracture.
        Ordering is hard-coded according to the grids defined in _make_grid
        """
        u_h = np.zeros(self.nd * self.g_h.num_cells)
        if self.nd == 3:
            u_h[0:6] = u_bot.ravel(order="f")
            u_h[12:18] = u_top.ravel(order="f")
        else:
            u_h[2:6] = u_bot.ravel(order="f")
            u_h[10:14] = u_top.ravel(order="f")

        # Project to interface
        e = (self.g_h, self.g_l)
        d_j = self.gb.edge_props(e)
        mg = d_j["mortar_grid"]
        trace = np.abs(pp.fvutils.vector_divergence(self.g_h)).T
        u_j = mg.primary_to_mortar_avg(nd=self.nd) * trace * u_h
        pp.set_iterate(d_j, {self.model.mortar_displacement_variable: u_j})

        # Parameters used by the propagation class
        for g, d in self.gb:
            if g.dim == self.nd:
                param = {"shear_modulus": self.mu, "poisson_ratio": self.poisson}
            else:
                param = {"SIFs_critical": np.ones((self.nd, g.num_faces))}
            pp.initialize_data(g, d, self.model.mechanics_parameter_key, param)

    def test_tension_2d(self):
        """ One cell purely tensile and one closed.
        Normal magnitude implies propagation.
        """
        self._make_grid(dim=2)
        # Define u and which of the two tips are expected to propagate
        u_bot = np.array([[0, 1], [3, 1]])
        u_top = np.array([[0, 1], [3, 4]])
        propagate = np.array([False, True])

        self._assign_data(u_bot, u_top)

        sifs, propagate_faces = self._compute_sifs(u_top, u_bot, propagate)
        angles = np.zeros(self.g_l.num_faces)
        self._verify(sifs, propagate_faces, angles)

    def test_shear_2d(self):
        """ Both tip cells have tangential jump, only the right one has normal
        Normal magnitude implies propagation.
        """
        # Make grid
        self._make_grid(dim=2)
        u_bot = np.array([[0, 0], [2, 1]])
        u_top = np.array([[1, 1], [2, 3]])
        propagate = np.array([False, True])
        self._assign_data(u_bot, u_top)

        # Face-wise for g_l
        sifs, propagate_faces = self._compute_sifs(u_top, u_bot, propagate)
        angles = np.zeros(self.g_l.num_faces)
        self._verify(sifs, propagate_faces, angles)

    def test_tension_3d(self):
        """ One cell purely tensile and one closed.
        Normal magnitude implies propagation.
        """
        self._make_grid(dim=3)
        # One cell purely tensile and one closed
        u_bot = np.array([[2, 1], [2, 3.3], [2, 4]])
        u_top = np.array([[2, 1], [2, 3.3], [3, 4]])
        propagate = np.array([True, False])
        self._assign_data(u_bot, u_top)

        sifs, propagate_faces = self._compute_sifs(u_top, u_bot, propagate)
        angles = np.zeros(self.g_l.num_faces)
        self._verify(sifs, propagate_faces, angles)

    def test_shear_3d(self):
        """ Both tip cells have tangential jump, only the right one has normal
        Normal magnitude implies NO propagation.
        """
        # Make grid
        self._make_grid(dim=3)
        # Both tip cells have tangential jump, only the first one has normal
        u_bot = np.array([[2, 1], [2.1, 3.3], [2, 4]])
        u_top = np.array([[2, 1.1], [2, 3.3], [2.1, 4]])
        # The small normal opening does not suffice to overcome the critical SIF.
        propagate = np.array([False, False])
        self._assign_data(u_bot, u_top)

        sifs, propagate_faces = self._compute_sifs(u_top, u_bot, propagate)
        angles = np.zeros(self.g_l.num_faces)
        self._verify(sifs, propagate_faces, angles)

    def _compute_sifs(self, u_top, u_bot, propagate):
        """ Compute face-wise SIFs and propagation tags
        Critical that this is correct
        """
        du = u_top[-1] - u_bot[-1]
        g_l = self.g_l
        sifs = np.zeros((self.nd, g_l.num_faces))
        if self.nd == 2:
            R_d = 1 / 8
            face_ind = np.argsort(self.g_l.face_centers[0])
            ind_0 = face_ind[0]
            ind_1 = face_ind[-1]
        else:
            R_d = 1 / 4
            ind_0 = np.argsort(
                pp.distances.point_pointset(
                    g_l.face_centers, np.array([0.25, 0.5, 0.5])
                )
            )[0]
            ind_1 = np.argsort(
                pp.distances.point_pointset(
                    g_l.face_centers, np.array([0.75, 0.5, 0.5])
                )
            )[0]

        K = np.sqrt(2 * np.pi / R_d) * self.mu / (3 - 4 * self.poisson + 1) * du
        sifs[0, ind_0] = K[0]
        sifs[0, ind_1] = K[1]

        # Tags for propagation
        propagate_faces = np.zeros(g_l.num_faces, dtype=bool)
        propagate_faces[ind_0] = propagate[0]
        propagate_faces[ind_1] = propagate[1]
        return sifs, propagate_faces

    def _make_grid(self, dim):
        if dim == 3:
            gb = setup_gb.grid_3d_2d()
        else:
            gb = setup_gb.grid_2d_1d([4, 2], 0.25, 0.75)
        gb.compute_geometry()

        pp.contact_conditions.set_projections(gb)

        # Model needs to know dimension
        self.model._Nd = dim
        # Set for convenient access without passing around:
        self.nd = dim
        self.gb = gb
        self.g_h = gb.grids_of_dimension(dim)[0]
        self.g_l = gb.grids_of_dimension(dim - 1)[0]

    def _verify(self, sifs, propagate, angles):
        data_h = self.gb.node_props(self.g_h)
        data_l = self.gb.node_props(self.g_l)
        p_l = data_l[pp.PARAMETERS][self.model.mechanics_parameter_key]
        data_edge = self.gb.edge_props((self.g_h, self.g_l))
        self.model._displacement_correlation(self.g_l, data_h, data_l, data_edge)

        self.model._propagation_criterion(data_l)

        self.model._angle_criterion(data_l)

        self.assertTrue(np.all(np.isclose(sifs, p_l["SIFs"])))
        self.assertTrue(np.all(np.isclose(propagate, p_l["propagate_faces"])))
        self.assertTrue(np.all(np.isclose(angles, p_l["propagation_angle_normal"])))
        # For pure mode I, we also have
        self.assertTrue(np.all(np.isclose(sifs[0], p_l["SIFs_equivalent"])))


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
        dof_manager = pp.DofManager(gb)
        assembler = pp.Assembler(gb, dof_manager)
        model.assembler = assembler

        # Define and initialize a state vector
        x = np.zeros(dof_manager.full_dof.sum())
        x[dof_manager.dof_ind(g_2d, self.cv2)] = cell_val_2d
        for g in g_1d:
            x[dof_manager.dof_ind(g, self.cv1)] = cell_val_1d[g]
            x[dof_manager.dof_ind((g_2d, g), self.mv)] = cell_val_mortar[g]

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
                np.all(x_new[dof_manager.dof_ind(g_2d, self.cv2)] == cell_val_2d)
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
                x_1d = x_new[dof_manager.dof_ind(g, self.cv1)]

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
                x_new[dof_manager.dof_ind(g, self.cv1)] = np.r_[
                    val_1d_prev[g], extended_1d + 1
                ]
                val_1d_prev[g] = x_new[dof_manager.dof_ind(g, self.cv1)]
                d[pp.STATE][self.cv1] = x_new[dof_manager.dof_ind(g, self.cv1)]
                val_1d_iterate_prev[g] = np.r_[val_1d_iterate_prev[g], extended_1d + 1]
                d[pp.STATE][pp.ITERATE][self.cv1] = val_1d_iterate_prev[g]

                ## Check mortar grid - see 1d case above for comments
                x_mortar = x_new[dof_manager.dof_ind((g_2d, g), self.mv)]

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

                x_new[dof_manager.dof_ind((g_2d, g), self.mv)] = np.r_[
                    val_mortar_prev[g][:sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                    val_mortar_prev[g][sz : 2 * sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                ]
                val_mortar_prev[g] = x_new[dof_manager.dof_ind((g_2d, g), self.mv)]
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
