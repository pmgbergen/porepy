"""
Tests of the propagation of fractures.

Content:
  * test_pick_propagation_face_conforming_propagation: Given a critically stressed
    fracture tip in the lower-dimensional grid, find which face in the
    higher-dimensional grid should be split. It employs the ConformingPropagation
    strategy.
  * FaceSplittingHostGrid: Various consistency tests for splitting faces in the higher
    dimensional grid, resembling propagation.
  * PropagationCriteria: Tests i) SIF computation by displacement correlation, ii) tags
    for which tips to propagate iii) propagation angles (only asserts all angles are
    zero).
  * VariableMappingInitializationUpdate: Checks that mapping of variables and
    assignment of new variable values in the FracturePropagation classes are correct.

Cases specifications for test_pick_propagation_face_conforming_propagation :

Case 1: Define a 2D medium that has a single fracture. Propagate first from both ends
and then from only one end. Since this is (almost) the simplest possible case, if the
test fails, either something fundamentally wrong has occurred with the propagation or a
basic assumption has been violated.

Case 2: An example of a simple 3D case with one fracture. The first step propagates in
two
 directions. In the second step, the fracture propagates in three directions, including
 both newly formed faces and the original fracture.

Case 3: An example of a simple 3D case with two fractures. As in case 2, the propagation
steps are similar.

"""

from typing import List, Union

import numpy as np
import pytest

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.applications.test_utils.arrays import compare_arrays

FractureNetwork = Union[FractureNetwork2d, FractureNetwork3d]

# @TODO: Notice that this is integration test collection. A set of unit test is needed.
# @TODO: Unit tests are required for conforming_propagation.py and propagation_model.py


def domains(case: int) -> pp.Domain:
    """Domain for each case"""
    if case == 1:
        return pp.Domain({"xmax": 4.0, "ymax": 3.0})
    elif case == 2:
        return pp.Domain({"xmax": 2.0, "ymax": 3.0, "zmax": 3.0})
    elif case == 3:
        return pp.Domain({"xmax": 4.0, "ymax": 3.0, "zmax": 3.0})


def geometry_fracture(case: int) -> List[np.array]:
    """Disjoint fractures by case"""
    if case == 1:
        data: List[np.array] = [np.array([[1, 2], [1, 1]])]
    elif case == 2:
        data: List[np.array] = [np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 1, 2, 2]])]
    elif case == 3:
        data: List[np.array] = [
            np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 1, 2, 2]]),
            np.array([[3, 3, 3, 3], [1, 2, 2, 1], [1, 1, 2, 2]]),
        ]
    return data


def generate_mdg(case: int):
    """Generates mgd for each case"""

    grid_type = "cartesian"
    domain: pp.Domain = domains(case)
    meshing_args: dict = {"cell_size": 1.0}
    fracture_geometry = geometry_fracture(case)
    if domain.dim == 2:
        disjoint_fractures = list(map(pp.LineFracture, fracture_geometry))
    else:
        disjoint_fractures = list(map(pp.PlaneFracture, fracture_geometry))
    fracture_network = pp.create_fracture_network(disjoint_fractures, domain)
    mdg = pp.create_mdg(grid_type, meshing_args, fracture_network)
    return mdg


def retrieve_md_targets_cells_and_angles(case: int):
    """Retrieves mgd, target cells, and angles involved in the propagation for each case."""

    mdg = generate_mdg(case)
    if case == 1:
        sd_c1 = mdg.subdomains(dim=mdg.dim_max() - 1)[0]
        # Target cells
        targets = [{sd_c1: np.array([21, 19])}, {sd_c1: np.array([22])}]
        angles = [{sd_c1: np.array([0, 0])}, {sd_c1: np.array([0])}]
    elif case == 2:
        sd_c1 = mdg.subdomains(dim=mdg.dim_max() - 1)[0]
        # Target cells
        targets = [{sd_c1: np.array([10, 4])}, {sd_c1: np.array([1, 7, 16])}]
        angles = [{sd_c1: np.array([0, 0])}, {sd_c1: np.array([0, 0, 0])}]
    elif case == 3:
        sd1_c1 = mdg.subdomains(dim=mdg.dim_max() - 1)[0]
        sd2_c1 = mdg.subdomains(dim=mdg.dim_max() - 1)[1]
        targets = [
            {sd1_c1: np.array([16, 6]), sd2_c1: np.array([], dtype=int)},
            {sd1_c1: np.array([1, 11, 26]), sd2_c1: np.array([8])},
        ]

        angles = [
            {sd1_c1: np.array([0, 0]), sd2_c1: np.array([], dtype=int)},
            {sd1_c1: np.array([0, 0, 0]), sd2_c1: np.array([0])},
        ]

    return mdg, targets, angles


def grid_3d_2d(nx=[2, 2, 2], x_start=0, x_stop=1):
    """
    Make the simplest possible fractured grid: 2d unit square with one horizontal
    fracture extending from 0 to fracture_x <= 1.

    """
    eps = 1e-10
    assert x_stop < 1 + eps and x_stop > -eps
    assert x_start < 1 + eps and x_stop > -eps

    f = np.array(
        [[x_start, x_stop, x_stop, x_start], [0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]
    )
    mdg = pp.meshing.cart_grid([f], nx, **{"physdims": [1, 1, 1]})
    return mdg


@pytest.mark.parametrize("case", [1, 2, 3])
def test_pick_propagation_face_conforming_propagation(case):
    """Verify that the right high-dimensional face is picked, based on tagged faces
    on the tips of the lower-dimensional grid. Thus, the test probes the function
    _pick_propagation_faces() in ConformingFracturePropagation.

    The overall idea of the test is to give a MixedDimensionalGrid, and a sequence of
    target faces in sd_primary to be split, based on face indices. For each target face,
    the closest (by distance) of the tip faces in g_l is found, and tagged for
    propagation. We then call the function to be tested to identify the face in
    sd_primary which should be propagated, and verify that the identified face is indeed
    the target.

    A key assumption in the above reasoning is that the propagation angle is known, and
    fixed to zero; non-zero values can also be accommodated, but that case is outside
    the standard coverage of the current functionality for fracture propagation.

    """

    mech_key = "mechanics"

    # Get problem geometry and description
    mdg, targets, angles = retrieve_md_targets_cells_and_angles(case)
    pp.set_local_coordinate_projections(mdg)

    # Bookkeeping
    sd_primary = mdg.subdomains(dim=mdg.dim_max())[0]
    data_primary = mdg.subdomain_data(sd_primary)
    data_primary[pp.TIME_STEP_SOLUTIONS] = {}

    # Propagation model; assign this some necessary fields.
    model = pp.ConformingFracturePropagation({"times_to_export": []})
    model.mechanics_parameter_key = mech_key
    model.nd = mdg.dim_max()

    # Loop over all propagation steps
    for step_target, step_angle in zip(targets, angles):
        # Map for faces to be split for all fracture grids
        propagation_map = {}

        # Loop over all fracture grids
        for sd_frac, data_frac in mdg.subdomains(
            dim=mdg.dim_max() - 1, return_data=True
        ):
            # Data dictionaries
            intf = mdg.subdomain_pair_to_interface((sd_primary, sd_frac))
            intf_data = mdg.interface_data(intf)

            # Faces on the tip o the fracture
            tip_faces = np.where(sd_frac.tags["tip_faces"])[0]

            # Data structure to tag tip faces to propagate from, and angles
            prop_face = np.zeros(sd_frac.num_faces, dtype=bool)
            prop_angle = np.zeros(sd_frac.num_faces)

            # Loop over all targets for this grid, tag faces to propagate
            for ti, ang in zip(step_target[sd_frac], step_angle[sd_frac]):
                # Coordinate of the face center of the target face
                fch = sd_primary.face_centers[:, ti].reshape((-1, 1))
                # Find the face in g_l (among tip faces) that is closest to this face
                hit = np.argmin(
                    np.sum((sd_frac.face_centers[:, tip_faces] - fch) ** 2, axis=0)
                )
                # Tag this face for propagation.
                prop_face[tip_faces[hit]] = 1
                prop_angle[tip_faces[hit]] = ang

            # Store information on what to propagate for this fracture
            data_frac[pp.PARAMETERS] = {
                mech_key: {
                    "propagation_angle_normal": prop_angle,
                    "propagate_faces": prop_face,
                }
            }

            # Use the functionality to find faces to split
            model._pick_propagation_faces(
                sd_primary, sd_frac, data_primary, data_frac, intf_data
            )
            _, face, _ = sparse_array_to_row_col_data(intf_data["propagation_face_map"])

            # This is the test, the targets and the identified faces should be the same
            assert np.all(np.sort(face) == np.sort(step_target[sd_frac]))

            # Store propagation map for this face - this is necessary to split the faces
            # and do the test for multiple propagation steps
            propagation_map[sd_frac] = face

        # We have updated the propagation map for all fractures, and can split faces in
        # sd_primary
        pp.propagate_fracture.propagate_fractures(mdg, propagation_map)
        # ... on to the next propagation step.


class TestFaceSplittingHostGrid:
    """Tests of splitting of faces in the higher-dimensional grid.

    The different tests have been written at different points in time, and could have
    benefited from a consolidation. On the other hand, the tests also cover different
    components of the propagation.

    """

    def test_equivalence_2d(self):
        """
        This central test checks that certain geometry, connectivity and tags are
        independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different initial fractures)
        4 grown in two steps
        """

        # Generate mixed-dimensional grids
        mdg_1, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            {"cell_size_x": 0.25, "cell_size_y": 0.5},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.0, 0.75])],
        )
        mdg_2, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            {"cell_size_x": 0.25, "cell_size_y": 0.5},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.0, 0.25])],
        )
        mdg_3, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            {"cell_size_x": 0.25, "cell_size_y": 0.5},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.0, 0.5])],
        )
        mdg_4, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            {"cell_size_x": 0.25, "cell_size_y": 0.5},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.0, 0.25])],
        )

        # Pick out the 1d grids in each bucket. These will be used to set which faces
        # in the 2d grid to split
        g2 = mdg_2.subdomains(dim=1)[0]
        g3 = mdg_3.subdomains(dim=1)[0]
        g4 = mdg_4.subdomains(dim=1)[0]

        # Do the splitting
        pp.propagate_fracture.propagate_fractures(mdg_2, {g2: np.array([15, 16])})
        pp.propagate_fracture.propagate_fractures(mdg_3, {g3: np.array([16])})
        # The fourth bucket is split twice
        pp.propagate_fracture.propagate_fractures(mdg_4, {g4: np.array([15])})
        pp.propagate_fracture.propagate_fractures(mdg_4, {g4: np.array([16])})

        # Check that the four grid buckets are equivalent
        _check_equivalent_md_grids([mdg_1, mdg_2, mdg_3, mdg_4])

    def test_equivalence_3d(self):
        """
        3d version of previous test.
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different initial fractures)
        4 grown in two steps
        """
        # Make buckets
        mdg_1 = grid_3d_2d([4, 2, 2], 0, 0.75)
        mdg_2 = grid_3d_2d([4, 2, 2], 0, 0.25)
        mdg_3 = grid_3d_2d([4, 2, 2], 0, 0.50)
        mdg_4 = grid_3d_2d([4, 2, 2], 0, 0.25)

        # Pick out the 2d grids in each bucket. These will be used to set which faces
        # in the 3d grid to split
        g2 = mdg_2.subdomains(dim=2)[0]
        g3 = mdg_3.subdomains(dim=2)[0]
        g4 = mdg_4.subdomains(dim=2)[0]

        # Do the splitting
        pp.propagate_fracture.propagate_fractures(mdg_2, {g2: np.array([53, 54])})
        pp.propagate_fracture.propagate_fractures(mdg_3, {g3: np.array([54])})
        # The fourth bucket is split twice
        pp.propagate_fracture.propagate_fractures(mdg_4, {g4: np.array([53])})
        pp.propagate_fracture.propagate_fractures(mdg_4, {g4: np.array([54])})

        # Check that the four grid buckets are equivalent
        _check_equivalent_md_grids([mdg_1, mdg_2, mdg_3, mdg_4])

    def test_two_fractures_2d(self):
        """Two fractures growing towards each other, but not meeting.

        Tests simultaneous growth of two fractures in multiple steps, and growth of one
        fracture in the presence of an inactive one.
        """
        f_1 = np.array([[0, 0.25], [0.5, 0.5]])
        f_2 = np.array([[1.75, 2], [0.5, 0.5]])
        mdg = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 0.5], [0.5, 0.5]])
        f_2 = np.array([[1.5, 2], [0.5, 0.5]])
        mdg_1 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 0.75], [0.5, 0.5]])
        f_2 = np.array([[1.25, 2], [0.5, 0.5]])
        mdg_2 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 1.0], [0.5, 0.5]])
        mdg_3 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        # Get the fracture grids
        sd_1 = mdg.subdomains(dim=1)[0]
        g2 = mdg.subdomains(dim=1)[1]

        # First propagation step
        faces = {sd_1: np.array([27]), g2: np.array([32])}
        pp.propagate_fracture.propagate_fractures(mdg, faces)
        _check_equivalent_md_grids([mdg, mdg_1])

        # Second step
        faces = {sd_1: np.array([28]), g2: np.array([31])}
        pp.propagate_fracture.propagate_fractures(mdg, faces)
        _check_equivalent_md_grids([mdg, mdg_2])

        # Final step - only one of the fractures grow
        faces = {sd_1: np.array([29]), g2: np.array([], dtype=int)}
        pp.propagate_fracture.propagate_fractures(mdg, faces)
        _check_equivalent_md_grids([mdg, mdg_3])

    def test_two_propagation_steps_3d(self):
        # Starting from a fracture one cell wide, three cells long, the first step
        # opens faces in the middle of the fracture, so that it looks roughly like
        #  O O O   <- Initial fracture
        #    0     <- face opened in the first step
        #
        # The second step requests opening of the two flanking faces
        mdg = self._make_grid()
        g_frac = mdg.subdomains(dim=2)[0]

        faces_to_split = [[43], [41, 45]]

        g = mdg.subdomains(dim=2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes

        gh = mdg.subdomains(dim=3)[0]

        new_fc = [
            np.array([[1.5, 2, 1.5], [1, 1.5, 2], [1.5, 2, 1.5]]),
            np.array([[1.5, 2, 2, 1.5], [0, 0.5, 2.5, 3], [1.5, 2, 2, 1.5]]),
        ]

        new_cell_volumes = [np.array([np.sqrt(2)]), np.sqrt(2) * np.ones(2)]

        # The first splitting will create no new nodes (only more tip nodes)
        # Second splitting generates 8 new nodes, since the domain is split in two
        num_nodes = gh.num_nodes + np.array([0, 8])

        for si, split in enumerate(faces_to_split):
            pp.propagate_fracture.propagate_fractures(mdg, {g_frac: np.array(split)})
            cc, fc, cv = self._verify(
                mdg, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    def test_three_from_bottom(self):
        # Starting from a fracture one cell wide, three cells long, the first step
        # opens one flanking face of the fracture, so that it looks roughly like
        #  O O O   <- Initial fracture
        #  0       <- face opened in the first step
        #
        # The second step requests the opening of the middle face, the final step
        # the second flaking face.
        mdg = self._make_grid()
        g_frac = mdg.subdomains(dim=2)[0]

        faces_to_split = [[41], [43], [45]]

        g = mdg.subdomains(dim=2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes

        gh = mdg.subdomains(dim=3)[0]

        new_fc = [
            np.array([[1.5, 2, 1.5], [0, 0.5, 1], [1.5, 2, 1.5]]),
            np.array([[2, 1.5], [1.5, 2], [2, 1.5]]),
            np.array([[2, 1.5], [2.5, 3], [2, 1.5]]),
        ]

        new_cell_volumes = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]

        num_nodes = gh.num_nodes + np.array([2, 4, 8])

        for si, split in enumerate(faces_to_split):
            pp.propagate_fracture.propagate_fractures(mdg, {g_frac: np.array(split)})
            cc, fc, cv = self._verify(
                mdg, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    def test_open_face_from_two_sides_simultaneously(self):
        # Starting from a fracture one cell wide, three cells long, the first step
        # opens faces on the flanks of the fracture, so that it looks roughly like
        #  O O O   <- Initial fracture
        #  O   O   <- faces opened in the first step
        #
        # The second step requests opening of the middle face from both sides.
        mdg = self._make_grid()
        g_frac = mdg.subdomains(dim=2)[0]

        faces_to_split = [[41, 45], [43, 43]]

        g = mdg.subdomains(dim=2)[0]
        cc = g.cell_centers
        fc = g.face_centers
        cv = g.cell_volumes

        gh = mdg.subdomains(dim=3)[0]

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
            pp.propagate_fracture.propagate_fractures(mdg, {g_frac: np.array(split)})
            cc, fc, cv = self._verify(
                mdg, split, cc, fc, cv, new_cell_volumes[si], new_fc[si], num_nodes[si]
            )

    #### Helpers below

    def _verify(self, mdg, split, cc, fc, cv, new_cell_volumes, new_fc, num_nodes):
        # Check that the geometry of a (propagated) MixedDimensionalGrid corresponds to
        # a given known geometry.

        gh = mdg.subdomains(dim=mdg.dim_max())[0]
        g = mdg.subdomains(dim=mdg.dim_max() - 1)[0]
        new_cc = gh.face_centers[:, split]
        if len(split) == 1:
            new_cc = new_cc.reshape((-1, 1))
        cc = np.append(cc, new_cc, axis=1)
        compare_arrays(g.cell_centers, cc)

        fc = np.append(fc, new_fc, axis=1)
        compare_arrays(g.face_centers, fc)

        cv = np.append(cv, new_cell_volumes)

        assert np.allclose(g.cell_volumes, cv)

        proj = mdg.subdomain_data(g)["tangential_normal_projection"]
        assert proj.normals.shape[1] == g.num_cells
        assert np.logical_or(np.all(proj.normals[2] < 0), np.all(proj.normals[2] > 0))

        hit = np.logical_and(g.face_centers[0] > 1.1, g.face_centers[0] < 1.9)
        assert np.allclose(g.face_areas[hit], np.sqrt(2))
        assert np.allclose(g.face_areas[np.logical_not(hit)], 1)

        assert num_nodes == gh.num_nodes

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
        mdg = pp.meshing.cart_grid([frac_1], [2, 3, 2])

        for sd in mdg.subdomains():
            hit = sd.nodes[0] > 1.5
            sd.nodes[2, hit] += 1

        mdg.compute_geometry()

        pp.set_local_coordinate_projections(mdg)

        return mdg


class MockPropagationModel(pp.ConformingFracturePropagation):
    """Helper class for variable mapping tests.

    Define a propagation model with a tailored method to populate new fields in
    variables. Should be easy to verify implementation. Subclass
    ConformingFracturePropagation, which is the simplest known non-ABC propagation
    class.
    """

    def _initialize_new_variable_values(self, g, d, var, dofs):
        cell_dof: int = dofs.get("cells")
        # Number of new variables is given by the size of the cell map.
        cell_map = d["cell_index_map"]
        n_new = cell_map.shape[0] - cell_map.shape[1]

        # Fill the new values with an arbitrary number for instance 42, because of course
        vals = np.full(n_new * cell_dof, 42)
        return vals


class TestPropagationCriteria:
    """Test of functionality to compute sifs and evaluate propagation onset and angle.

    Test logic:
        1. Set up 2d or 3d mdg with single fracture.
        2. Assign displacement and minimal parameters. To help assigning to the right
        side of the interface, the former are specified according to cells in sd_primary
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

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = pp.ConformingFracturePropagation({})
        self.model.mortar_displacement_variable = "mortar_variable"
        self.model.mechanics_parameter_key = "dummy_key"
        self.mu = 1.1
        self.poisson = 0.29

    def _assign_data(self, u_bot, u_top):
        """
        u_bot and u_top are the u_h in the neighbouring cells on the two sides of the
        fracture. Ordering is hard-coded according to the grids defined in _make_grid
        """
        u_h = np.zeros(self.nd * self.sd_primary.num_cells)
        if self.nd == 3:
            u_h[0:6] = u_bot.ravel(order="f")
            u_h[12:18] = u_top.ravel(order="f")
        else:
            u_h[2:6] = u_bot.ravel(order="f")
            u_h[10:14] = u_top.ravel(order="f")

        # Project to interface
        intf = self.mdg.subdomain_pair_to_interface((self.sd_primary, self.g_l))
        d_j = self.mdg.interface_data(intf)
        trace = self.sd_primary.trace(dim=self.nd)
        u_j = intf.primary_to_mortar_avg(nd=self.nd) * trace * u_h
        pp.set_solution_values(
            name=self.model.mortar_displacement_variable,
            values=u_j,
            data=d_j,
            iterate_index=0,
        )

        # Parameters used by the propagation class
        for g, d in self.mdg.subdomains(return_data=True):
            if g.dim == self.nd:
                param = {"shear_modulus": self.mu, "poisson_ratio": self.poisson}
            else:
                param = {"SIFs_critical": np.ones((self.nd, g.num_faces))}
            pp.initialize_data(g, d, self.model.mechanics_parameter_key, param)

    def test_tension_2d(self):
        """One cell purely tensile and one closed.
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
        """Both tip cells have tangential jump, only the right one has normal
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
        """One cell purely tensile and one closed.
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
        """Both tip cells have tangential jump, only the right one has normal
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
        """Compute face-wise SIFs and propagation tags
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
            mdg = grid_3d_2d()
        else:
            mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
                "cartesian",
                {"cell_size_x": 0.25, "cell_size_y": 0.5},
                fracture_indices=[1],
                fracture_endpoints=[np.array([0.25, 0.75])],
            )
        mdg.compute_geometry()

        pp.set_local_coordinate_projections(mdg)

        # Model needs to know dimension
        self.model.nd = dim
        # Set for convenient access without passing around:
        self.nd = dim
        self.mdg = mdg
        self.sd_primary = mdg.subdomains(dim=dim)[0]
        self.g_l = mdg.subdomains(dim=dim - 1)[0]

    def _verify(self, sifs, propagate, angles):
        data_h = self.mdg.subdomain_data(self.sd_primary)
        data_l = self.mdg.subdomain_data(self.g_l)
        p_l = data_l[pp.PARAMETERS][self.model.mechanics_parameter_key]
        intf = self.mdg.subdomain_pair_to_interface((self.sd_primary, self.g_l))
        data_edge = self.mdg.interface_data(intf)
        self.model._displacement_correlation(self.g_l, intf, data_h, data_l, data_edge)

        self.model._propagation_criterion(data_l)

        self.model._angle_criterion(data_l)

        assert np.all(np.isclose(sifs, p_l["SIFs"]))
        assert np.all(np.isclose(propagate, p_l["propagate_faces"]))
        assert np.all(np.isclose(angles, p_l["propagation_angle_normal"]))
        # For pure mode I, we also have
        assert np.all(np.isclose(sifs[0], p_l["SIFs_equivalent"]))


class TestVariableMappingInitializationUnderPropagation:
    """Test of functionality to update variables during propagation.

    In reality, three features are tested: First, the mappings for cell and face
    variables, generated by fracture propagation, second the usage of these mappings to
    update variables, and third the functionality to initialize new variables.

    Three cases are considered (all 2d, 3d should raise no further complication):
        1. Update on a domain a single fracture.
        2. Update on a domain with two fractures, both propagating.
        3. Update on a domain with two fractures, only one propagating.

    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.cv2 = "cell_variable_2d"
        self.cv1 = "cell_variable_1d"
        self.mv = "mortar_variable"

    def test_single_fracture_propagate(self):
        # Domain with 5x2 cells. The fracture is initially one face long, and will be
        # propagated to be three faces long, in two steps.
        frac = [np.array([[1, 2], [1, 1]])]
        mdg = pp.meshing.cart_grid(frac, [5, 2])

        g_frac = mdg.subdomains(dim=1)[0]
        # Two prolongation steps for the fracture.
        split_faces = [{g_frac: np.array([19])}, {g_frac: np.array([20])}]
        self._verify(mdg, split_faces)

    def test_two_fractures_propagate_both(self):
        # Domain with 5x3 cells, with two fractures.
        # Both fractures are initially one face long, and will be propagated in two steps

        frac = [np.array([[1, 2], [1, 1]]), np.array([[2, 3], [2, 2]])]
        mdg = pp.meshing.cart_grid(frac, [5, 3])

        g_1, g_2 = mdg.subdomains(dim=1)

        # Two prolongation steps for the fracture.
        # In the first step, one fracture grows, one is stuck.
        # In second step, the second fracture grows in both ends
        split_faces = [
            {g_1: np.array([25]), g_2: np.array([])},
            {g_1: np.array([26]), g_2: np.array([29, 31])},
        ]

        self._verify(mdg, split_faces)

    def _verify(self, mdg, split_faces):
        # Common function for all tests.
        # Propagate fracture, verify that variables are mapped correctly, and that
        # new variables are added with correct values

        # Individual grids
        g_2d = mdg.subdomains(dim=2)[0]
        g_1d = mdg.subdomains(dim=1)

        # Model used for fracture propagation
        model = MockPropagationModel({})
        model.mdg = mdg
        equation_system = pp.EquationSystem(mdg)
        model.equation_system = equation_system

        # self,
        # name: str,
        # dof_info: Optional[dict[GridEntity, int]] = None,
        # subdomains: Optional[list[pp.Grid]] = None,
        # interfaces: Optional[list[pp.MortarGrid]] = None,
        # tags: Optional[dict[str, Any]] = None,
        # Cell variable in 2d. Should stay constant throughout the process.
        cell_val_2d = np.random.rand(g_2d.num_cells)

        ## Initialize variables for all grids
        # Cell variable in 1d. Should be expanded, but initial size is num_vars_2d
        var_sz_1d = 2
        # 1d variables defined as a dict, indexed on the grid
        cell_val_1d = {g: np.random.rand(var_sz_1d) for g in g_1d}

        var_sz_mortar = 3
        cell_val_mortar = {g: np.random.rand(var_sz_mortar * 2) for g in g_1d}

        # Define variables on all grids.
        # Initialize the state by the known variable, and the iterate as twice that
        # value (mostly a why not)
        d = mdg.subdomain_data(g_2d)
        equation_system.create_variables(self.cv2, {"cells": 1}, [g_2d])

        val_sol = cell_val_2d
        val_it = 2 * cell_val_2d

        pp.set_solution_values(name=self.cv2, values=val_sol, data=d, time_step_index=0)
        pp.set_solution_values(name=self.cv2, values=val_it, data=d, iterate_index=0)

        for g in g_1d:
            d = mdg.subdomain_data(g)
            equation_system.create_variables(self.cv1, {"cells": var_sz_1d}, [g])

            val_sol = cell_val_1d[g]
            val_it = 2 * cell_val_1d[g]

            pp.set_solution_values(
                name=self.cv1, values=val_sol, data=d, time_step_index=0
            )
            pp.set_solution_values(
                name=self.cv1, values=val_it, data=d, iterate_index=0
            )

            intf = mdg.subdomain_pair_to_interface((g_2d, g))

            d = mdg.interface_data(intf)
            equation_system.create_variables(
                self.mv, {"cells": var_sz_mortar}, interfaces=[intf]
            )

            val_sol = cell_val_mortar[g]
            val_it = 2 * cell_val_mortar[g]

            pp.set_solution_values(
                name=self.mv, values=val_sol, data=d, time_step_index=0
            )
            pp.set_solution_values(name=self.mv, values=val_it, data=d, iterate_index=0)

        # Define and initialize a state vector
        x = np.zeros(equation_system.num_dofs())
        var_2d = equation_system.get_variables([self.cv2], [g_2d])
        x[equation_system.dofs_of(var_2d)] = cell_val_2d
        for g in g_1d:
            var_1d = equation_system.get_variables([self.cv1], [g])
            x[equation_system.dofs_of(var_1d)] = cell_val_1d[g]
            intf = mdg.subdomain_pair_to_interface((g_2d, g))
            var_mortar = equation_system.get_variables([self.mv], [intf])
            x[equation_system.dofs_of(var_mortar)] = cell_val_mortar[g]

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
            pp.propagate_fracture.propagate_fractures(mdg, split)

            # Update variables
            x_new = model._map_variables(x)

            # The values of the 2d cell should not change

            assert np.all(x_new[equation_system.dofs_of(var_2d)] == cell_val_2d)
            # Also check that pp.TIME_STEP_SOLUTIONS and ITERATES has been correctly
            # updated
            d = mdg.subdomain_data(g_2d)
            time_step_values_cv2 = pp.get_solution_values(
                name=self.cv2, data=d, time_step_index=0
            )
            assert np.all(time_step_values_cv2 == cell_val_2d)

            iterate_values_cv2 = pp.get_solution_values(
                name=self.cv2, data=d, iterate_index=0
            )
            assert np.all(iterate_values_cv2 == 2 * cell_val_2d)

            # Loop over all 1d grids, check both grid and the associated mortar grid
            for g in g_1d:
                num_new_cells = split[g].size

                # mapped variable
                var_1d = equation_system.get_variables([self.cv1], [g])
                x_1d = x_new[equation_system.dofs_of(var_1d)]

                # Extension of the 1d grid. All values should be 42 (see propagation
                # class)
                extended_1d = np.full(var_sz_1d * num_new_cells, 42)

                # True values for 1d (both grid and iterate)
                truth_1d = np.r_[val_1d_prev[g], extended_1d]
                truth_iterate = np.r_[val_1d_iterate_prev[g], extended_1d]
                assert np.allclose(x_1d, truth_1d)

                # Also check that pp.TIME_STEP_SOLUTIONS and ITERATE has been correctly
                # updated
                d = mdg.subdomain_data(g)

                time_step_values_cv1 = pp.get_solution_values(
                    name=self.cv1, data=d, time_step_index=0
                )
                assert np.all(time_step_values_cv1 == truth_1d)

                iterate_values_cv1 = pp.get_solution_values(
                    name=self.cv1, data=d, iterate_index=0
                )
                assert np.all(iterate_values_cv1 == truth_iterate)

                # The propagation model will assign the value 42 to new cells also in
                # the next step. To be sure values from the first propagation are mapped
                # correctly, we alter the true value (add 1), and update this both in
                # the solution vector, time step solutions and previous iterate
                # solutions.
                var_1d = equation_system.get_variables([self.cv1], [g])
                x_new[equation_system.dofs_of(var_1d)] = np.r_[
                    val_1d_prev[g], extended_1d + 1
                ]
                val_1d_prev[g] = x_new[equation_system.dofs_of(var_1d)]
                val_1d_iterate_prev[g] = np.r_[val_1d_iterate_prev[g], extended_1d + 1]

                pp.set_solution_values(
                    name=self.cv1, values=val_1d_prev[g], data=d, time_step_index=0
                )
                pp.set_solution_values(
                    name=self.cv1,
                    values=val_1d_iterate_prev[g],
                    data=d,
                    iterate_index=0,
                )

                ## Check mortar grid - see 1d case above for comments
                intf = mdg.subdomain_pair_to_interface((g_2d, g))
                var_intf = equation_system.get_variables([self.mv], [intf])
                x_mortar = x_new[equation_system.dofs_of(var_intf)]

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
                d = mdg.interface_data(intf)

                assert np.all(x_mortar == truth_mortar)
                assert np.all(
                    pp.get_solution_values(name=self.mv, data=d, time_step_index=0)
                    == truth_mortar
                )
                assert np.all(
                    pp.get_solution_values(name=self.mv, data=d, iterate_index=0)
                    == truth_iterate
                )
                x_new[equation_system.dofs_of(var_intf)] = np.r_[
                    val_mortar_prev[g][:sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                    val_mortar_prev[g][sz : 2 * sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                ]
                val_mortar_prev[g] = x_new[equation_system.dofs_of(var_intf)]
                val_mortar_iterate_prev[g] = np.r_[
                    val_mortar_iterate_prev[g][:sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                    val_mortar_iterate_prev[g][sz : 2 * sz],
                    np.full(var_sz_mortar * num_new_cells, 43),
                ]

                pp.set_solution_values(
                    name=self.mv, values=val_mortar_prev[g], data=d, time_step_index=0
                )
                pp.set_solution_values(
                    name=self.mv,
                    values=val_mortar_iterate_prev[g],
                    data=d,
                    iterate_index=0,
                )

                x = x_new


# Helper functions for the tests below


def _check_equivalent_md_grids(md_grids, decimals=12):
    """
    Checks agreement between number of cells, faces and nodes, their
    coordinates and the connectivity matrices cell_faces and face_nodes. Also
    checks the face tags.

    """
    dim_h = md_grids[0].dim_max()
    dim_l = dim_h - 1
    num_md_grids = len(md_grids)
    cell_maps_h, face_maps_h = [], []
    cell_maps_l, face_maps_l = num_md_grids * [{}], num_md_grids * [{}]

    # Check that all md-grids have the same number of grids in the lower dimension
    num_grids_l: int = len(md_grids[0].subdomains(dim=dim_h - 1))
    for mdg in md_grids:
        assert len(mdg.subdomains(dim=dim_h - 1)) == num_grids_l

    for dim in range(dim_l, dim_h + 1):
        for target_grid in range(len(md_grids[0].subdomains(dim=dim))):
            n_cells, n_faces, n_nodes = np.empty(0), np.empty(0), np.empty(0)
            nodes, face_centers, cell_centers = [], [], []
            cell_faces, face_nodes = [], []
            for mdg in md_grids:
                sd = mdg.subdomains(dim=dim)[target_grid]
                n_cells = np.append(n_cells, sd.num_cells)
                n_faces = np.append(n_faces, sd.num_faces)
                n_nodes = np.append(n_nodes, sd.num_nodes)
                cell_faces.append(sd.cell_faces)
                face_nodes.append(sd.face_nodes)
                cell_centers.append(sd.cell_centers)
                face_centers.append(sd.face_centers)
                nodes.append(sd.nodes)

            # Check that all md-grids have the same number of cells, faces and nodes
            assert np.unique(n_cells).size == 1
            assert np.unique(n_faces).size == 1
            assert np.unique(n_nodes).size == 1

            # Check that the coordinates agree
            cell_centers = np.round(cell_centers, decimals)
            nodes = np.round(nodes, decimals)
            face_centers = np.round(face_centers, decimals)
            for i in range(1, num_md_grids):
                assert np.all(
                    pp.utils.setmembership.ismember_rows(
                        cell_centers[0], cell_centers[i]
                    )[0]
                )
                assert np.all(
                    pp.utils.setmembership.ismember_rows(
                        face_centers[0], face_centers[i]
                    )[0]
                )
                assert np.all(
                    pp.utils.setmembership.ismember_rows(nodes[0], nodes[i])[0]
                )

            # Now we know all nodes, faces and cells are in all grids, we map them
            # to prepare cell_faces and face_nodes comparison
            sd_0 = md_grids[0].subdomains(dim=dim)[target_grid]
            for i in range(1, num_md_grids):
                mdg = md_grids[i]
                sd = mdg.subdomains(dim=dim)[target_grid]
                cell_map, face_map, node_map = _make_maps(sd_0, sd, mdg.dim_max())
                mapped_cf = sd.cell_faces[face_map][:, cell_map]
                mapped_fn = sd.face_nodes[node_map][:, face_map]

                assert np.sum(np.abs(sd_0.cell_faces) != np.abs(mapped_cf)) == 0
                assert np.sum(np.abs(sd_0.face_nodes) != np.abs(mapped_fn)) == 0
                if sd.dim == dim_h:
                    face_maps_h.append(face_map)
                    cell_maps_h.append(cell_map)
                else:
                    cell_maps_l[i][sd] = cell_map
                    face_maps_l[i][sd] = face_map

                # Also loop on the standard face tags to check that they are
                # identical between the md-grids.
                tag_keys = pp.utils.tags.standard_face_tags()
                for key in tag_keys:
                    assert np.all(np.isclose(sd_0.tags[key], sd.tags[key][face_map]))

    # Mortar grids
    sd_primary_0 = md_grids[0].subdomains(dim=dim_h)[0]
    for target_grid in range(len(md_grids[0].subdomains(dim=dim_l))):
        sd_secondary_0 = md_grids[0].subdomains(dim=dim_l)[target_grid]
        intf_0 = md_grids[0].subdomain_pair_to_interface((sd_primary_0, sd_secondary_0))
        proj_0 = intf_0.primary_to_mortar_int()
        for i in range(1, num_md_grids):
            sd_secondary_i = md_grids[i].subdomains(dim=dim_l)[target_grid]
            sd_primary_i = md_grids[i].subdomains(dim=dim_h)[0]
            intf_i = md_grids[i].subdomain_pair_to_interface(
                (sd_primary_i, sd_secondary_i)
            )
            proj_i = intf_i.primary_to_mortar_int()
            cm = cell_maps_l[i][sd_secondary_i]
            cm_extended = np.append(cm, cm + cm.size)
            fm = face_maps_h[i - 1]
            mapped_fc = proj_i[cm_extended, :][:, fm]
            assert np.sum(np.absolute(proj_0) - np.absolute(mapped_fc)) == 0
    return cell_maps_h, cell_maps_l, face_maps_h, face_maps_l


def _make_maps(g0, g1, n_digits=8, offset=0.11):
    """
    Given two grid with the same nodes, faces and cells, the mappings between
    these entities are constructed. Handles non-unique nodes and faces on next
    to fractures by exploiting neighbour information.
    Builds maps from g1 to g0, so g1.x[x_map]=g0.x, e.g.

    g1.tags[some_key][face_map] = g0.tags[some_key].

    Parameters:
        g0: Reference grid
        g1: Other grid
        n_digits: Tolerance in rounding before coordinate comparison
        offset: Weight determining how far the fracture neighbour nodes and faces
            are shifted (normally away from fracture) to ensure unique coordinates.
    """
    cell_map = pp.utils.setmembership.ismember_rows(
        np.around(g0.cell_centers, n_digits),
        np.around(g1.cell_centers, n_digits),
        sort=False,
    )[1]
    # Make face_centers unique by dragging them slightly away from the fracture

    fc0 = g0.face_centers.copy()
    fc1 = g1.face_centers.copy()
    n0 = g0.nodes.copy()
    n1 = g1.nodes.copy()
    fi0 = g0.tags["fracture_faces"]
    if np.any(fi0):
        fi1 = g1.tags["fracture_faces"]
        d0 = np.reshape(np.tile(g0.cell_faces[fi0, :].data, 3), (3, sum(fi0)))
        fn0 = g0.face_normals[:, fi0] * d0
        d1 = np.reshape(np.tile(g1.cell_faces[fi1, :].data, 3), (3, sum(fi1)))
        fn1 = g1.face_normals[:, fi1] * d1
        fc0[:, fi0] += fn0 * offset
        fc1[:, fi1] += fn1 * offset
        (ni0, fid0) = g0.face_nodes[:, fi0].nonzero()
        (ni1, fid1) = g1.face_nodes[:, fi1].nonzero()
        un, inv = np.unique(ni0, return_inverse=True)
        for i, node in enumerate(un):
            n0[:, node] += offset * np.mean(fn0[:, fid0[inv == i]], axis=1)
        un, inv = np.unique(ni1, return_inverse=True)
        for i, node in enumerate(un):
            n1[:, node] += offset * np.mean(fn1[:, fid1[inv == i]], axis=1)

    face_map = pp.utils.setmembership.ismember_rows(
        np.around(fc0, n_digits), np.around(fc1, n_digits), sort=False
    )[1]

    node_map = pp.utils.setmembership.ismember_rows(
        np.around(n0, n_digits), np.around(n1, n_digits), sort=False
    )[1]
    return cell_map, face_map, node_map
