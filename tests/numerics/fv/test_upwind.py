""" Various tests of the upwind discretization for transport problems on a single grid.

Both within a grid, and upwind coupling on mortar grids.

NOTE: Most tests check both the discretization matrix and the result from a call
to assemble_matrix_rhs (the latter functionality is in a sense outdated, but kept for
legacy reasons).

"""

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp


@pytest.fixture()
def flux_market():
    collection = {
        "test_1d_darcy_flux_positive": [2, 0, 0],
        "test_1d_darcy_flux_negative": [-2, 0, 0],
        "test_2d_cart_darcy_flux_positive": [2, 0, 0],
        "test_2d_cart_darcy_flux_negative": [-2, 0, 0],
        "test_3d_cart_darcy_flux_positive": [1, 0, 0],
        "test_3d_cart_darcy_flux_negative": [-1, 0, 0],
        "test_2d_simplex_darcy_flux_positive": [1, 0, 0],
        "test_2d_simplex_darcy_flux_negative": [-1, 0, 0],
        "test_tilted_1d_darcy_flux_positive": [1, 0, 0],
        "test_tilted_1d_darcy_flux_negative": [-1, 0, 0],
        "test_tilted_2d_cart_darcy_flux_positive": [1, 0, 0],
        "test_tilted_2d_cart_darcy_flux_negative": [-1, 0, 0],
        "test_tilted_2d_simplex_darcy_flux_positive": [1, 0, 0],
        "test_tilted_2d_simplex_darcy_flux_negative": [-1, 0, 0],
        "test_1d_darcy_flux_positive_rhs_dir": [2, 0, 0],
        "test_1d_darcy_flux_negative_rhs_dir": [-2, 0, 0],
    }
    return collection


@pytest.fixture()
def rotation_data_market():
    collection = {
        "test_1d_darcy_flux_positive": (0.0, [1, 0, 0]),
        "test_1d_darcy_flux_negative": (0.0, [1, 0, 0]),
        "test_2d_cart_darcy_flux_positive": (0.0, [1, 0, 0]),
        "test_2d_cart_darcy_flux_negative": (0.0, [1, 0, 0]),
        "test_3d_cart_darcy_flux_positive": (0.0, [1, 0, 0]),
        "test_3d_cart_darcy_flux_negative": (0.0, [1, 0, 0]),
        "test_2d_simplex_darcy_flux_positive": (0.0, [1, 0, 0]),
        "test_2d_simplex_darcy_flux_negative": (0.0, [1, 0, 0]),
        "test_tilted_1d_darcy_flux_positive": (-np.pi / 5.0, [0, 1, -1]),
        "test_tilted_1d_darcy_flux_negative": (-np.pi / 8.0, [-1, 1, -1]),
        "test_tilted_2d_cart_darcy_flux_positive": (np.pi / 4.0, [0, 1, 0]),
        "test_tilted_2d_cart_darcy_flux_negative": (np.pi / 6.0, [1, 1, 0]),
        "test_tilted_2d_simplex_darcy_flux_positive": (np.pi / 2.0, [1, 1, 0]),
        "test_tilted_2d_simplex_darcy_flux_negative": (-np.pi / 5.0, [1, 1, -1]),
        "test_1d_darcy_flux_positive_rhs_dir": (0.0, [1, 0, 0]),
        "test_1d_darcy_flux_negative_rhs_dir": (0.0, [1, 0, 0]),
    }
    return collection


@pytest.fixture()
def references_market():
    t1_lhs_known = np.array([[2, 0, 0], [-2, 2, 0], [0, -2, 0]])
    t1_deltaT_known = 1 / 12

    t2_lhs_known = np.array([[0, -2, 0], [0, 2, -2], [0, 0, 2]])
    t2_deltaT_known = 1 / 12

    t3_lhs_known = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, -1, 1, 0],
            [0, 0, 0, 0, -1, 0],
        ]
    )
    t3_deltaT_known = 1 / 12

    t4_lhs_known = np.array(
        [
            [0, -1, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 1, -1],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    t4_deltaT_known = 1 / 12

    t5_lhs_known = 0.25 * np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, -1, 0],
        ]
    )
    t5_deltaT_known = 1 / 4

    t6_lhs_known = 0.25 * np.array(
        [
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    t6_deltaT_known = 1 / 4

    t7_lhs_known = np.array([[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [-1, 0, 0, 1]])
    t7_deltaT_known = 1 / 6

    t8_lhs_known = np.array([[1, 0, 0, -1], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 1]])
    t8_deltaT_known = 1 / 6

    t9_lhs_known = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 0]])
    t9_deltaT_known = 1 / 6

    t10_lhs_known = np.array([[0, -1, 0], [0, 1, -1], [0, 0, 1]])
    t10_deltaT_known = 1 / 6

    t11_lhs_known = 0.5 * np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [-1, 1, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, -1, 1, 0],
            [0, 0, 0, 0, -1, 0],
        ]
    )
    t11_deltaT_known = 1 / 6

    t12_lhs_known = 0.5 * np.array(
        [
            [0, -1, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 1, -1],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    t12_deltaT_known = 1 / 6

    t13_lhs_known = np.array(
        [[1, -1, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [-1, 0, 0, 1]]
    )
    t13_deltaT_known = 1 / 6

    t14_lhs_known = np.array(
        [[1, 0, 0, -1], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 1]]
    )
    t14_deltaT_known = 1 / 6

    t15_lhs_known = np.array([[2, 0, 0], [-2, 2, 0], [0, -2, 2]])
    t15_deltaT_known = 1 / 12

    t16_lhs_known = np.array([[2, -2, 0], [0, 2, -2], [0, 0, 2]])
    t16_deltaT_known = 1 / 12

    collection = {
        "test_1d_darcy_flux_positive": (t1_lhs_known, t1_deltaT_known),
        "test_1d_darcy_flux_negative": (t2_lhs_known, t2_deltaT_known),
        "test_2d_cart_darcy_flux_positive": (t3_lhs_known, t3_deltaT_known),
        "test_2d_cart_darcy_flux_negative": (t4_lhs_known, t4_deltaT_known),
        "test_3d_cart_darcy_flux_positive": (t5_lhs_known, t5_deltaT_known),
        "test_3d_cart_darcy_flux_negative": (t6_lhs_known, t6_deltaT_known),
        "test_2d_simplex_darcy_flux_positive": (t7_lhs_known, t7_deltaT_known),
        "test_2d_simplex_darcy_flux_negative": (t8_lhs_known, t8_deltaT_known),
        "test_tilted_1d_darcy_flux_positive": (t9_lhs_known, t9_deltaT_known),
        "test_tilted_1d_darcy_flux_negative": (t10_lhs_known, t10_deltaT_known),
        "test_tilted_2d_cart_darcy_flux_positive": (
            t11_lhs_known,
            t11_deltaT_known,
        ),
        "test_tilted_2d_cart_darcy_flux_negative": (
            t12_lhs_known,
            t12_deltaT_known,
        ),
        "test_tilted_2d_simplex_darcy_flux_positive": (
            t13_lhs_known,
            t13_deltaT_known,
        ),
        "test_tilted_2d_simplex_darcy_flux_negative": (
            t14_lhs_known,
            t14_deltaT_known,
        ),
        "test_1d_darcy_flux_positive_rhs_dir": (t15_lhs_known, t15_deltaT_known),
        "test_1d_darcy_flux_negative_rhs_dir": (t16_lhs_known, t16_deltaT_known),
    }
    return collection


@pytest.fixture()
def rhs_market():
    collection = {
        "test_1d_darcy_flux_positive_rhs_dir": np.array([6, 0, 0]),
        "test_1d_darcy_flux_negative_rhs_dir": np.array([0, 0, 6]),
    }
    return collection


class TestUpwindDiscretization:
    """Tests some fixed-dimensional cases."""

    def _create_grid(self, grid_type, n_cells, phys_dims, angle, vector):
        if grid_type == "cartesian":
            sd = pp.CartGrid(n_cells, phys_dims)
        else:
            sd = pp.StructuredTriangleGrid(n_cells, phys_dims)
        R = pp.map_geometry.rotation_matrix(angle, vector)
        sd.nodes = np.dot(R, sd.nodes)
        sd.compute_geometry()
        return sd

    @pytest.fixture(autouse=True)
    def _retrieve_collections(
        self, flux_market, references_market, rotation_data_market, rhs_market
    ):
        self._flux_collection = flux_market
        self._reference_collection = references_market
        self._rotation_data_collection = rotation_data_market
        self._rhs_collection = rhs_market

    def _construct_discretization(self, sd, test_name, bc_type, angle, vector):
        R = pp.map_geometry.rotation_matrix(angle, vector)
        flux = self._flux_collection[test_name]
        upwind_obj = pp.Upwind()
        dis = upwind_obj.darcy_flux(sd, np.dot(R, flux))
        bf = sd.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(sd, bf, bf.size * [bc_type])
        if bc_type == "dir":
            bc_val = np.array([3, 0, 0, 3])
            specified_parameters = {"bc": bc, "bc_values": bc_val, "darcy_flux": dis}
        else:
            specified_parameters = {"bc": bc, "darcy_flux": dis}
        data = pp.initialize_default_data(sd, {}, "transport", specified_parameters)
        upwind_obj.discretize(sd, data)
        return upwind_obj, data

    def _compose_algebraic_representation(self, sd, upwind_obj, data):
        matrix_dictionary: dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
            upwind_obj.keyword
        ]
        upwind = matrix_dictionary[upwind_obj.upwind_matrix_key]
        param_dictionary: dict = data[pp.PARAMETERS][upwind_obj.keyword]
        flux_arr = param_dictionary[upwind_obj._flux_array_key]
        flux_mat = sps.dia_matrix((flux_arr, 0), shape=(sd.num_faces, sd.num_faces))
        div: sps.spmatrix = pp.fvutils.scalar_divergence(sd)

        bc_values: np.ndarray = param_dictionary["bc_values"]
        bc_discr_dir: sps.spmatrix = matrix_dictionary[
            upwind_obj.bound_transport_dir_matrix_key
        ]
        bc_discr_neu: sps.spmatrix = matrix_dictionary[
            upwind_obj.bound_transport_neu_matrix_key
        ]

        return (
            div * flux_mat * upwind,
            div * (bc_discr_neu + bc_discr_dir * flux_mat) * bc_values,
        )

    @pytest.mark.parametrize(
        "test_name, grid_type, bc_type, n_cells, phys_dims",
        [
            ("test_1d_darcy_flux_positive", "cartesian", "neu", 3, 1),
            ("test_1d_darcy_flux_negative", "cartesian", "neu", 3, 1),
            (
                "test_2d_cart_darcy_flux_positive",
                "cartesian",
                "neu",
                [3, 2],
                [1, 1],
            ),
            (
                "test_2d_cart_darcy_flux_negative",
                "cartesian",
                "neu",
                [3, 2],
                [1, 1],
            ),
            (
                "test_3d_cart_darcy_flux_positive",
                "cartesian",
                "neu",
                [2, 2, 2],
                [1, 1, 1],
            ),
            (
                "test_3d_cart_darcy_flux_negative",
                "cartesian",
                "neu",
                [2, 2, 2],
                [1, 1, 1],
            ),
            (
                "test_2d_simplex_darcy_flux_positive",
                "simplex",
                "neu",
                [2, 1],
                [1, 1],
            ),
            (
                "test_2d_simplex_darcy_flux_negative",
                "simplex",
                "neu",
                [2, 1],
                [1, 1],
            ),
            ("test_tilted_1d_darcy_flux_positive", "cartesian", "neu", 3, 1),
            ("test_tilted_1d_darcy_flux_negative", "cartesian", "neu", 3, 1),
            (
                "test_tilted_2d_cart_darcy_flux_positive",
                "cartesian",
                "neu",
                [3, 2],
                [1, 1],
            ),
            (
                "test_tilted_2d_cart_darcy_flux_negative",
                "cartesian",
                "neu",
                [3, 2],
                [1, 1],
            ),
            (
                "test_tilted_2d_simplex_darcy_flux_positive",
                "simplex",
                "neu",
                [2, 1],
                [1, 1],
            ),
            (
                "test_tilted_2d_simplex_darcy_flux_negative",
                "simplex",
                "neu",
                [2, 1],
                [1, 1],
            ),
            ("test_1d_darcy_flux_positive_rhs_dir", "cartesian", "dir", 3, 1),
            ("test_1d_darcy_flux_negative_rhs_dir", "cartesian", "dir", 3, 1),
        ],
    )
    def test_discretization(self, test_name, grid_type, bc_type, n_cells, phys_dims):
        angle, vector = self._rotation_data_collection[test_name]
        sd = self._create_grid(grid_type, n_cells, phys_dims, angle, vector)
        upwind_obj, data = self._construct_discretization(
            sd, test_name, bc_type, angle, vector
        )

        # compute matrix and delta_t
        lhs, rhs = self._compose_algebraic_representation(sd, upwind_obj, data)
        deltaT = upwind_obj.cfl(sd, data)

        # retrieve references
        lhs_known, deltaT_known = self._reference_collection[test_name]

        rtol = 1e-15
        atol = rtol
        assert np.allclose(lhs.todense(), lhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)
        if bc_type == "dir":
            # retrieve extra references
            rhs_known = self._rhs_collection[test_name]
            assert np.allclose(rhs, rhs_known, rtol, atol)
