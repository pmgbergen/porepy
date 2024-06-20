""" Various tests of the upwind discretization for transport problems on:
    - TestUpwind: for fixed dimensional problems
    - TestMixedDimensionalUpwind: for mixed-dimensional problems

NOTE: Most tests check both the discretization matrix and the result from a call
to assemble_matrix_rhs (the latter functionality is in a sense outdated, but kept for
legacy reasons).

"""
from __future__ import division

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve as sparse_solver

import porepy as pp
from porepy.applications.test_utils import (
    reference_dense_arrays,
    reference_sparse_arrays,
)


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
        "test_1d_darcy_flux_positive_rhs_dir": np.array([-6, 0, 0]),
        "test_1d_darcy_flux_negative_rhs_dir": np.array([0, 0, -6]),
    }
    return collection


class TestUpwind:
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
        """Set up and discretize a transport problem with upwind boundary conditions.

        Parameters:
            sd (Grid): Grid on which to discretize.
            test_name (str): Name of the test, used to retrieve flux and rotation data.
            bc_type (str): Type of boundary condition, either "dir" or "neu".
            angle (float): Angle of rotation of the flow field.
            vector (np.ndarray): Vector around which to rotate the flow field.

        """
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
        """Create discretization matrix and rhs vector from upwind object and data.

        This method essentially mimics the functionality of the assemble_matrix_rhs
        method in the upwind discretization class.
        """
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
        """Test the discretization matrix and delta_t for a single grid."""

        # create grid and discretize
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


class TestMixedDimensionalUpwind:
    """Test the discretization and assembly functionality of the mixed-dimensional
    upwind discretization.

    The tests are based on the the following steps:
        1. Specify a geometry and advection field
        2. Discretize the mixed-dimensional problem
        3. Assemble the discretization matrix and rhs vector for the md-problem.
        4. Compare the result to a reference solution.

    There is a tacit assumption that the order of the subdomains and interfaces in the
    md-grid is the same as the order of the subdomains and interfaces in the reference
    solution. In the md-grid, the order is dictated by the order in which fractures
    are specified and fed to the meshing algorithm. If this ever changes, these tests
    may break. Violations of this assumption will manifest as the assembled matrices
    and rhs vectors being permuted.

    """

    """Helper function for fill up dictionaries."""

    def _assign_discretization(self, mdg, disc, coupling_disc, variable):
        term = "advection"
        for _, data in mdg.subdomains(return_data=True):
            data[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
            data[pp.DISCRETIZATION] = {variable: {term: disc}}

        for intf, data in mdg.interfaces(return_data=True):
            sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
            data[pp.PRIMARY_VARIABLES] = {"lambda_u": {"cells": 1}}
            # Use the old Assembler-style specification of the coupling discretization
            data[pp.COUPLING_DISCRETIZATION] = {
                variable: {
                    sd_primary: (variable, term),
                    sd_secondary: (variable, term),
                    intf: ("lambda_u", coupling_disc),
                }
            }

    """Helper function for adding a vector flux for faces."""

    def _add_constant_darcy_flux(self, mdg, upwind, flux, a):
        # Add a constant darcy flux to the subdomains and interfaces
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            data[pp.PARAMETERS]["transport"]["darcy_flux"] = upwind.darcy_flux(
                sd, flux, aperture
            )
        for intf, data in mdg.interfaces(return_data=True):
            sd_primary, _ = mdg.interface_to_subdomain_pair(intf)
            param_primary = mdg.subdomain_data(sd_primary)[pp.PARAMETERS]
            darcy_flux = param_primary["transport"]["darcy_flux"]
            sign = np.zeros(sd_primary.num_faces)
            boundary_faces = sd_primary.get_all_boundary_faces()
            boundary_signs, _ = sd_primary.signs_and_cells_of_boundary_faces(
                boundary_faces
            )
            sign[boundary_faces] = boundary_signs

            sign = intf.primary_to_mortar_avg() * sign
            darcy_flux_e = sign * (intf.primary_to_mortar_avg() * darcy_flux)
            if pp.PARAMETERS not in data:
                data[pp.PARAMETERS] = pp.Parameters(
                    intf, ["transport"], [{"darcy_flux": darcy_flux_e}]
                )
            else:
                data[pp.PARAMETERS]["transport"]["darcy_flux"] = darcy_flux_e

    def _compose_algebraic_representation(self, mdg, upwind_obj, upwind_coupling_obj):
        """Create discretization matrix and rhs vector from mdg, upwind and
            upwind_coupling objects.

        This method essentially mimics the functionality of the antiquated assembler
        class.

        """

        # Objects such as grids and interface grids are perfectly hashable. In order to
        # perform block scatter operations, we have compiled a list of hashes here to
        # create an integer-based indexation map.
        sd_hashes = [hash(sd) for sd in mdg.subdomains()]
        intf_hashes = [hash(intf) for intf in mdg.interfaces()]
        hashes = sd_hashes + intf_hashes

        n_blocks = len(hashes)
        lhs = np.empty((n_blocks, n_blocks), dtype=sps.csr_matrix)
        rhs = np.empty(n_blocks, dtype=object)
        for intf, data in mdg.interfaces(return_data=True):
            h_sd, l_sd = mdg.interface_to_subdomain_pair(intf)
            h_data, l_data = mdg.subdomain_data(h_sd), mdg.subdomain_data(l_sd)
            upwind_obj.discretize(h_sd, h_data)
            upwind_obj.discretize(l_sd, l_data)
            upwind_coupling_obj.discretize(h_sd, l_sd, intf, h_data, l_data, data)

            bmat_loc = np.empty((3, 3), dtype=sps.csr_matrix)
            blocks = [h_sd.num_cells, l_sd.num_cells, intf.num_cells]
            for ib, isize in enumerate(blocks):
                for jb, jsize in enumerate(blocks):
                    bmat_loc[ib, jb] = sps.csr_matrix((isize, jsize))

            h_lhs, h_rhs = upwind_obj.assemble_matrix_rhs(h_sd, h_data)
            l_lhs, l_rhs = upwind_obj.assemble_matrix_rhs(l_sd, l_data)
            bmat_loc[0, 0] = h_lhs
            bmat_loc[1, 1] = l_lhs
            lhs_loc, rhs_loc = upwind_coupling_obj.assemble_matrix_rhs(
                h_sd, l_sd, intf, h_data, l_data, data, bmat_loc
            )
            rhs_loc[0] = h_rhs
            rhs_loc[1] = l_rhs

            # block scatter
            # The block scatter operation takes the discretization matrix associated with
            # each subdomain and inserts it into a sparse block structure (lhs and rhs) to
            # construct the final algebraic representation.
            h_idx = hashes.index(hash(h_sd))
            l_idx = hashes.index(hash(l_sd))
            i_idx = hashes.index(hash(intf))
            dest = np.array([h_idx, l_idx, i_idx])
            for i, ib in enumerate(dest):
                if isinstance(rhs_loc[i], np.ndarray):
                    rhs[ib] = rhs_loc[i]
                else:
                    rhs[ib] = np.array([rhs_loc[i]])
                for j, jb in enumerate(dest):
                    lhs[ib, jb] = lhs_loc[i, j]

        # The final algebraic representation
        lhs = bmat(lhs)
        rhs = np.concatenate(rhs)
        return lhs, rhs

    def _assertion(self, lhs, rhs, theta, lhs_ref, rhs_ref, theta_ref):
        """Check that the lhs, rhs and theta are close to the reference values.

        See class documentation for a comment on the order of the subdomains and
        interfaces.
        """
        atol = rtol = 1e-15
        assert np.allclose(lhs, lhs_ref, rtol, atol)
        assert np.allclose(rhs, rhs_ref, rtol, atol)
        assert np.allclose(theta, theta_ref, rtol, atol)

    """Test the case of 2d domain with 1d fractures."""

    def test_2d_1d(self):
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            {"cell_size": 0.5},
            fracture_indices=[0, 1],
        )

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        self._assign_discretization(mdg, upwind, upwind_coupling, variable)

        # define parameters
        tol = 1e-3
        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = sd.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(sd.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(sd, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})
            else:
                bound = pp.BoundaryCondition(sd, np.empty(0), np.empty(0))
                specified_parameters.update({"bc": bound})
            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        self._add_constant_darcy_flux(mdg, upwind, [1, 0, 0], a)
        lhs, rhs = self._compose_algebraic_representation(mdg, upwind, upwind_coupling)
        theta = sparse_solver(lhs, rhs)

        lhs_ref = reference_dense_arrays.test_upwind_coupling["test_2d_1d"]["lhs"]
        rhs_ref = reference_dense_arrays.test_upwind_coupling["test_2d_1d"]["rhs"]
        theta_ref = reference_dense_arrays.test_upwind_coupling["test_2d_1d"]["theta"]
        self._assertion(lhs.todense(), rhs, theta, lhs_ref, rhs_ref, theta_ref)

    """Test the case of 3d domain with 2d fractures."""

    def test_3d_2d(self):
        f = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        mdg = pp.meshing.cart_grid([f], [1, 1, 2], **{"physdims": [1, 1, 1]})
        mdg.compute_geometry()

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        self._assign_discretization(mdg, upwind, upwind_coupling, variable)

        # assign parameters
        tol = 1e-3
        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}

            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            bound_face_centers = sd.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[np.logical_or(left, right)] = ["dir"]

            bc_val = np.zeros(sd.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            bound = pp.BoundaryCondition(sd, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        self._add_constant_darcy_flux(mdg, upwind, [1, 0, 0], a)
        lhs, rhs = self._compose_algebraic_representation(mdg, upwind, upwind_coupling)
        theta = sparse_solver(lhs, rhs)

        lhs_ref = reference_dense_arrays.test_upwind_coupling["test_3d_2d"]["lhs"]
        rhs_ref = reference_dense_arrays.test_upwind_coupling["test_3d_2d"]["rhs"]
        theta_ref = reference_dense_arrays.test_upwind_coupling["test_3d_2d"]["theta"]

        self._assertion(lhs.todense(), rhs, theta, lhs_ref, rhs_ref, theta_ref)

    """Test the case of 3d domain with 2d fractures with multiple intersections down to
        dimension 0."""

    def test_3d_2d_1d_0d(self):
        f1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        f2 = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 1, 0], [0, 0, 1, 1]])
        f3 = np.array([[0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]])

        mdg = pp.meshing.cart_grid([f1, f2, f3], [2, 2, 2], **{"physdims": [1, 1, 1]})
        mdg.compute_geometry()

        # define discretization
        key = "transport"
        upwind = pp.Upwind(key)
        upwind_coupling = pp.UpwindCoupling(key)
        variable = "T"
        self._assign_discretization(mdg, upwind, upwind_coupling, variable)

        # assign parameters
        tol = 1e-3
        a = 1e-2
        for sd, data in mdg.subdomains(return_data=True):
            aperture = np.ones(sd.num_cells) * np.power(a, mdg.dim_max() - sd.dim)
            specified_parameters = {"aperture": aperture}
            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = sd.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(left, right)] = ["dir"]

                bc_val = np.zeros(sd.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                bound = pp.BoundaryCondition(sd, bound_faces, labels)
                specified_parameters.update({"bc": bound, "bc_values": bc_val})

            pp.initialize_default_data(sd, data, "transport", specified_parameters)

        for intf, data in mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, "transport", {})

        self._add_constant_darcy_flux(mdg, upwind, [1, 0, 0], a)
        lhs, rhs = self._compose_algebraic_representation(mdg, upwind, upwind_coupling)
        theta = sparse_solver(lhs, rhs)

        lhs_ref = reference_sparse_arrays.test_upwind_coupling["test_3d_2d_1d_0d"][
            "lhs"
        ]
        rhs_ref = reference_dense_arrays.test_upwind_coupling["test_3d_2d_1d_0d"]["rhs"]
        theta_ref = reference_dense_arrays.test_upwind_coupling["test_3d_2d_1d_0d"][
            "theta"
        ]

        self._assertion(
            lhs.todense(), rhs, theta, lhs_ref.todense(), rhs_ref, theta_ref
        )
