from __future__ import division

import numpy as np
import scipy.sparse as sps
from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import spsolve as sparse_solver
import porepy as pp
from porepy.applications.test_utils import (
    reference_dense_arrays,
    reference_sparse_arrays,
)

"""Test collection for mixed-dimensional upwind discretization with multiple dimensional
    couplings."""


class TestUpwindCoupling:

    """Helper function for fill up dictionaries."""

    def _assign_discretization(self, mdg, disc, coupling_disc, variable):
        term = "advection"
        for _, data in mdg.subdomains(return_data=True):
            data[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
            data[pp.DISCRETIZATION] = {variable: {term: disc}}

        for intf, data in mdg.interfaces(return_data=True):
            sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
            data[pp.PRIMARY_VARIABLES] = {"lambda_u": {"cells": 1}}
            data[pp.COUPLING_DISCRETIZATION] = {
                variable: {
                    sd_primary: (variable, term),
                    sd_secondary: (variable, term),
                    intf: ("lambda_u", coupling_disc),
                }
            }

    """Helper function for adding a vector flux for faces."""

    def _add_constant_darcy_flux(self, mdg, upwind, flux, a):
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

        This method essentially mimics the functionality of the antiquated assembler class
        """

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
            lhs_loc, rhs_loc = upwind_coupling_obj.assemble_matrix_rhs(h_sd, l_sd, intf, h_data,
                                                                   l_data, data, bmat_loc)
            rhs_loc[0] = h_rhs
            rhs_loc[1] = l_rhs

            # block scatter lhs and rhs
            h_idx = hashes.index(hash(h_sd))
            l_idx = hashes.index(hash(l_sd))
            i_idx = hashes.index(hash(intf))
            dest = np.array([h_idx, l_idx, i_idx])
            for i, ib in enumerate(dest):
                if isinstance(rhs_loc[i], np.ndarray):
                    rhs[ib] = rhs_loc[i]
                else:
                    print("loc: ", rhs_loc)
                    rhs[ib] = np.array([rhs_loc[i]])
                for j, jb in enumerate(dest):
                    lhs[ib, jb] = lhs_loc[i, j]

        lhs = bmat(lhs)
        rhs = np.concatenate(rhs)
        return lhs, rhs


    def _assertion(self, lhs, rhs, theta, lhs_ref, rhs_ref, theta_ref):
        atol = rtol = 1e-15
        assert np.allclose(lhs, lhs_ref, rtol, atol)
        assert np.allclose(rhs, rhs_ref, rtol, atol)
        assert np.allclose(theta, theta_ref, rtol, atol)

    """Test the case of 2d domain with 1d fractures."""

    def test_2d_1d(self):
        mdg, _ = pp.md_grids_2d.two_intersecting([2, 2], simplex=False)

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
