# %%
import numpy as np
import scipy.sparse as sps

import porepy as pp

# import pytest


# def _setup_cart_2d(nx, dir_faces=None):
#     g = pp.CartGrid(nx)
#     g.compute_geometry()
#     kxx = np.ones(g.num_cells)
#     perm = pp.SecondOrderTensor(kxx)
#     if dir_faces is None:
#         # If no Dirichlet faces are specified, set Dirichlet conditions on all faces.
#         dir_faces = g.tags["domain_boundary_faces"].nonzero()[0]
#     bound = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)
#     return g, perm, bound


# def _test_laplacian_stencil_cart_2d(discr_matrices_func):
#     """Apply TPFA or MPFA on Cartesian grid, should obtain Laplacian stencil."""
#     nx = np.array([3, 3])
#     dir_faces = np.array([0, 3, 12])
#     g, perm, bound = _setup_cart_2d(nx, dir_faces)
#     div, flux, bound_flux, _ = discr_matrices_func(g, perm, bound)
#     A = div * flux
#     b = -(div * bound_flux).A

#     # Checks on interior cell
#     mid = 4
#     assert A[mid, mid] == 4
#     assert A[mid - 1, mid] == -1
#     assert A[mid + 1, mid] == -1
#     assert A[mid - 3, mid] == -1
#     assert A[mid + 3, mid] == -1

#     # The first cell should have two Dirichlet bnds
#     assert A[0, 0] == 6
#     assert A[0, 1] == -1
#     assert A[0, 3] == -1

#     # Cell 3 has one Dirichlet, one Neumann face
#     assert A[2, 2] == 4
#     assert A[2, 1] == -1
#     assert A[2, 5] == -1

#     # Cell 2 has one Neumann face
#     assert A[1, 1] == 3
#     assert A[1, 0] == -1
#     assert A[1, 2] == -1
#     assert A[1, 4] == -1

#     assert b[1, 13] == -1


# class UnitTestAdTpfaFlux(  # type: ignore[misc]
#     pp.constitutive_laws.AdTpfaFlux, pp.fluid_mass_balance.SinglePhaseFlow
# ):
#     def initial_condition(self):
#         super().initial_condition()
#         for sd, data in self.mdg.subdomains(return_data=True):
#             pp.set_solution_values(
#                 name=self.pressure_variable,
#                 values=np.array([2, 3]),
#                 data=data,
#                 iterate_index=0,
#                 time_step_index=0,
#             )

#     def _set_grid(self):
#         g = pp.CartGrid([2, 1])
#         g.nodes = np.array(
#             [[0, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 2, 0], [3, 1, 0]]
#         ).T
#         g.compute_geometry()
#         g.face_centers[0, 3] = 1.5
#         g.cell_centers = np.array([[1, 0.5, 0], [2.5, 0.5, 0]]).T

#         mdg = pp.MixedDimensionalGrid()
#         mdg.add_subdomains([g])
#         self.mdg = mdg

#     def prepare_simulation(self):
#         super().prepare_simulation()

#         self._set_grid()
#         self.discretize(self.mdg)

#     def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
#         """Non-constant permeability tensor. Depends on pressure."""
#         nc = sum([sd.num_cells for sd in subdomains])
#         # K is a second order tensor having nd^2 entries per cell. 3d:
#         # Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
#         # 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
#         # 2d:
#         # Kxx, Kxy, Kyx, Kyy
#         # 0  , 1  , 2  , 3
#         tensor_dim = 3**2

#         # Set constant component of the permeability
#         all_vals = np.zeros(nc * tensor_dim, dtype=float)
#         all_vals[0] = 1
#         all_vals[4] = 2
#         all_vals[8] = 1
#         all_vals[9] = 2
#         all_vals[10] = 1
#         all_vals[12] = 1
#         all_vals[13] = 3
#         all_vals[17] = 1

#         cell_0_projection = pp.ad.Matrix(sps.csr_matrix(np.array([[1, 0], [0, 0]])))
#         cell_1_projection = pp.ad.Matrix(sps.csr_matrix(np.array([[0, 0], [0, 1]])))

#         e_xx = self.e_i(subdomains, i=0, dim=tensor_dim)
#         e_xy = self.e_i(subdomains, i=1, dim=tensor_dim)
#         e_yx = self.e_i(subdomains, i=3, dim=tensor_dim)
#         e_yy = self.e_i(subdomains, i=4, dim=tensor_dim)
#         p = self.pressure(subdomains)

#         # Non-constant component of the permeability in cell 0
#         cell_0_permeability = (
#             e_xx @ cell_0_projection @ p + e_yy @ cell_0_projection @ p**2
#         )
#         # Non-constant component of the permeability in cell 1
#         cell_1_permeability = (
#             pp.ad.Scalar(2) * e_xx @ cell_1_projection @ p**2
#             + pp.ad.Scalar(3) * e_yy @ cell_1_projection @ p**2
#         )

#         return (
#             pp.wrap_as_dense_ad_array(all_vals, name="Constant_permeability_component")
#             + cell_0_permeability
#             + cell_1_permeability
#         )


# class TestAdTpfaFlow(
#     pp.constitutive_laws.AdDarcyFlux,
#     pp.model_geometries.SquareDomainOrthogonalFractures,
#     pp.fluid_mass_balance.SinglePhaseFlow,
# ):
#     def initial_condition(self):
#         super().initial_condition()
#         for sd, data in self.mdg.subdomains(return_data=True):
#             pp.set_solution_values(
#                 name=self.pressure_variable,
#                 values=100 * np.ones(sd.num_cells),
#                 data=data,
#                 iterate_index=0,
#                 time_step_index=0,
#             )

#     def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
#         """Non-constant permeability tensor. Depends on pressure."""
#         nc = sum([sd.num_cells for sd in subdomains])
#         # K is a second order tensor having nd^2 entries per cell. 3d:
#         # Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
#         # 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
#         # 2d:
#         # Kxx, Kxy, Kyx, Kyy
#         # 0  , 1  , 2  , 3
#         tensor_dim = 3**2
#         all_vals = np.arange(nc * tensor_dim, dtype=float) + 1
#         # Set anisotropy by specifying the kyy entries
#         all_vals[self.nd + 1 :: tensor_dim] = 0.1 * (np.arange(nc) + 1)
#         scaled_vals = self.solid.convert_units(all_vals, "m^2")
#         e_xy = self.e_i(subdomains, i=1, dim=tensor_dim)
#         e_yy = self.e_i(subdomains, i=4, dim=tensor_dim)
#         p = self.pressure(subdomains)
#         return (
#             pp.wrap_as_dense_ad_array(scaled_vals, name="permeability")
#             + e_xy @ p
#             + pp.ad.Scalar(2) * e_yy @ p
#         )

#     def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
#         """Boundary condition values for Darcy flux.

#         Dirichlet boundary conditions are defined on the north and south boundaries,
#         with a constant value equal to the fluid's reference pressure (which will be 0
#         by default).

#         Parameters:
#             boundary_grid: Boundary grid for which to define boundary conditions.

#         Returns:
#             Boundary condition values array.

#         """
#         domain_sides = self.domain_boundary_sides(boundary_grid)
#         vals_loc = np.zeros(boundary_grid.num_cells)
#         vals_loc[domain_sides.north] = 200
#         vals_loc[domain_sides.south] = 100
#         return vals_loc

#     def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
#         """Boundary condition type for Darcy flux.

#         Dirichlet boundary conditions are defined on the north and south boundaries.

#         Parameters:
#             sd: Subdomain for which to define boundary conditions.

#         Returns:
#             bc: Boundary condition object.

#         """
#         domain_sides = self.domain_boundary_sides(sd)
#         # Define boundary condition on faces
#         return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

#     def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
#         """Discretization object for the Darcy flux term.

#         Parameters:
#             subdomains: List of subdomains where the Darcy flux is defined.

#         Returns:
#             Discretization of the Darcy flux.

#         """
#         # TODO: The ad.Discretizations may be purged altogether. Their current function
#         # is very similar to the ad.Geometry in that both basically wrap numpy/scipy
#         # arrays in ad arrays and collect them in a block matrix. This similarity could
#         # possibly be exploited. Revisit at some point.
#         return pp.ad.MpfaAd(self.darcy_keyword, subdomains)


# m = TestAdTpfaFlow({})
# m.prepare_simulation()
# g = m.mdg.subdomains()[0]
# g.nodes[:2, 0] += 0.1
# g.compute_geometry()
# m.set_discretization_parameters()
# m.discretize()  # This is needed to set up the discretization matrices. TODO: Fix in
# # the models.
# dummy = m.darcy_flux_discretization(m.mdg.subdomains()).flux
# dummy.discretize(m.mdg)


# o = m.darcy_flux(m.mdg.subdomains())
# t = o.value_and_jacobian(m.equation_system)
# p = m.pressure_trace(m.mdg.subdomains())
# pt = p.value_and_jacobian(m.equation_system)
# %%
class UnitTestAdTpfaFlux(
    pp.constitutive_laws.AdDarcyFlux, pp.fluid_mass_balance.SinglePhaseFlow
):
    """

    Notes for debugging:
        - Set the pressure to a constant value in the initial condition to test only the
          constant permeability component.
    """

    def __init__(self, params):
        super().__init__(params)
        self._neumann_face = 4

    def initial_condition(self):
        super().initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.set_solution_values(
                name=self.pressure_variable,
                values=np.array([2, 3], dtype=float),
                data=data,
                iterate_index=0,
                time_step_index=0,
            )

    def discretize(self):
        super().discretize()
        # Trick to compute the discretization matrices for the Darcy flux. This is done
        # automatically inside the constituitve laws if an Mpfa discretization is used,
        # but not for tpfa. In the latter case, the full discretization is computed as
        # part of the construction of Diff-Tpfa, and there is usually no need for a
        # separate construction of the transmissibility matrix. However, in this test we
        # will need it, so we force the computation here.
        dummy = self.darcy_flux_discretization(self.mdg.subdomains()).flux
        dummy.discretize(self.mdg)

    def set_geometry(self):
        # Create the geometry through domain amd fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        g = pp.CartGrid([2, 1])
        g.nodes = np.array(
            [[0, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 2, 0], [3, 1, 0]]
        ).T
        g.compute_geometry()
        g.face_centers[0, 3] = 1.5
        g.cell_centers = np.array([[1, 0.5, 0], [2.5, 0.5, 0]]).T

        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains([g])
        mdg.set_boundary_grid_projections()
        self.mdg = mdg
        self.nd = 2
        self.set_well_network()

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Non-constant permeability tensor. Depends on pressure."""
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0, size=0)

        nc = sum([sd.num_cells for sd in subdomains])
        # K is a second order tensor having nd^2 entries per cell. 3d:
        # Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
        # 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
        tensor_dim = 3**2

        # Set constant component of the permeability
        all_vals = np.zeros(nc * tensor_dim, dtype=float)
        all_vals[0] = 1
        all_vals[4] = 2
        all_vals[8] = 1
        all_vals[9] = 2
        all_vals[10] = 1
        all_vals[12] = 1
        all_vals[13] = 3
        all_vals[17] = 1

        cell_0_projection = pp.ad.SparseArray(
            sps.csr_matrix(np.array([[1, 0], [0, 0]]))
        )
        cell_1_projection = pp.ad.SparseArray(
            sps.csr_matrix(np.array([[0, 0], [0, 1]]))
        )

        e_xx = self.e_i(subdomains, i=0, dim=tensor_dim)
        e_xy = self.e_i(subdomains, i=1, dim=tensor_dim)
        e_yx = self.e_i(subdomains, i=3, dim=tensor_dim)
        e_yy = self.e_i(subdomains, i=4, dim=tensor_dim)
        p = self.pressure(subdomains)

        # Non-constant component of the permeability in cell 0
        cell_0_permeability = (
            e_xx @ cell_0_projection @ p + e_yy @ cell_0_projection @ p**2
        )
        # Non-constant component of the permeability in cell 1
        cell_1_permeability = (
            pp.ad.Scalar(2) * e_xx @ cell_1_projection @ p**2
            + e_xy @ cell_1_projection @ p
            + e_yx @ cell_1_projection @ p
            + pp.ad.Scalar(3) * e_yy @ cell_1_projection @ p**2
        )

        return (
            pp.wrap_as_dense_ad_array(all_vals, name="Constant_permeability_component")
            + cell_0_permeability
            + cell_1_permeability
        )

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        if self.params["base_discr"] == "tpfa":
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return pp.ad.MpfaAd(self.darcy_keyword, subdomains)

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned,
            requiring pressure values on the bonudary.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc_type = ["dir"] * boundary_faces.size

        hit = np.where(boundary_faces == self._neumann_face)[0][0]
        bc_type[hit] = "neu"

        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, bc_type)


def test_transmissibility_calculation(vector_source: bool = False, base_discr="tpfa"):
    # TODO: Parametrization of base_discr = ["tpfa", "mpfa"]
    model = UnitTestAdTpfaFlux({"base_discr": base_discr})
    model.prepare_simulation()

    face_indices = np.array([0, 2, 3, 5, 6])
    cell_indices = np.array([0, 1, 0, 0, 1])

    g = model.mdg.subdomains()[0]

    pressure = model.pressure(model.mdg.subdomains()).value(model.equation_system)

    k0 = np.array([[1 + pressure[0], 0], [0, 2 + pressure[0] ** 2]])
    k1 = np.array(
        [
            [2 + 2 * pressure[1] ** 2, 1 + pressure[1] ** 2],
            [1 + pressure[1] ** 2, 3 + 3 * pressure[1] ** 2],
        ]
    )
    permeability = [k0, k1]

    k0_diff = np.array([[1, 0], [0, 2 * pressure[0]]])
    k1_diff = 2 * pressure[1] * np.array([[2, 1], [1, 3]])
    permeability_diff = [k0_diff, k1_diff]

    computed_flux = model.darcy_flux(model.mdg.subdomains()).value_and_jacobian(
        model.equation_system
    )

    div = g.cell_faces.T

    data = model.mdg.subdomain_data(model.mdg.subdomains()[0])
    base_flux = data[pp.DISCRETIZATION_MATRICES][model.darcy_keyword]["flux"]

    def _compute_half_transmissibility_and_derivative(fi, ci):
        # Helper function to compute the half transmissibility (from cell center to
        # face) and its derivative for a single face-cell pair.
        n = g.face_normals[:2, fi]
        fc = g.face_centers[:2, fi]
        cc = g.cell_centers[:2, ci]
        k = permeability[ci]
        k_diff = permeability_diff[ci]

        fc_cc = np.reshape(fc - cc, (2, 1))

        fc_cc_dist = np.linalg.norm(fc_cc)

        trm = np.dot(n, np.dot(k, fc_cc) / np.power(fc_cc_dist, 2))
        trm_diff = np.dot(n, np.dot(k_diff, fc_cc) / np.power(fc_cc_dist, 2))
        return trm, trm_diff

    # Test flux calculation on boundary faces
    for fi, ci in zip(face_indices, cell_indices):
        p = pressure[ci]
        # Get half transmissibility and its derivative. There is no need to account for
        # sign changes (to reflect the direction of the normal vector of the face), as
        # this test only considers the numerical value of the transmissibility. The
        # effect of the sign change is tested elsewhere.
        trm, trm_diff = _compute_half_transmissibility_and_derivative(fi, ci)

        if base_discr == "tpfa":
            # If the base discretization is TPFA, we can directly compare the computed
            # flux with the calculated transmissibility times the pressure. This cannot
            # be done for mpfa, since 'trm' is a representation of the tpfa
            # transmissibility, not the mpfa one.
            assert np.isclose(trm * p, computed_flux.val[fi])

        # Fetch the transmissibility from the base discretization.
        diff = base_flux[fi].A.ravel()
        # Add tpfa-style contribution from the derivative of the transmissibility.
        diff[ci] += trm_diff * p
        assert np.allclose(diff, computed_flux.jac[fi].A.ravel())

    # On Neumann faces, the computed flux should be zero, as should the the
    # transmissibility
    assert computed_flux.val[model._neumann_face] == 0
    assert np.allclose(computed_flux.jac[model._neumann_face].A, 0)

    # Test flux calculation on internal face. This is a bit more involved, as we need to
    # compute the harmonic mean of the two transmissibilities and its derivative.
    fi = 1
    trm_0, trm_diff_0 = _compute_half_transmissibility_and_derivative(fi, 0)
    trm_1, trm_diff_1 = _compute_half_transmissibility_and_derivative(fi, 1)
    p0 = pressure[0]
    p1 = pressure[1]

    # Here we need to account for the sign change of the transmissibility.
    trm_0 *= div[0, fi]
    trm_1 *= div[1, fi]
    # Take the pressure difference between the two cells. Multiply with the sign of the
    # divergence for this face, to account for the direction of the normal vector (if
    # the normal vector is pointing into cell 1, div[1, fi] will be -1, thus we avoid
    # the situation where a positive pressure difference leads to a negative flux.
    p_diff = (p1 - p0) * div[1, fi]

    # Fetch the transmissibility from the base discretization.
    trm_full = base_flux[fi].A.ravel()
    # The derivative of the full transmissibility with respect to the two cell center
    # pressures, by the product rule.
    trm_diff_p0 = (
        trm_diff_0 * trm_1 / (trm_0 + trm_1)
        - trm_0 * trm_1 * trm_diff_0 / (trm_0 + trm_1) ** 2
    )
    trm_diff_p1 = (
        trm_diff_1 * trm_0 / (trm_0 + trm_1)
        - trm_1 * trm_0 * trm_diff_1 / (trm_0 + trm_1) ** 2
    )
    if base_discr == "tpfa":
        # If the base discretization is TPFA, we can directly compare the computed flux
        # with the calculated transmissibility times the pressure. This cannot be done
        # for mpfa, since 'trm' is a representation of the tpfa transmissibility, not
        # the mpfa one.
        assert np.isclose(trm_full.dot([p0, p1]), computed_flux.val[fi])

    trm_diff = np.array(
        [trm_full[0] + trm_diff_p0 * p_diff, trm_full[1] - trm_diff_p1 * p_diff]
    ).ravel()
    assert np.allclose(trm_diff, computed_flux.jac[fi].A)


test_transmissibility_calculation()
