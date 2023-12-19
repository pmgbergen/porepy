# %%
# import numpy as np
# import scipy.sparse as sps

# import porepy as pp
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
class UnitTestAdTpfaFlux(AdTpfaFlux, pp.fluid_mass_balance.SinglePhaseFlow):
    def initial_condition(self):
        super().initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.set_solution_values(
                name=self.pressure_variable,
                values=np.array([2, 3]),
                data=data,
                iterate_index=0,
                time_step_index=0,
            )

    def _set_grid(self):
        g = pp.CartGrid([2, 1])
        g.nodes = np.array(
            [[0, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 2, 0], [3, 1, 0]]
        ).T
        g.compute_geometry()
        g.face_centers[0, 3] = 1.5
        g.cell_centers = np.array([[1, 0.5, 0], [2.5, 0.5, 0]]).T

        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains([g])
        self.mdg = mdg

    def prepare_simulation(self):
        super().prepare_simulation()

        self._set_grid()
        self.discretize(self.mdg)

    def _permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Non-constant permeability tensor. Depends on pressure."""
        nc = sum([sd.num_cells for sd in subdomains])
        # K is a second order tensor having nd^2 entries per cell. 3d:
        # Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
        # 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
        # 2d:
        # Kxx, Kxy, Kyx, Kyy
        # 0  , 1  , 2  , 3
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

        cell_0_projection = pp.ad.Matrix(sps.csr_matrix(np.array([[1, 0], [0, 0]])))
        cell_1_projection = pp.ad.Matrix(sps.csr_matrix(np.array([[0, 0], [0, 1]])))

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
            + pp.ad.Scalar(3) * e_yy @ cell_1_projection @ p**2
        )

        return (
            pp.wrap_as_dense_ad_array(all_vals, name="Constant_permeability_component")
            + cell_0_permeability
            + cell_1_permeability
        )

    def test_transmissibility_calculation(self, vector_source: bool = False):
        face_indices = np.array([0, 2, 3, 5, 6])
        cell_indices = np.array([0, 1, 0, 0, 1])

        g = self.mdg.subdomains()[0]

        pressure = self.pressure(self.mdg.subdomains()).value(self.equation_system)

        k0 = np.array([[1 + pressure[0], 0], [0, 2 + pressure[0] ** 2]])
        k1 = np.array(
            [
                [2 + pressure[1] ** 2, 1 + pressure[1] ** 2],
                [1 + pressure[1] ** 2, 3 + pressure[1] ** 2],
            ]
        )
        permeability = [k0, k1]

        k0_diff = np.array([[1, 0], [0, 2 * pressure[0]]])
        k1_diff = 2 * pressure[1] * np.ones((2, 2))
        permeability_diff = [k0_diff, k1_diff]

        computed_flux = darcy_flux(self.mdg).value_and_jacobian(self.equation_system)

        for fi, ci in zip(face_indices, cell_indices):
            n = g.face_normals[:2, fi]
            fc = g.face_centers[:2, fi]
            cc = g.cell_centers[:2, ci]
            p = pressure[ci]
            k = permeability[ci]
            k_diff = permeability_diff[ci]

            fc_cc = np.reshape(fc - cc, (2, 1))

            fc_cc_dist = np.linalg.norm(fc_cc)

            trm = -np.dot(n, np.dot(k, fc_cc) / np.power(fc_cc_dist, 2))
            trm_diff = -np.dot(n, np.dot(k_diff, fc_cc) / np.power(fc_cc_dist, 2))

            assert np.isclose(tmp * p, computed_flux.val[fi])

            diff = trm + trm_diff * p
            assert np.isclose(diff, computed_flux.jac[fi, ci])

            other_ci = 1 if ci == 0 else 0
            assert np.isclose(computed_flux.jac[fi, other_ci], 0)
