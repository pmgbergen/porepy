# %%
from typing import Any, Callable, Literal, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp
import pytest


def _setup_cart_2d(nx, dir_faces=None):
    g = pp.CartGrid(nx)
    g.compute_geometry()
    kxx = np.ones(g.num_cells)
    perm = pp.SecondOrderTensor(kxx)
    if dir_faces is None:
        # If no Dirichlet faces are specified, set Dirichlet conditions on all faces.
        dir_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bound = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)
    return g, perm, bound


def _test_laplacian_stencil_cart_2d(discr_matrices_func):
    """Apply TPFA or MPFA on Cartesian grid, should obtain Laplacian stencil."""
    nx = np.array([3, 3])
    dir_faces = np.array([0, 3, 12])
    g, perm, bound = _setup_cart_2d(nx, dir_faces)
    div, flux, bound_flux, _ = discr_matrices_func(g, perm, bound)
    A = div * flux
    b = -(div * bound_flux).A

    # Checks on interior cell
    mid = 4
    assert A[mid, mid] == 4
    assert A[mid - 1, mid] == -1
    assert A[mid + 1, mid] == -1
    assert A[mid - 3, mid] == -1
    assert A[mid + 3, mid] == -1

    # The first cell should have two Dirichlet bnds
    assert A[0, 0] == 6
    assert A[0, 1] == -1
    assert A[0, 3] == -1

    # Cell 3 has one Dirichlet, one Neumann face
    assert A[2, 2] == 4
    assert A[2, 1] == -1
    assert A[2, 5] == -1

    # Cell 2 has one Neumann face
    assert A[1, 1] == 3
    assert A[1, 0] == -1
    assert A[1, 2] == -1
    assert A[1, 4] == -1

    assert b[1, 13] == -1


class AdTpfaFlow(pp.fluid_mass_balance.SinglePhaseFlow):
    def initial_condition(self):
        super().initial_condition()
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.set_solution_values(
                name=self.pressure_variable,
                values=100 * np.ones(sd.num_cells),
                data=data,
                iterate_index=0,
                time_step_index=0,
            )

    def _permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        nc = sum([sd.num_cells for sd in subdomains])
        # K is a second order tensor having nd^2 entries per cell. 3d:
        # Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
        # 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
        # 2d:
        # Kxx, Kxy, Kyx, Kyy
        # 0  , 1  , 2  , 3
        tensor_dim = 3**2
        all_vals = np.arange(nc * tensor_dim, dtype=float) + 1
        # Set anisotropy by specifying the kyy entries
        all_vals[self.nd + 1 :: tensor_dim] = 0.1 * (np.arange(nc) + 1)
        scaled_vals = self.solid.convert_units(all_vals, "m^2")
        e_xy = self.e_i(subdomains, i=1, dim=tensor_dim)
        e_yy = self.e_i(subdomains, i=4, dim=tensor_dim)
        p = self.pressure(subdomains)
        return (
            pp.wrap_as_dense_ad_array(scaled_vals, name="permeability")
            + e_xy @ p
            + pp.ad.Scalar(2) * e_yy @ p
        )
        # return pp.wrap_as_dense_ad_array(np.ones(nc * tensor_dim), name="permeability")

    def diffusive_flux(
        self,
        domains: pp.SubdomainsOrBoundaries,
        diffusivity_tensor: Callable[[list[pp.Grid]], pp.ad.Operator],
    ) -> pp.ad.Operator:
        """Discretization of Darcy's law.



        Parameters:
            domains: List of domains where the Darcy flux is defined.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Face-wise Darcy flux in cubic meters per second.

        """

        if len(domains) == 0 or all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=self.bc_data_darcy_flux_key,
                domains=domains,
            )

        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(domains, [1])
        intf_projection = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=1)

        (
            t_f_full,
            diff_discr,
            boundary_grids,
            hf_to_f,
            d_vec,
        ) = self._transmissibility_matrix(domains)

        one = pp.ad.Scalar(1)

        # Delete neu values in T_f, i.e. keep all non-neu values.
        dir_filter, neu_filter = diff_discr.boundary_filters(
            domains, boundary_grids, "bc_values_darcy_flux"
        )
        # Keep t_f_full for now, to be used in the pressure reconstruction.
        t_f = (one - neu_filter) * t_f_full

        # Sign of boundary faces.
        bnd_sgn = diff_discr.boundary_sign(domains)
        # Discretization of boundary conditions: On Neumann faces, we will simply add
        # the flux, with a sign change if the normal vector is pointing inwards.
        neu_bnd = neu_filter * bnd_sgn
        # On Dirichlet faces, the assigned Dirichlet value corresponds to a flux of
        # magnitude t_f. EK TODO: Why minus sign?
        dir_bnd = dir_filter * (-bnd_sgn * t_f)
        t_bnd = neu_bnd + dir_bnd

        # Vector source fc_cc is (nhf, nd*nhf) hf2f @ fc_cc @ [(f2hf3d @ t_f) * 3dc2hf3d
        # @ vec_source_3d]
        #
        # EK: Not sure about mpsa in the next sattement, but some sort of compatability
        # requirement is needed. For compatibility with mpsa, vector source is defined
        # as a cell-wise nd vector. As this discretization is in 3d, expand.
        cells_nd_to_hf_3d = diff_discr.half_face_map(
            domains, from_entity="cells", with_sign=False, dimensions=(3, 3)
        ) @ diff_discr.nd_to_3d(domains, self.nd)
        source_3d = pp.ad.SparseArray(cells_nd_to_hf_3d) @ self.vector_source(
            domains, material="fluid"
        )
        f_to_hf_3d = diff_discr.half_face_map(
            domains, from_entity="faces", with_sign=True, dimensions=(3, 1)
        )
        # hf_to_f is constructed with signs. This compensates for the distances of the
        # two half-faces in d_vec having opposite signs (they both run from cell center
        # to face center).
        #
        # EK: Can we be sure that the full distance vector points in the right
        # direction?
        vector_source = pp.ad.SparseArray(hf_to_f @ d_vec) @ (
            (pp.ad.SparseArray(f_to_hf_3d) @ t_f) * source_3d
        )

        pressure_difference = pp.ad.SparseArray(
            diff_discr.face_pairing_from_cell_array(domains)
        ) @ self.pressure(domains)

        # Get boundary condition values
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=self.pressure,
            neumann_operator=self.darcy_flux,
            bc_type=self.bc_type_darcy_flux,
            name="bc_values_darcy_flux",
        )
        base_discr = self.darcy_flux_discretization(domains)
        if isinstance(base_discr, pp.ad.MpfaAd):

            def f(T_f, p_diff, p):
                val = base_discr.flux.parse(self.mdg) @ p.val
                jac = base_discr.flux.parse(self.mdg) @ p.jac
                if hasattr(T_f, "jac"):
                    # Trick to get the correct shape of the jacobian. See the diagvec_mul methods.
                    jac += sps.diags(p_diff.val) @ T_f.jac
                return pp.ad.AdArray(val, jac)

            def g(vector_source_diff, vector_source_param):
                val = base_discr.vector_source @ vector_source_param
                #   jac = # how to get the correct shape?
                if hasattr(vector_source_diff, "jac"):
                    val += vector_source_diff.jac @ vector_source_param.val
                return pp.ad.AdArray(val, jac)

            flux_p = pp.ad.Function(f, "differentiable_mpfa")(
                t_f, pressure_difference, self.pressure(domains)
            )
            vector_source = pp.ad.Function(g, "differentiable_mpfa_vector_source")(
                vector_source, self.vector_source(domains, material="fluid")
            )
        else:
            flux_p = t_f * pressure_difference
        # self.set_discretization_parameters()
        # self.discretize()
        # flux_p.evaluate(self.equation_system)
        flux: pp.ad.Operator = (
            flux_p  # discr.flux @ self.pressure(domains)
            + t_bnd
            @ (
                boundary_operator
                + intf_projection.mortar_to_primary_int
                @ self.interface_darcy_flux(interfaces)
            )
            + base_discr.vector_source @ self.vector_source(domains, material="fluid")
        )

        # Development debugging. TODO: Remove
        # Compose equation

        # div = pp.ad.Divergence(domains)
        # eq = div @ ((T_f * pressure_difference) + bc_contr)
        # system = eq.evaluate(self.equation_system)
        # dp = sps.linalg.spsolve(system.jac, -system.val)
        return flux

    def _transmissibility_matrix(self, subdomains: list[pp.Grid]):
        # Construct half-face transmissbilities t_hf = 1 / / dist * d_vec @ n @ k_c.
        # Cell-wise diffusivity tensor, shape = (9 * n_cells,)
        k_c = self._permeability(subdomains)
        boundary_grids = self.subdomains_to_boundary_grids(subdomains)
        diff_discr = pp.numerics.fv.tpfa.DifferentiableTpfa(
            subdomains,
            boundary_grids,
            self.mdg,
        )
        n, d_vec, dist = diff_discr.half_face_geometry_matrices(subdomains)
        # Compose half-face transmissibilities
        d_n_by_dist = sps.diags(1 / dist) * d_vec @ n
        one = pp.ad.Scalar(1)
        t_hf_inv = one / (pp.ad.SparseArray(d_n_by_dist) @ k_c)
        # Compose full-face transmissibilities
        # Sum over half-faces to get transmissibility on faces.
        # Include sign to cancel the effect of the d_vec @ n having opposite signs on
        # the two half-faces.
        hf_to_f = diff_discr.half_face_map(
            subdomains, to_entity="faces", with_sign=True
        )
        t_f_full = one / (pp.ad.SparseArray(hf_to_f) @ t_hf_inv)
        return t_f_full, diff_discr, boundary_grids, hf_to_f, d_vec

    def pressure_trace(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)
        base_discr: Union[pp.ad.TpfaAd, pp.ad.MpfaAd] = self.darcy_flux_discretization(
            subdomains
        )
        p: pp.ad.MixedDimensionalVariable = self.pressure(subdomains)

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.pressure,
            neumann_operator=self.darcy_flux,
            bc_type=self.bc_type_darcy_flux,
            name="bc_values_darcy",
        )

        # Construct differentiable bound_pressure_face. Note that much of the following
        # code is copied from the diffusive_flux method. TODO: Refactor?
        t_f_full, *_ = self._transmissibility_matrix(subdomains)

        # Face contribution to boundary pressure is 1 on Dirichlet faces, -1/t_f_full on
        # Neumann faces (see Tpfa.discretize).
        bound_pressure_face = dir_filter - neu_filter * (one / t_f_full)

        if isinstance(base_discr, pp.ad.MpfaAd):

            def f(bound_pressure_face):
                val = base_discr.bound_pressure_face
                jac = bound_pressure_face.jac
                return pp.ad.AdArray(val, jac)

            bound_pressure_face = pp.ad.Function(f, "differentiable_mpfa")(
                bound_pressure_face
            )

        pressure_trace = (
            base_discr.bound_pressure_cell @ p  # independent of k
            + bound_pressure_face  # dependens on k
            * (projection.mortar_to_primary_int @ self.interface_darcy_flux(interfaces))
            + bound_pressure_face * boundary_operator
            + base_discr.bound_pressure_vector_source  # independent of k
            @ self.vector_source(subdomains, material="fluid")
        )
        return pressure_trace

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value equal to the fluid's reference pressure (which will be 0
        by default).

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(boundary_grid)
        vals_loc = np.zeros(boundary_grid.num_cells)
        vals_loc[domain_sides.north] = 200
        vals_loc[domain_sides.south] = 100
        return vals_loc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.MpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        # TODO: The ad.Discretizations may be purged altogether. Their current function
        # is very similar to the ad.Geometry in that both basically wrap numpy/scipy
        # arrays in ad arrays and collect them in a block matrix. This similarity could
        # possibly be exploited. Revisit at some point.
        return pp.ad.MpfaAd(self.darcy_keyword, subdomains)


m = AdTpfaFlow({})
m.prepare_simulation()
g = m.mdg.subdomains()[0]
g.nodes[:2, 0] += 0.1
g.compute_geometry()
o = m.diffusive_flux(m.mdg.subdomains(), m._permeability)
t = o.evaluate(m.equation_system)

# %%
