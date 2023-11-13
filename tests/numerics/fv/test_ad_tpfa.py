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
        p = self.pressure(subdomains)
        # return pp.wrap_as_dense_ad_array(scaled_vals, name="permeability") + e_xy @ p
        return pp.wrap_as_dense_ad_array(np.ones(nc * tensor_dim), name="permeability")

    def tpfa_ad(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """TPFA discretization.

        Parameters:
            subdomains: List of grids.

        Returns:
            Operator.
        """
        discr = pp.numerics.fv.tpfa.DifferentiableTpfa(self.mdg, subdomains=subdomains)

        T_f = discr.face_transmissibilities(subdomains)
        diff_p = discr.face_pairing_from_cell_vector(subdomains) @ self.pressure(
            subdomains
        )

        # Boundaries. From tpfa.py we have that:
        # - Neu faces should be
        #       zero in T_f
        #       equal sgn in T_bf. Sgn is cancelled by divergence in
        #            div @ (T_bf * bc_vals),
        #       thus ensuring outfluxes are always positive.
        # - Dir should be
        #       unchanged in T_f
        #       copied to T_bf with weight -sgn.

        # Preparation. Get boundary grids and filters.
        boundary_grids = self.subdomains_to_boundary_grids(subdomains)

        name = "bc_values_darcy_flux"
        dir_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_dir"), domains=boundary_grids
        )
        neu_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_neu"), domains=boundary_grids
        )
        proj = pp.ad.BoundaryProjection(self.mdg, subdomains=subdomains, dim=1)

        bnd_sgn = discr.boundary_sign(subdomains)
        # Neu values in T_bf = bnd_sgn.
        neu_bnd = proj.boundary_to_subdomain @ neu_filter * bnd_sgn
        # Dir values in T_bf = -bnd_sgn * T_f
        dir_bnd = (proj.boundary_to_subdomain @ dir_filter) * (-bnd_sgn * T_f)

        # Delete neu values in T_f, i.e. keep all non-neu values.
        T_f = (pp.ad.Scalar(1) - proj.boundary_to_subdomain @ neu_filter) * T_f

        # Get boundary condition values
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=self.pressure,
            neumann_operator=self.darcy_flux,
            bc_type=self.bc_type_darcy_flux,
            name=name,
        )
        bc_contr = (dir_bnd + neu_bnd) * boundary_operator
        # Compose equation
        div = pp.ad.Divergence(subdomains)
        eq = div @ ((T_f * diff_p) + bc_contr)
        system = eq.evaluate(self.equation_system)
        dp = sps.linalg.spsolve(system.jac, -system.val)
        return d_vec.T @ k_hf @ d_vec / dist

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


m = AdTpfaFlow({})
m.prepare_simulation()
o = m.tpfa_ad(m.mdg.subdomains())
t = o.evaluate(m.equation_system)

# %%
