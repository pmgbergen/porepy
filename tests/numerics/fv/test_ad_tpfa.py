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

    def _block_diagonal_grid_property_matrix(
        self,
        domains: list[pp.Grid],
        grid_property_getter: Callable[[pp.Grid], Any],
        name: Optional[str] = None,
    ) -> sps.spmatrix:
        """Construct mapping matrix for the connectivity between two grids entities.

        The mapping matrix is a block diagonal matrix where each block contains 1 where
        the two entities are connected, and 0 otherwise.

        Parameters:
            domains: List of grids.
            grid_property_getter: Function that returns the property of the grid that
                should be used for the mapping.
            name: Name of the operator.

        Returns:
            Mapping matrix.
        """
        blocks = []
        for g in domains:
            if g.dim == 0:
                # 0d subdomains have no faces, so the projection might not exist.
                # TODO: Implement this with proper handling of special cases (first and
                # second dimension being faces and cells, respectively).
                raise NotImplementedError
            else:
                mat_loc = grid_property_getter(g)
            blocks.append(mat_loc)

        block_matrix = pp.matrix_operations.optimized_compressed_storage(
            sps.block_diag(blocks)
        )
        return block_matrix

    def cells_to_half_faces(
        self, subdomains: list[pp.Grid], dim: int, with_sign=False
    ) -> sps.spmatrix:
        """Mapping from cells to half-faces.

        Parameters:
            subdomains: List of grids.
            dim: Dimension of the half-faces.

        Returns:
            Operator.
        """

        def get_matrix(g: pp.Grid) -> sps.csr_matrix:
            _, ci, sgn = sps.find(g.cell_faces)
            # Repeat dim times in f order
            row_inds = np.repeat(np.arange(ci.size), dim)
            col_inds = pp.fvutils.expand_indices_nd(ci, dim)
            if with_sign:
                vals = np.repeat(sgn, dim)
            else:
                vals = np.ones(col_inds.size)
            mat = sps.csr_matrix(
                (vals, (row_inds, col_inds)),
                shape=(ci.size, g.num_cells * dim),
            )
            return mat

        return self._block_diagonal_grid_property_matrix(
            subdomains,
            get_matrix,
        )

    def faces_to_half_faces(
        self, subdomains: list[pp.Grid], dim: int, with_sign=False
    ) -> sps.spmatrix:
        """Mapping from faces to half-faces.

        Parameters:
            subdomains: List of grids.
            dim: Dimension of the half-faces.

        Returns:
            Operator.
        """

        def get_matrix(g: pp.Grid) -> sps.csr_matrix:
            fi, _, sgn = sps.find(g.cell_faces)

            # Repeat dim times in f order
            row_inds = np.repeat(np.arange(fi.size), dim)
            col_inds = pp.fvutils.expand_indices_nd(fi, dim)
            if with_sign:
                vals = np.repeat(sgn, dim)
            else:
                vals = np.ones(col_inds.size)
            mat = sps.csr_matrix(
                (vals, (row_inds, col_inds)),
                shape=(fi.size, g.num_faces * dim),
            )
            return mat

        return self._block_diagonal_grid_property_matrix(
            subdomains,
            get_matrix,
        )

    def cell_face_vectors(self, subdomains: list[pp.Grid]) -> sps.spmatrix:
        """Distance between face centers and cell centers.

        Parameters:
            subdomains: List of grids.

        Returns:
            Operator.
        """

        vec_dim = 3

        def get_c_f_vec_matrix(g: pp.Grid) -> sps.csr_matrix:
            """Construct matrix of vectors connecting cell centers and face centers."""

            fi, ci, sgn = sps.find(g.cell_faces)
            # Construct vectors from cell centers to face centers.
            fc_cc = g.face_centers[:, fi] - g.cell_centers[:, ci]
            # Repeat dim times in f order
            num_hf = fi.size
            # Each row contains vec_dim entries and corresponds to one half-face.
            row_inds = np.repeat(np.arange(num_hf), vec_dim)
            # There are num_hf * vec_dim columns, each vec_dim-long block corresponding
            # to one half-face.
            col_inds = pp.fvutils.expand_indices_nd(np.arange(num_hf), vec_dim)
            vals = fc_cc.ravel("F")
            mat = sps.csr_matrix(
                (vals, (row_inds, col_inds)),
                shape=(num_hf, num_hf * vec_dim),
            )
            return mat

        dist_vec = self._block_diagonal_grid_property_matrix(
            subdomains,
            get_c_f_vec_matrix,
        )

        return dist_vec

    def normal_vectors(self, subdomains: list[pp.Grid]) -> sps.spmatrix:
        """Normal vectors on half-faces, repeated for each dimension.

        Parameters:
            subdomains: List of grids.

        Returns:
            Operator.
        """
        vector_dim = 3

        def get_matrix(g: pp.Grid) -> sps.csr_matrix:
            """Construct normal vector matrix. Each vector is repeated vector_dim times.

            Half-face i corresponds to rows
                vector_dim * i:vector_dim(i+1)
            and contains n_0^i, n_1^i, n_2^i. The column indices makes sure we hit the
            right permeability entries. The permeability being a tensor_dim * num_cells
            vector, we expand the cell indices to tensor_dim indices.

            Parameters:
                g: Grid.

            Returns:
                spmatrix ``(num_half_faces * vector_dim, num_cells * tensor_dim)``:
                    Normal vector matrix.

            """
            tensor_dim = vector_dim**2
            fi, ci, sgn = sps.find(g.cell_faces)
            num_hf = fi.size
            n = g.face_normals
            row_inds = np.repeat(np.arange(num_hf * vector_dim), vector_dim)
            col_inds = pp.fvutils.expand_indices_nd(ci, tensor_dim)
            repeat_fi = np.repeat(fi, vector_dim)
            vals = n[:, repeat_fi].ravel("F")
            mat = sps.csr_matrix(
                (vals, (row_inds, col_inds)),
                shape=(num_hf * vector_dim, g.num_cells * vector_dim**2),
            )
            return mat

        return self._block_diagonal_grid_property_matrix(
            subdomains,
            get_matrix,
        )

    def cell_face_distances(self, subdomains: list[pp.Grid]) -> sps.spmatrix:
        """Scalar distance between face centers and cell centers for each half face."""
        vals = []
        for g in subdomains:
            fi, ci, sgn = sps.find(g.cell_faces)
            fc_cc = g.face_centers[:, fi] - g.cell_centers[:, ci]
            vals.append(np.power(fc_cc, 2).sum(axis=0))
        return np.hstack(vals)

    def tpfa_ad(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """TPFA discretization.

        Parameters:
            subdomains: List of grids.

        Returns:
            Operator.
        """
        # Half-face transmissibilities are computed as
        # t_hf = d_vec @ n @ k_hf / dist
        # k_hf: Permeability on half-faces shape=(9 x num_half_faces,)
        # n: Normal vector on half-faces shape=(3 x num_half_faces,9 x num_half_faces)
        # d_vec: Vectors cell centers and face centers shape=(num_half_faces, 3 x num_half_faces)
        # dist: Distance between cell centers and face centers shape=(num_half_faces,)
        d_vec = self.cell_face_vectors(subdomains)
        dist = self.cell_face_distances(subdomains)
        k_c = self._permeability(subdomains)
        n = self.normal_vectors(subdomains)

        d_n_by_dist = sps.diags(1 / dist) * d_vec @ n
        one = pp.ad.Scalar(1)
        # t_hf = d_vec @ n @ k_hf / dist
        t_hf_inv = one / (pp.ad.SparseArray(d_n_by_dist) @ k_c)
        # Sum over half-faces to get transmissibility on faces.
        # Include sign to cancel the effect of the d_vec @ n having opposite signs on
        # the two half-faces.
        hf_to_f_mat = self.faces_to_half_faces(subdomains, dim=1, with_sign=True).T
        T_f = one / (pp.ad.SparseArray(hf_to_f_mat) @ t_hf_inv)

        # Construct difference operator to get p_l - p_r on faces. First map p to half-
        # faces, then to faces with the signed matrix.
        c_to_hf_mat = self.cells_to_half_faces(subdomains, dim=1, with_sign=False)
        diff_p = pp.ad.SparseArray(hf_to_f_mat @ c_to_hf_mat) @ self.pressure(
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

        # Construct sign vector on boundary from the signed hf_to_f_mat. The following
        # pairing of signs from half faces to faces works because there's only one half
        # face at each boundary face. Internally, the sign contributions cancel.
        one_vec = np.ones(hf_to_f_mat.shape[1])
        bnd_sgn = pp.ad.DenseArray(hf_to_f_mat @ one_vec)
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
