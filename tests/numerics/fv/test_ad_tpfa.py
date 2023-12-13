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


class AdTpfaFlux:
    """Differentiable discretization of a diffusive flux.

    The diffusive flux is given by

        q = - K grad p

    where K is the diffusivity tensor and p is the primary variable/potential. In the
    case of Darcy's law, the diffusivity tensor is the permeability tensor and the
    primary variable is the pressure. The implementation is agnostic to this, and can be
    used for other constitutive laws as well (e.g. Fourier's law).

    To use for a specific constitutive law, the following methods must be used when
    overriding specific methods:
    - diffusive_flux: Discretization of the diffusive flux. This method should be called
        by the overriding method (darcy_flux, fourier_flux etc).
    - potential_trace: Discretization of the potential on the subdomain boundaries. This
        method should be called by the overriding method (pressure_trace,
        temperature_trace etc).
    - todo: vector_source.

    Note: This class implicitly assumes conventions on naming of methods and BC values
    keys. Specifically, the BC values keys are assumed to be of the form
    "bc_values_" + flux_name, where flux_name is the name of the flux (e.g. "darcy_flux"
    or "fourier_flux"). The same goes for "inteface_" + flux_name and flux_name +
    "_discretization". These conventions are used to simplify the implementation of
    these methods. TODO: Consider making this more explicit. Also, marvel at the fact
    that this comment was largely written by a computer (including half of the
    last sentence).
    """

    def _permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """K is a second order tensor having nd^2 entries per cell.

        3d:
        Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
        0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
        2d:
        Kxx, Kxy, Kyx, Kyy
        0  , 1  , 2  , 3
        """

        nc = sum([sd.num_cells for sd in subdomains])
        tensor_dim = 3**2
        vals = self.solid.permeability() * np.ones(nc * tensor_dim)
        return pp.wrap_as_dense_ad_array(vals, name="Flattened permeability")

    def diffusive_flux(
        self,
        domains: pp.SubdomainsOrBoundaries,
        potential: Callable[[list[pp.Grid]], pp.ad.Operator],
        diffusivity_tensor: Callable[[list[pp.Grid]], pp.ad.Operator],
        flux_name: str,
    ) -> pp.ad.Operator:
        """Discretization of a diffusive constitutive law.

        Parameters:
            domains: List of domains where the flux is defined.
            diffusivity_tensor: Function returning the diffusivity tensor as an Ad
                operator. For Darcy's and Fourier's law, this is the permeability and
                thermal conductivity, respectively.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Face-wise integrated flux.

        """

        if len(domains) == 0 or all([isinstance(g, pp.BoundaryGrid) for g in domains]):
            # Note: in case of the empty subdomain list, the time dependent array is
            # still returned. Otherwise, this method produces an infinite recursion
            # loop. It does not affect real computations anyhow.
            return self.create_boundary_operator(  # type: ignore[call-arg]
                name=flux_name,
                domains=domains,
            )
        boundary_grids = self.subdomains_to_boundary_grids(domains)
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(domains, [1])
        intf_projection = pp.ad.MortarProjections(self.mdg, domains, interfaces, dim=1)

        # Compute the transmissibility matrix, see the called function for details. Also
        # obtain various helper objects.
        (
            t_f,
            diff_discr,
            hf_to_f,
            d_vec,
        ) = self._transmissibility_matrix(domains, diffusivity_tensor)

        # Treatment of boundary conditions.
        one = pp.ad.Scalar(1)
        dir_filter, neu_filter = diff_discr.boundary_filters(
            self.mdg, boundary_grids, "bc_values_" + flux_name
        )
        # Delete neu values in T_f, i.e. keep all non-neu values.
        t_f = (one - neu_filter) * t_f

        # Sign of boundary faces.
        bnd_sgn = diff_discr.boundary_sign(domains)
        # Discretization of boundary conditions: On Neumann faces, we will simply add
        # the flux, with a sign change if the normal vector is pointing inwards.
        neu_bnd = neu_filter * bnd_sgn
        # On Dirichlet faces, the assigned Dirichlet value corresponds to a flux of
        # magnitude t_f. EK TODO: Why minus sign?
        dir_bnd = dir_filter * (-bnd_sgn * t_f)
        t_bnd = neu_bnd + dir_bnd

        # Discretization of vector source: TODO: Consider moving to a separate method.
        #
        # The flux through a face with normal vector n_j, as seen from cell i, driven by
        # a vector source v_i in cell i, is given by
        #
        #   q_j = n_j^T K_i v_i
        #
        # A Tpfa-style discretization of this term will apply harmonic averaging of the
        # permeabilities (see function _transmissibility_matrix), and multiply with the
        # difference in vector source between the two cells. We have already computed
        # the transmissibility matrix, which computes the product of the permeability
        # tensor, the normal vector and a unit vector from cell to face center. To
        # convert this to a discretizaiton for the vector source, we first need to
        # project the vector source onto the unit vector from cell to face center.
        # Second, the vector source should be scaled by the distance from cell to face
        # center. This can be seen as compensating for the distance in the denominator
        # of the half-face transmissibility, or as converting the vector source into a
        # potential-like quantity before applying the flux calculation.

        # The vector source can be 2d or 3d, but the geometry, thus discretization, is
        # always 3d, thus we need to map from nd to 3d.
        cells_nd_to_3d = diff_discr.nd_to_3d(domains, self.nd)
        # Mapping from cells to half-faces of 3d quantities.
        cells_to_hf_3d = diff_discr.half_face_map(
            domains, from_entity="cells", with_sign=False, dimensions=(3, 3)
        )

        # Build a mapping for the cell-wise vector source, unravelled from the right:
        # First, map the vector source from nd to 3d. Second, map from cells to
        # half-faces. Third, project the vector source onto the vector from cell center
        # to half-face center (this is the vector which Tpfa uses as a proxy for the
        # full gradient, see comments in the method _transmissibility_matrix). As the
        # rows of d_vec have length equal to the distance, this compensates for the
        # distance in the denominator of the half-face transmissibility. Fourth, map
        # from half-faces to faces, using a mapping with signs, thereby taking the
        # difference between the two vector sources.
        vector_source_c_to_f = pp.ad.SparseArray(
            hf_to_f @ d_vec @ cells_to_hf_3d @ cells_nd_to_3d
        )

        # Fetch the constitutive law for the vector source.
        vector_source_cells = self.vector_source(domains, material="fluid")  # TODO

        # Compute the difference in pressure and vector source between the two cells on
        # the sides of each face.
        potential_difference = pp.ad.SparseArray(
            diff_discr.face_pairing_from_cell_array(domains)
        ) @ potential(domains)
        vector_source_difference = vector_source_c_to_f @ vector_source_cells

        # Fetch the discretization of the Darcy flux
        base_discr = getattr(self, flux_name + "_discretization")(domains)

        # Compose the discretization of the Darcy flux q = T(k(u)) * p, (where the k(u)
        # dependency can be replaced by other primary variables. The chain rule gives
        #
        #  dT = p * (dT/du) * du + T dp
        #
        # A similar expression holds for the vector source term. If the base
        # discretization (which calculates T in the above equation) is Tpfa, the full
        # expression will be obtained by the Ad machinery and there is no need for
        # special treatment. If the base discretization is Mpfa, we need to mix this
        # T-matrix with the the Tpfa-style approximation of dT/du, as is done in the
        # below if-statement.
        if isinstance(base_discr, pp.ad.MpfaAd):
            # To obtain a mixture of Tpfa and Mpfa, we utilize pp.ad.Function, one for
            # the flux and one for the vector source. Keep in mind that these functions
            # will be evaluated in forward mode, so that the inputs are not
            # Ad-operators, but numerical values.

            def flux_discretization(T_f, p_diff, p):
                # Take the differential of the product between the transmissibility
                # matrix and the pressure difference.

                # We know that base_discr.flux is a sparse matrix, so we can call parse
                # directly. At the time of evaluation, p will be an AdArray, thus we can
                # access its val and jac attributes.
                val = base_discr.flux.parse(self.mdg) @ p.val
                jac = base_discr.flux.parse(self.mdg) @ p.jac

                if hasattr(T_f, "jac"):
                    # Add the contribution to the Jacobian matrix from the derivative of
                    # the transmissibility matrix times the pressure difference. To see
                    # why this is correct, it may be useful to consider the flux over a
                    # single face (corresponding to one row in the Jacobian matrix).
                    jac += sps.diags(p_diff.val) @ T_f.jac

                return pp.ad.AdArray(val, jac)

            def vector_source_discretization(T_f, vs_diff, vs):
                # Take the differential of the flux associated with the vector source
                # term.

                # The vector source (vs) is an operator which, at the time of
                # evaluation, will be either a numpy or an AdArray (ex: a gravity term
                # with a constant and variable density, respectively). Thus an if-else
                # is needed to get hold of its value and Jacobian.
                if isinstance(vs, pp.ad.AdArray):
                    vs_val = vs.val
                    jac = vs.jac
                elif isinstance(vs, np.ndarray):
                    # The value is a numpy array, thus the Jacobian should be a zero
                    # matrix of the right size.
                    vs_val = vs
                    num_rows = vs_val.size
                    num_cols = self.equation_system.num_dofs()
                    jac = sps.csr_matrix((num_rows, num_cols))
                else:
                    # EK is not really sure about this (can it be a scalar?), but
                    # raising an error should uncover any such cases.
                    raise ValueError(
                        "vector_source_param must be an AdArray or numpy array"
                    )

                # The value of the vector source discretization is a simple product.
                val = base_discr.vector_source.parse(self.mdg) @ vs_val
                # The contribution from differentiating the vector source term to the
                # Jacobian of the flux.
                jac = base_discr.vector_source.parse(self.mdg) @ jac

                if hasattr(T_f, "jac"):
                    # At the time of evaluation, the difference in the vector source is
                    # either an AdArray or a numpy array. We anyhow need to get hold of
                    # its value.
                    if isinstance(vs_diff, pp.ad.AdArray):
                        vs_diff_val = vs_diff.val
                    elif isinstance(vs_diff, np.ndarray):
                        vs_diff_val = vs_diff

                    # Add the contribution to the Jacobian matrix from the derivative of
                    # the transmissibility matrix times the vector source difference.
                    jac += sps.diags(vs_diff_val) @ T_f.jac

                return pp.ad.AdArray(val, jac)

            flux_p = pp.ad.Function(flux_discretization, "differentiable_mpfa")(
                t_f, potential_difference, potential(domains)
            )
            vector_source_d = pp.ad.Function(
                vector_source_discretization, "differentiable_mpfa_vector_source"
            )(t_f, vector_source_difference, vector_source_cells)

        else:
            # The base discretization is Tpfa, so we can rely on the Ad machinery to
            # compose the full expression.
            flux_p = t_f * potential_difference
            vector_source_d = t_f * vector_source_difference

        # Get boundary condition values
        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=domains,
            dirichlet_operator=potential,
            neumann_operator=getattr(self, flux_name),
            bc_type=getattr(self, "bc_type_" + flux_name),
            name="bc_values_" + flux_name,
        )

        # Compose the full discretization of the Darcy flux, which consists of three
        # terms: The flux due to pressure differences, the flux due to boundary
        # conditions, and the flux due to the vector source.
        flux: pp.ad.Operator = (
            flux_p
            + t_bnd
            * (
                boundary_operator
                + intf_projection.mortar_to_primary_int
                @ getattr(self, "interface_" + flux_name)(interfaces)
            )
            + vector_source_d
        )
        flux.set_name("Differentiable diffusive flux")
        return flux

    def _transmissibility_matrix(
        self,
        subdomains: list[pp.Grid],
        diffusivity_tensor: Callable[[list[pp.Grid]], pp.ad.Operator],
    ) -> tuple[
        pp.ad.Operator,
        pp.numerics.fv.tpfa.DifferentiableTpfa,
        sps.spmatrix,
        sps.spmatrix,
    ]:
        """Compute the Tpfa transmissibility matrix for a list of subdomains."""
        # In Tpfa, the Darcy flux through a face with normal vector n_j, as seen from
        # cell i, is given by (subscripts indicate face or cell index)
        #
        #    q_j = n_j^T K_i e_ij (p_i - p_j) / dist_ij
        #
        # Here, K_i is the permeability tensor in cell i, e_ij is the unit vector from
        # cell i to face j, and dist_ij is the distance between the cell center and the
        # face center. Comparing with the continuous formulation, we see that the
        # pressure gradient is approximated by the pressure difference, divided by
        # distance, in the direction between cell and face centers. Writing out the
        # expression for the half-face transmissibility
        #
        #    t = n_r^T K_rs e_s / dist
        #
        # Here, subscripts indicate (Cartesian) dimensions, the summation convention is
        # applied, and dist again represent the distance from cell to face center. (EK:
        # the change of meaning of subscript is unfortunate, but it is very important to
        # understand the how the components of the permeability tensor and the normal
        # and distance vectors are multiplied.) This formulation can be reformulated to
        #
        #   t = n_r^T e_s K_rs / dist
        #
        # where the point is that, by right multiplying the permeability tensor, this
        # can be represented as an Ad operator (which upon parsing will be an AdArray
        # which only can be right multiplied). The below code implements this
        # formulation. The full transmissibility matrix is obtained by taking the
        # harmonic mean of the two half-face transmissibilities on each face.

        # The cell-wise permeability tensor is represented as an Ad operator which
        # evaluates to an AdArray with 9 * n_cells entries.
        k_c = diffusivity_tensor(subdomains)

        # Create the helper discretization object, which will be used to generate
        # grid-related quantities and mappings.
        diff_discr = pp.numerics.fv.tpfa.DifferentiableTpfa()

        # Get the normal vector, vector from cell center to face center (d_vec), and
        # distance from cell center to face center (dist) for each half-face.
        n, d_vec, dist = diff_discr.half_face_geometry_matrices(subdomains)

        # Compose the geometric part of the half-face transmissibilities. Note that
        # dividing d_vec by dist essentially form a unit vector from cell to face
        # center.
        d_n_by_dist = sps.diags(1 / dist) * d_vec @ n

        # Form the full half-face transmissibilities, and take its reciprocal, preparing
        # for a harmonic mean between the two half-face transmissibilities on ecah side
        # of a face.
        one = pp.ad.Scalar(1)
        t_hf_inv = one / (pp.ad.SparseArray(d_n_by_dist) @ k_c)

        # Compose full-face transmissibilities
        # Sum over half-faces to get transmissibility on faces.
        # Include sign to cancel the effect of the d_vec @ n having opposite signs on
        # the two half-faces.
        hf_to_f = diff_discr.half_face_map(
            subdomains, to_entity="faces", with_sign=True
        )
        # Take the harmonic mean of the two half-face transmissibilities.
        t_f_full = one / (pp.ad.SparseArray(hf_to_f) @ t_hf_inv)
        t_f_full.set_name("transmissibility matrix")
        return t_f_full, diff_discr, hf_to_f, d_vec

    def potential_trace(
        self,
        subdomains: list[pp.Grid],
        potential: Callable[[list[pp.Grid]], pp.ad.Operator],
        diffusivity_tensor: Callable[[list[pp.Grid]], pp.ad.Operator],
        flux_name: str,
    ) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        interfaces: list[pp.MortarGrid] = self.subdomains_to_interfaces(subdomains, [1])
        boundary_grids = self.subdomains_to_boundary_grids(subdomains)

        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        boundary_operator = self._combine_boundary_operators(  # type: ignore[call-arg]
            subdomains=subdomains,
            dirichlet_operator=potential,
            neumann_operator=getattr(self, flux_name),
            bc_type=getattr(self, "bc_type_" + flux_name),
            name="bc_values_" + flux_name,
        )
        base_discr = getattr(self, flux_name + "_discretization")(subdomains)
        # Obtain the transmissibilities in operator form. Ignore other outputs.
        t_f_full, *_ = self._transmissibility_matrix(subdomains, diffusivity_tensor)
        one = pp.ad.Scalar(1)
        # BC filters for Dirichlet and Neumann faces.

        diff_discr = pp.numerics.fv.tpfa.DifferentiableTpfa(
            subdomains,
            boundary_grids,
            self.mdg,
        )
        dir_filter, neu_filter = diff_discr.boundary_filters(
            self.mdg, boundary_grids, "bc_values_" + flux_name
        )

        # Face contribution to boundary potential is 1 on Dirichlet faces, -1/t_f_full
        # on Neumann faces (see Tpfa.discretize). Named "bound_pressure_face" and not
        # "bound_potential_face" to be consistent with the base discretization.
        bound_pressure_face = dir_filter - neu_filter * (one / t_f_full)

        if isinstance(base_discr, pp.ad.MpfaAd):

            def b_p_f(bound_pressure_face):
                val = base_discr.bound_pressure_face
                jac = bound_pressure_face.jac
                return pp.ad.AdArray(val, jac)

            bound_pressure_face = pp.ad.Function(b_p_f, "differentiable_mpfa")(
                bound_pressure_face
            )

        pressure_trace = (
            base_discr.bound_pressure_cell @ potential(subdomains)  # independent of k
            + bound_pressure_face  # dependens on k
            * (
                projection.mortar_to_primary_int
                @ getattr(self, "interface_" + flux_name)(interfaces)
            )
            + bound_pressure_face * boundary_operator
            + base_discr.bound_pressure_vector_source  # independent of k
            @ self.vector_source(subdomains, material="fluid")
        )
        return pressure_trace


class AdDarcyFlux(AdTpfaFlux):
    """Adaptive discretization of the Darcy flux from generic adaptive flux class."""

    def darcy_flux(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Discretization of Darcy's law.


        Parameters:
            domains: List of domains where the Darcy flux is defined.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Face-wise Darcy flux in cubic meters per second.

        """
        flux = self.diffusive_flux(
            domains, self.pressure, self._permeability, "darcy_flux"
        )
        flux.set_name("Differentiable Darcy flux")
        return flux

    def pressure_trace(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Pressure on the subdomain boundaries.

        Parameters:
            subdomains: List of subdomains where the pressure is defined.

        Returns:
            Pressure on the subdomain boundaries. Parsing the operator will return a
            face-wise array.

        """
        pressure_trace = self.potential_trace(
            domains, self.pressure, self._permeability, "darcy_flux"
        )
        pressure_trace.set_name("Differentiable pressure trace")
        return pressure_trace


class TestAdTpfaFlow(AdDarcyFlux, pp.fluid_mass_balance.SinglePhaseFlow):
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
        """Non-constant permeability tensor. Depends on pressure."""
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


m = TestAdTpfaFlow({})
m.prepare_simulation()
g = m.mdg.subdomains()[0]
g.nodes[:2, 0] += 0.1
g.compute_geometry()
m.set_discretization_parameters()
m.discretize()


o = m.diffusive_flux(m.mdg.subdomains(), m._permeability)
t = o.evaluate(m.equation_system)

# %%
