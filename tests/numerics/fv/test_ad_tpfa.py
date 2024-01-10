import pytest

import numpy as np
import scipy.sparse as sps

import porepy as pp

from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)


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
        self._neumann_flux = 1529
        self._nonzero_dirichlet_face = 5
        self._dirichlet_pressure = 1683

    def initial_condition(self):
        super().initial_condition()
        for _, data in self.mdg.subdomains(return_data=True):
            pp.set_solution_values(
                name=self.pressure_variable,
                values=np.array([2, 3], dtype=float),
                data=data,
                iterate_index=0,
                time_step_index=0,
            )

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

    def _cell_projection(self, cell_id: int) -> sps.csr_matrix:
        if cell_id == 0:
            return pp.ad.SparseArray(sps.csr_matrix(np.array([[1, 0], [0, 0]])))
        elif cell_id == 1:
            return pp.ad.SparseArray(sps.csr_matrix(np.array([[0, 0], [0, 1]])))
        else:
            raise ValueError(f"Cell id {cell_id} is not valid.")

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Non-constant permeability tensor. Depends on pressure.

        NOTE: *Do not* change this code without also updating the permeability in the
        test function.
        """
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

        e_xx = self.e_i(subdomains, i=0, dim=tensor_dim)
        e_xy = self.e_i(subdomains, i=1, dim=tensor_dim)
        e_yx = self.e_i(subdomains, i=3, dim=tensor_dim)
        e_yy = self.e_i(subdomains, i=4, dim=tensor_dim)
        p = self.pressure(subdomains)

        cell_0_projection = self._cell_projection(0)
        cell_1_projection = self._cell_projection(1)

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

    def vector_source_darcy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Vector source term for the Darcy flux.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Vector source term for the Darcy flux.

        """
        arr = self.params["vector_source"]

        v = pp.wrap_as_dense_ad_array(arr, name="Vector_source")
        return v

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

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for the fluid mass flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value equal to the fluid's reference pressure (which will be 0
        by default).

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        vals_loc = np.zeros(boundary_grid.num_cells)

        neumann_face_boundary = (
            boundary_grid.projection()[:, self._neumann_face].tocsc().indices[0]
        )
        vals_loc[neumann_face_boundary] = self._neumann_flux
        return vals_loc

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
        vals_loc = np.zeros(boundary_grid.num_cells)

        dirichlet_face_boundary = (
            boundary_grid.projection()[:, self._nonzero_dirichlet_face]
            .tocsc()
            .indices[0]
        )

        vals_loc[dirichlet_face_boundary] = self._dirichlet_pressure
        return vals_loc


@pytest.mark.parametrize("vector_source", [True, False])
@pytest.mark.parametrize("base_discr", ["tpfa", "mpfa"])
def test_transmissibility_calculation(vector_source: bool, base_discr: str):
    """Unit test for the calculation of differentiable tpfa transmissibilities.

    The function tests both the calculation of the transmissibility (and its derivative)
    and the calculation of the potential trace on a face.

    Description of relevant aspects tested for individual faces:
    0: Dirichlet face, diagonal permeability, normal vector aligned with x-axis
    1: Internal face.
    2: Dirichlet face, full-tensor permeability, normal vector aligned with x-axis
    3: Dirichlet face, diagonal permeability, normal vector aligned with y-axis
    4: Neumann face, non-zero BC value. Full-tensor permeability, normal vector aligned
         with y-axis.
    5: Dirichlet face, non-zero BC value. Diagonal permeability, normal vector with x
       and y components.
    6: Dirichlet face. Full-tensor permeability, normal vector with x and y components.

    Parameters:
        vector_source: If True, a vector source term will be included in the
            discretization. Else, the vector source term will be zero.
        base_discr: The base discretization to use. Either 'tpfa' or 'mpfa'.

    """

    if vector_source:
        vector_source_array = np.array([1, 2, 3, 5])
        vector_source_diff = vector_source_array[[2, 3]] - vector_source_array[[0, 1]]
    else:
        vector_source_array = np.zeros(4)
        vector_source_diff = np.zeros(2)

    model_params = {
        "base_discr": base_discr,
        "vector_source": vector_source_array,
    }

    model = UnitTestAdTpfaFlux(model_params)

    model.prepare_simulation()

    face_indices = np.array([0, 2, 3, 5, 6])
    cell_indices = np.array([0, 1, 0, 0, 1])

    g = model.mdg.subdomains()[0]

    pressure = model.pressure(model.mdg.subdomains()).value(model.equation_system)

    # The permeability and its derivative. *DO NOT* change this code without also
    # updating the permeability in the model class UnitTestAdTpfaFlux.
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

    # Base discretization matrices for the flux and the vector source.
    data = model.mdg.subdomain_data(model.mdg.subdomains()[0])
    base_flux = data[pp.DISCRETIZATION_MATRICES][model.darcy_keyword]["flux"]
    base_vector_source = data[pp.DISCRETIZATION_MATRICES][model.darcy_keyword][
        "vector_source"
    ]

    def _compute_half_transmissibility_and_derivative(
        fi: int, ci: int
    ) -> tuple[float, float]:
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

    def _project_vector_source(fi: int, ci: int) -> np.ndarray:
        # Helper function to project the vector source term onto the vector from cell
        # center to face center.
        if ci == 0:
            vs_cell = vector_source_array[[0, 1]]
        elif ci == 1:
            vs_cell = vector_source_array[[2, 3]]
        fc_cc = g.face_centers[:2, fi] - g.cell_centers[:2, ci]

        projected_vs = np.dot(fc_cc, vs_cell)

        return projected_vs

    # Test flux calculation on boundary faces
    for fi, ci in zip(face_indices, cell_indices):
        p = pressure[ci]
        # Get half transmissibility and its derivative. There is no need to account for
        # sign changes (to reflect the direction of the normal vector of the face), as
        # this test only considers the numerical value of the transmissibility. The
        # effect of the sign change is tested elsewhere.
        trm, trm_diff = _compute_half_transmissibility_and_derivative(fi, ci)

        # Project the vector source term onto the vector from cell center to face
        # center.
        projected_vs = _project_vector_source(fi, ci)

        if base_discr == "tpfa":
            # If the base discretization is TPFA, we can directly compare the computed
            # flux with the calculated transmissibility times the pressure. This cannot
            # be done for MPFA, since 'trm' is a representation of the tpfa
            # transmissibility, not the MPFA one.
            flux = trm * p
            if fi == model._nonzero_dirichlet_face:
                # If the face is assigned a non-zero Dirichlet value, we need to include
                # the contribution from the Dirichlet value.
                flux -= trm * model._dirichlet_pressure

            # If a vector source is present, the flux will be modified by the vector
            # source term.
            flux_without_vs = flux[0]
            flux += projected_vs * trm

            # Sanity check: The computed vector source flux, using the logic of the
            # implementation of differentiable TPFA should equal the flux computed
            # using the standard TPFA discretization.
            vector_source_flux = base_vector_source[fi] * vector_source_array
            assert np.isclose(projected_vs * trm, vector_source_flux)

            # Sanity check on the direction of the vector source term: A vector source
            # pointing in the same direction as the vector from cell center to face
            # center should give an increased flux in that direction. This will manifest
            # as an increase in the flux if the normal vector points out of the cell,
            # and a decrease otherwise.
            sgn_vs = np.sign(projected_vs * div[ci, fi])
            if sgn_vs > 0:
                assert flux > flux_without_vs
            elif sgn_vs < 0:
                assert flux < flux_without_vs
            assert np.isclose(flux, computed_flux.val[fi])

        # Fetch the transmissibility from the base discretization.
        diff = base_flux[fi].A.ravel()
        # Add tpfa-style contribution from the derivative of the transmissibility.
        diff[ci] += trm_diff * p

        if fi == model._nonzero_dirichlet_face:
            # If the face is assigned a non-zero Dirichlet value, the derivative of the
            # transmissibility will pick up a term that scales with the Dirichlet
            # value. The minus sign corresponds to the minus sign in the discretization.
            diff[ci] -= trm_diff * model._dirichlet_pressure

        diff[ci] += projected_vs * trm_diff

        assert np.allclose(diff, computed_flux.jac[fi].A.ravel())

    # On Neumann faces, the computed flux should equal the boundary condition. The
    # derivative should be zero.
    assert (
        computed_flux.val[model._neumann_face]
        == model._neumann_flux * div[1, model._neumann_face]
    )
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

    # We also need a difference in the vector source term. This must be projected onto
    # the vector from cell to face center for the respective cell.
    projected_vs_0 = _project_vector_source(fi, 0)
    projected_vs_1 = _project_vector_source(fi, 1)
    # Vector source difference. See comment related to p_diff for an explanation of the
    # divergence factor.
    vs_diff = (projected_vs_1 - projected_vs_0) * div[1, fi]

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
        flux = trm_full.dot([p0, p1]) + base_vector_source[fi] * vector_source_array

        assert np.isclose(flux, computed_flux.val[fi])

    trm_diff = np.array(
        [trm_full[0] + trm_diff_p0 * p_diff, trm_full[1] - trm_diff_p1 * p_diff]
    ).ravel()
    vector_source_diff = np.array(
        [trm_diff_p0 * vs_diff, -trm_diff_p1 * vs_diff]
    ).ravel()
    assert np.allclose(trm_diff + vector_source_diff, computed_flux.jac[fi].A)

    ####
    # Test the potential trace calculation
    potential_trace = model.potential_trace(
        model.mdg.subdomains(), model.pressure, model.permeability, "darcy_flux"
    ).value_and_jacobian(model.equation_system)

    # Base discretization matrix for the potential trace reconstruction, and for the
    # contribution from the vector source term.
    base_bound_pressure_cell = data[pp.DISCRETIZATION_MATRICES][model.darcy_keyword][
        "bound_pressure_cell"
    ]
    base_vector_source_bound = data[pp.DISCRETIZATION_MATRICES][model.darcy_keyword][
        "bound_pressure_vector_source"
    ]

    # On a Neumann face, the TPFA reconstruction of the potential at the face should be
    # the flux divided by the transmissibility.
    trm, trm_diff = _compute_half_transmissibility_and_derivative(
        model._neumann_face, 1
    )
    # The potential difference between the face and the adjacent cell, and its
    # derivative.
    dp = 1 / trm
    dp_diff = -1 * trm_diff / trm**2

    if base_discr == "tpfa":
        # For TPFA we can verify that the potential trace is reconstructed correctly.
        # For MPFA this check is not straightforward, as the potential trace
        # reconstruction involves other cells and boundary conditions on other faces.
        # Still, also for MPFA the present check gives a validation of the part of the
        # discretization that relates to the differentiable permeability.

        # The reconstruction of the potential trace consists of three terms: 1) The
        # pressure in the adjacent cell, 2) a pressure difference between the cell and
        # the face which drives the given flux, and 3) a contribution from the vector
        # source term.
        assert np.isclose(
            potential_trace.val[model._neumann_face],
            p1
            + dp * model._neumann_flux
            + (base_vector_source_bound[model._neumann_face] * vector_source_array)[0],
        )

    # Check the derivative of the potential trace reconstruction: Fetch the cell
    # contribution from the base discretization.
    cell_contribution = base_bound_pressure_cell[model._neumann_face].A.ravel()
    # For the cell next to the Neumann face, there is an extra contribution from the
    # derivative of the transmissibility.
    cell_contribution[ci] += dp_diff * model._neumann_flux

    assert np.allclose(potential_trace.jac[model._neumann_face].A, cell_contribution)

    # On a Dirichlet face, the potential trace should be equal to the Dirichlet value.
    assert np.isclose(
        potential_trace.val[model._nonzero_dirichlet_face], model._dirichlet_pressure
    )
    # The derivative of the potential trace with respect to the pressure should be zero,
    # as the potential trace is constant.
    assert np.allclose(
        potential_trace.jac[model._nonzero_dirichlet_face].A, 0, atol=1e-15
    )


class PoromechanicalTestDiffTpfa(
    SquareDomainOrthogonalFractures,
    pp.constitutive_laws.CubicLawPermeability,
    pp.constitutive_laws.AdDarcyFlux,
    pp.poromechanics.Poromechanics,
):
    """Helper class to test the derivative of the Darcy flux with respect to the mortar
    displacement.
    """

    def __init__(self, params):
        params.update(
            {
                "fracture_indices": [1],
                "grid_type": "cartesian",
                "meshing_arguments": {"cell_size_x": 0.5, "cell_size_y": 0.5},
            }
        )

        super().__init__(params)

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Non-constant permeability tensor, the y-component depends on pressure."""
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0, size=0)

        nc = sum([sd.num_cells for sd in subdomains])
        # K is a second order tensor having nd^2 entries per cell. 3d:
        # Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
        # 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
        tensor_dim = 3**2

        # Set constant component of the permeability - isotropic unit tensor
        all_vals = np.zeros(nc * tensor_dim, dtype=float)
        all_vals[0::tensor_dim] = 1
        all_vals[4::tensor_dim] = 1
        all_vals[8::tensor_dim] = 1

        # Basis vector for the yy-component
        e_yy = self.e_i(subdomains, i=4, dim=tensor_dim)

        return (
            pp.wrap_as_dense_ad_array(all_vals, name="Constant_permeability_component")
            + e_yy @ self.pressure(subdomains) ** 2
        )

    def initial_condition(self):
        """Set the initial condition for the problem.

        The interface displacement is non-zero in the y-direction to trigger a non-zero
        contribution from the derivative of the permeability with respect to the mortar
        displacement.
        """
        super().initial_condition()

        # Fetch the mortar interface and the subdomains.
        intf = self.mdg.interfaces()[0]
        g_2d, g_1d = self.mdg.subdomains()

        # Projection from the mortar to the primary grid, and directly from the mortar
        # cells to the high-dimensional cells. The latter uses an np.abs to avoid issues
        # with + and - in g.cell_faces.
        proj_high = intf.mortar_to_primary_int()
        mortar_to_high_cell = np.abs(g_2d.cell_faces.T @ proj_high)

        self.mortar_to_high_cell = mortar_to_high_cell

        # Set the mortar displacement, firs using the ordering of the 2d cells
        u_2d_x = np.array([0, 0, 0, 0])
        u_2d_y = np.array([0, 0, 1, 1])
        # .. and then map down to the mortar cells.
        u_mortar_x = mortar_to_high_cell.T @ u_2d_x
        u_mortar_y = mortar_to_high_cell.T @ u_2d_y

        # Define the full mortar displacement vector and set it in the equation system.
        u_mortar = np.vstack([u_mortar_x, u_mortar_y]).ravel("F")
        self.equation_system.set_variable_values(
            u_mortar, [self.interface_displacement_variable], iterate_index=0
        )

        # Store the y-component of the mortar displacement, using a Cartesian ordering
        # of the mortar cells (i.e., the same as the ordering of the 2d cells).
        self.u_mortar = u_2d_y

        # Fetch the global dof of the mortar displacement in the y-direction.
        dof_u_mortar_y = self.equation_system.dofs_of(
            [self.interface_displacement_variable]
        )[1::2]
        # Reorder the global dof to match the ordering of the 2d cells, and store it
        r, *_ = sps.find(mortar_to_high_cell)
        self.global_dof_u_mortar_y = dof_u_mortar_y[r]

        # Set the pressure variable in the 1d domain: The pressure is 2 in the leftmost
        # fracture cell, 0 in the rightmost fracture cell. This should give a flux
        # pointing to the right.
        if np.diff(g_1d.cell_centers[0])[0] > 0:
            p_1d = np.array([2, 0])
        else:
            p_1d = np.array([0, 2])

        p_1d_var = self.equation_system.get_variables(
            [self.pressure_variable], grids=[g_1d]
        )
        self.equation_system.set_variable_values(p_1d, p_1d_var, iterate_index=0)
        self.p_1d = p_1d

        # Set the interface Darcy flux to unity on all mortar cells.
        interface_flux = np.arange(intf.num_cells)
        self.equation_system.set_variable_values(
            interface_flux, [self.interface_darcy_flux_variable], iterate_index=0
        )
        self.interface_flux = interface_flux

        # Set the pressure in the 2d grid
        p_2d_var = self.equation_system.get_variables(
            [self.pressure_variable], grids=[g_2d]
        )
        p_2d = np.arange(g_2d.num_cells)
        self.equation_system.set_variable_values(p_2d, p_2d_var, iterate_index=0)
        self.p_2d = p_2d

        self.global_intf_ind = self.equation_system.dofs_of(
            [self.interface_darcy_flux_variable]
        )
        self.global_p_2d_ind = self.equation_system.dofs_of(p_2d_var)

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


@pytest.mark.parametrize("base_discr", ["tpfa", "mpfa"])
def test_derivatives_darcy_flux_potential_trace(base_discr: str):
    """Test the derivative of the Darcy flux with respect to the mortar displacement,
    and of the potential reconstruciton with respect to the interface flux.

    The test for the mortar displacement is done for a 2d domain with a single fracture,
    and a 1d domain with two cells. The mortar displacement is set to a constant value
    in the x-direction, but varying value in the y-direction. The permeability is given
    by the cubic law, and the derivative of the permeability with respect to the mortar
    displacement is non-zero. The test checks that the derivative of the Darcy flux with
    respect to the mortar displacement is correctly computed.

    The test for the interface flux uses the same overall setup, but the focus is on the
    2d domain, and the potential reconstruction on the internal boundary to the
    fracture. The permeability in the 2d domain is a function of the pressure, thus the
    pressure reconstruction operator should be dependent both on the 2d pressure and on
    the interface flux.

    """

    # Set up and discretize model
    model = PoromechanicalTestDiffTpfa({"base_discr": base_discr})
    model.prepare_simulation()
    model.discretize()

    # Fetch the mortar interface and the 1d subdomain.
    g_2d, g_1d = model.mdg.subdomains()

    ### First the test for the derivative of the Darcy flux with respect to the mortar
    # displacement.
    #
    # Get the y-component of the mortar displacement, ordered in the same way as the 2d
    # cells, that is
    #   2, 3
    #   0, 1
    # Thus the jumps in the mortar displacement are cell 3 - cell 2, and cell 1 - cell
    # 0.
    u_m = model.u_mortar

    # The permeability is given by the cubic law, calculate this and its derivative.
    resid_ap = model.residual_aperture([g_1d]).value(model.equation_system)
    k_0 = ((u_m[2] - u_m[0]) + resid_ap) ** 3 / 12
    k_1 = (u_m[3] - u_m[1] + resid_ap) ** 3 / 12

    # Derivative of the permeability with respect to the mortar displacement
    dk0_du2 = (u_m[2] - u_m[0] + resid_ap) ** 2 / 4
    dk0_du0 = -((u_m[2] - u_m[0] + resid_ap) ** 2) / 4
    dk1_du3 = (u_m[3] - u_m[1] + resid_ap) ** 2 / 4
    dk1_du1 = -((u_m[3] - u_m[1] + resid_ap) ** 2) / 4

    # Calculate the transmissibility. First, get the distance between the cell center and
    # the face center (will be equal on the two sides of the face).
    dist = np.abs(g_1d.face_centers[0, 1] - g_1d.cell_centers[0, 0])

    # Half transmissibility
    trm_0 = k_0 / dist
    trm_1 = k_1 / dist

    # The derivative of the transmissibility with respect to the mortar displacement is
    # given by the chain rule (a warm thanks to copilot):
    dtrm_du0 = (dk0_du0 / dist) * trm_1 / (trm_0 + trm_1) - trm_0 * trm_1 * dk0_du0 / (
        dist * (trm_0 + trm_1) ** 2
    )
    dtrm_du2 = (dk0_du2 / dist) * trm_1 / (trm_0 + trm_1) - trm_0 * trm_1 * dk0_du2 / (
        dist * (trm_0 + trm_1) ** 2
    )
    dtrm_du1 = (dk1_du1 / dist) * trm_0 / (trm_0 + trm_1) - trm_0 * trm_1 * dk1_du1 / (
        dist * (trm_0 + trm_1) ** 2
    )
    dtrm_du3 = (dk1_du3 / dist) * trm_0 / (trm_0 + trm_1) - trm_0 * trm_1 * dk1_du3 / (
        dist * (trm_0 + trm_1) ** 2
    )

    # We also need the pressure difference. Multiply with the sign of the divergence to
    # account for the direction of the normal vector.
    dp = (model.p_1d[1] - model.p_1d[0]) * g_1d.cell_faces[1, 1]

    # Finally, the true values that should be compared with the discretization.
    true_derivatives = dp * np.array([dtrm_du0, dtrm_du1, dtrm_du2, dtrm_du3])

    # The computed flux
    computed_flux = model.darcy_flux([g_1d]).value_and_jacobian(model.equation_system)
    # Pick out the middle face, and only those faces that are associated with the mortar
    # displacement in the y-direction.
    dt_du_computed = computed_flux.jac[1, model.global_dof_u_mortar_y].A.ravel()

    assert np.allclose(dt_du_computed, true_derivatives)

    ### Now the test for the derivative of the potential reconstruction with respect to
    # the interface flux.
    #
    # The potential reconstruction method
    potential_trace = model.potential_trace(
        model.mdg.subdomains(), model.pressure, model.permeability, "darcy_flux"
    ).value_and_jacobian(model.equation_system)

    # Fetch the permeability tensor from the data dictionary.
    data_2d = model.mdg.subdomain_data(model.mdg.subdomains()[0])
    k = data_2d[pp.PARAMETERS][model.darcy_keyword]["second_order_tensor"]
    # Only the yy-component is needed to calculate the reconstructed pressure, since the
    # face centers on the fracutre, and their respective 2d cell centers are aligned
    # with the x-axis.
    k_yy = k.values[1, 1, :]

    # Fetch the pressure in the 2d domain and the interface flux.
    p_2d = model.p_2d
    # The interface flux is mapped to the Cartesian ordering of the 2d cells.
    interface_flux = model.mortar_to_high_cell @ model.interface_flux

    # We know the distance from cell to face center (same for all cells and faces)
    dist = 0.5

    # The ordering of the fracture faces in the 2d domain is different from the
    # Cartesian ordering of the 2d cells, and from the ordering of the mortar cells. The
    # below code gives the fracture faces in the order corresponding to that of the 2d
    # cells.
    fracture_faces = np.where(g_2d.tags["fracture_faces"])[0]
    fracture_faces_cart_ordering = fracture_faces[
        g_2d.cell_faces[fracture_faces].indices
    ]

    # The reconstructed pressure is given by the sum of the pressure in the cell and the
    # pressure difference that drives the flux. The minus sign is needed since the
    # interface flux is defined as positive out of the high-dimensional cell.
    p_reconstructed = p_2d - interface_flux * dist / k_yy
    assert np.allclose(
        p_reconstructed, potential_trace.val[fracture_faces_cart_ordering]
    )

    # The Jacobian for the potential reconstruction should have two terms: One relating
    # to the pressure in the cell, and one relating to the interface flux.

    # We know that the yy-component of the permeability tensor contains a p^2 term
    dk_yy_dp = 2 * p_2d / dist
    # The Jacobian with respect to the pressure, represented as a diagonal matrix.
    true_jac_dp = np.diag(1 + interface_flux / (k_yy / dist) ** 2 * dk_yy_dp)

    # For ease of comparison, we reorder the Jacobian to match the ordering of the 2d
    # cells. No need to reorder the columns, as the 2d pressure already has the correct
    # ordering.
    assert np.allclose(
        true_jac_dp,
        potential_trace.jac[fracture_faces_cart_ordering][:, model.global_p_2d_ind].A,
    )

    # The computed Jacobian. Here we also need to reorder the columns from mortar to
    # high-dimensional cell ordering.
    computed_dp_dl = potential_trace.jac[fracture_faces_cart_ordering][
        :, model.mortar_to_high_cell @ model.global_intf_ind
    ].A
    # The true Jacobian with respect to the interface flux is found by differentiating
    # p_reconstructed with respect to the interface flux.
    true_dp_dl = np.diag(-dist / k_yy)
    assert np.allclose(computed_dp_dl, true_dp_dl)

    # Finally check there are no more terms in the Jacobian. We do no filtering of
    # elements that are essentially zero; EK would not have been surprised if that
    # turned out to be needed for Mpfa, but it seems to work without.
    assert potential_trace.jac[fracture_faces_cart_ordering].data.size == 8
