"""Tests for TPFA-related functionality.

The tests fall into two categories:
    1) Tests of the standard TPFA discretization.
    2) Tests of the differentiable TPFA discretization.

"""
import pytest

import scipy.sparse as sps
import numpy as np
import porepy as pp

from porepy.applications.test_utils import common_xpfa_tests as xpfa_tests
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
)
from porepy.applications.md_grids import model_geometries
from porepy.applications.test_utils import well_models


"""Local utility functions."""


def _discretization_matrices(g, perm, bound):
    kw = "flow"
    data = pp.initialize_data(
        g, {}, kw, {"second_order_tensor": perm, "bc": bound, "inverter": "python"}
    )
    discr = pp.Tpfa(kw)

    discr.discretize(g, data)
    flux = data[pp.DISCRETIZATION_MATRICES][kw][discr.flux_matrix_key]
    bound_flux = data[pp.DISCRETIZATION_MATRICES][kw][discr.bound_flux_matrix_key]
    vector_source = data[pp.DISCRETIZATION_MATRICES][kw][discr.vector_source_matrix_key]
    div = g.cell_faces.T
    return div, flux, bound_flux, vector_source


"""Tests below.

The tests are identical to the ones in test_mpfa.py, except for the discretization.
They are therefore defined in test_utils.common_xpfa_tests.py, and simply run here.
This is to avoid code duplication while adhering to the contract that code is tested
in its mirror file in the test directories.
"""


def test_laplacian_stencil_cart_2d():
    """Apply MPFA on Cartesian grid, should obtain Laplacian stencil.

    See test_tpfa.py for the original test. This test is identical, except for the
    discretization method used.
    """
    xpfa_tests._test_laplacian_stencil_cart_2d(_discretization_matrices)


def test_symmetric_bc_common_with_mpfa():
    """Outsourced to helper functions for convenient reuse in test_mpfa.py."""
    xpfa_tests._test_symmetry_field_2d_periodic_bc(_discretization_matrices)
    xpfa_tests._test_laplacian_stensil_cart_2d_periodic_bcs(_discretization_matrices)


@pytest.mark.parametrize(
    "test_method",
    [
        xpfa_tests._test_gravity_1d_ambient_dim_1,
        xpfa_tests._test_gravity_1d_ambient_dim_2,
        xpfa_tests._test_gravity_1d_ambient_dim_3,
        xpfa_tests._test_gravity_1d_ambient_dim_2_nodes_reverted,
        xpfa_tests._test_gravity_2d_horizontal_ambient_dim_3,
        xpfa_tests._test_gravity_2d_horizontal_ambient_dim_2,
        xpfa_tests._test_gravity_2d_horizontal_periodic_ambient_dim_2,
    ],
)
def test_tpfa_gravity_common_with_mpfa(test_method):
    """See test_utils.common_xpfa_tests.py for the original tests."""
    test_method("tpfa")


discr_instance = pp.Tpfa("flow")


class TestTpfaBoundaryPressure(xpfa_tests.XpfaBoundaryPressureTests):
    """Tests for the boundary pressure computation in MPFA. Accesses the fixture
    discr_instance, otherwise identical to the tests in test_utils.common_xpfa_tests.py
    and used in test_tpfa.py.

    """

    @property
    def discr_instance(self):
        """Return a tpfa instance."""
        return discr_instance


# Tests for differentiable TPFA


class _SetFluxDiscretizations:
    """Helper class with a method to set the Darcy flux variable."""

    def darcy_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpfaAd | pp.ad.TpfaAd:
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

    def fourier_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpfaAd | pp.ad.TpfaAd:
        """Discretization object for the Fourier flux term.

        Parameters:
            interfaces: List of mortar grids where the Fourier flux is defined.

        Returns:
            Discretization of the Fourier flux.

        """
        if self.params["base_discr"] == "tpfa":
            return pp.ad.TpfaAd(self.fourier_keyword, subdomains)
        else:
            return pp.ad.MpfaAd(self.fourier_keyword, subdomains)


class UnitTestAdTpfaFlux(
    pp.constitutive_laws.DarcysLawAd,
    _SetFluxDiscretizations,
    pp.fluid_mass_balance.SinglePhaseFlow,
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


class TestDiffTpfaGridsOfAllDimensions(
    CubeDomainOrthogonalFractures,
    _SetFluxDiscretizations,
    pp.constitutive_laws.CubicLawPermeability,
    pp.constitutive_laws.DarcysLawAd,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """Helper class to test that the methods for differentiating diffusive fluxes and
    potential reconstructions work on grids of all dimensions.
    """

    def __init__(self, params):
        params.update(
            {
                "fracture_indices": [0, 1, 2],
            }
        )
        if params["grid_type"] == "cartesian":
            params["meshing_arguments"] = {"cell_size": 0.5}
        else:  # Simplex
            params["mesh_args"] = {"mesh_size_frac": 0.5, "mesh_size_min": 0.5}

        super().__init__(params)

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
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
        """Set a random initial condition, to avoid the trivial case of a constant
        permeability tensor and trivial pressure and interface flux fields.
        """
        super().initial_condition()
        num_dofs = self.equation_system.num_dofs()
        values = np.random.rand(num_dofs)
        self.equation_system.set_variable_values(values, iterate_index=0)


@pytest.mark.parametrize("base_discr", ["tpfa", "mpfa"])
@pytest.mark.parametrize("grid_type", ["cartesian", "simplex"])
def test_diff_tpfa_on_grid_with_all_dimensions(base_discr: str, grid_type: str):
    """Test that the methods for differentiating diffusive fluxes and potential
    reconstructions work on grids of all dimensions.

    The method verifies that the methods can be called, and the resulting operators can
    be parsed, without raising exceptions. Also, the size and shape of the computed
    quantities are verified. The actual elements in the value vector and Jacobian matrix
    are not checked.

    """
    model = TestDiffTpfaGridsOfAllDimensions(
        {"base_discr": "tpfa", "grid_type": grid_type}
    )
    model.prepare_simulation()

    num_faces = sum([sd.num_faces for sd in model.mdg.subdomains()])
    num_dofs = model.equation_system.num_dofs()

    darcy_flux = model.darcy_flux(model.mdg.subdomains())
    darcy_value = darcy_flux.value(model.equation_system)
    assert darcy_value.size == num_faces

    darcy_jac = darcy_flux.value_and_jacobian(model.equation_system).jac
    assert darcy_jac.shape == (num_faces, num_dofs)

    potential_trace = model.potential_trace(
        model.mdg.subdomains(), model.pressure, model.permeability, "darcy_flux"
    )
    potential_value = potential_trace.value(model.equation_system)
    assert potential_value.size == num_faces

    potential_jac = potential_trace.value_and_jacobian(model.equation_system).jac
    assert potential_jac.shape == (num_faces, num_dofs)


# Test that a standard discretization and a differentiable discretization give the same
# linear system for a constant permeability.


class WithoutDiffTpfa(
    _SetFluxDiscretizations,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Helper class to test that the methods for differentiating diffusive fluxes and
    potential reconstructions work on grids of all dimensions.
    """

    def initial_condition(self):
        """Set a random initial condition, to avoid the trivial case of a constant
        pressure.
        """
        super().initial_condition()
        num_dofs = self.equation_system.num_dofs()
        np.random.seed(42)
        values = np.random.rand(num_dofs)
        self.equation_system.set_variable_values(values, iterate_index=0)


class WithDiffTpfa(
    pp.constitutive_laws.DarcysLawAd,
    pp.constitutive_laws.FouriersLawAd,
    WithoutDiffTpfa,
):
    """Helper class to test that the methods for differentiating diffusive fluxes and
    potential reconstructions work on grids of all dimensions.

    We use the default thermal conductivity, which should be constant (in fact same as
    permeability as defined below).
    """

    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Constant permeability tensor."""
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0, size=0)

        nc = sum([sd.num_cells for sd in subdomains])
        tensor_dim = 3**2

        all_vals = np.zeros(nc * tensor_dim, dtype=float)
        all_vals[0::tensor_dim] = 1
        all_vals[4::tensor_dim] = 1
        all_vals[8::tensor_dim] = 1

        return pp.wrap_as_dense_ad_array(
            all_vals, name="Constant_permeability_component"
        )


@pytest.mark.parametrize("base_discr", ["tpfa", "mpfa"])
def test_diff_tpfa_and_standard_tpfa_give_same_linear_system(base_discr: str):
    """Discretize the same problem with a standard TPFA discretization and a
    differentiable TPFA discretization, where the latter also has a constant
    permeability, but given on 'differentiable form'. The Jacobian matrix and the
    residual vectors should be the same.
    """
    model_without_diff = WithoutDiffTpfa({"base_discr": base_discr})
    model_with_diff = WithDiffTpfa({"base_discr": base_discr})

    matrix, vector = [], []

    for mod in [model_without_diff, model_with_diff]:
        mod.prepare_simulation()
        mod.assemble_linear_system()

        matrix.append(mod.linear_system[0])
        vector.append(mod.linear_system[1])

    assert np.allclose(matrix[0].A, matrix[1].A)
    assert np.allclose(vector[0], vector[1])


class TestDiffTpfaFractureTipsInternalBoundaries(
    model_geometries.OrthogonalFractures3d,
    well_models.OneVerticalWell,
    well_models.BoundaryConditionsWellSetup,
    _SetFluxDiscretizations,
    pp.constitutive_laws.DarcysLawAd,
    pp.constitutive_laws.FouriersLawAd,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Helper class to test that the methods for differentiating diffusive fluxes and
    potential reconstructions work as intended on fracture tips and internal boundaries.

    The model geometry consists of two fractures: One extending to the boundary, one
    that is immersed in the domain. The model also includes a well which intersects
    with one of the fractures.
    """

    def __init__(self, params):
        # From the default fractures, use one with a constant z-coordinate.
        params.update({"fracture_indices": [2]})
        super().__init__(params)

    def set_fractures(self) -> None:
        """In addition to the default fracture (which extends to the boundary), we add
        one fracture that is fully immersed in the domain.
        """
        super().set_fractures()

        # Add a fracture that is immersed in the domain, to test the discretization of
        # the Darcy flux on a fracture that is not on the boundary.
        frac = pp.PlaneFracture(
            np.array([[0.3, 0.3, 0.3, 0.3], [0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8]])
        )
        self._fractures.append(frac)

    def initial_condition(self):
        """Set a random initial condition, to avoid the trivial case of a constant
        pressure hiding difficulties.
        """
        super().initial_condition()
        num_dofs = self.equation_system.num_dofs()
        np.random.seed(42)
        values = np.random.rand(num_dofs)
        self.equation_system.set_variable_values(values, iterate_index=0)


@pytest.mark.parametrize("base_discr", ["tpfa", "mpfa"])
def test_flux_potential_trace_on_tips_and_internal_boundaries(base_discr: str):
    """Test that the flux and potential trace can be computed on fracture tips and
    internal boundaries.

    Both Darcy and Fourier fluxes are tested.

    For the fluxes, we test that the Jacobian matrix is zero on the Neumann faces (which
    include fracture tips, internal boundaries, and any external boundaries that are
    assigned a Neumann boundary condition). On tips we also test that the potential
    trace is equal to the pressure in the adjacent cell.

    """
    model = TestDiffTpfaFractureTipsInternalBoundaries({"base_discr": base_discr})
    model.prepare_simulation()

    mdg = model.mdg

    for sd in mdg.subdomains():
        data = mdg.subdomain_data(sd)

        # For both Darcy and Fourier flux, check that the Jacobian matrix is zero on
        # Neumann faces.
        bc_darcy = data[pp.PARAMETERS][model.darcy_keyword]["bc"]
        darcy_flux = model.darcy_flux([sd]).value_and_jacobian(model.equation_system)
        assert np.allclose(darcy_flux.jac[bc_darcy.is_neu].data, 0)

        bc_fourier = data[pp.PARAMETERS][model.fourier_keyword]["bc"]
        fourier_flux = model.fourier_flux([sd]).value_and_jacobian(
            model.equation_system
        )
        assert np.allclose(fourier_flux.jac[bc_fourier.is_neu].data, 0)

        # The potential trace should be equal to the potential in the adjacent cell on
        # fracture tip faces (but not on internal nor external boundaries, where
        # boundary conditions may change the boundary value).

        # Get the indices of the fracture tip faces and that of the adjacent cell.
        tip_faces = np.where(
            np.logical_and(
                sd.tags["tip_faces"], np.logical_not(sd.tags["domain_boundary_faces"])
            )
        )[0]
        _, tip_cells = sd.signs_and_cells_of_boundary_faces(tip_faces)

        # Check that the pressure trace is equal to the pressure in the adjacent cell.
        pressure_trace = model.potential_trace(
            [sd], model.pressure, model.permeability, "darcy_flux"
        ).value(model.equation_system)
        p = model.pressure([sd]).value(model.equation_system)
        assert np.allclose(pressure_trace[tip_faces], p[tip_cells])
        # Check that the temperature trace is equal to the temperature in the adjacent
        # cell.
        temperature_trace = model.potential_trace(
            [sd], model.temperature, model.thermal_conductivity, "fourier_flux"
        ).value(model.equation_system)
        T = model.temperature([sd]).value(model.equation_system)
        assert np.allclose(temperature_trace[tip_faces], T[tip_cells])
