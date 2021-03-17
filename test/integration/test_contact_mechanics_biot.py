"""
Integration tests for the Biot modell, with and without contact mechanics.

We have the full Biot equations in the matrix, and mass conservation and contact
conditions in the fracture. For the contact mechanical part of this
test, please refer to test_contact_mechanics.
"""
import test.common.contact_mechanics_examples
import unittest

import numpy as np

import porepy as pp
import porepy.models.contact_mechanics_biot_model as model


class TestBiot(unittest.TestCase):
    def _solve(self, setup):
        pp.run_time_dependent_model(setup, {})

        gb = setup.gb

        g = gb.grids_of_dimension(setup._Nd)[0]
        d = gb.node_props(g)

        u = d[pp.STATE][setup.displacement_variable]
        p = d[pp.STATE][setup.scalar_variable]

        return u, p

    def test_pull_north_negative_scalar(self):

        m = SetupContactMechanicsBiot()
        m.uy_north = 0.001
        m.with_fracture = False
        m.mesh_args["mesh_size_bound"] = 0.5
        u, p = self._solve(m)

        # By symmetry (reasonable to expect from this grid), the average x displacement
        # should be zero
        self.assertTrue(np.abs(np.sum(u[0::2])) < m.zero_tol)
        # Check that the expansion yields a negative pressure
        self.assertTrue(np.all(p < -m.zero_tol))

    def test_positive_reference_scalar(self):
        """
        Test nonzero (positive) p_reference. See GradP for documentation.
        """
        m = SetupContactMechanicsBiot()
        m.with_fracture = False
        m.nx = [4, 4]
        m.fix_only_bottom = True
        # Set nonzero reference state
        m.p_reference = 1

        u, p = self._solve(m)
        # The initial and boundary conditions for pressure are 0. The absence of
        # the positive reference pressure results in contraction of the medium,
        # which is fixed at the bottom.
        u_x = u[0::2]
        u_y = u[1::2]
        self.assertTrue(np.all(u_y < -m.zero_tol))
        # Mirror indices left and right half of domain
        ind_left = [0, 1, 4, 5, 8, 9, 12, 13]
        ind_right = [3, 2, 7, 6, 11, 10, 15, 14]
        # Contraction implies positive x displacement in left half
        self.assertTrue(np.all(u_x[ind_left] > m.zero_tol))
        # x displacement should be symmetric about x=0.5
        self.assertTrue(np.all(np.isclose(u_x[ind_right] + u_x[ind_left], 0)))

        # Check that the contraction yields a positive pressure
        self.assertTrue(np.all(p > m.zero_tol))
        # and that it is symmetric about x=0.5
        self.assertTrue(np.all(np.isclose(p[ind_right] - p[ind_left], 0)))
        # Increasing pressure towards y=0.5 from top
        self.assertTrue(
            np.all((p[np.arange(8, 12)] - p[np.arange(12, 16)]) > m.zero_tol)
        )
        # and bottom
        self.assertTrue(np.all((p[np.arange(4, 8)] - p[np.arange(0, 4)]) > m.zero_tol))

    def test_negative_reference_scalar(self):
        """Simplified version of test_positive_reference_scalar.
        Less testing of symmetry, only that the domain expands and pressure reduces.
        """
        m = SetupContactMechanicsBiot()
        m.with_fracture = False
        m.nx = [4, 4]
        m.fix_only_bottom = True
        # Set nonzero reference state
        m.p_reference = -1

        u, p = self._solve(m)
        # The initial and boundary conditions for pressure are 0. The absence of
        # the positive reference pressure results in contraction of the medium,
        # which is fixed at the bottom.
        u_x = u[0::2]
        u_y = u[1::2]
        self.assertTrue(np.all(u_y > m.zero_tol))

        # Mirror indices left and right half of domain
        ind_left = [0, 1, 4, 5, 8, 9, 12, 13]
        ind_right = [3, 2, 7, 6, 11, 10, 15, 14]
        # Expansion implies negative x displacement in left half
        self.assertTrue(np.all(u_x[ind_left] < -m.zero_tol))
        # x displacement should be symmetric about x=0.5
        self.assertTrue(np.all(np.isclose(u_x[ind_right] + u_x[ind_left], 0)))

        # Check that the expansion yields a negative pressure
        self.assertTrue(np.all(p < -m.zero_tol))


class TestContactMechanicsBiot(unittest.TestCase):
    def _solve(self, setup):
        pp.run_time_dependent_model(setup, {"convergence_tol": 1e-6})

        gb = setup.gb

        nd = setup._Nd

        g2 = gb.grids_of_dimension(nd)[0]
        g1 = gb.grids_of_dimension(nd - 1)[0]

        d_m = gb.edge_props((g1, g2))
        d_1 = gb.node_props(g1)

        mg = d_m["mortar_grid"]

        u_mortar = d_m[pp.STATE][setup.mortar_displacement_variable]
        contact_force = d_1[pp.STATE][setup.contact_traction_variable]
        fracture_pressure = d_1[pp.STATE][setup.scalar_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_secondary_avg(nd=nd)
            * mg.sign_of_mortar_sides(nd=nd)
            * u_mortar
        )
        projection = d_1["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((nd, -1), order="F")

        contact_force = contact_force.reshape((nd, -1), order="F")

        return u_mortar_local_decomposed, contact_force, fracture_pressure

    def _verify_aperture_computation(self, setup, u_mortar):
        # Verify the computation of apertures.
        g = setup.gb.grids_of_dimension(setup._Nd - 1)[0]
        opening = np.abs(u_mortar[1])
        aperture = setup._compute_aperture(g, from_iterate=False)

        self.assertTrue(np.allclose(aperture, setup.initial_aperture + opening))

    def test_pull_north_positive_opening(self):
        setup = SetupContactMechanicsBiot()
        setup.uy_north = 0.001

        setup.mesh_args = [2, 2]
        setup.simplex = False
        # setup.subtract_fracture_pressure = False
        u_mortar, contact_force, fracture_pressure = self._solve(setup)
        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))

        # Check aperture computation
        self._verify_aperture_computation(setup, u_mortar)

    def test_pull_south_positive_opening(self):

        setup = SetupContactMechanicsBiot()
        setup.uy_south = -0.001

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))

        # Check aperture computation
        self._verify_aperture_computation(setup, u_mortar)

    def test_push_north_zero_opening(self):

        setup = SetupContactMechanicsBiot()
        setup.uy_north = -0.001

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))

        # Check aperture computation
        self._verify_aperture_computation(setup, u_mortar)

    def test_push_south_zero_opening(self):

        setup = SetupContactMechanicsBiot()
        setup.uy_south = 0.001

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))

        # Check aperture computation
        self._verify_aperture_computation(setup, u_mortar)

    def test_positive_fracture_pressure_positive_opening(self):

        setup = SetupContactMechanicsBiot()
        setup.scalar_source_value = 0.001

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Fracture pressure is positive
        self.assertTrue(np.all(fracture_pressure > 1e-7))

        # Check aperture computation
        self._verify_aperture_computation(setup, u_mortar)

    def test_time_dependent_pull_north_negative_scalar(self):
        """To obtain the value used in the corresponding TM test,
        test_pull_north_reduce_to_tm, uncomment the line
        setup.subtract_fracture_pressure = False
        """
        setup = SetupContactMechanicsBiot()
        setup.uy_north = 0.001
        setup.end_time *= 3
        setup.mesh_args = [2, 2]
        setup.simplex = False
        # setup.subtract_fracture_pressure = False
        u_mortar, contact_force, fracture_pressure = self._solve(setup)
        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))
        # If the update of the mechanical BC values for the previous time step used in
        # div u is missing, the effect is similar to if the pull on the north is
        # increased in each time step. This leads to a too small fracture pressure.
        self.assertTrue(np.all(np.isclose(fracture_pressure, -4.31072866e-06)))

        # Check aperture computation
        self._verify_aperture_computation(setup, u_mortar)

    def test_pull_south_positive_reference_scalar(self):
        """
        Compare with and without nonzero reference (and initial) state.
        """
        m_ref = SetupContactMechanicsBiot()
        m_ref.uy_south = -0.001
        m_ref.subtract_fracture_pressure = False
        u_mortar_ref, contact_force_ref, fracture_pressure_ref = self._solve(m_ref)

        gb = m_ref.gb
        g = gb.grids_of_dimension(m_ref._Nd)[0]
        d = gb.node_props(g)

        m = SetupContactMechanicsBiot()
        m.subtract_fracture_pressure = False
        m.uy_south = -0.001
        m.p_reference = 1
        m.p_initial = 1

        u_mortar, contact_force, fracture_pressure = self._solve(m)

        self.assertTrue(np.all(np.isclose(u_mortar, u_mortar_ref)))
        self.assertTrue(np.all(np.isclose(contact_force, contact_force_ref)))
        self.assertTrue(
            np.all(np.isclose(fracture_pressure, fracture_pressure_ref + 1))
        )


class SetupContactMechanicsBiot(
    test.common.contact_mechanics_examples.ProblemDataTime, model.ContactMechanicsBiot
):
    def __init__(self):

        self.mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }

        super().__init__()

        self.ux_south = 0
        self.uy_south = 0
        self.ux_north = 0
        self.uy_north = 0
        self.scalar_source_value = 0
        self.p_reference = 0
        self.p_initial = 0
        self.zero_tol = 1e-8
        self.fix_only_bottom = False

    def _set_mechanics_parameters(self):
        super()._set_mechanics_parameters()
        for g, d in self.gb:
            if g.dim == self._Nd - 1:
                d[pp.PARAMETERS][self.mechanics_parameter_key]["dilation_angle"] = (
                    np.pi / 6
                )
            p_ref = self.p_reference * np.ones(g.num_cells)
            d[pp.PARAMETERS][self.mechanics_parameter_key]["p_reference"] = p_ref

    def _bc_values_scalar(self, g):
        """
        It may be convenient to have p_dir=p_initial!=0 when investigating p_reference.
        """
        all_bf, east, west, north, south, _, _ = self._domain_boundary_sides(g)
        val = np.zeros(g.num_faces)
        val[north + south] = self.p_initial
        return val

    def _initial_condition(self) -> None:
        """
        Assign possibly nonzero (non-default) initial value.
        """
        super()._initial_condition()

        for g, d in self.gb:
            # Initial value for the scalar variable.
            initial_scalar_value = self.p_initial * np.ones(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})

            d[pp.STATE][pp.ITERATE].update(
                {self.scalar_variable: initial_scalar_value.copy()}
            )

    def _bc_type_mechanics(self, g):
        """
        For nonzero reference temperature, we only want to fix one boundary.
        """
        if not self.fix_only_bottom:
            return super()._bc_type_mechanics(g)
        _, _, _, north, south, _, _ = self._domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, south, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc


if __name__ == "__main__":
    unittest.main()
