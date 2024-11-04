"""Tests for the momentum balance model class. """

from __future__ import annotations

import copy

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.models import (
    MomentumBalance,
    compare_scaled_model_quantities,
    compare_scaled_primary_variables,
)
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
    CubeDomainOrthogonalFractures,
)


class LinearModel(
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    MomentumBalance,
):
    pass


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({}, 0.0),
        ({"characteristic_displacement": 42}, -0.1),
        ({"porosity": 0.5}, 0.2),
    ],
)
def test_2d_single_fracture(solid_vals, north_displacement):
    """Test that the solution is qualitatively sound.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.

    """
    # Instantiate constants and store in params.
    solid = pp.SolidConstants(solid_vals)
    params = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"solid": solid},
        "u_north": [0.0, north_displacement],
    }

    # Create model and run simulation
    setup = LinearModel(params)
    pp.run_time_dependent_model(setup)

    # Check that the pressure is linear
    sd = setup.mdg.subdomains(dim=setup.nd)[0]
    var = setup.equation_system.get_variables([setup.displacement_variable], [sd])
    vals = setup.equation_system.get_variable_values(variables=var, time_step_index=0)
    if np.isclose(north_displacement, 0):
        assert np.allclose(vals, 0)
    else:
        if north_displacement < 0:
            # Boundary displacement is negative, so the y displacement should be
            # negative
            assert np.all(np.sign(vals[setup.nd - 1 :: setup.nd]) < 0)

            # Check that x displacement has the same sign as north_displacement for
            # x<0.5, and the opposite sign for x>0.5. To see why this makes sense, think
            # through what happens around the symmetry line of x=0.5 when pulling or
            # pushing the top (north) boundary.
            tol = 1e-10
            left = sd.cell_centers[0] < setup.domain.bounding_box["xmax"] / 2 - tol
            right = sd.cell_centers[0] > setup.domain.bounding_box["xmax"] / 2 + tol
            assert np.all(
                np.sign(vals[:: setup.nd][left]) == np.sign(north_displacement)
            )
            assert np.all(
                np.sign(vals[:: setup.nd][right]) == -np.sign(north_displacement)
            )
        else:
            # Check that y displacement is positive in top half of domain
            top = sd.cell_centers[1] > 0.5
            assert np.all(
                np.sign(vals[setup.nd - 1 :: setup.nd][top])
                == np.sign(north_displacement)
            )
            # Fracture cuts the domain in half, so the bottom half should be undisplaced.
            bottom = sd.cell_centers[1] < 0.5
            assert np.allclose(vals[setup.nd - 1 :: setup.nd][bottom], 0)
            # No displacement in x direction
            assert np.allclose(vals[:: setup.nd], 0)

    # Check that the displacement jump and traction are as expected
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)
    jump = setup.displacement_jump(sd_frac).value(setup.equation_system)
    traction = setup.contact_traction(sd_frac).value(setup.equation_system)
    if north_displacement > 0:
        # Normal component of displacement jump should be positive
        assert np.all(jump[setup.nd - 1 :: setup.nd] > 0)
        # Traction should be zero
        assert np.allclose(traction, 0)
    else:
        # Displacement jump should be zero
        assert np.all(np.isclose(jump, 0))
        # Normal traction should be non-positive. Zero if north_displacement is zero.
        if north_displacement < 0:
            assert np.all(traction[setup.nd - 1 :: setup.nd] <= 0)
        else:
            assert np.allclose(traction, 0)


@pytest.mark.parametrize("units", [{"m": 0.29, "kg": 0.31, "K": 4.1}])
@pytest.mark.parametrize("uy_north", [2e-4, -1e-4])
def test_unit_conversion(units: dict, uy_north: float):
    """Test that solution is independent of units.

    Parameters:
        units: Dictionary with keys as those in
            :class:`~pp.models.material_constants.MaterialConstants`.
        uy_north: Value of y displacement on the north boundary.

    """
    solid = pp.SolidConstants(pp.solid_values.extended_granite_values_for_testing)

    params = {
        "times_to_export": [],  # Suppress output for tests
        "fracture_indices": [0, 1],
        "cartesian": True,
        "u_north": [0.0, uy_north],
        "material_constants": {"solid": solid},
    }
    reference_params = copy.deepcopy(params)
    # Create model and run simulation.
    setup_0 = LinearModel(reference_params)
    pp.run_time_dependent_model(setup_0)

    params["units"] = pp.Units(**units)
    setup_1 = LinearModel(params)

    pp.run_time_dependent_model(setup_1)
    variables = [
        setup_1.displacement_variable,
        setup_1.interface_displacement_variable,
        setup_1.contact_traction_variable,
    ]
    variable_units = ["m", "m", "-"]
    compare_scaled_primary_variables(setup_0, setup_1, variables, variable_units)
    secondary_variables = ["stress", "displacement_jump"]
    secondary_units = ["Pa * m", "m"]
    domain_dimensions = [2, 1]
    compare_scaled_model_quantities(
        setup_0, setup_1, secondary_variables, secondary_units, domain_dimensions
    )


class LithostaticModel(
    pp.constitutive_laws.GravityForce, pp.momentum_balance.MomentumBalance
):
    """Model class to test the computation of lithostatic stress.

    The model sets up a column where the lateral sides (x-direction for 2d, x and y for
    3d) cannot move in the lateral plane, but is free to move in the vertical direction.
    The top boundary is free to move in all directions, and the bottom boundary is
    clamped. The test measures the vertical displacement and compares it to the
    analytical solution; moreover, it checks that the stress at the bottom of the
    column matches the weight of the column.

    """

    def __init__(self, params):
        super().__init__(params)

    def set_domain(self):
        """The domain is a column of height 10, with a non-trivial cross section."""
        if self.params["dim"] == 2:
            domain = {"xmin": 0, "xmax": 42, "ymin": 0, "ymax": 10}
        elif self.params["dim"] == 3:
            domain = {
                "xmin": 0,
                "xmax": 0.42,
                "ymin": 0,
                "ymax": 0.42,
                "zmin": 0,
                "zmax": 10,
            }

        self._domain = pp.Domain(domain)

    def meshing_arguments(self):
        # A single cell in the x and (for 3d) y direction. 100 cells in the z direction.
        if self.params["dim"] == 2:
            default_meshing_args: dict[str, float] = {
                "cell_size_x": 42.0,
                "cell_size_y": 0.4,
            }
        elif self.params["dim"] == 3:
            default_meshing_args: dict[str, float] = {
                "cell_size_x": 0.42,
                "cell_size_y": 0.42,
                "cell_size_z": 0.4,
            }
        return default_meshing_args

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Lateral sides: No motion in the x-direction (xy-plane for 3d), free motion in the
        vertical direction. Bottom: No motion. Top: Free motion.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        domain_sides = self.domain_boundary_sides(sd)

        if self.nd == 2:
            lateral_sides = np.logical_or.reduce((domain_sides.east, domain_sides.west))
            top_side = domain_sides.north
        elif self.nd == 3:
            lateral_sides = np.logical_or.reduce(
                (
                    domain_sides.east,
                    domain_sides.west,
                    domain_sides.north,
                    domain_sides.south,
                )
            )
            top_side = domain_sides.top

        bc.is_neu[self.nd - 1, lateral_sides] = True
        bc.is_dir[self.nd - 1, lateral_sides] = False
        bc.is_neu[:, top_side] = True
        bc.is_dir[:, top_side] = False

        return bc


@pytest.mark.parametrize("dim", [2, 3])
def test_lithostatic(dim: int):
    """Test that the solution is qualitatively sound.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.

    """
    # Create model and run simulation
    model = LithostaticModel({"dim": dim, "times_to_export": []})
    pp.run_stationary_model(model, {})

    # Check that the pressure is linear
    sd = model.mdg.subdomains(dim=model.nd)[0]
    # Fetch the displacement variable and convert it to an model.nd x model.num_cells
    # array.
    var = model.equation_system.get_variables([model.displacement_variable], [sd])
    vals = model.equation_system.get_variable_values(variables=var, time_step_index=0)
    vals = vals.reshape((model.nd, -1), order="F")

    # Analytical displacement.
    g = model.solid.convert_units(pp.GRAVITY_ACCELERATION, "m * s^-2")
    rho = model.solid.convert_units(model.solid.density(), "kg * m^-3")
    data = model.mdg.subdomain_data(sd)
    stiffness = data[pp.PARAMETERS][model.stress_keyword]["fourth_order_tensor"]
    E = 2 * stiffness.mu[0] + stiffness.lmbda[0]
    z_max = sd.nodes[model.nd - 1].max()
    z = sd.cell_centers[model.nd - 1]
    u_z = -rho * g / E * (z_max * z - z**2 / 2)

    for i in range(model.nd - 1):
        assert np.allclose(vals[i], 0)
    # EK, note for future reference (e.g., debugging): The difference between the
    # computed and analytical vertical displacement is uniform throughout the domain.
    # For reference, its value in 3d, with a grid size of 0.1 in the z-direction of
    # maginute and with a domain height of 10 is 0.0040861. Refining the grid with one
    # order of magnitude, seems to reduce the error by precisely (?) two orders of
    # magnitude. It is unclear to EK whether this is significant or not.
    assert np.allclose(vals[model.nd - 1], u_z, 7e-2)

    # Computed stress at the bottom of the domain.
    computed_stress = (
        model.stress([sd])
        .value(model.equation_system)
        .reshape((model.nd, -1), order="F")
    )
    bottom_face = np.where(model.domain_boundary_sides(sd).bottom)[0]
    bottom_traction = computed_stress[model.nd - 1, bottom_face]
    # The stress at the bottom of the domain should be equal to the weight of the
    # column.
    assert np.allclose(bottom_traction, -rho * g * sd.cell_volumes.sum())


"""Tests for elastoplastic split of fracture deformation below."""


class ElastoplasticModel2d(
    SquareDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.models.momentum_balance.MomentumBalance,
):

    pass


# We set a high shear modulus to make the domain stiff (especially for shear), and a
# low tangential fracture stiffness to make the fracture weak. This will make the
# fracture deform significantly more than the surrounding domain. Still, the domain
# will deform slightly, requiring a tolerance when comparing the results.
solid_vals_elastoplastic = {
    "fracture_tangential_stiffness": 1e-5,
    "shear_modulus": 1e6,
    "lame_lambda": 1e2,
}


def verify_elastoplastic_deformation(
    setup: pp.models.MomentumBalance,
    u_e_expected: list[pp.number],
    u_p_expected: list[pp.number],
    u_top_expected: list[pp.number],
    traction_expected: list[pp.number],
    tols: list[float],
    compare_means: bool = False,
):
    """Verify the results of a simulation.

    The function will raise an assertion error if the results do not match the expected
    values within the given tolerances.

    Parameters:
        setup: The model setup.
        u_e_expected: ``len=nd``

            Expected values of the elastic displacement jump in the x and y directions.
        u_p_expected: ``len=nd``

            Expected values of the plastic displacement jump in the x and y directions.
        u_top_expected: ``len=nd``

            Expected value of displacement in the cells above the fracture. nan values
            are ignored, i.e., the test will pass if all computed values match the
            corresponding non-nan expected values.
        tols: ``len=4``

            Tolerances for the comparisons. The first two are for the elastic and
            plastic displacement jumps, respectively, the third is for the displacement
            in the cells above the fracture and the fourth is for the traction on the
            fracture.
        compare_means: Whether to compare the means of the computed values to the
            expected values. If False, the values are compared element-wise.

    Returns:
        Tuple of the computed elastic displacement jump, plastic displacement jump,
        displacement in the cells above the fracture and traction on the fracture.


    """
    nd = setup.nd  # Shorthand for number of dimensions.
    # Get the indices of the tangential components in global coordinates. Hardcoded
    # based on the assumption that the fracture has constant y-coordinate.
    fracture_ind = 1
    tang_ind = np.setdiff1d(np.arange(nd), fracture_ind)
    matrix = setup.mdg.subdomains(dim=nd)[0]
    fractures = setup.mdg.subdomains(dim=nd - 1)
    assert len(fractures) == 1  # Below code assumes a single fracture.
    fracture = fractures[0]

    # Get plastic and elastic displacement jumps on the fracture in local coordinates.
    u_p_loc = setup.plastic_displacement_jump(fractures).value(setup.equation_system)
    u_e_loc = setup.elastic_displacement_jump(fractures).value(setup.equation_system)

    # Transform the jumps to global coordinates and corresponding to the j side of the
    # fracture being the one with the lower y-coordinate (jumps are k-j).

    # First, rotate the local displacements to global coordinates. Then, switch sign if
    # the fracture normal points downwards. This is needed to counteract the cases when
    # the j ("left") side of the fracture is the one with the higher y-coordinate, i.e.
    # the upper half of the domain.
    proj = setup.mdg.subdomain_data(fracture)["tangential_normal_projection"]
    n = proj.normals
    rot = proj.project_tangential_normal().T
    u_p = rot @ u_p_loc
    u_e = rot @ u_e_loc

    sign = np.tile(np.sign(n[1]), (setup.nd, 1)).ravel("F")
    u_p = (sign * u_p).reshape((nd, -1), order="F")
    u_e = (sign * u_e).reshape((nd, -1), order="F")

    u_domain = (
        setup.displacement([matrix])
        .value(setup.equation_system)
        .reshape((nd, -1), order="F")
    )
    u_top = u_domain[:, matrix.cell_centers[fracture_ind] > 0.5]
    # Compare the computed values to the expected values.
    if compare_means:
        assert np.allclose(np.mean(u_e, axis=1), u_e_expected, rtol=tols[0])
    else:
        assert np.allclose(u_e, np.reshape(u_e_expected, (nd, 1)), rtol=tols[0])

    if compare_means:
        assert np.allclose(np.mean(u_p, axis=1), u_p_expected, rtol=tols[1])
    else:
        assert np.allclose(u_p, np.reshape(u_p_expected, (nd, 1)), rtol=tols[1])

    # Matrix displacement above the fracture. We only compare those values that are not
    # nan in the expected values.
    mask_u = np.logical_not(np.isnan(u_top_expected))
    if compare_means:
        expected = np.array(u_top_expected)[mask_u]
        assert np.allclose(np.mean(u_top, axis=1)[mask_u], expected, rtol=tols[2])
    else:
        expected = np.reshape(u_top_expected, (nd, 1))[mask_u]
        assert np.allclose(u_top[mask_u], expected, rtol=tols[2])

    # Traction on the fracture.
    open_cells = u_p[fracture_ind] > 1e-10
    traction = (
        setup.characteristic_contact_traction([fracture])
        * setup.contact_traction([fracture])
    ).value(setup.equation_system)
    # Rotate to global coordinates.
    traction = rot @ traction
    traction = (sign * traction).reshape((nd, -1), order="F")

    # The two next assertions should hold for all cells. Thus, we use a very low
    # tolerance and don't compare means.
    # Check that tangential part of traction and u_e are parallel and have relative
    # magnitudes equal to stiffness for closed cells.
    assert np.allclose(
        traction[tang_ind][:, ~open_cells] / u_e[tang_ind][:, ~open_cells],
        setup.solid.fracture_tangential_stiffness(),
        atol=1e-10,
    )
    # Check that open cells have zero traction.
    assert np.allclose(traction[:, open_cells], 0, atol=1e-10)
    # Compare to expected values.
    assert np.allclose(traction, np.reshape(traction_expected, (nd, 1)), rtol=tols[3])
    return u_e, u_p, u_top, traction


@pytest.mark.parametrize(
    "u_north, u_e_expected,u_p_expected,u_expected,traction_expected",
    [
        # Compression and elastic shear. -2e6 is the expected value of the normal
        # traction, which is the negative (sign convention) and dominated by the shear
        # modulus (1e6, remember factor 2 for \mu in Hooke's law!). 1e-5 is the expected
        # value of the traction in the x direction, which is computed from the
        # tangential stiffness (1e-5) and the tangential displacement (1).
        ([1.0, -1.0], [1, 0], [0, 0], [1, np.nan], [1e-5, -2e6]),
        # Extension and plastic shear.
        ([1.0, 1.0], [0, 0], [1, 1], [1, 1], [0, 0]),
    ],
)
def test_elastoplastic_2d_single_fracture(
    u_north: list[pp.number],
    u_e_expected: list[pp.number],
    u_p_expected: list[pp.number],
    u_expected: list[pp.number],
    traction_expected: list[pp.number],
):
    """Test that the solution is qualitatively sound.

    Parameters:
        north_displacement: Value of displacement on the north boundary.
        u_e_expected: Expected values of the elastic displacement jump in global
            coordinates.
        u_p_expected: Expected values of the plastic displacement jump in global
            coordinates.
        u_expected: Expected value of displacement of cells above the fracture.
        traction_expected: Expected value of traction on the fracture in global
            coordinates.

    """
    # Instantiate constants and store in params.

    solid = pp.SolidConstants(solid_vals_elastoplastic)
    params = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"solid": solid},
        "u_north": u_north,
        "fracture_indices": [1],  # Single fracture with constant y coordinate.
    }

    # Create model and run simulation.
    setup = ElastoplasticModel2d(params)
    pp.run_time_dependent_model(setup, params)
    verify_elastoplastic_deformation(
        setup,
        u_e_expected,
        u_p_expected,
        u_expected,
        traction_expected,
        [1e-3, 1e-10, 1e-3, 1e-3],
    )


class ElastoplasticModel3d(
    CubeDomainOrthogonalFractures,
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    pp.models.momentum_balance.MomentumBalance,
):

    pass


cases_3d = [
    # Shear and, since u_y_north is negative, compression. With low tangential
    # stiffness, the fracture will deform elastically, accounting for almost all of the
    # displacement (very little in the matrix). The normal traction is dominated by the
    # shear modulus. The tangential traction is computed from the normal traction and
    # the tangential stiffness (see also test_2d_single_fracture). The relative
    # magnitude of the two tangential components is computed from the Pythagorean
    # theorem.
    (
        [2.0, -1.0, 3],
        [2, 0, 3],
        [0, 0, 0],
        [2, np.nan, 3],
        [2e-5, -2e6, 3e-5],
    ),
    # Shear and, since u_y_north is positive, extension. With opening, no elastic
    # deformation occurs, and the displacement is entirely plastic, identically 0
    # in the matrix.
    ([2.0, 1.0, 3], [0, 0, 0], [2, 1, 3], [2, 1, 3], [0, 0, 0]),
]


@pytest.mark.parametrize(
    "u_north,u_e_expected,u_p_expected,u_expected,traction_expected",
    cases_3d,
)
def test_elastoplastic_3d_single_fracture(
    u_north: list[pp.number],
    u_e_expected: list[pp.number],
    u_p_expected: list[pp.number],
    u_expected: list[pp.number],
    traction_expected: list[pp.number],
):
    """Test that the solution is qualitatively sound.

    The fracture is in the x-z plane, with constant y coordinate. Thus, x and z are the
    (global) tangential directions, and y is the normal direction.

    Parameters:
        u_north: Value of displacement on the north boundary.
        u_e_expected: Expected values of the elastic displacement jump in the x and y
            directions.
        u_p_expected: Expected values of the plastic displacement jump in the x and y
            directions.
        u_expected: Expected values of displacement in the x and z directions of cells
            above the fracture.
        traction_expected: Expected values of traction on the fracture in global
            coordinates.

    """
    # Instantiate constants and store in params.
    solid = pp.SolidConstants(solid_vals_elastoplastic)
    params = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"solid": solid},
        "fracture_indices": [1],
        "u_north": u_north,
    }

    # Create model and run simulation
    setup = ElastoplasticModel3d(params)
    pp.run_time_dependent_model(setup, params)
    verify_elastoplastic_deformation(
        setup,
        u_e_expected,
        u_p_expected,
        u_expected,
        traction_expected,
        [1e-3, 1e-10, 1e-3, 1e-3],
    )


class TimeDependentBCs(
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
):

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values.

        Initial value is u_y = self.solid.fracture_gap() at north boundary. Adding it on
        the boundary ensures a stress-free initial state. For positive times, a tailored
        displacement is imposed on the north boundary. The south boundary is fixed.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the problem,
            for each face in the subdomain.

        """
        sides = self.domain_boundary_sides(bg)
        values = np.zeros((self.nd, bg.num_cells))

        # Add fracture width on top if there is a fracture.
        if len(self.mdg.subdomains()) > 1:
            frac_val = self.solid.fracture_gap()
        else:
            frac_val = 0
        values[1, sides.north] = frac_val

        if bg.dim < self.nd - 1:
            return values.ravel("F")
        if self.time_manager.time > 1e-5:
            # Create slip for second time step.
            u_z = 50.0 if self.time_manager.time > 1.1 else 1.0
            u_n = np.tile([1, -1, u_z], (bg.num_cells, 1)).T
            values[:, sides.north] += self.solid.convert_units(u_n, "m")[:, sides.north]
        return values.ravel("F")


class ElastoplasticModelTimeDependentBCs(
    CubeDomainOrthogonalFractures,
    TimeDependentBCs,
    pp.models.momentum_balance.MomentumBalance,
):

    pass


def test_time_dependent_bc():
    solid = pp.SolidConstants(
        {
            "fracture_tangential_stiffness": 1e-1,
            "shear_modulus": 1e0,
            "lame_lambda": 1e0,
        }
    )
    params = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"solid": solid},
        "fracture_indices": [1],
        "time_manager": pp.TimeManager([0.0, 1.0], 1.0, True),
    }

    # Create model and run simulation. The north displacement is 1, -1, 1.
    setup = ElastoplasticModelTimeDependentBCs(params)
    pp.run_time_dependent_model(setup, params)
    tols = [5e-2, 1e-10, 1e-3, 5e-2]

    verify_elastoplastic_deformation(
        setup,
        [0.86, 0, 0.86],
        [0, 0, 0],
        [np.nan, np.nan, np.nan],
        [0.086, -2.54, 0.086],
        tols,
    )
    # Continue for one more time step. This time, the north displacement is 1, -1, 2.
    setup.time_manager = pp.TimeManager([1.0, 2.0], 1.0, True)
    params["prepare_simulation"] = False

    # Fixed values from a previous run. Both normal value (u_y=0) and ratio of
    # tangential displacements (1/50, see BC class) are correct.
    u_e = np.array([0.50754939, 0.0, 25.37756547])
    u_p = [0.40916401, 0.0, 20.45818325]
    traction = u_e * 1e-1
    traction[1] = -2.54
    # Same goes here. We expect -0.75, since the top coordinate is 0.75.
    u_top = [0.97917835, -0.75, 48.95893718]
    pp.run_time_dependent_model(setup, params)
    verify_elastoplastic_deformation(
        setup,
        u_e,
        u_p,
        u_top,
        traction,
        tols,
        compare_means=True,
    )
