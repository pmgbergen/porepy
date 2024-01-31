"""Tests for model variables.

"""
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


class LinearModel(
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    MomentumBalance,
):
    pass


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({}, 0),
        ({}, -0.1),
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
        "suppress_export": True,  # Suppress output for tests
        "material_constants": {"solid": solid},
        "uy_north": north_displacement,
    }

    # Create model and run simulation
    setup = LinearModel(params)
    pp.run_time_dependent_model(setup, params)

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
            # Fracture cuts the domain in half, so the bottom half should be undispalced
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
        # Normal traction should be non-positive. Zero if north_displacement is zero.,
        if north_displacement < 0:
            assert np.all(traction[setup.nd - 1 :: setup.nd] <= 0)
        else:
            assert np.allclose(traction, 0)


@pytest.mark.parametrize("units", [{"m": 0.29, "kg": 0.31, "K": 4.1}])
@pytest.mark.parametrize("uy_north", [0.2, -0.1])
def test_unit_conversion(units, uy_north):
    """Test that solution is independent of units.

    Parameters:
        units (dict): Dictionary with keys as those in
            :class:`~pp.models.material_constants.MaterialConstants`.
        uy_north (float): Value of displacement on the north boundary.

    """

    params = {
        "suppress_export": True,  # Suppress output for tests
        "fracture_indices": [0, 1],
        "cartesian": True,
        "uy_north": uy_north,
        "max_iterations": 10,
    }
    reference_params = copy.deepcopy(params)

    # Create model and run simulation
    setup_0 = LinearModel(reference_params)
    pp.run_time_dependent_model(setup_0, reference_params)

    params["units"] = pp.Units(**units)
    setup_1 = LinearModel(params)

    pp.run_time_dependent_model(setup_1, params)
    variables = [
        setup_1.displacement_variable,
        setup_1.interface_displacement_variable,
        setup_1.contact_traction_variable,
    ]
    variable_units = ["m", "m", "Pa"]
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
    model = LithostaticModel({"dim": dim, "suppress_export": True})
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
