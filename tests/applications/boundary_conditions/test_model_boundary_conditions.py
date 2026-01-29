"""This file tests:
- HydrostaticBoundaryPressureValues
- LithostaticBoundaryStressValues
- ThermalGradientBoundaryTemperatureValues

"""

from typing import Callable, Type

import numpy as np
import pytest
from pytest import param

import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    HydrostaticBoundaryPressureValues,
    LithostaticBoundaryStressValues,
    ThermalGradientBoundaryTemperatureValues,
)
from porepy.applications.md_grids.model_geometries import OrthogonalFractures3d


@pytest.mark.parametrize(
    "params",
    [
        param(
            {
                "mixin": HydrostaticBoundaryPressureValues,
                "method_to_evaluate": lambda model, bg: model.bc_values_pressure(bg),
            },
            id="HydrostaticBoundaryPressureValues",
        ),
        param(
            {
                "mixin": ThermalGradientBoundaryTemperatureValues,
                "method_to_evaluate": lambda model, bg: model.bc_values_temperature(bg),
            },
            id="ThermalGradientBoundaryTemperatureValues",
        ),
    ],
)
def test_gradient_scalar_boundary_values(params):
    mixin_type: Type[pp.PorePyModel] = params["mixin"]
    method_to_evaluate: Callable = params["method_to_evaluate"]

    class TestedModel(
        mixin_type,
        OrthogonalFractures3d,
        pp.MassAndEnergyBalance,
    ):
        pass

    tested_model = TestedModel()
    tested_model.prepare_simulation()

    for boundary_grid in tested_model.mdg.boundaries():
        values = method_to_evaluate(tested_model, boundary_grid)

        sides = tested_model.domain_boundary_sides(boundary_grid)

        max_value = values[sides.bottom][0]
        min_value = values[sides.top][0]
        # We test that the boundary values are:
        #      top     min val
        #            |--------|
        #            |        | min <= data <= max
        #            |--------|
        #   bottom     max val
        assert min_value < max_value
        # Min and max values should be the same in each cell of the side.
        np.testing.assert_array_equal(values[sides.bottom], max_value)
        np.testing.assert_array_equal(values[sides.top], min_value)

        # Other sides should contain values within the interval.
        other_sides = sides.east | sides.west | sides.north | sides.south
        assert np.all(values[other_sides] <= max_value)
        assert np.all(values[other_sides] >= min_value)


def test_lithostatic_boundary_stress_values():
    class TestedModel(
        LithostaticBoundaryStressValues,
        OrthogonalFractures3d,
        pp.MomentumBalance,
    ):
        pass

    tested_model = TestedModel()
    tested_model.prepare_simulation()

    # Lithostatic boundary condition requires non-zero time.
    tested_model.time_manager.time = 1

    for boundary_grid in tested_model.mdg.boundaries():
        values = tested_model.bc_values_stress(boundary_grid)
        sides = tested_model.domain_boundary_sides(boundary_grid)

        # Expanding the indices to reflect vector data, 3 DoFs per cell.
        bottom = np.repeat(sides.bottom, 3)
        top = np.repeat(sides.top, 3)
        west = np.repeat(sides.west, 3)
        east = np.repeat(sides.east, 3)
        north = np.repeat(sides.north, 3)
        south = np.repeat(sides.south, 3)

        # Shear stresses must be zeros.
        np.testing.assert_array_equal(values[bottom | top][0::3], 0)
        np.testing.assert_array_equal(values[bottom | top][1::3], 0)
        np.testing.assert_array_equal(values[east | west][1::3], 0)
        np.testing.assert_array_equal(values[east | west][2::3], 0)
        np.testing.assert_array_equal(values[north | south][0::3], 0)
        np.testing.assert_array_equal(values[north | south][2::3], 0)

        # Normal stresses.
        max_value = values[bottom][2::3].max()
        min_value = values[top][2::3].min()
        assert min_value < max_value

        # For this geometry, east and west sides contain no cells for the fracture
        # boundary.
        east_value = values[east][0::3].mean() if np.any(east) else 0
        west_value = values[west][0::3].mean() if np.any(west) else 0
        north_value = values[north][1::3].mean()
        south_value = values[south][1::3].mean()

        # All mean values of the sides are within the bounds.
        assert min_value <= abs(east_value) < max_value
        assert min_value <= abs(west_value) < max_value
        assert min_value <= abs(north_value) < max_value
        assert min_value <= abs(south_value) < max_value

        # Forces on opposite sides should have opposite sign.
        assert east_value <= 0
        assert west_value >= 0
        assert north_value <= 0
        assert south_value >= 0

        # Forces on opposite sides should equilibrate each other, the domain is static.
        np.testing.assert_almost_equal(east_value + west_value, 0)
        np.testing.assert_almost_equal(north_value + south_value, 0)
