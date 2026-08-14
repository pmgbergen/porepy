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
    BoundaryConditionsMechanicsNeumann,
    HydrostaticBoundaryPressureValues,
    LithostaticBoundaryStressValues,
    ThermalGradientBoundaryTemperatureValues,
)
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)


def _geometry_mixin(model_dim: int):
    if model_dim == 3:
        return CubeDomainOrthogonalFractures
    elif model_dim == 2:
        return SquareDomainOrthogonalFractures
    else:
        raise ValueError(model_dim)


@pytest.fixture(params=[2, 3], ids=["2d", "3d"])
def momentum_model(request):
    model_dim = request.param
    geometry_mixin_type = _geometry_mixin(model_dim)

    class TestedModel(
        BoundaryConditionsMechanicsNeumann,
        LithostaticBoundaryStressValues,
        geometry_mixin_type,
        pp.MomentumBalance,
    ):
        pass

    model = TestedModel()
    model.prepare_simulation()
    return model


@pytest.fixture
def momentum_boundary_matrix_grid(momentum_model):
    matrix_grids = [
        sd for sd in momentum_model.mdg.subdomains() if sd.dim == momentum_model.nd
    ]
    assert len(matrix_grids) == 1
    return matrix_grids[0]


@pytest.mark.parametrize("model_dim", [2, 3])
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
def test_gradient_scalar_boundary_values(params, model_dim: int):
    mixin_type: Type[pp.PorePyModel] = params["mixin"]
    method_to_evaluate: Callable = params["method_to_evaluate"]

    if model_dim == 3:
        geometry_mixin_type = CubeDomainOrthogonalFractures
    elif model_dim == 2:
        geometry_mixin_type = SquareDomainOrthogonalFractures
    else:
        raise ValueError(model_dim)

    class TestedModel(
        mixin_type,
        geometry_mixin_type,
        pp.MassAndEnergyBalance,
    ):
        pass

    tested_model = TestedModel()
    thermal_gradient = 0.03
    surface_temperature = 293.15
    tested_model.params.update(
        {
            "thermal_gradient": thermal_gradient,
            "surface_temperature": surface_temperature,
        }
    )
    tested_model.prepare_simulation()

    for boundary_grid in tested_model.mdg.boundaries():
        values = method_to_evaluate(tested_model, boundary_grid)

        sides = tested_model.domain_boundary_sides(boundary_grid)

        if model_dim == 3:
            max_value_side = sides.bottom
            min_value_side = sides.top
            other_sides = sides.east | sides.west | sides.north | sides.south
            vertical_index = 2
        elif model_dim == 2:
            max_value_side = sides.south
            min_value_side = sides.north
            other_sides = sides.east | sides.west
            vertical_index = 1

        # Explicitly compute the expected values at the boundary based on the
        # depth of the cell centers.
        domain_key = "zmax" if model_dim == 3 else "ymax"
        depth_bc = (
            tested_model.domain.bounding_box[domain_key]
            - boundary_grid.cell_centers[vertical_index]
        )

        if isinstance(tested_model, HydrostaticBoundaryPressureValues):
            # The expected values are computed as: P = rho * g * h + P_atm.
            rho = tested_model.fluid.reference_component.density
            expected_values = (
                pp.GRAVITY_ACCELERATION * rho * depth_bc + pp.ATMOSPHERIC_PRESSURE
            )
        elif isinstance(tested_model, ThermalGradientBoundaryTemperatureValues):
            # The expected values are computed as: T = T_{surface} + G * h.
            expected_values = surface_temperature + thermal_gradient * depth_bc

        max_value = expected_values[max_value_side][0]
        min_value = expected_values[min_value_side][0]

        # We test that the boundary values are:
        #      top     min val
        #            |--------|
        #            |        | min <= data <= max
        #            |--------|
        #   bottom     max val
        assert min_value < max_value
        # Min and max values should be the same in each cell of the side.
        np.testing.assert_array_equal(values[max_value_side], max_value)
        np.testing.assert_array_equal(values[min_value_side], min_value)

        # Other sides should contain values within the interval.
        assert np.all(values[other_sides] <= max_value)
        assert np.all(values[other_sides] >= min_value)

        # Check the values vary linearly in depth
        z_cell_centers = boundary_grid.cell_centers[vertical_index]
        z_top = np.mean(z_cell_centers[min_value_side])
        z_bottom = np.mean(z_cell_centers[max_value_side])

        slope = (max_value - min_value) / (z_bottom - z_top)
        intercept = min_value - slope * z_top
        expected_values = slope * z_cell_centers + intercept

        np.testing.assert_allclose(
            values[other_sides],
            expected_values[other_sides],
            rtol=1e-10,
            atol=1e-10,
        )


def test_lithostatic_boundary_stress_values(momentum_model: pp.PorePyModel):
    # Scaling of the lithostatic stress.
    stress_multipliers = np.array([1, 2, 0.1])
    model = momentum_model
    model.params.update({"lithostatic_stress_multipliers": stress_multipliers})

    # Lithostatic boundary condition requires non-zero time.
    model.time_data.time = 1

    nd = model.nd

    for boundary_grid in model.mdg.boundaries():
        values = model.bc_values_stress(boundary_grid)
        sides = model.domain_boundary_sides(boundary_grid)

        # Expanding the indices to reflect vector data, `nd` DoFs per cell.
        bottom = np.repeat(sides.bottom, nd)
        top = np.repeat(sides.top, nd)
        west = np.repeat(sides.west, nd)
        east = np.repeat(sides.east, nd)
        north = np.repeat(sides.north, nd)
        south = np.repeat(sides.south, nd)

        # Shear stresses must be zeros.
        np.testing.assert_array_equal(values[east | west][1::nd], 0)
        np.testing.assert_array_equal(values[north | south][0::nd], 0)
        if model.nd == 3:
            np.testing.assert_array_equal(values[bottom | top][0::nd], 0)
            np.testing.assert_array_equal(values[bottom | top][1::nd], 0)
            np.testing.assert_array_equal(values[east | west][2::nd], 0)
            np.testing.assert_array_equal(values[north | south][2::nd], 0)

        vertical_index = 2 if model.nd == 3 else 1

        # Explicitly compute expected normal stresses in each direction.
        domain_key = "zmax" if model.nd == 3 else "ymax"
        depth_bc = (
            model.domain.bounding_box[domain_key]
            - boundary_grid.cell_centers[vertical_index]
        )
        rho_f = model.fluid.reference_component.density
        rho_s = model.solid.density
        phi = model.solid.porosity
        rho_eff = rho_s * (1 - phi) + rho_f * phi
        gravity = rho_eff * pp.GRAVITY_ACCELERATION

        expected_stress = [
            stress_multipliers[i] * gravity * depth_bc * boundary_grid.cell_volumes
            for i in range(3)
        ]

        # Values from the model should be equal to explicitly expected results.
        expected_east = -expected_stress[0][sides.east]
        expected_west = +expected_stress[0][sides.east]
        np.testing.assert_allclose(values[east][0::nd], expected_east)
        np.testing.assert_allclose(values[west][0::nd], expected_west)
        if model.nd == 3:
            expected_north = -expected_stress[1][sides.north]
            expected_south = +expected_stress[1][sides.south]
            np.testing.assert_allclose(values[north][1::nd], expected_north)
            np.testing.assert_allclose(values[south][1::nd], expected_south)

        # For this geometry, some sides contain no cells for the fracture boundary.
        east_value = values[east].mean() if np.any(east) else 0.0
        west_value = values[west].mean() if np.any(west) else 0.0

        # Forces on opposite sides should equilibrate each other, the domain is static.
        np.testing.assert_almost_equal(east_value + west_value, 0)
        if model.nd == 3:
            north_value = float(values[north].mean())
            south_value = float(values[south].mean())
            np.testing.assert_almost_equal(north_value + south_value, 0)


def test_mechanics_bcs_neumann(
    momentum_model,
    momentum_boundary_matrix_grid,
):
    """Test anchor ordering and component constraints for Neumann mechanics BCs.

    The anchor faces are ordered west-south-east in 2D and 3D. The in-plane
    constraints are identical across dimensions, and 3D additionally fixes z.

    """
    sd = momentum_boundary_matrix_grid
    sides = momentum_model.domain_boundary_sides(sd)
    bc = momentum_model.bc_type_mechanics(sd)
    faces = momentum_model.faces_to_fix(sd)

    assert len(faces) == 3
    assert np.any(bc.is_dir)
    assert not np.all(bc.is_dir)

    # Anchors are ordered west, south, east.
    assert sides.west[faces[0]]
    assert sides.south[faces[1]]
    assert sides.east[faces[2]]

    west_mask = bc.is_dir[:, faces[0]]
    south_mask = bc.is_dir[:, faces[1]]
    east_mask = bc.is_dir[:, faces[2]]

    # In-plane constraints are shared in 2D and 3D.
    np.testing.assert_array_equal(west_mask[:2], np.array([False, True]))
    np.testing.assert_array_equal(south_mask[:2], np.array([True, False]))
    np.testing.assert_array_equal(east_mask[:2], np.array([False, True]))

    if momentum_model.nd == 3:
        assert west_mask[2]
        assert south_mask[2]
        assert east_mask[2]

    for face in faces:
        assert np.array_equal(bc.is_neu[:, face], ~bc.is_dir[:, face])
