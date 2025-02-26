"""Tests for contact mechanics."""

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import (
    ContactMechanicsTester,
    _add_mixin,
)

grid_classes = {2: SquareDomainOrthogonalFractures, 3: CubeDomainOrthogonalFractures}


@pytest.mark.parametrize("nd", list(grid_classes.keys()))
def test_contact_mechanics(nd):
    solid = pp.SolidConstants(**pp.solid_values.extended_granite_values_for_testing)
    solid_vals = {
        "fracture_tangential_stiffness": 1.0e0,  # [Pa m^-1]
        "fracture_normal_stiffness": 1.0e0,  # [Pa m^-1]
        "maximum_elastic_fracture_opening": 2.0,  # [m]
    }
    solid = pp.SolidConstants(**solid_vals)

    # One 3d vector for each time step. The vector is applied to the "top" interface.
    displacement_vals = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]).T[:nd]
    params = {
        "times_to_export": [],  # Suppress output for tests
        "fracture_indices": [1],
        "cartesian": True,
        "material_constants": {"solid": solid},
        "interface_displacement_parameter_values": displacement_vals,
    }
    model_class = _add_mixin(grid_classes[nd], ContactMechanicsTester)
    model = model_class(params)
    pp.run_time_dependent_model(model)
    fractures = model.mdg.subdomains(dim=nd - 1)
    # Get displacement jump in global coordinates.
    displacement_jump_global = model.equation_system.evaluate(
        model.local_coordinates(fractures).transpose()
        @ model.displacement_jump(fractures)
    ).reshape((nd, -1), order="F")
    displacement_jump_local = model.equation_system.evaluate(
        model.displacement_jump(fractures)
    ).reshape((nd, -1), order="F")
    # Check if the top side is the first side of the fracture. Remember, the jump is
    # defined as u_2 - u_1, where u_1 is the displacement of the first side of the
    # fracture. If the top side is the first side, the jump will be the negative of the
    direction_vec = np.array([0.0, 1.0, 0.0])
    _, _, top_side_first = pp.sides_of_fracture(
        model.mdg.interfaces()[0], model.mdg.subdomains()[0], direction_vec
    )
    # Check that the jump is equal to the applied displacement.
    expected_jump = displacement_vals[:, 1].reshape((nd, -1))
    if top_side_first:
        expected_jump *= -1
    np.testing.assert_allclose(displacement_jump_global - expected_jump, 0)
    # Check the contact traction.
    scaled_traction = model.contact_traction(
        fractures
    ) * model.characteristic_contact_traction(fractures)
    traction = model.equation_system.evaluate(scaled_traction).reshape(
        (nd, -1), order="F"
    )
    # In the normal direction, we have according to the Barton-Bandis model
    # \Delta u_n = \Delta u_n^{max}
    #         + \frac{\Delta u_n^{max} \sigma_n}{\Delta u_n^{max} K_n - \sigma_n}
    # Solving for \sigma_n gives
    # \sigma_n = K_n \Delta u_n^{max} * (1 - \Delta u_n^{max} / \Delta u_n)
    k_n = solid.fracture_normal_stiffness
    u_n_max = solid.maximum_elastic_fracture_opening
    # Use local coordinates since the inverted relation is defined in local coordinates.
    u_n = displacement_jump_local[-1]
    sigma_n = k_n * u_n_max * (1 - u_n_max / u_n)
    # Check that the traction is equal to the calculated value.
    np.testing.assert_allclose(traction[-1] - sigma_n, 0)
    # In the tangential direction, we have
    # \sigma_t = k_t * \Delta u_t
    # Note that in 3d, we test both the nonzero and zero displacement jumps.
    inds_t = np.arange(nd - 1)
    sigma_t = solid.fracture_tangential_stiffness * displacement_jump_local[inds_t]
    np.testing.assert_allclose(traction[inds_t] - sigma_t, 0)
