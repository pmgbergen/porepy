"""Tests for contact mechanics."""

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import ContactMechanicsTester, add_mixin
from porepy.models.contact_mechanics import (
    RadialReturnTangentialContactMechanicsEquation,
)

grid_classes = {2: SquareDomainOrthogonalFractures, 3: CubeDomainOrthogonalFractures}

# Define the two formulation variants
tester_classes = {
    "standard": ContactMechanicsTester,
    "radial_return": add_mixin(
        RadialReturnTangentialContactMechanicsEquation, ContactMechanicsTester
    ),
}


@pytest.mark.parametrize("nd", list(grid_classes.keys()))
@pytest.mark.parametrize("formulation", list(tester_classes.keys()))
def test_contact_mechanics(nd, formulation):
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
    model_class = add_mixin(grid_classes[nd], tester_classes[formulation])
    model: pp.PorePyModel = model_class(params)
    pp.ModelRunner(model).run()
    fractures = model.mdg.subdomains(dim=nd - 1)
    # Get displacement jump in global coordinates.
    displacement_jump_global = model.equation_system.evaluate(
        model.local_coordinates(fractures).transpose()
        @ model.displacement_jump(fractures)
    ).reshape((nd, -1), order="F")
    displacement_jump_local = model.equation_system.evaluate(
        model.displacement_jump(fractures)
    ).reshape((nd, -1), order="F")
    # Check if the positive side is the first side of the fracture. Remember, the jump
    # is defined as u_2 - u_1, where u_1 is the displacement of the first side of the
    # fracture. If the negative side is the first side, the jump will be the negative of
    # the applied displacement.
    direction_vec = np.array([0.0, 1.0, 0.0])
    _, _, positive_side_first = pp.sides_of_fracture(
        model.mdg.interfaces()[0], model.mdg.subdomains()[0], direction_vec
    )
    # Check that the jump is equal to the applied displacement.
    expected_jump = displacement_vals[:, 1].reshape((nd, -1))
    if not positive_side_first:
        expected_jump *= -1
    np.testing.assert_allclose(displacement_jump_global - expected_jump, 0, atol=1e-12)
    # Check the contact traction.
    scalar_to_nd = pp.ad.sum_projection_list(model.basis(fractures, dim=nd))
    scaled_traction = (
        scalar_to_nd @ model.characteristic_contact_traction(fractures)
    ) * model.contact_traction(fractures)
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
    np.testing.assert_allclose(traction[-1] - sigma_n, 0, atol=1e-15)
    # In the tangential direction, we have
    # \sigma_t = k_t * \Delta u_t
    # Note that in 3d, we test both the nonzero and zero displacement jumps.
    inds_t = np.arange(nd - 1)
    sigma_t = solid.fracture_tangential_stiffness * displacement_jump_local[inds_t]
    np.testing.assert_allclose(traction[inds_t] - sigma_t, 0, atol=1e-15)


@pytest.mark.parametrize("nd", list(grid_classes.keys()))
@pytest.mark.parametrize("formulation", list(tester_classes.keys()))
@pytest.mark.parametrize("friction_coefficient", [0.0, 0.5])
def test_friction_constraint(nd, formulation, friction_coefficient):
    """Test whether friction constraint ||t_t|| <= b_p (friction bound) is respected."""

    solid_vals = {
        "fracture_tangential_stiffness": 0.5e0,
        "fracture_normal_stiffness": 1.0e0,
        "maximum_elastic_fracture_opening": 2.0,
        "friction_coefficient": friction_coefficient,
    }
    solid = pp.SolidConstants(**solid_vals)

    # Large displacement to trigger sliding.
    displacement_vals = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]]).T[:nd]
    params = {
        "times_to_export": [],
        "fracture_indices": [1],
        "cartesian": True,
        "material_constants": {"solid": solid},
        "interface_displacement_parameter_values": displacement_vals,
    }

    model_class = add_mixin(grid_classes[nd], tester_classes[formulation])
    model: pp.PorePyModel = model_class(params)
    pp.ModelRunner(model).run()

    fractures = model.mdg.subdomains(dim=nd - 1)

    # Evaluate tangential traction and friction bound.
    nd_vec_to_tangential = model.tangential_component(fractures)
    t_t = nd_vec_to_tangential @ model.contact_traction(fractures)

    t_t_vals = model.equation_system.evaluate(t_t).reshape((nd - 1, -1), order="F")
    norm_t_t = np.linalg.norm(t_t_vals, axis=0)

    # Evaluate friction bound.
    friction_bound = model.equation_system.evaluate(
        model.friction_bound(fractures)
    ).ravel()

    # Should enfore ||t_t|| <= b_p (with small tolerance).
    tol = 1e-10
    assert np.all(norm_t_t <= friction_bound + tol), (
        f"Friction constraint violated: ||t_t|| = {norm_t_t.max()}, "
        f"b_p = {friction_bound.max()}"
    )


@pytest.mark.parametrize("nd", list(grid_classes.keys()))
@pytest.mark.parametrize("formulation", list(tester_classes.keys()))
def test_contact_mechanics_convergence(nd, formulation):
    """Check successful convergence behavior."""
    solid_vals = {
        "fracture_tangential_stiffness": 1.0e0,
        "fracture_normal_stiffness": 1.0e0,
        "maximum_elastic_fracture_opening": 2.0,
    }
    solid = pp.SolidConstants(**solid_vals)

    displacement_vals = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]).T[:nd]

    params = {
        "times_to_export": [],
        "fracture_indices": [1],
        "cartesian": True,
        "material_constants": {"solid": solid},
        "interface_displacement_parameter_values": displacement_vals,
    }

    model_class = add_mixin(grid_classes[nd], tester_classes[formulation])
    model = model_class(params)

    # Run the simulation.
    status = pp.ModelRunner(model).run()

    assert status.is_success()
