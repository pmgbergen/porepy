"""Integration tests for the fracture damage model.

These tests run full time-dependent simulations with displacement-controlled boundary
conditions and verify qualitative behaviour: monotonicity of damage history, correct
response to load reversal, and expected damage lengths. They take roughly a minute each
and are intended as a complement to the unit tests.
"""

import copy
from typing import Any, Sequence, cast

import numpy as np
import pytest

import porepy as pp
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.examples import fracture_damage as damage_example


def _assert_damage_values_equal_between_steps(
    vals: Sequence[object],
    names: Sequence[str],
    from_idx: int,
    to_idx: int,
    from_time: int,
    to_time: int,
) -> None:
    """Assert equal values of damage fields for all names between two steps."""
    for name in names:
        val_from = cast(np.ndarray, getattr(vals[from_idx], "approx_" + name))
        val_to = cast(np.ndarray, getattr(vals[to_idx], "approx_" + name))
        np.testing.assert_allclose(
            val_from,
            val_to,
            atol=1e-8,
            err_msg=f"Mismatch between damage values at t={from_time} and t={to_time}"
            f" for {name}",
        )


def _assert_increment_damage_length_zero(length_1: np.ndarray, step: int) -> None:
    """Assert that a given displacement increment gives zero damage length."""
    np.testing.assert_allclose(
        length_1,
        0.0,
        atol=1e-8,
        err_msg=f"Damage length after {step} step not close to zero: {length_1}.",
    )


# Turn off shear dilation to better control the test. Although physically nonsensical,
# dilation damage with zero dilation angle is mathematically well-defined.
_solid_paramms = damage_example.solid_params.copy()
_solid_paramms.update({"dilation_angle": 0.0})
material_constants = {"solid": FractureDamageSolidConstants(**_solid_paramms)}  # type: ignore[arg-type]


def run_displacement_controlled_setup(
    isotropic: bool,
    dim: int,
    damages: Sequence[str],
    custom_model_params: dict[str, object] | None = None,
    custom_solver_params: dict[str, object] | None = None,
) -> tuple[list[Any], Any]:
    """Build and run the fracture damage setup used in these integration tests.

    Any custom parameters are used to update the default model or solver parameters.
    """
    model_class, params, solver_params = (
        damage_example.create_displacement_controlled_setup(
            isotropic=isotropic,
            dim=dim,
            damages=damages,
        )
    )
    # Copy to avoid messing with entries of the custom paramameters.
    if custom_model_params is not None:
        params.update(copy.deepcopy(custom_model_params))
    if custom_solver_params is not None:
        solver_params.update(copy.deepcopy(custom_solver_params))

    model = model_class(params)
    pp.ModelRunner(
        model,
        nonlinear_solver=pp.solvers.ConstraintLineSearchNonlinearSolver(solver_params),
    ).run()
    return model.results, model


@pytest.mark.parametrize("dim", [2, 3])
def test_isotropic_damage(dim: int):
    """Run one time step with both dilation and friction and verify against physically
    sensible/intuitive expectations.

    We have Dirichlet conditions as driving force, with values as per
    north_displacements in the parameter dictionary. The intuition for each assertion is
    given below.

    The error computed using the exact solution is not tested here, since the exact
    solution is based on the assumption of a known normal traction on the north
    boundary.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
        dim: Spatial dimension of the bulk domain (2 or 3).

    Raises:
        AssertionError: If the results do not match the exact solution.
    """
    damages = ["dilation", "friction"]
    # Model instance may be useful for debugging.
    vals, model = run_displacement_controlled_setup(
        True,
        dim,
        damages,
        custom_model_params={
            "material_constants": material_constants,
            "times_to_export": [],  # Suppress export of data for testing.
        },
    )
    # Note on counting time steps: The initial time step is not included in the results.
    # Thus, vals[0] corresponds to t=1 in the time manager, and vals[1]-vals[0] is
    # referred to as "first increment" below.
    for damage in damages:
        # Test both damage and damage history variable.
        names = [damage + "_damage_state", "damage_history"]
        # I) First two displacement jumps are identical, yielding identical damage
        # values after first two steps.
        _assert_damage_values_equal_between_steps(vals, names, 0, 1, 1, 2)

        # II) Third displacement jump is the negative of the second. For isotropic
        # damage length, this leads to a further decrease in...
        # i) damage state/history.
        name = names[0]
        val1 = cast(np.ndarray, getattr(vals[1], "approx_" + name))
        val2 = cast(np.ndarray, getattr(vals[2], "approx_" + name))
        assert np.all(val2 < val1 - 1e-2), (
            f"Damage did not decrease for {name} at t=3: {val2}/{val1}."
        )
        # Due to exponential decay, the decrease in damage is smaller than a factor 2.
        assert np.all(val2 * 2 > val1), (
            f"Damage decrease too large for {name} at t=3: {val2}/{val1}."
        )
        # ii) ... corresponding to a threefold increase in damage history, since
        # cumulative length 3 = length 2 + (length 2 - (-length 2) ), where the
        # parenthesis is the last step.
        name = names[1]
        val1 = cast(np.ndarray, getattr(vals[1], "approx_" + name))
        val2 = cast(np.ndarray, getattr(vals[2], "approx_" + name))
        np.testing.assert_allclose(
            val2,
            3 * val1,
            rtol=2e-4,  # Allow some tolerance due to numerical errors.
            err_msg=f"Damage history mismatch for {name} at t=3: {val2}/{val1}.",
        )

        # III) Fourth displacement jump and corresponding increment are nonzero in the
        # tangential direction, but the normal displacement is large enough to open the
        # fracture. Thus, we expect no additional damage.
        _assert_damage_values_equal_between_steps(vals, names, 2, 3, 3, 4)
    # Test damage lengths. Each value is the contribution from that increment.
    length_0 = vals[0].approx_damage_length
    length_1 = vals[1].approx_damage_length
    length_2 = vals[2].approx_damage_length
    length_3 = vals[3].approx_damage_length
    # 1) The second displacement increment is zero.
    _assert_increment_damage_length_zero(length_1, 2)

    # 2) Isotropic damage length depends on the magnitude, not the direction, of the
    # displacement jump. Since the first and third jump are in opposite directions, the
    # third damage length should be twice the first.
    np.testing.assert_allclose(
        length_0 * 2,
        length_2,
        rtol=1e-3,  # Allow some tolerance due to numerical errors.
        err_msg=f"Damage length mismatch after first and third step: "
        f"{length_0}/{length_2}.",
    )

    # 3) After fourth step, damage *length* (as opposed to damage above) should be
    # positive due to nonzero tangential increment, even though damage does not increase
    # due to the fracture being open.
    # The prescribed tangential boundary displacement is only approached, not attained,
    # part of the tangential motion being taken up elastically. That share is linear in
    # the friction coefficient -- a relative shortfall of 0.31 * mu_b to four digits, so
    # 0.31% at the fixture's 0.01. The tolerance leaves half again as much room, tight
    # enough that raising the fixture's friction much above that fails here rather than
    # passing silently.
    expected_3 = 1e-4 if dim == 2 else np.sqrt(5) * 1e-4
    np.testing.assert_allclose(
        length_3,
        expected_3,
        rtol=5e-3,
        err_msg=f"Damage length is wrong after fourth step: {length_3}",
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_anisotropic_damage(dim: int):
    """Run one time step with both dilation and friction and verify against physically
    sensible/intuitive expectations.

    We have Dirichlet conditions as driving force, with values as per
    north_displacements in the parameter dictionary. The intuition for each assertion is
    given below.

    The error computed using the exact solution is not tested here, since the exact
    solution is based on the assumption of a known normal traction on the north
    boundary.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
        dim: Spatial dimension of the bulk domain (2 or 3).

    Raises:
        AssertionError: If the results do not match the exact solution.
    """
    damages = ["dilation", "friction"]

    # Model instance may be useful for debugging.
    vals, model = run_displacement_controlled_setup(
        False,
        dim,
        damages,
        custom_model_params={
            "material_constants": material_constants,
            "times_to_export": [],
        },  # Suppress export of data for testing.
    )
    # Note on counting time steps: The initial time step is not included in the results.
    # Thus, vals[0] corresponds to t=1 in the time manager, and vals[1]-vals[0] is
    # referred to as "first increment" below.
    for damage in damages:
        names = [damage + "_damage_state", "damage_history"]
        # I) First two displacement jumps are identical, yielding identical damage
        # values after first two steps.
        _assert_damage_values_equal_between_steps(vals, names, 0, 1, 1, 2)

        # II) Third displacement jump is the negative of the second. For anisotropic
        # damage length, this yields identical damage values after third step.
        for name in names:
            val1 = cast(np.ndarray, getattr(vals[1], "approx_" + name))
            val2 = cast(np.ndarray, getattr(vals[2], "approx_" + name))
            np.testing.assert_allclose(
                val1,
                val2,
                atol=1e-8,
                err_msg=f"Mismatch between damage values at t=2 and t=3 for {name}",
            )

        # III) Fourth displacement jump and increment are nonzero in the tangential
        # direction, but the normal displacement is large enough to open the fracture.
        # Thus, we expect no additional damage.
        _assert_damage_values_equal_between_steps(vals, names, 2, 3, 3, 4)
    # Test damage lengths. Each value is the contribution from that increment.
    length_0 = vals[0].approx_damage_length
    length_1 = vals[1].approx_damage_length
    length_2 = vals[2].approx_damage_length
    length_3 = vals[3].approx_damage_length
    # 1) The second displacement increment is zero.
    _assert_increment_damage_length_zero(length_1, 2)

    # 2) Anisotropic damage length depends on the direction of the displacement jump.
    # Since the first and third jump are in opposite directions and only the part of the
    # increment aligned with the current jump contributes, the damage lengths
    # should be equal.
    np.testing.assert_allclose(
        length_0,
        length_2,
        rtol=1e-8,
        err_msg=f"Damage length mismatch after first and third step: "
        f"{length_0}/{length_2}.",
    )

    # 3) After fourth step, damage *length* (as opposed to damage above) should increase
    # due to nonzero tangential increment, even though damage does not increase due to
    # the fracture being open.
    # The prescribed tangential boundary displacement is only approached, not attained,
    # part of the tangential motion being taken up elastically. That share is linear in
    # the friction coefficient -- a relative shortfall of 0.31 * mu_b to four digits, so
    # 0.31% at the fixture's 0.01. The tolerance leaves half again as much room, tight
    # enough that raising the fixture's friction much above that fails here rather than
    # passing silently.
    expected_3 = 1e-4 if dim == 2 else np.sqrt(1 / 2) * 1e-4
    np.testing.assert_allclose(
        length_3,
        expected_3,
        rtol=5e-3,
        err_msg=f"Damage length is wrong after fourth step: {length_3}",
    )
