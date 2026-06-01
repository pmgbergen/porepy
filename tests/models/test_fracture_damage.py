"""Integration tests for the fracture damage model.

These tests run full time-dependent simulations with displacement-controlled boundary
conditions and verify qualitative behaviour: monotonicity of damage history, correct
response to load reversal, and expected damage lengths. Tests are marked skipped due
to their runtime (~1 min each) and are intended as a complement to the unit tests.
"""

import copy
from typing import Sequence, cast

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import add_mixin
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.examples import fracture_damage as damage_examples
from porepy.models import fracture_damage as damage_models
from porepy.numerics.nonlinear.line_search import ConstraintLineSearchNonlinearSolver


def run_displacement_controlled_setup(
    isotropic: bool,
    dim: int,
    damages: Sequence[str],
):
    """Run a time-dependent simulation with displacement-controlled BCs.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
        dim: Spatial dimension of the bulk domain (2 or 3).
        damages: Iterable of damage types to include (subset of
            {"dilation", "friction"}).

    Returns:
        A list of VerificationDataSaving instances containing results for each time
        step.
    """
    params = copy.deepcopy(damage_examples.model_params)
    model_class = damage_examples.FractureDamageMomentumBalance

    # Choose damage length (isotropic vs anisotropic) and exact solution.
    if isotropic:
        params["exact_solution"] = damage_examples.ExactSolutionIsotropic
        model_class = add_mixin(
            damage_models.IsotropicFractureDamageLength, model_class
        )
    else:
        params["exact_solution"] = damage_examples.ExactSolutionAnisotropic
        model_class = add_mixin(
            damage_models.AnisotropicFractureDamageLength, model_class
        )

    # Add requested damage equations and variables.
    for name in damages:
        model_class = add_mixin(damage_examples.damage_types[name], model_class)

    # Add geometry mixin for the target dimension.
    geom = (
        SquareDomainOrthogonalFractures if dim == 2 else CubeDomainOrthogonalFractures
    )
    model_class = add_mixin(geom, model_class)

    # 5 time instants => 4 forward steps.
    params.update(
        {
            "time_manager": pp.TimeManager(np.arange(0, 5), 1, True),
            "north_displacements": params["north_displacements"][:dim],
        }
    )
    params["material_constants"] = {
        "solid": FractureDamageSolidConstants(**damage_examples.solid_params),  # type: ignore[arg-type]
    }
    solver_params = {
        "nl_max_iterations": 50,  # Hard nonlinear problems - expect slow convergence
        "nl_convergence_res_atol": 1e-8,
        "nl_convergence_inc_atol": 1e-8,
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "local_line_search": True,
    }

    # Set y component somewhat smaller than (fracture gap + maximum elastic opening).
    # This results in compression and sensible tractions.
    params["north_displacements"][1] = 0.98e-3
    # Modify north displacement BC to get open fracture (=> no additional damage) in
    # 4th time increment.
    params["north_displacements"][1, 4] = 3e-3
    # Turn off shear dilation to better control the test. Although physically
    # nonsensical, dilation damage with zero dilation angle is mathematically
    # well-defined.
    solid_params = damage_examples.solid_params.copy()
    solid_params["dilation_angle"] = 0.0
    # Similarly, simplify problem by removing elastic normal deformation.
    params["material_constants"]["solid"] = FractureDamageSolidConstants(**solid_params)  # type: ignore[arg-type]
    m = model_class(params)
    pp.ModelRunner(m, solver_params).run()

    return m.results, m


@pytest.mark.parametrize("dim", [2, 3])
# The tests take about a minute and are not critical, rather a supplement to test_damage
# @pytest.mark.skipped  # reason: slow
def test_isotropic_damage(dim: int):
    """Run one time step with both dilation and friction and verify against physically
    sensible/intuitive expectations.

    By disabling the Neumann normal stress boundary condition on the north side, we have
    Dirichlet conditions as driving force, with values as per north_displacements in the
    parameter dictionary. The intuition for each assertion is given below.

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
    vals, model = run_displacement_controlled_setup(True, dim, damages)
    # Note on counting time steps: The initial time step is not included in the results.
    # Thus, vals[0] corresponds to t=1 in the time manager, and vals[1]-vals[0] is
    # referred to as "first increment" below.
    for damage in damages:
        # Test both damage and damage history variable.
        names = [damage + "_damage_state", damage + "_damage_history"]
        # I) First two displacement jumps are identical, yielding identical damage
        # values after first two steps.
        for name in names:
            val0 = cast(np.ndarray, getattr(vals[0], "approx_" + name))
            val1 = cast(np.ndarray, getattr(vals[1], "approx_" + name))
            np.testing.assert_allclose(
                val0,
                val1,
                atol=1e-8,
                err_msg=f"Mismatch between damage values at t=1 and t=2 for {name}",
            )

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
        for name in names:
            val2 = cast(np.ndarray, getattr(vals[2], "approx_" + name))
            val3 = cast(np.ndarray, getattr(vals[3], "approx_" + name))
            np.testing.assert_allclose(
                val2,
                val3,
                atol=1e-8,
                err_msg=f"Mismatch between damage values at t=3 and t=4 for {name}.",
            )
    # Test damage lengths. Each value is the contribution from that increment.
    length_0 = vals[0].approx_damage_length
    length_1 = vals[1].approx_damage_length
    length_2 = vals[2].approx_damage_length
    length_3 = vals[3].approx_damage_length
    # 1) The second displacement increment is zero.
    np.testing.assert_allclose(
        length_1,
        0.0,
        atol=1e-8,
        err_msg=f"Damage length after second step not close to zero: {length_1}.",
    )

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
    expected_3 = 1e-4 if dim == 2 else np.sqrt(5) * 1e-4
    np.testing.assert_allclose(
        length_3,
        expected_3,
        rtol=3e-3,
        err_msg=f"Damage length is wrong after fourth step: {length_3}",
    )


@pytest.mark.parametrize("dim", [2, 3])
# The tests take about a minute and are not critical, rather a supplement to test_damage
# @pytest.mark.skipped  # reason: slow
def test_anisotropic_damage(dim: int):
    """Run one time step with both dilation and friction and verify against physically
    sensible/intuitive expectations.

    By disabling the Neumann normal stress boundary condition on the north side, we have
    Dirichlet conditions as driving force, with values as per north_displacements in the
    parameter dictionary. The intuition for each assertion is given below.

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
    vals, model = run_displacement_controlled_setup(False, dim, damages)
    # Note on counting time steps: The initial time step is not included in the results.
    # Thus, vals[0] corresponds to t=1 in the time manager, and vals[1]-vals[0] is
    # referred to as "first increment" below.
    for damage in damages:
        names = [damage + "_damage_state", damage + "_damage_history"]
        # I) First two displacement jumps are identical, yielding identical damage
        # values after first two steps.
        for name in names:
            val0 = cast(np.ndarray, getattr(vals[0], "approx_" + name))
            val1 = cast(np.ndarray, getattr(vals[1], "approx_" + name))
            np.testing.assert_allclose(
                val0,
                val1,
                atol=1e-8,
                err_msg=f"Mismatch between damage values at t=1 and t=2 for {name}",
            )

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
        for name in names:
            val2 = cast(np.ndarray, getattr(vals[2], "approx_" + name))
            val3 = cast(np.ndarray, getattr(vals[3], "approx_" + name))
            np.testing.assert_allclose(
                val2,
                val3,
                atol=1e-8,
                err_msg=f"Mismatch between damage values at t=3 and t=4 for {name}",
            )
    # Test damage lengths. Each value is the contribution from that increment.
    length_0 = vals[0].approx_damage_length
    length_1 = vals[1].approx_damage_length
    length_2 = vals[2].approx_damage_length
    length_3 = vals[3].approx_damage_length
    # 1) The second displacement increment is zero.
    np.testing.assert_allclose(
        length_1,
        0.0,
        atol=1e-8,
        err_msg=f"Damage length after second step not close to zero: {length_1}.",
    )

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
    expected_3 = 1e-4 if dim == 2 else np.sqrt(1 / 2) * 1e-4
    np.testing.assert_allclose(
        length_3,
        expected_3,
        rtol=3e-3,
        err_msg=f"Damage length is wrong after fourth step: {length_3}",
    )
