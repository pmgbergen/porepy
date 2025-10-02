import copy
from typing import Literal, Sequence, cast

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

geometry_mixins = {
    "2": SquareDomainOrthogonalFractures,
    "3": CubeDomainOrthogonalFractures,
}
damage_types = {
    "dilation": damage_examples.DilationDamageMomentumBalance,
    "friction": damage_examples.FrictionDamageMomentumBalance,
}

# Methods to compare will be provided later by the user. Example names could be
# ["friction_damage", "dilation_damage"]. Leave empty to skip comparisons.
METHODS_TO_COMPARE: list[str] = []


def setup(
    isotropic: bool,
    dim: int,
    damages: Sequence[str],
    time_steps: int,
):
    """Construct a model for a time-dependent run using shared setup logic.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
    dim: Spatial dimension of the bulk domain (2 or 3).
    damages: Iterable of damage types to include (subset of {"dilation", "friction"}).
    time_steps: Number of time steps to simulate.
        Number of time instants passed to the TimeManager (N); this yields N-1
        forward steps. For a single forward step, pass 2.

    Returns:
        A configured model instance ready to be run with pp.run_time_dependent_model.
    """
    params_local = copy.deepcopy(damage_examples.model_params)
    model_class = damage_examples.FractureDamageMomentumBalance

    # Choose damage length (isotropic vs anisotropic) and exact solution
    if isotropic:
        params_local["exact_solution"] = damage_examples.ExactSolutionIsotropic
        model_class = add_mixin(
            damage_models.IsotropicFractureDamageLength, model_class
        )
    else:
        params_local["exact_solution"] = damage_examples.ExactSolutionAnisotropic
        model_class = add_mixin(
            damage_models.AnisotropicFractureDamageLength, model_class
        )

    # Add requested damage equations and variables.
    for name in damages:
        model_class = add_mixin(damage_types[name], model_class)

    # Add geometry mixin for the target dimension.
    model_class = add_mixin(geometry_mixins[str(dim)], model_class)

    # Configure time manager: np.arange(0, time_steps) gives time_steps-1 steps
    params_local.update(
        {
            "time_manager": pp.TimeManager(np.arange(0, time_steps), 1, True),
            # Trim displacement BCs to requested dimension
            "north_displacements": params_local["north_displacements"][:dim],
            "adaptive_indicator_scaling": True,
        }
    )

    params_local["material_constants"] = {
        "solid": FractureDamageSolidConstants(**damage_examples.solid_params),  # type: ignore[arg-type]
    }
    solver_params = {
        "max_iterations": 50,  # Hard nonlinear problems - expect slow convergence
        "nl_convergence_tol_res": 1e-8,
        "nl_convergence_tol": 1e-8,
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "local_line_search": True,
    }

    return model_class, params_local, solver_params


def run_displacement_controlled_setup(
    isotropic: bool,
    dim: int,
    damages: Sequence[str],
):
    """Run a time-dependent simulation with shared setup logic.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
        dim: Spatial dimension of the bulk domain (2 or 3).
        damages: Iterable of damage types to include (subset of
            {"dilation", "friction"}).

    Returns:
        A list of VerificationDataSaving instances containing results for each time
        step.
    """
    cls, model_params, solver_params = setup(isotropic, dim, damages, 5)

    model_params.pop("north_stress", None)  # Disable stress BC for this test.
    # Set y component somewhat smaller than (fracture gap + maximum elastic opening).
    # This results in compression and sensible tractions.
    model_params["north_displacements"][1] = 1.8e-3
    # Modify north displacement BC to get open fracture (=> no additional damage) in
    # 4th time increment.
    model_params["north_displacements"][1, 4] = 3e-3
    # Turn off shear dilation to better control the test. Although physically
    # nonsensical, dilation damage with zero dilation angle is mathematically
    # well-defined.
    solid_params = damage_examples.solid_params.copy()
    solid_params["dilation_angle"] = 0.0
    # Similarly, simplify problem by removing elastic normal deformation.
    model_params["material_constants"]["solid"] = FractureDamageSolidConstants(
        **solid_params
    )  # type: ignore[arg-type]
    m = cls(model_params)
    pp.run_time_dependent_model(m, solver_params)

    return m.results, m


@pytest.mark.parametrize("dim", [2, 3])
# The tests take about a minute and are not critical, rather a supplement to test_damage
@pytest.mark.skipped  # reason: slow
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
        names = [damage + "_damage", damage + "_damage_history"]
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
        # i) damage
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
        # ii) ... corresponding to a threefold increase in damage history.
        name = names[1]
        val1 = cast(np.ndarray, getattr(vals[1], "approx_" + name))
        val2 = cast(np.ndarray, getattr(vals[2], "approx_" + name))
        np.testing.assert_allclose(
            val2,
            3 * val1,
            rtol=1e-5,  # Allow some tolerance due to numerical errors.
            err_msg=f"Damage history mismatch for {name} at t=3: {val2}/{val1}.",
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
        rtol=1e-5,  # Allow some tolerance due to numerical errors.
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
        rtol=1e-3,
        err_msg=f"Damage length is too small after fourth step: {length_3}",
    )


@pytest.mark.parametrize("dim", [2, 3])
# The tests take about a minute and are not critical, rather a supplement to test_damage
@pytest.mark.skipped  # reason: slow
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
        names = [damage + "_damage", damage + "_damage_history"]
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
        rtol=1e-3,
        err_msg=f"Damage length is too small after fourth step: {length_3}",
    )


@pytest.mark.parametrize("isotropic", [True, False])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "damages",
    [["dilation"], ["friction"], ["dilation", "friction"]],
)
@pytest.mark.parametrize(
    "time_steps", [2, pytest.param(5, marks=pytest.mark.skipped(reason="slow"))]
)  # The skipped tests take several minutes.
def test_damage(
    isotropic: bool,
    dim: int,
    damages: list[Literal["friction", "dilation"]],
    time_steps: int,
):
    """Run damage test for a specific model.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
        dim: Spatial dimension of the domain (2 or 3).
        damages: List of damage types to include in the test.
        time_steps: Number of time steps to simulate.

    """
    # Reuse shared builder for consistency across tests.
    cls, model_params, solver_params = setup(
        isotropic=isotropic, dim=dim, damages=damages, time_steps=time_steps
    )
    m = cls(model_params)
    pp.run_time_dependent_model(m, solver_params)
    # Initial time step is not included in VerificationDataSaving.
    for t in np.arange(time_steps - 1):
        # Retrieve stored results for time step t + 1. Test against exact solution for
        # a list of method names defined in the model and the exact solution class.
        for name in damage_examples.DATA_SAVING_METHOD_NAMES:
            # Skip if the model does not contain the specified damage type.
            if "dilation" in name and "dilation" not in damages:
                continue
            if "friction" in name and "friction" not in damages:
                continue
            results = m.results[t]

            exact = getattr(results, "exact_" + name)
            approx = getattr(results, "approx_" + name)

            np.testing.assert_allclose(
                approx,
                exact,
                atol=1e-7,
                rtol=6e-3,  # Lenient relative tolerance due to simplifications in exact
                # solution.
                err_msg=f"Mismatch for {name} at step {t + 1}",
            )
