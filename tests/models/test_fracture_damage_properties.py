"""Property tests for the fracture damage model.

These tests verify qualitative invariants that any correct implementation must satisfy.

Properties tested:

    1. Zero tangential slip → damage history remains zero.
    2. Open fracture (tensile state) → no damage accumulates even with nonzero
        tangential displacement jump.
    3. Monotonically increasing tangential slip → non-decreasing damage history.
    4. Non-decreasing damage history → non-increasing damage factor d.
    5. Damage factor d stays in [d0, 1] for all time steps.
    6. After a load reversal in 2D, the isotropic model accumulates strictly more
        history than the anisotropic model.
    7. For purely one-directional loading the two models agree on damage history.

All tests use a 2D square domain with a single horizontal fracture, driven purely by
Dirichlet displacement boundary conditions on boundary faces satisfying y > 0.5. The
dilation angle is set to zero to decouple the normal mechanics from the tangential slip,
making the setup better conditioned. The friction coefficient is kept small (0.01) so
that the fracture slip closely tracks the applied BC displacement.

TODO: Decide on placement. These tests are more high-level than typical unit tests, and
could be placed in test_fracture_damage.py. Kept here until review for a less cluttered
test_fracture_damage.py.
"""

import copy
from typing import Sequence, cast

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import add_mixin
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.examples import fracture_damage as damage_examples
from porepy.models import fracture_damage as damage_models
from porepy.numerics.solvers.line_search import ConstraintLineSearchNonlinearSolver

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NORMAL_DISP_COMPRESSION = 8e-4
"""North-boundary y-displacement that keeps the fracture closed (in compression).

This value is chosen to be below the fracture gap opening, so the fracture remains in
contact with a sensible compressive traction."""

_NORMAL_DISP_OPEN = 2.0e-3
"""North-boundary y-displacement large enough to open the fracture (tensile state)."""


def _solid_params(dilation_angle: float = 0.0) -> dict:
    """Solid material parameters tuned for stable property tests.

    Zero dilation angle decouples the normal mechanics from tangential slip and removes
    one source of nonlinearity.  The small friction coefficient (0.01, inherited from
    the example defaults) means that fracture slip approximately equals the applied BC
    displacement, making the expected behaviour easy to reason about.
    """
    p = damage_examples.solid_params.copy()
    p["dilation_angle"] = dilation_angle
    return p


def _solver_params() -> dict:
    return {
        "nl_max_iterations": 50,
        "nl_convergence_res_atol": 1e-8,
        "nl_convergence_inc_atol": 1e-8,
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "local_line_search": True,
    }


def build_and_run(
    north_displacements: np.ndarray,
    damages: Sequence[str],
    isotropic: bool,
    solid_params: dict | None = None,
) -> tuple[list, object]:
    """Build a 2D fracture damage model and run it.

    Parameters:
        north_displacements: Array of shape (2, T) with x (tangential) and y (normal)
            displacement values prescribed on the north boundary at each time step
            t = 0, ..., T-1. Step 0 is the initial condition.
        damages: Damage types to activate, a non-empty subset of
            {"dilation", "friction"}.
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
        solid_params: Solid material parameters. Defaults to _solid_params().

    Returns:
        Tuple of (results, model), where results is a list of DamageSaveData objects
        (one per forward time step) and model is the model instance after the run.
    """
    if solid_params is None:
        solid_params = _solid_params()

    num_time_steps = north_displacements.shape[1]

    params = copy.deepcopy(damage_examples.model_params)
    params["north_displacements"] = north_displacements
    params["time_manager"] = pp.TimeManager(np.arange(0, num_time_steps), 1, True)
    params["material_constants"] = {
        "solid": FractureDamageSolidConstants(**solid_params),  # type: ignore[arg-type]
    }
    # The exact_solution key is required by DamageDataSaving.  We set it here so
    # collect_data does not raise, but the tests only inspect approx_* values.
    params["exact_solution"] = (
        damage_examples.ExactSolutionIsotropic
        if isotropic
        else damage_examples.ExactSolutionAnisotropic
    )

    # Assemble model class.
    model_class = damage_examples.FractureDamageMomentumBalance
    length_mixin = (
        damage_models.IsotropicFractureDamageLength
        if isotropic
        else damage_models.AnisotropicFractureDamageLength
    )
    model_class = add_mixin(length_mixin, model_class)
    for name in damages:
        model_class = add_mixin(damage_examples.damage_types[name], model_class)
    model_class = add_mixin(SquareDomainOrthogonalFractures, model_class)

    model = model_class(params)
    pp.ModelRunner(model, _solver_params()).run()
    return model.results, model


def _histories(results: list, damage: str) -> list[np.ndarray]:
    """Return the list of damage history arrays (one per step)."""
    return [
        cast(np.ndarray, getattr(r, f"approx_{damage}_damage_history")) for r in results
    ]


def _damages(results: list, damage: str) -> list[np.ndarray]:
    """Return the list of damage factor d arrays (one per step)."""
    return [
        cast(np.ndarray, getattr(r, f"approx_{damage}_damage_state")) for r in results
    ]


# ---------------------------------------------------------------------------
# 1. Zero tangential slip → zero damage history
# ---------------------------------------------------------------------------


def test_no_damage_for_zero_tangential_slip():
    """With compression but no tangential slip the damage history stays zero.

    The fracture is always in contact (positive normal compression), but the applied
    x-displacement is zero throughout.  No plastic tangential slip means no accumulation
    in the convolution integral, so the damage history variable must remain zero.
    """
    num_steps = 3
    north = np.zeros((2, num_steps))
    north[1, :] = _NORMAL_DISP_COMPRESSION  # y: compression, no tangential

    results, _ = build_and_run(north, ["dilation", "friction"], isotropic=True)

    for damage in ["dilation", "friction"]:
        for i, h in enumerate(_histories(results, damage)):
            np.testing.assert_allclose(
                h,
                0.0,
                atol=1e-6,
                err_msg=(
                    f"{damage} damage history non-zero at step {i + 1} "
                    f"with no tangential slip: {h}"
                ),
            )


# ---------------------------------------------------------------------------
# 2. Open fracture → no damage accumulation
# ---------------------------------------------------------------------------


def test_no_damage_for_open_fracture():
    """Tangential slip under tensile normal loading does not accumulate damage.

    When the fracture is open the contact state characteristic equals 1, which forces
    the damage equation to keep the history variable at its previous value. Since the
    initial value is zero and the fracture remains open throughout, the history must
    stay zero at every step, even with non-zero applied tangential displacements.
    """
    num_steps = 3
    north = np.zeros((2, num_steps))
    north[1, :] = _NORMAL_DISP_OPEN  # large y: fracture stays open
    north[0, 1] = 1.0e-4  # tangential displacement jump in open state
    north[0, 2] = 2.0e-4

    results, _ = build_and_run(north, ["dilation", "friction"], isotropic=True)

    for damage in ["dilation", "friction"]:
        for i, h in enumerate(_histories(results, damage)):
            np.testing.assert_allclose(
                h,
                0.0,
                atol=1e-6,
                err_msg=(
                    f"{damage} damage history non-zero at step {i + 1} "
                    f"with open fracture: {h}"
                ),
            )


# ---------------------------------------------------------------------------
# Shared fixture: 4-step monotone tangential loading.
# Used by tests 3, 4, and 5 which share identical setup parameters.
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(("dilation", True), id="dilation-isotropic"),
        pytest.param(
            ("dilation", False),
            id="dilation-anisotropic",
            marks=pytest.mark.skipped,  # reason: slow
        ),
        pytest.param(("friction", True), id="friction-isotropic"),
        pytest.param(
            ("friction", False),
            id="friction-anisotropic",
            marks=pytest.mark.skipped,  # reason: slow
        ),
    ],
)
def monotone_slip_results(request):
    """Results from a 4-step monotone tangential loading run.

    The solver is invoked once per ``(damage, isotropic)`` combination and the cached
    ``(results, model, damage, isotropic)`` tuple is shared among the three tests that
    depend on it (non-decreasing history, non-increasing factor, bounded factor).  This
    avoids running three identical simulations per combination.
    """
    damage, isotropic = request.param
    num_steps = 4
    north = np.zeros((2, num_steps))
    north[1, :] = _NORMAL_DISP_COMPRESSION
    north[0, :] = np.array([0.0, 1.0e-4, 2.0e-4, 3.0e-4])
    results, model = build_and_run(north, [damage], isotropic=isotropic)
    return results, model, damage, isotropic


# ---------------------------------------------------------------------------
# 3. Monotonic tangential slip → non-decreasing damage history
# ---------------------------------------------------------------------------


def test_damage_history_non_decreasing_monotonic_slip(monotone_slip_results):
    """Non-negative slip increments must yield a non-decreasing damage history.

    The convolution integral accumulates |length| * coefficient contributions that are
    always >= 0.  Under monotonic loading in one direction every step adds a positive
    term, so Lambda(t+1) >= Lambda(t).
    """
    results, _, damage, isotropic = monotone_slip_results
    histories = _histories(results, damage)
    for i in range(1, len(histories)):
        assert np.all(histories[i] >= histories[i - 1] - 1e-10), (
            f"{damage} (isotropic={isotropic}) history decreased at step {i + 1}: "
            f"{histories[i - 1]} → {histories[i]}"
        )


# ---------------------------------------------------------------------------
# 4. Non-decreasing history → non-increasing damage factor d
# ---------------------------------------------------------------------------


def test_damage_factor_non_increasing_monotonic_slip(monotone_slip_results):
    """Monotonically increasing Lambda must yield non-increasing d.

    The damage factor is d = d0 + (1 - d0) * exp(-Lambda).  Since exp is strictly
    decreasing, d decreases as Lambda increases.  After the first non-zero slip
    increment d must be strictly below 1 and must not increase on subsequent steps.
    """
    results, _, damage, isotropic = monotone_slip_results
    factors = _damages(results, damage)
    for i in range(1, len(factors)):
        assert np.all(factors[i] <= factors[i - 1] + 1e-10), (
            f"{damage} (isotropic={isotropic}) damage factor increased at step {i + 1}:"
            f" {factors[i - 1]} → {factors[i]}"
        )


# ---------------------------------------------------------------------------
# 5. Damage factor bounded in [d0, 1]
# ---------------------------------------------------------------------------


def test_damage_bounded(monotone_slip_results):
    """The damage factor d must lie in [d0, 1] at every time step.

    The analytic formula d = d0 + (1-d0)*exp(-Lambda) with Lambda >= 0 guarantees this.
    The test checks that the numerical implementation preserves the bounds.
    """
    results, model, damage, isotropic = monotone_slip_results
    d0 = (
        model.solid.residual_dilation_damage
        if damage == "dilation"
        else model.solid.residual_friction_damage
    )
    for i, d in enumerate(_damages(results, damage)):
        assert np.all(d >= d0 - 1e-10), f"{damage} d below d0={d0} at step {i + 1}: {d}"
        assert np.all(d <= 1.0 + 1e-10), f"{damage} d above 1.0 at step {i + 1}: {d}"


# ---------------------------------------------------------------------------
# 6. After a load reversal: isotropic accumulates strictly more history
# ---------------------------------------------------------------------------


@pytest.mark.skipped  # reason: slow
def test_isotropic_more_history_after_reversal():
    """Isotropic model accumulates strictly more history after a load reversal.

    Loading profile (north x-displacement): 0 → +d → −d.

    *Isotropic*: history += |Δu_t| at every step, regardless of direction. After two
        increments of magnitude d the history is proportional to 2d.

    *Anisotropic*: length is the projection of each past increment onto the current
        cumulative slip direction. After a full reversal the first increment projects
        negatively onto the current direction (clipped to 0). Only the second increment
        contributes. The history is proportional to d.

    Hence Lambda_iso > Lambda_aniso after the reversal.  Because d = d0 + (1-d0)exp(-Λ)
    is strictly decreasing in Λ, the damage factors satisfy d_iso < d_aniso.
    """
    # Use a displacement magnitude large enough that the fracture slip changes sign
    # despite the low but non-zero friction.
    slip = 2.0e-4
    num_steps = 3
    north = np.zeros((2, num_steps))
    north[1, :] = _NORMAL_DISP_COMPRESSION
    north[0, 1] = +slip  # forward step
    north[0, 2] = -slip  # equal-magnitude reversal

    results_iso, _ = build_and_run(north, ["dilation", "friction"], isotropic=True)
    results_aniso, _ = build_and_run(north, ["dilation", "friction"], isotropic=False)

    for damage in ["dilation", "friction"]:
        h_iso = _histories(results_iso, damage)[-1]
        h_aniso = _histories(results_aniso, damage)[-1]

        assert np.all(h_iso > h_aniso + 1e-12), (
            f"{damage}: isotropic history {h_iso} not strictly greater than "
            f"anisotropic {h_aniso} after load reversal."
        )


# ---------------------------------------------------------------------------
# 7. Single-direction loading: isotropic and anisotropic histories agree
# ---------------------------------------------------------------------------


@pytest.mark.skipped  # reason: slow
def test_isotropic_anisotropic_agree_for_unidirectional_loading():
    """Isotropic and anisotropic lengths are identical for purely one-directional slip.

    When all slip increments are in the same direction the normalised direction vector m
    is constant and aligned with each increment.  The anisotropic formula then reduces
    to the isotropic one (both evaluate to |Δu_t| per step), so the two models must
    produce the same damage history and the same damage factor.
    """
    num_steps = 3
    north = np.zeros((2, num_steps))
    north[1, :] = _NORMAL_DISP_COMPRESSION
    north[0, :] = np.array([0.0, 1.0e-4, 2.0e-4])  # monotone in one direction

    results_iso, _ = build_and_run(north, ["dilation", "friction"], isotropic=True)
    results_aniso, _ = build_and_run(north, ["dilation", "friction"], isotropic=False)

    for damage in ["dilation", "friction"]:
        for i, (h_iso, h_aniso) in enumerate(
            zip(_histories(results_iso, damage), _histories(results_aniso, damage))
        ):
            np.testing.assert_allclose(
                h_iso,
                h_aniso,
                rtol=1e-4,
                err_msg=(
                    f"{damage}: isotropic/anisotropic history mismatch "
                    f"at step {i + 1}. "
                    f"iso={h_iso}, aniso={h_aniso}"
                ),
            )
