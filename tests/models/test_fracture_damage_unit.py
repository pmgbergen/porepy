"""Unit tests for fracture damage constitutive laws.

These tests prescribe primary variable values directly in the equation system, bypassing
the PDE solver, to verify that derived operators evaluate to their analytically expected
values.

No time integration is performed. ``prepare_simulation`` is called once to fully
initialize the equation system and geometry. Values are then injected directly via
``equation_system.set_variable_values`` at iterate index 0.

Covered formulas
----------------
- Damage factor:
    ``d_alpha = d0_alpha + (1 - d0_alpha) * exp(-max(Lambda, 0) / Lambda_c_alpha)``
- Stress partition:
    ``a_s = 1 - clip(1 - sigma_n / sigma_T, 0, 1) ** K``
- Composed friction coefficient:
    ``mu* = (mu_b + tan psi)/(1 - mu_b tan psi) + a_s mu_p0 d_f``,
    with ``tan psi = (1 - a_s) tan psi_0 d_d``
- Damage evolution coefficient (shared by both channels, Archard):
    ``k = pos_normal * char_traction / sqrt(Lambda_c_d * Lambda_c_f)``
- Damage length:
    - Isotropic, k=0: ``L = |u_t_iterate − u_t_ts0|``
    - Anisotropic, k=0: ``L = |max(0, m · u_t_ts0) − |u_t_iterate||``

where ``t_n_nondim`` is the nondimensional normal contact traction (negative when the
fracture is in compression), ``char_traction`` is
``numerical.characteristic_contact_traction``, ``pos_normal = -t_n_nondim``, ``m =
u_t_iterate / |u_t_iterate|`` and ``u_t_ts0`` is the value at ``time_step_index=0``.

TODO: Decide on placement. Everything is defined in constitutive_laws bar the length
operator, which is defined in the momentum models.fracture_damage.

"""

from typing import Any

import numpy as np
import pytest

import porepy as pp
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.examples import fracture_damage as damage_example

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepared_model(
    isotropic: bool = True,
    damages: list[str] | None = None,
    dim: int = 2,
    solid_overrides: dict[str, Any] | None = None,
) -> Any:
    """Build and prepare (but not run) a fracture damage model.

    ``prepare_simulation`` is called so that the equation system, all variables, and the
    geometry are fully initialized.  No time step is advanced and no nonlinear solve is
    performed.

    Tests modify the equation system directly at iterate index 0 to prescribe variable
    values. Hence not a fixture.

    Parameters:
        isotropic: If ``True``, use ``IsotropicFractureDamageLength``; otherwise use
            ``AnisotropicFractureDamageLength``.
        damages: Damage types to activate, a non-empty subset of
            ``{"dilation", "friction"}``.  Defaults to both.
        dim: Spatial dimension of the bulk domain (2 or 3).
        solid_overrides: Solid constants to override on top of the example's, e.g. to
            place the transitional traction where a test wants it.

    Returns:
        A prepared model instance with a fully initialized equation system.
    """
    if damages is None:
        damages = ["dilation", "friction"]

    model_class, params, _ = damage_example.create_displacement_controlled_setup(
        isotropic=isotropic,
        damages=damages,
        dim=dim,
    )
    # Zero displacement BCs: shape (dim, 2) — dim components × 2 time steps.
    params["north_displacements"] = np.zeros((dim, 2))
    params["time_manager"] = pp.TimeManager([0, 1], 1, True)
    params["exact_solution"] = (
        damage_example.ExactSolutionIsotropic
        if isotropic
        else damage_example.ExactSolutionAnisotropic
    )
    if solid_overrides is not None:
        solid = damage_example.solid_params.copy()
        solid.update(solid_overrides)
        params["material_constants"] = {
            "solid": FractureDamageSolidConstants(**solid)  # type: ignore[arg-type]
        }
    model = model_class(params)
    model.prepare_simulation()
    return model


# ---------------------------------------------------------------------------
# 1.  Damage state formula: d = d0 + (1 - d0) * exp(-Lambda)
# ---------------------------------------------------------------------------


def _nondimensional_wear_energy_scale(model, damage: str) -> float:
    """Return ``Lambda_c^alpha`` scaled the same way as the history variable.

    The history is nondimensionalized by the characteristic wear energy, so the scale it
    is compared against must be too. Tests prescribe histories as multiples of this
    quantity, which keeps them independent of the fixture's characteristic scales.
    """
    fractures = model.mdg.subdomains(dim=model.nd - 1)
    scale = getattr(model, f"{damage}_wear_energy_scale")(fractures)
    return float(
        np.mean(
            model.equation_system.evaluate(
                scale / model.characteristic_wear_energy(fractures)
            )
        )
    )


class TestDamageStateFormula:
    """Algebraic formula ``d = d0 + (1 - d0) * exp(-Lambda / Lambda_c)``.

    The AD implementation clips Lambda below at zero before exponentiating; no upper
    clip is applied. Histories are prescribed as multiples of the (nondimensionalized)
    wear energy scale, so the tests read directly as values of the exponent.
    """

    @staticmethod
    def _prepared_model_with_fractures(
        isotropic: bool = True,
        damages: list[str] | None = None,
        dim: int = 2,
    ) -> tuple[Any, list[pp.Grid], int]:
        """Return a prepared model together with fracture subdomains and cell count."""
        model = _prepared_model(isotropic=isotropic, damages=damages, dim=dim)
        fractures = model.mdg.subdomains(dim=model.nd - 1)
        nc = sum(sd.num_cells for sd in fractures)
        return model, fractures, nc

    @staticmethod
    def _set_history(model, damage: str, exponent: float) -> None:
        """Prescribe a history of ``exponent * Lambda_c^alpha`` on all cells."""
        fractures = model.mdg.subdomains(dim=model.nd - 1)
        nc = sum(sd.num_cells for sd in fractures)
        scale = _nondimensional_wear_energy_scale(model, damage)
        model.equation_system.set_variable_values(
            exponent * scale * np.ones(nc),
            variables=[model.damage_history(fractures)],
            iterate_index=0,
        )

    @pytest.mark.parametrize("damage", ["dilation", "friction"])
    @pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0, 3.0])
    def test_damage_state_matches_formula(self, damage: str, exponent: float):
        """State evaluates to ``d0 + (1-d0)*exp(-Lambda/Lambda_c)`` for both types.

        Parameters:
            damage: Damage type, either ``"dilation"`` or ``"friction"``.
            exponent: History prescribed as this multiple of the wear energy scale.
        """
        model, fractures, nc = self._prepared_model_with_fractures(damages=[damage])
        d0 = float(getattr(model.solid, f"residual_{damage}_damage"))

        self._set_history(model, damage, exponent)

        d = getattr(model, f"{damage}_damage_state")(fractures).value(
            model.equation_system
        )
        expected = d0 + (1.0 - d0) * np.exp(-exponent)
        np.testing.assert_allclose(d, expected * np.ones(nc), rtol=1e-12)

    @pytest.mark.parametrize("damage", ["dilation", "friction"])
    def test_damage_state_is_one_for_zero_history(self, damage: str):
        """Lambda = 0 means no accumulated damage: state must equal 1."""
        model, fractures, nc = self._prepared_model_with_fractures(damages=[damage])

        model.equation_system.set_variable_values(
            np.zeros(nc),
            variables=[model.damage_history(fractures)],
            iterate_index=0,
        )
        d = getattr(model, f"{damage}_damage_state")(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(d, np.ones(nc), rtol=1e-12)

    def test_dilation_damage_approaches_d0_at_large_history(self):
        """A history of ten scales drives the damage state to d0."""
        model, fractures, nc = self._prepared_model_with_fractures(damages=["dilation"])
        d0 = float(model.solid.residual_dilation_damage)

        self._set_history(model, "dilation", 10.0)

        d = model.dilation_damage_state(fractures).value(model.equation_system)
        expected = d0 + (1.0 - d0) * np.exp(-10.0)
        np.testing.assert_allclose(d, expected * np.ones(nc), rtol=1e-12)
        # exp(-10) ~ 4.5e-5, so the state has effectively reached its residual.
        assert np.all(d - d0 < 1e-4 * (1.0 - d0))

    def test_damage_state_is_monotone_in_history(self):
        """A larger history value produces a smaller (more damaged) state."""
        model, fractures, nc = self._prepared_model_with_fractures(damages=["dilation"])

        def _eval(exponent: float) -> float:
            self._set_history(model, "dilation", exponent)
            return float(
                np.mean(
                    model.dilation_damage_state(fractures).value(model.equation_system)
                )
            )

        assert _eval(0.0) > _eval(1.0) > _eval(3.0)


# ---------------------------------------------------------------------------
# 2.  Damage evolution coefficients and normalized traction
# ---------------------------------------------------------------------------


class TestDamageEvolutionCoefficients:
    r"""Tests for the damage evolution coefficient and normalized traction.

    All tests prescribe the contact traction variable to a known nondimensional normal
    value (zero tangential component) and compare the evaluated operator with the
    analytically computed reference.

    The coefficient implements Archard's wear law,

    .. math::
        k = |t_n| / u_{char},

    which is *linear* in the normal traction. A single coefficient serves both channels:
    what distinguishes them is the wear energy scale in the softening, which is the
    subject of :meth:`test_channels_differ_only_by_wear_energy_scale`.

    The formula is written in terms of nondimensional tractions because the contact
    traction variable is nondimensionalized by ``char_traction``, which is multiplied
    back in before dividing by the characteristic wear energy.
    """

    @staticmethod
    def _fractures(model):
        return model.mdg.subdomains(dim=model.nd - 1)

    def _set_normal_traction(self, model, t_n_nondim: float) -> None:
        """Prescribe a uniform nondimensional normal traction to all fracture cells.

        The tangential component is set to zero.

        Parameters:
            model: A prepared model instance.
            t_n_nondim: Nondimensional normal traction (negative = compression).
        """
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)

        values = np.zeros(nc * model.nd)
        # In local fracture coordinates the normal component is last (index nd-1).
        values[model.nd - 1 :: model.nd] = t_n_nondim

        model.equation_system.set_variable_values(
            values,
            variables=[model.contact_traction(fractures)],
            iterate_index=0,
        )

    def _material_constants(self, model):
        """Return (char_traction, reference_wear_energy) as floats.

        ``char_traction`` is obtained by evaluating the operator, which uses the Young's
        modulus and characteristic displacement (not the scalar stored in
        ``numerical.characteristic_contact_traction``).
        """
        fractures = self._fractures(model)
        evaluate = model.equation_system.evaluate
        char_t = float(
            np.mean(evaluate(model.characteristic_contact_traction(fractures)))
        )
        reference_energy = float(
            np.mean(evaluate(model.characteristic_wear_energy(fractures)))
        )
        return char_t, reference_energy

    def test_evolution_coefficient_formula(self):
        """Coefficient equals ``|t_n| * char_traction / sqrt(Lc_d * Lc_f)``."""
        model = _prepared_model(damages=["dilation"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        char_t, reference_energy = self._material_constants(model)

        self._set_normal_traction(model, -0.4)
        expected = 0.4 * char_t / reference_energy

        result = model.damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(result, expected * np.ones(nc), rtol=1e-10)

    def test_reference_energy_is_geometric_mean_of_the_scales(self):
        """The nondimensionalisation reference is ``sqrt(Lc_d * Lc_f)``.

        A mechanical reference such as ``char_traction * u_char`` would be an elastic
        energy, unrelated to the wear energies it is meant to scale, and would leave the
        nondimensional history far from unity. Pinning the reference to the two scales
        keeps it at order one whatever they are.
        """
        model = _prepared_model(damages=["dilation", "friction"])
        fractures = self._fractures(model)
        _, reference_energy = self._material_constants(model)

        expected = np.sqrt(
            model.solid.dilation_wear_energy_scale
            * model.solid.friction_wear_energy_scale
        )
        np.testing.assert_allclose(reference_energy, expected, rtol=1e-12)

        # The nondimensional scales are reciprocal square roots of the ratio, hence of
        # order one, and so is the history they are compared against.
        scale_d = _nondimensional_wear_energy_scale(model, "dilation")
        scale_f = _nondimensional_wear_energy_scale(model, "friction")
        np.testing.assert_allclose(scale_d * scale_f, 1.0, rtol=1e-12)

    def test_evolution_coefficient_is_linear_in_traction(self):
        """The coefficient is linear in the normal traction, with no turning point.

        Sampled across a decade of traction, so a wear rate that saturated, turned over
        or reversed anywhere in that range would fail. A failure most likely means a
        nonlinear factor has been introduced into the driver.
        """
        model = _prepared_model(damages=["dilation"])
        fractures = self._fractures(model)
        char_t, _ = self._material_constants(model)
        strength_scale = 1e8  # representative rock strength [Pa], sets the range

        def _eval(fraction_of_strength: float) -> np.ndarray:
            self._set_normal_traction(
                model, -fraction_of_strength * strength_scale / char_t
            )
            return model.damage_evolution_coefficient(fractures).value(
                model.equation_system
            )

        base = _eval(0.1)
        for factor in (2.0, 4.0, 5.0, 9.0):
            np.testing.assert_allclose(_eval(0.1 * factor), factor * base, rtol=1e-10)

    def test_channels_differ_only_by_wear_energy_scale(self):
        """Only the softening scale distinguishes the two channels.

        Setting the two histories to the same value must give damage states related by
        the wear energy scales alone. A failure here means a per-channel factor has
        crept back into the driver, which is what a single history rules out.
        """
        model = _prepared_model(damages=["dilation", "friction"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)

        scale_d = _nondimensional_wear_energy_scale(model, "dilation")
        scale_f = _nondimensional_wear_energy_scale(model, "friction")

        for history in (0.2 * scale_d, scale_d, 2.0 * scale_d):
            model.equation_system.set_variable_values(
                history * np.ones(nc),
                variables=[model.damage_history(fractures)],
                iterate_index=0,
            )
            evaluate = model.equation_system.evaluate
            d_dil = evaluate(model.dilation_damage_state(fractures))
            d_fri = evaluate(model.friction_damage_state(fractures))

            d0_d = model.solid.residual_dilation_damage
            d0_f = model.solid.residual_friction_damage
            # Strip the residuals; the remaining factors are
            # exp(-Lambda/Lambda_c^alpha), so their logs are in the ratio
            # Lambda_c^f / Lambda_c^d.
            decay_d = (d_dil - d0_d) / (1.0 - d0_d)
            decay_f = (d_fri - d0_f) / (1.0 - d0_f)
            np.testing.assert_allclose(
                np.log(decay_f) / np.log(decay_d), scale_d / scale_f, rtol=1e-10
            )

    def test_coefficient_is_negligible_at_zero_traction(self):
        """Zero normal traction (open fracture) gives a negligible coefficient.

        The positive-normal-traction helper clips the contact traction to a maximum of
        ``-1e-15`` (nondim) before negating, so a traction of zero produces ``pos_normal
        = 1e-15`` rather than exactly zero. The resulting coefficient is many orders of
        magnitude below any physically relevant value.
        """
        model = _prepared_model(damages=["dilation"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        char_t, reference_energy = self._material_constants(model)

        self._set_normal_traction(model, 0.0)

        clip_floor = 1e-15  # pos_normal_nondim produced by the clip
        expected = clip_floor * char_t / reference_energy

        result = model.damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(result, expected * np.ones(nc), rtol=1e-6)

    def test_coefficient_is_non_negative(self):
        """The coefficient is non-negative under compression."""
        model = _prepared_model(damages=["dilation", "friction"])
        fractures = self._fractures(model)

        self._set_normal_traction(model, -0.1)

        coefficient = model.damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        assert np.all(coefficient >= 0), "Damage coefficient must be non-negative"


# ---------------------------------------------------------------------------
# 3.  Stress partition
# ---------------------------------------------------------------------------


SIGMA_T = 4.0e7
"""Transitional normal traction [Pa] used throughout the partition tests."""


class TestStressPartition:
    r"""Ladanyi-Archambault partition ``a_s = 1 - (1 - sigma_n / sigma_T) ** K``.

    Tests prescribe the normal contact traction as a fraction of ``sigma_T`` and compare
    against the closed form. The fraction is the natural coordinate here: it is what the
    partition is a function of, and it keeps the tests independent of the fixture's
    characteristic traction.
    """

    @staticmethod
    def _fractures(model):
        return model.mdg.subdomains(dim=model.nd - 1)

    @staticmethod
    def _model(exponent: float = 1.5):
        return _prepared_model(
            damages=["dilation", "friction"],
            solid_overrides={
                "transitional_normal_traction": SIGMA_T,
                "stress_partition_exponent": exponent,
            },
        )

    def _set_traction_fraction(self, model, fraction: float) -> None:
        """Prescribe a normal traction of ``fraction * sigma_T``.

        Positive ``fraction`` is compression; negative is tension.
        """
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        char_t = float(
            np.mean(
                model.equation_system.evaluate(
                    model.characteristic_contact_traction(fractures)
                )
            )
        )
        values = np.zeros(nc * model.nd)
        # In local fracture coordinates the normal component is last, and compression is
        # negative.
        values[model.nd - 1 :: model.nd] = -fraction * SIGMA_T / char_t
        model.equation_system.set_variable_values(
            values, variables=[model.contact_traction(fractures)], iterate_index=0
        )

    def _partition(self, model, fraction: float) -> np.ndarray:
        self._set_traction_fraction(model, fraction)
        return np.asarray(
            model.stress_partition(self._fractures(model)).value(model.equation_system)
        )

    @pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.75, 0.99])
    def test_partition_matches_formula(self, fraction: float):
        """Below the transition the partition is the closed form."""
        model = self._model()
        expected = 1.0 - (1.0 - fraction) ** 1.5
        np.testing.assert_allclose(
            self._partition(model, fraction), expected, rtol=1e-10
        )

    def test_partition_is_one_above_the_transition(self):
        """Above ``sigma_T`` the asperities are fully sheared, ``a_s = 1``.

        This is the test that the base is clipped *before* it is raised. With the clip
        applied afterwards the base is negative here and a non-integer power of it has
        no real value, so the failure is not a wrong number but a complex or NaN one.
        """
        model = self._model()
        for fraction in (1.0, 1.5, 4.0):
            np.testing.assert_allclose(
                self._partition(model, fraction), 1.0, rtol=1e-12
            )

    def test_partition_vanishes_in_tension(self):
        """A fracture in tension carries no sheared contact, ``a_s = 0``.

        The upper clip is what enforces this: without it the base exceeds one and the
        partition would go negative, i.e. the sliding fraction would exceed the whole
        contact.
        """
        model = self._model()
        for fraction in (-0.5, -2.0):
            np.testing.assert_allclose(self._partition(model, fraction), 0.0, atol=1e-9)

    def test_partition_is_bounded_and_monotone(self):
        """``a_s`` is confined to [0, 1] and increases with the normal traction."""
        model = self._model()
        fractions = np.linspace(-1.0, 3.0, 41)
        values = np.array(
            [float(np.mean(self._partition(model, f))) for f in fractions]
        )

        assert np.all(values >= -1e-9) and np.all(values <= 1.0 + 1e-12)
        assert np.all(np.diff(values) >= -1e-12)

    def test_partition_is_smooth_at_the_transition(self):
        """No kink at ``sigma_n = sigma_T``: the slope vanishes from both sides.

        This asserts an observed *rate* rather than a threshold, which is worth
        spelling out because it is not the style used elsewhere in this file.

        The quantity of interest is the one-sided derivative on each side of the
        transition. Both are zero: above ``sigma_T`` the partition is clipped flat, and
        below it ``da_s/dsigma_n ~ (1 - sigma_n/sigma_T) ** (K - 1)``, which for
        ``K > 1`` decays to zero as the transition is approached. So ``a_s`` is C1
        there, not merely continuous.

        A derivative that is zero cannot be checked against a tolerance directly: a
        centred difference straddling the transition is not zero but ``sqrt(h)/2`` for
        ``K = 1.5``, so any fixed bound on it is really a statement about ``h`` and says
        nothing about the law. Comparing two step sizes removes ``h`` from the claim.
        The quotient's leading term is ``h ** (K - 1)``, so quartering the step must
        halve it, exactly and independently of the constants -- hence the tight ``rtol``
        on a ratio of two crude finite differences.

        The test discriminates sharply. A hard switch at ``sigma_T`` has a quotient that
        *grows* as ``1/h``; a linear ramp (``K = 1``) has one that is constant in ``h``;
        only ``K = 1.5`` gives the factor of two. Clipping after the power rather than
        before would not reach this test at all, since the evaluation below the
        transition would already have failed.
        """

        def slope(model, centre: float, step: float) -> float:
            below = float(np.mean(self._partition(model, centre - step)))
            above = float(np.mean(self._partition(model, centre + step)))
            return (above - below) / (2 * step)

        def refinement_ratio(model) -> float:
            """Quotient at the transition, coarse step over a four times finer one."""
            return slope(model, 1.0, 1e-3) / slope(model, 1.0, 2.5e-4)

        model = self._model()
        np.testing.assert_allclose(refinement_ratio(model), 2.0, rtol=1e-6)

        # Well below the transition the slope is order one, so the vanishing above is a
        # property of the transition and not of the partition being flat everywhere.
        assert slope(model, 0.5, 1e-3) > 1.0

        # The rate is what carries the claim, so check that it can come out otherwise:
        # at K = 1 the partition is a ramp with a genuine corner at the transition, and
        # the same quotient is then independent of the step instead of halving with it.
        np.testing.assert_allclose(
            refinement_ratio(self._model(exponent=1.0)), 1.0, rtol=1e-6
        )

    def test_partition_is_inert_by_default(self):
        """With no transitional traction set, all contact is sliding contact.

        The default is infinite, so the model reduces to the pure sliding law rather
        than silently assuming full ploughing.
        """
        model = _prepared_model(damages=["dilation", "friction"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        values = np.zeros(nc * model.nd)
        values[model.nd - 1 :: model.nd] = -0.4
        model.equation_system.set_variable_values(
            values, variables=[model.contact_traction(fractures)], iterate_index=0
        )
        np.testing.assert_allclose(
            model.stress_partition(fractures).value(model.equation_system), 0.0
        )


# ---------------------------------------------------------------------------
# 4.  Composed friction coefficient
# ---------------------------------------------------------------------------


MU_B = 0.5
"""Basic friction coefficient used throughout the composition tests."""

PSI_0 = 0.05
"""Intact dilation angle [rad] used throughout the composition tests."""

MU_P0 = 0.3
"""Ploughing coefficient in the fully-ploughing limit."""


class TestComposedFriction:
    r"""Composition ``mu* = (mu_b + tan psi)/(1 - mu_b tan psi) + mu_p``.

    with ``tan psi = (1 - a_s) tan psi_0 d^d`` and ``mu_p = a_s mu_p0 d^f``. Tests
    prescribe the normal traction as a fraction of ``sigma_T`` and the history as a
    multiple of the wear energy scale, then compare against values computed in numpy
    from the same two inputs.
    """

    @staticmethod
    def _model(
        residual_dilation: float = 1.0,
        residual_friction: float = 1.0,
        dilation_angle: float = PSI_0,
        friction_coefficient: float = MU_B,
    ):
        return _prepared_model(
            damages=["dilation", "friction"],
            solid_overrides={
                "transitional_normal_traction": SIGMA_T,
                "stress_partition_exponent": 1.5,
                "ploughing_friction_coefficient": MU_P0,
                "friction_coefficient": friction_coefficient,
                "dilation_angle": dilation_angle,
                "residual_dilation_damage": residual_dilation,
                "residual_friction_damage": residual_friction,
            },
        )

    @staticmethod
    def _fractures(model):
        return model.mdg.subdomains(dim=model.nd - 1)

    def _set_state(self, model, traction_fraction: float, exponent: float = 0.0):
        """Prescribe ``sigma_n = fraction * sigma_T`` and ``Lambda = exponent * Lc^f``.

        The history is expressed against the friction scale. The dilation channel sees
        the same history divided by its own scale, which the closed forms below account
        for.
        """
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        evaluate = model.equation_system.evaluate
        char_t = float(
            np.mean(evaluate(model.characteristic_contact_traction(fractures)))
        )
        traction = np.zeros(nc * model.nd)
        traction[model.nd - 1 :: model.nd] = -traction_fraction * SIGMA_T / char_t
        model.equation_system.set_variable_values(
            traction, variables=[model.contact_traction(fractures)], iterate_index=0
        )
        scale_f = _nondimensional_wear_energy_scale(model, "friction")
        model.equation_system.set_variable_values(
            exponent * scale_f * np.ones(nc),
            variables=[model.damage_history(fractures)],
            iterate_index=0,
        )

    @staticmethod
    def _mean(model, operator) -> float:
        """Return the mean of an operator evaluated at iterate index 0."""
        return float(np.mean(model.equation_system.evaluate(operator)))

    # -- Patton recovery, brief section 6.2 ---------------------------------------

    def test_patton_recovery_at_vanishing_traction(self):
        """``mu* -> tan(phi_b + psi_0)`` as ``sigma_n -> 0`` with intact asperities.

        At vanishing traction nothing is sheared through, so ``a_s = 0`` removes the
        ploughing term and leaves the full dilation angle. The composition is then a
        tangent addition, and it must reproduce Patton's sliding envelope exactly. This
        is the strongest single check on the composition: it ties three separate pieces
        -- the partition, the dilation scaling and the tangent-addition formula -- to
        one closed-form value that none of them contains.
        """
        model = self._model()
        self._set_state(model, traction_fraction=1e-9)

        expected = np.tan(np.arctan(MU_B) + PSI_0)
        np.testing.assert_allclose(
            self._mean(model, model.friction_coefficient(self._fractures(model))),
            expected,
            rtol=1e-8,
        )

    def test_patton_recovery_at_the_transition(self):
        """``mu* = mu_b + mu_p0`` at ``sigma_n = sigma_T`` with intact asperities.

        The other end of the envelope: all contact is sheared through, so the dilation
        term vanishes with ``1 - a_s`` and the ploughing term is at its limit. That the
        dilation contribution disappears *exactly*, leaving no residue of the tangent
        addition, is what this pins down.
        """
        model = self._model()
        self._set_state(model, traction_fraction=1.0)

        np.testing.assert_allclose(
            self._mean(model, model.friction_coefficient(self._fractures(model))),
            MU_B + MU_P0,
            rtol=1e-12,
        )

    # -- The composition away from the two limits ---------------------------------

    @pytest.mark.parametrize("traction_fraction", [0.15, 0.4, 0.8])
    @pytest.mark.parametrize("exponent", [0.0, 0.7, 2.5])
    def test_composition_matches_formula(
        self, traction_fraction: float, exponent: float
    ):
        """``mu*`` matches the closed form between the limits and under damage.

        Parameters:
            traction_fraction: Normal traction as a fraction of ``sigma_T``.
            exponent: History as this multiple of the friction wear energy scale.
        """
        residual_d, residual_f = 0.6, 0.3
        model = self._model(residual_dilation=residual_d, residual_friction=residual_f)
        self._set_state(model, traction_fraction, exponent)

        # The dilation channel reads the same history against its own scale.
        scale_ratio = _nondimensional_wear_energy_scale(
            model, "friction"
        ) / _nondimensional_wear_energy_scale(model, "dilation")

        a_s = 1.0 - (1.0 - traction_fraction) ** 1.5
        d_f = residual_f + (1.0 - residual_f) * np.exp(-exponent)
        d_d = residual_d + (1.0 - residual_d) * np.exp(-exponent * scale_ratio)
        tan_psi = (1.0 - a_s) * np.tan(PSI_0) * d_d
        expected = (MU_B + tan_psi) / (1.0 - MU_B * tan_psi) + a_s * MU_P0 * d_f

        np.testing.assert_allclose(
            self._mean(model, model.friction_coefficient(self._fractures(model))),
            expected,
            rtol=1e-10,
        )

    def test_basic_friction_is_the_floor(self):
        """Fully worn asperities leave ``mu_b`` and nothing else.

        ``mu_b`` is a property of the rock surfaces, not of their geometry, so no amount
        of wear removes it. With both residual states at zero and a large history, both
        the dilation and the ploughing term must vanish, whatever the traction.
        """
        model = self._model(residual_dilation=0.0, residual_friction=0.0)
        for traction_fraction in (0.05, 0.5, 1.0):
            self._set_state(model, traction_fraction, exponent=200.0)
            np.testing.assert_allclose(
                self._mean(model, model.friction_coefficient(self._fractures(model))),
                MU_B,
                rtol=1e-12,
            )

    def test_dissipation_is_positive(self):
        r"""``mu* - tan psi > 0`` everywhere in the admissible parameter set.

        The dissipation per unit slip is ``(mu* - tan psi) sigma_n``: the frictional
        work less the part recovered as dilation. Positivity is what makes the law
        thermodynamically admissible, and it is an identity rather than a numerical
        accident, since

            mu* - tan psi = mu_b (1 + tan^2 psi)/(1 - mu_b tan psi) + mu_p,

        which is positive term by term whenever ``mu_b tan psi < 1``. The test is
        therefore checking the implementation against the algebra, not exploring a
        risk: a failure means the composition was assembled wrongly, not that the
        parameters strayed.

        It is asserted over a grid of traction and history because ``tan psi`` and
        ``mu_p`` move in opposite directions as either is varied, so a sign error in one
        term can be masked at any single point.
        """
        model = self._model(residual_dilation=0.2, residual_friction=0.0)
        fractures = self._fractures(model)

        for traction_fraction in (0.01, 0.2, 0.6, 1.0, 2.0):
            for exponent in (0.0, 0.5, 2.0, 10.0):
                self._set_state(model, traction_fraction, exponent)
                dissipation = self._mean(
                    model,
                    model.friction_coefficient(fractures)
                    - model.tangent_dilation_angle(fractures),
                )
                assert dissipation > 0.0, (
                    f"Dissipation {dissipation} not positive at "
                    f"sigma_n/sigma_T={traction_fraction}, Lambda/Lc^f={exponent}"
                )

    # -- The pole ------------------------------------------------------------------

    def test_pole_is_rejected_at_setup(self):
        """Parameters at the pole raise rather than producing a huge friction bound.

        ``mu*`` diverges as ``mu_b tan psi -> 1``. Since ``tan psi <= tan psi_0``, no
        state reachable during a run is closer to the pole than the parameters are, so
        the check belongs to the parameters and can be made once.

        The raise happens inside ``prepare_simulation``, where ``set_equations`` builds
        the friction bound -- before any solve, which is the point of checking there.
        """
        with pytest.raises(ValueError, match="pole"):
            self._model(friction_coefficient=1.0, dilation_angle=np.arctan(1.0) + 0.01)

    def test_admissible_parameters_are_accepted(self):
        """The guard does not fire for a steep but admissible dilation angle."""
        model = self._model(friction_coefficient=1.0, dilation_angle=np.arctan(0.9))
        self._set_state(model, traction_fraction=0.3)
        assert np.isfinite(
            self._mean(model, model.friction_coefficient(self._fractures(model)))
        )


# ---------------------------------------------------------------------------
# 5.  Mated fracture gap (g_0)
# ---------------------------------------------------------------------------


class TestMatedFractureGap:
    """The mated aperture wears down with the dilation damage state."""

    def test_reference_gap_carries_the_dilation_damage(self):
        """``g_0`` is scaled by ``d^d``, so the mated gap closes towards ``d_0^d g_0``.

        The reference gap is the aperture held by the asperities in the mated
        configuration, so it is subject to the same wear as the dilation angle they
        also set.
        """
        g_0, residual_d = 3.0e-4, 0.4
        model = _prepared_model(
            damages=["dilation", "friction"],
            solid_overrides={
                "fracture_gap": g_0,
                "residual_dilation_damage": residual_d,
            },
        )
        fractures = model.mdg.subdomains(dim=model.nd - 1)
        nc = sum(sd.num_cells for sd in fractures)
        scale_d = _nondimensional_wear_energy_scale(model, "dilation")
        evaluate = model.equation_system.evaluate

        for exponent in (0.0, 1.0, 50.0):
            model.equation_system.set_variable_values(
                exponent * scale_d * np.ones(nc),
                variables=[model.damage_history(fractures)],
                iterate_index=0,
            )
            d_d = residual_d + (1.0 - residual_d) * np.exp(-exponent)
            np.testing.assert_allclose(
                evaluate(model.reference_fracture_gap(fractures)),
                d_d * g_0,
                rtol=1e-12,
            )


# ---------------------------------------------------------------------------
# 6.  Damage length kernel
# ---------------------------------------------------------------------------


class TestDamageLength:
    """Unit tests for the ``damage_length`` operator.

    ``damage_length(subdomains, k)`` returns the tangential slip kernel for the
    *k*-th time step contribution.  For ``k=0`` (the implicit term):

    - **Isotropic**: ``L = |u_t_iterate − u_t_ts0|``
    - **Anisotropic**: ``L = |max(0, m · u_t_ts0) − |u_t_iterate||``,
      where ``m = u_t_iterate / |u_t_iterate|`` and ``u_t_ts0`` is the value
      at ``time_step_index=0``.

    For ``k ≥ 1`` (explicit, past steps):

    - **Isotropic**: ``L = |u_ts{k−1} − u_ts{k}|``
    - **Anisotropic**: ``L = |max(0, m · u_ts{k−1}) − max(0, m · u_ts{k})|``

    Values are injected via ``interface_displacement`` (contact traction is not
    touched; the fracture tangential stiffness sentinel −1 ensures
    ``elastic_tangential = 0``, so ``plastic_jump = displacement_jump``).

    Time-step-to-storage mapping
    ----------------------------
    ``u_t.previous_timestep(k)`` reads from:

    - ``k = 0``: current Newton iterate (``iterate_index=0``)
    - ``k ≥ 1``: ``time_step_index = k − 1``
    """

    @staticmethod
    def _fractures(model):
        return model.mdg.subdomains(dim=model.nd - 1)

    def _set_tangential_jump(
        self,
        model,
        u_tx: float,
        u_tz: float = 0.0,
        *,
        iterate: bool = False,
        time_step_index: int | None = None,
    ) -> None:
        """Inject a tangential plastic displacement jump.

        Sets the positive-side mortar interface displacement cells so that the computed
        tangential displacement jump equals ``(u_tx, u_tz)`` (2D: only ``u_tx`` is used;
        3D: both x and z components) for every fracture cell. Contact traction is zeroed
        at the same depth so that storage for all required variables is allocated and
        the elastic jump evaluates to zero.

        Parameters:
            model: A prepared model instance.
            u_tx: x-component of the tangential jump (applied to all cells).
            u_tz: z-component of the tangential jump (3D only; ignored in 2D).
            iterate: If ``True``, store at the current Newton iterate depth.
            time_step_index: If given, store at this time-step depth index.
        """
        fractures = model.mdg.subdomains(dim=model.nd - 1)
        interfaces = model.mdg.interfaces(dim=model.nd - 1)
        nc = sum(sd.num_cells for sd in fractures)
        ni = sum(intf.num_cells for intf in interfaces)
        nd = model.nd

        u_int = np.zeros(ni * nd)
        # The positive-side mortar cells are the second half of the sorted
        # interface cell list.  Their x-component drives a +1 tangential jump.
        u_int[(ni // 2) * nd :: nd] = u_tx
        if nd == 3:
            # z is the second tangential direction for a y-normal fracture.
            u_int[(ni // 2) * nd + 2 :: nd] = u_tz

        def _setv_int(**kw):
            model.equation_system.set_variable_values(
                u_int,
                variables=[model.interface_displacement(interfaces)],
                **kw,
            )

        def _setv_trac(**kw):
            model.equation_system.set_variable_values(
                np.zeros(nc * nd),
                variables=[model.contact_traction(fractures)],
                **kw,
            )

        if iterate:
            _setv_int(iterate_index=0)
            _setv_trac(iterate_index=0)
        if time_step_index is not None:
            _setv_int(time_step_index=time_step_index)
            _setv_trac(time_step_index=time_step_index)

    # ---- Isotropic and anisotropic agree for aligned steps ----

    @pytest.mark.parametrize(
        "isotropic", [True, False], ids=["isotropic", "anisotropic"]
    )
    def test_length_equals_increment_for_aligned_step(self, isotropic: bool):
        """``L = |u_now − u_prev|`` for a positive forward step (both models agree).

        When the step stays in the positive direction (``u_now > u_prev ≥ 0``), the
        anisotropic direction ``m = +1`` and ``max(0, m·u) = u``, so both formulas
        reduce to ``|u_now − u_prev|``.

        ``u_now`` is injected via the iterate (read by ``previous_timestep(0)``);
        ``u_prev`` via ``time_step_index=0`` (read by ``previous_timestep(1)``).
        """
        model = _prepared_model(isotropic=isotropic, damages=["dilation"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        u_now, u_prev = 2.0e-4, 0.5e-4

        self._set_tangential_jump(model, u_now, iterate=True)
        self._set_tangential_jump(model, u_prev, time_step_index=0)

        L, _ = model.damage_length(fractures, 0)
        np.testing.assert_allclose(
            L.value(model.equation_system),
            abs(u_now - u_prev) * np.ones(nc),
            rtol=1e-12,
        )

    # ---- Isotropic-specific ----

    def test_isotropic_reversal_length_is_twice_forward_length(self):
        """Sign reversal gives twice the forward step length.

        Forward step (0 → +d) and reversal (+d → −d) have the same step size *d*, but
        the reversal spans 2*d.  The ratio ``L_reversal / L_forward = 2`` reflects the
        unsigned cumulative nature of the isotropic formula.
        """
        model = _prepared_model(isotropic=True, damages=["dilation"])
        fractures = self._fractures(model)
        d = 1.5e-4

        # Forward step: u_now = +d, u_prev = 0
        self._set_tangential_jump(model, d, iterate=True)
        self._set_tangential_jump(model, 0.0, time_step_index=0)
        L_fwd, _ = model.damage_length(fractures, 0)
        L_fwd_val = L_fwd.value(model.equation_system)

        # Reversal step: u_now = −d, u_prev = +d
        self._set_tangential_jump(model, -d, iterate=True)
        self._set_tangential_jump(model, d, time_step_index=0)
        L_rev, _ = model.damage_length(fractures, 0)
        L_rev_val = L_rev.value(model.equation_system)

        np.testing.assert_allclose(L_rev_val, 2.0 * L_fwd_val, rtol=1e-12)

    # ---- Anisotropic-specific ----

    def test_anisotropic_length_zero_for_step_opposite_to_current_direction(self):
        """A past step opposite to the current direction contributes zero length.

        This is the key structural difference from the isotropic model.  The anisotropic
        formula evaluates every past step's contribution using the **current** slip
        direction ``m``.  When a past step was in the direction opposite to ``m``, both
        ``max(0, m · u)`` terms vanish and the length is zero.

        Setup (``k = 1``, an explicit past-step contribution):

        - Iterate = ``−1e−4``  → ``m = −1``
        - ``time_step_index=0`` = ``+d``  (past step went 0 → +d, opposite to m)
        - ``time_step_index=1`` = ``0``

        Expected:

        - Isotropic: ``|+d − 0| = d``  (non-zero)
        - Anisotropic: ``|max(0, −d) − max(0, 0)| = 0``
        """
        iso_model = _prepared_model(isotropic=True, damages=["dilation"])
        aniso_model = _prepared_model(isotropic=False, damages=["dilation"])
        d = 2.0e-4

        for m in (iso_model, aniso_model):
            self._set_tangential_jump(m, -1.0e-4, iterate=True)
            self._set_tangential_jump(m, d, time_step_index=0)
            self._set_tangential_jump(m, 0.0, time_step_index=1)

        nc_iso = sum(sd.num_cells for sd in self._fractures(iso_model))

        L_iso, _ = iso_model.damage_length(self._fractures(iso_model), 1)
        L_aniso, _ = aniso_model.damage_length(self._fractures(aniso_model), 1)

        np.testing.assert_allclose(
            L_iso.value(iso_model.equation_system), d * np.ones(nc_iso), rtol=1e-12
        )
        np.testing.assert_allclose(
            L_aniso.value(aniso_model.equation_system),
            np.zeros(nc_iso),
            atol=1e-15,
        )

    @pytest.mark.parametrize(
        "isotropic", [True, False], ids=["isotropic", "anisotropic"]
    )
    def test_length_zero_for_zero_increment(self, isotropic: bool):
        """No net slip → zero damage length for both isotropic and anisotropic."""
        model = _prepared_model(isotropic=isotropic, damages=["dilation"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)

        self._set_tangential_jump(model, 3.0e-4, iterate=True)
        self._set_tangential_jump(model, 3.0e-4, time_step_index=0)

        L, _ = model.damage_length(fractures, 0)
        np.testing.assert_allclose(
            L.value(model.equation_system), np.zeros(nc), atol=1e-15
        )

    # ---- 3D tests ----

    def test_isotropic_3d_norm_of_2d_tangential_increment(self):
        """In 3D ``damage_length`` uses ``l2_norm(2)`` over both tangential components.

        With ``u_t_iterate = (3e-4, 4e-4)`` (x and z) and ``u_t_ts0 = 0``,
        the Euclidean norm gives ``L = 5e-4`` via the Pythagorean triple
        ``3² + 4² = 5²``, exercising ``l2_norm(2)`` instead of the 2D
        ``l2_norm(1)`` (= absolute value).
        """
        model = _prepared_model(isotropic=True, damages=["dilation"], dim=3)
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        u_tx, u_tz = 3.0e-4, 4.0e-4

        self._set_tangential_jump(model, u_tx, u_tz, iterate=True)
        self._set_tangential_jump(model, 0.0, 0.0, time_step_index=0)

        L, _ = model.damage_length(fractures, 0)
        np.testing.assert_allclose(
            L.value(model.equation_system),
            np.sqrt(u_tx**2 + u_tz**2) * np.ones(nc),  # = 5e-4
            rtol=1e-10,
        )

    def test_anisotropic_3d_oblique_direction_dot_product(self):
        """In 3D the anisotropic kernel uses a 2D dot product ``m · u_t``.

        Setup (``time_step_index = 0``, implicit term):

        - iterate = ``(d, d)`` → ``m = (1/√2, 1/√2)`` (diagonal slip direction)
        - ts0 = ``0``

        The formula evaluates::

            L = |max(0, m·iterate) − max(0, m·ts0)|
              = |m·(d, d) − 0|
              = |(d/√2 + d/√2)|
              = d·√2

        This verifies that ``normalized_tangential_plastic_jump`` uses ``l2_norm(2)``
        for normalization and that the projection ``tangential_to_scalar @ (m_t * u_t)``
        performs a genuine 2D inner product in 3D.
        """
        model = _prepared_model(isotropic=False, damages=["dilation"], dim=3)
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        d = 1.0e-4

        # Diagonal iterate → m = (1/√2, 1/√2)
        self._set_tangential_jump(model, d, d, iterate=True)
        self._set_tangential_jump(model, 0.0, 0.0, time_step_index=0)

        L, _ = model.damage_length(fractures, 0)
        # m · (d, d) = d/√2 + d/√2 = d·√2
        np.testing.assert_allclose(
            L.value(model.equation_system),
            d * np.sqrt(2.0) * np.ones(nc),
            rtol=1e-10,
        )

    @pytest.mark.parametrize(
        "isotropic", [True, False], ids=["isotropic", "anisotropic"]
    )
    def test_3d_zero_check_uses_norm_not_component_sum(self, isotropic: bool):
        """The increment measure must not vanish for a cancelling 3D increment.

        ``damage_convolution_integral`` drops a history term when this second return
        value falls below tolerance, so it may vanish only when the increment itself
        does. With ``u_t = (a, -a)`` the tangential components sum to zero while the
        increment is plainly non-zero. A failure here means the check has reverted to
        summing components, which silently discards real slip history in 3D.
        """
        model = _prepared_model(isotropic=isotropic, damages=["dilation"], dim=3)
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        a = 3.0e-4

        self._set_tangential_jump(model, a, -a, iterate=True)
        self._set_tangential_jump(model, 0.0, 0.0, time_step_index=0)

        _, increment_norm = model.damage_length(fractures, 0)
        np.testing.assert_allclose(
            increment_norm.value(model.equation_system),
            a * np.sqrt(2.0) * np.ones(nc),
            rtol=1e-10,
        )
