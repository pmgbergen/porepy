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
    ``d = d0 + (1 - d0) * exp(-clip(Lambda, 0, 10))``
- Normalized traction:
    ``(-t_n_nondim) / (0.2 * UCS / char_traction)``
- Friction damage evolution coefficient:
    ``3 * normalized_traction / roughness``
- Dilation damage evolution coefficient:
    ``log(UCS / (char_traction * pos_normal)) * normalized_traction / roughness``
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
from porepy.examples import fracture_damage as damage_example

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepared_model(
    isotropic: bool = True,
    damages: list[str] | None = None,
    dim: int = 2,
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

    model = model_class(params)
    model.prepare_simulation()
    return model


# ---------------------------------------------------------------------------
# 1.  Damage state formula: d = d0 + (1 - d0) * exp(-Lambda)
# ---------------------------------------------------------------------------


class TestDamageStateFormula:
    """Algebraic formula ``d = d0 + (1 - d0) * exp(-Lambda)``.

    The AD implementation clips Lambda to ``[0, 10]`` before exponentiating. The tests
    use values strictly inside ``(0, 10)`` so that the clip does not change the input
    and the formula reduces to the unclipped expression.
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

    @pytest.mark.parametrize("damage", ["dilation", "friction"])
    @pytest.mark.parametrize("lambda_val", [0.0, 0.5, 1.0, 3.0])
    def test_damage_state_matches_formula(self, damage: str, lambda_val: float):
        """Damage state evaluates to ``d0 + (1-d0)*exp(-Lambda)`` for both types.

        Parameters:
            damage: Damage type, either ``"dilation"`` or ``"friction"``.
            lambda_val: Damage history value prescribed to all fracture cells.
        """
        model, fractures, nc = self._prepared_model_with_fractures(damages=[damage])
        d0 = float(getattr(model.solid, f"residual_{damage}_damage"))

        model.equation_system.set_variable_values(
            lambda_val * np.ones(nc),
            variables=[getattr(model, f"{damage}_damage_history")(fractures)],
            iterate_index=0,
        )

        d = getattr(model, f"{damage}_damage_state")(fractures).value(
            model.equation_system
        )
        expected = d0 + (1.0 - d0) * np.exp(-lambda_val)
        np.testing.assert_allclose(d, expected * np.ones(nc), rtol=1e-12)

    @pytest.mark.parametrize("damage", ["dilation", "friction"])
    def test_damage_state_is_one_for_zero_history(self, damage: str):
        """Lambda = 0 means no accumulated damage: state must equal 1."""
        model, fractures, nc = self._prepared_model_with_fractures(damages=[damage])

        model.equation_system.set_variable_values(
            np.zeros(nc),
            variables=[getattr(model, f"{damage}_damage_history")(fractures)],
            iterate_index=0,
        )
        d = getattr(model, f"{damage}_damage_state")(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(d, np.ones(nc), rtol=1e-12)

    def test_dilation_damage_approaches_d0_at_large_history(self):
        """Lambda = 10 (clip maximum) drives the damage state to d0."""
        model, fractures, nc = self._prepared_model_with_fractures(damages=["dilation"])
        d0 = float(model.solid.residual_dilation_damage)

        # The clip is at 10, so exp(-10) ≈ 4.5e-5 is the residual offset.
        model.equation_system.set_variable_values(
            10.0 * np.ones(nc),
            variables=[model.dilation_damage_history(fractures)],
            iterate_index=0,
        )
        d = model.residual_dilation_damage(fractures).value(model.equation_system)
        expected = d0 + (1.0 - d0) * np.exp(-10.0)
        np.testing.assert_allclose(d, expected * np.ones(nc), rtol=1e-4)

    def test_damage_state_is_monotone_in_history(self):
        """A larger history value produces a smaller (more damaged) state."""
        model, fractures, nc = self._prepared_model_with_fractures(damages=["dilation"])

        def _eval(lam: float) -> float:
            model.equation_system.set_variable_values(
                lam * np.ones(nc),
                variables=[model.dilation_damage_history(fractures)],
                iterate_index=0,
            )
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
    """Tests for the damage evolution coefficient formulas and normalized traction.

    All tests prescribe the contact traction variable to a known nondimensional normal
    value (zero tangential component) and compare the evaluated operator with the
    analytically computed reference.

    The key material parameters (from the default example solid constants) are:

    - ``UCS = 1e8 Pa``
    - ``roughness = 1e-4 m``
    - ``char_traction = numerical.characteristic_contact_traction = 1.0 Pa`` (the
      default in ``NumericalConstants``)

    The formulas are written in terms of nondimensional tractions because the contact
    traction variable is nondimensionalized by ``char_traction``.
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
        """Return (char_traction, UCS, roughness) as floats.

        ``char_traction`` is obtained by evaluating the operator, which uses the Young's
        modulus and characteristic displacement (not the scalar stored in
        ``numerical.characteristic_contact_traction``).
        """
        fractures = self._fractures(model)
        char_t = float(
            np.mean(
                model.characteristic_contact_traction(fractures).value(
                    model.equation_system
                )
            )
        )
        ucs = float(model.solid.uniaxial_compressive_strength)
        roughness = float(model.solid.characteristic_fracture_roughness)
        return char_t, ucs, roughness

    def test_normalized_traction_at_transitional_strength(self):
        """At the transitional normal strength the normalized traction equals 1.

        The transitional strength is ``0.2 * UCS``.  Setting ``t_n_nondim = -0.2 * UCS /
        char_traction`` places the traction exactly at this level, so the normalized
        traction should equal 1.
        """
        model = _prepared_model(damages=["dilation"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        char_t, ucs, _ = self._material_constants(model)

        self._set_normal_traction(model, -0.2 * ucs / char_t)

        result = model.normalized_traction_for_damage(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(result, np.ones(nc), rtol=1e-10)

    def test_normalized_traction_scales_linearly_with_traction(self):
        """Doubling the compressive traction doubles the normalized traction."""
        model = _prepared_model(damages=["dilation"])
        fractures = self._fractures(model)
        char_t, ucs, _ = self._material_constants(model)
        base_t = -0.2 * ucs / char_t  # normalized = 1 at this level

        def _eval(factor: float) -> np.ndarray:
            self._set_normal_traction(model, factor * base_t)
            return model.normalized_traction_for_damage(fractures).value(
                model.equation_system
            )

        np.testing.assert_allclose(_eval(2.0), 2.0 * _eval(1.0), rtol=1e-10)

    def test_friction_damage_evolution_coefficient_formula(self):
        """Friction coefficient equals ``3 * normalized_traction / roughness``.

        At a traction of ``0.4 * UCS / char_traction`` (twice the transitional strength)
        the normalized traction is 2, so the coefficient should equal ``3 * 2 /
        roughness = 6 / roughness``.
        """
        model = _prepared_model(damages=["friction"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        char_t, ucs, roughness = self._material_constants(model)

        # normalized_traction = 2
        self._set_normal_traction(model, -0.4 * ucs / char_t)
        normalized = 2.0
        expected = 3.0 * normalized / roughness

        result = model.friction_damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(result, expected * np.ones(nc), rtol=1e-10)

    def test_friction_coefficient_is_negligible_at_zero_traction(self):
        """Zero normal traction (open fracture) → negligible friction coefficient.

        The positive-normal-traction helper clips the contact traction to a maximum of
        ``-1e-15`` (nondim) before negating, so a traction of zero produces ``pos_normal
        = 1e-15`` rather than exactly zero.  The resulting friction coefficient is then:

        .. math::

            k_{f} = \\frac{3 \\cdot 10^{-15}}{0.2 \\, UCS / t_{char} \\cdot
                \\text{roughness}}

        which is many orders of magnitude smaller than any physically relevant value.
        """
        model = _prepared_model(damages=["friction"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        char_t, ucs, roughness = self._material_constants(model)

        self._set_normal_traction(model, 0.0)

        clip_floor = 1e-15  # pos_normal_nondim produced by the clip
        transitional_nondim = 0.2 * ucs / char_t
        expected_clip_value = 3.0 * (clip_floor / transitional_nondim) / roughness

        result = model.friction_damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(result, expected_clip_value * np.ones(nc), rtol=1e-6)

    def test_dilation_damage_evolution_coefficient_formula(self):
        """Dilation coefficient equals ``K_ad * normalized_traction / roughness``.

        At a traction of ``0.4 * UCS / char_traction`` the parameters are::

            pos_normal = 0.4 * UCS / char_traction
            K_ad = log(UCS / (char_traction * pos_normal)) = log(1 / 0.4)
            normalized_traction = 2.0
            expected = K_ad * 2.0 / roughness
        """
        model = _prepared_model(damages=["dilation"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        char_t, ucs, roughness = self._material_constants(model)

        t_n_nondim = -0.4 * ucs / char_t
        self._set_normal_traction(model, t_n_nondim)

        pos_normal_nondim = -t_n_nondim  # = 0.4 * ucs / char_t
        dimensionless_strength = ucs / char_t
        K_ad = np.log(dimensionless_strength / pos_normal_nondim)
        normalized = 2.0
        expected = K_ad * normalized / roughness

        result = model.dilation_damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(result, expected * np.ones(nc), rtol=1e-10)

    def test_dilation_damage_evolution_coefficient_at_ucs(self):
        """Dilation coefficient diverges (K_ad → 0) as traction approaches UCS.

        At ``t_n_nondim = -UCS / char_traction`` the positive normal traction equals
        ``UCS / char_traction``, so ``K_ad = log(1) = 0`` and the coefficient is zero.
        """
        model = _prepared_model(damages=["dilation"])
        fractures = self._fractures(model)
        nc = sum(sd.num_cells for sd in fractures)
        char_t, ucs, _ = self._material_constants(model)

        self._set_normal_traction(model, -ucs / char_t)

        result = model.dilation_damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        np.testing.assert_allclose(result, np.zeros(nc), atol=1e-6)

    def test_dilation_and_friction_coefficients_have_same_sign(self):
        """Both coefficients are non-negative under compression."""
        model = _prepared_model(damages=["dilation", "friction"])
        fractures = self._fractures(model)
        char_t, ucs, _ = self._material_constants(model)

        # Traction well inside the valid range (0.1 * UCS, below UCS)
        self._set_normal_traction(model, -0.1 * ucs / char_t)

        c_dil = model.dilation_damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        c_fri = model.friction_damage_evolution_coefficient(fractures).value(
            model.equation_system
        )
        assert np.all(c_dil >= 0), "Dilation coefficient must be non-negative"
        assert np.all(c_fri >= 0), "Friction coefficient must be non-negative"


# ---------------------------------------------------------------------------
# 3.  Damage length kernel
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
    def test_3d_zero_check_does_not_trigger_for_cancelling_increment(
        self, isotropic: bool
    ):
        """The increment measure must not vanish for a cancelling 3D increment.

        ``damage_convolution_integral`` drops a history term when this second return
        value falls below tolerance, so it may vanish only when the increment itself
        does. With ``u_t = (a, -a)`` and (0, 0) at the previous time step, the
        tangential components sum to zero while the increment is plainly non-zero.
        """
        add_contribution = self._compute_add_contribution(
            time_val=0.0, isotropic=isotropic
        )
        assert add_contribution, (
            "Increment norm must not vanish for cancelling 3D increment"
        )

    @pytest.mark.parametrize(
        "isotropic", [True, False], ids=["isotropic", "anisotropic"]
    )
    def test_3d_zero_check_triggers_for_zero_increment(self, isotropic: bool):
        """The increment norm must vanish when the tangential jump is unchanged.

        The second return value of ``damage_length`` is used to determine whether a
        history term contributes to the convolution integral.  It must vanish only
        when the tangential jump is unchanged, which is tested here in 3D.
        """
        # Same time_val as used for iterate in the helper implies a zero increment (no
        # change from previous time step).
        add_contribution = self._compute_add_contribution(
            time_val=3.0e-4, isotropic=isotropic
        )
        assert not add_contribution, (
            "Increment norm must vanish for unchanged tangential jump."
        )

    def _compute_add_contribution(self, time_val: float, isotropic: bool) -> bool:
        """Return whether the increment norm is non-zero for a 3D tangential jump."""
        model = _prepared_model(isotropic=isotropic, damages=["dilation"], dim=3)
        fractures = self._fractures(model)
        a = 3.0e-4

        self._set_tangential_jump(model, a, -a, iterate=True)
        self._set_tangential_jump(model, time_val, -time_val, time_step_index=0)
        coeff = model.dilation_damage_evolution_coefficient(
            fractures
        ).previous_timestep(0)
        _, increment_norm = model.damage_length(fractures, 0)
        return model._check_constant_contribution(increment_norm * coeff)
