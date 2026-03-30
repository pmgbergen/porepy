"""Test module for the persistent variable flash, its instantiation and assembly of
residuals and Jacobians for different flash specifications.

"""

from __future__ import annotations

from itertools import product
from threading import Lock

import numpy as np
import pytest

import porepy as pp
import porepy.compositional as pc
import porepy.compositional.flash as pf
import porepy.compositional.peng_robinson as pr
from porepy.applications.test_utils.derivative_testing import (
    assert_order_at_least,
    get_EOC_taylor,
)
from tests.compositional.peng_robinson import components, comps_and_phases, pr_eos

_flash_cache: dict[tuple[str, ...], pr.CompiledPengRobinson] = {}
"""Caching expensive to create flash classes."""
_cache_lock = Lock()
"""Threading lock in case of parallel test execution, to avoid race conditions between
different test processes."""


@pytest.fixture(scope="module")
def flash(
    comps_and_phases: tuple[int, str],
    components: list[pp.compositional.FluidComponent],
    pr_eos: pr.CompiledPengRobinson,
    request,
) -> pf.CompiledPersistentVariableFlash:
    """Flash instance for indicated components and requested EoS."""
    ncomp = comps_and_phases[0]
    p = comps_and_phases[1]
    nphase = len(p)
    nliq = p.count("L")
    ngas = p.count("V")
    assert ngas <= 1
    assert len(components) == ncomp

    eos: pp.compositional.EquationOfState
    if request.param == "PR":
        eos = pr_eos
    else:
        raise ValueError(f"Flash fixture not covering EoS request: {request.param}")

    phases: list[pp.Phase] = []
    if ngas:
        phases.append(pp.Phase(pp.compositional.PhysicalState.gas, "V", eos))

    for i in range(nliq):
        phases.append(pp.Phase(pp.compositional.PhysicalState.liquid, f"L{i}", eos))

    assert len(phases) == nphase
    for p in phases:
        p.components = components

    fluid = pp.Fluid(components, phases)

    # Default initializer supports only 2-phase mixtures.
    class DummyInitializer(pf.FlashInitializer):
        def __init__(self, fluid, params=None):
            pass

        def __getitem__(self, key):
            return lambda x: x

        def compile(self, *args):
            pass

    cache_key = tuple(
        str(c) for c in [comps_and_phases[0], comps_and_phases[1], request.param]
    )

    with _cache_lock:
        if cache_key in _flash_cache:
            fl = _flash_cache[cache_key]
        else:
            fl = pf.CompiledPersistentVariableFlash(
                fluid, params={"initializer": DummyInitializer}
            )
            fl.compile()
            _flash_cache[cache_key] = fl

    return fl


@pytest.mark.skipped(reason="slow due to compilation.")
@pytest.mark.xfail(raises=pp.compositional.CompositionalModellingError)
@pytest.mark.parametrize("comps_and_phases", [(1, "L"), (2, "V")], indirect=True)
@pytest.mark.parametrize("flash", ["PR"], indirect=True)
def test_error_when_flashing_with_one_phase(
    flash: pf.CompiledPersistentVariableFlash,
    comps_and_phases: tuple[int, str],
) -> None:
    """Testing that the flash class should raise an error if the modeles assumes only
    1 phase.

    Failures occurres in fixture fetching.

    """
    assert False, "Fixture fetching should fail with CompositionalModellingError."


def _get_base_dim(cp: tuple[int, str], spec: pc.FlashSpec) -> int:
    """Calculates the base dimension of a flash system based on numbers of components,
    phases and flash specification"""
    ncomp = cp[0]
    nphase = len(cp[1])
    # Phase fractions and partial fractions
    base_dim = ncomp * nphase + nphase - 1

    if spec >= pc.FlashSpec.vT:  # Pressure variable.
        base_dim += 1

    if spec not in (pc.FlashSpec.pT, pc.FlashSpec.vT):  # Temperature variable.
        base_dim += 1

    return base_dim


def _dh_from_cp(
    cp: tuple[int, str], spec: pc.FlashSpec
) -> list[tuple[np.ndarray, np.ndarray]]:
    ncomp = cp[0]
    nphase = len(cp[1])
    dim_gen_arg = pf.dim_gen_arg(ncomp, nphase, spec)
    # Base dimension covers phase fractions and extended partial fractions.
    dim_base = _get_base_dim(cp, spec)
    directions = np.hstack(
        (np.zeros((dim_base, dim_gen_arg - dim_base)), np.eye(dim_base))
    )
    h_fractions = np.logspace(0, -6, 7)
    h_p = np.logspace(3, -3, 7)
    h_T = np.logspace(2, -4, 7)

    h_all: list[np.ndarray] = [h_fractions] * (nphase - 1 + ncomp * nphase)
    if spec not in (pc.FlashSpec.pT, pc.FlashSpec.vT):
        h_all = [h_T] + h_all
    if spec >= pc.FlashSpec.vT:
        h_all = [h_p] + h_all

    return [(d, h) for d, h in zip(directions, h_all)]


@pytest.mark.skipped(reason="slow due to compilation.")
@pytest.mark.parametrize(
    ["comps_and_phases", "flash_spec", "d", "h"],
    [
        (cp, spec, d, h)
        for cp, spec in product(
            [
                (1, "VL"),
                (2, "VL"),
                (2, "VLL"),
                (2, "LL"),
                (3, "VL"),
                (3, "VLLL"),
                (3, "LL"),
                (3, "LLL"),
            ],
            pf.CompiledPersistentVariableFlash.SUPPORTED_SPECIFICATIONS,
        )
        for d, h in _dh_from_cp(cp, spec)
    ],
    indirect=["comps_and_phases"],
)
@pytest.mark.parametrize("flash", ["PR"], indirect=True)
def test_assembly_of_flash_systems(
    flash: pf.CompiledPersistentVariableFlash,
    comps_and_phases: tuple[int, str],
    d: np.ndarray,
    h: np.ndarray,
    flash_spec: pc.FlashSpec,
) -> None:
    """Tests the assembly of flash systems:

    1. Availability after compilation
    2. Signature (callable with generic arg)
    3. Jacobian and residual of expected size.
    4. Jacobian approximates the system properly (Taylor expansion close to 2nd order)

    """
    ncomp = comps_and_phases[0]
    nphase = len(comps_and_phases[1])
    dim_gen_arg = pf.dim_gen_arg(ncomp, nphase, flash_spec)

    # Base dimension covers phase fractions and extended partial fractions.
    dim_base = _get_base_dim(comps_and_phases, flash_spec)

    assert flash._eos.nc == ncomp, "Failure in test setup."
    assert flash._eos.is_compiled, "EoS not compiled."

    # If flash not available, this will raise a key error. Error in test setup.
    res = flash.residuals[flash_spec]
    jac = flash.jacobians[flash_spec]

    # Assume state to be in an area of the domain where the residual is smooth, despite
    # complementary conditions
    z = np.ones(ncomp) / ncomp
    # NOTE: scale with 0.9 to avoid linear dependent complementarity conditions
    y = np.ones(nphase) / nphase * 0.9
    x = np.ones((nphase, ncomp)) / ncomp * 0.9
    p = 1e7
    T = 400.0
    # Isochoric of isobaric.
    if flash_spec >= pc.FlashSpec.vT:
        state1 = 1e-5
    else:
        state1 = p
    # Can only be energetic, if relevant at all.
    state2 = 3e4

    x0 = pf.assemble_generic_arg(x, y, z, p, T, state1, state2, np.zeros(0), flash_spec)

    def func(x):
        r = res(x)
        assert r.shape == (dim_base,)
        return r

    def dfunc(x):
        j = jac(x)
        assert j.shape == (dim_base, dim_base)
        return np.hstack((np.zeros((dim_base, dim_gen_arg - dim_base)), j))

    orders = get_EOC_taylor(func, dfunc, x0.copy(), d, h)
    assert_order_at_least(
        orders,
        2.0,
        tol=2e-2,
        asymptotic=3,
        err_msg=f"({flash_spec.name}, {comps_and_phases}, {d})",
    )
