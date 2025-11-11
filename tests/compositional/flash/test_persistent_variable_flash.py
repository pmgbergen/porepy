"""Test module for the persistent variable flash, its instantiation and assembly of
residuals and Jacobians for different flash specifications.

"""

from __future__ import annotations

import numpy as np
import pytest

# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

import porepy as pp
import porepy.compositional.flash as pf
import porepy.compositional.peng_robinson as pr

from tests.compositional.peng_robinson.test_cubic_polynomial import (
    get_EOC_taylor,
    assert_order_at_least,
)


@pytest.fixture(scope="module")
def comps_and_phases(request) -> tuple[int, str]:
    """Indirect flash parametrization for fixing number of components and phases
    (and their type)."""
    return request.param


@pytest.fixture(scope="module")
def components(
    comps_and_phases: tuple[int, str],
) -> list[pp.compositional.FluidComponent]:
    """Fluid components on which the flash was tested."""
    h2o = pp.compositional.FluidComponent(
        name="H2O",
        acentric_factor=0.3443,
        critical_pressure=22064000.0,
        critical_specific_volume=5.59480372671e-05,
        critical_temperature=647.096,
        molar_mass=0.01801528,
    )
    co2 = pp.compositional.FluidComponent(
        name="CO2",
        acentric_factor=0.22394,
        critical_pressure=7377300.0,
        critical_specific_volume=9.41184770731e-05,
        critical_temperature=304.1282,
        molar_mass=0.04400950000000001,
    )
    h2s = pp.compositional.FluidComponent(
        name="H2S",
        acentric_factor=0.1005,
        critical_pressure=9000000.0,
        critical_specific_volume=9.81354268891e-05,
        critical_temperature=373.1,
        molar_mass=0.03408088,
    )
    # n2 = pp.compositional.FluidComponent(
    #     name="N2",
    #     acentric_factor=0.0372,
    #     critical_pressure=3395800.0,
    #     critical_specific_volume=8.94142472662e-05,
    #     critical_temperature=126.192,
    #     molar_mass=0.0280134,
    # )

    comps = [h2o, co2, h2s]
    ncomp = comps_and_phases[0]
    assert ncomp > 0
    assert ncomp <= len(comps)
    return comps[:ncomp]


@pytest.fixture(scope="module")
def pr_eos(
    components: list[pp.compositional.FluidComponent], comps_and_phases: tuple[int, str]
) -> pr.CompiledPengRobinson:
    """Peng-Robinson EoS, un-compiled for the test case."""
    bips = np.array(
        [
            [0.0, 0.0394, 0.0952, 0.0],
            [0.0394, 0.0, 0.0967, 0.1652],
            [0.0952, 0.0967, 0.0, -0.0122],
            [0.0, 0.1652, -0.0122, 0.0],
        ],
    )
    h_ideal = [pr.h_ideal_H2O, pr.h_ideal_CO2, pr.h_ideal_H2S, pr.h_ideal_N2]

    ncomp = comps_and_phases[0]
    assert ncomp == len(components)

    eos = pr.CompiledPengRobinson(
        components=components,
        ideal_enthalpies=h_ideal[:ncomp],
        bip_matrix=bips[:ncomp, :ncomp],
    )
    return eos


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

    # Default initializer supports only 2-phase mixtures.
    class DummyInitializer(pf.FlashInitializer):
        def __init__(self, fluid, params=None):
            pass

        def compile(self, *args):
            pass

    fluid = pp.Fluid(components, phases)
    fl = pf.CompiledPersistentVariableFlash(fluid, {"initializer": DummyInitializer})
    return fl


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


@pytest.mark.parametrize(
    "flash_spec", [pf.FlashSpec.pT, pf.FlashSpec.ph, pf.FlashSpec.vh]
)
@pytest.mark.parametrize(
    "comps_and_phases",
    [(1, "VL"), (2, "VL"), (2, "VLL"), (3, "VL"), (3, "VLLL")],
    indirect=True,
)
@pytest.mark.parametrize("flash", ["PR"], indirect=True)
def test_assembly_of_flash_systems(
    flash: pf.CompiledPersistentVariableFlash,
    comps_and_phases: tuple[int, str],
    flash_spec: pf.FlashSpec,
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
    base_dim = ncomp * nphase + nphase - 1

    match flash_spec:
        case pf.FlashSpec.pT | pf.FlashSpec.vT:
            pass
        case pf.FlashSpec.ph:
            base_dim += 1
        case pf.FlashSpec.vh | pf.FlashSpec.vu:
            base_dim += 2 + nphase - 1
        case _:
            assert False, "Uncovered flash specification in test."

    # Directions for Taylor test.
    directions = np.hstack(
        (np.zeros((base_dim, dim_gen_arg - base_dim)), np.eye(base_dim))
    )

    # This takes some time, but should not fail.
    flash.compile(flash_spec)
    # If flash not available, this will raise an Key error.
    res = flash.residuals[flash_spec]
    jac = flash.jacobians[flash_spec]

    # Assume state to be in an area of the domain where the residual is smooth, despite
    # complementary conditions
    z = np.ones(ncomp) / ncomp
    y = np.ones(nphase) / nphase * 0.5
    sat = y.copy()
    x = np.ones((nphase, ncomp)) / ncomp * 0.5
    p = 1e7
    T = 400.0
    # Isochoric of isobaric.
    if flash_spec >= pf.FlashSpec.vT:
        state1 = 1e-5
    else:
        state1 = p
    # Can only be energetic, if relevant at all.
    state2 = -3e4

    x0 = pf.assemble_generic_arg(
        sat, x, y, z, p, T, state1, state2, np.zeros(0), flash_spec
    )
    h = np.logspace(-2, -10, 8)

    def func(*x):
        xg = np.array(x)
        r = res(xg)
        assert r.shape == (base_dim,)
        return r

    def dfunc(*x):
        xg = np.array(x)
        j = jac(xg)
        j.shape == (base_dim, base_dim)
        return np.hstack((np.zeros((base_dim, dim_gen_arg - base_dim)), j))

    for d in directions:
        orders = get_EOC_taylor(func, dfunc, x0.copy(), d, h)
        assert_order_at_least(orders, 2.0, tol=2e-2, asymptotic=6)
