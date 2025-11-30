"""Contains some fixtures shared by different testing modules."""

from __future__ import annotations

from threading import Lock
from typing import Literal

import numpy as np
import pytest

import porepy as pp
import porepy.compositional.peng_robinson as pr
from porepy.compositional._numba_interface import njit
from porepy.compositional.compiled_eos import (
    PROPERTY_DERIVATIVE_FUNC_SIGNATURE,
    PROPERTY_FUNC_SIGNATURE,
)

_COMPILER = njit
"""Decorator for compiling functions in this module.

Uses :func:`~porepy.compositional._numba_interface.njit`.

"""


def calculate_expected_order(
    gaslike: bool,
    tol: float,
    /,
    smooth_sc: float = 0.0,
    smooth3: float = 0.0,
    AB: tuple[float, float] | np.ndarray | None = None,
    pTx: tuple[float, float, np.ndarray] | tuple[float, float] | None = None,
    eos: pr.CompiledPengRobinson | None = None,
) -> Literal[1, 2]:
    """Calculates the expected order of approximation of the Taylor expansion.

    By default we expect order 2. But if the compressibility factor is extended in the
    super-critical area, and smoothing is applied, the expected order drops to 1.

    Also, if smoothing is applied in the sub-critical area with 3 roots, the expected
    order is 1.

    Parameters:
        gaslike: True if gaslike root, False if liquid-like root.
        tol: Tolerance for root case detection.
        smooth3: Smoothing factor in the sub-critical 3-root area.
        smooth_sc: Smoothing factor in the super-critical extension case.
        AB: If given, determines the extension case based on the cohesion and covolume
            pair.
        pTx: If AB is not given, but this, determines AB based on pressure, temperature
            and partial fractions (if not pure fluid)
        eos: The equation of state instance must given in case pTx is given.

    Returns:
        The expected order, which is either 2 or 1.

    """
    expected_order = 2

    if AB is not None:
        A = AB[0]
        B = AB[1]
    elif pTx is not None:
        assert eos is not None, "Need EoS in case pTx is used for order determination."
        p = pTx[0]
        T = pTx[1]
        if len(pTx) == 3:
            xn = pTx[2]
        else:
            xn = np.ones(1)
        RT = pp.compositional.THD_REF.R_U * T
        a = pr.a_VdW(T, xn, eos.Tcs, eos.omegas, eos.acs, eos.bips)
        b = np.dot(xn, eos.bcs)
        A = a * p / (RT * RT)
        B = b * p / RT

    ec = pr.is_extended_factor(A, B, gaslike, tol)

    # Order loss for super-critical smoothing because derivatives of smoothing weights
    # are not computed.
    if ec >= 10 and smooth_sc > 0.0:
        expected_order = 1

    # Order loss for extended super-critical root near the zero-cohesion limit.
    if 10 <= ec < 20 and not gaslike and np.abs(A) < 1e-5:
        expected_order = 1

    # Order loss because of enforcement of lower B bound for liquid root.
    if B <= pr.COVOLUME_LIMIT and not gaslike:
        expected_order = 1

    # Order loss because of liquid-saturated curve intersecting with covolume limit.
    if B <= pr.COVOLUME_LIMIT and gaslike and A >= 0.25:
        expected_order = 1

    # Order loss due to limit case (0,0) is interaction of all root cases and Z becomes
    # non-smooth.
    if np.linalg.norm((A, B)) <= 1e-5:
        expected_order = 1

    # Order loss for sub-critical 3-root smoothing because derivatives of smoothing
    # weights are not computed.
    if (
        pr.get_root_case(pr.c_from_AB(A, B), tol) == 3
        and smooth3 > 0
        and not pr.is_supercritical(A, B)
    ):
        expected_order = 1

    return expected_order


class PRLBC(pr.CompiledPengRobinson, pr.LBCViscosity):
    """Combined Peng-Robinson EoS and LBC viscosity model, for testing purposes.

    Thermal conductivities are set to 1.0, with zero derivatives.

    """

    def get_kappa_function(self):
        @_COMPILER(PROPERTY_FUNC_SIGNATURE)
        def kappa_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return 1.0

        return kappa_c

    def get_grad_kappa_function(self):
        @_COMPILER(PROPERTY_DERIVATIVE_FUNC_SIGNATURE)
        def dkappa_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            if xn.size > 1:
                return np.zeros(2 + xn.shape[0])
            else:
                return np.zeros(2)

        return dkappa_c


@pytest.fixture(scope="session")
def comps_and_phases(request) -> tuple[int, str]:
    """Indirect flash parametrization for fixing number of components and phases
    (and their type)."""
    return request.param


@pytest.fixture(scope="session")
def components(
    comps_and_phases: tuple[int, str], request
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
    assert ncomp > 0, "Must request at least one component."
    assert ncomp <= len(comps), f"Can only request {len(comps)} components max."
    return comps[:ncomp]


_pr_eos_cache: dict[tuple[str, ...], pr.CompiledPengRobinson] = {}
"""Caching expensive to create Peng-Robinson EoS instances."""
_cache_lock = Lock()
"""Threading lock in case of parallel test execution, to avoid race conditions between
different test processes."""


@pytest.fixture(scope="session")
def pr_eos(
    components: list[pp.compositional.FluidComponent],
    comps_and_phases: tuple[int, str],
    request,
) -> pr.CompiledPengRobinson:
    """Peng-Robinson + LBC viscosity EoS, compiled and cached for each component
    configuration for all tests in a session."""

    cache_key = tuple(c.name for c in components)
    if cache_key in _pr_eos_cache:
        return _pr_eos_cache[cache_key]

    bips = np.array(
        [
            [0.0, 0.0394, 0.0952, 0.0],
            [0.0394, 0.0, 0.0967, 0.1652],
            [0.0952, 0.0967, 0.0, -0.0122],
            [0.0, 0.1652, -0.0122, 0.0],
        ],
    )
    ideal_fluids = [
        pp.compositional.ideal.IdealH2O,
        pp.compositional.ideal.IdealCO2,
        pp.compositional.ideal.IdealH2S,
        pp.compositional.ideal.IdealN2,
    ]

    ncomp = comps_and_phases[0]
    assert ncomp == len(components), "Failure in test setup."

    with _cache_lock:
        eos = PRLBC(
            components=components,
            ideal_fluids=ideal_fluids[:ncomp],
            bip_matrix=bips[:ncomp, :ncomp],
        )
        eos.compile()
        _pr_eos_cache[cache_key] = eos

    return eos
