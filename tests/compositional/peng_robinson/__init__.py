"""Contains some fixtures shared by different testing modules."""

from __future__ import annotations

from threading import Lock

import pytest

import numpy as np
import numba as nb

import porepy as pp
import porepy.compositional.peng_robinson as pr


class PRLBC(pr.CompiledPengRobinson, pr.LBCViscosity):
    """Combined Peng-Robinson EoS and LBC viscosity model, for testing purposes.

    Thermal conductivities are set to 1.0, with zero derivatives.

    """

    def get_conductivity_function(self):
        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def kappa_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return 1.0

        return kappa_c

    def get_conductivity_derivative_function(self):
        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dkappa_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(2 + xn.shape[0], dtype=np.float64)

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
    h_ideal = [pr.h_ideal_H2O, pr.h_ideal_CO2, pr.h_ideal_H2S, pr.h_ideal_N2]

    ncomp = comps_and_phases[0]
    assert ncomp == len(components), "Failure in test setup."

    with _cache_lock:
        eos = PRLBC(
            components=components,
            ideal_enthalpies=h_ideal[:ncomp],
            bip_matrix=bips[:ncomp, :ncomp],
        )
        eos.compile()
        _pr_eos_cache[cache_key] = eos

        # def _clear_cache():
        #     _pr_eos_cache.pop(cache_key, None)

        # request.addfinalizer(_clear_cache)

    return eos
