"""Tests the dataclass implementations for storing fluid properties."""

import random
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from numpy.typing import NDArray

from porepy.compositional.states import (
    ExtensiveProperties,
    FluidProperties,
    IntensiveProperties,
    PhaseProperties,
)
from porepy.compositional.utils import PhysicalState

# Attributes of the dataclasses to be tested.
_INTENSIVE_ATTRIBUTS: list[str] = ["p", "T", "z"]
_EXTENSIVE_ATTRIBUTS: list[str] = ["h", "u", "rho"]
_PHASE_ATTRIBUTES: list[str] = [
    "x",
    "dh",
    "du",
    "drho",
    "phis",
    "dphis",
    "mu",
    "dmu",
    "kappa",
    "dkappa",
]
_FLUID_ATTRIBUTES: list[str] = ["y", "sat"]


def atleast_2d(arr: np.ndarray) -> np.ndarray:
    """Ensures proper transposition of 1D arrays to 2D arrays, while leaving 2D arrays
    unchanged."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array.")


def get_random_props(
    num_vals: int, num_comp: int, num_phase: int, num_dep: int
) -> FluidProperties:
    """Create fluid properties filled with random values."""

    possible_states = [v for v in PhysicalState]

    base_intensive = IntensiveProperties(
        p=np.random.rand(num_vals),
        T=np.random.rand(num_vals),
        z=np.random.rand(num_comp, num_vals),
    )

    base_extensive = ExtensiveProperties(
        h=np.random.rand(num_vals),
        u=np.random.rand(num_vals),
        rho=np.random.rand(num_vals),
    )

    return FluidProperties(
        **asdict(deepcopy(base_intensive)),
        **asdict(deepcopy(base_extensive)),
        y=np.random.rand(num_phase, num_vals),
        sat=np.random.rand(num_phase, num_vals),
        phases=[
            PhaseProperties(
                **asdict(deepcopy(base_extensive)),
                state=random.choice(possible_states),
                x=np.random.rand(num_comp, num_vals),
                phis=np.random.rand(num_comp, num_vals),
                dh=np.random.rand(num_dep, num_vals),
                du=np.random.rand(num_dep, num_vals),
                drho=np.random.rand(num_dep, num_vals),
                dphis=np.random.rand(num_comp, num_dep, num_vals),
                mu=np.random.rand(num_vals),
                dmu=np.random.rand(num_dep, num_vals),
                kappa=np.random.rand(num_vals),
                dkappa=np.random.rand(num_dep, num_vals),
            )
            for _ in range(num_phase)
        ],
    )


@pytest.mark.parametrize(
    "num_vals, key",
    [
        # NOTE Numpy dropped support to set array elements (integer-indexing) with
        # arrays of shape (1,).
        # (1, 0),
        # (2, 0),
        # (6, 0),
        (1, np.array([True])),
        (1, np.array([0])),
        (2, slice(0, 1)),
        (2, slice(0, 2)),
        (2, slice(1, 2)),
        (2, np.array([True, False])),
        (2, np.array([False, True])),
        (2, np.array([True, True])),
        (2, np.array([0])),
        (2, np.array([0, 1])),
        (6, slice(0, 3)),
        (6, slice(0, 6, 1)),
        (6, slice(0, 6, 2)),
        (6, np.array([True, False, True, False, True, False])),
        (6, np.array([5])),
        (6, np.array([0, 2, 4])),
    ],
)
@pytest.mark.parametrize("num_comp", [1, 2])
@pytest.mark.parametrize("num_phase", [1, 2])
@pytest.mark.parametrize("num_dep", [1, 2])
def test_setter_getter_fluidproperties(
    key: int | slice | NDArray[np.int_] | NDArray[np.bool_],
    num_vals: int,
    num_comp: int,
    num_phase: int,
    num_dep: int,
) -> None:
    """Tests the setitem and getitem overloads of the fluid-property dataclass.

    This test suffices to indirectly test also the lower dataclasses such as
    phase-properties, intensive and extensive states.
    """

    base = get_random_props(num_vals, num_comp, num_phase, num_dep)
    reference = deepcopy(base)

    if isinstance(key, int):
        expected_size = 1
    elif isinstance(key, slice):
        expected_size = len(range(*key.indices(num_vals)))
    elif isinstance(key, np.ndarray):
        if key.dtype == bool:
            expected_size = int(np.sum(key))
        elif key.dtype == int and key.ndim == 1:
            expected_size = len(key)
        else:
            assert False, "Invalid key array."
    else:
        assert False, "Invalid key type."

    # Checking the substate and whether it has the right values.
    sub = base[key]

    for a in _FLUID_ATTRIBUTES + _INTENSIVE_ATTRIBUTS + _EXTENSIVE_ATTRIBUTS:
        sub_a: np.ndarray = getattr(sub, a)
        ref_a: np.ndarray

        err_msg = f"Unexpected shape for {a} ({sub_a.shape})"

        # Catch fractional quantities before others due to different shape
        if a == "z":
            assert sub_a.shape == (num_comp, expected_size), err_msg
            ref_a = atleast_2d(getattr(reference, a)[:, key])
        elif a in ["y", "sat"]:
            assert sub_a.shape == (num_phase, expected_size), err_msg
            ref_a = atleast_2d(getattr(reference, a)[:, key])
        elif a in _INTENSIVE_ATTRIBUTS + _EXTENSIVE_ATTRIBUTS:
            assert sub_a.shape == (expected_size,), err_msg
            ref_a = getattr(reference, a)[key]
        else:
            assert False, f"Uncovered attribute {a}"

        np.testing.assert_array_equal(sub_a, ref_a, err_msg=a)

    for phase, phase_ref in zip(sub.phases, reference.phases):
        assert phase.state == phase_ref.state, "Physical state not expected to change."
        for a in _EXTENSIVE_ATTRIBUTS + _PHASE_ATTRIBUTES:
            sub_a: np.ndarray = getattr(phase, a)
            ref_a: np.ndarray

            err_msg = f"Unexpected shape for {a} ({sub_a.shape})"

            if a in ["x", "phis"]:
                assert sub_a.shape == (num_comp, expected_size), err_msg
                ref_a = atleast_2d(getattr(phase_ref, a)[:, key])
            elif a == "dphis":
                assert sub_a.shape == (num_comp, num_dep, expected_size), err_msg
                ref_a = getattr(phase_ref, a)[:, :, key]
                if ref_a.ndim == 2:
                    ref_a = ref_a[:, :, np.newaxis]
            elif a[0] == "d":
                assert sub_a.shape == (num_dep, expected_size), err_msg
                ref_a = atleast_2d(getattr(phase_ref, a)[:, key])
            elif a in _EXTENSIVE_ATTRIBUTS + _PHASE_ATTRIBUTES:
                assert sub_a.shape == (expected_size,), err_msg
                ref_a = getattr(phase_ref, a)[key]
            else:
                assert False, f"Uncovered attribute {a}"

            np.testing.assert_array_equal(sub_a, ref_a, err_msg=a)

    # Testing the setting of subproperties
    base_1 = get_random_props(expected_size, num_comp, num_phase, num_dep)

    base[key] = base_1

    for a in _FLUID_ATTRIBUTES + _INTENSIVE_ATTRIBUTS + _EXTENSIVE_ATTRIBUTS:
        mod_a: np.ndarray = getattr(base, a)
        mod_ref_a: np.ndarray

        err_msg = f"Unexpected shape for {a} ({mod_a.shape})"

        # Catch fractional quantities before others due to different shape
        if a == "z":
            assert mod_a.shape == (num_comp, num_vals), err_msg
            mod_ref_a = getattr(reference, a).copy()
            mod_ref_a[:, key] = getattr(base_1, a)
        elif a in ["y", "sat"]:
            assert mod_a.shape == (num_phase, num_vals), err_msg
            mod_ref_a = getattr(reference, a).copy()
            mod_ref_a[:, key] = getattr(base_1, a)
        elif a in _INTENSIVE_ATTRIBUTS + _EXTENSIVE_ATTRIBUTS:
            assert mod_a.shape == (num_vals,), err_msg
            mod_ref_a = getattr(reference, a).copy()
            mod_ref_a[key] = getattr(base_1, a)
        else:
            assert False, f"Uncovered attribute {a}"

        np.testing.assert_array_equal(mod_a, mod_ref_a, err_msg=a)

    for phase, phase_ref, phase_1 in zip(base.phases, reference.phases, base_1.phases):
        assert phase.state == phase_ref.state, "Physical state not expected to change."
        for a in _EXTENSIVE_ATTRIBUTS + _PHASE_ATTRIBUTES:
            mod_a: np.ndarray = getattr(phase, a)
            mod_ref_a: np.ndarray

            err_msg = f"Unexpected shape for {a} ({mod_a.size})"

            if a in ["x", "phis"]:
                assert mod_a.shape == (num_comp, num_vals), err_msg
                mod_ref_a = getattr(phase_ref, a).copy()
                mod_ref_a[:, key] = getattr(phase_1, a)
            elif a == "dphis":
                assert mod_a.shape == (num_comp, num_dep, num_vals), err_msg
                mod_ref_a = getattr(phase_ref, a).copy()
                mod_ref_a[:, :, key] = getattr(phase_1, a)
            elif a[0] == "d":
                assert mod_a.shape == (num_dep, num_vals), err_msg
                mod_ref_a = getattr(phase_ref, a).copy()
                mod_ref_a[:, key] = getattr(phase_1, a)
            elif a in _EXTENSIVE_ATTRIBUTS + _PHASE_ATTRIBUTES:
                assert mod_a.shape == (num_vals,), err_msg
                mod_ref_a = getattr(phase_ref, a).copy()
                mod_ref_a[key] = getattr(phase_1, a)
            else:
                assert False, f"Uncovered attribute {a}"

            np.testing.assert_array_equal(mod_a, mod_ref_a, err_msg=a)
