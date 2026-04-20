"""Testing functionality in abstract flash module."""

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from numpy.typing import NDArray

from porepy.compositional.states import PhaseProperties
from porepy.compositional.utils import FlashSpec
from tests.compositional.test_states import get_random_props


@pytest.mark.parametrize(
    "p_spec",
    [FlashSpec.pT, FlashSpec.ph],
)
@pytest.mark.parametrize(
    "v_spec",
    [FlashSpec.vT, FlashSpec.vh, FlashSpec.vu],
)
def test_flash_specs(p_spec: FlashSpec, v_spec: FlashSpec) -> None:
    """The flash specifications must fulfill certain criteria when performing logical
    operations on them."""
    assert v_spec != p_spec, "All spec. must be logically unequal."
    assert v_spec > FlashSpec.none, "No spec. must have lowest order."
    assert p_spec > FlashSpec.none, "No spec. must have lowest order."
    assert p_spec < v_spec, "Isobaric spec. must be of lower order than isochoric spec."
    assert p_spec >= FlashSpec.pT, (
        "Isobaric-isothermal spec. must be lowest order isobaric spec."
    )
    assert v_spec >= FlashSpec.vT, (
        "Isochoric-isothermal spec. must be lowest order isochoric spec."
    )


@pytest.mark.parametrize(
    "num_vals, key",
    [
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
def test_setter_getter_flash_results(
    num_vals: int,
    key: int | slice | NDArray[np.int_] | NDArray[np.bool_],
) -> None:
    """Tests the setter and getter implementation of the flash results data
    structure.

    This tests only the additional functionality implemented in the overloads of
    ``setitem`` and ``getitem`` of the flash results class.

    """
    from porepy.compositional.flash.abstract_flash import FlashResults

    # These numbers should not impact the test. Respective functionality should be
    # tested separately for the property dataclasses.
    num_dep = 2
    num_comp = 2
    num_phase = 2
    vals = np.arange(num_vals, dtype=int)

    base = FlashResults(
        specification=FlashSpec.pT,
        dofs=10,
        size=10,
        clocktime_init=1.0,
        clocktime_solve=1.0,
        exitcode=vals.copy(),
        num_iter=vals.copy(),
        **asdict(get_random_props(num_vals, num_comp, num_phase, num_dep)),
    )
    # NOTE: asdict turns phase properties into dictionaries.
    phases = []
    for phase in base.phases:
        phases.append(PhaseProperties(**phase))
    base.phases = phases

    reference = deepcopy(base)

    # Checking the extracted dataclass.
    sub = base[key]
    assert isinstance(sub, FlashResults), "Sliced object must be of type FlashResults."
    assert np.array_equal(sub.exitcode, reference.exitcode[key]), (
        "Exit codes must be correctly sliced."
    )
    assert np.array_equal(sub.num_iter, reference.num_iter[key]), (
        "Number of iterations must be correctly sliced."
    )
    assert sub.specification == reference.specification, (
        "Specification must be unchanged by slicing."
    )
    assert sub.dofs == reference.dofs, "DOFs must be unchanged by slicing."
    assert sub.clocktime_init == reference.clocktime_init, (
        "Clocktime init must be unchanged by slicing."
    )
    assert sub.clocktime_solve == reference.clocktime_solve, (
        "Clocktime solve must be unchanged by slicing."
    )

    # Calculate resulting size.
    if isinstance(key, int):
        expected_size = 1
    elif isinstance(key, slice):
        expected_size = len(range(*key.indices(reference.size)))
    elif isinstance(key, np.ndarray):
        if key.dtype == bool:
            expected_size = int(np.sum(key))
        elif key.dtype == int and key.ndim == 1:
            expected_size = len(key)
        else:
            assert False, "Invalid key array."
    else:
        assert False, "Invalid key type."

    assert sub.size == expected_size, "Size must be correctly calculated."

    # Checking the modified parent class when setting a substate
    base_1 = FlashResults(
        specification=FlashSpec.ph,
        dofs=20,
        size=expected_size,
        clocktime_init=2.0,
        clocktime_solve=2.0,
        exitcode=vals[key].copy() + 10,
        num_iter=vals[key].copy() + 10,
        **asdict(get_random_props(expected_size, num_comp, num_phase, num_dep)),
    )
    phases = []
    for phase in base_1.phases:
        phases.append(PhaseProperties(**phase))
    base_1.phases = phases
    base[key] = base_1

    assert base.specification == reference.specification, (
        "Specification must be unchanged by setitem."
    )
    assert base.dofs == reference.dofs, "DOFs must be unchanged by setitem."
    assert base.size == reference.size, "Size must be unchanged by setitem."
    assert base.clocktime_init == reference.clocktime_init, (
        "Clocktime init must be unchanged by setitem."
    )
    assert base.clocktime_solve == reference.clocktime_solve, (
        "Clocktime solve must be unchanged by setitem."
    )
    assert np.array_equal(base.num_iter, reference.num_iter), (
        "Number of iterations must be unchanged by setitem."
    )

    # Exit codes are the only one changed
    reference_codes = reference.exitcode.copy()
    reference_codes[key] = base_1.exitcode
    assert np.array_equal(base.exitcode, reference_codes), (
        "Exitcodes must match after setitem."
    )
