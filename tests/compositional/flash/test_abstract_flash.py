"""Testing functionality in abstract flash module."""

import pytest

from porepy.compositional.flash import FlashSpec


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
