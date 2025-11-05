"""Testing functionality in abstract flash module."""

import pytest

import porepy.compositional.flash as ppf


@pytest.mark.parametrize(
    "p_spec",
    [ppf.FlashSpec.pT, ppf.FlashSpec.ph],
)
@pytest.mark.parametrize(
    "v_spec",
    [ppf.FlashSpec.vT, ppf.FlashSpec.vh, ppf.FlashSpec.vu],
)
def test_flash_specs(p_spec: ppf.FlashSpec, v_spec: ppf.FlashSpec) -> None:
    """The flash specifications must fulfill certain criteria when performing logical
    operations on them."""
    assert v_spec != p_spec, "All spec. must be logically unequal."
    assert v_spec > ppf.FlashSpec.none, "No spec. must have lowest order."
    assert p_spec > ppf.FlashSpec.none, "No spec. must have lowest order."
    assert p_spec < v_spec, "Isobaric spec. must be of lower order than isochoric spec."
    assert p_spec >= ppf.FlashSpec.pT, (
        "Isobaric-isothermal spec. must be lowest order isobaric spec."
    )
    assert v_spec >= ppf.FlashSpec.vT, (
        "Isochoric-isothermal spec. must be lowest order isochoric spec."
    )
