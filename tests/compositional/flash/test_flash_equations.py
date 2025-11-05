"""Module testing assembly of common flash equations as well as generic argument
parsing."""

import pytest

# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np

import porepy.compositional.flash as flash


@pytest.mark.parametrize(
    "spec", [spec for spec in flash.FlashSpec if spec != flash.FlashSpec.none]
)
@pytest.mark.parametrize("with_params", [True, False])
@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [1, 2, 5])
def test_assembly_and_parsing_of_generic_flash_argument(
    ncomp: int,
    nphase: int,
    spec: flash.FlashSpec,
    with_params: bool,
) -> None:
    if with_params:
        params = np.random.random((np.random.randint(1, 10),))
    else:
        params = np.zeros((0,))

    # Make sure all values distinct.
    non_params = np.random.choice(
        np.arange(0, 101), replace=False, size=(flash.dim_gen_arg(ncomp, nphase, spec),)
    )

    Xgen = np.hstack([params, non_params])

    sat, x, y, z, p, T, state1, state2, pars = flash.parse_generic_arg(
        Xgen, ncomp, nphase, spec
    )

    assert sat.shape == (nphase,), "Saturations of unexpected shape"
    assert x.shape == (nphase, ncomp), "Partial fractions of unexpected shape"
    assert y.shape == (nphase,), "Phase fractions of unexpected shape"
    assert z.shape == (ncomp,), "Overall compositions of unexpected shape"
    assert isinstance(p, float), "Pressure expected to be float"
    assert isinstance(T, float), "Temperature expected to be float"
    assert isinstance(state1, float), "State value 1 expected to be float"
    assert isinstance(state2, float), "State value 2 expected to be float"
    assert pars.shape == params.shape, "Parsed parameters of unexpected shape"
    assert np.all(pars == params), "Expecting parameters to not change."

    # Expecting to be unequal in any case.
    assert p != T, "Expecting pressure and temperature to be distinct."
    assert state1 != state2, "Expecting state values to be distinct."

    if spec in [flash.FlashSpec.pT, flash.FlashSpec.vT]:
        assert "T" == spec.name[1], "Expecting character T in isothermal spec."
        assert T == state2, (
            "State value 2 and temperature expected to be equal in isothermal spec."
        )
    else:
        assert T != state2, (
            "State value 2 and temperature expected to be distinct in non-isothermal "
            "spec."
        )
    if "p" == spec.name[0]:
        assert p == state1, (
            "State value 1 and pressure expected to be equal in isobaric spec."
        )
    elif "v" == spec.name[0]:
        assert p != state1, (
            "State value 1 and pressure expected to be distinct in isochoric spec."
        )
        assert T != state1, (
            "State value 1 and temperature expected to be distinct in isochoric spec."
        )
    else:
        assert False, "Uncovered specification."

    # For non-isobaric and non-isothermal specifications, all values must be distinct
    if spec > flash.FlashSpec.vT:
        assert len(set([p, T, state1, state2])) == 4, (
            "State values, pressure and temperature expected to be distinct for "
            "isochoric, non-isothermal spec."
        )

    Xgen2 = flash.assemble_generic_arg(sat, x, y, z, p, T, state1, state2, pars, spec)
    assert np.all(Xgen == Xgen2), (
        "Parsed and re-assembled generic arg expected to be equal to original arg,"
    )

    # Sanity check that the values remain the same and there is no accidental
    # cancelation of errors.
    sat2, x2, y2, z2, p2, T2, state12, state22, pars2 = flash.parse_generic_arg(
        Xgen2.copy(), ncomp, nphase, spec
    )
    assert np.all(sat2 == sat)
    assert np.all(x2 == x)
    assert np.all(y2 == y)
    assert np.all(z == z)
    assert np.all(p2 == p)
    assert np.all(T2 == T)
    assert np.all(state12 == state1)
    assert np.all(state22 == state2)
    assert np.all(pars2 == pars)

    Xgen3 = flash.assemble_generic_arg(
        sat2, x2, y2, z2, p2, T2, state12, state22, pars2, spec
    )
    assert np.all(Xgen == Xgen3)
