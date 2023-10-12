"""Tests for utility functions for generation of default domains."""
import pytest

import porepy as pp


def check_key_value(domain, keys, values):
    tol = 1e-12
    for dim, key in enumerate(keys):
        if key in domain:
            assert abs(domain.pop(key) - values[dim]) < tol
        else:
            assert False  # Did not find correct key
    assert len(domain) == 0, "Domain should be empty now."


@pytest.mark.parametrize(
    "physdims_domain",
    [
        # Unit cube
        ([1, 1, 1], pp.UnitCubeDomain()),
        # Unit square
        ([1, 1], pp.UnitSquareDomain()),
        # Cube
        ([3.14, 1, 5], pp.CubeDomain([3.14, 1, 5])),
        # Square
        ([2.71, 1e5], pp.SquareDomain([2.71, 1e5])),
    ],
)
def test_default_domains(physdims_domain):
    physdims = physdims_domain[0]
    domain = physdims_domain[1]
    if len(physdims) == 3:
        keys = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]
        values = [0, 0, 0]
    else:
        keys = ["xmin", "ymin", "xmax", "ymax"]
        values = [0, 0]

    values += physdims

    check_key_value(domain, keys, values)


def test_negative_domain():
    physdims = [1, -1]
    with pytest.raises(ValueError):
        _ = pp.SquareDomain(physdims)
