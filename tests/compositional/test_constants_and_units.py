"""
Constitutive library tests.
    - Units
       - Setting default and modified unit objects
       - Relationships between base and derived units
    - Material properties
        - Unit conversion
        - Default values
        - Setting modified values TODO
"""
from __future__ import annotations

import numpy as np
import pytest

import porepy as pp


# TODO remove
@pytest.fixture
def subdomains() -> list[pp.Grid]:
    """Create a list of grids for the test."""
    g1 = pp.CartGrid(np.array([1, 1]), np.array([1, 1]))
    g1.compute_geometry()
    # 2d grid
    g2 = pp.CartGrid(np.array([2, 1]), np.array([1, 1]))
    g2.compute_geometry()
    # 3d
    g3 = pp.CartGrid(np.array([3, 1, 1]), np.array([1, 1, 1]))
    g3.compute_geometry()
    return [g1, g2, g3]


@pytest.fixture
def base_units() -> list[str]:
    """Create a list of base units defined in Units."""
    return ["m", "kg", "s", "K", "mol", "rad"]


@pytest.fixture
def derived_units() -> list[str]:
    """Create a list of derived units defined in Units."""
    return ["Pa", "N", "J", "W", "degree"]


def test_default_units(base_units, derived_units):
    """Test that the default units are defined properly.

    All base and derived units should be defined, and the default unit for each
    should be one. We check that the lists derived_units and base_units are cover
    all attributes of the unit object (useful if new units are added and the test should
    be updated). Also check that instantiating with invalid units raises an error.

    """
    units = pp.Units()
    for unit in base_units + derived_units:
        assert hasattr(units, unit)
        if unit == "degree":
            # Only one derived unit with a different default value
            assert np.isclose(getattr(units, unit), 180 / np.pi)
        else:
            assert np.isclose(getattr(units, unit), 1)

    # Get list of all public attributes excluding methods
    exclude = ['convert_units']
    attributes = [
        attr
        for attr in dir(units)
        if not (attr.startswith("_") or attr in exclude)
    ]

    # Check that all attributes are base or derived units.
    # An error in this test likely means a new unit has been added, but is not covered
    # by the test. Consider updating the test (both the present one and other tests
    # in this file).
    for attr in attributes:
        assert attr in base_units + derived_units

    # Check that instantiating with invalid unit name raises an error
    with pytest.raises(ValueError):
        pp.Units(invalid_unit=1)
    # Check that instantiating with invalid unit value raises an error
    with pytest.raises(ValueError):
        # Only integers and floats are allowed
        pp.Units(m="invalid")


# Two sets of units to be used in the tests below. All basic units are modified in at
# least one of the two sets, except the radian, which is not expected to be rescaled
# in a manner similar to the other units.
modification_dictionaries = [dict(m=2, kg=3.7, s=1), dict(m=3, mol=2.5)]


@pytest.mark.parametrize("modify_dict", modification_dictionaries)
def test_modified_units(modify_dict, derived_units, base_units):
    """Test that the derived units are defined properly."""
    # Pass dictionary to constructor as keyword arguments
    units = pp.Units(**modify_dict)
    # First check that base units are defined properly
    for unit in base_units:
        if unit in modify_dict:
            assert np.isclose(getattr(units, unit), modify_dict[unit])
        else:
            assert np.isclose(getattr(units, unit), 1)

    # Test Pascal in base units
    assert np.isclose(units.Pa, units.kg / units.m / units.s**2)
    # And in derived units
    assert np.isclose(units.Pa, units.N / units.m**2)
    # Test Newton in base units
    assert np.isclose(units.N, units.kg * units.m / units.s**2)
    # And in derived units
    assert np.isclose(units.N, units.Pa * units.m**2)
    # Test Joule in base units
    assert np.isclose(units.J, units.kg * units.m**2 / units.s**2)
    # And in derived units
    assert np.isclose(units.J, units.N * units.m)
    # Test Watts in base units
    assert np.isclose(units.W, units.kg * units.m**2 / units.s**3)
    # And in derived units
    assert np.isclose(units.W, units.J / units.s)
    # Test degree in base units
    assert np.isclose(units.degree, units.rad * 180 / np.pi)

    # Check that all derived units are covered. Note that check_default_units asserts
    # that all derived units are included in the fixtures in this test. A failure below
    # would likely indicate that a new unit has been added to pp.Units, and to the set
    # of derived units, as returned by the fixture derived_units, but not to the present
    # test.
    # Note that if a unit is added to pp.Units, but not to the set of derived units,
    # the present test will not fail, but the test_default_units will, see comments
    # towards the end of that test.

    # These are the units figuring in the above assertions.
    units_covered = ["Pa", "N", "J", "W", "degree"]
    assert set(units_covered) == set(derived_units)


@pytest.mark.parametrize("modify_dict", modification_dictionaries)
def test_convert_units(modify_dict, base_units):
    """Test that the conversion between units works as expected.

    Conversion includes parsing and evaluating the string and dividing the value by the
    resulting unit.
    """
    # Set up a unit class with modified units passed as keyword arguments.
    units = pp.Units(**modify_dict)

    # Test that the conversion works for base units.
    for unit in base_units:
        # Retrieve the value of the unit. If the unit is not modified, the default value
        # is 1.
        val = modify_dict.get(unit, 1)
        # Assert that scaling 1 * unit returns the inverse of the scale set for unit
        assert np.isclose(units.convert_units(1, unit), 1 / val)
        # Assert that scaling 1 / unit**2 returns the square of the scale set for unit
        assert np.isclose(units.convert_units(1, f"{unit}^-2"), val**2)

    # Test that the conversion works for combinations of units
    # Get pascal from modified base units
    pascal = units.kg / units.m / units.s**2

    # Combine the Pascal with other units (somewhat randomly chosen)
    expected = pascal / units.m**2
    # The conversion method in the material class should be equivalent to manual scaling
    # as done above.
    assert np.isclose(units.convert_units(1, "m^2 *Pa^-1"), expected)
    expected = pascal * units.s / units.m**2
    assert np.isclose(units.convert_units(1, "m^2*Pa^-1*s^-1"), expected)

    # Test that invalid units raise an error
    invalid_units = ["invalid", "m^2*invalid", "^2", "s^-1*", "2", "m**2"]
    for unit in invalid_units:
        with pytest.raises(AttributeError):
            units.convert_units(1, unit)

    # Test that the different ways of defining a dimensionless unit work
    dimensionless_units = ["", "1", "-", "   "]
    for unit in dimensionless_units:
        assert np.isclose(units.convert_units(1, unit), 1)
