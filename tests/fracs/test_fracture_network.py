"""Testing common functionality related to the fracture nework. These functions are
covered:
- create_fracture_network

"""

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.domains import unit_cube_domain as unit_domain
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d


# Set line fracture plane fracture as fixture functions to reuse them later
@pytest.fixture
def line_fracture():
    return pp.LineFracture(np.array([[0.5, 0.5], [0, 1]]))


@pytest.fixture
def plane_fracture():
    return pp.PlaneFracture(np.array([[0, 2, 2, 0], [0, 2, 2, 0], [-1, -1, 1, 1]]))


@pytest.mark.parametrize("fractures", [[], None])
def test_error_raised_if_fractures_and_domain_are_none(fractures):
    """Check that an error is raised when both fractures and domain are given as
    None/ empty"""
    with pytest.raises(ValueError) as excinfo:
        msg = "'fractures' and 'domain' cannot both be empty/None."
        pp.create_fracture_network(fractures, None)
    assert msg in str(excinfo.value)


# -----> Test #2
@pytest.fixture()
def fractures_lists(line_fracture, plane_fracture):
    fractures_1 = [1]
    fractures_2 = [line_fracture, "just_another_fracture"]
    fractures_3 = [line_fracture, plane_fracture]
    fractures_4 = [plane_fracture, line_fracture]
    return [fractures_1, fractures_2, fractures_3, fractures_4]


@pytest.mark.parametrize("tst_idx", [0, 1, 2, 3])
def test_error_if_fractures_has_heterogeneous_types(fractures_lists, tst_idx):
    """Check that a type error is raised if the list of fractures contain
    non-homogeneous types, e.g., it should be all PlaneFractures or
    LineFractures."""
    with pytest.raises(TypeError) as excinfo:
        msg = "All fracture objects must be of the same type."
        pp.create_fracture_network(fractures_lists[tst_idx], None)
    assert msg in str(excinfo.value)


# -----> Test #3
@pytest.mark.parametrize("domain", [unit_domain(2), unit_domain(3)])
def test_empty_list_and_none_create_the_same_network(domain):
    """Checks that the same fracture network is generated if an empty fracture
    list OR None is given."""
    fn_none = pp.create_fracture_network(None, domain)
    fn_empty_list = pp.create_fracture_network([], domain)
    assert fn_none.num_frac() == fn_empty_list.num_frac()
    assert fn_none.domain == fn_empty_list.domain


# -----> Test #4
def test_error_raised_if_different_dimensions(line_fracture):
    """Checks that an error is raised if the dimensions inferred from the set of
    fractures and the domain are different."""
    with pytest.raises(ValueError) as excinfo:
        msg = "The dimensions inferred from 'fractures' and 'domain' do not match."
        pp.create_fracture_network([line_fracture], unit_domain(3))
    assert msg in str(excinfo.value)


# -----> Test #5
def test_warning_raised_if_run_checks_true_for_dim_not_3():
    """Checks if the warning is properly raised when ``run_checks=True`` for a
    fracture network with a dimensionality different from 3."""
    warn_msg = "'run_checks=True' has no effect if dimension != 3."
    with pytest.warns() as record:
        pp.create_fracture_network(None, domain=unit_domain(2), run_checks=True)
    assert str(record[0].message) == warn_msg


# -----> Test #6
@pytest.fixture()
def fractures_list_2d(line_fracture):
    list_1 = []
    list_2 = [line_fracture]
    return [list_1, list_2]


@pytest.mark.parametrize("tst_idx", [0, 1])
def test_create_fracture_network_2d(fractures_list_2d, tst_idx):
    """Test if 2d fracture networks constructed with FractureNetwork2d and with
    create_fracture_network() are the same."""
    fn_with_function = pp.create_fracture_network(
        fractures_list_2d[tst_idx], unit_domain(2)
    )
    fn_with_class = FractureNetwork2d(fractures_list_2d[tst_idx], unit_domain(2))
    assert isinstance(fn_with_function, FractureNetwork2d)
    assert fn_with_function.num_frac() == fn_with_class.num_frac()
    assert fn_with_function.domain == fn_with_class.domain


# -----> Test #7
@pytest.fixture()
def fractures_list_3d(plane_fracture):
    list_1 = []
    list_2 = [plane_fracture]
    return [list_1, list_2]


@pytest.mark.parametrize("tst_idx", [0, 1])
def test_create_fracture_network_3d(fractures_list_3d, tst_idx):
    """Test if 3d fracture networks constructed with FractureNetwork3d and with
    create_fracture_network() are the same."""
    fn_with_function = pp.create_fracture_network(
        fractures_list_3d[tst_idx], unit_domain(3)
    )
    fn_with_class = FractureNetwork3d(fractures_list_3d[tst_idx], unit_domain(3))
    assert isinstance(fn_with_function, FractureNetwork3d)
    assert fn_with_function.num_frac() == fn_with_class.num_frac()
    assert fn_with_function.domain == fn_with_class.domain
