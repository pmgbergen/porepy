""" Tests for variable storage in the data fields "state" and "iterate".
"""
from typing import Dict

import pytest

import porepy as pp


@pytest.fixture
def empty_dict() -> Dict:
    return {}


class TestState:
    def test_add_empty_state(self, empty_dict):
        """Add an empty state dictionary"""
        d = empty_dict
        pp.set_state(d)
        assert 'stored_solutions' in d

    def test_add_empty_iterate(self, empty_dict):
        """Add an empty iterate dictionary"""
        d = empty_dict
        pp.set_iterate(d)
        assert 'stored_iterates' in d

    def test_add_state_twice(self, empty_dict):
        """Add two state dictionaries.

        The existing foo value should be overwritten, while bar should be kept.
        """
        d = empty_dict
        d1 = {"foo": {0: 1}, "bar": {0: 2}}
        d2 = {"foo": {0: 3}, "spam": {0: 4}}

        pp.set_state(d, d1)
        pp.set_state(d, d2)
        for key, val in zip(["foo", "bar", "spam"], [3, 2, 4]):
            assert key in d['stored_solutions']
            assert 0 in d['stored_solutions'][key]
            assert d['stored_solutions'][key][0] == val

    def test_add_iterate_twice_and_state(self, empty_dict):
        """Add two state dictionaries.

        The existing foo value should be overwritten, while bar should be kept.
        Setting values in 'stored_solutions' should not affect the iterate values.
        """
        d = empty_dict
        d1 = {"foo": 1, "bar": 2}
        d2 = {"foo": 3, "spam": 4}

        pp.set_iterate(d, d1)
        pp.set_iterate(d, d2)
        pp.set_state(d, {"foo": 5})
        for key, val in zip(["foo", "bar", "spam"], [3, 2, 4]):
            assert key in d['stored_iterates']
            assert d['stored_iterates'][key] == val
        assert d['stored_solutions']["foo"] == 5
