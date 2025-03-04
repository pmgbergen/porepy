"""
Module containing configuration functions for Pytest.

Credits: https://jwodder.github.io/kbits/posts/pytest-mark-off/ (Option 1).
"""

import pytest


def pytest_addoption(parser):
    """Adopt a new flag to run all tests, including skipped ones."""
    parser.addoption(
        "--run-skipped",
        action="store_true",
        default=False,
        help="Run skipped tests",
    )


def pytest_collection_modifyitems(config, items):
    """Identify tests mark with 'skipped' at collection."""
    if not config.getoption("--run-skipped"):
        skipper = pytest.mark.skip(reason="Only run when --run-skipped is given")
        for item in items:
            if "skipped" in item.keywords:
                item.add_marker(skipper)


def pytest_configure(config):
    # See https://docs.pytest.org/en/stable/how-to/mark.html
    config.addinivalue_line(
        "markers", "skipped: Mark test to be run only once a week and not during PR."
    )
