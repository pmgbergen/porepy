"""
Module containing configuration functions for Pytest.

Credits: https://jwodder.github.io/kbits/posts/pytest-mark-off/ (Option 1).
"""

import os
import sys

# NOTE: Put this on top before import pytest to disable numba entirely and reliably for
# the whole test session.
if "--disable-jit" in sys.argv:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adopt a new flag to run all tests, including skipped ones."""
    parser.addoption(
        "--run-skipped",
        action="store_true",
        default=False,
        help="Run skipped tests",
    )
    parser.addoption(
        "--disable-jit",
        action="store_true",
        default=False,
        help="Disable Numba JIT compilation (by default it is enabled for tests)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Identify tests mark with 'skipped' at collection."""
    if not config.getoption("--run-skipped"):
        skipper = pytest.mark.skip(reason="Only run when --run-skipped is given")
        for item in items:
            if "skipped" in item.keywords:
                item.add_marker(skipper)


def pytest_configure(config: pytest.Config) -> None:
    # See https://docs.pytest.org/en/stable/how-to/mark.html
    config.addinivalue_line(
        "markers", "skipped: Mark test to be run only once a week and not during PR."
    )
