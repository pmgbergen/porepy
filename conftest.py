"""
Module containing configuration functions for Pytest.

Credits: https://jwodder.github.io/kbits/posts/pytest-mark-off/ (Option 1).
"""

import os
import glob
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


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files():
    """Cleanup generated test files after the test session."""
    yield  # Let the tests run first

    # Get the current working directory
    current_dir = os.getcwd()

    # Define the patterns for files to delete
    patterns = [
        os.path.join(current_dir, "gmsh_frac_file_*.msh"),
        os.path.join(current_dir, "*.geo_unrolled"),
    ]

    # Delete matching files
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
