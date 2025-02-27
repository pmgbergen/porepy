"""Tests of functionality of :class:`~porepy.viz.diagnostics_mixin.DiagnosticsMixin`."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from porepy.applications.test_utils.models import Poromechanics
from porepy.viz.diagnostics_mixin import DiagnosticsMixin


class PoromechanicsWithDiagnostics(DiagnosticsMixin, Poromechanics):
    pass


@pytest.fixture
def setup() -> PoromechanicsWithDiagnostics:
    """Assembles test PorePy model setup."""
    setup = PoromechanicsWithDiagnostics()
    # Common preprocessing is done to assemble the linear system.
    setup.prepare_simulation()
    setup.before_nonlinear_loop()
    setup.before_nonlinear_iteration()
    setup.assemble_linear_system()
    return setup


@patch("matplotlib.pyplot.show")
# It changes plt.show to a mockup that does nothing.
# Thus, we prevent blocking of plt.show.
# This provides first argument to the function, which we don't use.
def test_diagnostics_mixin_basic(_, setup: PoromechanicsWithDiagnostics) -> None:
    """Tests basic functionality."""
    diagnostics_data = setup.run_diagnostics(
        default_handlers=("max", "cond"),
        grouping=None,
    )
    for block in diagnostics_data.values():
        if not block["is_empty_block"]:
            assert "max" in block
            assert "cond" in block
            assert block["max"] > 0
            assert block["cond"] > 0


@patch("matplotlib.pyplot.show")
def test_diagnostics_mixin_custom_handler(
    _, setup: PoromechanicsWithDiagnostics
) -> None:
    """Making a custom handler and testing grouping="dense"."""
    tracked_variable = setup.contact_traction_variable
    tracked_equation = "normal_fracture_deformation_equation"

    # Safeguard in case if the equation name is once changed.
    assert tracked_equation in setup.equation_system.equations

    def custom_handler(mat, equation_name, variable_name) -> float:
        if equation_name == tracked_equation and variable_name == tracked_variable:
            return mat.shape[0]
        return 0

    diagnostics_data = setup.run_diagnostics(
        grouping="dense",
        additional_handlers={"custom_handler": custom_handler},
    )

    # Run along all diagnostics data. Discard empty blocks. Assert that nonzero is only
    # in the tracked equation and variable combination.
    for submatrix_data in diagnostics_data.values():
        if submatrix_data["is_empty_block"]:
            assert "custom_handler" not in submatrix_data
            continue
        if (
            submatrix_data["variable_name"] == tracked_variable
            and submatrix_data["equation_name"] == tracked_equation
        ):
            assert submatrix_data["custom_handler"] != 0
        else:
            assert submatrix_data["custom_handler"] == 0


@patch("matplotlib.pyplot.show")
def test_diagnostics_mixin_grouping(_, setup: PoromechanicsWithDiagnostics) -> None:
    """Testing custom grouping. We want to investigate only the interface."""
    # Collecting only the interface variable names.
    interface_variable_names = [
        var.name for var in setup.equation_system.variables if len(var.interfaces) > 0
    ]

    def is_interface_block(mat, equation_name, variable_name) -> float:
        if variable_name in interface_variable_names:
            return 1
        return 0

    grouping = [setup.mdg.interfaces()]
    diagnostics_data = setup.run_diagnostics(
        grouping=grouping,
        additional_handlers={"is_interface_block": is_interface_block},
    )
    for subdomain_data in diagnostics_data.values():
        if not subdomain_data["is_empty_block"]:
            assert subdomain_data["is_interface_block"] == 1

    # And checking that keyword "interfaces" will prodice the same result.
    diagnostics_data_new = setup.run_diagnostics(
        grouping="interfaces",
        additional_handlers={"is_interface_block": is_interface_block},
    )

    assert len(diagnostics_data_new) == len(diagnostics_data)
    for subdomain_data, subdomain_data_new in zip(
        diagnostics_data.values(), diagnostics_data_new.values()
    ):
        assert subdomain_data["is_empty_block"] == subdomain_data_new["is_empty_block"]
        if not subdomain_data_new["is_empty_block"]:
            assert (
                subdomain_data["is_interface_block"]
                == subdomain_data_new["is_interface_block"]
            )
