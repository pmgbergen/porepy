""" Tests of functionality of :class:`~porepy.viz.diagnostics_mixin.DiagnosticsMixin`.

"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from porepy.viz.diagnostics_mixin import DiagnosticsMixin
from tests.integration.models.setup_utils import Poromechanics


class PoromechanicsWithDiagnostics(DiagnosticsMixin, Poromechanics):
    pass


@pytest.fixture
def setup() -> PoromechanicsWithDiagnostics:
    """Assembles test PorePy model setup."""
    setup = PoromechanicsWithDiagnostics()
    # Common preprocessing is done to assemble the linear system.
    setup.prepare_simulation()
    setup.before_newton_loop()
    setup.before_newton_iteration()
    setup.assemble_linear_system()
    return setup


@patch("matplotlib.pyplot.show")
# It changes plt.show to a mockup that does nothing.
# Thus, we prevent blocking of plt.show.
# This provides first argument to the function, which we don't use.
def test_diagnostics_mixin_basic(_, setup: PoromechanicsWithDiagnostics) -> None:
    """Tests basic functionality."""
    diagnostics_data = setup.run_diagnostics(
        is_plotting_condition_number=True, is_plotting_max=True, grouping=None
    )
    assert "Block condition number" in diagnostics_data[0, 0]
    assert "Absolute maximum value" in diagnostics_data[0, 0]


@patch("matplotlib.pyplot.show")
def test_diagnostics_mixin_custom_handler(
    _, setup: PoromechanicsWithDiagnostics
) -> None:
    """Making a custom handler and testing grouping="dense"."""
    last_equation_name = tuple(setup.equation_system.equations.keys())[0]
    last_variable_name = setup.equation_system.variables[0].name

    def custom_handler(mat, equation_name, variable_name) -> float:
        if equation_name == last_equation_name and variable_name == last_variable_name:
            return mat.shape[0]
        return 0

    diagnostics_data = setup.run_diagnostics(
        is_plotting_condition_number=False,
        is_plotting_max=False,
        grouping="dense",
        additional_handlers={"custom_handler": custom_handler},
    )

    assert diagnostics_data[0, 0]["custom_handler"] != 0
    assert diagnostics_data[0, 1]["custom_handler"] == 0
    assert diagnostics_data[1, 0]["custom_handler"] == 0
    assert diagnostics_data[1, 1]["custom_handler"] == 0


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
        is_plotting_condition_number=False,
        is_plotting_max=False,
        grouping=grouping,
        additional_handlers={"is_interface_block": is_interface_block},
    )
    for subdomain_data in diagnostics_data.values():
        if not subdomain_data["is_empty_block"]:
            assert subdomain_data["is_interface_block"] == 1

    # And checking that keyword "interfaces" will prodice the same result.
    diagnostics_data_new = setup.run_diagnostics(
        is_plotting_condition_number=False,
        is_plotting_max=False,
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
