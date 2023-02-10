""" Tests of functionality of porepy.vis.diagnostics_mixin.DiagnosticsMixin.
"""

import unittest
from unittest.mock import patch

from porepy.viz.diagnostics_mixin import DiagnosticsMixin
from tests.integration.test_contact_mechanics_biot import SetupContactMechanicsBiot


class BiotWithDiagnostics(DiagnosticsMixin, SetupContactMechanicsBiot):
    pass


@patch("matplotlib.pyplot.show")
# It changes plt.show to a mockup that does nothing.
# Thus, we prevent blocking of plt.show.
# This provides first argument to the function, which we don't use.
def test_diagnostics_mixin(_) -> None:
    """Runs over the functionality of
    :class:`~porepy.viz.diagnostics_mixin.DiagnosticsMixin`.

    Plotting functionality is based on seaborn which might not be installed in the
    testing environment. In this case the mixin must fall back to text output mode,
    and test must run succeed. Thus, we test everything except plotting.
    """
    setup = BiotWithDiagnostics()
    setup.with_fracture = True  # type: ignore
    setup.ux_north = 0.01

    # Common preprocessing is done to assemble the linear system.
    setup.prepare_simulation()
    setup.before_newton_loop()
    setup.before_newton_iteration()
    setup.assemble_linear_system()

    setup.show_diagnostics(
        is_plotting_condition_number=True, is_plotting_max=True, grouping=None
    )
    setup.show_diagnostics(
        is_plotting_condition_number=True, is_plotting_max=True, grouping="subdomains"
    )


if __name__ == "__main__":
    unittest.main()
