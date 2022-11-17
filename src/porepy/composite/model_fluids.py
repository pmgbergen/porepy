"""Contains concrete implementations of components for the fluid phases in the porous medium.
In the current setting, we expect these substances to only appear in liquid or gaseous form.
i.e. they are associated with the flow.
"""

from .component import FluidComponent

__all__ = ["H2O"]


class H2O(FluidComponent):
    """Fluid component representing water.

    The physical properties and parameters are found in respective references.

    """

    @staticmethod
    def molar_mass():
        """`Source <https://en.wikipedia.org/wiki/Water_vapor>`_."""
        return 0.0180152833

    @staticmethod
    def critical_pressure():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        return 22060

    @staticmethod
    def critical_temperature():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        return 647.096

    @staticmethod
    def triple_point_pressure():
        """`Source <https://en.wikipedia.org/wiki/Water_(data_page)>`_."""
        return 0.61173

    @staticmethod
    def triple_point_temperature():
        """`Source <https://en.wikipedia.org/wiki/Water_(data_page)>`_."""
        return 273.1600
