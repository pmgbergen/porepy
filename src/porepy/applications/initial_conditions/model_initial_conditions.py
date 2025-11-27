from typing import Callable

from numpy.typing import NDArray

from porepy.applications.boundary_conditions.model_boundary_conditions import (
    HydrostaticPressureValues,
    ThermalGradientTemperatureValues,
)
from porepy.grids.grid import Grid


class InitialConditionHydrostaticPressureValues(HydrostaticPressureValues):
    """Initial conditions for the flow with hydrostatic pressure in the domain.

    Pressure values corresponding to hydrostatic pressure are defined in the entire
    domain. These will be used as initial conditions for pressure and any derived
    quantities defined on the subdomains, such as the density-mobility product.

    """

    depth: Callable[[NDArray], NDArray]
    """Function to compute depth of points."""

    def ic_values_pressure(self, sd: Grid) -> NDArray:
        """Pressure values.

        Parameters:
            sd: Subdomain grid for which initial values are to be returned.

        Returns:
            Array of initial values, with one value for each cell in the subdomain.

        """
        depth = self.depth(sd.cell_centers)

        values = self.hydrostatic_pressure(depth)
        return values


class InitialConditionThermalGradientTemperatureValues(
    ThermalGradientTemperatureValues
):
    """Initial conditions for the thermal problem with linear temperature gradient
    in the domain.

    Temperature values corresponding to a linear thermal gradient are defined in the
    entire domain. These will be used as initial conditions for temperature and any
    derived quantities defined on the subdomains.

    """

    depth: Callable[[NDArray], NDArray]
    """Function to compute depth of points."""

    def ic_values_temperature(self, sd: Grid) -> NDArray:
        """Temperature values.

        Parameters:
            sd: Subdomain grid for which initial values are to be returned.

        Returns:
            Array of initial values, with one value for each cell in the subdomain.

        """
        return self.temperature_at_depth(self.depth(sd.cell_centers))
