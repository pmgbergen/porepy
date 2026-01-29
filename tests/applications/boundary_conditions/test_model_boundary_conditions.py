import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
)


# def test_hydrostatic_boundary_pressure_values():
#     model = HydrostaticBoundaryPressureValues()
#     parent_grid = pp.Grid(dim=3)
#     boundary_grid = pp.BoundaryGrid()
#     model.bc_values_pressure
