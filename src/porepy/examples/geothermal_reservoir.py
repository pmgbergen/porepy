"""Example script for simulating a geothermal reservoir with wells and fractures.

This script defines a geothermal reservoir model with two wells and two fractures. The
model includes constitutive laws, boundary conditions, initial conditions, and export
functionality. The model is solved using a line search nonlinear solver.

Some of the functionality is tailored to handle well boundary conditions, while other
parts are fetched from PorePy mixin classes.

Note that the simulations as defined herein is relatively expensive, mainly due to the
high number of time steps. Adjust the time schedule, spatial grid or other parameters as
needed for quicker test runs.

"""

import logging
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from numpy.typing import NDArray

import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsNeumann,
    HydrostaticBoundaryPressureValues,
    LithostaticBoundaryStressValues,
    ThermalGradientBoundaryTemperatureValues,
)
from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
)
from porepy.applications.md_grids.model_geometries import (
    TwoEllipticFractures3d,
    TwoWells3d,
)
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


class WellBoundaryConditions(pp.PorePyModel):
    """Class defining boundary conditions values for geothermal reservoir models with
    wells.

    We impose the well protocols as boundary conditions on the appropriate grids. For
    the boundaries of the 3d matrix grid (and any fracture grids), we impose whatever
    boundary conditions are defined in the super classes. On the well grids, we
    prescribe Dirichlet data. We do not explicitly set the BC type on the well grids,
    so it is assumed that the BC type is defined elsewhere, e.g., in a mixin class. If
    that class specifies Neumann BCs on the well grids, an extension of the model is
    needed to change the BC type to Dirichlet or to prescribe flux values from
    protocols.

    Super calls in methods `bc_values_pressure` and `bc_values_temperature` suggest that
    this class should be used as a mixin with higher MRO priority than classes defining
    non-well boundary conditions, e.g. `HydrostaticBoundaryPressureValues` and
    `ThermalGradientBoundaryTemperatureValues`.
    """

    if TYPE_CHECKING:
        well_names: list[str]
        """List of well tags, e.g. ['injection_well', 'production_well']."""

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Return boundary values for pressure on all boundaries.

        Parameters:
            bg: The boundary grid for which to return the BC values.

        Returns:
            The boundary values for pressure on the given boundary grid.
        """
        sd = bg.parent
        # Ignore super call for type checking, as it is assumed to be present for this
        # mixin class.
        values = super().bc_values_pressure(bg)  # type: ignore[misc]
        if self.is_well_grid(sd):
            well = self.well_network.wells[sd.well_num]
            well_tag = well.tags["well_name"]
            # Find indices of the well boundary sides.
            domain_sides = self.domain_boundary_sides(bg)
            # The top of the domain is '.top' in 3d, '.north' in 2d.
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            # Set pressure values according to the well protocol.
            values[inds] = self.units.convert_units(
                self.get_well_value(
                    self.well_protocols(well_tag, "pressures"),
                    self.time_data.schedule,
                    self.time_data.time,
                ),
                "Pa",
            )
        return values

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Return boundary values for temperature on all boundaries.

        Parameters:
            bg: The boundary grid for which to return the BC values.

        Returns:
            The boundary values for temperature on the given boundary grid.
        """
        sd = bg.parent
        values = super().bc_values_temperature(bg)  # type: ignore[misc]
        if self.is_well_grid(sd):
            well_tag = self.well_names[sd.well_num]
            # Find indices of the well boundary sides.
            domain_sides = self.domain_boundary_sides(bg)
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            # Set temperature values according to the well protocol.
            values[inds] = self.units.convert_units(
                self.get_well_value(
                    self.well_protocols(well_tag, "temperatures"),
                    self.time_data.schedule,
                    self.time_data.time,
                ),
                "K",
            )
        return values

    def get_well_value(
        self,
        values: np.ndarray,
        times: np.ndarray,
        current_time: float,
    ) -> float:
        """Return the well value at the current time.

        Parameters:
            values: Array with time-dependent well values.
            times: Array with time points corresponding to the well values.
                Linear interpolation is used to find the value at the current time.
            current_time: Current simulation time.

        Returns:
            The well value at the current time.

        Raises:
            ValueError: If the current time is outside the range of the provided times.
        """
        if current_time < times[0]:
            raise ValueError("Current time is before the start of the well protocol.")
        elif current_time > times[-1]:
            raise ValueError("Current time is after the end of the well protocol.")
        else:
            return float(np.interp(current_time, times, values))

    def well_protocols(self, well_tag: str, variable: str) -> NDArray[np.float64]:
        """Return the time-dependent protocol array for a given well and variable.

        The value is read from ``self.params`` under the key
        ``"{well_tag}_{variable}"``. A scalar is broadcast to all schedule times; an
        array must match schedule length.

        Parameters:
            well_tag: Name of the well (e.g. ``"injection_well"``).
            variable: Protocol variable name (e.g. ``"pressures"``, ``"temperatures"``,
                ``"mass_rates"``).

        Returns:
            Array of protocol values, one entry per scheduled time point.
        """
        num_times = self.time_data.schedule.size
        raw = self.params.get(f"{well_tag}_{variable}", 0.0)
        if isinstance(raw, (int, float)):
            return np.full(num_times, float(raw))
        values = np.asarray(raw, dtype=float)
        if values.size != num_times:
            raise ValueError(
                f"Protocol '{well_tag}_{variable}' has {values.size} entries, "
                f"expected {num_times} (one per schedule point)."
            )
        return values


class NeumannWellBCsFirstTimeInterval(pp.PorePyModel):
    """Class defining Neumann BCs on well grids during the first time interval.

    Rediscretization happens when calling rediscretize_fluxes in the solution strategy
    class. By default, both diffusive fluxes in lower-dimensional subdomains are tagged
    for rediscretization for a Thermoporomechanics model.
    """

    if TYPE_CHECKING:
        darcy_flux_discretization: Callable[
            [list[pp.Grid]], pp.ad.MpfaAd | pp.ad.TpfaAd
        ]
        add_nonlinear_diffusive_flux_discretization: Callable[
            [pp.ad.MergedOperator], None
        ]
        remove_nonlinear_diffusive_flux_discretization: Callable[
            [pp.ad.MergedOperator], bool
        ]

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Return Neumann BC for Darcy flux on well grids during first time interval.

        Parameters:
            sd: The subdomain for which to return the BC type.

        Returns:
            The boundary condition type for Darcy flux on the given subdomain.
        """
        if self.is_well_grid(sd) and self.time_data.time <= self.time_data.schedule[1]:
            # Before start of injection, impose Neumann BCs on well grids. A zero-flux
            # condition is imposed by default when no BC values are specified. The <=
            # comparison ensures that the BCs are kept as Neumann as long as the time
            # step is within the first time interval [0, t1], where t1 is the start time
            # of injection, consistent with the implicit time stepping employed in
            # PorePy.
            domain_sides = self.domain_boundary_sides(sd)
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            return pp.BoundaryCondition(sd, inds, "neu")
        else:
            return super().bc_type_darcy_flux(sd)  # type: ignore[misc]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Return Neumann BC for Fourier flux on well grids during first time interval.

        Parameters:
            sd: The subdomain for which to return the BC type.

        Returns:
            The boundary condition type for Fourier flux on the given subdomain.
        """
        if self.is_well_grid(sd) and self.time_data.time <= self.time_data.schedule[1]:
            # Before start of injection, impose Neumann BCs on well grids. A zero-flux
            # condition is imposed by default when no BC values are specified. The <=
            # comparison ensures that the BCs are kept as Neumann as long as the time
            # step is within the first time interval [0, t1], where t1 is the start time
            # of injection, consistent with the implicit time stepping employed in
            # PorePy.
            domain_sides = self.domain_boundary_sides(sd)
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            return pp.BoundaryCondition(sd, inds, "neu")
        else:
            return super().bc_type_fourier_flux(sd)  # type: ignore[misc]


class GeothermalReservoirWellBCs(  # type: ignore[misc]
    # Constituive laws
    pp.constitutive_laws.GravityForce,
    pp.constitutive_laws.CubicLawPermeability,
    # BC mixins
    NeumannWellBCsFirstTimeInterval,
    WellBoundaryConditions,
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
    BoundaryConditionsMechanicsNeumann,
    LithostaticBoundaryStressValues,
    # Initial condition mixins
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
    # Geometry mixins
    TwoWells3d,
    TwoEllipticFractures3d,
    # Export mixins
    FractureDeformationExporting,
    # Uncomment the following line to enable iteration exporting, e.g. for debugging.
    # pp.viz.data_saving_model_mixin.IterationExporting,
    # Helper mixin for the line search solution strategy, see also solver_params below.
    pp.models.solution_strategy.ContactIndicators,
    # Base class
    pp.Thermoporomechanics,
):
    """Class defining a geothermal reservoir model with wells and fractures. Collects
    constitutive laws, boundary conditions, and geometry mixins.
    """


def set_model_params():
    # Adjust solid values, while using default values for water.
    solid_values = cast(dict[str, float], pp.solid_values.basalt.copy())
    solid_values.update(
        {
            "dilation_angle": 0.1,  # [rad]
            # Uncomment next two lines to include elastic fracture deformation, aka
            # "Barton-Bandis" model for normal fracture deformation.
            # "fracture_normal_stiffness": 1.1e8,  # [Pa m^-1]
            # "maximum_elastic_fracture_opening": 1e-3,  # [m]
            "normal_permeability": 1.0e-10,  # [m^2]
            "residual_aperture": 1e-3,  # [m]
            "well_radius": 0.1,  # [m]
        }
    )

    # Define time schedule for the simulation.
    schedule = np.array([0, pp.HOUR, 10 * pp.HOUR, 100 * pp.DAY])

    # Add initialization time interval.
    dt_init = 3 * pp.YEAR
    schedule += dt_init * 2.5
    schedule = np.insert(schedule, 0, 0.0)
    # Define injection pressures as list of len = schedule.size. For other protocol
    # values, broadcasting of single values is used for simplicity. The following
    # schedule is somewhat arbitrary, but meant to represent a ramping up of injection
    # pressures over time. The initial low pressure represents a start from near
    # hydrostatic conditions.
    # We ramp up from 1e5 to 5e6 Pa during initialization (well is closed using a
    # Neumann BC), then ramp up to 9e6 Pa at injection start (1 hour), then increase to
    # 11e6 Pa after 10 hours, and finally to 15e6 Pa after 200 days.
    injection_pressures = [1e5, 5e6, 9e6, 11e6, 15e6]  # [Pa]
    # Convenient shortening of simulation schedule for quick simulations. The point is
    # that injection_pressures must match the size of schedule.
    schedule_length = schedule.size  # Change to 3 or 4 for quicker runs.
    schedule = schedule[:schedule_length]
    injection_pressures = injection_pressures[:schedule_length]

    # Define domain sizes (x, y, z) and fracture size.
    length_scale = 1e3  # [m]
    fracture_size = 0.2  # [-], fraction of length_scale
    domain_sizes = np.array(
        [1.0 * length_scale, 1.0 * length_scale, 1.0 * length_scale]
    )  # [m]
    # Define model parameters. See file example_params.py and the mixins used to define
    # :class:`GeothermalReservoirWellBCs` for other options.
    model_params = {
        # Set time manager.
        "time_manager": pp.TimeManager(
            schedule=schedule,
            dt_init=dt_init,
            constant_dt=False,
            dt_min_max=(0.1 * pp.HOUR, max(pp.YEAR, dt_init)),
            iter_optimal_range=(6, 10),  # Allow more iterations than default.
            iter_relax_factors=(0.5, 1.8),  # More aggressive relaxation
        ),
        # Set physical parameters.
        "lithostatic_stress_multipliers": np.array([0.8, 1.2, 1.0]),
        "injection_well_temperatures": 300.00,
        "injection_well_pressures": injection_pressures,
        # The produced fluid is hotter than the injected one.
        "production_well_temperatures": 350.0,
        "production_well_pressures": pp.ATMOSPHERIC_PRESSURE,  # = 1.01325e5 Pa
        "material_constants": {
            "solid": pp.SolidConstants(**solid_values),  # type: ignore[arg-type]
            "fluid": pp.FluidComponent(**pp.fluid_values.water),  # type: ignore[arg-type]
            "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=300.0, pressure=1e6
        ),  # type: ignore[arg-type]
        "units": pp.Units(m=1.0, kg=1.0e5, K=1.0),
        # Set geometry and meshing related parameters.
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": length_scale / 5.0,
            "cell_size_fracture": fracture_size * length_scale / 2.5,
        },
        "well_meshing_arguments": {
            "cell_size": length_scale / 10.0,
        },
        "fracture_params": {  # Other options are available in the geometry mixin.
            "fracture_major_axes": np.array(
                (fracture_size, fracture_size)
            ),  # dimensionless, scaled by domain size
            "dip_angles": np.array((np.pi / 4, -np.pi / 4)),  # [rad]
        },
        "domain_sizes": domain_sizes,
        # Line search: Scale the indicator used for the local_line_search (see below)
        # adaptively to increase robustness.
        "adaptive_indicator_scaling": 1,
        # Set folder name for results.
        "folder_name": "geothermal_reservoir",
        # Add the length scale and fracture size here to make it available for
        # modifications of the parameter dictionary in testing and runscripts.
        "length_scale": length_scale,
        "fracture_size": fracture_size,
    }
    return model_params


def set_solver_params():
    # Define parameters for the non-linear solver.
    solver_params = {
        "prepare_simulation": True,
        "nl_max_iterations": 25,  # Max iterations of a nonlinear solver (Newton)
        "nl_convergence_inc_atol": 1e-7,  # Increment norm
        "nl_convergence_res_atol": 1e-7,  # Residual norm
        "nl_divergence_inc_atol": 1e12,
        "nl_divergence_res_atol": 1e12,
        # Line search / Solution Strategies. These are considered "advanced" options,
        # improving the robustness of the nonlinear solver at the cost of some
        # additional computational overhead. Delete/comment the following lines for the
        # default Newton's method.
        "nonlinear_solver": pp.solvers.ConstraintLineSearchNonlinearSolver,
        # Set to 1 to use turn on a residual-based line search. This involves some extra
        # residual evaluations and may be quite costly.
        "global_line_search": 0,
        # Set to 0 to use turn off the tailored line search, see the class
        # ConstraintLineSearchNonlinearSolver. This line search is cheap and has proven
        # effective for (some versions of) this particular simulation setup.
        "local_line_search": 1,
    }
    return solver_params


def run_example() -> list[pp.PorePyModel]:
    """Run the geothermal reservoir example and return the model."""
    models: list[pp.PorePyModel] = []
    model = GeothermalReservoirWellBCs(set_model_params())
    pp.ModelRunner(model, set_solver_params()).run()
    models.append(model)
    return models


if __name__ == "__main__":
    run_example()
