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
from dataclasses import dataclass, field
from enum import Enum
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


class WellBCType(Enum):
    """Type of the boundary condition imposed at the top of a well."""

    DIRICHLET = "dirichlet"
    """The well values (e.g. pressure, temperature) are imposed."""
    NEUMANN = "neumann"
    """Zero flux, i.e. the well is closed."""
    ROBIN = "robin"
    """Mixed condition. Not implemented by the boundary condition classes below."""


@dataclass(frozen=True)
class WellInterval:
    """One point of a well schedule: the start of an interval and the well values there.

    A well schedule is a list of these, sorted by ``start``. The values are linearly
    interpolated in time between consecutive starts, so each interval spans from its
    ``start`` to the start of the next one.

    Attributes:
        start: Start time of the interval [s].
        values: Mapping from protocol variable name (e.g. ``"pressures"``,
            ``"temperatures"``) to the value at ``start``, in SI units.
        bc_type: Type of the boundary condition of the well in this interval. See
            :meth:`WellSchedule.well_bc_type`.

    """

    start: float
    values: dict[str, float] = field(default_factory=dict)
    bc_type: WellBCType = WellBCType.DIRICHLET


class WellSchedule(pp.PorePyModel):
    """Mixin providing the well schedules, which define the well protocols.

    The well schedule is distinct from the time manager's schedule, which only controls
    time stepping. It is read from ``self.params["well_schedule"]``, a dictionary
    mapping the well name (e.g. ``"injection_well"``) to a list of
    :class:`WellInterval`, sorted by start time.

    """

    def parent_well_tag(self, sd: pp.Grid) -> str:
        """Return the name of the well that a well grid belongs to."""
        return self.well_network.wells[sd.well_num].tags["well_name"]

    def well_intervals(self, well_tag: str) -> list[WellInterval]:
        """Return the validated schedule of a well.

        Parameters:
            well_tag: Name of the well (e.g. ``"injection_well"``).

        Returns:
            The list of intervals of the well, sorted by start time.

        Raises:
            ValueError: If the well has no schedule, or the intervals are empty or not
                strictly increasing in start time.

        """
        schedules = self.params.get("well_schedule", {})
        if well_tag not in schedules or len(schedules[well_tag]) == 0:
            raise ValueError(f"No well schedule provided for well '{well_tag}'.")
        intervals = list(schedules[well_tag])
        starts = np.array([interval.start for interval in intervals])
        if np.any(np.diff(starts) <= 0):
            raise ValueError(
                f"Well schedule of '{well_tag}' must have strictly increasing starts."
            )
        return intervals

    def well_value(self, well_tag: str, variable: str) -> float:
        """Return the value of a well protocol variable at the current time.

        The value is linearly interpolated between the interval starts, and kept
        constant from the last start until the end of the simulation.

        Parameters:
            well_tag: Name of the well.
            variable: Protocol variable name (e.g. ``"pressures"``).

        Returns:
            The value of the variable at the current time, in SI units.

        Raises:
            ValueError: If the current time is outside the well schedule, or an
                interval does not define the variable.

        """
        intervals = self.well_intervals(well_tag)
        starts = np.array([interval.start for interval in intervals])
        t = self.time_manager.time
        # Raises if the time is outside the well schedule. Beyond the last start, the
        # values of the last interval are kept constant until the end of the simulation.
        self.well_interval_index(well_tag, t)
        for interval in intervals:
            if variable not in interval.values:
                raise ValueError(
                    f"Well schedule of '{well_tag}' does not define '{variable}' in "
                    f"the interval starting at {interval.start}."
                )
        values = np.array([interval.values[variable] for interval in intervals])
        return float(np.interp(t, starts, values))

    def well_interval_index(self, well_tag: str, time: float) -> int:
        """Return the index of the well interval containing a time.

        Intervals are right-closed, consistent with the implicit time stepping employed
        in PorePy: a time equal to the start of an interval belongs to the previous
        interval, except for the very first start, which belongs to the first interval.
        The last interval ends at the end of the simulation, as given by the time
        manager's schedule.

        Parameters:
            well_tag: Name of the well.
            time: The time to locate, typically the current time.

        Returns:
            The index of the interval in the list returned by :meth:`well_intervals`.

        Raises:
            ValueError: If the time is outside the well schedule, i.e. before the first
                start or after the end of the simulation.

        """
        starts = np.array(
            [interval.start for interval in self.well_intervals(well_tag)]
        )
        ends = np.append(starts[1:], self.time_manager.schedule.get_array()[-1])
        if time < starts[0] or time > ends[-1]:
            raise ValueError(f"Time {time} is outside the schedule of '{well_tag}'.")
        return int(np.searchsorted(ends, time, side="left"))

    def well_interval_bounds(self, well_tag: str, time: float) -> tuple[float, float]:
        """Return the start and end time of the well interval containing a time.

        See :meth:`well_interval_index` for the interval convention.

        Parameters:
            well_tag: Name of the well.
            time: The time to locate, typically the current time.

        Returns:
            Tuple ``(start, end)``, where ``end`` is the start of the next interval, or
            the end of the simulation for the last interval.

        """
        i = self.well_interval_index(well_tag, time)
        intervals = self.well_intervals(well_tag)
        if i + 1 < len(intervals):
            end = intervals[i + 1].start
        else:
            end = float(self.time_manager.schedule.get_array()[-1])
        return float(intervals[i].start), float(end)

    def well_bc_type(self, well_tag: str) -> WellBCType:
        """Return the boundary condition type of the well at the current time, as
        defined by the active interval of its schedule.

        Parameters:
            well_tag: Name of the well.

        """
        intervals = self.well_intervals(well_tag)
        return intervals[
            self.well_interval_index(well_tag, self.time_manager.time)
        ].bc_type

    def well_bc_intervals(
        self, well_tag: str, bc_type: WellBCType
    ) -> list[tuple[float, float]]:
        """Return the time ranges ``(start, end)`` in which the well has a BC type.

        Parameters:
            well_tag: Name of the well.
            bc_type: The boundary condition type to look for.

        Returns:
            One tuple per interval of the schedule of the given type. The end of the
            last interval is the end of the simulation.

        """
        intervals = self.well_intervals(well_tag)
        t_end = float(self.time_manager.schedule.get_array()[-1])
        ends = [interval.start for interval in intervals[1:]] + [t_end]
        return [
            (interval.start, end)
            for interval, end in zip(intervals, ends)
            if interval.bc_type == bc_type
        ]


class WellBoundaryConditions(WellSchedule):
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
                self.well_value(well_tag, "pressures"), "Pa"
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
                self.well_value(well_tag, "temperatures"), "K"
            )
        return values


class NeumannWellBCsFirstTimeInterval(WellSchedule):
    """Class defining Neumann BCs on well grids during the Neumann intervals of the well
    schedule, see :class:`WellInterval`. In the example, this is the first time interval.

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
        bc_type = (
            self.well_bc_type(self.parent_well_tag(sd))
            if self.is_well_grid(sd)
            else None
        )
        if bc_type == WellBCType.ROBIN:
            raise NotImplementedError("Robin boundary conditions on wells.")
        if bc_type == WellBCType.NEUMANN:
            # Impose Neumann BCs on well grids in the Neumann intervals of the well
            # schedule, e.g. before start of injection. A zero-flux condition is imposed
            # by default when no BC values are specified. Intervals are right-closed,
            # so the BCs are kept as Neumann as long as the time step ends within the
            # interval, consistent with the implicit time stepping employed in PorePy.
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
        bc_type = (
            self.well_bc_type(self.parent_well_tag(sd))
            if self.is_well_grid(sd)
            else None
        )
        if bc_type == WellBCType.ROBIN:
            raise NotImplementedError("Robin boundary conditions on wells.")
        if bc_type == WellBCType.NEUMANN:
            # Impose Neumann BCs on well grids in the Neumann intervals of the well
            # schedule, e.g. before start of injection. A zero-flux condition is imposed
            # by default when no BC values are specified. Intervals are right-closed,
            # so the BCs are kept as Neumann as long as the time step ends within the
            # interval, consistent with the implicit time stepping employed in PorePy.
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
    # Define the well schedules, one list of WellInterval per well, with the values
    # linearly interpolated between interval starts. The injection pressures are
    # somewhat arbitrary, but meant to represent a ramping up of injection pressures
    # over time. The initial low pressure represents a start from near hydrostatic
    # conditions. We ramp up from 1e5 to 5e6 Pa during initialization (well is closed
    # using a Neumann BC), then ramp up to 9e6 Pa at injection start (1 hour), then
    # increase to 11e6 Pa after 10 hours, and finally to 15e6 Pa after 100 days.
    injection_pressures = [1e5, 5e6, 9e6, 11e6, 15e6]  # [Pa]
    # Convenient shortening of simulation schedule for quick simulations. The point is
    # that the well intervals follow the time manager's schedule.
    schedule_length = schedule.size  # Change to 3 or 4 for quicker runs.
    schedule = schedule[:schedule_length]
    injection_pressures = injection_pressures[:schedule_length]
    well_schedule = {
        "injection_well": [
            WellInterval(
                start,
                {"pressures": pressure, "temperatures": 300.0},
                bc_type=(  # The well is closed during initialization.
                    WellBCType.NEUMANN if i == 0 else WellBCType.DIRICHLET
                ),
            )
            for i, (start, pressure) in enumerate(zip(schedule, injection_pressures))
        ],
        # The produced fluid is hotter than the injected one.
        "production_well": [
            WellInterval(
                start,
                {"pressures": pp.ATMOSPHERIC_PRESSURE, "temperatures": 350.0},
                bc_type=WellBCType.NEUMANN if i == 0 else WellBCType.DIRICHLET,
            )
            for i, start in enumerate(schedule)
        ],
    }

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
        "well_schedule": well_schedule,
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
