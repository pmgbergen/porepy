"""This mixin class is used to save data from a model to file.

It is combined with a Model class and provides methods for saving data to file.

We provide basic Exporter functionality, but the user is free to override and extend
this class to suit their needs. This could include, e.g., saving data to a database,
or to a file format other than vtu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

import porepy as pp
from porepy.viz.exporter import DataInput


class DataSavingMixin:
    """Class for saving data from a simulation model.

    Contract with other classes:
        The model should/may call save_data_time_step() at the end of each time step.

    """

    equation_system: pp.EquationSystem
    """Equation system manager."""
    params: dict[str, Any]
    """Dictionary of parameters. May contain data saving parameters."""
    time_manager: pp.TimeManager
    """Time manager for the simulation."""
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the simulation."""
    restart_options: dict
    """Dictionary of parameters for restarting from pvd."""
    units: pp.Units
    """Units for the simulation."""
    fluid: pp.FluidConstants
    """Fluid constants for the simulation."""
    nd: int
    """Number of spatial dimensions for the simulation."""
    nonlinear_solver_statistics: pp.SolverStatistics
    """Non-linear solver statistics for nonlinear solver."""

    def save_data_time_step(self) -> None:
        """Export the model state at a given time step, and log time.
        The options for exporting times are:
            * None: All time steps are exported
            * list: Export if time is in the list. If the list is empty, then no times
              are exported.

        In addition, save the solver statistics to file if the option is set.

        """

        # Fetching the desired times to export
        times_to_export = self.params.get("times_to_export", None)
        if times_to_export is None:
            # Export all time steps if times are not specified.
            do_export = True
        else:
            # If times are specified, export should only occur if the current time is in
            # the list of times to export.
            do_export = bool(
                np.any(np.isclose(self.time_manager.time, times_to_export))
            )

        if do_export:
            self.write_pvd_and_vtu()

        # Save solver statistics to file
        self.nonlinear_solver_statistics.save()

    def write_pvd_and_vtu(self) -> None:
        """Helper function for writing the .vtu and .pvd files and time information."""
        self.exporter.write_vtu(self.data_to_export(), time_dependent=True)
        if self.restart_options.get("restart", False):
            # For a pvd file addressing all time steps (before and after restart
            # time), resume based on restart input pvd file through append.
            pvd_file = self.restart_options["pvd_file"]
            self.exporter.write_pvd(append=True, from_pvd_file=pvd_file)
        else:
            self.exporter.write_pvd()
        self.time_manager.write_time_information()

    def data_to_export(self) -> list[DataInput]:
        """Return data to be exported.

        Return type should comply with pp.exporter.DataInput.

        Returns:
            List containing all (grid, name, scaled_values) tuples.

        """
        data = []
        variables = self.equation_system.variables
        for var in variables:
            scaled_values = self.equation_system.get_variable_values(
                variables=[var], time_step_index=0
            )
            units = var.tags["si_units"]
            values = self.fluid.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))

        # Add secondary variables/derived quantities.
        # All models are expected to have the dimension reduction methods for aperture
        # and specific volume. More methods may be added as needed, e.g. by overriding
        # this method:
        #   def data_to_export(self):
        #       data = super().data_to_export()
        #       data.append(
        #           (grid, "name", self._evaluate_and_scale(sd, "name", "units"))
        #       )
        #       return data
        for dim in range(self.nd + 1):
            for sd in self.mdg.subdomains(dim=dim):
                if dim < self.nd:
                    data.append(
                        (sd, "aperture", self._evaluate_and_scale(sd, "aperture", "m"))
                    )
                data.append(
                    (
                        sd,
                        "specific_volume",
                        self._evaluate_and_scale(
                            sd, "specific_volume", f"m^{self.nd - sd.dim}"
                        ),
                    )
                )

        # We combine grids and mortar grids. This is supported by the exporter, but not
        # by the type hints in the exporter module. Hence, we ignore the type hints.
        return data  # type: ignore[return-value]

    def _evaluate_and_scale(
        self,
        grid: Union[pp.Grid, pp.MortarGrid],
        method_name: str,
        units: str,
    ) -> np.ndarray:
        """Evaluate a method for a derived quantity and scale the result to SI units.

        Parameters:
            grid: Grid or mortar grid for which the method should be evaluated.
            method_name: Name of the method to be evaluated.
            units: Units of the quantity returned by the method. Should be parsable by
                :meth:`porepy.fluid.FluidConstants.convert_units`.

        Returns:
            Array of values for the quantity, scaled to SI units.

        """
        vals_scaled = getattr(self, method_name)([grid]).value(self.equation_system)
        vals = self.fluid.convert_units(vals_scaled, units, to_si=True)
        return vals

    def initialize_data_saving(self) -> None:
        """Initialize data saving.

        This method is called by :meth:`prepare_simulation` to initialize the exporter,
        and any other data saving functionality (e.g., empty data containers to be
        appended in :meth:`save_data_time_step`).

        In addition, set path for storing solver statistics data to file for each time step.

        """
        self.exporter = pp.Exporter(
            self.mdg,
            self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
            length_scale=self.units.m,
        )

        if "solver_statistics_file_name" in self.params:
            self.nonlinear_solver_statistics.path = (
                Path(self.params["folder_name"])
                / self.params["solver_statistics_file_name"]
            )

    def load_data_from_vtu(
        self,
        vtu_files: Union[Path, list[Path]],
        time_index: int,
        times_file: Optional[Path] = None,
        keys: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> None:
        """Initialize data in the model by reading from a pvd file.

        Parameters:
            vtu_files: path(s) to vtu file(s)
            keys: keywords addressing cell data to be transferred. If 'None', the
                mixed-dimensional grid is checked for keywords corresponding to primary
                variables identified through pp.TIME_STEP_SOLUTIONS.
            keyword arguments: see documentation of
                :meth:`porepy.viz.exporter.Exporter.import_state_from_vtu`

        Raises:
            ValueError: if incompatible file type provided.

        """
        # Sanity check
        if not (
            isinstance(vtu_files, list)
            and all([vtu_file.suffix == ".vtu" for vtu_file in vtu_files])
        ) and not (isinstance(vtu_files, Path) and vtu_files.suffix == ".vtu"):
            raise ValueError

        # Load states and read time index, connecting data and time history
        self.exporter.import_state_from_vtu(vtu_files, keys, **kwargs)

        # Load time and time step size
        self.time_manager.load_time_information(times_file)
        self.time_manager.set_from_history(time_index)
        self.exporter._time_step_counter = time_index

    def load_data_from_pvd(
        self,
        pvd_file: Path,
        is_mdg_pvd: bool = False,
        times_file: Optional[Path] = None,
        keys: Optional[Union[str, list[str]]] = None,
    ) -> None:
        """Initialize data in the model by reading from a pvd file.

        Parameters:
            pvd_file: path to pvd file with exported vtu files.
            is_mdg_pvd: flag controlling whether pvd file is a mdg file, i.e., generated
                with Exporter._export_mdg_pvd() or Exporter.write_pvd().
            times_file: path to json file storing history of time and time step size.
            keys: keywords addressing cell data to be transferred. If 'None', the
                mixed-dimensional grid is checked for keywords corresponding to primary
                variables identified through pp.TIME_STEP_SOLUTIONS.

        Raises:
            ValueError: if incompatible file type provided.

        """
        # Sanity check
        if not pvd_file.suffix == ".pvd":
            raise ValueError

        # Import data and determine time index corresponding to the pvd file
        time_index: int = self.exporter.import_from_pvd(pvd_file, is_mdg_pvd, keys)

        # Load time and time step size
        self.time_manager.load_time_information(times_file)
        self.time_manager.set_from_history(time_index)
        self.exporter._time_step_counter = time_index


class VerificationDataSaving(DataSavingMixin):
    """Class to store relevant data for a generic verification setup."""

    nonlinear_solver_statistics: pp.SolverStatistics
    """Non-linear solver statistics, also keeping track of the number of iterations."""

    _is_time_dependent: Callable[[], bool]
    """Whether the problem is time-dependent."""

    results: list
    """List of objects containing the results of the verification."""

    def save_data_time_step(self) -> None:
        """Save data to the `results` list."""
        if not self._is_time_dependent():  # stationary problem
            if (
                self.nonlinear_solver_statistics.num_iteration > 0
            ):  # avoid saving initial condition
                collected_data = self.collect_data()
                self.results.append(collected_data)
        else:  # time-dependent problem
            t = self.time_manager.time  # current time
            scheduled = self.time_manager.schedule[1:]  # scheduled times except t_init
            if any(np.isclose(t, scheduled)):
                collected_data = self.collect_data()
                self.results.append(collected_data)

    def collect_data(self):
        """Collect relevant data for the verification setup."""
        raise NotImplementedError()
