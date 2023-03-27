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

    def save_data_time_step(self) -> None:
        """Export the model state at a given time step, and log time."""
        if not self.suppress_export:
            self.exporter.write_vtu(self.data_to_export(), time_dependent=True)
            if self.restart_options["restart"]:
                global_pvd_file = self.restart_options.get(
                    "global_file", self.restart_options["pvd_file"]
                )
                self.exporter.write_pvd(append=True, from_pvd_file=global_pvd_file)
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
            scaled_values = self.equation_system.get_variable_values([var])
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
                            sd, "specific_volume", f"m^{self.nd-sd.dim}"
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
        vals_scaled = getattr(self, method_name)([grid]).evaluate(self.equation_system)
        if isinstance(vals_scaled, pp.ad.AdArray):
            vals_scaled = vals_scaled.val
        vals = self.fluid.convert_units(vals_scaled, units, to_si=True)
        return vals

    def initialize_data_saving(self) -> None:
        """Initialize data saving.

        This method is called by :meth:`prepare_simulation` to initialize the exporter,
        and any other data saving functionality (e.g., empty data containers to be
        appended in :meth:`save_data_time_step`).

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

    def load_data_from_pvd(self, options: dict, **kwargs) -> None:
        """Initialize data in the model by reading from file.

        Parameters:
            options: dictionary with restart options.

        Raises:
            ValueError: if incompatible file type provided.
        """
        # Load states and read time index, connecting data and time history
        pvd_file: Optional[str] = options.get("pvd_file")
        assert isinstance(pvd_file, str)
        if Path(pvd_file).suffix == ".vtu":
            self.exporter.import_from_vtu(pvd_file, **kwargs)
            time_index: Optional[int] = options.get("time_index")

        elif Path(pvd_file).suffix == ".pvd":
            time_index = self.exporter.import_from_pvd(pvd_file, **kwargs)

        else:
            raise ValueError("Only vtu and pvd files supported for import.")

        # Load time and time step size
        times_file: Optional[str] = options.get("times_file", None)
        self.time_manager.load_time_information(times_file)
        assert isinstance(time_index, int)
        self.time_manager.set_from_history(time_index)
        self.exporter._time_step_counter = time_index

    @property
    def suppress_export(self) -> bool:
        """Suppress export of data to file."""
        return self.params.get("suppress_export", False)


class VerificationDataSaving(DataSavingMixin):
    """Class to store relevant data for a generic verification setup."""

    _nonlinear_iteration: int
    """Number of non-linear iterations needed to solve the system. Used only as an
    indicator to avoid saving the initial conditions.

    """

    _is_time_dependent: Callable[[], bool]
    """Whether the problem is time-dependent."""

    results: list
    """List of objects containing the results of the verification."""

    def save_data_time_step(self) -> None:
        """Save data to the `results` list."""
        if not self._is_time_dependent():  # stationary problem
            if self._nonlinear_iteration > 0:  # avoid saving initial condition
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
