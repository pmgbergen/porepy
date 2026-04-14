"""This mixin class is used to save data from a model to file.

It is combined with a Model class and provides methods for saving data to file.

We provide basic Exporter functionality, but the user is free to override and extend
this class to suit their needs. This could include, e.g., saving data to a database,
or to a file format other than vtu.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import NDArray

import porepy as pp
from porepy.viz.exporter import DataInput


class DataSavingMixin(pp.PorePyModel):
    """Class for saving data from a simulation model.

    Contract with other classes:
        The model should/may call save_data_time_step() at the end of each time step.

    """

    def save_data_time_step(self) -> None:
        """Export the model state at a given time step and log time.

        The options for exporting times can be given as ``params['times_to_export']``:

        - ``None``: All time steps are exported.
        - ``list``: Export if time is in the list. If the list is empty, then no
          times are exported.

        In addition, save the solver statistics to file if the option is set.

        Finally, :meth:`collect_data` is called and stored in :attr:`results` for
        data collection and verification in runtime.

        """

        # Fetching the desired times to export.
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

        # Save solver statistics to file.
        self.nonlinear_solver_statistics.save()

        # Collecting and storing data in runtime for analysis. If default value of None
        # is returned, nothing is stored to not burden memory.
        if not self._is_time_dependent():
            collected_data = self.collect_data()
            if collected_data is not None:
                self.results.append(collected_data)
        else:
            t = self.time_manager.time  # current time
            scheduled = self.time_manager.schedule[1:]  # scheduled times except t_init
            if any(np.isclose(t, scheduled)):
                collected_data = self.collect_data()
                if collected_data is not None:
                    self.results.append(collected_data)

    def collect_data(self) -> Any:
        """Collect relevant simulation data to be stored in attr:`results`.

        Override to collect data respectively. By default, this method returns None and
        nothing is stored.

        For stationary problems, this method is called in every iteration. For time
        dependent problems, it is called after convergence of a time step which is
        scheduled by the time manager.

        Returns:
            Any data structure relevant for future verification. By default, None.
            If it is not None, it is stored in :attr:`results`.

        """
        return None

    def write_pvd_and_vtu(self) -> None:
        """Helper function for writing the .vtu and .pvd files and time information."""
        self.time_manager.write_time_information(
            Path(self.params["folder_name"]) / "times.json"
        )
        self.exporter.write_vtu(self.data_to_export(), time_dependent=True)
        times = np.array(self.time_manager.exported_times)
        if self.restart_options.get("restart", False):
            # For a pvd file addressing all time steps (before and after restart
            # time), resume based on restart input pvd file through append.
            pvd_file = self.restart_options["pvd_file"]
            self.exporter.write_pvd(times=times, append=True, from_pvd_file=pvd_file)
        else:
            self.exporter.write_pvd(times=times)

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
                variables=[var], iterate_index=0
            )
            units = var.tags["si_units"]
            values = self.units.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))

        # Add secondary variables/derived quantities.
        # All models are expected to have the dimension reduction methods for aperture
        # and specific volume. More methods may be added as needed, e.g. by overriding
        # this method as follows:
        #   def data_to_export(self):
        #       data = super().data_to_export()
        #       sds = self.mdg.subdomains()
        #       cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        #       Add more derived quantities here:
        #       q = self.evaluate_and_scale(sds, "some_derived_quantity", "m^3")
        #       for id, sd in enumerate(sds):
        #           data.append(
        #               (
        #                   sd,
        #                   "some_derived_quantity",
        #                   q[cell_offsets[id] : cell_offsets[id + 1]],
        #               )
        #           )
        #       return data

        # We combine grids and mortar grids. This is supported by the exporter, but not
        # by the type hints in the exporter module. Hence, we ignore the type hints.
        return data  # type: ignore[return-value]

    def evaluate_and_scale(
        self,
        grids: Sequence[pp.Grid] | Sequence[pp.MortarGrid],
        method_name: str,
        units: str,
    ) -> np.ndarray:
        """Evaluate a method for a derived quantity and scale the result to SI units.

        Parameters:
            grids: Sequence of grids or mortar grids for which the method should be
                evaluated.
            method_name: Name of the method to be evaluated.
            units: Units of the quantity returned by the method. Should be parsable by
                :meth:`porepy.models.units.Units.convert_units`.

        Returns:
            Array of values for the quantity, scaled to SI units.

        """
        vals_scaled = cast(
            np.ndarray,
            self.equation_system.evaluate(getattr(self, method_name)(grids)),
        )
        vals = self.units.convert_units(vals_scaled, units, to_si=True)
        return vals

    def initialize_data_saving(self) -> None:
        """Initialize data saving.

        This method is called by
        :meth:`~porepy.models.solution_strategy.SolutionStrategy.prepare_simulation` to
        initialize the exporter, and any other data saving functionality
        (e.g., empty data containers to be appended in :meth:`save_data_time_step`).

        In addition, it sets the path for storing solver statistics data to file for
        each time step.

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

        if (
            "solver_statistics_file_name" in self.params
            and self.params["solver_statistics_file_name"] is not None
        ):
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
            vtu_files: Path(s) to vtu file(s).
            time_index: Index of the time step to be loaded.
            times_file: Path to json file storing history of time and time step size. If
                ``None``, the same default path as in :meth:`write_pvd_and_vtu` is used.
            keys: Keywords addressing cell data to be transferred. If ``None``, the
                mixed-dimensional grid is checked for keywords corresponding to primary
                variables identified through ``pp.TIME_STEP_SOLUTIONS``.
            **kwargs: See documentation of
                :meth:`porepy.viz.exporter.Exporter.import_state_from_vtu`

        Raises:
            ValueError: If incompatible file types are provided.

        """
        # Sanity check
        if not (
            isinstance(vtu_files, list)
            and all([vtu_file.suffix == ".vtu" for vtu_file in vtu_files])
        ) and not (isinstance(vtu_files, Path) and vtu_files.suffix == ".vtu"):
            raise ValueError

        # Default path for times_file.
        if times_file is None:
            times_file = Path(self.params["folder_name"]) / "times.json"

        # Load states and read time index, connecting data and time history.
        self.exporter.import_state_from_vtu(vtu_files, keys, **kwargs)

        # Load time and time step size.
        self.time_manager.load_time_information(times_file)
        self.time_manager.set_time_and_dt_from_exported_steps(time_index)
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
            pvd_file: Path to pvd file with exported vtu files.
            is_mdg_pvd: Flag controlling whether pvd file is a mdg file, i.e., generated
                with ``Exporter._export_mdg_pvd()`` or ``Exporter.write_pvd()``.
            times_file: Path to json file storing history of time and time step size. If
                ``None``, the same default path as in :meth:`write_pvd_and_vtu` is used.
            keys: Keywords addressing cell data to be transferred. If ``None``, the
                mixed-dimensional grid is checked for keywords corresponding to primary
                variables identified through ``pp.TIME_STEP_SOLUTIONS``.

        Raises:
            ValueError: if incompatible file type provided.

        """
        # Sanity check
        if not pvd_file.suffix == ".pvd":
            raise ValueError

        # Default path for times_file.
        if times_file is None:
            times_file = Path(self.params["folder_name"]) / "times.json"

        # Import data and determine time index corresponding to the pvd file.
        time_index: int = self.exporter.import_from_pvd(pvd_file, is_mdg_pvd, keys)

        # Load time and time step size.
        self.time_manager.load_time_information(times_file)
        self.time_manager.set_time_and_dt_from_exported_steps(time_index)
        self.exporter._time_step_counter = time_index


class IterationExporting(pp.PorePyModel):
    if TYPE_CHECKING:
        data_to_export: Callable[[], Any]

    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Having a separate exporter for iterations avoids distinguishing between
        # iterations and time steps in the regular exporter's history.
        folder = Path(self.params["folder_name"])
        folder_iterations = folder.parent / (folder.name + "_iterations")
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"],
            folder_name=folder_iterations,
            length_scale=self.units.m,
        )

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Override to customize data to be exported at each nonlinear iteration.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        return self.data_to_export()

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.
        """
        # To make sure the nonlinear iteration index does not interfere with the time
        # part, we multiply the latter by the next power of ten above the maximum number
        # of nonlinear iterations. Default value set to 10 in accordance with the
        # default value used in NewtonSolver.
        n = self.params.get("nl_max_iterations", 10)
        r = np.ceil(np.log10(n + 1))
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iterations
            + 10**r * self.time_manager.time_index,
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)  # type: ignore[misc]
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()

    def prepare_simulation(self):
        """Prepare simulation.

        This method is called before the simulation starts. It initializes the iteration
        exporter and writes the initial state to a vtu file.

        """
        super().prepare_simulation()
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()


class ResidualExporting:
    """Class for exporting residuals of the equation system.

    Note: This class is primarily intended for debugging purposes in combination with
    the IterationExporting mixin. At the end of a time step, the residuals of
    accumulation terms may be non-zero due to time and iteration solution coinciding.
    """

    if TYPE_CHECKING:
        equation_system: pp.EquationSystem

    def data_to_export(self) -> list[DataInput]:
        """Return data to be exported, including residuals.

        Returns:
            List containing all (grid, name, values) tuples.

        """
        data = super().data_to_export()  # type: ignore[misc]

        # Add residuals. Loop over equations in the equation system.
        for name, operator in self.equation_system.equations.items():
            residuals = cast(NDArray, self.equation_system.evaluate(operator))

            # Get image_info as dict[GridEntity, int], where
            # GridEntity = Literal["cells", "faces", "nodes"]
            image_info = self.equation_system.equation_image_size_info[name]
            dof_start, dof_end = 0, 0
            for g in self.equation_system.equation_image_space_composition[name].keys():
                # Add number of dofs for each entity in image_info.
                for entity, num in image_info.items():
                    dof_end += getattr(g, "num_" + entity) * num
                # Append residuals for current grid.
                data.append(
                    (
                        g,
                        "residual_" + name,
                        residuals[dof_start:dof_end],
                    )
                )
                # Update dof_start for next grid.
                dof_start = dof_end
        return data


class FractureDeformationExporting(pp.PorePyModel):
    """Class for exporting fracture-specific quantities.

    Adds the fracture-specific secondary variables
    - displacement jump,
    - aperture,
    - rescaled traction,
    - slip tendency = -||shear_traction||/normal_traction
    """

    def data_to_export(self) -> list:
        """Returns data for exporting.

        Returns:
            A list of tuples (subdomain, variable name, variable values).
        """
        # Start with data from super class. This includes standard variables.
        data = super().data_to_export()  # type: ignore[misc]
        # Add the three fracture-specific vector quantities.
        sds = self.mdg.subdomains(dim=self.nd - 1)
        cell_offsets_nd = np.cumsum([0] + [sd.num_cells * self.nd for sd in sds])
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        displacement_jump = self.evaluate_and_scale(sds, "displacement_jump", "m")
        char = self.evaluate_and_scale(sds, "characteristic_contact_traction", "Pa")
        friction_coefficient = self.evaluate_and_scale(sds, "friction_coefficient", "")

        # Both characteristic traction and friction coefficients are frequently floats.
        # Ensure they are arrays to allow element-wise operations below, thus covering
        # the case of spatially homogeneous quantities. Heterogeneous quantities are
        # at least possible for the friction coefficient.
        size = sum([sd.num_cells for sd in sds])

        def ensure_array(
            quantity: NDArray | float,
        ) -> NDArray:
            if isinstance(quantity, float):
                # Cast to cell-wise array.
                return quantity * np.ones(size)
            else:
                return quantity

        char = ensure_array(char)
        friction_coefficient = ensure_array(friction_coefficient)
        traction = self.evaluate_and_scale(sds, "contact_traction", "-").reshape(
            (self.nd, -1), order="F"
        )
        # Compute apertures, which are scalar quantities.
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        apertures = self.evaluate_and_scale(sds, "aperture", "m")
        slip_tendency = self.compute_slip_tendency(traction, friction_coefficient)

        # Loop over the fracture subdomains.
        for id, sd in enumerate(sds):
            # Export the displacement jump.
            data.append(
                (
                    sd,
                    "displacement_jump",
                    displacement_jump[cell_offsets_nd[id] : cell_offsets_nd[id + 1]],
                )
            )
            # Export the slip tendency, defined as the ratio of the shear traction to
            # the normal traction.
            data.append(
                (
                    sd,
                    "slip_tendency",
                    slip_tendency[cell_offsets[id] : cell_offsets[id + 1]],
                )
            )
            # Rescale traction by characteristic contact traction.
            traction_loc = traction[:, cell_offsets[id] : cell_offsets[id + 1]]
            traction_loc *= char[cell_offsets[id] : cell_offsets[id + 1]]
            data.append((sd, "contact_traction_in_Pa", traction_loc.ravel("F")))

            data.append(
                (
                    sd,
                    "aperture",
                    apertures[cell_offsets[id] : cell_offsets[id + 1]],
                )
            )
        return data

    @staticmethod
    def compute_slip_tendency(
        traction: np.ndarray,
        friction_coefficient: np.ndarray,
        atol: float = 1e-8,
    ) -> np.ndarray:
        """Compute slip tendency as the ratio of shear traction to normal traction.

        For cells where normal traction is zero (open fractures), slip tendency is
        physically undefined and set to NaN.

        Parameters:
            traction: Array of shape (nd, num_cells) containing the traction vector for
                each cell.
            friction_coefficient: Array of shape (num_cells,) containing the friction
                coefficient for each cell.
            atol: Absolute tolerance for detecting zero normal traction. Defaults to
                1e-8, matching numpy's default.

            Returns:
                Array of shape (num_cells,) containing the slip tendency for each cell.
                NaN values indicate cells with zero normal traction (open fractures).

        Notes:
            Slip tendency convention:
            - Positive values indicate tendency to slip (shear traction exceeds
              friction limit).
            - Compressive normal traction is negative.
            - Regular values are positive for fractures in contact.
            - NaN indicates undefined slip tendency (zero normal traction).
        """
        # Compute slip tendency.
        # Minus to follow convention that positive slip tendency indicates slip and
        # compressive normal traction is negative.
        slip_tendency = -np.linalg.norm(traction[:-1], axis=0) / (
            traction[-1] * friction_coefficient
        )

        # Set to NaN where normal traction is zero (undefined).
        zero_inds = np.isclose(traction[-1], 0, atol=atol)
        slip_tendency[zero_inds] = np.nan

        return slip_tendency
