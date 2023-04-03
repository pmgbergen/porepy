"""Module containing a class for performing spatio-temporal convergence analysis."""
from __future__ import annotations

from copy import deepcopy
from typing import Optional, Union, NamedTuple
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import porepy as pp


class PlotVar(NamedTuple):
    """Class for representing a variable that will be plotted in a convergence plot."""

    name: str
    """Name of the attribute to be accessed in a result data class, e.g.,
    `error_matrix_pressure`.
    
    """

    label: str
    """Label of the variable that should be included in the legend of the plot."""

    value: np.ndarray
    """Value of the variable. Typically, some type of L2-error."""


class ConvergenceAnalysis:
    """Class for performing a convergence analysis for a given model.

    Current support:
        - Simplicial and Cartesian grids in 2d and 3d. Tensor grids are not supported.
        - Static and time-dependent models.
        - If the model is time-dependent, we assume that the ``TimeManager`` is
          instantiated with a constant time step.

    Parameters:
        model_class: Model class for which the analysis will be performed.
        model_params: Model parameters. We assume that it contains all parameters
            necessary to set up a valid instance of :data:`model_class`.
        levels: ``default=1``

            Number of refinement levels associated to the convergence analysis
        in_space: ``default=True``

            Whether convergence in space should be performed.
        in_time:  ``default=False``

            Whether convergence in time should be performed.
        spatial_rate: ``default=1``

            Rate at which the mesh size(s) should be refined. For example, use
            ``spatial_rate=2`` for halving the mesh size(s) in-between levels.
        temporal_rate: ``default=1``

            Rate at which the time step size should be refined. For example,
            use ``temporal_rate=2`` for halving the time step size in-between levels.

    """

    def __init__(
        self,
        model_class,
        model_params: dict,
        levels: int = 1,
        in_space: bool = True,
        in_time: bool = False,
        spatial_rate: int = 1,
        temporal_rate: int = 1,
    ):
        # Sanity check
        if not in_space and not in_time:
            raise ValueError("At least one type of analysis should be performed.")

        self.model_class = model_class
        """Model class that should be used to run the simulations and perform the
        convergence analysis.

        """

        self.levels: int = levels
        """Number of levels of the convergence analysis."""

        self.in_space: bool = in_space
        """Whether a spatial analysis should be performed."""

        self.in_time: bool = in_time
        """Whether a temporal analysis should be performed."""

        self.spatial_rate: int = spatial_rate
        """Rate at which the mesh size should be refined. A value of ``2``
        corresponds to halving the mesh size(s) between :attr:`levels`.

        """

        self.temporal_rate: int = temporal_rate
        """Rate at which the time step size should be refined. A value of ``2``
        corresponds to halving the time step size between :attr:`levels`.

        """

        # Initialize setup and retrieve spatial and temporal data
        setup = model_class(model_params)
        setup.prepare_simulation()

        self._init_setup = setup
        """"Initial setup."""

        self._is_time_dependent: bool = setup._is_time_dependent()
        """Whether the problem is time-dependent or not."""

        # Sanity check
        if not self._is_time_dependent and self.in_time:
            raise ValueError("Analysis in time not available for stationary models.")

        # Check whether the grid is simplicial or not
        g = setup.mdg.subdomains()[0]
        if isinstance(g, pp.TriangleGrid) or isinstance(g, pp.TetrahedralGrid):
            self._is_simplicial = True
        else:
            self._is_simplicial = False

        # Retrieve list of mesh arguments
        list_of_mesh_arguments: list[
            Union[dict[str, pp.number], np.ndarray]
        ] = self._get_list_of_mesh_arguments()

        # Retrieve list of time managers
        list_of_time_managers: Union[
            list[pp.TimeManager], None
        ] = self._get_list_of_time_managers()

        # Create list of model parameters
        list_of_params: list[dict] = []
        for lvl in range(self.levels):
            params = deepcopy(model_params)
            params["mesh_arguments"] = list_of_mesh_arguments[lvl]
            if list_of_time_managers is not None:
                params["time_manager"] = list_of_time_managers[lvl]
            list_of_params.append(params)

        self.model_params: list[dict] = list_of_params
        """List of model parameters associated to each run."""

    def _get_list_of_mesh_arguments(self) -> list[dict[str, pp.number]]:
        """Obtain list of mesh arguments.

        Returns:
            List of mesh arguments. Length of list is ``levels``.

        """
        # Retrieve initial mesh arguments
        init_mesh_args = deepcopy(self._init_setup.mesh_arguments())

        # Prepare factors for the spatial analysis
        factors = 1 / (self.spatial_rate ** np.arange(self.levels))

        # Loop over levels
        if self._is_simplicial:
            list_mesh_args: list[dict[str, pp.number]] = []
            for lvl in range(self.levels):
                factor: pp.number = factors[lvl]
                mesh_args = {}
                for key in init_mesh_args:
                    mesh_args[key] = init_mesh_args[key] * factor
                list_mesh_args.append(mesh_args)
        else:
            list_mesh_args: list[dict[str, pp.number]] = []
            for lvl in range(self.levels):
                factor: pp.number = factors[lvl]
                mesh_args = {}
                for key in init_mesh_args:
                    mesh_args[key] = init_mesh_args[key] * factor
                list_mesh_args.append(mesh_args)

        return list_mesh_args

    def _get_list_of_time_managers(self) -> Union[list[pp.TimeManager], None]:
        """Obtain list of time managers.

        Returns:
            List of time managers. Length of list is ``levels``.

        """
        if self._is_time_dependent:
            init_time_manager: pp.TimeManager = self._init_setup.time_manager
        else:
            return None

        # Sanity check
        if not init_time_manager.is_constant:
            msg = "Analysis in time only supports constant time step."
            raise NotImplementedError(msg)

        # Prepare factors for the temporal analysis
        factors = 1 / (self.temporal_rate ** np.arange(self.levels))

        # Loop over levels
        list_time_managers: list[pp.TimeManager] = []
        for lvl in range(self.levels):
            factor = factors[lvl]
            time_manager = pp.TimeManager(
                schedule=init_time_manager.schedule,
                dt_init=init_time_manager.dt_init * factor,
                constant_dt=True,
            )
            list_time_managers.append(time_manager)

        return list_time_managers

    def run_analysis(self) -> list:
        """Run convergence analysis."""
        convergence_results: list = []
        for level in range(self.levels):
            setup = self.model_class(self.model_params[level])
            if not setup._is_time_dependent():
                # Run stationary model
                pp.run_stationary_model(setup, self.model_params[level])
                # Complement information in results
                setattr(setup.results[-1], "num_dofs", setup.equation_system.num_dofs())
                setattr(setup.results[-1], "cell_diam", setup.mdg.diameter())
            else:
                # Run time-dependent model
                pp.run_time_dependent_model(setup, self.model_params[level])
                # Complement information in results
                setattr(setup.results[-1], "num_dofs", setup.equation_system.num_dofs())
                setattr(setup.results[-1], "cell_diam", setup.mdg.diameter())
                setattr(setup.results[-1], "dt", setup.time_manager.dt)

            convergence_results.append(setup.results[-1])
        return convergence_results

    def order_of_convergence(
            self,
            list_of_results: list,
            variables: Optional[list[str]] = None,
            in_space=True,
            in_time=False,
    ) -> dict[str, float]:
        """Compute order of convergence (OOC) for a given set of variables.

        Raises:
            ValueError
                If both `in_space` and `in_time` are set to True simultaneously.

            ValueError
                If neither `in_space` nor `in_time` are set to True.

            ValueError
                If `in_time=True` but the model is stationary.

        Parameters:
            list_of_results: List containing the results of the convergence analysis.
                Typically, the output of ``self.run_analysis()``.
            variables: names of the variables for which the OOC should be computed. The
                ``item`` of the list must match the attribute ``"error_" + item`` from
                each item from the ``list_of_results``. If not given, all attributes
                starting with "error_" will be included in the analysis.
            in_space: Whether the OOC should be computed w.r.t. the inverse of the
                cell diameter.
            in_time: Whether the OOC should be computed w.r.t. the inverse of the
                time step size.

        Returns:
            List of tuples. Each item of the list corresponds to a pair ``(var, ooc)``,
            where ``var: str`` is the name of the variable (from the list of variables)
            and ``ooc: float`` is the estimated OOC obtained via linear regression.

        """
        # Sanity checks
        if in_space and in_time:
            raise ValueError("Both 'in_space' and 'in_time' cannot be True.")
        if not in_space and not in_time:
            raise ValueError("Expected either 'in_space' or 'in_time' to be True.")
        if in_time and not self._is_time_dependent:
            raise ValueError("Analysis in time requires a time-dependent model.")

        # Obtain x-values
        if in_space:
            cell_diams = np.array([result.cell_diam for result in list_of_results])
            # Invert and apply logarithm in space
            x_vals = self.log_space(1 / cell_diams)
        else:
            time_steps = np.array([result.dt for result in list_of_results])
            # Invert and apply logarithm in time
            x_vals = self.log_time(1 / time_steps)

        # Get variable names and labels
        if variables is None:
            # Retrieve all attributes from the data class
            attributes: list[str] = list(vars(list_of_results[0]).keys())
            # Filter attributes that start with ``error_``
            names: list[str] = [att for att in attributes if att.startswith("error_")]
        else:
            names = ["error_" + att for att in variables]

        # Obtain y-values
        y_vals = []
        for name in names:
            y_val = []
            # Loop over lists of results
            for result in list_of_results:
                y_val.append(getattr(result, name))
            # Append to the `y_vals` list, using the right log base
            if in_space:
                y_vals.append(self.log_space(np.array(y_val)))
            else:
                y_vals.append(self.log_time(np.array(y_val)))

        # Perform linear regression and populate return dictionary
        # Keywords of the dictionary will have the prefix 'ooc_' before the `name`
        # Values of order of convergence will correspond to the negative of the slope
        ooc_dict = {}
        for idx, name in enumerate(names):
            slope, intercept, r, p, std_err = stats.linregress(x_vals, y_vals[idx])
            ooc_name = "ooc_" + name.lstrip("error_")
            ooc_val = -slope
            ooc_dict[ooc_name] = ooc_val

        return ooc_dict

    def plot_spatial_rates(
            self,
            list_of_results: list,
            vars_to_plot: Optional[list[tuple[str, str]]] = None,
            plot_first_order_line=False,
            plot_second_order_line=False,
            save_img=False,
    ) -> None:
        """Convergence plot in space.

        Parameters:
            list_of_results: List of ``results`` data classes associated with a given
                model class.
            vars_to_plot: List of 2-tuples containing the name and label identifying
                the errors associated with the variables that should be included in the
                plot. Note, however, that the name should NOT include the prefix
                ``error_``. The label can be arbitrary and is only used for plotting
                purposes. For example, if the error associated to the pressure in the
                highest-dimensional domain is ``error_matrix_pressure``, a valid
                2-tuple is ("matrix_pressure", "p2"). If ``vars_to_plot`` is not
                given, all attributes whose prefix match the string "error_" will be
                included in the plot, and the suffix will be used as a label.
                Defaults to None.
            plot_first_order_line: Whether to include the reference line
                corresponding to first order convergence. Note that some manual
                tweaking might be required to position the line in a decent place. If
                that is the case, you should override the method ``plot_first_order()``.
            plot_second_order_line: Whether to include the reference line
                corresponding to second order convergence. Note that some manual
                tweaking might be required to position the line in a decent place. If
                that is the case, you should override the method
                ``plot_second_order()``.
            save_img: Whether to save the image in PDF format. The plot will be save
                under the name "convergence_plot.pdf".

        """

        # Get x_axis
        mesh_sizes = np.array([result.cell_diam for result in list_of_results])

        # Get variable names and labels
        if vars_to_plot is None:
            # Retrieve all attributes from the data class
            attributes: list[str] = list(vars(list_of_results[0]).keys())
            # Filter attributes that start with ``error_``
            names: list[str] = [att for att in attributes if att.startswith("error_")]
            # Assign as label the suffix of ``error_`` corresponding to that name
            labels: list[str] = [att[6:] for att in names]
        else:
            names, labels = [], []
            for var in vars_to_plot:
                names.append("error_" + var[0])
                labels.append(var[1])

        # Now we can construct the named tuple containing name, label, and vals
        plot_vars = []
        for name, label in zip(names, labels):
            vals = []
            for result in list_of_results:
                vals.append(getattr(result, name))
            plot_vars.append(PlotVar(name=name, label=label, value=np.asarray(vals)))

        # Plot
        cmap = mcolors.ListedColormap(plt.cm.tab10.colors[: len(plot_vars)])
        fig, ax = plt.subplots(nrows=1, ncols=2)

        # Plot first order convergence rate line
        if plot_first_order_line:
            self.plot_first_order(ax[0], ax[1], mesh_sizes)

        # Plot second order convergence rate line
        if plot_second_order_line:
            self.plot_second_order(ax[0], ax[1], mesh_sizes)

        # Plot data
        for idx, var in enumerate(plot_vars):

            # Data plot
            ax[0].plot(
                self.log_space(1/mesh_sizes),
                self.log_space(var.value),
                linestyle="-",
                linewidth=3,
                marker="o",
                markersize=6,
                color=cmap.colors[idx],
            )
            ax[0].set_xlabel(fr"$\log_{self.spatial_rate}$($1/h$)", fontsize=14)
            ax[0].set_ylabel(fr"$\log_{self.spatial_rate}$(error)", fontsize=14)
            ax[0].grid(True)

            # Legend plot
            ax[1].plot(
                [],
                [],
                linestyle="-",
                linewidth=3,
                marker="o",
                markersize=6,
                color=cmap.colors[idx],
                label=var.label
            )

            # Add legend
            ax[1].plot()
            ax[1].axis("off")
            ax[1].legend(
                bbox_to_anchor=(1.25, 0.5),
                loc="center right",
                fontsize=13,
            )

        plt.tight_layout()
        # plt.show()

        if save_img:
            plt.savefig(fname="convergence_plot.pdf")

    # -----> Utility methods
    def log_space(self, array: np.ndarray) -> np.ndarray:
        return np.emath.logn(self.spatial_rate, array)

    def log_time(self, array: np.ndarray) -> np.ndarray:
        return np.emath.logn(self.temporal_rate, array)

    # -----> Auxiliary plotting methods
    def plot_first_order(
            self,
            data_axis,
            legend_axis,
            mesh_sizes: np.ndarray,
    ) -> None:
        """Plot first order line.

        Parameters:
            data_axis: Axis corresponding to the left plot (the data plot).
            legend_axis: Axis corresponding to the righ plot (the legend plot).
            mesh_sizes: Array containing the target mesh sizes.

        """
        x0 = self.log_space(1/mesh_sizes[0])
        x1 = self.log_space(1/mesh_sizes[-1])
        y0 = -3  # this often requires tweaking
        y1 = y0 - (x1 - x0)
        data_axis.plot(
            [x0, x1],
            [y0, y1],
            "k-",
            linewidth=3,
            label="First order",
        )
        legend_axis.plot(
            [],
            [],
            "k-",
            linewidth=3,
            label="First order",
        )

    def plot_second_order(
            self,
            data_axis,
            legend_axis,
            mesh_sizes: np.ndarray
    ) -> None:
        """Plot second order line.

        Parameters:
            data_axis: Axis corresponding to the left plot (the data plot).
            legend_axis: Axis corresponding to the righ plot (the legend plot).
            mesh_sizes: Array containing the target mesh sizes.

        """
        x0 = self.log_space(1/mesh_sizes[0])
        x1 = self.log_space(1/mesh_sizes[-1])
        y0 = -2  # this requires tweaking
        y1 = y0 - 2 * (x1 - x0)
        data_axis.plot(
            [x0, x1],
            [y0, y1],
            "k--",
            linewidth=3,
        )
        legend_axis.plot(
            [],
            [],
            "k--",
            linewidth=3,
            label="Second order",
        )


