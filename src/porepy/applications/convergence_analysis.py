"""Module containing a class for performing spatio-temporal convergence analysis."""
from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import Literal, Optional, Union, Dict, List, Callable

import numpy as np
from scipy import stats

import porepy as pp
from porepy.utils.txt_io import TxtData, export_data_to_txt

logger = logging.getLogger(__name__)

class ConvergenceAnalysis:
    """Perform spatio-temporal convergence analysis on a given model.

    The class :class:`ConvergenceAnalysis` takes a PorePy model and its parameter
    dictionary to run a batch of simulations with successively refined mesh sizes and
    time steps and collects the results (i.e., the data classes containing the errors)
    in a list using the :meth:`run_analysis` method. Refinement rates (in time and
    space) are given at instantiation.

    Useful methods of this class include

        - :meth:`run_analysis`: Runs a batch of simulations with successively refined
          mesh sizes and/or time steps.
        - :meth:`export_errors_to_txt`: Exports errors into a ``txt`` file.
        - :meth:`order_of_convergence`: Estimates the observed order of convergence
          of a given set of variables using linear regression.

    Note:
        Current support of the class includes

            - Simplicial and Cartesian grids in 2d and 3d. Tensor grids are not
              supported.

            - Static and time-dependent models.

            - If the model is time-dependent, we require the ``TimeManager`` to be
              instantiated with a constant time step.

    Raises:

        ValueError
            If ``spatial_refinement_rate`` < 1.

        ValueError
            If ``temporal_refinement_rate`` < 1.

        ValueError
            For a stationary model, if ``temporal_refinement_rate`` > 1.

        NotImplementedError
            For a time-dependent model, if the time manager contains a non-constant
            time step.

        Warning
            If both refinement rates are equal to 1.


    Parameters:
        model_class: Model class for which the analysis will be performed.
        model_params: Model parameters. We assume that it contains all parameters
            necessary to set up a valid instance of :data:`model_class`.
        levels: ``default=1``

            Number of refinement levels associated to the convergence analysis.
        spatial_refinement_rate: ``default=1``

            Rate at which the mesh size should be refined. For example, use
            ``spatial_refinement_rate=2`` for halving the mesh size in-between levels.
        temporal_refinement_rate: ``default=1``

            Rate at which the time step size should be refined. For example, use
             ``temporal_refinement_rate=2`` for halving the time step size in-between
            levels.

    """

    def __init__(
        self,
        model_class,
        model_params: dict,
        levels: int = 1,
        spatial_refinement_rate: int = 1,
        temporal_refinement_rate: int = 1,
    ):
        # Sanity checks
        if spatial_refinement_rate < 1 or temporal_refinement_rate < 1:
            raise ValueError("Refinement rate cannot be less than 1.")

        if spatial_refinement_rate == 1 and temporal_refinement_rate == 1:
            warnings.warn("No refinement (in space or time) will be performed.")

        self.model_class = model_class
        """Model class that should be used to run the simulations and perform the
        convergence analysis.

        """

        self.levels: int = levels
        """Number of levels of the convergence analysis."""

        self.spatial_refinement_rate: int = spatial_refinement_rate
        """Rate at which the mesh size should be refined. A value of ``2``
        corresponds to halving the mesh size(s) between :attr:`levels`.

        """

        self.temporal_refinement_rate: int = temporal_refinement_rate
        """Rate at which the time step size should be refined. A value of ``2``
        corresponds to halving the time step size between :attr:`levels`.

        """

        # Initialize setup and retrieve spatial and temporal data
        setup = model_class(deepcopy(model_params))  # make a deep copy of dictionary
        setup.prepare_simulation()

        # Store initial setup
        self._init_setup = setup
        """Initial setup containing the 'base-line' information."""

        # We need to know whether the model is time-dependent or not
        self._is_time_dependent: bool = setup._is_time_dependent()
        """Whether the model is time-dependent."""

        if not self._is_time_dependent and self.temporal_refinement_rate > 1:
            raise ValueError("Analysis in time not available for stationary models.")

        # Retrieve list of meshing arguments
        # The list is of length ``levels`` and contains the ``meshing_arguments``
        # dictionaries needed to run the simulations.
        list_of_meshing_arguments: list[
            dict[str, float]
        ] = self._get_list_of_meshing_arguments()

        # Retrieve list of time managers
        # The list is of length ``levels`` and contains the ``pp.TimeManager``s
        # needed to run the simulations. ``None`` if the model is stationary.
        list_of_time_managers: Union[
            list[pp.TimeManager], None
        ] = self._get_list_of_time_managers()

        # Generate list of model parameters
        # Having the initial model parameter, the list of meshing arguments, and the
        # list of time managers, we can create the ready-to-be-fed list of model
        # parameters necessary for running the simulations
        list_of_params: list[dict] = []
        for lvl in range(self.levels):
            params = deepcopy(model_params)
            params["meshing_arguments"] = list_of_meshing_arguments[lvl]
            if list_of_time_managers is not None:
                params["time_manager"] = list_of_time_managers[lvl]
            list_of_params.append(params)

        self.model_params: list[dict] = list_of_params
        """List of model parameters associated to each simulation run."""

    def run_analysis(self) -> list:
        """Run convergence analysis.

        Returns:
            List of results (i.e., data classes containing the errors) for each
            refinement level. Note that for time-dependent models, only the result
            corresponding to the final time is appended to the list.

        """
        convergence_results: list = []
        for level in range(self.levels):
            setup = self.model_class(deepcopy(self.model_params[level]))
            if not setup._is_time_dependent():
                # Run stationary model
                pp.run_stationary_model(setup, deepcopy(self.model_params[level]))
                # Complement information in results
                setattr(setup.results[-1], "cell_diameter", setup.mdg.diameter())
            else:
                # Run time-dependent model
                pp.run_time_dependent_model(setup, deepcopy(self.model_params[level]))
                # Complement information in results
                setattr(setup.results[-1], "cell_diameter", setup.mdg.diameter())
                setattr(setup.results[-1], "dt", setup.time_manager.dt)

            convergence_results.append(setup.results[-1])
        return convergence_results

    def export_errors_to_txt(
        self,
        list_of_results: list,
        variables_to_export: Optional[list[str]] = None,
        file_name="error_analysis.txt",
    ) -> None:
        """Write errors into a ``txt`` file.

        The format is the following one:

            - First column contains the cell diameters.
            - Second column contains the time steps (if the model is time-dependent).
            - The rest of the columns contain the errors for each variable in
              ``variables``.

        Parameters:
            list_of_results: List containing the results of the convergence analysis.
                Typically, the output of :meth:`run_analysis`.
            variables_to_export: names of the variables for which the TXT file will be
                generated. If ``variables`` is not given, all the variables present
                in the txt file will be collected.
            file_name: Name of the output file. Default is "error_analysis.txt".

        """
        # Filter variables from the list of results
        var_names: list[str] = self._filter_variables_from_list_of_results(
            list_of_results=list_of_results,
            variables=variables_to_export,
        )

        # Filter errors to be exported
        errors_to_export: dict[str, np.ndarray] = {}
        for name in var_names:
            # Loop over lists of results
            var_error: list[float] = []
            for result in list_of_results:
                var_error.append(getattr(result, name))
            # Append to the dictionary
            errors_to_export[name] = np.array(var_error)

        # Prepare to export
        list_of_txt_data: list[TxtData] = []
        # Append cell diameters
        cell_diameters = np.array([result.cell_diameter for result in list_of_results])
        list_of_txt_data.append(
            TxtData(
                header="cell_diameter",
                array=cell_diameters,
                format=self._set_column_data_format("cell_diameter"),
            )
        )
        # Append time steps (if the analysis employs a time-dependent model)
        if self._is_time_dependent:
            time_steps = np.array([result.dt for result in list_of_results])
            list_of_txt_data.append(
                TxtData(
                    header="time_step",
                    array=time_steps,
                    format=self._set_column_data_format("time_step"),
                )
            )
        # Now, append the errors
        for key in errors_to_export.keys():
            list_of_txt_data.append(
                TxtData(
                    header=key,
                    array=errors_to_export[key],
                    format=self._set_column_data_format(key),
                )
            )

        # Finally, call the function to write into the txt
        export_data_to_txt(list_of_txt_data, file_name)

    def order_of_convergence(
        self,
        list_of_results: list,
        variables: Optional[list[str]] = None,
        x_axis: Literal["cell_diameter", "time_step"] = "cell_diameter",
        base_log_x_axis: int = 2,
        base_log_y_axis: int = 2,
        data_range: slice = slice(None, None, None),
    ) -> dict[str, float]:
        """Compute order of convergence (OOC) for a given set of variables.

        Note:
            The OOC is computed by fitting a line for log_{base_log_y_axis}(error)
            vs. log_{base_log_x_axis}(x_axis).

        Raises:
            ValueError
                If ``x_axis`` is ``"time_step"`` and the model is stationary.

        Parameters:
            list_of_results: List containing the results of the convergence analysis.
                Typically, the output of :meth:`run_analysis`.
            variables: ``default=None``

                Names of the variables for which the OOC should be computed. The
                ``item`` of the list must match the attribute ``"error_" + item`` from
                each item from the ``list_of_results``. If not given, all attributes
                starting with "error_" will be included in the analysis.
            x_axis: ``default=cell_diameter``

                Type of data in the x-axis used to compute the OOC.
            base_log_x_axis: ``default=2``

                Base of the logarithm for the data in the x-axis.
            base_log_y_axis: ``default=2``

                Base of the logarithm for the data in the y-axis.

            data_range: ``default=slice(None, None, None)``

                Data range used for the estimation of the observed order of convergence.
                It should be given as a slice. For example, if
                ``data_range=slice(-2, None, None)``, then only the last two data points
                from the list of results will be used. If not given, the whole range
                (i.e., errors collected at each level) will be used. Intended usage of
                this parameter is to employ only a subset of the data range (e.g.,
                the one corresponding to the asymptotic range) to determine the
                observed order of convergence.

        Returns:
            Dictionary containing the OOC for the given variables.

        """
        # Sanity check
        if x_axis == "time_step" and not self._is_time_dependent:
            msg = "Order of convergence cannot be estimated as a function of the time "
            msg += "step for a stationary model."
            raise ValueError(msg)

        # Get x-data
        if x_axis == "cell_diameter":
            x_data = np.array([result.cell_diameter for result in list_of_results])
        elif x_axis == "time_step":
            x_data = np.array([result.dt for result in list_of_results])
        else:
            msg = "'x_axis' must be either 'cell_diameter' or 'time_step'."
            raise NotImplementedError(msg)

        # Apply logarithm to x_data
        x_vals = np.emath.logn(base_log_x_axis, x_data)

        # Filter variables from the list of results
        names: list[str] = self._filter_variables_from_list_of_results(
            list_of_results=list_of_results,
            variables=variables,
        )

        # Obtain y-values
        y_vals: list[np.ndarray] = []
        for name in names:
            y_val: list[float] = []
            # Loop over lists of results
            for result in list_of_results:
                y_val.append(getattr(result, name))
            y_vals.append(np.emath.logn(base_log_y_axis, np.array(y_val)))

        # Filter the data to be used according to the given data range
        x_vals_filtered = x_vals[data_range]
        y_vals_filtered = [y_val[data_range] for y_val in y_vals]

        # Perform linear regression and populate the return dictionary
        # Keywords of the dictionary will have the prefix "ooc_" before the `name`
        ooc_dict: dict[str, float] = {}
        for idx, name in enumerate(names):
            slope, *_ = stats.linregress(x_vals_filtered, y_vals_filtered[idx])
            ooc_name = "ooc_" + name.lstrip("error_")  # strip the prefix "error_"
            ooc_dict[ooc_name] = slope

        return ooc_dict

    # -----> Helper methods
    def _get_list_of_meshing_arguments(self) -> list[dict[str, float]]:
        """Obtain list of meshing arguments dictionaries.

        Returns:
            List of meshing arguments dictionaries. Length of list is ``levels``.

        """
        # Retrieve initial meshing arguments
        init_mesh_args = deepcopy(self._init_setup.meshing_arguments())

        # Prepare factors for the spatial analysis
        factors = 1 / (self.spatial_refinement_rate ** np.arange(self.levels))

        # Loop through the levels and populate the list
        list_meshing_args: list[dict[str, float]] = []
        for lvl in range(self.levels):
            factor: pp.number = factors[lvl]
            meshing_args: dict[str, float] = {}
            for key in init_mesh_args:
                meshing_args[key] = init_mesh_args[key] * factor
            list_meshing_args.append(meshing_args)

        return list_meshing_args

    def _get_list_of_time_managers(self) -> Union[list[pp.TimeManager], None]:
        """Obtain list of time managers.

        Returns:
            List of time managers. Length of list is ``levels``. ``None`` is returned
            if the model is stationary.

        """
        if not self._is_time_dependent:
            return None

        # Retrieve initial time manager
        init_time_manager: pp.TimeManager = self._init_setup.time_manager

        # Sanity check
        if not init_time_manager.is_constant:
            msg = "Analysis in time only supports constant time step."
            raise NotImplementedError(msg)

        # Prepare factors for the temporal analysis
        factors = 1 / (self.temporal_refinement_rate ** np.arange(self.levels))

        # Loop over levels and populate the list
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

    def _filter_variables_from_list_of_results(
        self,
        list_of_results: list,
        variables: list[str] | None,
    ) -> list[str]:
        """Filter variables from the list of results.

        Parameters:
            list_of_results: List containing the results of the convergence analysis.
                Typically, the output of :meth:`run_analysis`.
            variables: List of strings containing the variables that should be
                filtered from the list of results. If ``None``, all variables with
                the prefix "error_" will be filtered.

        Returns:
            List of strings containing the filtered variables.

        """
        if variables is None:
            # Retrieve all attributes from the data class. Note that we use the first
            # result from the list of results to retrieve this information. Thus, we
            # assume that all other results contain (minimally) the same information.
            attributes: list[str] = list(vars(list_of_results[0]).keys())
            # Filter attributes that whose names contain the prefix ``'error_'``
            names = [attr for attr in attributes if attr.startswith("error_")]
        else:
            # Not much to do here, since the user gives the variables that should be
            # retrieved
            names = variables

        return names

    def _set_column_data_format(self, header: str) -> str:
        """Set column data format.

        Intended usage is to give the option of inheriting the class and override
        this method to define custom data formats for a given column.

        Parameters:
            header: Name of the header.

        Returns:
            String data format. As used to instantiate a
            :class:`~porepy.utils.txt_io.TxtData` object.

        """
        return "%2.2e"

    @staticmethod
    def grid_error(
            mdg: pp.MixedDimensionalGrid,
            mdg_ref: pp.MixedDimensionalGrid,
            variable: List[str],
            variable_dof: List[int],
    ) -> dict:
        """Compute grid errors a grid bucket and refined reference grid bucket

        Assumes that the coarse grid bucket has a property 'coarse_fine_cell_mapping'
        assigned on each subdomain, which maps from coarse to fine cells according to the
        method 'coarse_fine_cell_mapping(...)'.

        Parameters:
            mdg: "Coarse" mixed-dimensional grid.
            mdg_ref: "Fine" mixed-dimensional grid.
            variable: List defining which variables to compute error over.
            variable_dof: List specifying the number of degrees of freedom for each variable
                in the list 'variable'.

        Returns:
            errors: Dictionary with top level keys as node_number, within which for each
                variable, the error is reported.

        """
        warnings.warn(
            "This method will soon be removed from PorePy",
            DeprecationWarning,
        )

        assert len(variable) == len(variable_dof), (
            "Each variable must have associated " "with it a number of degrees of freedom."
        )
        n_variables = len(variable)

        errors: Dict = {}

        grids = mdg.subdomains()
        grids_ref = mdg_ref.subdomains()
        n_grids = len(grids)

        for i in np.arange(n_grids):
            g, g_ref = grids[i], grids_ref[i]
            mapping = mdg.subdomain_data(g)["coarse_fine_cell_mapping"]

            # Get time step solutions
            data = mdg.subdomain_data(g)
            data_ref = mdg_ref.subdomain_data(g_ref)
            solutions = data[pp.TIME_STEP_SOLUTIONS]
            solutions_ref = data_ref[pp.TIME_STEP_SOLUTIONS]
            node_number = data["node_number"]

            # Initialize errors
            errors[node_number] = {}

            for var_idx in range(0, n_variables):
                var = variable[var_idx]
                var_dof = variable_dof[var_idx]

                # Check if the variable exists on both the grid and reference grid
                solution_keys = set(solutions.keys())
                solution_ref_keys = set(solutions_ref.keys())
                check_keys = solution_keys.intersection(solution_ref_keys)
                if var not in check_keys:
                    logger.info(
                        f"{var} not present on grid number "
                        f"{node_number} of dim {g.dim}."
                    )
                    continue

                # Compute errors relative to the reference grid
                # TODO: Should the solution be divided by g.cell_volumes or similar?
                # TODO: If scaling is used, consider that - or use the export-ready variables,
                #   'u_exp', 'p_exp', etc.
                sol = (
                    solutions[var][0].reshape((var_dof, -1), order="F").T
                )  # (num_cells x var_dof)
                mapped_sol: np.ndarray = mapping.dot(sol)  # (num_cells x variable_dof)
                sol_ref = (
                    solutions_ref[var][0].reshape((var_dof, -1), order="F").T
                )  # (num_cells x var_dof)

                # axis=0 gives component-wise norm.
                absolute_error = np.linalg.norm(mapped_sol - sol_ref, axis=0)

                norm_ref = np.linalg.norm(sol_ref)
                if np.any(norm_ref < 1e-10):
                    logger.info(
                        f"Relative error not reportable. "
                        f"Norm of reference state is {norm_ref}. "
                        f"Reporting absolute error"
                    )
                    error = absolute_error
                    is_relative = False
                else:
                    error = absolute_error / norm_ref
                    is_relative = True

                errors[node_number][var] = {
                    "error": error,
                    "is_relative": is_relative,
                }

        return errors

    @staticmethod
    def project(g: pp.GridLike, fun: Callable):
        """
        Project a scalar or vector function on a piece-wise constant space.

        Parameters
        ----------
        g : grid
            Grid, or a subclass, with geometry fields computed.
        fun : function
            Scalar or vector function.

        Return
        ------
        out: np.ndarray (dim of fun, g.num_cells)
            Function interpolated in the cell centers.

        Examples
        --------

        def fun_p(pt): return np.sin(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1])

        def fun_u(pt): return [\
                          -2*np.pi*np.cos(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1]),
                          -2*np.pi*np.sin(2*np.pi*pt[0])*np.cos(2*np.pi*pt[1])]
        p_ex = interpolate(g, fun_p)
        u_ex = interpolate(g, fun_u)

        """

        return np.array([fun(pt) for pt in g.cell_centers.T]).T

    @staticmethod
    def norm_L2(g: pp.GridLike, val: np.ndarray):
        """
        Compute the L2 norm of a scalar or vector field.

        Parameters
        ----------
        g:
            Grid, or a subclass, with geometry fields computed.
        val:
            Scalar or vector field (dim of val = g.num_cells).

        Return
        ------
        out: double
            The L2 norm of the input field.

        Examples
        --------

        def fun_p(pt): return np.sin(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1])
        p_ex = interpolate(g, fun_p)
        norm_ex = norm_L2(g, p_ex)

        """

        val = np.asarray(val)
        norm_sq = lambda v: np.sum(np.multiply(np.square(v), g.cell_volumes))
        if val.ndim == 1:
            return np.sqrt(norm_sq(val))
        return np.sqrt(np.sum([norm_sq(v) for v in val]))

    @staticmethod
    def l2_error(
            grid: pp.GridLike,
            true_array: np.ndarray,
            approx_array: np.ndarray,
            is_scalar: bool,
            is_cc: bool,
            relative: bool = False,
    ) -> pp.number:
        """Compute discrete L2-error as given in [1].

        It is possible to compute the absolute error (default) or the relative error.

        Raises:
            NotImplementedError if a mortar grid is given and ``is_cc=False``.
            ZeroDivisionError if the denominator in the relative error is zero.

        Parameters:
            grid: Either a subdomain grid or a mortar grid.
            true_array: Array containing the true values of a given variable.
            approx_array: Array containing the approximate values of a given variable.
            is_scalar: Whether the variable is a scalar quantity. Use ``False`` for
                vector quantities. For example, ``is_scalar=True`` for pressure, whereas
                ``is_scalar=False`` for displacement.
            is_cc: Whether the variable is associated to cell centers. Use ``False``
                for variables associated to face centers. For example, ``is_cc=True``
                for pressures, whereas ``is_scalar=False`` for subdomain fluxes.
            relative: Compute the relative error (if True) or the absolute error (if False).

        Returns:
            Discrete L2-error between the true and approximated arrays.

        References:

            - [1] Nordbotten, J. M. (2016). Stable cell-centered finite volume
              discretization for Biot equations. SIAM Journal on Numerical Analysis,
              54(2), 942-968.

        """
        # Sanity check
        if isinstance(grid, pp.MortarGrid) and not is_cc:
            raise NotImplementedError("Interface variables can only be cell-centered.")

        # Obtain proper measure, e.g., cell volumes for cell-centered quantities and face
        # areas for face-centered quantities.
        if is_cc:
            meas = grid.cell_volumes
        else:
            assert isinstance(grid, pp.Grid)  # to please mypy
            meas = grid.face_areas

        if not is_scalar:
            meas = meas.repeat(grid.dim)

        # Obtain numerator and denominator to determine the error.
        numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))
        denominator = np.sqrt(np.sum(meas * np.abs(true_array) ** 2)) if relative else 1.0

        # Deal with the case when the denominator is zero when computing the relative error.
        if np.isclose(denominator, 0):
            raise ZeroDivisionError("Attempted division by zero.")

        return numerator / denominator