"""Module containing a class for performing spatio-temporal convergence analysis."""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from scipy import stats

import porepy as pp


class ConvergenceAnalysis:
    """Performing a convergence analysis on a model.

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

            Number of refinement levels associated to the convergence analysis.
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
        # Sanity checks
        if not in_space and not in_time:
            raise ValueError("At least one type of analysis should be performed.")

        if not in_space and spatial_rate > 1:
            warnings.warn("'spatial_rate' is not being used.")
            spatial_rate = 1

        if not in_time and temporal_rate > 1:
            warnings.warn("'temporal_rate' is not being used.")
            temporal_rate = 1

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
        list_of_meshing_arguments: list[
            Union[dict[str, pp.number], np.ndarray]
        ] = self._get_list_of_meshing_arguments()

        # Retrieve list of time managers
        list_of_time_managers: Union[
            list[pp.TimeManager], None
        ] = self._get_list_of_time_managers()

        # Create list of model parameters
        list_of_params: list[dict] = []
        for lvl in range(self.levels):
            params = deepcopy(model_params)
            params["meshing_arguments"] = list_of_meshing_arguments[lvl]
            if list_of_time_managers is not None:
                params["time_manager"] = list_of_time_managers[lvl]
            list_of_params.append(params)

        self.model_params: list[dict] = list_of_params
        """List of model parameters associated to each run."""

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

    # -----> Getter methods
    def _get_list_of_meshing_arguments(self) -> list[dict[str, float]]:
        """Obtain list of meshing arguments.

        Returns:
            List of mesh arguments. Length of list is ``levels``.

        """
        # Retrieve initial mesh arguments
        init_mesh_args = deepcopy(self._init_setup.meshing_arguments())

        # Prepare factors for the spatial analysis
        factors = 1 / (self.spatial_rate ** np.arange(self.levels))

        # Loop through the levels and obtain mesh size accordingly
        list_meshing_args: list[dict[str, float]] = []
        for lvl in range(self.levels):
            factor: pp.number = factors[lvl]
            meshing_args = {}
            for key in init_mesh_args:
                meshing_args[key] = init_mesh_args[key] * factor
            list_meshing_args.append(meshing_args)

        return list_meshing_args

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

    # -----> Helper methods
    def log_space(self, array: np.ndarray) -> np.ndarray:
        return np.emath.logn(self.spatial_rate, array)

    def log_time(self, array: np.ndarray) -> np.ndarray:
        return np.emath.logn(self.temporal_rate, array)