"""This file contains a complete set of top-level parameters for a PorePy model, along
with a solver to run it. Use it as a starting point and copy the relevant parameters for
your own problem.

"""

import numpy as np

import porepy as pp
from porepy.applications.material_values.fluid_values import (
    extended_water_values_for_testing as water,
)
from porepy.applications.material_values.numerical_values import (
    extended_numerical_values_for_testing as numerical_values,
)
from porepy.applications.material_values.reference_values import (
    extended_reference_values_for_testing as reference_values,
)
from porepy.applications.material_values.solid_values import (
    extended_granite_values_for_testing as granite,
)
from porepy.numerics.nonlinear.line_search import ConstraintLineSearchNonlinearSolver

# Used for conversion of units.
units = pp.Units()

# NOTE: When expanding this list, please mark those parameters that are not used by
# the default models by 'not invoked by default'.

model_params = {
    "linear_solver": "pypardiso",
    "units": units,
    "time_manager": pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True),
    "reference_variable_values": pp.ReferenceVariableValues(**reference_values),  # type: ignore[arg-type]
    "material_constants": {
        "solid": pp.SolidConstants(**granite),  # type: ignore[arg-type]
        "fluid": pp.FluidComponent(**water),  # type: ignore[arg-type]
        "numerical": pp.NumericalConstants(**numerical_values),  # type: ignore[arg-type]
    },
    # Meshing
    "grid_type": "cartesian",
    # Depending on the grid type, some subset of the meshing arguments below are used.
    "meshing_arguments": {
        # Use this one for the basic grid. Used as default if no more specific cell size
        # is given.
        "cell_size": units.convert_units(0.5, "m"),
        # Or the more refined parameters.
        "cell_size_x": units.convert_units(0.5, "m"),  # For TensorGrid
        "cell_size_y": units.convert_units(0.5, "m"),  # For TensorGrid
        "cell_size_z": units.convert_units(0.5, "m"),  # For TensorGrid
        "cell_size_fracture": units.convert_units(0.25, "m"),  # For unstructured grids
        "cell_size_boundary": units.convert_units(0.5, "m"),  # For unstructured grids
        "cell_size_min": units.convert_units(0.51, "m"),  # For unstructured grids
        "circumcenter_threshold": 0,  # Relative distance to move cell centers from
        # centroids to circumcenters in [0, 1). 0 implies no movement.
    },
    "meshing_kwargs": {},
    # Used to export the fracture network to enable later reuse. Not invoked by default.
    "csv_file_name": "fracture_network.csv",
    # Used to export the gmsh geometry and the created mesh file.
    "gmsh_file_name": "gmsh_frac_file.msh",
    # Exporting and restarting
    "nonlinear_solver_statistics": pp.SolverStatistics,  # Must be a class, not instance
    "folder_name": "visualization",
    "file_name": "data",
    "solver_statistics_file_name": "statistics",
    "export_constants_separately": False,
    "times_to_export": None,
    "restart_options": {
        "restart": False,
        "pvd_file": None,
        "is_mdg_pvd": False,
        "vtu_files": None,
        "times_file": None,
        "time_index": -1,
    },
    # Compositional model (multiphase multicomponent flow)
    "fractional_flow": True,
    "enable_buoyancy_effects": False,
    "apply_schur_complement_reduction": False,
    "schur_complement_inverter": None,
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "phase_property_params": None,  # See Phase.compute_properties for details.
    # Contact mechanics
    "traction_estimate_p_mean": 5.0,
    "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively for robustness.
}

solver_params = {
    "prepare_simulation": True,
    "progressbars": True,  # make sure you installed tqdm
    "_nl_progress_bar_position": 0,  # TODO: You don't want to change it.
    # Sufficient to steer control default convergence and divergence criteria.
    "nl_max_iterations": 10,  # Max iterations of a nonlinear solver (Newton)
    "nl_convergence_inc_atol": 1e-6,  # Increment norm
    "nl_convergence_res_atol": 1e-6,  # Residual norm
    "nl_convergence_inc_rtol": 1e-4,  # Increment norm (relative)
    "nl_convergence_res_rtol": 1e-4,  # Residual norm (relative)
    "nl_divergence_inc_atol": np.inf,
    "nl_divergence_res_atol": np.inf,
    "nl_metric": pp.EuclideanMetric(),  # Metric for norms.
    # Detailed convergence and divergence criteria - overwrite the defaults.
    "nl_convergence_criteria": {
        "inc_abs": pp.IncrementBasedAbsoluteCriterion(
            tol=1e-6, metric=pp.EuclideanMetric()
        ),
        "inc_rel": pp.IncrementBasedRelativeCriterion(
            tol=1e-4, metric=pp.EuclideanMetric()
        ),
        "res_abs": pp.ResidualBasedAbsoluteCriterion(
            tol=1e-6, metric=pp.EuclideanMetric()
        ),
        "res_rel": pp.ResidualBasedRelativeCriterion(
            tol=1e-4, metric=pp.EuclideanMetric()
        ),
    },
    "nl_divergence_criteria": {
        "max_iter": pp.MaxIterationsCriterion(max_iterations=10),
        "inc_nan": pp.IncrementBasedNanCriterion(),
        "res_nan": pp.ResidualBasedNanCriterion(),
        "inc_max": pp.IncrementBasedAbsoluteDivergenceCriterion(
            tol=1e20, metric=pp.EuclideanMetric()
        ),
        "res_max": pp.ResidualBasedAbsoluteDivergenceCriterion(
            tol=1e20, metric=pp.EuclideanMetric()
        ),
    },
    # Line search / Solution Strategies. These are considered "advanced" options.
    # Delete the following lines for the default Newton's method.
    "nonlinear_solver": ConstraintLineSearchNonlinearSolver,  # Must be a class.
    "global_line_search": 0,  # Set to 1 to use turn on a residual-based line search.
    "residual_line_search_interval_size": 1e-1,
    "residual_line_search_num_steps": 5,
    "local_line_search": 1,  # Set to 0 to use turn off the tailored line search.
    "relative_constraint_transition_tolerance": 2e-1,
    "constraint_violation_tolerance": 3e-1,
    "min_line_search_weight": 1e-10,
}
