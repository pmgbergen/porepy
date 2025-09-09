import copy
from typing import Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import add_mixin
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.examples import fracture_damage as damage_examples
from porepy.models import fracture_damage as damage_models
from porepy.numerics.nonlinear.line_search import ConstraintLineSearchNonlinearSolver

geometry_mixins = {
    "2": SquareDomainOrthogonalFractures,
    "3": CubeDomainOrthogonalFractures,
}
damage_types = {
    "dilation": damage_examples.DilationDamageMomentumBalance,
    "friction": damage_examples.FrictionDamageMomentumBalance,
}

# Methods to compare will be provided later by the user. Example names could be
# ["friction_damage", "dilation_damage"]. Leave empty to skip comparisons.
METHODS_TO_COMPARE: list[str] = []


def build_single_step_model(
    isotropic: bool,
    dim: int,
    damages: Sequence[str],
    time_steps: int,
    params: dict[str, Any] | None = None,
):
    """Construct a model for a time-dependent run using shared setup logic.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
    dim: Spatial dimension of the bulk domain (2 or 3).
    damages: Iterable of damage types to include (subset of {"dilation", "friction"}).
    time_steps: Number of time steps to simulate.
        Number of time instants passed to the TimeManager (N); this yields N-1
        forward steps. For a single forward step, pass 2.

    Returns:
        A configured model instance ready to be run with pp.run_time_dependent_model.
    """
    params_local = copy.deepcopy(damage_examples.model_params)
    model_class = damage_examples.FractureDamageMomentumBalance

    # Choose damage length (isotropic vs anisotropic) and exact solution
    if isotropic:
        params_local["exact_solution"] = damage_examples.ExactSolutionIsotropic
        model_class = add_mixin(
            damage_models.IsotropicFractureDamageLength, model_class
        )
    else:
        params_local["exact_solution"] = damage_examples.ExactSolutionAnisotropic
        model_class = add_mixin(
            damage_models.AnisotropicFractureDamageLength, model_class
        )

    # Add requested damage equations and variables.
    for name in damages:
        model_class = add_mixin(damage_types[name], model_class)

    # Add geometry mixin for the target dimension.
    model_class = add_mixin(geometry_mixins[str(dim)], model_class)

    # Configure time manager: np.arange(0, time_steps) gives time_steps-1 steps
    params_local.update(
        {
            "time_manager": pp.TimeManager(np.arange(0, time_steps), 1, True),
            # Trim displacement BCs to requested dimension
            "north_displacements": params_local["north_displacements"][:dim],
        }
    )

    # Use equal initial damage in both modes for simplicity
    solid_params_local = {
        **damage_examples.solid_params,
        **{"initial_friction_damage": 0.5, "initial_dilation_damage": 0.5},
    }
    params_local["material_constants"] = {
        "solid": FractureDamageSolidConstants(**solid_params_local)  # type: ignore[arg-type]
    }
    # Finally, add model parameters passed to this function.
    params_local.update(params or {})
    # Initialize the model with the updated parameters and run problem.
    m = model_class(params_local)
    pp.run_time_dependent_model(
        m,
        {
            "max_iterations": 50,  # Hard nonlinear problems - expect slow convergence
            "nl_convergence_tol_res": 1e-8,
            "nl_convergence_tol": 1e-8,
            "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
            "adaptive_indicator_scaling": True,
            "local_line_search": True,
        },
    )
    # Initialize test names for assertions
    test_names = [
        "friction_damage",
        "dilation_damage",
    ]
    # Initial time step is not included in VerificationDataSaving
    for t in np.arange(time_steps - 1):
        # Retrieve stored results for time step t + 1
        results = m.results[t]
        for name in test_names:
            # Retrieve the normalized error for the current time step and the current
            # variable.
            e = getattr(results, name + "_error")
            # Uncomment to print the error. Useful for debugging. Might require running
            # the test with pytest -s.
            val = getattr(results, "exact_" + name)
            print(f"Time step {t + 1}, {name}: {e}")
            print(f"Time step {t + 1}, {name} ana: {val}")
            print(
                f"Time step {t + 1}, {name} num: {getattr(results, 'approx_' + name)}"
            )

            # Assert that the error is small.
            np.testing.assert_allclose(e, 0, atol=1e-8)
