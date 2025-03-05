import copy
from typing import Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import _add_mixin
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.examples import fracture_damage as damage

geometry_mixins = [
    SquareDomainOrthogonalFractures,
    CubeDomainOrthogonalFractures,
]
# Additional solid parameters used for parameterizing the tests for different regimes
# of frictional and dilational damage.
additional_solid_params = {
    "friction": {
        "initial_friction_damage": 1.0,  # Initial value 1 implies no frictional damage
        "initial_dilation_damage": 2.0,
    },
    "dilation": {
        "initial_friction_damage": 1.5,
        "initial_dilation_damage": 1.0,  # Initial value 1 implies no dilational damage
    },
    "both": {
        "initial_friction_damage": 1.1,
        "initial_dilation_damage": 1.1,
    },
}


@pytest.mark.parametrize("isotropic", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "regime",
    ["dilation", "friction", "both"],
)
@pytest.mark.parametrize(
    "time_steps", [2, pytest.param(6, marks=[pytest.mark.skip(reason="slow")])]
)
def test_damage(
    isotropic: bool,
    dim: int,
    regime: Literal["dilation", "friction", "both"],
    time_steps: int,
):
    # Copy the model parameters to avoid modifying the original dictionary.
    # Deepcopy is needed to avoid modifying mutable objects in the dictionary,
    # presumably the time manager.
    params_local = copy.deepcopy(damage.model_params)
    if isotropic:
        # Use the isotropic damage model.
        model_class = damage.IsotropicFractureDamage
        # Change the exact solution to the isotropic version.
        params_local["exact_solution"] = damage.ExactSolutionIsotropic
    else:
        # Use the anisotropic damage model.
        model_class = damage.AnisotropicFractureDamage
    # Add geometry mixin.
    model_class = _add_mixin(geometry_mixins[dim - 2], model_class)

    # Combine the solid parameters giving priority to the additional parameters.
    # solid_params is not modified and can be reused in other tests.
    solid_params_local = {**damage.solid_params, **additional_solid_params[regime]}

    params_local.update(
        {
            "material_constants": {
                "solid": FractureDamageSolidConstants(**solid_params_local)
            },
            "time_manager": pp.TimeManager(np.arange(0, time_steps), 1, True),
            "interface_displacement_parameter_values": params_local[
                "interface_displacement_parameter_values"
            ][:dim],
        }
    )

    m = model_class(params_local)
    # Some simulations do not converge in the default number of iterations. Only
    # slightly increase the number of iterations, thus capturing any future deterioration
    # in the convergence and avoiding excessive run times.

    pp.run_time_dependent_model(
        m,
        {
            "max_iterations": 45,
        },
    )
    # Initialize test names for assertions
    test_names = [
        "friction_damage",
        "dilation_damage",
        "damage_history",
    ]
    # Initial time step is not included in VerificationDataSaving
    for t in np.arange(time_steps - 1):
        # Retrieve stored results for time step t + 1
        results = m.results[t]
        for name in test_names:
            # Retrieve the normalized error for the current time step and the current variable.
            e = getattr(results, name + "_error")
            # Uncomment to print the error. Useful for debugging. Might require running
            # the test with pytest -s.
            # print(f"Time step {t + 1}, {name}: {e}")
            # Assert that the error is small
            assert e < 1e-6
