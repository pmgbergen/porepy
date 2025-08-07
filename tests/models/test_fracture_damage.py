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
from porepy.examples import fracture_damage as damage
from porepy.models.fracture_damage import AnisotropicFractureDamageEquations

geometry_mixins = {
    "2": SquareDomainOrthogonalFractures,
    "3": CubeDomainOrthogonalFractures,
}
# Additional solid parameters used for parameterizing the tests for different regimes
# of frictional and dilational damage.
additional_solid_params = {
    "dilation": {
        "initial_friction_damage": 1.0,  # Initial value 1 implies no frictional damage
        "initial_dilation_damage": 0.7,
    },
    "friction": {
        "initial_friction_damage": 0.5,
        "initial_dilation_damage": 1.0,  # Initial value 1 implies no dilational damage
    },
    "both": {
        "initial_friction_damage": 0.5,
        "initial_dilation_damage": 0.5,
    },
}


@pytest.mark.parametrize("isotropic", [True, False])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize(
    "regime",
    ["dilation"],  # "friction", "both"],
)
@pytest.mark.parametrize(
    "time_steps", [7, pytest.param(7, marks=pytest.mark.skipped(reason="slow"))]
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
    model_class = damage.FractureDamageMomentumBalance

    if isotropic:
        # Use the isotropic damage model.
        # Change the exact solution to the isotropic version.
        params_local["exact_solution"] = damage.ExactSolutionIsotropic
    else:
        # Use the anisotropic damage model.
        model_class = add_mixin(AnisotropicFractureDamageEquations, model_class)
    # Add geometry mixin.
    model_class = add_mixin(geometry_mixins[str(dim)], model_class)
    # Add the simplified White coefficients for the damage model.
    # model_class = add_mixin(damage.FractureDamageCoefficientsWhite, model_class)
    # Combine the solid parameters giving priority to the additional parameters.
    # solid_params is not modified and can be reused in other tests.
    solid_params_local = {**damage.solid_params, **additional_solid_params[regime]}

    params_local.update(
        {
            "material_constants": {
                "solid": FractureDamageSolidConstants(**solid_params_local)
            },
            "time_manager": pp.TimeManager(np.arange(0, time_steps), 1, True),
            "north_displacements": params_local["north_displacements"][:dim],
        }
    )

    m = model_class(params_local)
    pp.run_time_dependent_model(
        m,
        {
            "max_iterations": 20,
            "nl_convergence_tol_res": 1e-8,
            "nl_convergence_tol": 1e-8,
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
