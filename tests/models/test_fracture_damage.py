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
from porepy.applications.test_utils.reference_dense_arrays import test_damage_history
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


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize("isotropic", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "regime",
    ["dilation", "friction", "both"],
)
def test_damage(
    isotropic: bool,
    dim: int,
    regime: Literal["dilation", "friction", "both"],
):

    # Copy the model parameters to avoid modifying the original dictionary.
    # Deepcopy is needed to avoid modifying mutable objects in the dictionary,
    # presumably the time manager.
    params_local = copy.deepcopy(damage.model_params)
    if isotropic:
        # Use the isotropic damage model.
        model_class = damage.IsotropicModel
        # Change the exact solution to the isotropic version.
        params_local["exact_solution"] = damage.ExactSolutionIsotropic
    else:
        # Use the anisotropic damage model.
        model_class = damage.AnisotropicModel
    # Add geometry mixin.
    model_class = _add_mixin(geometry_mixins[dim - 2], model_class)

    # Combine the solid parameters giving priority to the additional parameters.
    # solid_params is not modified and can be reused in other tests.
    solid_params_local = {**damage.solid_params, **additional_solid_params[regime]}
    params_local["material_constants"] = {
        "solid": FractureDamageSolidConstants(**solid_params_local)
    }
    params_local["north_displacements"] = params_local["north_displacements"][:dim]
    # North displacements are time dependent and implicitly define the number of time steps.
    num_time_steps = params_local["north_displacements"].shape[1]
    m = model_class(params_local)
    # Some simulations do not converge in the default number of iterations. Only
    # slightly increase the number of iterations, thus capturing any future detoriation
    # in the convergence and avoiding excessive run times.
    pp.run_time_dependent_model(m, {"max_iterations": 15})
    # return m
    test_names = [
        "friction_damage",
        "dilation_damage",
        "damage_history",
    ]
    # Initial time step is not included in VerificationDataSaving
    for t in np.arange(num_time_steps - 1):
        # Retrieve stored results for time step t + 1
        results = m.results[t]
        for name in test_names:
            # Retrieve the normalized error for the current time step and the current variable.
            e = getattr(results, name + "_error")
            print(f"Time step {t + 1}, {name}: {e}")
            # Assert that the error is small
            assert e < 1e-1


class AnisotropicDamage3D(
    CubeDomainOrthogonalFractures,
    damage.AnisotropicModel,
):
    pass


def test_anisotropic_damage_values():
    params_local = copy.deepcopy(damage.model_params)
    solid_params_local = {**damage.solid_params, **additional_solid_params["both"]}
    # Set higher friction coefficient to improve convergence.
    solid_params_local["friction_coefficient"] = 0.5
    params_local["material_constants"] = {
        "solid": FractureDamageSolidConstants(**solid_params_local)
    }
    m = AnisotropicDamage3D(params_local)
    pp.run_time_dependent_model(m, {"max_iterations": 15})

    # North displacements are time dependent and implicitly define the number of time steps.
    num_time_steps = params_local["north_displacements"].shape[1]

    values_to_test = test_damage_history["test_anisotropic_damage_values"]
    for name, val_known in values_to_test.items():
        for t in np.arange(num_time_steps - 1):
            results = m.results[t]
            val_num = getattr(results, "approx_" + name)
            assert np.allclose(val_num, val_known[t])
            # Uncomment to print the values. Useful for debugging. Might require running
            # the test with pytest -s.
            # print(f"{name}: {val_num}")
