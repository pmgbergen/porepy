"""
Tests for the N-phase, N-component buoyancy-driven flow model.

This file verifies mass and energy conservation and the reciprocity of buoyancy
fluxes in an immiscible flow simulation under gravity.

It covers two multicomponent fluid systems:
- N = 2: Two phases (aqueous liquid, gas) and two components (e.g., H₂O, CH₄).
- N = 3: Three phases (aqueous liquid, oleic liquid, gas) and
three components (e.g., H₂O, CO₂, CH₄).

Simulations are run in 2D and 3D for several conservation tolerances, and
the observed conservation is checked to be of the expected order. After each
time step the following are tested:
1. Reciprocal buoyancy fluxes: Component buoyancy fluxes are equal and opposite.
2. Mass conservation: The change in the total volume of independent phases over
   the simulation time remains within the specified tolerance, demonstrating a
   mass-conservative discretization of the buoyancy term.
3. Energy conservation: The change in total fluid energy over the simulation
   time remains within the specified tolerance, demonstrating an energy-conservative
   discretization of the energy convective buoyancy terms.
"""

from typing import Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.models import add_mixin
from tests.functional.setups.buoyancy_flow_model import (
    BuoyancyFlowModel2N,
    BuoyancyFlowModel3N,
    ModelGeometry2D,
    ModelGeometry3D,
    ModelMDGeometry2D,
    ModelMDGeometry3D,
    to_Mega,
)

# Parameterization list for both tests
Parameterization = [
    (BuoyancyFlowModel2N, 2, 4),
    (BuoyancyFlowModel2N, 3, 4),
    (BuoyancyFlowModel3N, 2, 4),
    (BuoyancyFlowModel3N, 3, 4),
]


def _run_buoyancy_model(
    model_class: type,
    dim: Literal[2, 3],
    expected_order_loss: int,
    md: bool = False,
) -> None:
    """Run buoyancy flow simulation for given parameters."""

    # The residual tolerance for Newton should be related to the expected (requested)
    # order loss.
    residual_tolerance = 10.0 ** (-expected_order_loss)
    day = 86400
    if md:
        tf = 0.5 * day
        dt = 0.25 * day
        geometry2d = ModelMDGeometry2D
        geometry3d = ModelMDGeometry3D
    else:
        tf = 2.0 * day
        dt = 1.0 * day
        geometry2d = ModelGeometry2D
        geometry3d = ModelGeometry3D

    solid_constants = pp.SolidConstants(
        permeability=1.0e-14,
        porosity=0.1,
        thermal_conductivity=2.0 * to_Mega,
        density=2500.0,
        specific_heat_capacity=1000.0 * to_Mega,
    )
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=50,
        print_info=True,
    )
    model_params = {
        "fractional_flow": True,
        "enable_buoyancy_effects": True,
        "material_constants": {"solid": solid_constants},
        "time_manager": time_manager,
        "apply_schur_complement_reduction": False,
        "expected_order_loss": expected_order_loss,
    }
    # Combine geometry with model class.
    geometry_class = geometry2d if dim == 2 else geometry3d
    model_class = add_mixin(geometry_class, model_class)
    model = model_class(model_params)
    # Use a Lebesgue metric for the residual convergence criterion, since this will
    # strictly bound the residual error in the mass conservation equations.
    solver_params = {
        "nl_convergence_criteria": {
            "res_abs": pp.ResidualBasedAbsoluteCriterion(
                tol=residual_tolerance, metric=pp.EquationBasedLebesgueMetric(model)
            ),
        },
        "nl_divergence_criteria": {
            "max_iter": pp.MaxIterationsCriterion(max_iterations=50),
        },
    }

    pp.run_time_dependent_model(model, solver_params)


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize("model_class, dim, expected_order_loss", Parameterization)
@pytest.mark.parametrize("md", [True])  # False skipped to limit computational cost.
def test_buoyancy_model(model_class, dim: Literal[2, 3], expected_order_loss, md):
    """Test buoyancy-driven flow model (FD)."""
    _run_buoyancy_model(model_class, dim, expected_order_loss, md=md)
