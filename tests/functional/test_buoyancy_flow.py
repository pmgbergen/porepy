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

import numpy as np
import pytest

import porepy as pp
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
    (BuoyancyFlowModel2N, True, 4),
    (BuoyancyFlowModel2N, False, 4),
    (BuoyancyFlowModel3N, True, 4),
    (BuoyancyFlowModel3N, False, 4),
]


def _run_buoyancy_model(
    model_class: type,
    mesh_2d_Q: bool,
    expected_order_loss: int,
    md: bool = False,
) -> None:
    """Run buoyancy flow simulation for given parameters."""
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
    params = {
        "fractional_flow": True,
        "enable_buoyancy_effects": True,
        "material_constants": {"solid": solid_constants},
        "time_manager": time_manager,
        "apply_schur_complement_reduction": False,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": residual_tolerance,
        "max_iterations": 50,
        "expected_order_loss": expected_order_loss,
    }
    # Combine geometry with model class
    if mesh_2d_Q:

        class Model2D(geometry2d, model_class):
            pass

        model = Model2D(params)
    else:

        class Model3D(geometry3d, model_class):
            pass

        model = Model3D(params)
    pp.run_time_dependent_model(model, params)


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize(
    "model_class, mesh_2d_Q, expected_order_loss", Parameterization
)
@pytest.mark.parametrize("md", [True])  # False skipped to limit computational cost.
def test_buoyancy_model(model_class, mesh_2d_Q, expected_order_loss, md):
    """Test buoyancy-driven flow model (FD)."""
    _run_buoyancy_model(model_class, mesh_2d_Q, expected_order_loss, md=md)
