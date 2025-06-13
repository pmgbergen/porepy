"""
Tests for the 2-phase, 2-component buoyancy-driven flow model.

This test file verifies the mass conservation and reciprocal nature of buoyancy fluxes
in a compositional flow simulation. The simulation involves two phases (liquid and gas)
and two components (H2O and CO2) under the influence of gravity.

The test runs simulations in both 2D and 3D and for various mass conservation
tolerances. It checks two primary conditions after each simulation run:
1. Reciprocal Buoyancy Fluxes: It asserts that the buoyancy fluxes of the two
   components are equal and opposite, which is a fundamental physical property.
2. Mass Conservation: It verifies that the change in the total volume of the gas
   phase over the simulation time is within a specified tolerance, ensuring that the
   discretization of the buoyancy term is mass-conservative.
"""

import pytest
import numpy as np
import porepy as pp
from tests.functional.setups.buoyancy_flow_model import ModelGeometry2D, ModelGeometry3D, BuoyancyFlowModel

@pytest.mark.parametrize(
    "mesh_2d_Q, expected_order_mass_loss",
    [
        (True, 2),
        (True, 4),
        (True, 6),
        (False, 2),
        (False, 4),
        (False, 6),
    ],
)
def test_buoyancy_model(mesh_2d_Q: bool, expected_order_mass_loss: int) -> None:
    """
    Runs the buoyancy-driven flow simulation and checks for mass conservation and
    reciprocal buoyancy fluxes.

    Parameters:
        mesh_2d_Q (bool): If True, runs a 2D simulation. Otherwise, runs a 3D simulation.
        expected_order_mass_loss (int): The expected order of magnitude for the mass loss,
                                        used to set the residual tolerance.
    """
    residual_tolerance = 10.0 ** (-expected_order_mass_loss)

    day = 86400
    tf = 500.0 * day
    dt = 50.0 * day
    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=50,
        print_info=True,
    )

    solid_constants = pp.SolidConstants(
        permeability=1.0e-14,
        porosity=0.1,
        thermal_conductivity=2.0e-6,  # to_Mega already applied
        density=2500.0,
        specific_heat_capacity=1000.0e-6,  # to_Mega already applied
    )
    material_constants = {"solid": solid_constants}
    params = {
        "rediscretize_darcy_flux": True,
        "rediscretize_fourier_flux": True,
        "fractional_flow": True,
        "material_constants": material_constants,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "reduce_linear_system": False,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": residual_tolerance,
        "max_iterations": 100,
        "expected_order_mass_loss": expected_order_mass_loss,
    }

    # Combine the geometry with the main model class
    if mesh_2d_Q:
        # Define the 2D model by inheriting from ModelGeometry2D
        class Model2D(ModelGeometry2D, BuoyancyFlowModel):
            pass

        model = Model2D(params)
    else:
        # Define the 3D model by inheriting from ModelGeometry3D
        class Model3D(ModelGeometry3D, BuoyancyFlowModel):
            pass

        model = Model3D(params)

    model.prepare_simulation()
    pp.run_time_dependent_model(model, params)