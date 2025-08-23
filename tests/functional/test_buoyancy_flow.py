"""
Tests for the N-phase, N-component buoyancy-driven flow model.

This test file verifies mass conservation and the reciprocity of buoyancy
fluxes in an immiscible flow simulation under the influence of gravity.

The tests cover two systems:
- N=2: Two phases (liquid-aqueous, gas) and two components (e.g., H₂O, CH₄).
- N=3: Three phases (liquid-aqueous, liquid-oleic, gas) and three components (e.g., H₂O, CO₂, CH₄).

The test runs simulations in both 2D and 3D and for various mass conservation
tolerances. It checks two primary conditions after each simulation run:
1. Reciprocal Buoyancy Fluxes: It asserts that the buoyancy fluxes of the
   components are equal and opposite.
2. Mass Conservation: It verifies that the change in the total volume of independent phases
   over the simulation time is within a specified tolerance, ensuring that the
   discretization of the buoyancy term is mass-conservative.
"""

from typing import Type
import pytest
import numpy as np
import porepy as pp
from tests.functional.setups.buoyancy_flow_model import ModelGeometry2D, ModelGeometry3D
from tests.functional.setups.buoyancy_flow_model import ModelMDGeometry2D
from tests.functional.setups.buoyancy_flow_model import BuoyancyFlowModel2N, BuoyancyFlowModel3N


@pytest.mark.parametrize(
    "model_class, mesh_2d_Q, expected_order_loss",
    [
        (BuoyancyFlowModel2N, True, 2),
        # (BuoyancyFlowModel2N, True, 4),
        # (BuoyancyFlowModel2N, True, 6),
        # (BuoyancyFlowModel2N, False, 2),
        # (BuoyancyFlowModel2N, False, 4),
        # (BuoyancyFlowModel2N, False, 6),
        (BuoyancyFlowModel3N, True, 2),
        # (BuoyancyFlowModel3N, True, 4),
        # (BuoyancyFlowModel3N, True, 6),
        # (BuoyancyFlowModel3N, False, 2),
        # (BuoyancyFlowModel3N, False, 4),
        # (BuoyancyFlowModel3N, False, 6),
    ],
)
def test_buoyancy_model(
    model_class: Type[pp.PorePyModel],
    mesh_2d_Q: bool,
    expected_order_loss: int,
) -> None:
    """
    Runs the buoyancy-driven flow simulation and checks for mass conservation and
    reciprocal buoyancy fluxes.

    Parameters:
        model_class (Type[pp.PorePyModel]): The buoyancy flow model class to test
                                                     (BuoyancyFlowModel2N or BuoyancyFlowModel3N).
        mesh_2d_Q (bool): If True, runs a 2D simulation. Otherwise, runs a 3D simulation.
        expected_order_mass_loss (int): The expected order of magnitude for the mass loss,
                                        used to set the residual tolerance.
    """
    residual_tolerance = 15.0 ** (-expected_order_loss)

    day = 86400
    tf = 2.0 * day
    dt = 1.0 * day
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
        thermal_conductivity=2.0 * 1e-6,  # to_Mega already applied
        density=2500.0,
        specific_heat_capacity=1000.0 * 1e-6,  # to_Mega already applied
    )
    material_constants = {"solid": solid_constants}
    params = {
        "fractional_flow": True,
        "buoyancy_on": True,
        "material_constants": material_constants,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "apply_schur_complement_reduction": False,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": residual_tolerance,
        "max_iterations": 50,
        "expected_order_loss": expected_order_loss,
    }

    # Combine the geometry with the main model class
    if mesh_2d_Q:
        # Define the 2D model by inheriting from ModelGeometry2D and the parametrized model
        class Model2D(ModelGeometry2D, model_class):
            pass

        model = Model2D(params)
    else:
        # Define the 3D model by inheriting from ModelGeometry3D and the parametrized model
        class Model3D(ModelGeometry3D, model_class):
            pass

        model = Model3D(params)

    model.prepare_simulation()
    pp.run_time_dependent_model(model, params)

@pytest.mark.parametrize(
    "model_class, mesh_2d_Q, expected_order_loss",
    [
        (BuoyancyFlowModel2N, True, 2),
        # (BuoyancyFlowModel2N, True, 3),
        # (BuoyancyFlowModel2N, True, 4),
        # (BuoyancyFlowModel2N, False, 2),
        # (BuoyancyFlowModel2N, False, 3),
        # (BuoyancyFlowModel2N, False, 4),
        (BuoyancyFlowModel3N, True, 2),
        # (BuoyancyFlowModel3N, True, 4),
        # (BuoyancyFlowModel3N, True, 6),
        # (BuoyancyFlowModel3N, False, 2),
        # (BuoyancyFlowModel3N, False, 4),
        # (BuoyancyFlowModel3N, False, 6),
    ],
)

def test_buoyancy_md_model(
    model_class: Type[pp.PorePyModel],
    mesh_2d_Q: bool,
    expected_order_loss: int,
) -> None:
    """
    Runs the md buoyancy-driven flow simulation and checks for mass conservation and
    reciprocal buoyancy fluxes.

    Parameters:
        model_class (Type[pp.PorePyModel]): The buoyancy flow model class to test
                                                     (BuoyancyFlowModel2N or BuoyancyFlowModel3N).
        mesh_2d_Q (bool): If True, runs a 2D simulation. Otherwise, runs a 3D simulation.
        expected_order_mass_loss (int): The expected order of magnitude for the mass loss,
                                        used to set the residual tolerance.
    """
    residual_tolerance = 15.0 ** (-expected_order_loss)

    day = 86400
    tf = 0.5 * day
    dt = 0.25 * day
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
        thermal_conductivity=2.0 * 1e-6,  # to_Mega already applied
        density=2500.0,
        specific_heat_capacity=1000.0 * 1e-6,  # to_Mega already applied
    )
    material_constants = {"solid": solid_constants}
    params = {
        "fractional_flow": True,
        "buoyancy_on": True,
        "material_constants": material_constants,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "apply_schur_complement_reduction": False,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": residual_tolerance,
        "max_iterations": 50,
        "expected_order_loss": expected_order_loss,
    }

    # Combine the geometry with the main model class
    if mesh_2d_Q:
        # Define the 2D model by inheriting from ModelGeometry2D and the parametrized model
        class Model2D(ModelMDGeometry2D, model_class):
            pass

        model = Model2D(params)
    else:
        # Define the 3D model by inheriting from ModelGeometry3D and the parametrized model
        class Model3D(ModelGeometry3D, model_class):
            pass

        model = Model3D(params)

    model.prepare_simulation()
    pp.run_time_dependent_model(model, params)