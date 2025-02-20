import copy
import math

import pytest

import porepy as pp
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from tests.functional.setups.manu_sneddon_2d import ManuSneddonSetup2d


@pytest.fixture(scope="module")
def actual_ooc() -> dict:
    """Performs convergence analysis of the Sneddon setup.

    This setup validates the linear elasticity model for the analytical Sneddon solution
    in 2D, describing the analytical displacement on the fracture. The problem consists
    of a 2D domain with a fracture at a given angle and internal pressure.

    Returns:
        A dictionary containing the experimental order of convergence for the
        displacement.

    """
    # Angle of the fracture in degrees

    # Simulation parameters
    theta_deg = 30.0
    a = 0.3
    height = 1.0
    theta_rad = math.radians(90 - theta_deg)

    # Set up the material constants
    poi = 0.25
    shear_modulus = 1
    lam = (
        2 * shear_modulus * poi / (1 - 2 * poi)
    )  # Convertion formula from shear modulus and poission to lame lambda parameter

    solid = pp.SolidConstants(shear_modulus=shear_modulus, lame_lambda=lam)

    params = {
        "prepare_simulation": True,
        "material_constants": {"solid": solid},
        "a": a,  # Half-length of the fracture
        "domain_size": height,  # Length of square domain
        "p0": 1e-4,  # Internal pressure of fracture
        "poi": poi,  # Possion ratio (Not standard in solid constants)
        "meshing_arguments": {"cell_size": 0.03},
        "grid_type": "simplex",
        "theta_rad": theta_rad,
        "num_bem_segments": 1000,
    }

    # Convergence analysis setup
    conv_analysis = ConvergenceAnalysis(
        model_class=ManuSneddonSetup2d,
        model_params=copy.deepcopy(params),
        levels=2,
        spatial_refinement_rate=2,
    )

    # Calculate and return the order of convergence for the displacement
    order_dict = conv_analysis.order_of_convergence(conv_analysis.run_analysis())
    return order_dict


def test_order_of_convergence(actual_ooc: dict) -> None:
    """Test observed order of convergence."""
    # We  the order of L2 convergence on the fracture of displacement to be about 1.0
    assert 0.85 < actual_ooc["ooc_displacement"]
