import copy
import math

import numpy as np
import pytest

import porepy as pp
import tests.functional.setups.manu_sneddon_2d as manu_sneddon_2d
from porepy.applications.convergence_analysis import ConvergenceAnalysis

# Set up the material constants
poi = 0.25
shear_modulus = 1
lam = (
    2 * shear_modulus * poi / (1 - 2 * poi)
)  # Convertion formula from shear modulus and poission to lame lambda parameter

solid = pp.SolidConstants(shear_modulus=shear_modulus, lame_lambda=lam)


def compute_frac_pts(
    theta_rad: float, a: float, height: float, length: float
) -> np.ndarray:
    """Assuming the fracture center is at the coordinate (height/2, length/2),
    compute the endpoints of a fracture given its orientation and fracture length.

    Parameters:
        theta_rad: Angle of the fracture in radians
        a: Half-length of the fracture.
        height: Height of the domain.
        length: Width of the domain.

    Returns:
        A 2x2 array where each column represents the coordinates of an end point of the
        fracture in 2D. The first column corresponds to one end point, and the second
        column corresponds to the other.

    """
    # Rotate the fracture with an angle theta_rad
    y_0 = height / 2 - a * np.cos(theta_rad)
    x_0 = length / 2 - a * np.sin(theta_rad)
    y_1 = height / 2 + a * np.cos(theta_rad)
    x_1 = length / 2 + a * np.sin(theta_rad)

    frac_pts = np.array([[x_0, y_0], [x_1, y_1]]).T
    return frac_pts


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
    length = 1.0
    theta_rad = math.radians(90 - theta_deg)

    params = {
        "prepare_simulation": True,
        "material_constants": {"solid": solid},
        "a": a,  # Half-length of the fracture
        "height": height,  # Height of the domain
        "length": length,  # Length of the domain
        "p0": 1e-4,  # Internal pressure of fracture
        "poi": poi,  # Possion ratio (Not standard in solid constants)
        "meshing_arguments": {"cell_size": 0.03},
        "theta": theta_rad,
    }

    # Convert angle to radians and compute fracture points
    params["frac_pts"] = compute_frac_pts(
        theta_rad=theta_rad, a=a, height=height, length=length
    )

    # Model for the convergence analysis
    model = manu_sneddon_2d.ManuSneddonSetup2d

    # Convergence analysis setup
    conv_analysis = ConvergenceAnalysis(
        model_class=model,
        model_params=copy.deepcopy(params),
        levels=2,
        spatial_refinement_rate=2,
        temporal_refinement_rate=1,
    )

    # Calculate and return the order of convergence for the displacement
    order_dict = conv_analysis.order_of_convergence(
        conv_analysis.run_analysis(), data_range=slice(None, None, None)
    )
    return order_dict


def test_order_of_convergence(actual_ooc: dict) -> None:
    """Test observed order of convergence."""
    # We  the order of L2 convergence on the fracture of displacement to be about 1.0
    assert 0.85 < actual_ooc["ooc_displacement"]
