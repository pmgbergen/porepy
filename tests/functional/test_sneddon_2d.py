import porepy as pp
import numpy as np
import pytest
import math
import copy
import  tests.functional.setups.manu_sneddon_2d as manu_sneddon_2d
from porepy.applications.convergence_analysis import ConvergenceAnalysis

# ----> Set up the material constants
poi = 0.25
shear_modulus = 1
lam = (
    2 * shear_modulus * poi / (1 - 2 * poi)
)  # Convertion formula from shear modulus and poission to lame lambda parameter

solid = pp.SolidConstants(shear_modulus=shear_modulus, lame_lambda=lam)
   

def compute_frac_pts(
    theta_rad: float, a: float, height: float, length: float
) -> np.ndarray:
    """
    Assuming the fracture center is at the coordinate (height/2, length/2),
    compute the endpoints of a fracture given its orientation and fracture length.

    Parameters:
        theta_rad: Angle of the fracture in radians
        a: Half-length of the fracture.
        height: Height of the domain.
        length: Width of the domain.

    Returns:
        frac_pts : A 2x2 array where each column represents the coordinates of an end point of the fracture in 2D.
            The first column corresponds to one end point, and the second column corresponds to the other.

    """
    # Rotate the fracture with an angle theta_rad
    y_0 = height / 2 - a * np.cos(theta_rad)
    x_0 = length / 2 - a * np.sin(theta_rad)
    y_1 = height / 2 + a * np.cos(theta_rad)
    x_1 = length / 2 + a * np.sin(theta_rad)

    frac_pts = np.array([[x_0, y_0], [x_1, y_1]]).T
    return frac_pts


# ----> Retrieve actual order of convergence
@pytest.fixture(scope="module")
def actual_ooc(
) -> float:
    """Retrieve actual order of convergence.
    
    Returns: A dictionary containing the order of convergence for the displacement.
    """

    # Angle of the fracture
    theta = 30

    params = {
        "prepare_simulation": True,
        "material_constants": {"solid": solid},
        "a" : 0.3,
        "height": 1.0,
        "length": 1.0,
        "p0": 1e-4,
        "poi": poi,
        "meshing_arguments": {"cell_size": 0.03},
    }
    
    # Construct the fracture points for the given angle and length
    params["theta"] = math.radians(90 - theta)
    params["frac_pts"]  = compute_frac_pts(params["theta"], params["a"], height=params["height"], length=params["length"])
    
    model =  manu_sneddon_2d.MomentumBalanceGeometryBC

    # Construct the convergence analysis object, which does embedd the model, parameters refinementlevels
    # for the convergence analysis
    conv_analysis = ConvergenceAnalysis(
        model_class=model,
        model_params= copy.deepcopy(params),
        levels=2,
        spatial_refinement_rate=2,
        temporal_refinement_rate=1
    )
    
    # Dictonary containing the order of convergence for the displacement
    order_dict = conv_analysis.order_of_convergence(conv_analysis.run_analysis(), data_range=slice(None, None, None))
    return order_dict



def test_order_of_convergence(
    actual_ooc,
) -> None:
    """Test observed order of convergence.
    """
    # We require the order of convergence to always be about 1.0 
    assert 0.85 <   actual_ooc["ooc_displacement"]  
