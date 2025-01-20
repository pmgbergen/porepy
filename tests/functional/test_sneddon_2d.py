from porepy.applications.convergence_analysis import ConvergenceAnalysis
import porepy as pp
import numpy as np
import pytest

import test_sneddon_2d




# ----> Retrieve actual L2-errors
@pytest.fixture(scope="module")
def actual_l2_errors(material_constants: dict) -> :
    """Run verification setups and retrieve results for the scheduled times.

    Parameters:
        material_constants: Dictionary containing the material constant classes.

    Returns:
        List of lists of dictionaries of actual relative errors. The outer list contains
        two items, the first contains the results for 2d and the second contains the
        results for 3d. Both inner lists contain three items each, each of which is a
        dictionary of results for the scheduled times, i.e., 0.5 [s] and 1.0 [s].

    """

    # Define model parameters
    model_params = {
        "grid_type": "cartesian",
        "material_constants": material_constants,
        "meshing_arguments": {"cell_size": 0.25},
        "manufactured_solution": "nordbotten_2016",
        "time_manager": pp.TimeManager([0, 0.5, 1.0], 0.5, True),
    }

    # Retrieve actual L2-relative errors.
    errors = []
    # Loop through models, i.e., 2d and 3d.
    model  = SneddonSetup2d()
    
    # Make deep copy of params to avoid nasty bugs.
    setup = model(deepcopy(model_params))
    pp.run_time_dependent_model(setup)
    
    errors_setup: list[dict[str, float]] = []
    # Loop through results, i.e., results for each scheduled time.
    for result in setup.results:
        errors_setup.append(
            {
                "error_pressure": getattr(result, "error_pressure"),
            }
        )
    errors.append(errors_setup)
    return errors


# ----> Retrieve actual order of convergence
@pytest.fixture(scope="module")
def actual_ooc(
) -> float:
    """Retrieve actual order of convergence.

    """
    
    
       
    
    ooc = []
    # Loop through the models
    ooc_setup: list[dict[str, float]] = []
    # Loop through grid type
    # We do not perform a convergence analysis with simplices in 3d
    grid_type = "simplex"
        
    conv_analysis = ConvergenceAnalysis(
        model_class=model,
        model_params=deepcopy(params),
        levels=4,
        spatial_refinement_rate=2,
        temporal_refinement_rate=4,
    )
    order = conv_analysis.order_of_convergence(conv_analysis.run_analysis())
    

    return order 


# ----> Set desired order of convergence
@pytest.fixture(scope="module")
def desired_ooc() -> float:
    """Set desired order of convergence.

    Returns:
        List of lists of dictionaries, containing the desired order of convergence.

    """
    desired_ooc_2d = 2.0      
 
    return desired_ooc_2d



def test_order_of_convergence(
    actual_ooc,
    desired_ooc,
) -> None:
    """Test observed order of convergence.

    Note:
        We set more flexible tolerances for simplicial grids compared to Cartesian
        grids. This is because we would like to allow for slight changes in the
        order of convergence if the meshes change, i.e., in newer versions of Gmsh.

    Parameters:
        var: Name of the variable to be tested.
        grid_type_idx: Index to identify the grid type; `0` for Cartesian, and `1`
            for simplices.
        dim_idx: Index to identify the dimensionality of the problem; `0` for 2d, and
            `1` for 3d.
        actual_ooc: List of lists of dictionaries containing the actual observed
            order of convergence.
        desired_ooc: List of lists of dictionaries containing the desired observed
            order of convergence.

    """
    # We require the order of convergence to always be larger than 1.0
    assert 1.0 < actual_ooc


    assert np.isclose(
        desired_ooc,
        actual_ooc,
        atol=1e-1,  # allow for an absolute difference of 0.1 in OOC
        rtol=5e-1,  # allow for 5% of relative difference in OOC
    )
