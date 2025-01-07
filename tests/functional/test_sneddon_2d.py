from porepy.applications.convergence_analysis import ConvergenceAnalysis


# ----> Retrieve actual order of convergence
@pytest.fixture(scope="module")
def actual_ooc(
    material_constants: dict, reference_values: pp.ReferenceVariableValues
) -> list[list[dict[str, float]]]:
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
    ooc_results = conv_analysis.order_of_convergence(conv_analysis.run_analysis())
    ooc.append(ooc_results)

    return ooc


# ----> Set desired order of convergence
@pytest.fixture(scope="module")
def desired_ooc() -> list[list[dict[str, float]]]:
    """Set desired order of convergence.

    Returns:
        List of lists of dictionaries, containing the desired order of convergence.

    """
    desired_ooc_2d = 1.8757689483196147        
 
    return desired_ooc_2d


@pytest.mark.skipped  # reason: slow
@pytest.mark.parametrize("grid_type_idx", [0, 1])
def test_order_of_convergence(
    var: str,
    dim_idx: int,
    grid_type_idx: int,
    actual_ooc: list[list[dict[str, float]]],
    desired_ooc: list[list[dict[str, float]]],
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
    if not (dim_idx == 1 and grid_type_idx == 1):  # no analysis for 3d and simplices
        assert 1.0 < actual_ooc[dim_idx][grid_type_idx]["ooc_" + var]

    if grid_type_idx == 0:  # Cartesian
        assert np.isclose(
            desired_ooc[dim_idx][grid_type_idx]["ooc_" + var],
            actual_ooc[dim_idx][grid_type_idx]["ooc_" + var],
            atol=1e-3,  # allow for an absolute difference of 0.001 in OOC
            rtol=1e-3,  # allow for 0.1% of relative difference in OOC
        )
    else:  # Simplex
        if dim_idx == 0:  # no analysis for 3d and simplices
            assert np.isclose(
                desired_ooc[dim_idx][grid_type_idx]["ooc_" + var],
                actual_ooc[dim_idx][grid_type_idx]["ooc_" + var],
                atol=1e-1,  # allow for an absolute difference of 0.1 in OOC
                rtol=5e-1,  # allow for 5% of relative difference in OOC
            )
