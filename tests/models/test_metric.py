import numpy as np
import pytest
from deepdiff import DeepDiff

import porepy as pp
from porepy.applications.md_grids.model_geometries import OrthogonalFractures3d


@pytest.fixture(scope="module")
def ortho3d_model() -> pp.PorePyModel:
    # Set up parameters for a unit cube with three orthogonal fractures
    params = {
        "domain_size": 1.0,
        "fracture_indices": [0, 1, 2],  # Use all three orthogonal fractures
        "material_constants": {
            "solid": pp.SolidConstants(residual_aperture=1),
        },
    }

    class Model(OrthogonalFractures3d, pp.Poromechanics):
        pass

    model = Model(params)
    model.prepare_simulation()
    return model


def test_euclidean_metric_basic():
    """Simple unit test of EuclideanMetric - independent of models."""
    m = pp.EuclideanMetric()
    arr = np.array([3.0, 4.0])
    assert np.isclose(m(arr), 5.0 / np.sqrt(2))
    assert m(np.array([])) == 0.0
    assert m(np.array([1.0])) == 1.0
    for i in range(1, 10):
        arr = np.ones(i)
        assert np.isclose(m(arr), 1.0)


def test_euclidean_metric_on_model(ortho3d_model):
    """Test that one arrays return 1.

    Note the scaling of the Euclidean metric - it divides by sqrt(size).

    """
    m = pp.EuclideanMetric()
    for g in ortho3d_model.mdg.subdomains():
        arr = np.ones(g.num_cells)
        assert np.isclose(m(arr), 1)


def test_variable_based_euclidean_metric_on_model(ortho3d_model):
    m = pp.VariableBasedEuclideanMetric(ortho3d_model)
    dummy_variable = ortho3d_model.equation_system.get_variable_values(
        time_step_index=0
    )
    dummy_variable.fill(1.0)
    metric_values = m(dummy_variable)
    for _, value in metric_values.items():
        assert np.isclose(value, 1.0)


def test_variable_based_lebesgue_metric_on_model(ortho3d_model):
    """Test that one arrays return correct Lebesgue metric."""

    # Create a dummy variable array filled with ones.
    dummy_variable = ortho3d_model.equation_system.get_variable_values(
        time_step_index=0
    )
    dummy_variable.fill(1.0)

    # Compute the corresponding Lebesgue metric.
    m = pp.VariableBasedLebesgueMetric(ortho3d_model)
    metric_values = m(dummy_variable)

    # Manually compute expected values - L2 integral of 1 over the domain
    # (incl. dimensionality and sqrt).
    variables = ortho3d_model.equation_system.variables
    result = {v.name: 0.0 for v in variables}
    for v in variables:
        domain = v.domain
        volume = domain.cell_volumes.sum()
        dimensionality = v._cells + v._faces + v._nodes
        result[v.name] += volume * dimensionality
    for name in result:
        result[name] = np.sqrt(result[name])

    for name in result:
        assert np.isclose(result[name], metric_values[name])


def test_equation_based_euclidean_metric_on_model(ortho3d_model):
    # Generate a dummy residual array filled with ones.
    # NOTE: Evaluate Jacobian to initialize the equation system properly.
    _, dummy_residual_array = ortho3d_model.equation_system.assemble()
    dummy_residual_array.fill(1.0)

    # Compute Lebesgue metric values.
    m = pp.EquationBasedEuclideanMetric(ortho3d_model)
    metric_values = m(dummy_residual_array)

    # Define expected values - 1's for scaled Euclidean norm.
    equations = ortho3d_model.equation_system._equations
    result = {name: 1.0 for name in equations}

    # Since there is no wells, the well equation should have a zero contribution.
    if "well_flux_equation" in result:
        result["well_flux_equation"] = 0.0

    # Make sure that the dictionaries are the same.
    deepdiff_result = DeepDiff(
        result,
        metric_values,
        significant_digits=6,
        ignore_order=True,
        number_format_notation="e",
        ignore_numeric_type_changes=True,
    )
    assert deepdiff_result == {}


def test_equation_based_lebesgue_metric_on_model(ortho3d_model):
    """Test whether the integration of 1-s over the domain results in volume."""

    # Fetch the equations.
    equations = ortho3d_model.equation_system._equations

    # Generate a dummy residual array filled with ones scaled with the cell volumes.
    # NOTE: Evaluate Jacobian to initialize the equation system properly.
    _, dummy_residual_array = ortho3d_model.equation_system.assemble()
    dummy_residual_array.fill(1.0)

    # Scale with the right cell volumes.
    for eqn in equations:
        domains = ortho3d_model.equation_system._equation_image_space_composition[
            eqn
        ].keys()
        if len(domains) == 0:
            continue
        indices = ortho3d_model.equation_system.assembled_equation_indices[eqn]
        cell_volumes = np.hstack([_sd.cell_volumes for _sd in domains])
        eq_dim = ortho3d_model.equation_system._equation_image_size_info[eqn]["cells"]
        dummy_residual_array[indices] *= np.repeat(cell_volumes, repeats=eq_dim)

    # Compute Lebesgue metric values.
    m = pp.EquationBasedLebesgueMetric(ortho3d_model)
    metric_values = m(dummy_residual_array)

    # Define expected values - L2 integral of 1 over the domain
    # (incl. dimensionality and sqrt).
    result = {name: 0.0 for name in equations}
    for eqn in equations:
        domains = ortho3d_model.equation_system._equation_image_space_composition[
            eqn
        ].keys()
        volume = sum([domain.cell_volumes.sum() for domain in domains])
        dimensionality = ortho3d_model.equation_system._equation_image_size_info[eqn][
            "cells"
        ]
        result[eqn] += volume * dimensionality
    for name in result:
        result[name] = np.sqrt(result[name])

    # Make sure that the dictionaries are the same.
    deepdiff_result = DeepDiff(
        result,
        metric_values,
        significant_digits=6,
        ignore_order=True,
        number_format_notation="e",
        ignore_numeric_type_changes=True,
    )
    assert deepdiff_result == {}
