"""
Module containing tests for cases 3a and 3b from the 2d benchmark study [1].

The tests check whether effective permeabilities and boundary conditions are correctly
assigned. Since we solved the actual model, we also implicitly check that the model does
not crash. However, we are currently not checking that the solution is the "correct"
one.

Reference:
    - [1] Flemisch, B., Berre, I., Boon, W., Fumagalli, A., Schwenck, N., Scotti, A.,
      ... & Tatomir, A. (2018). Benchmarks for single-phase flow in fractured porous
      media. Advances in Water Resources, 111, 239-258.

"""
import numpy as np
import pytest

import porepy as pp


from porepy.examples.flow_benchmark_2d_case_3 import (
    FlowBenchmark2dCase3aModel,
    FlowBenchmark2dCase3bModel,
    solid_constants,
)
from porepy.applications.test_utils.benchmarks import EffectivePermeability
from typing import Literal


class Model3aWithEffectivePermeability(
    EffectivePermeability,
    FlowBenchmark2dCase3aModel,
):
    ...


class Model3bWithEffectivePermeability(
    EffectivePermeability,
    FlowBenchmark2dCase3bModel,
):
    ...


@pytest.fixture(scope="module", params=["tpfa", "mpfa"])
def flux_discretization(request) -> Literal["tpfa", "mpfa"]:
    return request.param


@pytest.fixture(scope="module", params=["a", "b"])
def case(request) -> Literal["a", "b"]:
    return request.param


@pytest.fixture(scope="module")
def model(
    flux_discretization: Literal["tpfa", "mpfa"], case: Literal["a", "b"]
) -> Model3aWithEffectivePermeability | Model3bWithEffectivePermeability:
    """
    Run benchmark model for the given flux discretization and case variant.

    Parameters:
        flux_discretization: Either 'tpfa' or 'mpfa'
        case: Either 'a' (top-to-bottom flow ) or 'b' (left-to-right flow).

    Returns:
        A solved instance of the model.

    """
    model_params = {
        "material_constants": {"solid": solid_constants},
        "grid_type": "simplex",
        "meshing_arguments": {"cell_size": 0.1},
        "flux_discretization": flux_discretization,
    }
    if case == "a":
        model = Model3aWithEffectivePermeability(model_params)
    elif case == "b":
        model = Model3bWithEffectivePermeability(model_params)
    else:
        ValueError("Parameter combination not admissible.")
    pp.run_time_dependent_model(model)
    return model


@pytest.mark.skipped  # reason: slow
def test_effective_tangential_permeability(model) -> None:
    """Test if the permeability values are consistent with the benchmark specification.

    We expect the following values:

    Subdomain dim   | Value
    -----------------------
    2d              | 1
    1d (conductive) | 1
    1d (blocking)   | 1E-8

    Parameters:
        model: A solved instance of Model3aWithEffectivePermeability or
            Model3bWithEffectivePermeability.

    """
    for sd in model.mdg.subdomains():
        val = model.effective_tangential_permeability([sd]).value(model.equation_system)
        if sd.dim == 2:
            np.testing.assert_array_almost_equal(val, 1.0)
        elif sd.dim == 1:
            if sd.frac_num in [3, 4]:  # indices of blocking fractures
                np.testing.assert_array_almost_equal(val, 1e-8)
            else:
                np.testing.assert_array_almost_equal(val, 1.0)
        else:  # 0d tangential permeabilities are meaningless
            continue


@pytest.mark.skipped  # reason: slow
def test_effective_normal_permeability(model) -> None:
    """Test if the permeability values are consistent with the benchmark specification.

    We expect the following values:

    Interface dim   | Value
    -----------------------
    1d (conductive) | 2e8
    1d (blocking)   | 2
    0d (conductive) | 2e4
    0d (blocking)   | 4e-4

    0d conductive interfaces are the ones intersecting two conductive fractures, whereas
    0d blocking interfaces are the ones intersecting at least one blocking fracture.

    Parameters:
        model: A solved instance of Model3aWithEffectivePermeability or
            Model3bWithEffectivePermeability.

    """
    for intf in model.mdg.interfaces():
        val = model.effective_normal_permeability([intf]).value(model.equation_system)
        sd_high, sd_low = model.mdg.interface_to_subdomain_pair(intf)
        if intf.dim == 1:
            if sd_low.frac_num in [3, 4]:  # blocking 1d interface
                np.testing.assert_array_almost_equal(val, 2)
            else:  # conductive 1d interface
                np.testing.assert_array_almost_equal(val, 2e8)
        else:
            # Get the fractures intersecting the interface.
            interfaces_lower = model.subdomains_to_interfaces([sd_low], [1])
            # Get neighboring subdomains to the 0d interface
            neighbors = model.interfaces_to_subdomains(interfaces_lower)
            blocking_sd_high = [
                sd for sd in neighbors if sd.dim == 1 and sd.frac_num in [3, 4]
            ]
            # If the 0d interface is neighboring a 1d blocking fracture, the length
            # of the `block_sd_high` list should be larger than zero
            if len(blocking_sd_high) > 0:  # blocking 0d interface
                np.testing.assert_array_almost_equal(val, 4e-4)
            else:  # conductive 0d interface
                np.testing.assert_array_almost_equal(val, 2e4)


@pytest.mark.skipped  # reason: slow
def test_boundary_specification(model) -> None:
    """Check that the boundary specification is correct.

    Parameters:
        model: A solved instance of Model3aWithEffectivePermeability or
            Model3bWithEffectivePermeability.

    """
    bg, data_bg = model.mdg.boundaries(return_data=True, dim=1)[0]
    if isinstance(model, Model3aWithEffectivePermeability):
        # North boundary
        north_side = model.domain_boundary_sides(bg).north
        north_val = data_bg["iterate_solutions"]["pressure"][0][north_side]
        np.testing.assert_array_almost_equal(north_val, 4)
        # South boundary
        south_side = model.domain_boundary_sides(bg).south
        south_val = data_bg["iterate_solutions"]["pressure"][0][south_side]
        np.testing.assert_array_almost_equal(south_val, 1)
    elif isinstance(model, Model3bWithEffectivePermeability):
        # West boundary
        west_side = model.domain_boundary_sides(bg).west
        west_val = data_bg["iterate_solutions"]["pressure"][0][west_side]
        np.testing.assert_array_almost_equal(west_val, 4)
        # East boundary
        east_side = model.domain_boundary_sides(bg).east
        east_val = data_bg["iterate_solutions"]["pressure"][0][east_side]
        np.testing.assert_array_almost_equal(east_val, 1)
    else:
        raise ValueError("Model not recognized.")
