"""
Module containing tests for the flow three-dimensional benchmark, case 3 from [1].

The tests check whether effective permeabilities and boundary are correctly assigned.
Since we solved the actual model, we also implicitly check that the model does not
crash. However, we are currently not checking that the solution is the "correct" one.

Reference:
    - [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Gläser, D., Keilegavlen,
      E., ... & Zulian, P. (2021). Verification benchmarks for single-phase flow in
      three-dimensional fractured porous media. Advances in Water Resources, 147,
      103759. https://doi.org/10.1016/j.advwatres.2020.103759

"""
import numpy as np
import pytest

import porepy as pp

from porepy.examples.flow_benchmark_3d_case_3 import (
    FlowBenchmark3dCase3Model,
    solid_constants,
)
from porepy.applications.test_utils.benchmarks import EffectivePermeability
from typing import Literal


class ModelWithEffectivePermeability(
    EffectivePermeability,
    FlowBenchmark3dCase3Model,
):
    """Model with functionality to calculate effective permeabilities."""


@pytest.fixture(scope="module", params=["tpfa", "mpfa"])
def flux_discretization(request) -> Literal["tpfa", "mpfa"]:
    return request.param


@pytest.fixture(scope="module")
def model(
    flux_discretization: Literal["tpfa", "mpfa"]
) -> ModelWithEffectivePermeability:
    """Run the benchmark model with the coarsest mesh resolution.

    Parameters:
        flux_discretization: Either 'tpfa' or 'mpfa'

    Returns:
        The solved model, an instance of `ModelWithEffectivePermeability`.

    """
    model_params = {
        "material_constants": {"solid": solid_constants},
        "flux_discretization": flux_discretization,
    }
    model = ModelWithEffectivePermeability(model_params)
    pp.run_time_dependent_model(model)
    return model


@pytest.mark.skipped  # reason: slow
def test_effective_tangential_permeability_values(model) -> None:
    """Test if the permeability values are consistent with the benchmark specification.

    The values are specified in Table 5 from [1]. We expect a value of effective
    tangential permeability = 1 for the 3d subdomain, 1e2 for the fractures, and 1.0
    for the fracture intersections.

    Parameters:
        model: ModelWithEffectivePermeability
            Solved model. Returned by the `model()` fixture.

    """
    for sd in model.mdg.subdomains():
        val = model.effective_tangential_permeability([sd]).value(model.equation_system)
        if sd.dim == 3:
            np.testing.assert_array_almost_equal(val, 1.0)
        elif sd.dim == 2:
            np.testing.assert_array_almost_equal(val, 1e2)
        else:  # sd.dim == 1
            np.testing.assert_array_almost_equal(val, 1.0)


@pytest.mark.skipped  # reason: slow
def test_effective_normal_permeability_values(model) -> None:
    """Test if the permeability values are consistent with the benchmark specification.

    The values are specified in Table 5, from [1]. Specifically, we expect a value of
    normal permeability = 2e6 for 2d interfaces, and a value of normal permeability
    = 2e4 for 1d interfaces.

    Parameters:
        model: ModelWithEffectivePermeability
            Solved model. Returned by the `model()` fixture.

    """
    for intf in model.mdg.interfaces():
        val = model.effective_normal_permeability([intf]).value(model.equation_system)
        if intf.dim == 2:
            np.testing.assert_array_almost_equal(val, 2e6)
        else:  # intf.dim == 1
            np.testing.assert_array_almost_equal(val, 2e4)


@pytest.mark.skipped  # reason: slow
def test_boundary_specification(model) -> None:
    """Check that the inlet and outlet boundaries are correctly specified.

    At the inlet boundary, we check if the total amount of fluid is entering into the
    domain. At the outlet boundary, we check that the pressure value is zero.

    """
    bg, data_bg = model.mdg.boundaries(return_data=True, dim=2)[0]

    # Inlet boundary
    south_side = model.domain_boundary_sides(bg).south
    inlet_flux = np.sum(data_bg["iterate_solutions"]["darcy_flux"][0][south_side])
    assert np.isclose(inlet_flux, -1 / 3, atol=1e-5)

    # Outlet boundary
    north_side = model.domain_boundary_sides(bg).north
    outlet_pressure = np.sum(data_bg["iterate_solutions"]["pressure"][0][north_side])
    assert np.isclose(outlet_pressure, 0, atol=1e-5)
