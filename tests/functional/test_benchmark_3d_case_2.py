"""
Module containing tests for the flow three-dimensional benchmark, case 2 from [1].

The tests check whether effective permeabilities and boundary are correctly assigned.
Since we solved the actual model, we also implicitly check that the model does not
crash. However, we are currently not checking that the solution is the "correct" one.

Reference:
    - [1] Berre, I., Boon, W. M., Flemisch, B., Fumagalli, A., Gläser, D., Keilegavlen,
      E., ... & Zulian, P. (2021). Verification benchmarks for single-phase flow in
      three-dimensional fractured porous media. Advances in Water Resources, 147,
      103759. https://doi.org/10.1016/j.advwatres.2020.103759

"""

from typing import Literal

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.benchmarks import EffectivePermeability
from porepy.examples.flow_benchmark_3d_case_2 import (
    FlowBenchmark3dCase2Model,
    solid_constants_blocking,
    solid_constants_conductive,
)


class ModelWithEffectivePermeability(
    EffectivePermeability,
    FlowBenchmark3dCase2Model,
):
    """Model with functionality to calculate effective permeabilities."""


@pytest.fixture(scope="module", params=[
    pytest.param("tpfa", id="tpfa"),
    pytest.param("mpfa", id="mpfa", marks=pytest.mark.skipped)
])
def flux_discretization(request) -> Literal["tpfa", "mpfa"]:
    return request.param

@pytest.fixture(scope="module")
def model_conductive(
    flux_discretization: Literal["tpfa", "mpfa"],
) -> ModelWithEffectivePermeability:
    """Run benchmark model for case 1: Conductive fractures.

    Parameters:
        flux_discretization: Either 'tpfa' or 'mpfa'.

    Returns:
        The solved model, an instance of `ModelWithEffectivePermeability`.

    """
    model_params = {
        "material_constants": {"solid": solid_constants_conductive},
        "flux_discretization": flux_discretization,
        "times_to_export": [],  # Suppress output for tests
    }
    model = ModelWithEffectivePermeability(model_params)
    pp.run_time_dependent_model(model)
    return model


@pytest.fixture(scope="module")
def model_blocking(
    flux_discretization: Literal["tpfa", "mpfa"],
) -> ModelWithEffectivePermeability:
    """Run benchmark model for case 2: blocking fractures.

    Parameters:
        flux_discretization: Either 'tpfa' or 'mpfa'.

    Returns:
        The solved model, an instance of `ModelWithEffectivePermeability`.

    """
    model_params = {
        "material_constants": {"solid": solid_constants_blocking},
        "flux_discretization": flux_discretization,
        "times_to_export": [],  # Suppress output for tests
    }
    model = ModelWithEffectivePermeability(model_params)
    pp.run_time_dependent_model(model)
    return model


@pytest.mark.parametrize("model_fixture", [
    pytest.param("model_conductive", id="conductive"),
    pytest.param("model_blocking", id="blocking", marks=pytest.mark.skipped)
])
def test_effective_tangential_permeability_values(
        request,
        model_fixture: str,
        flux_discretization: Literal['tpfa', 'mpfa'],
) -> None:
    """Test if the permeability values are consistent with the benchmark specification.

    The values are specified in Table 4 from [1]. Expected values are:

        3D domain:  1 (high perm regions), 0.1 (low perm regions)
        2D domains: 1 (conductive case), 1e-8 (blocking case)
        1D domains: 1e-4 (conductive case), 1e-12 (blocking case)

    Parameters:
        model: ModelWithEffectivePermeability
            Solved model. Returned by the `model()` fixture.

    """
    # Get the actual model instance from the fixture name
    model = request.getfixturevalue(model_fixture)

    for sd in model.mdg.subdomains():
        val = model.equation_system.evaluate(
            model.effective_tangential_permeability([sd])
        )

        if sd.dim == 3:
            msk = model._low_perm_zones(sd)
            np.testing.assert_array_almost_equal(val[~msk], 1.0)
            np.testing.assert_array_almost_equal(val[msk], 0.1)
        elif sd.dim == 2:
            if model_fixture == "model_conductive":
                np.testing.assert_array_almost_equal(val, 1)
            else:
                np.testing.assert_array_almost_equal(val, 1e-8)
        elif sd.dim == 1:
            if model_fixture == "model_conductive":
                np.testing.assert_array_almost_equal(val, 1e-4)
            else:
                np.testing.assert_array_almost_equal(val, 1e-12)
        else:
            continue  # zero-dimensional subdomains


@pytest.mark.parametrize("model_fixture", [
    pytest.param("model_conductive", id="conductive"),
    pytest.param("model_blocking", id="blocking", marks=pytest.mark.skipped)
])
def test_effective_normal_permeability_values(
        request,
        model_fixture: str,
        flux_discretization: Literal['tpfa', 'mpfa'],
) -> None:
    """Test if the permeability values are consistent with the benchmark specification.

    The values are specified in Table 4, from [1]. Expected values are:

        2D interfaces: 2e8 (conductive case), 2 (blocking case)
        1D interfaces: 2e4 (conductive case), 2e-4 (blocking case)
        0D interfaces: 2 (conductive case), 2e-8 (blocking case).

    Parameters:
        model: ModelWithEffectivePermeability
            Solved model. Returned by the `model()` fixture.

    """
    # Get the actual model instance from the fixture name
    model = request.getfixturevalue(model_fixture)

    for intf in model.mdg.interfaces():
        val = model.equation_system.evaluate(
            model.effective_normal_permeability([intf])
        )
        if intf.dim == 2:
            if model_fixture == "model_conductive":
                np.testing.assert_array_almost_equal(val, 2e8)
            else:
                np.testing.assert_array_almost_equal(val, 2)
        elif intf.dim == 1:
            if model_fixture == "model_conductive":
                np.testing.assert_array_almost_equal(val, 2e4)
            else:
                np.testing.assert_array_almost_equal(val, 2e-4)
        else:  # 0d domains
            if model_fixture == "model_conductive":
                np.testing.assert_array_almost_equal(val, 2)
            else:
                np.testing.assert_array_almost_equal(val, 2e-8)


@pytest.mark.parametrize("model_fixture", [
    pytest.param("model_conductive", id="conductive"),
    pytest.param("model_blocking", id="blocking", marks=pytest.mark.skipped)
])
def test_boundary_specification(
        request,
        model_fixture: str,
        flux_discretization: Literal['tpfa', 'mpfa'],
    ) -> None:
    """Check that the inlet and outlet boundaries are correctly specified.

    At the inlet boundary, we check if the total amount of fluid is entering into the
    domain. At the outlet boundary, we check that the pressure value is one.

    """
    # Get the actual model instance from the fixture name
    model = request.getfixturevalue(model_fixture)

    # Retrieve inlet and outlet faces from the boundary grid
    bg, data_bg = model.mdg.boundaries(return_data=True, dim=2)[0]
    cc = bg.cell_centers

    # Check on inlet faces
    inlet_faces = np.logical_and.reduce(
        tuple(cc[i, :] < 0.25 + 1e-8 for i in range(3))
    )
    inlet_flux = np.sum(data_bg["iterate_solutions"]["darcy_flux"][0][inlet_faces])
    assert np.isclose(inlet_flux, -0.1875, atol=1e-5)

    # Check on outlet faces
    outlet_faces = np.logical_and.reduce(
        tuple(cc[i, :] > 0.875 + 1e-8 for i in range(3))
    )
    outlet_pressure = data_bg["iterate_solutions"]["pressure"][0][outlet_faces]
    np.testing.assert_allclose(outlet_pressure, 1, atol=1e-5)
