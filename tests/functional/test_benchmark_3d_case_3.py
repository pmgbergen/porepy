"""
Module containing tests for the flow three-dimensional benchmark, case 3.

The tests check whether effective permeabilities and boundary are correctly assigned.
Since we solved the actual model, we also implicitly check that the model does not
crash. However, we are currently not checking that the solution is the "correct" one.

"""
import sys

import numpy as np
import pytest

import porepy as pp

# Append the top PorePy folder to the path, to allow for imports of the examples folder
sys.path.append("../..")

from examples.flow_benchmark_3d_case_3 import (
    solid_constants,
    FlowBenchmark3dCase3Model,
)


class ModelWithEffectivePermeability(FlowBenchmark3dCase3Model):
    """Mixin that contains the computation of effective normal permeabilities."""

    def effective_normal_permeability(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """
        Computes the effective normal permeability, see Eq. 6b from [1].

        Note:
            The effective normal permeability is the scalar that multiplies the pressure
            jump in the continuous interface law.

        Reference:
            [1] https://doi.org/10.1016/j.advwatres.2020.103759

        Parameters:
            interfaces: List of interfaces.

        Returns:
            Wrapped ad operator containing the effective normal permeabilities for the
            given list of interfaces.
        """
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        normal_gradient = pp.ad.Scalar(2) * (
            projection.secondary_to_mortar_avg
            @ self.aperture(subdomains) ** pp.ad.Scalar(-1)
        )

        effective_normal_permeability = (
            self.specific_volume(interfaces)
            * self.normal_permeability(interfaces)
            * normal_gradient
        )

        return effective_normal_permeability


@pytest.fixture(scope="module")
def model() -> ModelWithEffectivePermeability:
    """Run the benchmark model with the coarsest mesh resolution.

    Returns:
        The solved model, an instance of `ModelWithEffectivePermeability`.

    """
    model = ModelWithEffectivePermeability(
        {"material_constants": {"solid": solid_constants}}
    )
    pp.run_time_dependent_model(model, {})
    return model


@pytest.mark.skipped  # reason: slow
def test_effective_tangential_permeability_values(model) -> None:
    """Test if the permeability values are consistent with the benchmark specification.

    Parameters:
        model: ModelWithEffectivePermeability
            Solved model. Returned by the `model()` fixture.

    """
    for sd, d in model.mdg.subdomains(return_data=True):
        val = np.mean(
            d[pp.PARAMETERS][model.darcy_keyword]["second_order_tensor"].values[0][0]
        )
        if sd.dim == 3:
            np.testing.assert_allclose(val, 1)
        elif sd.dim == 2:
            np.testing.assert_allclose(val, 1e2)
        else:  # sd.dim == 1
            np.testing.assert_allclose(val, 1)


@pytest.mark.skipped  # reason: slow
def test_effective_normal_permeability_values(model) -> None:
    """Test if the permeability values are consistent with the benchmark specification.

    Parameters:
        model: ModelWithEffectivePermeability
            Solved model. Returned by the `model()` fixture.

    """
    for intf in model.mdg.interfaces():
        val = np.mean(
            model.effective_normal_permeability([intf]).value(model.equation_system)
        )
        if intf.dim == 2:
            np.testing.assert_allclose(val, 2e6)
        else:
            np.testing.assert_allclose(val, 2e4)


@pytest.mark.skipped  # reason: slow
def test_boundary_specification(model) -> None:
    """Check that the inlet and outlet boundaries are correctly specified.

    Note:
        At the inlet boundary, we check if the total amount of fluid is entering into
        the domain. At the outlet boundary, we check that the pressure value is zero.

    """
    bg, data_bg = model.mdg.boundaries(return_data=True, dim=2)[0]

    # Inlet boundary
    south_side = model.domain_boundary_sides(bg).south
    inlet_flux = np.sum(data_bg["iterate_solutions"]["darcy_flux"][0][south_side])
    np.testing.assert_allclose(inlet_flux, desired=-1 / 3, atol=1e-5)

    # Outlet boundary
    north_side = model.domain_boundary_sides(bg).north
    outlet_pressure = np.sum(data_bg["iterate_solutions"]["pressure"][0][north_side])
    np.testing.assert_allclose(outlet_pressure, desired=0, atol=1e-5)
