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
    solid_constants
)


class ModelWithEffectivePermeability(FlowBenchmark3dCase3Model):
    """Mixin that contains the computation of effective permeabilities."""

    def effective_tangential_permeability(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Retrieves the effective tangential permeability, see Eq. 6a from [1].

        This method implicitly assumes that, in each subdomain, the effective
        tangential permeability can be fully represented by one scalar per cell.

        The effective tangential permeability is the permeability tensor multiplied
        by the specific volume. PorePy "transforms" the intrinsic permeability into
        an effective one using the method `operator_to_SecondOrderTensor` defined in
        the mixin class `~porepy.models.constitutive_laws.SecondOrderTensorUtils`.

        Parameters:
            subdomains: list of pp.Grid
                List of subdomain grids.

        Returns:
            Wrapped ad operator containing the effective tangential permeabilities
            for the given list of subdomains.

        """
        values = []
        size = self.mdg.num_subdomain_cells()
        for sd in subdomains:
            d = self.mdg.subdomain_data(sd)
            val_loc = d[pp.PARAMETERS][self.darcy_keyword][
                "second_order_tensor"
            ].values[0][0]
            values.append(val_loc)
        return pp.wrap_as_dense_ad_array(
            np.hstack(values), size, "effective_tangential_permeability"
        )

    def effective_normal_permeability(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Operator:
        """
        Computes the effective normal permeability, see Eq. 6b from [1].

        The effective normal permeability is the scalar that multiplies the pressure
        jump in the continuous interface law.

        Parameters:
            interfaces: List of pp.MortarGrid
                List of interface grids.

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
        effective_normal_permeability.set_name("effective_normal_permeability")

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
