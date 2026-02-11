"""Unit tests for various metric implementations in PorePy."""

from typing import Literal

import numpy as np
import pytest
import sympy as sp
from deepdiff import DeepDiff

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.applications.md_grids.model_geometries import OrthogonalFractures3d


@pytest.fixture(scope="module")
def orthogonal_3d_model() -> pp.PorePyModel:
    # Set up parameters for a unit cube with three orthogonal fractures.
    params = {
        "domain_size": 1.0,
        "fracture_indices": [0, 1, 2],  # Use all three orthogonal fractures.
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


def test_euclidean_metric_on_model(orthogonal_3d_model):
    """Test that one arrays return 1.

    Note the scaling of the Euclidean metric - it divides by sqrt(size).

    """
    m = pp.EuclideanMetric()
    for g in orthogonal_3d_model.mdg.subdomains():
        arr = np.ones(g.num_cells)
        assert np.isclose(m(arr), 1)


def test_variable_based_euclidean_metric_on_model(orthogonal_3d_model):
    m = pp.VariableBasedEuclideanMetric(orthogonal_3d_model)
    dummy_variable = orthogonal_3d_model.equation_system.get_variable_values(
        time_step_index=0
    )
    dummy_variable.fill(1.0)
    metric_values = m(dummy_variable)
    for _, value in metric_values.items():
        assert np.isclose(value, 1.0)


def test_variable_based_lebesgue_metric_on_model(orthogonal_3d_model):
    """Test that one arrays return correct Lebesgue metric."""

    # Create a dummy variable array filled with ones.
    dummy_variable = orthogonal_3d_model.equation_system.get_variable_values(
        time_step_index=0
    )
    dummy_variable.fill(1.0)

    # Compute the corresponding Lebesgue metric.
    m = pp.VariableBasedLebesgueMetric(orthogonal_3d_model)
    metric_values = m(dummy_variable)

    # Manually compute expected values - L2 integral of 1 over the domain
    # (incl. dimensionality and sqrt).
    variables = orthogonal_3d_model.equation_system.variables
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


def test_equation_based_euclidean_metric_on_model(orthogonal_3d_model):
    # Generate a dummy residual array filled with ones.
    # NOTE: Evaluate Jacobian to initialize the equation system properly.
    _, dummy_residual_array = orthogonal_3d_model.equation_system.assemble()
    dummy_residual_array.fill(1.0)

    # Compute Lebesgue metric values.
    m = pp.EquationBasedEuclideanMetric(orthogonal_3d_model)
    metric_values = m(dummy_residual_array)

    # Define expected values - 1's for scaled Euclidean norm.
    equations = orthogonal_3d_model.equation_system.equations
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


def test_equation_based_lebesgue_metric_on_model(orthogonal_3d_model):
    """Test whether the integration of 1-s over the domain results in volume."""

    # Fetch the equations.
    equations = orthogonal_3d_model.equation_system.equations

    # Generate a dummy residual array filled with ones scaled with the cell volumes.
    # NOTE: Evaluate Jacobian to initialize the equation system properly.
    _, dummy_residual_array = orthogonal_3d_model.equation_system.assemble()
    dummy_residual_array.fill(1.0)

    # Scale with the right cell volumes.
    for eqn in equations:
        domains = orthogonal_3d_model.equation_system._equation_image_space_composition[
            eqn
        ].keys()
        if len(domains) == 0:
            continue
        indices = orthogonal_3d_model.equation_system.assembled_equation_indices[eqn]
        cell_volumes = np.hstack([_sd.cell_volumes for _sd in domains])
        eq_dim = orthogonal_3d_model.equation_system._equation_image_size_info[eqn][
            "cells"
        ]
        dummy_residual_array[indices] *= np.repeat(cell_volumes, repeats=eq_dim)

    # Compute Lebesgue metric values.
    m = pp.EquationBasedLebesgueMetric(orthogonal_3d_model)
    metric_values = m(dummy_residual_array)

    # Define expected values - L2 integral of 1 over the domain
    # (incl. dimensionality and sqrt).
    result = {name: 0.0 for name in equations}
    for eqn in equations:
        domains = orthogonal_3d_model.equation_system._equation_image_space_composition[
            eqn
        ].keys()
        volume = sum([domain.cell_volumes.sum() for domain in domains])
        dimensionality = orthogonal_3d_model.equation_system._equation_image_size_info[
            eqn
        ]["cells"]
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


class UnitCube:
    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(3, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        return {"cell_size": 0.05, "cell_size_boundary": 0.05}

    def grid_type(self) -> Literal["simplex"]:
        return "simplex"


class DummyVariables(pp.VariableMixin):
    def create_variables(self) -> None:
        self.equation_system.create_variables(
            "dummy_variable_x",
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "-"},
        )
        self.equation_system.create_variables(
            "dummy_variable_y",
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "-"},
        )
        self.equation_system.create_variables(
            "dummy_variable_z",
            subdomains=self.mdg.subdomains(),
            tags={"si_units": "-"},
        )

    def dummy_variable_x(self, subdomains):
        return self.equation_system.md_variable("dummy_variable_x", subdomains)

    def dummy_variable_y(self, subdomains):
        return self.equation_system.md_variable("dummy_variable_y", subdomains)

    def dummy_variable_z(self, subdomains):
        return self.equation_system.md_variable("dummy_variable_z", subdomains)


class DummyEquations(pp.PorePyModel):
    def set_equations(self):
        subdomains = self.mdg.subdomains()
        sd_eq = self.sd_eq(subdomains)
        self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})

    def sd_eq(self, subdomains):
        variable_x = self.dummy_variable_x(subdomains)
        variable_y = self.dummy_variable_y(subdomains)
        variable_z = self.dummy_variable_z(subdomains)
        coeff = self.params.get("coeff", [0])
        exp_x = self.params.get("exp_x", [0])
        exp_y = self.params.get("exp_y", [0])
        exp_z = self.params.get("exp_z", [0])
        polynomial_expression = sum(
            [
                pp.ad.Scalar(c)
                * variable_x ** pp.ad.Scalar(e_x)
                * variable_y ** pp.ad.Scalar(e_y)
                * variable_z ** pp.ad.Scalar(e_z)
                for c, e_x, e_y, e_z in zip(coeff, exp_x, exp_y, exp_z)
            ]
        )
        # Compute mass weighted integral of the polynomial expression
        # to mimick a typical equation residual.
        mass_weighted_expression = self.volume_integral(
            polynomial_expression, subdomains, 1
        )
        mass_weighted_expression.set_name("sd_eq")
        return mass_weighted_expression


class SimpleVolumeIntegralMixin(pp.models.constitutive_laws.DimensionReduction):
    """Fetch only volume integral from BalanceEquation."""

    def volume_integral(
        self,
        integrand: pp.ad.Operator,
        grids: pp.GridLikeSequence,
        dim: int,
    ) -> pp.ad.Operator:
        return pp.BalanceEquation.volume_integral(self, integrand, grids, dim)


class DummyModel(  # type: ignore[misc]
    UnitCube,
    DummyVariables,
    DummyEquations,
    SimpleVolumeIntegralMixin,
    pp.SolutionStrategy,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.BoundaryConditionMixin,
    pp.InitialConditionMixin,
    pp.DataSavingMixin,
): ...


def test_variable_based_lebesgue_metric_with_model():
    """Compare analytical and numerical Lebesgue norms for random polynomial expressions
    via variables."""

    # Define symbols
    x, y, z = sp.symbols("x y z")

    # Define a random polynomial in x,y,z.
    coeffs = np.random.randint(-5, 6, size=10)
    exponents_x = np.random.randint(0, 4, size=10)
    exponents_y = np.random.randint(0, 4, size=10)
    exponents_z = np.random.randint(0, 4, size=10)
    expr = sum(
        c * x**e_x * y**e_y * z**e_z
        for c, e_x, e_y, e_z in zip(coeffs, exponents_x, exponents_y, exponents_z)
    )

    # Compute the analytical L2 norm over the unit cube.
    l2_norm_analytical = float(
        sp.sqrt(sp.integrate(expr**2, (x, 0, 1), (y, 0, 1), (z, 0, 1)))
    )

    # Evaluate the numerical norm using the VariableBasedLebesgueMetric.
    # Use a minimal setup of a dummy model.
    model = DummyModel()
    m_var = pp.VariableBasedEuclideanMetric(model)
    model.prepare_simulation()

    # Use cell centers to define the polynomial expression.
    assert len(model.mdg.subdomains()) == 1
    cell_center_x = model.mdg.subdomains()[0].cell_centers[0, :]
    cell_center_y = model.mdg.subdomains()[0].cell_centers[1, :]
    cell_center_z = model.mdg.subdomains()[0].cell_centers[2, :]
    polynomial_expression = np.zeros_like(cell_center_x)
    for c, e_x, e_y, e_z in zip(coeffs, exponents_x, exponents_y, exponents_z):
        polynomial_expression += (
            c * cell_center_x**e_x * cell_center_y**e_y * cell_center_z**e_z
        )

    # Exploit any scalar variable in the model - here the "dummy_variable_x".
    variable_array = model.equation_system.get_variable_values(time_step_index=0)
    variable_x_index = model.equation_system.dofs_of(["dummy_variable_x"])
    variable_array[variable_x_index] = polynomial_expression

    # Compute the Lebesgue norm.
    metric_values_var = m_var(variable_array)
    l2_norm_numerical = metric_values_var["dummy_variable_x"]

    # Allow for small numerical errors due to numerical integration.
    assert np.isclose(l2_norm_numerical, l2_norm_analytical, rtol=1e-1), (
        """Numerical and analytical L2 norms do not match. """
        f"""Numerical: {l2_norm_numerical} """
        f"""Analytical: {l2_norm_analytical} """
    )


def test_equation_based_lebesgue_metric_with_model():
    """Compare analytical and numerical Lebesgue norms for random polynomial expressions
    via equations."""

    # Define symbols
    x, y, z = sp.symbols("x y z")

    # Define a random polynomial in x,y,z.
    coeffs = np.random.randint(-5, 6, size=10)
    exponents_x = np.random.randint(0, 4, size=10)
    exponents_y = np.random.randint(0, 4, size=10)
    exponents_z = np.random.randint(0, 4, size=10)
    expr = sum(
        c * x**e_x * y**e_y * z**e_z
        for c, e_x, e_y, e_z in zip(coeffs, exponents_x, exponents_y, exponents_z)
    )

    # Compute the analytical L2 norm over the unit cube.
    l2_norm_analytical = float(
        sp.sqrt(sp.integrate(expr**2, (x, 0, 1), (y, 0, 1), (z, 0, 1)))
    )

    # Evaluate the numerical norm using the EquationBasedLebesgueMetric.
    # Pass the coefficients and exponents to the model parameters
    # to define the polynomial expression in the equation.
    model = DummyModel(
        {
            "coeff": coeffs,
            "exp_x": exponents_x,
            "exp_y": exponents_y,
            "exp_z": exponents_z,
        }
    )
    m_eq = pp.EquationBasedLebesgueMetric(model)
    model.prepare_simulation()

    # Use cell centers and pass as values for the model variables.
    assert len(model.mdg.subdomains()) == 1
    cell_centers = model.mdg.subdomains()[0].cell_centers.ravel(order="F")
    model.equation_system.set_variable_values(cell_centers, iterate_index=0)

    # Compute the Lebesgue norm of the equation which corresponds to the
    # mass weighted polynomial expression, defined above.
    _, dummy_residual_array = model.equation_system.assemble()
    metric_values_eq = m_eq(dummy_residual_array)
    l2_norm_numerical = metric_values_eq["sd_eq"]

    # Allow for small numerical errors due to numerical integration.
    assert np.isclose(l2_norm_numerical, l2_norm_analytical, rtol=1e-1), (
        """Numerical and analytical L2 norms do not match. """
        f"""Numerical: {l2_norm_numerical} """
        f"""Analytical: {l2_norm_analytical} """
    )
