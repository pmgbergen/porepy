"""Unit tests for various metric implementations in PorePy.

Tested basic functionality and integration in models for the following metrics:
- EuclideanMetric
- VariableBasedEuclideanMetric
- VariableBasedLebesgueMetric
- EquationBasedEuclideanMetric
- EquationBasedLebesgueMetric

The tests combine simple unit tests and comparisons of norm computations for:
- one arrays and one functions over domains
- Gauss summation of squares for Euclidean metrics
- random polynomial expressions for L2 norms

"""

from typing import Literal

import numpy as np
import pytest
import sympy as sp
from deepdiff import DeepDiff

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)


@pytest.fixture(scope="module")
def orthogonal_2d_model() -> pp.PorePyModel:
    """Set up parameters for a unit square with two orthogonal fractures."""
    params = {
        "domain_size": 1.0,
        "fracture_indices": [0, 1],  # Use both orthogonal fractures.
        "cell_size": 0.5,
        "grid_type": "simplex",
        "material_constants": {
            "solid": pp.SolidConstants(residual_aperture=1),
        },
    }

    class Model(SquareDomainOrthogonalFractures, pp.Poromechanics):
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


@pytest.mark.parametrize(
    "assignment, expected_value",
    [
        (np.ones, lambda n: 1.0),
        (np.arange, lambda n: np.sqrt(n * (n - 1) * (2 * n - 1) / 6) / np.sqrt(n)),
    ],
)
def test_euclidean_metric_on_grids(orthogonal_2d_model, assignment, expected_value):
    """Test integration of EuclideanMetric in models with grids."""
    m = pp.EuclideanMetric()
    for g in orthogonal_2d_model.mdg.subdomains():
        arr = assignment(g.num_cells)
        assert np.isclose(m(arr), expected_value(g.num_cells))


@pytest.mark.parametrize(
    "assignment, expected_value",
    [
        (np.ones, lambda n: 1.0),
        (np.arange, lambda n: np.sqrt(n * (n - 1) * (2 * n - 1) / 6) / np.sqrt(n)),
    ],
)
def test_variable_based_euclidean_metric_on_grids(
    orthogonal_2d_model, assignment, expected_value
):
    """Test integration of VariableBasedEuclideanMetric in models with grids."""
    m = pp.VariableBasedEuclideanMetric(orthogonal_2d_model)

    # Fetch variable names and corresponding dofs for each variable block.
    variable_names = set(
        [v.name for v in orthogonal_2d_model.equation_system.variables]
    )
    variable_dofs = {name: [] for name in variable_names}
    for variable in orthogonal_2d_model.equation_system.variables:
        variable_dofs[variable.name].extend(
            orthogonal_2d_model.equation_system.dofs_of([variable])
        )

    # Create a dummy variable and assign each variable block with the values.
    dummy_variable = orthogonal_2d_model.equation_system.get_variable_values(
        time_step_index=0
    )
    for name in variable_names:
        dofs = variable_dofs[name]
        dummy_variable[dofs] = assignment(len(dofs))

    # Compute the corresponding Euclidean metric.
    metric_values = m(dummy_variable)

    # Check keys
    assert set(metric_values.keys()) == variable_names

    # Check values
    for key, value in metric_values.items():
        dofs = variable_dofs[key]
        assert np.isclose(value, expected_value(len(dofs)))


def test_variable_based_lebesgue_metric_on_grids(orthogonal_2d_model):
    """Test integration of VariableBasedLebesgueMetric in models with grids.

    Check that the integration of 1-s over the domain results in the expected L2 norm,
    which includes the volume of the domain and the dimensionality of the variable.

    """

    # Create a dummy variable array filled with ones.
    dummy_variable = orthogonal_2d_model.equation_system.get_variable_values(
        time_step_index=0
    )
    dummy_variable.fill(1.0)

    # Compute the corresponding Lebesgue metric.
    m = pp.VariableBasedLebesgueMetric(orthogonal_2d_model)
    metric_values = m(dummy_variable)

    # Manually compute expected values - L2 integral of 1 over the domain
    # (incl. dimensionality and sqrt).
    variables = orthogonal_2d_model.equation_system.variables
    result = {v.name: 0.0 for v in variables}
    for v in variables:
        domain = v.domain
        volume = domain.cell_volumes.sum()
        dimensionality = v._cells
        result[v.name] += volume * dimensionality
    for name in result:
        result[name] = np.sqrt(result[name])

    for name in result:
        assert np.isclose(result[name], metric_values[name])


@pytest.mark.parametrize(
    "assignment, expected_value",
    [
        (np.ones, lambda n: 1.0),
        (np.arange, lambda n: np.sqrt(n * (n - 1) * (2 * n - 1) / 6) / np.sqrt(n)),
    ],
)
def test_equation_based_euclidean_metric_on_grids(
    orthogonal_2d_model, assignment, expected_value
):
    """Test integration of EquationBasedEuclideanMetric in models with grids."""
    # Generate a dummy residual array filled with ones.
    # NOTE: Evaluate Jacobian to initialize the equation system properly.
    _, dummy_residual_array = orthogonal_2d_model.equation_system.assemble()

    # Define array and expected norm values.
    equations = orthogonal_2d_model.equation_system.equations
    result = {}
    for name in equations:
        if name not in orthogonal_2d_model.equation_system.assembled_equation_indices:
            continue
        dofs = orthogonal_2d_model.equation_system.assembled_equation_indices[name]
        if len(dofs) == 0:
            # Expect zero norm for empty equations
            result[name] = 0.0
            continue
        dummy_residual_array[dofs] = assignment(len(dofs))
        result[name] = expected_value(len(dofs))

    # Compute Lebesgue metric values.
    m = pp.EquationBasedEuclideanMetric(orthogonal_2d_model)
    metric_values = m(dummy_residual_array)

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


def test_equation_based_lebesgue_metric_on_grid(orthogonal_2d_model):
    """Test whether the integration of 1-s over the domain results in volume."""

    # Fetch the equations.
    equations = orthogonal_2d_model.equation_system.equations

    # Generate a dummy residual array filled with ones scaled with the cell volumes.
    # NOTE: Evaluate Jacobian to initialize the equation system properly.
    _, dummy_residual_array = orthogonal_2d_model.equation_system.assemble()
    dummy_residual_array.fill(1.0)

    # Scale with the right cell volumes.
    # Simultaneously compute the expected L2 norm of the 1 vector (incl dimensionality).
    result = {name: 0.0 for name in equations}
    for eqn in equations:
        domains = orthogonal_2d_model.equation_system.equation_image_space_composition[
            eqn
        ].keys()
        if len(domains) == 0:
            continue
        indices = orthogonal_2d_model.equation_system.assembled_equation_indices[eqn]
        cell_volumes = np.hstack([_sd.cell_volumes for _sd in domains])
        eq_dim = orthogonal_2d_model.equation_system.equation_image_size_info[eqn][
            "cells"
        ]
        dummy_residual_array[indices] *= np.repeat(cell_volumes, repeats=eq_dim)
        result[eqn] += sum(cell_volumes) * eq_dim

    # Take square root to get L2 norm.
    for name in result:
        result[name] = np.sqrt(result[name])

    # Compute Lebesgue metric values.
    m = pp.EquationBasedLebesgueMetric(orthogonal_2d_model)
    metric_values = m(dummy_residual_array)

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


class UnitSquare:
    """Model geometry for a unit square domain."""

    def set_domain(self) -> None:
        """Set domain."""
        self._domain = nd_cube_domain(2, 1.0)

    def meshing_arguments(self) -> dict[str, float]:
        """Set meshing arguments."""
        return {"cell_size": 0.5, "cell_size_boundary": 0.5}

    def grid_type(self) -> Literal["simplex"]:
        # Use a simplex grid to ensure we deal with non-trivial cell volumes and
        # coordinates.
        return "simplex"


class DummyVariables(pp.VariableMixin):
    """Define dummy variables for testing metrics."""

    def create_variables(self) -> None:
        """Create dummy variables associated with spatial coordinates."""
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

    def dummy_variable_x(self, subdomains):
        """Fetch dummy variable x."""
        return self.equation_system.md_variable("dummy_variable_x", subdomains)

    def dummy_variable_y(self, subdomains):
        """Fetch dummy variable y."""
        return self.equation_system.md_variable("dummy_variable_y", subdomains)


class DummyEquations(pp.PorePyModel):
    """Define dummy equations for testing metrics."""

    def sd_eq(self, subdomains: pp.GridLikeSequence) -> pp.ad.Operator:
        """Polynom in the variables, integrated over the domain with mass weighting."""

        # Treat variables as spatial coordinates.
        variable_x = self.dummy_variable_x(subdomains)
        variable_y = self.dummy_variable_y(subdomains)
        # Define a polynomial expression in the variables, with coefficients and
        # exponents obtained from model parameters.
        coeff = self.params.get("coeff", [0])
        exp_x = self.params.get("exp_x", [0])
        exp_y = self.params.get("exp_y", [0])
        polynomial_expression = pp.ad.sum_operator_list(
            [
                pp.ad.Scalar(c)
                * variable_x ** pp.ad.Scalar(e_x)
                * variable_y ** pp.ad.Scalar(e_y)
                for c, e_x, e_y in zip(coeff, exp_x, exp_y)
            ]
        )
        # Compute mass weighted integral of the polynomial expression
        # to mimick a typical equation residual.
        mass_weighted_expression = self.volume_integral(
            polynomial_expression, subdomains, 1
        )
        mass_weighted_expression.set_name("sd_eq")
        return mass_weighted_expression

    def set_equations(self):
        """Set dummy equations based on the polynomial expression."""
        subdomains = self.mdg.subdomains()
        sd_eq = self.sd_eq(subdomains)
        self.equation_system.set_equation(sd_eq, subdomains, {"cells": 1})


class SimpleVolumeIntegralMixin(pp.models.constitutive_laws.DimensionReduction):
    """Fetch only volume integral from BalanceEquation."""

    def volume_integral(
        self,
        integrand: pp.ad.Operator,
        grids: pp.GridLikeSequence,
        dim: int,
    ) -> pp.ad.Operator:
        """Fetch only volume integral from BalanceEquation for simple cases."""
        return pp.BalanceEquation.volume_integral(self, integrand, grids, dim)


class DummyModel(  # type: ignore[misc]
    UnitSquare,
    DummyVariables,
    DummyEquations,
    SimpleVolumeIntegralMixin,
    pp.SolutionStrategy,
    pp.FluidMixin,
    pp.ModelGeometry,
    pp.BoundaryConditionMixin,
    pp.InitialConditionMixin,
    pp.DataSavingMixin,
):
    """A dummy model combining necessary components to test metrics with random
    polynomial expressions."""


@pytest.fixture
def random_polynomial_setup():
    """Fixture providing a random polynomial expression and its analytical L2 norm."""
    # Define symbols
    x, y = sp.symbols("x y")

    # Define a random polynomial in x,y,z.
    np.random.seed(42)
    coeffs = np.random.randint(-5, 6, size=10)
    exponents_x = np.random.randint(0, 4, size=10)
    exponents_y = np.random.randint(0, 4, size=10)
    expr = sum(
        c * x**e_x * y**e_y for c, e_x, e_y in zip(coeffs, exponents_x, exponents_y)
    )

    # Compute the analytical L2 norm over the unit cube.
    l2_norm_analytical = float(sp.sqrt(sp.integrate(expr**2, (x, 0, 1), (y, 0, 1))))

    return {
        "coeffs": coeffs,
        "exponents_x": exponents_x,
        "exponents_y": exponents_y,
        "l2_norm_analytical": l2_norm_analytical,
    }


def test_variable_based_lebesgue_metric_with_model(random_polynomial_setup):
    """Test integral of a random polynomial expression via variables."""
    # Evaluate the numerical norm using the VariableBasedLebesgueMetric.
    model = DummyModel()
    m_var = pp.VariableBasedEuclideanMetric(model)
    model.prepare_simulation()

    # Use cell centers to define the polynomial expression.
    assert len(model.mdg.subdomains()) == 1
    cell_center_x = model.mdg.subdomains()[0].cell_centers[0, :]
    cell_center_y = model.mdg.subdomains()[0].cell_centers[1, :]
    polynomial_expression = np.zeros_like(cell_center_x)
    for c, e_x, e_y in zip(
        random_polynomial_setup["coeffs"],
        random_polynomial_setup["exponents_x"],
        random_polynomial_setup["exponents_y"],
    ):
        polynomial_expression += c * cell_center_x**e_x * cell_center_y**e_y

    # Exploit any scalar variable in the model - here the "dummy_variable_x".
    variable_array = model.equation_system.get_variable_values(time_step_index=0)
    variable_x_index = model.equation_system.dofs_of(["dummy_variable_x"])
    variable_array[variable_x_index] = polynomial_expression

    # Compute the Lebesgue norm.
    metric_values_var = m_var(variable_array)
    l2_norm_numerical = metric_values_var["dummy_variable_x"]

    # Allow for small numerical errors due to numerical integration.
    assert np.isclose(
        l2_norm_numerical, random_polynomial_setup["l2_norm_analytical"], rtol=1e-1
    ), (
        """Numerical and analytical L2 norms do not match. """
        f"""Numerical: {l2_norm_numerical} """
        f"""Analytical: {random_polynomial_setup["l2_norm_analytical"]} """
    )


def test_equation_based_lebesgue_metric_with_model(random_polynomial_setup):
    """Test integral of a random polynomial expression via equations."""
    # Evaluate the numerical norm using the EquationBasedLebesgueMetric.
    model = DummyModel(
        {
            "coeff": random_polynomial_setup["coeffs"],
            "exp_x": random_polynomial_setup["exponents_x"],
            "exp_y": random_polynomial_setup["exponents_y"],
        }
    )
    m_eq = pp.EquationBasedLebesgueMetric(model)
    model.prepare_simulation()

    # Use cell centers and pass as values for the model variables.
    assert len(model.mdg.subdomains()) == 1
    cell_centers = model.mdg.subdomains()[0].cell_centers[:2].ravel(order="C")
    model.equation_system.set_variable_values(cell_centers, iterate_index=0)

    # Compute the Lebesgue norm of the equation which corresponds to the
    # mass weighted polynomial expression, defined above.
    _, dummy_residual_array = model.equation_system.assemble()
    metric_values_eq = m_eq(dummy_residual_array)
    l2_norm_numerical = metric_values_eq["sd_eq"]

    # Allow for small numerical errors due to numerical integration.
    assert np.isclose(
        l2_norm_numerical, random_polynomial_setup["l2_norm_analytical"], rtol=1e-1
    ), (
        """Numerical and analytical L2 norms do not match. """
        f"""Numerical: {l2_norm_numerical} """
        f"""Analytical: {random_polynomial_setup["l2_norm_analytical"]} """
    )
