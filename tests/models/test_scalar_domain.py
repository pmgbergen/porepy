"""Integration tests verifying that material-property Scalars carry grid domain info.

After implementing Scalar domain support, methods such as `solid_density(subdomains)`
should return a Scalar whose `operator_domain` reflects the provided grids.
"""

from __future__ import annotations

import pytest

import porepy as pp
from porepy.applications.test_utils import models as test_models
from porepy.numerics.ad.operators import DomainType, GridEntity, OperatorSpace, Scalar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mass_balance_model():
    """A prepared 2-D mass balance model for use across tests in this module."""
    return test_models.model("mass_balance", dim=2, num_fracs=1)


@pytest.fixture(scope="module")
def momentum_balance_model():
    """A prepared 2-D momentum balance model."""
    return test_models.model("momentum_balance", dim=2, num_fracs=1)


@pytest.fixture(scope="module")
def poromechanics_model():
    """A prepared 2-D poromechanics model."""
    return test_models.model("poromechanics", dim=2, num_fracs=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_scalar_has_subdomain_domain(
    scalar: pp.ad.Operator, expected_grids: list
) -> None:
    """Assert that *scalar* carries subdomain domain info matching *expected_grids*."""
    assert isinstance(scalar, Scalar)
    dom = scalar.operator_domain
    assert dom is not None, "Scalar should have a non-None domain"
    assert dom.domain_type == DomainType.subdomains
    assert set(dom.grids) == set(expected_grids)
    assert dom.dof_info == {
        GridEntity.cells: 1
    }, "Domain-bearing Scalar should be cell-wise"
    assert scalar.operator_domain == scalar.operator_range


def _assert_scalar_has_interface_domain(
    scalar: pp.ad.Operator, expected_grids: list
) -> None:
    """Assert that *scalar* carries interface domain info matching *expected_grids*."""
    assert isinstance(scalar, Scalar)
    dom = scalar.operator_domain
    assert dom is not None
    assert dom.domain_type == DomainType.interfaces
    assert set(dom.grids) == set(expected_grids)
    assert dom.dof_info == {GridEntity.cells: 1}
    assert scalar.operator_domain == scalar.operator_range


# ---------------------------------------------------------------------------
# Tests: material-property Scalars from constitutive_laws
# ---------------------------------------------------------------------------


class TestSolidPropertyScalarDomains:
    """Solid material property methods should return domain-bearing Scalars."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "solid_density",
            "shear_modulus",
            "lame_lambda",
            "youngs_modulus",
            "bulk_modulus",
        ],
    )
    def test_elastic_constant_has_subdomain_domain(
        self, method_name, momentum_balance_model
    ):
        m = momentum_balance_model
        subdomains = m.mdg.subdomains()
        method = getattr(m, method_name)
        scalar = method(subdomains)
        _assert_scalar_has_subdomain_domain(scalar, subdomains)

    def test_biot_coefficient_has_subdomain_domain(self, poromechanics_model):
        m = poromechanics_model
        subdomains = m.mdg.subdomains()
        scalar = m.biot_coefficient(subdomains)
        _assert_scalar_has_subdomain_domain(scalar, subdomains)

    def test_porosity_has_subdomain_domain(self, mass_balance_model):
        m = mass_balance_model
        subdomains = m.mdg.subdomains()
        scalar = m.porosity(subdomains)
        _assert_scalar_has_subdomain_domain(scalar, subdomains)

    def test_normal_permeability_has_interface_domain(self, mass_balance_model):
        m = mass_balance_model
        interfaces = m.mdg.interfaces()
        if not interfaces:
            pytest.skip("No interfaces in this model setup")
        scalar = m.normal_permeability(interfaces)
        _assert_scalar_has_interface_domain(scalar, interfaces)


class TestFluidPropertyScalarDomains:
    """Fluid material property methods should return domain-bearing Scalars."""

    def test_fluid_compressibility_has_subdomain_domain(self, mass_balance_model):
        m = mass_balance_model
        subdomains = m.mdg.subdomains()
        scalar = m.fluid_compressibility(subdomains)
        _assert_scalar_has_subdomain_domain(scalar, subdomains)


# ---------------------------------------------------------------------------
# Tests: scalar remains compatible with arithmetic in equations
# ---------------------------------------------------------------------------


class TestScalarDomainArithmetic:
    """Domain-bearing Scalars must remain compatible in arithmetic with operators
    defined on the same grids and compose to unclear domains on mismatched grids."""

    def test_scalar_times_scalar_result_carries_domain(self, mass_balance_model):
        m = mass_balance_model
        subdomains = m.mdg.subdomains()
        # porosity returns a Scalar with subdomain domain; multiply by a plain Scalar
        phi = m.porosity(subdomains)
        s = Scalar(2.0, domains=subdomains)
        result = s * phi
        assert result.operator_domain is not None
        assert result.operator_domain.domain_type == DomainType.subdomains

    def test_plain_scalar_times_domain_scalar_inherits_domain(
        self, mass_balance_model
    ):
        m = mass_balance_model
        subdomains = m.mdg.subdomains()
        phi = m.porosity(subdomains)
        s = Scalar(2.0)  # plain scalar (no domain)
        result = s * phi
        assert result.operator_domain is not None
