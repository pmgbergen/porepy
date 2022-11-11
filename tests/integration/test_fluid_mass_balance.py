"""Tests for fluid mass balance models."""

from inspect import signature

import numpy as np
import pytest

import porepy as pp
from porepy.models import fluid_mass_balance as fmb


class FracGeom(pp.ModelGeometry):
    def set_fracture_network(self) -> None:
        p = np.array([[0, 1], [0.5, 0.5]])
        e = np.array([[0], [1]])
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        self.fracture_network = pp.FractureNetwork2d(p, e, domain)

    def mesh_arguments(self) -> dict:
        return {"mesh_size_frac": 0.3, "mesh_size_bound": 0.3}


class IncompressibleCombined(
    FracGeom,
    fmb.MassBalanceEquations,
    fmb.ConstitutiveEquationsIncompressibleFlow,
    fmb.VariablesSinglePhaseFlow,
    fmb.SolutionStrategyIncompressibleFlow,
    pp.DataSavingMixin,
):
    """Demonstration of how to combine in a class which can be used with
    pp.run_stationary_problem (once cleanup has been done).
    """

    pass


@pytest.fixture
def setup():

    ob = IncompressibleCombined({})
    ob.prepare_simulation()
    return ob


@pytest.mark.parametrize(
    "method_name",
    [
        "bc_values_darcy_flux",
        "bc_values_mobrho",
        "viscosity",
        "fluid_source",
        "mobility",
        "fluid_density",
        "aperture",
        "specific_volume",
        "darcy_flux",
        "interface_fluid_flux",
        "fluid_flux",
        "pressure_trace",
        "porosity",
        "reference_pressure",
    ],
)
def test_ad_parsing_constitutive_laws(setup, method_name):
    """Test that the ad parsing works as expected."""
    method = getattr(setup, method_name)
    sig = signature(method)
    assert len(sig.parameters) == 1
    if "subdomains" in sig.parameters:
        op = method(subdomains=setup.mdg.subdomains())
    elif "interfaces" in sig.parameters:
        op = method(interfaces=setup.mdg.interfaces())

    assert isinstance(op, pp.ad.Operator)
    op.evaluate(setup.equation_system)


def test_special_signatures(setup) -> None:
    """Test that the ad parsing works as expected.

    This test is for methods with special signatures:
    - vector_source
    """
    op0 = setup.vector_source(grids=setup.mdg.subdomains(), material="fluid")
    op1 = setup.vector_source(grids=setup.mdg.interfaces(), material="fluid")
    op0.evaluate(setup.equation_system)
    op1.evaluate(setup.equation_system)


@pytest.mark.parametrize(
    "variable_name",
    [
        "pressure",
        "interface_darcy_flux",
    ],
)
@pytest.mark.parametrize(
    "variable_inds",
    [
        [0],
        [0, 1],
    ],
)
def test_ad_parsing_variables(setup, variable_name, variable_inds):
    """Test that the ad parsing works as expected."""
    variable = getattr(setup, variable_name)
    sig = signature(variable)
    assert len(sig.parameters) == 1
    if "subdomains" in sig.parameters:
        domains = setup.mdg.subdomains()
    elif "interfaces" in sig.parameters:
        domains = setup.mdg.interfaces()

    # Pick out the relevant domains
    domains = [domains[i] for i in variable_inds if i < len(domains)]
    op = variable(domains)

    assert isinstance(op, pp.ad.Operator)
    # The operator should be evaluateable
    evaluation = op.evaluate(setup.equation_system)
    # Check that value and Jacobian are of the correct shape
    sz_tot = setup.equation_system.num_dofs()
    sz_var = sum([d.num_cells for d in domains])
    assert evaluation.val.size == sz_var
    assert evaluation.jac.shape == (sz_var, sz_tot)


@pytest.mark.parametrize(
    "equation_name",
    [
        "fluid_mass_balance_equation",
        "interface_darcy_flux_equation",
    ],
)
@pytest.mark.parametrize(
    "domain_inds",
    [
        [0],
        [0, 1],
    ],
)
def test_parse_equations(setup, equation_name, domain_inds):
    """Test that equation parsing works as expected."""
    if "balance" in equation_name:
        domains = setup.mdg.subdomains()
    elif "flux" in equation_name:
        domains = setup.mdg.interfaces()
    else:
        raise ValueError("Unknown equation type")

    # Pick out the relevant domains
    domains = [domains[i] for i in domain_inds if i < len(domains)]
    setup.equation_system.assemble_subsystem({equation_name: domains})
