"""Tests for model variables.

"""
from inspect import signature

import pytest

from .setup_utils import model


@pytest.mark.parametrize(
    "model_type,equation_name,domain_inds",
    [
        ("mass_balance", "mass_balance_equation", []),
        ("mass_balance", "interface_darcy_flux_equation", []),
        ("force_balance", "matrix_force_balance_equation", [0]),
        ("force_balance", "interface_force_balance_equation", [0]),
        ("force_balance", "normal_fracture_deformation_equation", [1]),
        ("force_balance", "tangential_fracture_deformation_equation", [1]),
    ],
)
def test_parse_equations(model_type, equation_name, domain_inds):
    """Test that equation parsing works as expected."""
    setup = model(model_type)
    # TODO: Add EquationManager method for retrieving equation domains
    method = getattr(setup, equation_name)
    sig = signature(method)
    assert len(sig.parameters) == 1
    if "subdomains" in sig.parameters:
        domains = setup.mdg.subdomains()
    elif "interfaces" in sig.parameters:
        domains = setup.mdg.interfaces()

    # Pick out the relevant domains
    if len(domain_inds) > 0:
        domains = [domains[i] for i in domain_inds if i < len(domains)]
    setup.equation_system.assemble_subsystem({equation_name: domains})
