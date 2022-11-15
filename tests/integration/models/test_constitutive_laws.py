"""Tests for constitutive laws.

"""
from inspect import signature

import pytest

import porepy as pp

from .setup_utils import model


@pytest.mark.parametrize(
    "model_type,method_name,domain_inds",
    [  # Fluid mass balance
        ("mass_balance", "bc_values_darcy_flux", []),
        ("mass_balance", "bc_values_mobrho", []),
        ("mass_balance", "viscosity", []),
        ("mass_balance", "fluid_source", []),
        ("mass_balance", "mobility", []),
        ("mass_balance", "fluid_density", []),
        ("mass_balance", "aperture", []),
        ("mass_balance", "specific_volume", []),
        ("mass_balance", "darcy_flux", []),
        ("mass_balance", "interface_fluid_flux", []),
        ("mass_balance", "fluid_flux", []),
        ("mass_balance", "pressure_trace", []),
        ("mass_balance", "porosity", []),
        ("mass_balance", "reference_pressure", []),
        # Force balance
        ("force_balance", "bc_values_mechanics", []),
        ("force_balance", "body_force", [0]),
        ("force_balance", "stress", [0]),
        ("force_balance", "solid_density", []),
        ("force_balance", "lame_lambda", []),
        ("force_balance", "shear_modulus", []),
        ("force_balance", "youngs_modulus", []),
        ("force_balance", "bulk_modulus", []),
        ("force_balance", "reference_gap", [1]),
        ("force_balance", "friction_coefficient", [1]),
        ("force_balance", "friction_bound", [1]),
    ],
)
def test_parse_constitutive_laws(model_type, method_name, domain_inds):
    """Test that the ad parsing works as expected.


    Parameters:
        method_name (str): Name of the method to test.
        domain_inds (list of int): Indices of the domains for which the method
            should be called. Some methods are only defined for a subset of
            domains.

    """
    setup = model(model_type)
    method = getattr(setup, method_name)
    sig = signature(method)
    assert len(sig.parameters) == 1
    if "subdomains" in sig.parameters:
        domains = setup.mdg.subdomains()
    elif "interfaces" in sig.parameters:
        domains = setup.mdg.interfaces()
    if len(domain_inds) > 0:
        domains = [domains[i] for i in domain_inds]
    op = method(domains)
    assert isinstance(op, pp.ad.Operator)
    op.evaluate(setup.equation_system)
