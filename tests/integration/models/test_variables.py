"""Tests for model variables.

"""
from __future__ import annotations

from inspect import signature

import pytest

import porepy as pp

from .setup_utils import model


@pytest.mark.parametrize(
    "model_type,variable_name,domain_inds",
    [
        ("momentum_balance", "displacement", [0]),
        ("momentum_balance", "interface_displacement", []),
        ("momentum_balance", "contact_traction", [1]),
        ("mass_balance", "pressure", [0]),
        ("mass_balance", "pressure", []),
        ("mass_balance", "interface_darcy_flux", []),
        ("energy_balance", "temperature", []),
        ("energy_balance", "interface_enthalpy_flux", []),
        ("energy_balance", "interface_fourier_flux", []),
        ("poromechanics", "displacement", [0]),
        ("poromechanics", "interface_displacement", []),
        ("poromechanics", "contact_traction", [1]),
        ("poromechanics", "pressure", [0]),
        ("thermoporomechanics", "displacement", [0]),
        ("thermoporomechanics", "interface_displacement", []),
        ("thermoporomechanics", "temperature", []),
        ("thermoporomechanics", "contact_traction", [1]),
        ("thermoporomechanics", "pressure", [0]),
        ("thermoporomechanics", "interface_enthalpy_flux", []),
    ],
)
def test_parse_variables(model_type, variable_name, domain_inds):
    """Test that the ad parsing works as expected."""
    # Set up an object of the prescribed model
    # Use dimension two and one fracture.
    setup = model(model_type, dim=2, num_fracs=1)
    # Fetch the relevant method to get the variable from this model.
    variable = getattr(setup, variable_name)
    # Fetch the signature of the method.
    sig = signature(variable)

    # The assumption is that the method to be tested takes as input only its domain
    # of definition, that is a list of subdomains or interfaces. The test framework is
    # not compatible with methods that take other arguments (and such a method would also
    # break the implicit contract of the constitutive laws).
    assert len(sig.parameters) == 1

    # The domain is a list of either subdomains or interfaces.
    if "subdomains" in sig.parameters:
        domains = setup.mdg.subdomains()
    elif "interfaces" in sig.parameters:
        domains = setup.mdg.interfaces()
    # If relevant, filter out the domains that are not to be tested.
    if len(domain_inds) > 0:
        domains = [domains[i] for i in domain_inds]

    # Pick out the relevant domains
    if len(domain_inds) > 0:
        domains = [domains[i] for i in domain_inds if i < len(domains)]
    op = variable(domains)

    assert isinstance(op, pp.ad.Operator)

    # The operator should be evaluateable
    evaluation = op.evaluate(setup.equation_system)
    # Check that value and Jacobian are of the correct shape
    sz_tot = setup.equation_system.num_dofs()
    sz_var = setup.equation_system.dofs_of(op.sub_vars).size
    assert evaluation.val.size == sz_var
    assert evaluation.jac.shape == (sz_var, sz_tot)

    scalar = pp.ad.Scalar(1)
    op_2 = scalar * op
    op_2.evaluate(setup.equation_system)
