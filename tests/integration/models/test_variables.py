"""Tests for model variables.

"""
from inspect import signature

import numpy as np
import pytest

import porepy as pp

from .setup_utils import model


@pytest.mark.parametrize(
    "model_type,variable_name,domain_inds",
    [
        ("force_balance", "displacement", [0]),
        ("force_balance", "interface_displacement", []),
        ("force_balance", "contact_traction", [1]),
        ("mass_balance", "pressure", [0]),
        ("mass_balance", "pressure", []),
        ("mass_balance", "interface_darcy_flux", []),
    ],
)
def test_parse_variables(model_type, variable_name, domain_inds):
    """Test that the ad parsing works as expected."""
    setup = model(model_type)
    variable = getattr(setup, variable_name)
    sig = signature(variable)
    assert len(sig.parameters) == 1
    if "subdomains" in sig.parameters:
        # Fracture contact traction
        domains = setup.mdg.subdomains()
    elif "interfaces" in sig.parameters:
        # Displacement in matrix and matrix-fracture interfaces
        domains = setup.mdg.interfaces()
    else:
        raise ValueError

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
