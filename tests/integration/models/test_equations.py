"""Tests 'final' equations employed in the models.

The tests here considers equations that are present in the global system to be solved,
as opposed to simpler relations (typically constitutive relations) that are used to
construct these global equations.

See also: tests.integration.models.test_constitutive_laws.

"""
from inspect import signature

import pytest

from .setup_utils import model


@pytest.mark.parametrize(
    "model_type,equation_name,domain_dimension",
    [
        ("mass_balance", "mass_balance_equation", []),
        ("mass_balance", "interface_darcy_flux_equation", []),
        ("momentum_balance", "momentum_balance_equation", [2]),
        ("momentum_balance", "interface_force_balance_equation", [1]),
        ("momentum_balance", "normal_fracture_deformation_equation", [1]),
        ("momentum_balance", "tangential_fracture_deformation_equation", [1]),
        ("energy_balance", "energy_balance_equation", []),
        ("energy_balance", "interface_enthalpy_flux_equation", []),
        ("energy_balance", "interface_fourier_flux_equation", []),
        # Energy balance inherits mass balance equations. Test one of these as well.
        ("energy_balance", "mass_balance_equation", []),
        ("poromechanics", "mass_balance_equation", []),
        ("poromechanics", "momentum_balance_equation", [0]),
        ("poromechanics", "interface_force_balance_equation", [0]),
    ],
)
@pytest.mark.parametrize("num_fracs", [0, 1, 2])
def test_parse_equations(model_type, equation_name, domain_dimension, num_fracs):
    """Test that equation parsing works as expected.

    Currently tested are the relevant equations in the mass balance and momentum balance
    models. To add a new model, add a new class in setup_utils.py and expand the
    test parameterization accordingly.

    Parameters:
        model_type: Type of model to test. Currently supported are "mass_balance" and
            "momentum_balance". To add a new model, add a new class in setup_utils.py.
        equation_name (str): Name of the method to test.
        domain_inds (list of int): Indices of the domains for which the method
            should be called. Some methods are only defined for a subset of
            domains.

    """
    if len(domain_dimension) > 0:
        if domain_dimension[0] == 2 and num_fracs > 0:
            return
        elif domain_dimension[0] == 1 and num_fracs == 0:
            return

    # Set up an object of the prescribed model
    setup = model(model_type, num_fracs=num_fracs)
    # Fetch the relevant equation of this model.
    method = getattr(setup, equation_name)
    # Fetch the signature of the method.
    sig = signature(method)

    # The assumption is that the method to be tested takes as input only its domain
    # of definition, that is a list of subdomains or interfaces. The test framework is
    # not compatible with methods that take other arguments (and such a method would also
    # break the implicit contract of the constitutive laws).
    assert len(sig.parameters) == 1

    # The domain is a list of either subdomains or interfaces.
    if "subdomains" in sig.parameters:
        # If relevant, filter out the domains that are not to be tested.
        if len(domain_dimension) > 0:
            domains = setup.mdg.subdomains(dim=domain_dimension[0])
        else:
            domains = setup.mdg.subdomains()
    elif "interfaces" in sig.parameters:
        if len(domain_dimension) > 0:
            domains = setup.mdg.interfaces(dim=domain_dimension[0])
        else:
            domains = setup.mdg.interfaces()
    # Call the discretization method.
    setup.equation_system.discretize()
    print(setup.mdg)

    # Assemble the matrix and right hand side for the given equation. An error here will
    # indicate that something is wrong with the way the conservation law combines
    # terms and factors (e.g., grids, parameters, variables, other methods etc.) to form
    # an Ad operator object.
    setup.equation_system.assemble_subsystem({equation_name: domains})


test_parse_equations(
    model_type="mass_balance",
    equation_name="mass_balance_equation",
    domain_dimension=[],
    num_fracs=0,
)
