"""Tests for constitutive laws in the models.

The module contains a setup for testing composition and evaluation (in the Ad sense) of
constitutive laws for models. The setup is applied to the constitutive laws in the
models subpackage.

In addition to checking that evaluation is possible, the tests also check that the
compound classes contain the specified constitutive laws, thus revealing mistakes if
the classes are refactored.

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
        ("mass_balance", "fluid_viscosity", []),
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
        # Momentum balance
        ("momentum_balance", "bc_values_mechanics", []),
        # The body force and stress are only meaningful in the top dimension
        ("momentum_balance", "body_force", [0]),
        ("momentum_balance", "stress", [0]),
        ("momentum_balance", "solid_density", []),
        ("momentum_balance", "lame_lambda", []),
        ("momentum_balance", "shear_modulus", []),
        ("momentum_balance", "youngs_modulus", []),
        ("momentum_balance", "bulk_modulus", []),
        # Reference gap, friction coefficient and friction bound are only meaningful for
        # a fracture of co-dimension 1.
        ("momentum_balance", "reference_gap", [1]),
        ("momentum_balance", "friction_coefficient", [1]),
        ("momentum_balance", "friction_bound", [1]),
        # Energy balance
        ("energy_balance", "bc_values_enthalpy_flux", []),
        ("energy_balance", "bc_values_fourier_flux", []),
        ("energy_balance", "energy_source", []),
        ("energy_balance", "enthalpy_flux", []),
        ("energy_balance", "fourier_flux", []),
        ("energy_balance", "interface_enthalpy_flux", []),
        ("energy_balance", "interface_fourier_flux", []),
        ("energy_balance", "reference_temperature", []),
        ("energy_balance", "normal_thermal_conductivity", []),
        ("energy_balance", "aperture", []),
        ("energy_balance", "specific_volume", []),
        # Poromechanics
        ("poromechanics", "biot_alpha", [0]),
    ],
)
def test_parse_constitutive_laws(model_type, method_name, domain_inds):
    """Test that the ad parsing of constitutive laws works as intended.

    The workflow in the test is as follows: An object of the prescribed model is
    created, and the specified method is called. The three most likely reason a test may
    fail are:
        1. The method to be tested combines its components (grids, variables etc.) in a
           wrong way, causing an error in the call to the method.
        2. Something goes wrong in discretization.
        3. Something goes wroing in parsing of the Ad operator tree, perhaps due to
           combining matrices and arrays of incompatible dimensions.

    Parameters:
        model_type: Type of model to test. Currently supported are "mass_balance" and
            "momentum_balance". To add a new model, add a new class in setup_utils.py.
        method_name (str): Name of the method to test.
        domain_inds (list of int): Indices of the domains for which the method
            should be called. Some methods are only defined for a subset of
            domains.

    """
    # Set up an object of the prescribed model
    setup = model(model_type)
    # Fetch the relevant method of this model.
    method = getattr(setup, method_name)
    # Fetch the signature of the method.
    sig = signature(method)

    # The assumption is that the method to be tested takes as input only its domain of
    # definition, that is a list of subdomains or interfaces. The test framework is not
    # compatible with methods that take other arguments (and such a method would also
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

    # Call the method with the domain as argument. An error here will indicate that
    # something is wrong with the way the method combines terms and factors (e.g., grids
    # parameters, variables, other methods) to form an Ad operator object.
    op = method(domains)
    # The method should return an Ad object.
    assert isinstance(op, pp.ad.Operator)

    # Carry out discretization.
    op.discretize(setup.mdg)
    # Evaluate the discretized operator. An error here would typically be caused by the
    # method combining terms and factors of the wrong size etc. This could be a problem
    # with the constitutive law, or it could signify that something has changed in the
    # Ad machinery which makes the evaluation of the operator fail.
    op.evaluate(setup.equation_system)
