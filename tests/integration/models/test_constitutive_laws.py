"""Tests for constitutive laws in the models.

The module contains a setup for testing composition and evaluation (in the Ad sense) of
constitutive laws for models. The setup is applied to the constitutive laws in the
models subpackage.

In addition to checking that evaluation is possible, the tests also check that the
compound classes contain the specified constitutive laws, thus revealing mistakes if
the classes are refactored.

"""
import numpy as np
import pytest

import porepy as pp

from . import setup_utils


@pytest.mark.parametrize(
    "model_type,method_name,domain_dimension",
    [  # Fluid mass balance
        ("mass_balance", "bc_values_darcy_flux", None),
        ("mass_balance", "bc_values_mobrho", None),
        ("mass_balance", "fluid_viscosity", None),
        ("mass_balance", "fluid_source", None),
        ("mass_balance", "mobility", None),
        ("mass_balance", "fluid_density", None),
        ("mass_balance", "aperture", None),
        ("mass_balance", "specific_volume", None),
        ("mass_balance", "darcy_flux", None),
        ("mass_balance", "interface_fluid_flux", None),
        ("mass_balance", "fluid_flux", None),
        ("mass_balance", "pressure_trace", None),
        ("mass_balance", "porosity", None),
        ("mass_balance", "reference_pressure", None),
        # Momentum balance
        ("momentum_balance", "bc_values_mechanics", None),
        # The body force and stress are only meaningful in the top dimension
        ("momentum_balance", "body_force", 2),
        ("momentum_balance", "stress", 2),
        ("momentum_balance", "solid_density", None),
        ("momentum_balance", "lame_lambda", None),
        ("momentum_balance", "shear_modulus", None),
        ("momentum_balance", "youngs_modulus", None),
        ("momentum_balance", "bulk_modulus", None),
        # Reference gap, friction coefficient and friction bound are only meaningful for
        # a fracture of co-dimension 1.
        ("momentum_balance", "reference_gap", 1),
        ("momentum_balance", "friction_coefficient", 1),
        ("momentum_balance", "friction_bound", 1),
        # Energy balance
        ("energy_balance", "bc_values_enthalpy_flux", None),
        ("energy_balance", "bc_values_fourier_flux", None),
        ("energy_balance", "energy_source", None),
        ("energy_balance", "enthalpy_flux", None),
        ("energy_balance", "fourier_flux", None),
        ("energy_balance", "interface_enthalpy_flux", None),
        ("energy_balance", "interface_fourier_flux", None),
        ("energy_balance", "reference_temperature", None),
        ("energy_balance", "normal_thermal_conductivity", None),
        ("energy_balance", "aperture", None),
        ("energy_balance", "specific_volume", None),
        # Poromechanics
        ("poromechanics", "reference_porosity", None),
        ("poromechanics", "biot_coefficient", 2),
        ("poromechanics", "matrix_porosity", 2),
        ("poromechanics", "porosity", 1),
        ("poromechanics", "porosity", None),
        ("poromechanics", "pressure_stress", 2),
        ("poromechanics", "stress", 2),
        ("poromechanics", "specific_volume", None),
        ("poromechanics", "aperture", None),
        ("poromechanics", "fracture_pressure_stress", 1),
    ],
)
@pytest.mark.parametrize("num_fracs", [0, 1, 2])
def test_parse_constitutive_laws(model_type, method_name, domain_dimension, num_fracs):
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
    if domain_dimension is not None:
        if domain_dimension == 2 and num_fracs > 0:
            return
        elif domain_dimension == 1 and num_fracs == 0:
            return

    # Set up an object of the prescribed model
    setup = setup_utils.model(model_type, num_fracs=num_fracs)
    # Fetch the relevant method of this model and extract the domains for which it is
    # defined.
    method = getattr(setup, method_name)
    domains = setup_utils.domains_from_method_name(setup.mdg, method, domain_dimension)

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


@pytest.mark.parametrize(
    "method_name, domain_dimension, expected",
    [
        # Viscosity is constant by default, so test should be trivial.
        ("fluid_viscosity", None, setup_utils.water_values["viscosity"]),
    ],
)
def test_return_values(method_name, domain_dimension, expected):
    """Test that the value of the parsed operator is as expected."""
    # The thermoporoelastic model covers most constitutive laws, so we use it for the
    # test.
    # Assign non-trivial values to the parameters to avoid masking errors. TODO: Same
    # for variables?
    fluid = pp.FluidConstants(setup_utils.water_values)
    solid = pp.SolidConstants(setup_utils.granite_values)
    params = {"material_constants": {"fluid": fluid, "solid": solid}}
    setup = pp.thermoporomechanics.Thermoporomechanics(params)
    setup.prepare_simulation()

    # Fetch the relevant method for this test and the relevant domains.
    method = getattr(setup, method_name)
    domains = setup_utils.domains_from_method_name(setup.mdg, method, domain_dimension)

    # Call the method with the domain as argument.
    op = method(domains)
    # Evaluate the operator.
    value = op.evaluate(setup.equation_system)
    if isinstance(value, pp.ad.Ad_array):
        value = value.val
    assert np.allclose(value, expected)
