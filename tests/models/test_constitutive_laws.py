"""Tests for constitutive laws in the models.

The first function is for testing composition and evaluation (in the Ad sense) of
constitutive laws for models. The setup is applied to the constitutive laws in the
models subpackage.

In addition to checking that evaluation is possible, the tests also check that the
compound classes contain the specified constitutive laws, thus revealing mistakes if the
classes are refactored.

The first test does not test that the constitutive laws evaluates to reasonable numbers.
That is done in the second and third tests. TODO: If those are extended to cover more
constitutive laws, the first test might be removed.


"""
from __future__ import annotations

from typing import Any, Literal, Type

import numpy as np
import pytest

import porepy as pp
import porepy.models.constitutive_laws as c_l
from porepy.applications.test_utils import models

solid_values = pp.solid_values.granite
solid_values.update(
    {
        "fracture_normal_stiffness": 1529,
        "maximum_fracture_closure": 1e-4,
        "fracture_gap": 1e-4,
        "residual_aperture": 0.01,
    }
)


@pytest.mark.parametrize(
    "model_type,method_name,only_codimension",
    [  # Fluid mass balance
        ("mass_balance", "mobility_rho", None),
        ("mass_balance", "fluid_viscosity", None),
        ("mass_balance", "fluid_source", None),
        ("mass_balance", "mobility", None),
        ("mass_balance", "fluid_density", None),
        ("mass_balance", "aperture", None),
        ("mass_balance", "darcy_flux", None),
        ("mass_balance", "interface_fluid_flux", None),
        ("mass_balance", "fluid_flux", None),
        ("mass_balance", "pressure_trace", None),
        ("mass_balance", "porosity", None),
        ("mass_balance", "reference_pressure", None),
        # The body force and stress are only meaningful in the top dimension
        ("momentum_balance", "body_force", 0),
        ("momentum_balance", "stress", 0),
        ("momentum_balance", "solid_density", None),
        ("momentum_balance", "lame_lambda", None),
        ("momentum_balance", "shear_modulus", None),
        ("momentum_balance", "youngs_modulus", None),
        ("momentum_balance", "bulk_modulus", None),
        # Reference gap, friction coefficient and friction bound are only meaningful for
        # a fracture of co-dimension 1.
        ("momentum_balance", "reference_fracture_gap", 1),
        ("momentum_balance", "friction_coefficient", 1),
        ("momentum_balance", "friction_bound", 1),
        # Energy balance
        ("energy_balance", "energy_source", None),
        ("energy_balance", "enthalpy_flux", None),
        ("energy_balance", "fourier_flux", None),
        ("energy_balance", "interface_enthalpy_flux", None),
        ("energy_balance", "interface_fourier_flux", None),
        ("energy_balance", "reference_temperature", None),
        ("energy_balance", "normal_thermal_conductivity", None),
        ("energy_balance", "aperture", None),
        # Poromechanics
        ("poromechanics", "reference_porosity", None),
        ("poromechanics", "biot_coefficient", 0),
        ("poromechanics", "matrix_porosity", 0),
        ("poromechanics", "porosity", None),
        ("poromechanics", "pressure_stress", 0),
        ("poromechanics", "stress", 0),
        ("poromechanics", "aperture", None),
        ("poromechanics", "fracture_pressure_stress", 1),
    ],
)
# Run the test for models with and without fractures. We skip the case of more than one
# fracture, since it seems unlikely this will uncover any errors that will not be found
# with the simpler setups. Activate more fractures if needed in debugging.
@pytest.mark.parametrize(
    "num_fracs",
    [  # Number of fractures
        0,
        1,
        pytest.param(2, marks=pytest.mark.skipped),
        pytest.param(3, marks=pytest.mark.skipped),
    ],
)
# By default we run only a 2d test. Activate 3d if needed in debugging.
@pytest.mark.parametrize("domain_dim", [2, pytest.param(3, marks=pytest.mark.skipped)])
def test_parse_constitutive_laws(
    model_type, method_name, only_codimension, num_fracs, domain_dim
):
    """Test that the ad parsing of constitutive laws works as intended.

    The workflow in the test is as follows: An object of the prescribed model is
    created, and the specified method is called. The three most likely reason a test may
    fail are:
        1. The method to be tested combines its components (grids, variables etc.) in a
           wrong way, causing an error in the call to the method.
        2. Something goes wrong in discretization.
        3. Something goes wroing in parsing of the Ad operator tree, perhaps due to
           combining matrices and arrays of incompatible dimensions.

    Note: Some methods are unfit for this test due to not having signature equal to
    either subdomains or interfaces.

    Parameters:
        model_type: Type of model to test. Currently supported are "mass_balance" and
            "momentum_balance". To add a new model, add a new class in models.py.
        method_name (str): Name of the method to test.
        only_codimension (list of int): Indices of the domains for which the method
            should be called. Some methods are only defined for a subset of domains. If
            not specified (None), all subdomains or interfaces in the model will be
            considered.

    """
    if only_codimension is not None:
        if only_codimension == 0 and num_fracs > 0:
            # If the test is to be run on the top domain only, we need not consider
            # models with fractures (note that since we iterate over num_fracs, the
            # test will be run for num_fracs = 0, which is the top domain only)
            return
        elif only_codimension == 1 and num_fracs == 0:
            # If the point of the test is to check the method on a fracture, we need
            # not consider models without fractures.
            return
        else:
            # We will run the test, but only on the specified codimension.
            dimensions_to_assemble = domain_dim - only_codimension
    else:
        # Test on subdomains or interfaces of all dimensions
        dimensions_to_assemble = None

    if domain_dim == 2 and num_fracs > 2:
        # The 2d models are not defined for more than two fractures.
        return

    # Set up an object of the prescribed model
    setup = models.model(model_type, domain_dim, num_fracs=num_fracs)
    # Fetch the relevant method of this model and extract the domains for which it is
    # defined.
    method = getattr(setup, method_name)
    domains = models.subdomains_or_interfaces_from_method_name(
        setup.mdg, method, dimensions_to_assemble
    )

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
    op.evaluate_value_and_jacobian(setup.equation_system)


# Shorthand for values with many digits. Used to compute expected values in the tests.
lbda = pp.solid_values.granite["lame_lambda"]
mu = pp.solid_values.granite["shear_modulus"]
bulk = lbda + 2 / 3 * mu


@pytest.mark.parametrize(
    "model, method_name, expected, dimension",
    [
        (models.Poromechanics, "fluid_density", 998.2 * np.exp(4.559e-10 * 2), None),
        (
            models.Thermoporomechanics,
            "fluid_density",
            # \rho = \rho_0 \exp(compressibility p - thermal_expansion T)
            998.2 * np.exp(4.559e-10 * 2 - 2.068e-4 * 3),
            None,
        ),
        (
            models.MassAndEnergyBalance,
            "thermal_conductivity",
            # Porosity weighted average of the solid and fluid thermal conductivities
            (1 - 1.3e-2) * 3.1 + 1.3e-2 * 0.5975,
            None,
        ),
        (
            models.MassAndEnergyBalance,
            "solid_enthalpy",
            # c_p T
            720.7 * 3,
            None,
        ),
        (
            models.MassAndEnergyBalance,
            "fluid_enthalpy",
            # c_p T
            4182 * 3,
            None,
        ),
        (
            models.Poromechanics,
            "matrix_porosity",
            # phi_0 + (alpha - phi_ref) * (1 - alpha) / bulk p. Only pressure, not
            # div(u), is included in this test.
            1.3e-2 + (0.47 - 1.3e-2) * (1 - 0.47) / bulk * 2,
            2,  # Matrix porosity is only defined in Nd
        ),
        (
            models.Thermoporomechanics,
            "matrix_porosity",
            # phi_0 + (alpha - phi_ref) * (1 - alpha) / bulk * p
            # - (alpha - phi_0) * thermal expansion * T
            #  Only pressure and temperature, not div(u), is included in this test.
            (
                1.3e-2
                + (0.47 - 1.3e-2) * (1 - 0.47) / bulk * 2
                - (0.47 - 1.3e-2) * 9.66e-6 * 3
            ),
            2,  # Matrix porosity is only defined in Nd
        ),
        (
            models.MomentumBalance,
            "bulk_modulus",
            # \lambda + 2/3 \mu
            bulk,
            None,
        ),
        (
            models.MomentumBalance,
            "shear_modulus",
            # \mu
            mu,
            None,
        ),
        (
            models.MomentumBalance,
            "youngs_modulus",
            # \mu (3 \lambda + 2 \mu) / (\lambda + \mu)
            mu * (3 * lbda + 2 * mu) / (lbda + mu),
            None,
        ),
        (
            models.MomentumBalance,
            "elastic_normal_fracture_deformation",
            # (-normal_traction) * maximum_closure /
            #    (normal_stiffness * maximum_closure + (-normal_traction))
            (1 * 1e-4) / (1529 * 1e-4 + 1),
            1,
        ),
        (
            # Tets permeability for the matrix domain. Should give the matrix
            # permeability
            models._add_mixin(c_l.CubicLawPermeability, models.MassBalance),
            "permeability",
            5.0e-18,
            2,
        ),
        (
            # Test the permeability for a fracture domain. This should be computed
            # by the cubic law (i.e., aperture squared by 12, an aditional aperture
            # scaling to get the transmissivity is taken care of elsewhere).
            models._add_mixin(c_l.CubicLawPermeability, models.MassBalance),
            "permeability",
            0.01**2 / 12,
            1,
        ),
        (
            # Test the permeability for an intersection. The reasoning is the same as
            # for the 1-d domain.
            models._add_mixin(c_l.CubicLawPermeability, models.MassBalance),
            "permeability",
            0.01**2 / 12,
            0,
        ),
    ],
)
def test_evaluated_values(
    model: Type[models.Poromechanics]
    | Type[models.Thermoporomechanics]
    | Type[models.MassAndEnergyBalance]
    | Type[models.MomentumBalance]
    | Type[_],  # noqa
    method_name: Literal[
        "fluid_density",
        "thermal_conductivity",
        "solid_enthalpy",
        "fluid_enthalpy",
        "matrix_porosity",
        "bulk_modulus",
        "shear_modulus",
        "youngs_modulus",
        "elastic_normal_fracture_deformation",
        "permeability",
    ],
    expected: Any | float | Literal[2370, 12540],
    dimension: Literal[2, 1, 0] | None,
):
    """Test that the value of the parsed operator is as expected.

    Parameters:
        model: Model class to be tested. Likely composed of several mixins.
        method_name: The specific method to be tested.
        expected: The expected value from the evaluation.
        dimension: Dimension on which the method will be invoked. If None, all
            dimensions in the MixedDimensionalGrid will be considered.

    """
    # The thermoporoelastic model covers most constitutive laws, so we use it for the
    # test.
    # Assign non-trivial values to the parameters to avoid masking errors.
    solid = pp.SolidConstants(solid_values)
    fluid = pp.FluidConstants(pp.fluid_values.water)
    params = {
        "material_constants": {"solid": solid, "fluid": fluid},
        "fracture_indices": [0, 1],
    }

    setup = model(params)
    setup.prepare_simulation()

    # Set variable values different from default zeros in iterate.
    # This yields non-zero perturbations.
    if hasattr(setup, "pressure_variable"):
        setup.equation_system.set_variable_values(
            2 * np.ones(setup.mdg.num_subdomain_cells()),
            [setup.pressure_variable],
            iterate_index=0,
        )
    if hasattr(setup, "temperature_variable"):
        setup.equation_system.set_variable_values(
            3 * np.ones(setup.mdg.num_subdomain_cells()),
            [setup.temperature_variable],
            iterate_index=0,
        )
    method = getattr(setup, method_name)

    # Call the method with the domain as argument. An error here will indicate that
    # something is wrong with the way the method combines terms and factors (e.g., grids
    # parameters, variables, other methods) to form an Ad operator object.
    # TODO: At some point, we should also consider interface laws.
    op = method(setup.mdg.subdomains(dim=dimension))
    if isinstance(op, np.ndarray):
        # This will happen if the return type of method is a pp.ad.DenseArray. EK is not
        # sure if we still have such methods among the constitutive laws, but the test
        # is correct, so keeping this should not do any harm.
        assert np.allclose(op, expected)
        return

    # The method should return an Ad object.
    assert isinstance(op, pp.ad.Operator)

    # Carry out discretization.
    op.discretize(setup.mdg)
    # Evaluate the discretized operator. An error here would typically be caused by the
    # method combining terms and factors of the wrong size etc. This could be a problem
    # with the constitutive law, or it could signify that something has changed in the
    # Ad machinery which makes the evaluation of the operator fail.
    val = op.evaluate_value(setup.equation_system)
    # Strict tolerance. We know analytical expected values, and some of the
    # perturbations are small relative to
    assert np.allclose(val, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "geometry, domain_dimension, expected",
    [
        (models.OrthogonalFractures3d, 1, [0.02, 0.02**2, 0.02]),
        (models.OrthogonalFractures3d, 0, [0.02, 0.02**3, 0.02**2]),
        (models.RectangularDomainThreeFractures, 0, [0.02, 0.02**2, 0.02]),
        (models.RectangularDomainThreeFractures, 2, [1, 1, 42]),
    ],
)
def test_dimension_reduction_values(
    geometry: pp.ModelGeometry, domain_dimension: int, expected: list[float]
):
    """Test that the value of the parsed operator is as expected.

    Parameters:
        geometry: The model geometry to use for the test.
        domain_dimension: The dimension of the domains to test.
        expected: The expected values of the:
            - subdomain aperture
            - subdomain specific volume
            - interface specific volume

    EK: This test could possibly be merged into the above test_evaluated_values(), but
    the risk of obfuscating the implementation. It was therefore decided to keep this
    method separate, at least for now.

    """
    # Assign non-trivial values to the parameters to avoid masking errors.
    solid = pp.SolidConstants({"residual_aperture": 0.02})
    params = {"material_constants": {"solid": solid}, "num_fracs": 3}
    if geometry is models.RectangularDomainThreeFractures:
        params["fracture_indices"] = [0, 1]

    class Model(geometry, pp.constitutive_laws.DimensionReduction, models.NoPhysics):
        pass

    setup = Model(params)
    setup.prepare_simulation()

    subdomains = setup.mdg.subdomains(dim=domain_dimension)
    interfaces = setup.mdg.interfaces(dim=domain_dimension)
    # Check aperture and specific volume values
    aperture = setup.aperture(subdomains).evaluate_value(setup.equation_system)
    assert np.allclose(aperture.data, expected[0])
    for grids, expected_value in zip([subdomains, interfaces], expected[1:]):
        specific_volume = setup.specific_volume(grids).evaluate_value(
            setup.equation_system
        )
        assert np.allclose(specific_volume.data, expected_value)
