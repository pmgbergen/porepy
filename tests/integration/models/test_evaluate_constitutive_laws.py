"""This module provides quantitative tests of the constitutive laws in the models.

The main test method is test_evaluated_values, but there is also a dedicated method
for testing geometries and dimension reduction.

"""
from __future__ import annotations

import numpy as np
import pytest

import porepy as pp
import porepy.models.constitutive_laws as c_l

from . import setup_utils


def _add_mixin(mixin, parent):
    """Helper method to dynamically construct a class by adding a mixin.

    Multiple mixins can be added by nested calls to this method.

    Reference:
        https://www.geeksforgeeks.org/create-classes-dynamically-in-python/

    """
    name = parent.__name__
    # IMPLEMENTATION NOTE: The last curly bracket can be used to add code to the created
    # class (empty brackets is equivalent to a ``pass``). In principle, we could add
    # this as an extra parameter to this function, but at the moment it is unclear why
    # such an addition could not be made in the mixin class instead.
    cls = type(f"Extended_{name}", (mixin, parent), {})
    return cls


@pytest.mark.parametrize(
    "model, method_name, expected, dimension",
    [
        (setup_utils.Poromechanics, "fluid_density", 1000 * np.exp(4e-10 * 2), None),
        (
            setup_utils.Thermoporomechanics,
            "fluid_density",
            # \rho = \rho_0 \exp(compressibility p - thermal_expansion T)
            1000 * np.exp(4e-10 * 2 - 2.1e-4 * 3),
            None,
        ),
        (
            setup_utils.MassAndEnergyBalance,
            "thermal_conductivity",
            # Porosity weighted average of the solid and fluid thermal conductivities
            (1 - 7e-3) * 2.5 + 7e-3 * 0.6,
            None,
        ),
        (
            setup_utils.MassAndEnergyBalance,
            "solid_enthalpy",
            # c_p T
            790 * 3,
            None,
        ),
        (
            setup_utils.MassAndEnergyBalance,
            "fluid_enthalpy",
            # c_p T
            4180 * 3,
            None,
        ),
        (
            setup_utils.Poromechanics,
            "matrix_porosity",
            # phi_0 + (alpha - phi_ref) * (1 - alpha) / bulk p. Only pressure, not
            # div(u), is included in this test.
            7e-3 + (0.8 - 7e-3) * (1 - 0.8) / (11.11 * pp.GIGA) * 2,
            None,
        ),
        (
            setup_utils.Thermoporomechanics,
            "matrix_porosity",
            # phi_0 + (alpha - phi_ref) * (1 - alpha) / bulk * p
            # - (alpha - phi_0) * thermal expansion * T
            #  Only pressure and temperature, not div(u), is included in this test.
            (
                7e-3
                + (0.8 - 7e-3) * (1 - 0.8) / (11.11 * pp.GIGA) * 2
                - (0.8 - 7e-3) * 1e-5 * 3
            ),
            None,
        ),
        (
            setup_utils.MomentumBalance,
            "bulk_modulus",
            # \lambda + 2/3 \mu
            11.11 * pp.GIGA + 2 / 3 * 16.67 * pp.GIGA,
            None,
        ),
        (
            setup_utils.MomentumBalance,
            "shear_modulus",
            # \mu
            16.67 * pp.GIGA,
            None,
        ),
        (
            setup_utils.MomentumBalance,
            "youngs_modulus",
            # \mu (3 \lambda + 2 \mu) / (\lambda + \mu)
            16.67 * (3 * 11.11 + 2 * 16.67) / (11.11 + 16.67) * pp.GIGA,
            None,
        ),
        (
            _add_mixin(pp.constitutive_laws.BartonBandis, setup_utils.MomentumBalance),
            "elastic_normal_fracture_deformation",
            # (-normal_traction) * maximum_closure /
            #    (normal_stiffness * maximum_closure + (-normal_traction))
            (1 * 0.1683) / (1529 * 0.1683 + 1),
            1,
        ),
        (
            _add_mixin(c_l.CubicLawPermeability, setup_utils.MassBalance),
            "permeability",
            1e-20,
            2,
        ),
        (
            _add_mixin(c_l.CubicLawPermeability, setup_utils.MassBalance),
            "permeability",
            0.01**3 / 12,
            1,
        ),
        (
            _add_mixin(c_l.CubicLawPermeability, setup_utils.MassBalance),
            "permeability",
            0.01**4 / 12,
            0,
        ),
    ],
)
def test_evaluated_values(model, method_name, expected, dimension):
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
    solid = pp.SolidConstants(setup_utils.granite_values)
    fluid = pp.FluidConstants(setup_utils.water_values)
    params = {
        "material_constants": {"solid": solid, "fluid": fluid},
    }

    setup = model(params)
    setup.prepare_simulation()

    # Set variable values different from default zeros in iterate.
    # This yields non-zero perturbations.
    if hasattr(setup, "pressure_variable"):
        setup.equation_system.set_variable_values(
            2 * np.ones(setup.mdg.num_subdomain_cells()),
            [setup.pressure_variable],
            to_iterate=True,
        )
    if hasattr(setup, "temperature_variable"):
        setup.equation_system.set_variable_values(
            3 * np.ones(setup.mdg.num_subdomain_cells()),
            [setup.temperature_variable],
            to_iterate=True,
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
    val = op.evaluate(setup.equation_system)
    if isinstance(val, pp.ad.AdArray):
        val = val.val
    # Strict tolerance. We know analytical expected values, and some of the
    # perturbations are small relative to
    assert np.allclose(val, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "geometry, domain_dimension, expected",
    [
        (setup_utils.OrthogonalFractures3d, 1, [0.02, 0.02**2, 0.02]),
        (setup_utils.OrthogonalFractures3d, 0, [0.02, 0.02**3, 0.02**2]),
        (setup_utils.RectangularDomainThreeFractures, 0, [0.02, 0.02**2, 0.02]),
        (setup_utils.RectangularDomainThreeFractures, 2, [1, 1, 42]),
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
    if geometry is setup_utils.RectangularDomainThreeFractures:
        params["fracture_indices"] = [0, 1]

    class Model(
        geometry, pp.constitutive_laws.DimensionReduction, setup_utils.NoPhysics
    ):
        pass

    setup = Model(params)
    setup.prepare_simulation()

    subdomains = setup.mdg.subdomains(dim=domain_dimension)
    interfaces = setup.mdg.interfaces(dim=domain_dimension)
    # Check aperture and specific volume values
    aperture = setup.aperture(subdomains).evaluate(setup.equation_system)
    if isinstance(aperture, pp.ad.AdArray):
        aperture = aperture.val
    assert np.allclose(aperture.data, expected[0])
    for grids, expected_value in zip([subdomains, interfaces], expected[1:]):
        specific_volume = setup.specific_volume(grids).evaluate(setup.equation_system)
        if isinstance(specific_volume, pp.ad.AdArray):
            specific_volume = specific_volume.val
        assert np.allclose(specific_volume.data, expected_value)
