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
import scipy.sparse as sps
import pytest

import porepy as pp
import porepy.models.constitutive_laws as c_l
from porepy.applications.test_utils import models
from porepy.applications.test_utils.reference_dense_arrays import (
    test_constitutive_laws as reference_dense_arrays,
)
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.applications.discretizations.flux_discretization import FluxDiscretization


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
    op.value_and_jacobian(setup.equation_system)


# Shorthand for values with many digits. Used to compute expected values in the tests.
lbda = pp.solid_values.granite["lame_lambda"]
mu = pp.solid_values.granite["shear_modulus"]
bulk = lbda + 2 / 3 * mu
reference_arrays = reference_dense_arrays["test_evaluated_values"]


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
            # Porosity weighted average of the solid and fluid thermal conductivities.
            # Expand to format for an isotropic second order tensor.
            ((1 - 1.3e-2) * 3.1 + 1.3e-2 * 0.5975)
            * reference_arrays["isotropic_second_order_tensor"],
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
            # permeability. 9 * (nc = 32) entries of isotropic permeability.
            models._add_mixin(c_l.CubicLawPermeability, models.MassBalance),
            "permeability",
            5.0e-18 * reference_arrays["isotropic_second_order_tensor"][: 9 * 32],
            2,
        ),
        (
            # Test the permeability for a fracture domain. This should be computed by
            # the cubic law (i.e., aperture squared by 12, an aditional aperture scaling
            # to get the transmissivity is taken care of elsewhere). 9 * (nc = 6)
            # entries of isotropic permeability.
            models._add_mixin(c_l.CubicLawPermeability, models.MassBalance),
            "permeability",
            0.01**2 / 12 * reference_arrays["isotropic_second_order_tensor"][: 9 * 6],
            1,
        ),
        (
            # Test the permeability for an intersection. The reasoning is the same as
            # for the 1-d domain. 9 entries of isotropic permeability.
            models._add_mixin(c_l.CubicLawPermeability, models.MassBalance),
            "permeability",
            0.01**2 / 12 * np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
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
    val = op.value(setup.equation_system)
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
    aperture = setup.aperture(subdomains).value(setup.equation_system)
    assert np.allclose(aperture.data, expected[0])
    for grids, expected_value in zip([subdomains, interfaces], expected[1:]):
        specific_volume = setup.specific_volume(grids).value(setup.equation_system)
        assert np.allclose(specific_volume.data, expected_value)


class PoromechanicalTestDiffTpfa(
    FluxDiscretization,
    SquareDomainOrthogonalFractures,
    pp.constitutive_laws.CubicLawPermeability,
    pp.constitutive_laws.DarcysLawAd,
    pp.poromechanics.Poromechanics,
):
    """Helper class to test the derivative of the Darcy flux with respect to the mortar
    displacement.
    """

    def __init__(self, params):
        params.update(
            {
                "fracture_indices": [1],
                "grid_type": "cartesian",
                "meshing_arguments": {"cell_size_x": 0.5, "cell_size_y": 0.5},
            }
        )

        super().__init__(params)

    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Non-constant permeability tensor, the y-component depends on pressure."""
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0, size=0)

        nc = sum([sd.num_cells for sd in subdomains])
        # K is a second order tensor having nd^2 entries per cell. 3d:
        # Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz
        # 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8
        tensor_dim = 3**2

        # Set constant component of the permeability - isotropic unit tensor
        all_vals = np.zeros(nc * tensor_dim, dtype=float)
        all_vals[0::tensor_dim] = 1
        all_vals[4::tensor_dim] = 1
        all_vals[8::tensor_dim] = 1

        # Basis vector for the yy-component
        e_yy = self.e_i(subdomains, i=4, dim=tensor_dim)

        return (
            pp.wrap_as_dense_ad_array(all_vals, name="Constant_permeability_component")
            + e_yy @ self.pressure(subdomains) ** 2
        )

    def initial_condition(self):
        """Set the initial condition for the problem.

        The interface displacement is non-zero in the y-direction to trigger a non-zero
        contribution from the derivative of the permeability with respect to the mortar
        displacement.
        """
        super().initial_condition()

        # Fetch the mortar interface and the subdomains.
        intf = self.mdg.interfaces()[0]
        g_2d, g_1d = self.mdg.subdomains()

        # Projection from the mortar to the primary grid, and directly from the mortar
        # cells to the high-dimensional cells. The latter uses an np.abs to avoid issues
        # with + and - in g.cell_faces.
        proj_high = intf.mortar_to_primary_int()
        mortar_to_high_cell = np.abs(g_2d.cell_faces.T @ proj_high)

        self.mortar_to_high_cell = mortar_to_high_cell

        # Set the mortar displacement, firs using the ordering of the 2d cells
        u_2d_x = np.array([0, 0, 0, 0])
        u_2d_y = np.array([0, 0, 1, 2])
        # .. and then map down to the mortar cells.
        u_mortar_x = mortar_to_high_cell.T @ u_2d_x
        u_mortar_y = mortar_to_high_cell.T @ u_2d_y

        # Define the full mortar displacement vector and set it in the equation system.
        u_mortar = np.vstack([u_mortar_x, u_mortar_y]).ravel("F")
        self.equation_system.set_variable_values(
            u_mortar, [self.interface_displacement_variable], iterate_index=0
        )

        # Store the y-component of the mortar displacement, using a Cartesian ordering
        # of the mortar cells (i.e., the same as the ordering of the 2d cells).
        self.u_mortar = u_2d_y

        # Fetch the global dof of the mortar displacement in the y-direction.
        dof_u_mortar_y = self.equation_system.dofs_of(
            [self.interface_displacement_variable]
        )[1::2]
        # Reorder the global dof to match the ordering of the 2d cells, and store it
        r, *_ = sps.find(mortar_to_high_cell)
        self.global_dof_u_mortar_y = dof_u_mortar_y[r]

        # Set the pressure variable in the 1d domain: The pressure is 2 in the leftmost
        # fracture cell, 0 in the rightmost fracture cell. This should give a flux
        # pointing to the right.
        if np.diff(g_1d.cell_centers[0])[0] > 0:
            p_1d = np.array([2, 0])
        else:
            p_1d = np.array([0, 2])

        p_1d_var = self.equation_system.get_variables(
            [self.pressure_variable], grids=[g_1d]
        )
        self.equation_system.set_variable_values(p_1d, p_1d_var, iterate_index=0)
        self.p_1d = p_1d

        # Set the interface Darcy flux to unity on all mortar cells.
        interface_flux = np.arange(intf.num_cells)
        self.equation_system.set_variable_values(
            interface_flux, [self.interface_darcy_flux_variable], iterate_index=0
        )
        self.interface_flux = interface_flux

        # Set the pressure in the 2d grid
        p_2d_var = self.equation_system.get_variables(
            [self.pressure_variable], grids=[g_2d]
        )
        p_2d = np.arange(g_2d.num_cells)
        self.equation_system.set_variable_values(p_2d, p_2d_var, iterate_index=0)
        self.p_2d = p_2d

        self.global_intf_ind = self.equation_system.dofs_of(
            [self.interface_darcy_flux_variable]
        )
        self.global_p_2d_ind = self.equation_system.dofs_of(p_2d_var)


@pytest.mark.parametrize("base_discr", ["tpfa", "mpfa"])
def test_derivatives_darcy_flux_potential_trace(base_discr: str):
    """Test the derivative of the Darcy flux with respect to the mortar displacement,
    and of the potential reconstruciton with respect to the interface flux.

    The test for the mortar displacement is done for a 2d domain with a single fracture,
    and a 1d domain with two cells. The mortar displacement is set to a constant value
    in the x-direction, but varying value in the y-direction. The permeability is given
    by the cubic law, and the derivative of the permeability with respect to the mortar
    displacement is non-zero. The test checks that the derivative of the Darcy flux with
    respect to the mortar displacement is correctly computed.

    The test for the interface flux uses the same overall setup, but the focus is on the
    2d domain, and the potential reconstruction on the internal boundary to the
    fracture. The permeability in the 2d domain is a function of the pressure, thus the
    pressure reconstruction operator should be dependent both on the 2d pressure and on
    the interface flux.

    """

    # Set up and discretize model
    model = PoromechanicalTestDiffTpfa({"darcy_flux_discretization": base_discr})
    model.prepare_simulation()
    model.discretize()

    # Fetch the mortar interface and the 1d subdomain.
    g_2d, g_1d = model.mdg.subdomains()

    ### First the test for the derivative of the Darcy flux with respect to the mortar
    # displacement.
    #
    # Get the y-component of the mortar displacement, ordered in the same way as the 2d
    # cells, that is
    #   2, 3
    #   0, 1
    # Thus the jumps in the mortar displacement are cell 3 - cell 2, and cell 1 - cell
    # 0.
    u_m = model.u_mortar

    # The permeability is given by the cubic law, calculate this and its derivative.
    resid_ap = model.residual_aperture([g_1d]).value(model.equation_system)
    k_0 = ((u_m[2] - u_m[0]) + resid_ap) ** 3 / 12
    k_1 = (u_m[3] - u_m[1] + resid_ap) ** 3 / 12

    # Derivative of the permeability with respect to the mortar displacement
    dk0_du2 = (u_m[2] - u_m[0] + resid_ap) ** 2 / 4
    dk0_du0 = -((u_m[2] - u_m[0] + resid_ap) ** 2) / 4
    dk1_du3 = (u_m[3] - u_m[1] + resid_ap) ** 2 / 4
    dk1_du1 = -((u_m[3] - u_m[1] + resid_ap) ** 2) / 4

    # Calculate the transmissibility. First, get the distance between the cell center and
    # the face center (will be equal on the two sides of the face).
    dist = np.abs(g_1d.face_centers[0, 1] - g_1d.cell_centers[0, 0])

    # Half transmissibility
    trm_0 = k_0 / dist
    trm_1 = k_1 / dist

    # The derivative of the transmissibility with respect to the mortar displacement is
    # given by the chain rule (a warm thanks to copilot):
    dtrm_du0 = (dk0_du0 / dist) * trm_1 / (trm_0 + trm_1) - trm_0 * trm_1 * dk0_du0 / (
        dist * (trm_0 + trm_1) ** 2
    )
    dtrm_du2 = (dk0_du2 / dist) * trm_1 / (trm_0 + trm_1) - trm_0 * trm_1 * dk0_du2 / (
        dist * (trm_0 + trm_1) ** 2
    )
    dtrm_du1 = (dk1_du1 / dist) * trm_0 / (trm_0 + trm_1) - trm_0 * trm_1 * dk1_du1 / (
        dist * (trm_0 + trm_1) ** 2
    )
    dtrm_du3 = (dk1_du3 / dist) * trm_0 / (trm_0 + trm_1) - trm_0 * trm_1 * dk1_du3 / (
        dist * (trm_0 + trm_1) ** 2
    )

    # We also need the pressure difference. Multiply with the sign of the divergence to
    # account for the direction of the normal vector.
    dp = (model.p_1d[1] - model.p_1d[0]) * g_1d.cell_faces[1, 1]

    # Finally, the true values that should be compared with the discretization.
    true_derivatives = dp * np.array([dtrm_du0, dtrm_du1, dtrm_du2, dtrm_du3])

    # The computed flux
    computed_flux = model.darcy_flux([g_1d]).value_and_jacobian(model.equation_system)
    # Pick out the middle face, and only those faces that are associated with the mortar
    # displacement in the y-direction. The column indices must also be reordered to
    # match the ordering of the 2d cells (which was used to set 'true_derivatives').
    dt_du_computed = computed_flux.jac[
        1, (model.mortar_to_high_cell @ model.global_dof_u_mortar_y).astype(int)
    ].A.ravel()

    assert np.allclose(dt_du_computed, true_derivatives)

    ### Now the test for the derivative of the potential reconstruction with respect to
    # the interface flux.
    #
    # The potential reconstruction method
    potential_trace = model.potential_trace(
        model.mdg.subdomains(), model.pressure, model.permeability, "darcy_flux"
    ).value_and_jacobian(model.equation_system)

    # Fetch the permeability tensor from the data dictionary.
    data_2d = model.mdg.subdomain_data(model.mdg.subdomains()[0])
    k = data_2d[pp.PARAMETERS][model.darcy_keyword]["second_order_tensor"]
    # Only the yy-component is needed to calculate the reconstructed pressure, since the
    # face centers on the fracutre, and their respective 2d cell centers are aligned
    # with the x-axis.
    k_yy = k.values[1, 1, :]

    # Fetch the pressure in the 2d domain and the interface flux.
    p_2d = model.p_2d
    # The interface flux is mapped to the Cartesian ordering of the 2d cells.
    interface_flux = model.mortar_to_high_cell @ model.interface_flux

    # We know the distance from cell to face center (same for all cells and faces)
    dist = 0.5

    # The ordering of the fracture faces in the 2d domain is different from the
    # Cartesian ordering of the 2d cells, and from the ordering of the mortar cells. The
    # below code gives the fracture faces in the order corresponding to that of the 2d
    # cells.
    fracture_faces = np.where(g_2d.tags["fracture_faces"])[0]
    fracture_faces_cart_ordering = fracture_faces[
        g_2d.cell_faces[fracture_faces].indices
    ]

    # The reconstructed pressure is given by the sum of the pressure in the cell and the
    # pressure difference that drives the flux. The minus sign is needed since the
    # interface flux is defined as positive out of the high-dimensional cell.
    p_reconstructed = p_2d - interface_flux * dist / k_yy
    assert np.allclose(
        p_reconstructed, potential_trace.val[fracture_faces_cart_ordering]
    )

    # The Jacobian for the potential reconstruction should have two terms: One relating
    # to the pressure in the cell, and one relating to the interface flux.

    # We know that the yy-component of the permeability tensor contains a p^2 term
    dk_yy_dp = 2 * p_2d / dist
    # The Jacobian with respect to the pressure, represented as a diagonal matrix.
    true_jac_dp = np.diag(1 + interface_flux / (k_yy / dist) ** 2 * dk_yy_dp)

    # For ease of comparison, we reorder the Jacobian to match the ordering of the 2d
    # cells. No need to reorder the columns, as the 2d pressure already has the correct
    # ordering.
    assert np.allclose(
        true_jac_dp,
        potential_trace.jac[fracture_faces_cart_ordering][:, model.global_p_2d_ind].A,
    )

    # The computed Jacobian. Here we also need to reorder the columns from mortar to
    # high-dimensional cell ordering.
    computed_dp_dl = potential_trace.jac[fracture_faces_cart_ordering][
        :, model.mortar_to_high_cell @ model.global_intf_ind
    ].A
    # The true Jacobian with respect to the interface flux is found by differentiating
    # p_reconstructed with respect to the interface flux.
    true_dp_dl = np.diag(-dist / k_yy)
    assert np.allclose(computed_dp_dl, true_dp_dl)

    # Finally check there are no more terms in the Jacobian. We do no filtering of
    # elements that are essentially zero; EK would not have been surprised if that
    # turned out to be needed for Mpfa, but it seems to work without.
    assert potential_trace.jac[fracture_faces_cart_ordering].data.size == 8
