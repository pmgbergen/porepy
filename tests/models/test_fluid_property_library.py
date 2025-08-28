import pytest
import numpy as np
import porepy as pp
from tests.functional.setups.buoyancy_flow_model import ModelGeometry2D
from tests.functional.setups.buoyancy_flow_model import ModelMDGeometry2D
from tests.functional.setups.buoyancy_flow_model import (
    BuoyancyFlowModel2N,
    BuoyancyFlowModel3N,
)
from tests.functional.setups.buoyancy_flow_model import to_Mega


# Parameterization list for both tests
Parameterization = [
    (BuoyancyFlowModel2N),
    (BuoyancyFlowModel3N),
]


def _build_buoyancy_model(
    model_class: type,
    md: bool = False,
) -> None:
    """build buoyancy flow model for given parameters."""
    day = 86400
    tf = 0.5 * day
    dt = 0.25 * day
    solid_constants = pp.SolidConstants(
        permeability=1.0e-14,
        porosity=0.1,
        thermal_conductivity=2.0 * to_Mega,
        density=2500.0,
        specific_heat_capacity=1000.0 * to_Mega,
    )
    if md:
        geometry2d = ModelMDGeometry2D
    else:
        geometry2d = ModelGeometry2D

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
        iter_max=50,
        print_info=True,
    )
    params = {
        "fractional_flow": True,
        "buoyancy_on": True,
        "material_constants": {"solid": solid_constants},
        "time_manager": time_manager,
        "apply_schur_complement_reduction": False,
        "nl_convergence_tol": np.inf,
        "nl_convergence_tol_res": np.inf,
    }

    # Combine geometry with model class
    class Model2D(geometry2d, model_class):
        pass

    model = Model2D(params)
    model.prepare_simulation()
    return model


def __common_assertions(model):
    """
    Assert fluid configuration and keys for a given phase context.
    """

    phase_context = model.fluid.phases
    unkown_phase = pp.Phase(0, "unkown")
    assert model.phase_pairs_for(unkown_phase) == []

    phase_names = [phase.name for phase in phase_context]
    for phase in phase_context:
        reduced_phase_names = [name for name in phase_names if name != phase.name]
        pairs = model.phase_pairs_for(phase)
        for pair in pairs:
            assert phase == pair[0]
            if len(reduced_phase_names) == 1:
                assert pair[1].name == reduced_phase_names[0]
            else:
                assert (
                    pair[1].name == reduced_phase_names[0]
                    or pair[1].name == reduced_phase_names[1]
                )

            assert (
                model.buoyancy_key(pair[0], pair[1])
                == "buoyancy_" + pair[0].name + "_" + pair[1].name
            )
            assert (
                model.buoyant_flux_array_key(pair[0], pair[1])
                == "buoyant_flux_" + pair[0].name + "_" + pair[1].name
            )
            assert (
                model.buoyancy_intf_key(pair[0], pair[1])
                == "buoyancy_intf_" + pair[0].name + "_" + pair[1].name
            )
            assert (
                model.buoyant_intf_flux_array_key(pair[0], pair[1])
                == "buoyant_intf_flux_" + pair[0].name + "_" + pair[1].name
            )


def __subdomains_assertions(model):
    """
    Assert that subdomain fluxes are correctly computed for the model.
    """

    # test keys
    phase_context = model.fluid.phases
    component_context = model.fluid.components
    subdomains = model.mdg.subdomains()

    eval = lambda op: model.equation_system.evaluate(op)
    are_equal = lambda a, b: np.allclose(a, b)

    rho_hat = eval(model.fractionally_weighted_density(subdomains))
    if len(phase_context) == 2:
        rho_hat_expected = np.array(
            [
                1000.0,
                207.92079208,
                1000.0,
                207.92079208,
                1000.0,
                207.92079208,
                200.0,
                207.92079208,
                200.0,
                207.92079208,
                1000.0,
                207.92079208,
                1000.0,
                207.92079208,
                1000.0,
                207.92079208,
                200.0,
                207.92079208,
                200.0,
                207.92079208,
                1000.0,
                207.92079208,
                1000.0,
                207.92079208,
                1000.0,
            ]
        )
    else:
        rho_hat_expected = np.array(
            [
                1000.0,
                271.92982456,
                1000.0,
                271.92982456,
                1000.0,
                271.92982456,
                252.25225225,
                271.92982456,
                252.25225225,
                271.92982456,
                1000.0,
                271.92982456,
                1000.0,
                271.92982456,
                1000.0,
                271.92982456,
                252.25225225,
                271.92982456,
                252.25225225,
                271.92982456,
                1000.0,
                271.92982456,
                1000.0,
                271.92982456,
                1000.0,
            ]
        )
    assert are_equal(rho_hat, rho_hat_expected)

    w_flux = eval(model.density_driven_flux(subdomains, pp.ad.Scalar(1.0)))
    if len(phase_context) == 2:
        w_flux_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.85134888e-07,
                -1.79270887e-06,
                -1.85134888e-07,
                -1.79270887e-06,
                -1.85134888e-07,
                -1.85134888e-07,
                -1.79270887e-06,
                -1.85134888e-07,
                -1.79270887e-06,
                -1.85134888e-07,
                -1.85134888e-07,
                -1.79270887e-06,
                -1.85134888e-07,
                -1.79270887e-06,
                -1.85134888e-07,
                -1.85134888e-07,
                -1.79270887e-06,
                -1.85134888e-07,
                -1.79270887e-06,
                -1.85134888e-07,
                0.0,
                0.0,
                -9.80665000e-08,
                0.0,
                0.0,
            ]
        )
    else:
        w_flux_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.79694758e-07,
                -1.23819178e-06,
                -1.79694758e-07,
                -1.23819178e-06,
                -1.79694758e-07,
                -1.79694758e-07,
                -1.23819178e-06,
                -1.79694758e-07,
                -1.23819178e-06,
                -1.79694758e-07,
                -1.79694758e-07,
                -1.23819178e-06,
                -1.79694758e-07,
                -1.23819178e-07,
                -1.79694758e-07,
                -1.79694758e-07,
                -1.23819178e-06,
                -1.79694758e-07,
                -1.23819178e-06,
                -1.79694758e-07,
                0.0,
                0.0,
                -9.80665000e-08,
                0.0,
                0.0,
            ]
        )

    assert are_equal(w_flux, w_flux_expected)

    buoyancy_fluxes = []
    for component in component_context:
        c_buoyancy_flux = eval(model.component_buoyancy(component, subdomains))
        buoyancy_fluxes.append(c_buoyancy_flux)
    overall_buoyancy_flux = np.sum(np.array(buoyancy_fluxes), axis=0)
    assert are_equal(overall_buoyancy_flux, np.zeros_like(overall_buoyancy_flux))

    h_buoyancy_flux = eval(model.enthalpy_buoyancy(subdomains))
    if len(phase_context) == 2:
        h_buoyancy_flux_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.96215821e-19,
                1.40590834e-05,
                2.96215821e-19,
                1.40590834e-05,
                2.96215821e-19,
                1.45189599e-06,
                0.0,
                1.45189599e-06,
                0.0,
                1.45189599e-06,
                2.96215821e-19,
                1.40590834e-05,
                2.96215821e-19,
                1.40590834e-05,
                2.96215821e-19,
                1.45189599e-06,
                0.0,
                1.45189599e-06,
                0.0,
                1.45189599e-06,
                0.0,
                0.0,
                1.56906400e-19,
                0.0,
                0.0,
            ]
        )
    else:
        h_buoyancy_flux_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                3.06379562e-19,
                5.48783061e-05,
                3.06379562e-19,
                5.48783061e-05,
                3.06379562e-19,
                7.96431059e-06,
                3.33139011e-05,
                7.96431059e-06,
                3.33139011e-05,
                7.96431059e-06,
                3.06379562e-19,
                5.48783061e-05,
                3.06379562e-19,
                5.48783061e-05,
                3.06379562e-19,
                7.96431059e-06,
                3.33139011e-05,
                7.96431059e-06,
                3.33139011e-05,
                7.96431059e-06,
                0.0,
                0.0,
                1.67203382e-19,
                0.0,
                0.0,
            ]
        )
    assert np.allclose(
        h_buoyancy_flux, h_buoyancy_flux_expected
    ), "Buoyancy flux values do not match expected results."
    assert (
        h_buoyancy_flux.shape == h_buoyancy_flux_expected.shape
    ), "Buoyancy flux output shape mismatch."


def __interface_assertions(model):
    """
    Assert that interface fluxes are correctly computed for the model.
    """
    # test keys
    phase_context = model.fluid.phases
    component_context = model.fluid.components
    subdomains = model.mdg.subdomains()
    interfaces = model.mdg.interfaces()

    eval = lambda op: model.equation_system.evaluate(op)
    are_equal = lambda a, b: np.allclose(a, b)

    intf_w_flux = eval(
        model.interface_density_driven_flux(interfaces, pp.ad.Scalar(1.0))
    )
    if len(phase_context) == 2:
        intf_w_flux_expected = np.array(
            [
                1.96133000e-06,
                1.65078608e-06,
                1.96133000e-06,
                -1.96133000e-06,
                -1.65078608e-06,
                -1.96133000e-06,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.96133000e-07,
                1.96133000e-07,
                -1.96133000e-07,
                1.96133000e-07,
                -1.96133000e-07,
                1.96133000e-07,
            ]
        )
    else:
        intf_w_flux_expected = np.array(
            [
                1.46533982e-06,
                1.07201462e-06,
                1.46533982e-06,
                -1.46533982e-06,
                -1.07201462e-06,
                -1.46533982e-06,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.46533982e-07,
                1.46533982e-07,
                -1.46533982e-07,
                1.46533982e-07,
                -1.46533982e-07,
                1.46533982e-07,
            ]
        )

    assert are_equal(intf_w_flux, intf_w_flux_expected)

    buoyancy_flux_jumps = []
    for component in component_context:
        c_buoyancy_flux_jump = eval(
            model.component_buoyancy_jump(component, subdomains)
        )
        buoyancy_flux_jumps.append(c_buoyancy_flux_jump)
    overall_buoyancy_flux_jump = np.sum(np.array(buoyancy_flux_jumps), axis=0)
    assert are_equal(
        overall_buoyancy_flux_jump, np.zeros_like(overall_buoyancy_flux_jump)
    )

    h_buoyancy_flux_jump = eval(model.enthalpy_buoyancy_jump(subdomains))
    if len(phase_context) == 2:
        h_buoyancy_flux_jump_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
    else:
        h_buoyancy_flux_jump_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -3.94253834e-06,
                0.0,
                0.0,
                -3.94253834e-06,
            ]
        )
    assert are_equal(h_buoyancy_flux_jump, h_buoyancy_flux_jump_expected)


@pytest.mark.parametrize("model_class", Parameterization)
def akatest_fluid_buoyancy_fd(model_class):
    """Test buoyancy-driven flow model (FD)."""
    fd_model = _build_buoyancy_model(model_class, md=False)
    __common_assertions(fd_model)
    __subdomains_assertions(fd_model)


@pytest.mark.parametrize("model_class", Parameterization)
def test_fluid_buoyancy_md(model_class):
    """Test buoyancy-driven flow model (MD)."""
    md_model = _build_buoyancy_model(model_class, md=True)
    __common_assertions(md_model)
    __interface_assertions(md_model)
