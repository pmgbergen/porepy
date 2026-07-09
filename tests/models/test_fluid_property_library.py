import numpy as np
import pytest

import porepy as pp
from tests.functional.setups.buoyancy_flow_model import (
    buoyancy_flow_model,
    ModelGeometry2D,
    ModelMDGeometry2D,
    to_Mega,
)

# Test suite for FluidBuoyancy class functionality.
# Validates buoyancy operators by instantiating models and evaluating
# computed operators.

# Parameterization for testing both 2-component and 3-component models
Parameterization = [
    (buoyancy_flow_model(2)),  # 2-component buoyancy flow model
    (buoyancy_flow_model(3)),  # 3-component buoyancy flow model
]


def _build_buoyancy_model(
    model_class: type,
    md: bool = False,
) -> None:
    """
    Build a buoyancy flow model for given parameters.

    Args:
        model_class: The model class to instantiate (2N or 3N component model)
        md: Whether to use mixed-dimensional geometry (default: False)

    Returns:
        Configured and prepared buoyancy flow model instance
    """
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
        "enable_buoyancy_effects": True,
        "material_constants": {"solid": solid_constants},
        "time_manager": time_manager,
        "apply_schur_complement_reduction": False,
        "nl_convergence_inc_atol": np.inf,
        "nl_convergence_res_atol": np.inf,
    }

    # Combine geometry with model class
    class Model2D(geometry2d, model_class):
        pass

    model = Model2D(params)
    model.prepare_simulation()
    return model


def __common_assertions(model):
    """
    Verify fluid configuration and phase pair relationships.

    Tests:
    - Phase pair generation logic for unknown phases
    - Correct pairing of phases in multi-phase contexts
    - Consistency of buoyancy-related dictionary keys
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

            # The refactored FluidBuoyancy uses a single two-direction hybrid-upwind
            # keyword per ordered phase pair (see HUpwind); the former per-array keys
            # (buoyant_flux_array_key, buoyancy_intf_key, ...) no longer exist.
            assert (
                model.hybrid_upwind_key(pair[0], pair[1])
                == "hybrid_upwind_" + pair[0].name + "_" + pair[1].name
            )


def __subdomains_assertions(model):
    """Verify subdomain buoyancy invariants:

    - the fractionally-weighted density is a mobility-weighted average of the phase
      densities, so it lies within their range and is non-degenerate;
    - component buoyancy fluxes are reciprocal (sum to zero);
    - the density-driven and enthalpy buoyancy fluxes are finite.

    Geometry/IC-independent, so there are no golden arrays to update when those change.
    """
    phase_context = model.fluid.phases
    component_context = model.fluid.components
    subdomains = model.mdg.subdomains()
    eval = lambda op: model.equation_system.evaluate(op)

    rho_hat = eval(model.fractionally_weighted_density(subdomains))
    phase_densities = [eval(phase.density(subdomains)) for phase in phase_context]
    lo, hi = np.min(phase_densities), np.max(phase_densities)
    assert np.all(rho_hat >= lo - 1e-9) and np.all(rho_hat <= hi + 1e-9)
    assert rho_hat.min() < rho_hat.max()  # composition varies -> density varies

    overall_buoyancy_flux = np.sum(
        [eval(model.component_buoyancy(c, subdomains)) for c in component_context],
        axis=0,
    )
    assert np.allclose(overall_buoyancy_flux, 0.0)

    assert np.all(
        np.isfinite(eval(model.density_driven_flux(subdomains, pp.ad.Scalar(1.0))))
    )
    assert np.all(np.isfinite(eval(model.enthalpy_buoyancy(subdomains))))



def __interface_assertions(model):
    """Verify interface flux invariants for mixed-dimensional cases:

    - the interface density-driven (gravity) flux is balanced across the interface
      (antisymmetric, so it sums to zero) and non-trivial;
    - component-wise buoyancy flux jumps are reciprocal (sum to zero);
    - the enthalpy buoyancy flux jump vanishes (temperature is held at zero here).

    These are geometry-independent, so they need not be updated when the mesh changes.
    """
    component_context = model.fluid.components
    subdomains = model.mdg.subdomains()
    interfaces = model.mdg.interfaces()
    eval = lambda op: model.equation_system.evaluate(op)

    intf_w_flux = eval(
        model.interface_density_driven_flux(interfaces, pp.ad.Scalar(1.0))
    )
    assert np.isclose(intf_w_flux.sum(), 0.0)
    assert np.any(np.abs(intf_w_flux) > 0.0)

    buoyancy_flux_jumps = [
        eval(model.component_buoyancy_jump(component, subdomains))
        for component in component_context
    ]
    overall_buoyancy_flux_jump = np.sum(np.array(buoyancy_flux_jumps), axis=0)
    assert np.allclose(overall_buoyancy_flux_jump, 0.0)

    h_buoyancy_flux_jump = eval(model.enthalpy_buoyancy_jump(subdomains))
    assert np.allclose(h_buoyancy_flux_jump, 0.0)


@pytest.mark.parametrize("model_class", Parameterization)
def test_fluid_buoyancy_fd(model_class):
    """Test FluidBuoyancy class with fixed-dimensional (FD) geometry."""
    fd_model = _build_buoyancy_model(model_class, md=False)
    __common_assertions(fd_model)
    __subdomains_assertions(fd_model)


@pytest.mark.parametrize("model_class", Parameterization)
def test_fluid_buoyancy_md(model_class):
    """Test FluidBuoyancy class with mixed-dimensional (MD) geometry."""
    md_model = _build_buoyancy_model(model_class, md=True)
    __common_assertions(md_model)
    __interface_assertions(md_model)
