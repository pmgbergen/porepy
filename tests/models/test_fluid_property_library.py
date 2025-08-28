
import pytest
import numpy as np
import porepy as pp
from tests.functional.setups.buoyancy_flow_model import ModelGeometry2D, ModelGeometry3D
from tests.functional.setups.buoyancy_flow_model import ModelMDGeometry2D, ModelMDGeometry3D
from tests.functional.setups.buoyancy_flow_model import BuoyancyFlowModel2N, BuoyancyFlowModel3N
from tests.functional.setups.buoyancy_flow_model import to_Mega


# Parameterization list for both tests
Parameterization = [
    (BuoyancyFlowModel2N, True),
    (BuoyancyFlowModel2N, True),
    (BuoyancyFlowModel2N, True),
    (BuoyancyFlowModel2N, False),
    (BuoyancyFlowModel2N, False),
    (BuoyancyFlowModel2N, False),
    (BuoyancyFlowModel3N, True),
    (BuoyancyFlowModel3N, True),
    (BuoyancyFlowModel3N, True),
    (BuoyancyFlowModel3N, False),
    (BuoyancyFlowModel3N, False),
    (BuoyancyFlowModel3N, False),
]


def _build_buoyancy_model(
    model_class: type,
    mesh_2d_Q: bool,
    md: bool = False,
) -> None:
    """Run buoyancy flow simulation for given parameters."""
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
        geometry3d = ModelMDGeometry3D
    else:
        geometry2d = ModelGeometry2D
        geometry3d = ModelGeometry3D

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
    if mesh_2d_Q:
        class Model2D(geometry2d, model_class): pass
        model = Model2D(params)
    else:
        class Model3D(geometry3d, model_class): pass
        model = Model3D(params)
    model.prepare_simulation()
    return model

def __common_assertions(model):

    # test keys
    phase_context = model.fluid.phases
    component_context = model.fluid.components
    unkown_phase = pp.Phase(0,"unkown")
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
                assert pair[1].name == reduced_phase_names[0] or pair[1].name == reduced_phase_names[1]

            assert model.buoyancy_key(pair[0], pair[1]) == "buoyancy_" + pair[0].name + "_" + pair[1].name
            assert model.buoyant_flux_array_key(pair[0], pair[1]) == "buoyant_flux_" + pair[0].name + "_" + pair[1].name
            assert model.buoyancy_intf_key(pair[0], pair[1]) == "buoyancy_intf_" + pair[0].name + "_" + pair[1].name
            assert model.buoyant_intf_flux_array_key(pair[0], pair[1]) == "buoyant_intf_flux_" + pair[0].name + "_" + pair[1].name


@pytest.mark.parametrize("model_class, mesh_2d_Q", Parameterization)
def test_fluid_buoyancy_fd(model_class, mesh_2d_Q):
    """Test buoyancy-driven flow model (FD)."""
    fd_model = _build_buoyancy_model(model_class, mesh_2d_Q, md=False)
    __common_assertions(fd_model)


@pytest.mark.parametrize("model_class, mesh_2d_Q", Parameterization)
def test_fluid_buoyancy_md(model_class, mesh_2d_Q):
    """Test buoyancy-driven flow model (MD)."""
    md_model = _build_buoyancy_model(model_class, mesh_2d_Q, md=True)
    __common_assertions(md_model)