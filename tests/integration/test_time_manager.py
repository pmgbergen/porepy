"""
This module contains tests that are devised to guarantee a correct integration between the
TimeManager class and the PorePy models.
"""

import porepy as pp
import numpy as np
import pytest

# Model classes to be tested
all_models = [
    pp.IncompressibleFlow,
    pp.SlightlyCompressibleFlow,
    pp.ContactMechanics,
    pp.ContactMechanicsBiot,
    pp.THM
]

time_dependent_models = [
    pp.SlightlyCompressibleFlow,
    pp.ContactMechanicsBiot,
    pp.THM
]


# -----> Testing defaults
@pytest.mark.parametrize("model_class", all_models)
def test_default_behaviour(model_class):
    """Test the default behaviour of the models"""

    model = model_class(params={"use_ad": True})

    if not model._is_time_dependent():
        pp.run_stationary_model(model, model.params)
        assert not hasattr(model, "time_manager")
    else:
        pp.run_time_dependent_model(model, model.params)
        assert model.time_manager.is_constant
        assert model.time_manager.time_init == 0
        assert model.time_manager.time_final == 1
        assert model.time_manager.time == 1
        assert model.time_manager.dt == 1
        assert model._ad.time_step.evaluate(model.dof_manager) == model.time_manager.dt


# -----> Testing constant time step
tm0 = pp.TimeManager([0, 1, 2], 1, constant_dt=True)
tm1 = pp.TimeManager([0, 0.01, 0.02, 0.05], 0.01, constant_dt=True)
tm2 = pp.TimeManager([0, pp.HOUR, 2*pp.HOUR, 10*pp.HOUR], pp.HOUR, constant_dt=True)


@pytest.mark.parametrize("model_class", time_dependent_models)
@pytest.mark.parametrize("time_manager", [tm0, tm1, tm2])
def test_constant_time_step(request, model_class, time_manager):
    """Test models when constant time step is used"""

    model = model_class(params={"use_ad": True, "time_manager": time_manager})
    pp.run_time_dependent_model(model, model.params)
    if "0" in request.node.callspec.id:
        assert model.time_manager.is_constant
        assert model.time_manager.time_init == 0
        assert model.time_manager.time_final == 2
        assert model.time_manager.dt == 1
        assert model.time_manager.time == 2
    elif "1" in request.node.callspec.id:
        assert model.time_manager.is_constant
        assert model.time_manager.time_init == 0
        assert model.time_manager.time_final == 0.05
        assert model.time_manager.dt == 0.01
        assert model.time_manager.time == 0.05
    elif "2" in request.node.callspec.id:
        assert model.time_manager.is_constant
        assert model.time_manager.time_init == 0
        assert model.time_manager.time_final == 10 * pp.HOUR
        assert model.time_manager.dt == pp.HOUR
        assert model.time_manager.time == 10 * pp.HOUR


# -----> Testing linear models
# TODO: Replace [pp.SlightlyCompressibleFlow] by time_dependent_models
@pytest.mark.parametrize("model_class", [pp.SlightlyCompressibleFlow])
def test_linear_non_constant_dt(model_class):
    """Test linear models when non-constant time steps are used"""
    time_manager = pp.TimeManager(schedule=[0, 1], dt_init=0.01)
    model = model_class(params={"use_ad": True, "time_manager": time_manager})
    with pytest.raises(NotImplementedError) as excinfo:
        msg = "Currently, time step cannot be adapted when the problem is linear."
        pp.run_time_dependent_model(model, model.params)
    assert msg in str(excinfo.value)

# -----> Testing non-linear models