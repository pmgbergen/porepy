"""
This module contains tests that are devised to guarantee a correct integration between the
TimeManager class and the PorePy models.
"""

import porepy as pp
import numpy as np
import pytest


class TestDefaultBehaviour:
    """Class that checks the default behaviour of the models"""

    default_models = [
        pp.IncompressibleFlow(params={"use_ad": True}),
        pp.SlightlyCompressibleFlow(params={"use_ad": True}),
        pp.ContactMechanics(params={"use_ad": True}),
        pp.ContactMechanicsBiot(params={"use_ad": True}),
        pp.THM(params={"use_ad": True}),
    ]

    @pytest.mark.parametrize("model", default_models)
    def test_default_behaviour(self, model):
        """Checks default time behaviour of models"""

        # Time-independent models should not have a time manager
        if not model._is_time_dependent():
            pp.run_stationary_model(model, model.params)
            assert not hasattr(model, "time_manager")
        # Time-dependent models should return default values after running the models
        else:
            pp.run_time_dependent_model(model, model.params)
            assert model.time_manager.time_init == 0
            assert model.time_manager.time_final == 1
            assert model.time_manager.time == 1
            assert model.time_manager.dt == 1
            