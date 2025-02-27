"""Module testing the initial condition mixin and its functionality.

All base models are tested by mixing in non-trivial IC.

Multiphysics models are tested through the base models they inherit.

"""

from __future__ import annotations

from copy import copy
from functools import cached_property
from typing import Callable, Optional

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import _add_mixin
from porepy.examples.mandel_biot import MandelSetup
from porepy.examples.tracer_flow import TracerFlowSetup
from porepy.models.derived_models.biot import BiotPoromechanics


class ICTestMixin(pp.PorePyModel):
    """Mixin defining interface for testing results of initialization procedure."""

    @property
    def time_step_indices(self) -> np.ndarray:
        """For testing whether IC values are copied to all time step indices."""
        return np.array([0, 1])

    @property
    def iterate_indices(self) -> np.ndarray:
        """For testing whether IC values are copied to all iterate indices."""
        return np.array([0, 1])

    @cached_property
    def IC_VALUES_PER_VARIABLE(self) -> dict[pp.ad.Variable, np.ndarray]:
        """Every method ``ic_values_*`` should store the values it returns
        for the respective ATOMIC variable (single-grid) in this cached property for
        testing reasons."""
        return {}

    def process_initialization(
        self,
        variable_method: Callable[[list[pp.GridLike]], pp.ad.MixedDimensionalVariable],
        domain: pp.GridLike,
    ) -> np.ndarray:
        """A method returning IC values for a variable represented by the method the
        respective variable mixin defines, on the grid on which it is initialized."""
        variable = variable_method([domain]).sub_vars[0]
        shape = self.equation_system.dofs_of([variable]).shape
        np.random.seed(42)
        value = np.random.rand(*shape)
        self.IC_VALUES_PER_VARIABLE[variable] = value
        return value

    def prepare_simulation(self) -> None:
        """Partial model set-up including only steps required to create and initialize
        variables for testing purpose."""
        self.set_materials()
        self.set_geometry()
        self.initialize_data_saving()
        self.set_equation_system_manager()
        self.create_variables()
        self.assign_thermodynamic_properties_to_phases()
        self.initial_condition()
        self.initialize_time_and_iteration_indices()


class ICTestSinglePhaseFlowMixin(pp.PorePyModel):
    """Single phase flow class with non-trivial IC for every variable."""

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return self.process_initialization(self.pressure, sd)

    def ic_values_interface_darcy_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.interface_darcy_flux, intf)

    def ic_values_well_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.well_flux, intf)


class ICTestEnergyMixin(pp.PorePyModel):
    """Non-trivial IC for the energy-related variables."""

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return self.process_initialization(self.temperature, sd)

    def ic_values_interface_fourier_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.interface_fourier_flux, intf)

    def ic_values_interface_enthalpy_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.interface_enthalpy_flux, intf)

    def ic_values_well_enthalpy_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.well_enthalpy_flux, intf)


class ICTestMomentumBalancerMixin(pp.PorePyModel):
    """Non-trivial IC for the mechanics-related variables."""

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        return self.process_initialization(self.displacement, sd)

    def ic_values_interface_displacement(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.interface_displacement, intf)


class ICTestContactMechanicsMixin(pp.PorePyModel):
    """Non-trivial IC for contact mechanics variables."""

    def ic_values_contact_traction(self, sd: pp.Grid) -> np.ndarray:
        return self.process_initialization(self.contact_traction, sd)


class ICTestTracerFlowMixin(pp.PorePyModel):
    """Non-trivial IC values for fractions in tracer flow"""

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        return self.process_initialization(component.fraction, sd)


@pytest.mark.parametrize(
    "geometry_model",
    [None, SquareDomainOrthogonalFractures, CubeDomainOrthogonalFractures],
)
@pytest.mark.parametrize(
    "model_class,ic_mixin_models",
    [
        (
            pp.fluid_mass_balance.SinglePhaseFlow,
            [ICTestSinglePhaseFlowMixin],
        ),
        (
            pp.momentum_balance.MomentumBalance,
            [ICTestMomentumBalancerMixin],
        ),
        (
            pp.contact_mechanics.ContactMechanics,
            [ICTestContactMechanicsMixin],
        ),
        (
            pp.poromechanics.Poromechanics,
            [
                ICTestSinglePhaseFlowMixin,
                ICTestMomentumBalancerMixin,
                ICTestContactMechanicsMixin,
            ],
        ),
        (
            pp.thermoporomechanics.Thermoporomechanics,
            [
                ICTestSinglePhaseFlowMixin,
                ICTestMomentumBalancerMixin,
                ICTestContactMechanicsMixin,
                ICTestEnergyMixin,
            ],
        ),
        (
            BiotPoromechanics,
            [
                ICTestSinglePhaseFlowMixin,
                ICTestMomentumBalancerMixin,
                ICTestContactMechanicsMixin,
            ],
        ),
        (
            MandelSetup,
            [
                ICTestSinglePhaseFlowMixin,
                ICTestMomentumBalancerMixin,
                ICTestContactMechanicsMixin,
            ],
        ),
        (
            TracerFlowSetup,
            [
                ICTestSinglePhaseFlowMixin,
                ICTestTracerFlowMixin,
            ],
        ),
    ],
)
def test_initial_values_are_set(
    model_class: type[pp.PorePyModel],
    ic_mixin_models: list[type[pp.PorePyModel]],
    geometry_model: Optional[type[pp.PorePyModel]],
) -> None:
    """Test for PorePy models checking whether IC values are correctly set and
    initialized for all iterate and time step indices."""

    local_model_class: type[pp.PorePyModel] = model_class

    # Add all IC value mixins which are relevant for the tested class
    for ic_mixin_model in ic_mixin_models:
        local_model_class = _add_mixin(ic_mixin_model, local_model_class)

    # Add the geometry mixin to test on various geometries.
    # None means the default grid is used (no fractures)
    # Also check that the geometry used in the test is not already part of the tested
    # model class. Otherwise Python will throw an error regarding unresolvable MRO.
    if geometry_model is not None and geometry_model not in local_model_class.__mro__:
        local_model_class = _add_mixin(geometry_model, local_model_class)

    # Add the IC test mixin. Add type hint for some extra support
    local_model_class: type[ICTestMixin] = _add_mixin(ICTestMixin, local_model_class)

    model_params = {"times_to_export": []}

    model = local_model_class(model_params)
    # Create variables and initialize
    model.prepare_simulation()

    # Use the functionality of the IC test mixin to evaluate every fixed-dimensional
    # variable and check if the returned value is the same as the one set.

    for var, expected_value in model.IC_VALUES_PER_VARIABLE.items():
        current_value = model.equation_system.evaluate(var)

        # First, check we get an array as expected.
        assert isinstance(current_value, np.ndarray)
        assert len(current_value.shape) == 1

        # Check the values at the current iterate.
        np.testing.assert_allclose(current_value, expected_value, rtol=0.0, atol=1e-16)

        # Check the values at all previous iterates.
        var_i = copy(var)
        for _ in model.iterate_indices:
            var_i = var_i.previous_iteration()
            prev_iter_value = model.equation_system.evaluate(var_i)
            np.testing.assert_allclose(
                prev_iter_value, expected_value, rtol=0.0, atol=1e-16
            )

        # Check the values at all previous time steps.
        var_t = copy(var)
        for _ in model.iterate_indices:
            var_t = var_t.previous_iteration()
            prev_time_value = model.equation_system.evaluate(var_t)
            np.testing.assert_allclose(
                prev_time_value, expected_value, rtol=0.0, atol=1e-16
            )
