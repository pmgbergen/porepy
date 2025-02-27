"""Module testing the initial condition mixin and its functionality.

All base models are tested by mixing in non-trivial IC.

Multiphysics models are tested through the base models they inherit.

"""

from __future__ import annotations

from copy import copy
from functools import cached_property
from itertools import permutations
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


def create_local_model_class(
    model_class: type[pp.PorePyModel], mixin_models: list[type[pp.PorePyModel]]
) -> type[pp.PorePyModel]:
    """Helper method to add mixins to a model class for testing purpose.

    Note that the order in ``mixin_models`` has an effect. First element will be mixed
    in first, last element last.

    """
    local_model_class: type[pp.PorePyModel] = model_class

    for mixin_model in mixin_models:
        local_model_class = _add_mixin(mixin_model, local_model_class)

    return local_model_class


class ICTestAllVariablesMixin(pp.PorePyModel):
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
    def IC_VALUES_PER_VARIABLE(self) -> dict[pp.ad.Variable, tuple[np.ndarray, bool]]:
        """Every method ``ic_values_*`` should store the values it returns
        for the respective ATOMIC variable (single-grid) in this cached property for
        testing reasons, as well as the flag whether the variable is initialized as a
        primary variable."""
        return {}

    def process_initialization(
        self,
        variable_method: Callable[[list[pp.GridLike]], pp.ad.MixedDimensionalVariable],
        domain: pp.GridLike,
        is_primary: bool,
    ) -> np.ndarray:
        """A method returning IC values for a variable represented by the method the
        respective variable mixin defines, on the grid on which it is initialized.

        The boolean is_primary must be set according to where the initialization of
        the variable is implemented: Either in set_initial_values_primary_variables
        or in the outer scope of initial_condition.

        """
        variable = variable_method([domain]).sub_vars[0]
        shape = self.equation_system.dofs_of([variable]).shape
        np.random.seed(42)
        value = np.random.rand(*shape)
        self.IC_VALUES_PER_VARIABLE[variable] = (value, is_primary)
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


class ICTestPrimaryVariablesMixin(ICTestAllVariablesMixin):
    """A special mixin for testing purpose which overloads the method setting
    initial values for primary variables.

    The overload asserts that all values for variables flagged as primary are set
    after the super-framework finishes with the method
    set_initial_values_primary_variables.
    It also tests that the evaluation of non-primary variables will fail because no
    values were set so far.

    """

    def set_initial_values_primary_variables(self) -> None:
        super().set_initial_values_primary_variables()

        # Keeping track of all the variables tested.
        all_variables = []

        # After the IC values for primaries are set, we assert that the primary
        # variables can be evaluated and the value is as expected.

        for var, data in self.IC_VALUES_PER_VARIABLE.items():
            expected_value, is_primary = data

            # So far only primary variables were set.
            assert is_primary

            # Check that the value is as expected.
            current_value = self.equation_system.evaluate(var)
            np.testing.assert_allclose(
                current_value, expected_value, rtol=0.0, atol=1e-16
            )

            all_variables.append(var)

        # Now we loop over all variables and check if they are were initialized.
        # If they were not initialized with random values as per this test mixin,
        # they should be by default initialized with zeros by the InitialConditionMixin
        primary_variables = list(self.IC_VALUES_PER_VARIABLE.keys())
        for var in self.equation_system.variables:
            if var not in primary_variables:
                val = self.equation_system.evaluate(var)
                np.testing.assert_allclose(val, 0.0, rtol=0.0, atol=1e-16)

                all_variables.append(var)

        # Final check that we have indeed tested all variables.
        # This is primarily a sanity check for the tests.
        # I.e., all variables are covered and flagged as either primary or non-primary.
        assert set(all_variables) == set(self.equation_system.variables)


class ICTestSinglePhaseFlowMixin(pp.PorePyModel):
    """Single phase flow class with non-trivial IC for every variable.

    The only primary variable is pressure.

    """

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return self.process_initialization(self.pressure, sd, True)

    def ic_values_interface_darcy_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.interface_darcy_flux, intf, False)

    def ic_values_well_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.well_flux, intf, False)


class ICTestEnergyMixin(pp.PorePyModel):
    """Non-trivial IC for the energy-related variables.

    The only primary variable is temperature.

    """

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return self.process_initialization(self.temperature, sd, True)

    def ic_values_interface_fourier_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.interface_fourier_flux, intf, False)

    def ic_values_interface_enthalpy_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.interface_enthalpy_flux, intf, False)

    def ic_values_well_enthalpy_flux(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.well_enthalpy_flux, intf, False)


class ICTestMomentumBalancerMixin(pp.PorePyModel):
    """Non-trivial IC for the mechanics-related variables.

    Both displacement and interface displacement are considered primary.

    """

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        return self.process_initialization(self.displacement, sd, True)

    def ic_values_interface_displacement(self, intf: pp.MortarGrid) -> np.ndarray:
        return self.process_initialization(self.interface_displacement, intf, True)


class ICTestContactMechanicsMixin(pp.PorePyModel):
    """Non-trivial IC for contact mechanics variables.

    The contact traction is considered a primary variable.

    """

    def ic_values_contact_traction(self, sd: pp.Grid) -> np.ndarray:
        return self.process_initialization(self.contact_traction, sd, True)


class ICTestTracerFlowMixin(pp.PorePyModel):
    """Non-trivial IC values for fractions in tracer flow.

    Overall component fractions are considered primary.

    """

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        return self.process_initialization(component.fraction, sd, True)


model_configurations: list[tuple[type[pp.PorePyModel], list[type[pp.PorePyModel]]]] = [
    (
        pp.fluid_mass_balance.SinglePhaseFlow,
        [ICTestSinglePhaseFlowMixin],
    ),
    (
        pp.momentum_balance.MomentumBalance,
        [
            ICTestMomentumBalancerMixin,
            ICTestContactMechanicsMixin,
        ],
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
]
"""Configuration of a tested models and relevant IC mixins.

Note:
    The model must be compatible with any geometry model mixin from the applications
    library.

    It must also be able to execute prepare_simulation with default model params
    (empty dict).

    The Terzaghi setup for example cannot be tested this way because of its geometry.
    But considering that the standard Biot model and Mandel setup are testable, we can
    consider it covered.

"""


@pytest.mark.parametrize(
    "ic_test_mixin", [ICTestAllVariablesMixin, ICTestPrimaryVariablesMixin]
)
@pytest.mark.parametrize(
    "geometry_model",
    [None, SquareDomainOrthogonalFractures, CubeDomainOrthogonalFractures],
)
@pytest.mark.parametrize(
    "model_class,ic_mixin_models",
    model_configurations,
)
def test_initial_values_are_set(
    model_class: type[pp.PorePyModel],
    ic_mixin_models: list[type[pp.PorePyModel]],
    geometry_model: Optional[type[pp.PorePyModel]],
    ic_test_mixin: type[pp.PorePyModel],
) -> None:
    """Test for PorePy models checking whether IC values are correctly set and
    initialized for all iterate and time step indices.

    Tests also if the values of variables flagged as primary by the IC test mixin are
    indeed set first, and then the others.
    That test is implemented in the a special IC test mixin via assertions, not directly
    here.

    """

    # Creating the local list of mixins to be added to the model
    # Make shallow copy to not mess with other tests.
    local_mixins = [mixin_class for mixin_class in ic_mixin_models]

    # Add the geometry mixin to test on various geometries.
    # None means the default grid is used (no fractures)
    # Also check that the geometry used in the test is not already part of the tested
    # model class. Otherwise Python will throw an error regarding unresolvable MRO.
    if geometry_model is not None and geometry_model not in model_class.__mro__:
        local_mixins += [geometry_model]

    # Add the IC test mixin relevant for this test.
    local_mixins += [ic_test_mixin]

    # create local model class and add some extra typing support.
    local_model_class: type[ICTestAllVariablesMixin] = create_local_model_class(
        model_class, local_mixins
    )

    model = local_model_class({"times_to_export": []})
    # Create variables and initialize
    # NOTE there are some tests included here when the ICTestPrimaryVariableMixin is
    # used.
    model.prepare_simulation()

    # Use the functionality of the IC test mixin to evaluate every fixed-dimensional
    # variable and check if the returned value is the same as the one set.

    for var, data in model.IC_VALUES_PER_VARIABLE.items():
        expected_value, _ = data
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


@pytest.mark.parametrize(
    "ic_test_mixin", [ICTestAllVariablesMixin, ICTestPrimaryVariablesMixin]
)
@pytest.mark.parametrize(
    "permutations_ic_thermoporomechanics",
    list(
        permutations(
            [
                pp.energy_balance.InitialConditionsEnergy,
                pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow,
                pp.momentum_balance.InitialConditionsMomentumBalance,
                pp.contact_mechanics.InitialConditionsContactTraction,
            ]
        )
    ),
)
def test_thermoporomechanics_initialization_is_mro_insensitive(
    permutations_ic_thermoporomechanics: tuple[type[pp.PorePyModel], ...],
    ic_test_mixin: type[pp.PorePyModel],
) -> None:
    """This test uses the Thermoporomechanics model and tests that the
    Initialization procedure is insensitive w.r.t. to the order of how individual
    initial condition mixins are mixed in.

    The Thermoporomechanics model encompasses all base models/physics and ensures the
    widest test coverage of all models.

    """

    # First construct a thermoporomechanics class without IC mixins at all
    class LocalThermoporomechanics(
        pp.thermoporomechanics.SolutionStrategyThermoporomechanics,
        pp.thermoporomechanics.EquationsThermoporomechanics,
        pp.thermoporomechanics.VariablesThermoporomechanics,
        pp.thermoporomechanics.BoundaryConditionsThermoporomechanics,
        pp.thermoporomechanics.ConstitutiveLawsThermoporomechanics,
        pp.FluidMixin,
        pp.ModelGeometry,
        pp.DataSavingMixin,
    ):
        pass

    # Now we construct the list of classes to be mixed in, starting with the permutation
    # of IC mixins in the original Thermoporomechanics model.
    # Then the Test IC mixins with custom IC values are mixed in, finally the geometry
    # and the test mixin.
    local_mixins: list[type[pp.PorePyModel]] = (
        [_ for _ in permutations_ic_thermoporomechanics]
        + [
            ICTestSinglePhaseFlowMixin,
            ICTestMomentumBalancerMixin,
            ICTestContactMechanicsMixin,
            ICTestEnergyMixin,
        ]
        + [CubeDomainOrthogonalFractures, ic_test_mixin]
    )

    model_class: type[ICTestAllVariablesMixin] = create_local_model_class(
        LocalThermoporomechanics, local_mixins
    )

    # The rest of the test is analogous to test_initial_values_are_set, but shorter.
    model = model_class({"times_to_export": []})
    model.prepare_simulation()

    for var, data in model.IC_VALUES_PER_VARIABLE.items():
        expected_value, _ = data
        current_value = model.equation_system.evaluate(var)

        assert isinstance(current_value, np.ndarray)
        assert len(current_value.shape) == 1

        np.testing.assert_allclose(current_value, expected_value, rtol=0.0, atol=1e-16)
