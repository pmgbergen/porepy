"""
Test functionalities in the example case of the geomthermal reservoir.

"""

from typing import Literal

import numpy as np
import pytest

import porepy as pp
import porepy.applications.md_grids.model_geometries
from porepy.applications.discretizations.flux_discretization import FluxDiscretization
from porepy.applications.test_utils import well_models
from porepy.examples.geothermal_reservoir import (
    BoundaryConditionsMechanicsNeumann,
    GeothermalReservoirWellBCs,
    NeumannWellBCsFirstTimeInterval,
    WellBoundaryConditions,
    set_model_params,
    set_solver_params,
)
from porepy.numerics.nonlinear import line_search


class FastMeshingMixin:
    """Helper mixin to enforce a coarse grid without having to set the meshing arguments
    in each model and test separately."""

    def meshing_arguments(self) -> dict:
        return {"cell_size": 0.25}

    def grid_type(self) -> Literal["cartesian"]:
        return "cartesian"


class GeothermalModelNeumann(
    FastMeshingMixin,
    well_models.OneVerticalWell,
    porepy.applications.md_grids.model_geometries.CubeDomainOrthogonalFractures,
    NeumannWellBCsFirstTimeInterval,
    pp.Poromechanics,
):
    pass


@pytest.fixture
def neuBC_model():
    model = GeothermalModelNeumann()
    model.prepare_simulation()
    return model


@pytest.fixture
def well_subdomains(neuBC_model):
    model = neuBC_model
    wells = [sd for sd in model.mdg.subdomains() if model.is_well_grid(sd)]

    assert len(wells) > 0
    return wells


def test_NeumannWellBCs_in_FirstTimeInterval(neuBC_model, well_subdomains):
    """
    Test that well grids have Neumann BCs during the first time interval.
    """
    model = neuBC_model
    model.time_manager.time = model.time_manager.schedule[0]
    for sd in well_subdomains:
        bc = model.bc_type_darcy_flux(sd)
        assert not np.any(bc.is_dir)


class OneVerticalInjectionWell(well_models.OneVerticalWell):
    def set_well_network(self) -> None:
        super().set_well_network()
        self.well_network.wells[0].tags["well_name"] = "injection_well"


class GeothermalModelWell(
    FastMeshingMixin,
    OneVerticalInjectionWell,
    porepy.applications.md_grids.model_geometries.CubeDomainOrthogonalFractures,
    WellBoundaryConditions,
    pp.Thermoporomechanics,
):
    @property
    def well_names(self):
        return ["injection_well"]


@pytest.fixture
def well_bc_model():
    params = {
        "injection_well_pressures": [1e6, 1e6],
        "injection_well_temperatures": [300.00, 300.00],
    }
    model = GeothermalModelWell(params)
    model.prepare_simulation()
    return model


def test_well_bcs_pressure(well_bc_model):
    """
    Test the boundary conditions of one well for pressure.
    """
    model = well_bc_model
    model.time_manager.time = model.time_manager.schedule[0]
    wells = [sd for sd in model.mdg.subdomains() if model.is_well_grid(sd)]
    assert len(wells) == 1

    expected_value = model.units.convert_units(1e6, "Pa")

    bg = model.mdg.subdomain_to_boundary_grid(wells[0])
    values = model.bc_values_pressure(bg)

    assert np.any(np.isclose(values, expected_value))


def test_well_bcs_temperature(well_bc_model):
    """
    Test the boundary conditions of one well for temperature.
    """
    model = well_bc_model
    model.time_manager.time = model.time_manager.schedule[0]
    wells = [sd for sd in model.mdg.subdomains() if model.is_well_grid(sd)]
    assert len(wells) == 1

    expected_value = model.units.convert_units(300.00, "K")

    bg = model.mdg.subdomain_to_boundary_grid(wells[0])
    values = model.bc_values_temperature(bg)

    assert np.any(np.isclose(values, expected_value))


class GeothermalModelMechanics(
    FastMeshingMixin,
    well_models.OneVerticalWell,
    porepy.applications.md_grids.model_geometries.CubeDomainOrthogonalFractures,
    BoundaryConditionsMechanicsNeumann,
    pp.Poromechanics,
):
    pass


@pytest.fixture
def mechanics_bc_model():
    model = GeothermalModelMechanics()
    model.prepare_simulation()
    return model


def test_mechanics_bcs_neumann(mechanics_bc_model):
    """
    Test the boundary conditions of mechanics.
    """
    model = mechanics_bc_model
    matrix_grids = [sd for sd in model.mdg.subdomains() if sd.dim == model.nd]
    assert len(matrix_grids) == 1

    sd = matrix_grids[0]
    bc = model.bc_type_mechanics(sd)

    faces = model.faces_to_fix(sd)

    expected_dir = [
        np.array([False, True, True]),
        np.array([False, True, True]),
        np.array([True, False, True]),
    ]

    assert len(faces) == 3
    assert np.any(bc.is_dir)
    assert not np.all(bc.is_dir)

    for i, value in zip(faces, expected_dir):
        assert np.array_equal(bc.is_dir[:, i], value)
        assert np.array_equal(bc.is_neu[:, i], ~value)


# MARK: Integration


@pytest.mark.skipped(reason="slow")
def test_geothermal_reservoir():
    """This is a slow integration test, which runs the whole model with realistic
    parameters in a relatively coarse grid (~5k dofs total). The tests ensures that:
    Initialization in the porous medium:
        - pressure equilibrates
        - temperature equilibrates
        - displacement equilibrates
    - Injection well and adjuscent fracture:
        - pressure increases
        - temperature decreases
    - Production well and adjuscent fracture:
        - pressure increases (not in the fracture)
        - temperature decreases

    """
    # MARK: Setup

    # Get the model parameteres as defined in the example. Compared to that setup, we
    # will introduce some simplifications to speed up the test:
    # - The injection schedule and simulation time are shortened and simplified.
    # - The grid is coarser.
    # - Diffusive terms are discretized using TPFA instead of MPFA.
    # - Boundary conditions for the injection well are changed to make testing of flow
    #   in the well more reliable.
    model_params = set_model_params()

    # The model setup is mostly copied from porepy/examples/geothermal_reservoir.py

    # Initial time step. Note that the tests check near-equality between the final two
    # time steps of the initialization phase. Hence dt_init, INITIALIZATION_LENGTH and
    # the TimeManager should be set up such that the system is in equilibrium no later
    # than the penultimate time step. This is verified below.
    dt_init = 20 * pp.YEAR
    # The initialization phase is run for INITIALIZATION_LENGTH * dt_init. A value of
    # 3.1 * dt_init has been found sufficient for the system to reach a near-equilibrium
    # state in this test setup while keeping the overall runtime reasonable.
    INITIALIZATION_LENGTH = 3.1
    schedule = np.array(
        [
            0,  # Initialization, wells are off.
            dt_init * INITIALIZATION_LENGTH,  # Initialization done, wells are pumping.
            dt_init * INITIALIZATION_LENGTH + 100 * pp.SECOND,  # Simulation ends.
        ]
    )

    # Injection pressure schedule, its size == schedule.size
    injection_pressures = [1e5, 1e5, 5e5]  # [Pa]

    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=dt_init,
        constant_dt=False,
        dt_min_max=(10 * pp.SECOND, dt_init),
        iter_optimal_range=(6, 10),  # Allow more iterations than default.
        iter_relax_factors=(0.5, 1.8),  # More aggressive relaxation
    )

    length_scale = model_params["length_scale"]
    fracture_size = model_params["fracture_size"]

    # Define model parameters.
    model_params.update(
        {
            "darcy_flux_discretization": "tpfa",
            "fourier_flux_discretization": "tpfa",
            # Set time manager.
            "time_manager": time_manager,
            # Set physical parameters.
            "injection_well_temperatures": 250.00,
            "injection_well_pressures": injection_pressures,
            # Set geometry and meshing related parameters.
            "meshing_arguments": {
                "cell_size": length_scale / 4.0,
                "cell_size_fracture": fracture_size * length_scale * 0.7,
                "background_transition_multiplier": 6.0,
            },
        }
    )

    # Data saved in the simulation for the test.
    pressure_data_initialization = []
    temperature_data_initialization = []
    displacement_data_initialization = []
    pressure_data_injection_well = []
    darcy_flux_data_injection_well = []
    temperature_data_injection_well = []
    pressure_data_production_well = []
    darcy_flux_data_production_well = []
    temperature_data_production_well = []
    pressure_data_injection_fracture = []
    temperature_data_injection_fracture = []
    pressure_data_production_fracture = []
    temperature_data_production_fracture = []

    class ModelForTest(FluxDiscretization, GeothermalReservoirWellBCs):
        def after_nonlinear_convergence(self):
            # YZ: We don't use the model method collect_data, because it is triggered
            # both for checkpoints and for failed time steps. We don't need the latter.
            # This should be reconsidered when the behavior of collect_data is fixed.
            super().after_nonlinear_convergence()
            mdg: pp.MixedDimensionalGrid = self.mdg
            matrix = mdg.subdomains(dim=self.nd)
            t = self.time_manager.time
            if t <= self.time_manager.schedule[1]:
                pressure_data_initialization.append(
                    self.equation_system.evaluate(self.pressure(matrix))
                )
                temperature_data_initialization.append(
                    self.equation_system.evaluate(self.temperature(matrix))
                )
                displacement_data_initialization.append(
                    self.equation_system.evaluate(self.displacement(matrix))
                )

            if np.any(abs(t - self.time_manager.schedule) < 1e-6):
                # Hitting the checkpoint.
                self._collect_data_for_tests()

        def _collect_data_for_tests(self):
            mdg = self.mdg
            # Extracting indices that identify the injection and the production wells.
            inj_well_index = next(
                well.index
                for well in self.well_network.wells
                if well.tags["well_name"] == "injection_well"
            )
            prod_well_index = next(
                well.index
                for well in self.well_network.wells
                if well.tags["well_name"] == "production_well"
            )

            # Identifying injection and production well subdomains, one well can be
            # defined on multiple subdomains.
            all_wells = [sd for sd in mdg.subdomains() if self.is_well_grid(sd)]
            injection_wells = [
                well
                for well in all_wells
                if well.tags["parent_well_index"] == inj_well_index
            ]
            production_wells = [
                well
                for well in all_wells
                if well.tags["parent_well_index"] == prod_well_index
            ]
            # Fluxes are defined along normal vectors. Below, we check whether fluxes
            # point upwards or downwards along the z axis. Hence, we multiply the flux
            # values by the sign of the z component of the face normal.
            injection_signs = [np.sign(sd.face_normals[2]) for sd in injection_wells]
            production_signs = [np.sign(sd.face_normals[2]) for sd in production_wells]

            def identify_well_fracture(parent_well_index):
                # Identifying the fracture corresponding to this well index.

                # Getting the intersections well-fracture.
                intersections_0d = mdg.subdomains(dim=0)
                # Getting the intersection that corresponds to the this well.
                injection_intersection_0d = [
                    x
                    for x in intersections_0d
                    if x.tags["parent_well_index"] == parent_well_index
                ][0]
                # Getting the corresponding interfaces. It's 3 interfaces: two codim==1
                # (with the well subdomains above and below the intersection) and one
                # codim==2 (with the fracture). The latter is what we need.
                interfaces_of_0d_sd = self.mdg.subdomain_to_interfaces(
                    injection_intersection_0d
                )
                intf_codim_2 = [x for x in interfaces_of_0d_sd if x.codim == 2][0]
                # Getting subdomains of this interface (well intersection and fracture.)
                fracture_and_intersection = self.mdg.interface_to_subdomain_pair(
                    intf_codim_2
                )
                # Keeping only the fracture (it should be one element in this list).
                injection_fractures = [
                    x for x in fracture_and_intersection if x.dim == self.nd - 1
                ]
                assert len(injection_fractures) == 1
                return injection_fractures

            injection_fractures = identify_well_fracture(
                parent_well_index=inj_well_index
            )
            production_fractures = identify_well_fracture(
                parent_well_index=prod_well_index
            )

            # Saving the data for the tests.
            pressure_data_injection_well.append(
                self.equation_system.evaluate(self.pressure(injection_wells))
            )
            temperature_data_injection_well.append(
                self.equation_system.evaluate(self.temperature(injection_wells))
            )
            pressure_data_production_well.append(
                self.equation_system.evaluate(self.pressure(production_wells))
            )
            darcy_flux_data_production_well.append(
                self.equation_system.evaluate(self.darcy_flux(production_wells))
                * np.hstack(production_signs)
            )
            temperature_data_production_well.append(
                self.equation_system.evaluate(self.temperature(production_wells))
            )
            pressure_data_injection_fracture.append(
                self.equation_system.evaluate(self.pressure(injection_fractures))
            )
            darcy_flux_data_injection_well.append(
                self.equation_system.evaluate(self.darcy_flux(injection_wells))
                * np.hstack(injection_signs)
            )
            temperature_data_injection_fracture.append(
                self.equation_system.evaluate(self.temperature(injection_fractures))
            )
            pressure_data_production_fracture.append(
                self.equation_system.evaluate(self.pressure(production_fractures))
            )
            temperature_data_production_fracture.append(
                self.equation_system.evaluate(self.temperature(production_fractures))
            )

    model = ModelForTest(model_params)
    # Use a less strict convergence criterion to speed up the test.
    solver_params = set_solver_params()
    solver_params["nl_convergence_inc_atol"] = 1e-5

    pp.run_time_dependent_model(model, solver_params)

    # MARK: Tests

    # Test 1: Verify that the initialization stage (from schedule[0] to schedule[1])
    # reached the steady state. We check that the last two states during the
    # initialization are close.
    assert len(pressure_data_initialization) >= 2
    np.testing.assert_allclose(
        pressure_data_initialization[-2],
        pressure_data_initialization[-1],
        atol=5e-4,
        rtol=0,
    )
    np.testing.assert_allclose(
        temperature_data_initialization[-2],
        temperature_data_initialization[-1],
        atol=2e-2,
        rtol=0,
    )
    np.testing.assert_allclose(
        displacement_data_initialization[-2],
        displacement_data_initialization[-1],
        atol=1e-5,
        rtol=0,
    )

    # Test 2: Injection starts.

    # Test 2.1: The pressure increases and the temperature decreases in the injection
    # well, as cold water is injected into the reservoir. Injection fluxes should be
    # negative (down the well). Allow tolerance for the flux, due to the cells below the
    # well fracture intersection, which should be very small but can have some numerical
    # noise.
    flux_tolerance = 2e-14
    assert (
        pressure_data_injection_well[1].mean() > pressure_data_injection_well[0].mean()
    )
    assert (
        temperature_data_injection_well[1].mean()
        < temperature_data_injection_well[0].mean()
    )
    assert np.all(darcy_flux_data_injection_well[-1] < flux_tolerance)
    # To avoid masking the test above using a too large tolerance, we verify that the
    # average flux is well below the tolerance.
    assert np.mean(darcy_flux_data_injection_well[-1]) < -flux_tolerance * 10

    # Test 2.2: The production well temperature increases, as hot water enters the well
    # from the reservoir. Production fluxes should be positive (up the well). Pressure
    # can increase or decrease and is not tested.
    assert (
        temperature_data_production_well[1].mean()
        > temperature_data_production_well[0].mean()
    )
    assert np.all(darcy_flux_data_production_well[-1] > -flux_tolerance)
    # To avoid masking the test above using a too large tolerance, we verify that the
    # average flux is well above the tolerance.
    assert np.mean(darcy_flux_data_production_well[-1]) > flux_tolerance * 10

    # Test 2.3: The same for the injection fracture.
    assert (
        pressure_data_injection_fracture[1].mean()
        > pressure_data_injection_fracture[0].mean()
    )
