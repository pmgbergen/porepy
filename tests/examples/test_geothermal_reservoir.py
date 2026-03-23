"""
Test functionalities in the example case of the geomthermal reservoir.

"""

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
)
from porepy.numerics.nonlinear import line_search


class geothermal_model_neu(
    well_models.OneVerticalWell,
    porepy.applications.md_grids.model_geometries.CubeDomainOrthogonalFractures,
    NeumannWellBCsFirstTimeInterval,
    pp.Poromechanics,
):
    pass


@pytest.fixture
def neuBC_model():
    model = geothermal_model_neu()
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


class geomhermal_model_well(
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
    model = geomhermal_model_well(params)
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


class geothermal_model_mechanics(
    well_models.OneVerticalWell,
    porepy.applications.md_grids.model_geometries.CubeDomainOrthogonalFractures,
    BoundaryConditionsMechanicsNeumann,
    pp.Poromechanics,
):
    pass


@pytest.fixture
def mechcanics_bc_model():
    model = geothermal_model_mechanics()
    model.prepare_simulation()
    return model


def test_mechanics_bcs_neumann(mechcanics_bc_model):
    """
    Test the boundary conditions of mechanics.
    """
    model = mechcanics_bc_model
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

    # The model setup is mostly copied from porepy/examples/geothermal_reservoir.py

    dt_init = 3 * pp.YEAR
    # 6 * dt_init is enough to equilibrate the system. However, exactly 6 produces a
    # bug with the time_manager, which does not adjust the schedule. Therefore, using
    # 6.1 instead. This can be reconcidered by just 6 later, when the time_manager works
    # more robustly.
    INITIALIZATION_LENGTH = 6.1
    schedule = np.array(
        [
            0,  # Initialization, wells are off.
            dt_init * INITIALIZATION_LENGTH,  # Initialization done, wells are pumping.
            dt_init * INITIALIZATION_LENGTH + pp.HOUR,  # Simulation ends.
        ]
    )

    # Injection pressure schedule, its size == schedule.size
    injection_pressures = [1e5, 5e6, 9e6]  # [Pa]

    # Adjust solid values, while using default values for water.
    solid_values = pp.solid_values.basalt
    solid_values.update(
        {
            "dilation_angle": 0.1,  # [rad]
            "normal_permeability": 1.0e-10,  # [m^2]
            "residual_aperture": 1e-3,  # [m]
            "well_radius": 0.1,  # [m]
        }
    )
    # Define domain sizes (x, y, z) and fracture size.
    length_scale = 1e3  # [m]
    fracture_size = 0.15  # [-], fraction of length_scale
    domain_sizes = np.array(
        [1.0 * length_scale, 1.0 * length_scale, 1.0 * length_scale]
    )  # [m]
    # Define model parameters.
    model_params = {
        "darcy_flux_discretization": "tpfa",
        "fourier_flux_discretization": "tpfa",
        # Set time manager.
        "time_manager": pp.TimeManager(
            schedule=schedule,
            dt_init=dt_init,
            constant_dt=False,
            dt_min_max=(0.1 * pp.HOUR, max(pp.YEAR, dt_init)),
            iter_optimal_range=(6, 10),  # Allow more iterations than default.
            iter_relax_factors=(0.5, 1.8),  # More aggressive relaxation
        ),
        # Set physical parameters.
        "lithostatic_stress_multipliers": np.array([0.8, 1.2, 1.0]),
        "injection_well_temperatures": 250.00,
        "injection_well_pressures": injection_pressures,
        "production_well_temperatures": 300.0,
        "production_well_pressures": pp.ATMOSPHERIC_PRESSURE,  # = 1.01325e5 Pa
        "material_constants": {
            "solid": pp.SolidConstants(**solid_values),  # type: ignore[arg-type]
            "fluid": pp.FluidComponent(
                **pp.fluid_values.water,
            ),  # type: ignore[arg-type]
            "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),
        },
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=300, pressure=1e6
        ),  # type: ignore[arg-type]
        "units": pp.Units(m=1.0, kg=1.0e5, K=1.0),
        # Set geometry and meshing related parameters.
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": length_scale / 5.0,
            "cell_size_fracture": fracture_size * length_scale,
        },
        "fracture_params": {  # Other options are available in the geometry mixin.
            "fracture_major_axes": np.array((fracture_size, fracture_size * 1.2)),
            "num_points": np.array((9, 8)),  # Number of points to define each fracture
            "dip_angles": np.array((np.pi / 4, np.pi / 2)),  # Slanted and vertical
        },
        "domain_sizes": domain_sizes,
        # Line search: Scale the indicator used for the local_line_search (see below)
        # adaptively to increase robustness.
        "adaptive_indicator_scaling": 1,
        # Set folder name for results.
        "folder_name": "geothermal_reservoir",
    }

    # Data saved in the simulation for the test.
    pressure_data_initialization = []
    temperature_data_initialization = []
    displacement_data_initialization = []
    pressure_data_injection_well = []
    temperature_data_injection_well = []
    pressure_data_production_well = []
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
            temperature_data_production_well.append(
                self.equation_system.evaluate(self.temperature(production_wells))
            )
            pressure_data_injection_fracture.append(
                self.equation_system.evaluate(self.pressure(injection_fractures))
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
    solver_params = {
        "prepare_simulation": True,
        "max_iterations": 25,  # Max iterations of a nonlinear solver (Newton)
        "nl_divergence_tol": 1e20,
        "nl_convergence_inc_atol": 1e-7,  # Increment norm
        "nl_convergence_res_atol": 1e-7,  # Residual norm
        "nonlinear_solver": line_search.ConstraintLineSearchNonlinearSolver,
        "global_line_search": 0,
        "local_line_search": 1,
    }

    pp.run_time_dependent_model(model, solver_params)

    # MARK: Tests

    # Test 1: Verify that the initialization stage (from schedule[0] to schedule[1])
    # reached the steady state. We check that the last two states during the
    # initialization are close.
    assert len(pressure_data_initialization) >= 2

    np.testing.assert_allclose(
        pressure_data_initialization[-2],
        pressure_data_initialization[-1],
        atol=1e-3,
        rtol=0,
    )
    np.testing.assert_allclose(
        temperature_data_initialization[-2],
        temperature_data_initialization[-1],
        atol=1e-2,
        rtol=0,
    )
    np.testing.assert_allclose(
        displacement_data_initialization[-2],
        displacement_data_initialization[-1],
        atol=1e-5,
        rtol=0,
    )

    # Test 2.1: Injection starts.

    # We check that the pressure increases and the temperature decreases in the
    # injection well.
    assert (
        pressure_data_injection_well[1].mean() > pressure_data_injection_well[0].mean()
    )
    assert (
        temperature_data_injection_well[1].mean()
        < temperature_data_injection_well[0].mean()
    )

    # Test 2.2: The opposite for the production well.
    assert (
        pressure_data_production_well[1].mean()
        < pressure_data_production_well[0].mean()
    )
    assert (
        temperature_data_production_well[1].mean()
        > temperature_data_production_well[0].mean()
    )

    # Test 2.3: The same for the injection fracture.
    assert (
        pressure_data_injection_fracture[1].mean()
        > pressure_data_injection_fracture[0].mean()
    )
    assert (
        temperature_data_injection_fracture[1].mean()
        < temperature_data_injection_fracture[0].mean()
    )

    # Test 2.4: Pressure should decrease for the production fracture.
    assert (
        pressure_data_production_fracture[1].mean()
        < pressure_data_production_fracture[0].mean()
    )
