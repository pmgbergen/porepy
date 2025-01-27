"""Tests for thermoporomechanics.

The positive fracture gap value ensures positive aperture for all simulation. This is
needed to avoid degenerate mass and energy balance equations in the fracture.

"""

from __future__ import annotations

import copy

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids import model_geometries
from porepy.applications.test_utils import well_models
from porepy.applications.test_utils.models import (
    Thermoporomechanics,
    compare_scaled_model_quantities,
    compare_scaled_primary_variables,
)


class NonzeroFractureGapPoromechanics(pp.PorePyModel):
    """Adjust bc values and initial condition."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def initial_condition(self):
        """Set initial condition.

        Set initial displacement compatible with fracture gap for matrix subdomain and
        matrix-fracture interface. Also initialize boundary conditions to fracture gap
        value on top half of the matrix.

        """
        super().initial_condition()
        # Initial pressure equals reference pressure (defaults to zero).
        self.equation_system.set_variable_values(
            self.reference_variable_values.pressure * np.ones(self.mdg.num_subdomain_cells()),
            [self.pressure_variable],
            time_step_index=0,
            iterate_index=0,
        )
        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        # Initial displacement.
        if len(self.mdg.subdomains()) > 1:
            top_cells = sd.cell_centers[1] > self.units.convert_units(0.5, "m")
            vals = np.zeros((self.nd, sd.num_cells))
            vals[1, top_cells] = self.solid.fracture_gap
            self.equation_system.set_variable_values(
                vals.ravel("F"),
                [self.displacement_variable],
                time_step_index=0,
                iterate_index=0,
            )
            # Find mortar cells on the top boundary
            intf = self.mdg.interfaces()[0]
            # Identify by normal vector in sd_primary
            faces_primary = intf.primary_to_mortar_int().tocsr().indices
            switcher = pp.grid_utils.switch_sign_if_inwards_normal(
                sd,
                self.nd,
                faces_primary,
            )

            normals = (switcher * sd.face_normals[: sd.dim].ravel("F")).reshape(
                sd.dim, -1, order="F"
            )
            intf_normals = normals[:, faces_primary]
            top_cells = intf_normals[1, :] < 0

            # Set mortar displacement to zero on bottom and fracture gap value on top
            vals = np.zeros((self.nd, intf.num_cells))
            vals[1, top_cells] = (
                self.solid.fracture_gap
                + self.solid.maximum_elastic_fracture_opening
            )
            self.equation_system.set_variable_values(
                vals.ravel("F"),
                [self.interface_displacement_variable],
                time_step_index=0,
                iterate_index=0,
            )

    def fracture_stress(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Fracture stress on interfaces.

        The "else" mimicks old subtract_p_frac=False behavior, which is
        useful for testing reference pressure.

        Parameters:
            interfaces: List of interfaces where the stress is defined.

        Returns:
            Poromechanical stress operator on matrix-fracture interfaces.

        """
        if not all([intf.dim == self.nd - 1 for intf in interfaces]):
            raise ValueError("Interfaces must be of dimension nd - 1.")

        if getattr(self, "subtract_p_frac", True):
            # Contact traction and fracture pressure.
            return super().fracture_stress(interfaces)
        else:
            # Only contact traction.
            return pp.constitutive_laws.LinearElasticMechanicalStress.fracture_stress(
                self, interfaces
            )

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid source term.

        Add a source term in fractures.

        Parameters:
            subdomains: List of subdomains where the source term is defined.

        Returns:
            Fluid source term operator.

        """
        internal_boundaries = super().fluid_source(subdomains)
        if "fracture_source_value" not in self.params:
            return internal_boundaries

        vals = []
        for sd in subdomains:
            if sd.dim == self.nd:
                vals.append(np.zeros(sd.num_cells))
            else:
                val = self.units.convert_units(
                    self.params["fracture_source_value"], "kg * s ^ -1"
                )
                vals.append(val * np.ones(sd.num_cells))
        fracture_source = pp.wrap_as_dense_ad_array(
            np.hstack(vals), name="fracture_fluid_source"
        )
        return internal_boundaries + fracture_source




def get_variables_poromechanics(
    setup
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Utility function to extract variables from a setup.

    Parameters:
        setup: A setup for a fractured domain.

    Returns:
        Tuple containing the following variables:
            np.ndarray: Displacement values.
            np.ndarray: Pressure values.
            np.ndarray: Fracture pressure values.
            np.ndarray: Displacement jump values.
            np.ndarray: Contact traction values.

    """
    sd = setup.mdg.subdomains(dim=setup.nd)[0]
    u_var = setup.equation_system.get_variables([setup.displacement_variable], [sd])
    u_vals = setup.equation_system.get_variable_values(
        variables=u_var, 
        time_step_index=0
    ).reshape(setup.nd, -1, order="F")

    p_var = setup.equation_system.get_variables(
        [setup.pressure_variable], setup.mdg.subdomains()
    )
    p_vals = setup.equation_system.get_variable_values(
        variables=p_var, time_step_index=0
    )
    p_var = setup.equation_system.get_variables(
        [setup.pressure_variable], setup.mdg.subdomains(dim=setup.nd - 1)
    )
    p_frac = setup.equation_system.get_variable_values(
        variables=p_var, time_step_index=0
    )
    # Fracture
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)
    jump = (
        setup.displacement_jump(sd_frac)
        .value(setup.equation_system)
        .reshape(setup.nd, -1, order="F")
    )
    traction = (
        setup.contact_traction(sd_frac)
        .value(setup.equation_system)
        .reshape(setup.nd, -1, order="F")
    )
    return u_vals, p_vals, p_frac, jump, traction




class TailoredThermoporomechanics(
    NonzeroFractureGapPoromechanics,
    pp.model_boundary_conditions.TimeDependentMechanicalBCsDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsEnergyDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsMassDirNorthSouth,
    Thermoporomechanics,
):
    pass


def create_fractured_setup(
    solid_vals: dict, fluid_vals: dict, params: dict
) -> TailoredThermoporomechanics:
    """Create a setup for a 2d problem with a single fracture.

    Parameters:
        solid_vals: Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        fluid_vals: Dictionary with keys as those in :class:`pp.FluidComponent`
            and corresponding values.
        params: Dictionary with keys as those in params of
            :class:`TailoredThermoporomechanics`.

    Returns:
        setup: Model object for the problem.

    """
    # Instantiate constants and store in params.
    solid_vals["fracture_gap"] = 0.042
    solid_vals["residual_aperture"] = 1e-10
    solid_vals["biot_coefficient"] = 1.0
    solid_vals["thermal_expansion"] = 1e-1
    fluid_vals["compressibility"] = 1
    fluid_vals["thermal_expansion"] = 1e-1
    solid = pp.SolidConstants(**solid_vals)
    fluid = pp.FluidComponent(**fluid_vals)

    default = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"solid": solid, "fluid": fluid},
        "max_iterations": 20,
    }
    default.update(params)
    setup = TailoredThermoporomechanics(default)
    return setup


def get_variables(setup):
    u_vals, p_vals, p_frac, jump, traction = get_variables_poromechanics(setup)
    t_var = setup.equation_system.get_variables(
        [setup.temperature_variable], setup.mdg.subdomains()
    )
    t_vals = setup.equation_system.get_variable_values(
        variables=t_var, time_step_index=0
    )
    t_var = setup.equation_system.get_variables(
        [setup.temperature_variable], setup.mdg.subdomains(dim=setup.nd - 1)
    )
    t_frac = setup.equation_system.get_variable_values(
        variables=t_var, time_step_index=0
    )
    return u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac


def test_thermoporomechanics_model_no_modification():
    """Test that the raw contact thermoporomechanics model with no modifications can be
    run with no error messages.

    Failure of this test would signify rather fundamental problems in the model.

    """
    mod = pp.Thermoporomechanics({})
    pp.run_stationary_model(mod, {})


def test_pull_north_positive_opening():

    setup = create_fractured_setup({}, {}, {"u_north": [0.0, 0.001]})
    pp.run_time_dependent_model(setup)
    u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac = get_variables(setup)

    # All components should be open in the normal direction
    assert np.all(jump[1] > 0)

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero

    # NB: This assumes the contact force is expressed in local coordinates
    assert np.all(np.abs(traction) < 1e-7)

    # Check that the dilation of the fracture yields a negative fracture pressure
    assert np.all(p_frac < -1e-7)
    assert np.all(t_frac < -1e-7)


def test_pull_south_positive_opening():

    setup = create_fractured_setup({}, {}, {"u_south": [0.0, -0.001]})
    pp.run_time_dependent_model(setup)
    u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac = get_variables(setup)

    # All components should be open in the normal direction
    assert np.all(jump[1] > 0)

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero

    # NB: This assumes the contact force is expressed in local coordinates
    assert np.all(np.abs(traction) < 1e-7)

    # Check that the dilation yields a negative pressure and temperature
    assert np.all(p_vals < -1e-7)
    assert np.all(t_vals < -1e-7)


def test_push_north_zero_opening():

    setup = create_fractured_setup({}, {}, {"u_north": [0.0, -0.001]})
    pp.run_time_dependent_model(setup)
    u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac = get_variables(setup)

    # All components should be closed in the normal direction
    assert np.allclose(jump[1], setup.solid.fracture_gap)

    # Contact force in normal direction should be negative
    assert np.all(traction[1] < 0)

    # Compression of the domain yields a (slightly) positive fracture pressure
    assert np.all(p_frac > 1e-10)
    assert np.all(t_frac > 1e-10)


def test_positive_p_frac_positive_opening():
    setup = create_fractured_setup({}, {}, {"fracture_source_value": 0.001})
    pp.run_time_dependent_model(setup)
    _, _, p_frac, jump, traction, _, t_frac = get_variables(setup)

    # All components should be open in the normal direction.
    assert np.all(jump[1] > setup.solid.fracture_gap)

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero.

    # NB: This assumes the contact force is expressed in local coordinates.
    assert np.all(np.abs(traction) < 1e-7)

    # Fracture pressure and temperature are both positive.
    assert np.allclose(p_frac, 4.8e-4, atol=1e-5)
    assert np.allclose(t_frac, 8.3e-6, atol=1e-7)


def test_robin_boundary_flux():
    """Tests model setup with Robin boundary conditions.

    Neumann and Robin boundary values are set by the same method
    (bc_values_"flux_name"). This test ensures that the correct values are assigned to
    the correct domain boundary sides (Robin values to Robin boundaries, Neumann values
    to Neumann boundaries). The test also covers checking if the Dirichlet values are
    assigned to the correct boundary sides. Thus, this test also checks that there is no
    overlap between the three boundary condition types.

    Model setup:
    * West boundary: Robin
    * East boundary: Neumann
    * North and south boundaries: Dirichlet

    Flux types that are tested: Mechanical stress, Fourier flux and Darcy flux.

    """

    class TailoredPoromechanicsRobin(
        pp.test_utils.models.RobinDirichletNeumannConditions,
        Thermoporomechanics,
    ):
        def set_domain(self) -> None:
            self._domain = pp.domains.unit_cube_domain(dimension=2)

    model_params = {
        "meshing_arguments": {"cell_size": 0.5},
        "grid_type": "cartesian",
        "pressure_north": 1e-3,
        "pressure_south": -1e-3,
        "darcy_flux_west": 1e-2,
        "darcy_flux_east": -1e-2,
        "fourier_flux_west": 2e-2,
        "fourier_flux_east": -2e-2,
        "mechanical_stress_west": 3e-2,
        "mechanical_stress_east": -3e-2,
    }

    model = TailoredPoromechanicsRobin(model_params)
    pp.run_time_dependent_model(model)

    subdomain = model.mdg.subdomains(dim=model.nd, return_data=True)[0][0]

    bc_operators = {
        "darcy_flux": model.combine_boundary_operators_darcy_flux,
        "fourier_flux": model.combine_boundary_operators_fourier_flux,
        "mechanical_stress": model.combine_boundary_operators_mechanical_stress,
    }

    # Create dictionary of evaluated boundary operators in bc_operators
    values = {
        key: operator([subdomain]).value(model.equation_system)
        for key, operator in bc_operators.items()
    }

    # Reshape mechanical stress values
    values["mechanical_stress"] = values["mechanical_stress"].reshape(
        (model.nd, subdomain.num_faces), order="F"
    )

    # Get boundary sides and assert boundary condition values
    bounds = model.domain_boundary_sides(subdomain)

    assert np.allclose(
        values["darcy_flux"][bounds.west], model_params["darcy_flux_west"]
    )
    assert np.allclose(
        values["darcy_flux"][bounds.east], model_params["darcy_flux_east"]
    )

    assert np.allclose(
        values["fourier_flux"][bounds.west], model_params["fourier_flux_west"]
    )
    assert np.allclose(
        values["fourier_flux"][bounds.east], model_params["fourier_flux_east"]
    )

    assert np.allclose(
        values["mechanical_stress"][0][bounds.west],
        model_params["mechanical_stress_west"],
    )
    assert np.allclose(
        values["mechanical_stress"][0][bounds.east],
        model_params["mechanical_stress_east"],
    )

    # Final check to see that the Dirichlet values are also assigned as expected. The
    # different bc types should not overlap. If all these tests pass, that suggest there
    # is no such overlap.

    # First constructing the pressure operator and evaluating it, then finding the
    # indices of the north and south boundaries in the pressure_values array, before
    # finally asserting that the values are correct.
    bg = model.mdg.subdomain_to_boundary_grid(subdomain)
    pressure_operator = model.pressure([bg])
    pressure_values = pressure_operator.value(model.equation_system)

    ind_north = np.nonzero(np.isin(bounds.all_bf, np.where(bounds.north)[0]))[0]
    ind_south = np.nonzero(np.isin(bounds.all_bf, np.where(bounds.south)[0]))[0]

    assert np.allclose(pressure_values[ind_north], model_params["pressure_north"])
    assert np.allclose(pressure_values[ind_south], model_params["pressure_south"])


class ThermoporomechanicsWell(
    well_models.OneVerticalWell,
    model_geometries.OrthogonalFractures3d,
    well_models.BoundaryConditionsWellSetup,
    pp.Poromechanics,
):
    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.units.convert_units(1, "m")
        h = 0.1 * ls
        mesh_sizes = {
            "cell_size": h,
        }
        return mesh_sizes


def test_thermoporomechanics_well():
    """Test that the thermoporomechanics model runs without errors."""
    # These parameters hopefully yield a relatively easy problem
    model_params = {
        "fracture_indices": [2],
        "well_flux": -1e-2,
    }
    setup = ThermoporomechanicsWell(model_params)
    pp.run_time_dependent_model(setup)


# test_positive_p_frac_positive_opening()
test_thermoporomechanics_well()