"""Tests for wells in the models.

The module contains a setup for testing well-related functionality in the models.


"""
from __future__ import annotations

import numpy as np

# import numpy as np
import pytest

import porepy as pp

from . import setup_utils


@pytest.mark.parametrize(
    "base_model",
    [
        (pp.fluid_mass_balance.SinglePhaseFlow),
        (pp.mass_and_energy_balance.MassAndEnergyBalance),
    ],
)
@pytest.mark.parametrize("num_wells", [1])
def test_run_models(base_model, num_wells):
    """Test that the model actually runs."""

    solid = pp.SolidConstants({"residual_aperture": 0.01, "permeability": 42})
    params = {
        "material_constants": {"solid": solid},
        "fracture_indices": [2],
        "num_wells": num_wells,
        "grid_type": "simplex",
    }

    class LocalThermoporomechanics(
        setup_utils.WellGeometryMixin, setup_utils.OrthogonalFractures3d, base_model
    ):
        pass

    setup = LocalThermoporomechanics(params)
    pp.run_time_dependent_model(setup, params)


class OneVerticalWell:
    domain: pp.Domain
    """Domain for the model."""

    def set_well_network(self) -> None:
        """Assign well network class."""
        points = np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 1]])
        mesh_size = self.solid.convert_units(1 / 10.0, "m")
        self.well_network = pp.WellNetwork3d(
            domain=self.domain,
            wells=[pp.Well(points)],
            parameters={"mesh_size": mesh_size},
        )

    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.solid.convert_units(1, "m")
        h = 0.15 * ls
        mesh_sizes = {
            "cell_size_fracture": h,
            "cell_size_boundary": h,
            "cell_size_min": 0.2 * h,
        }

        return mesh_sizes

    def grid_type(self) -> str:
        return "simplex"


class BoundaryConditionsWellSetup(
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow
):
    def _bc_type(self, sd: pp.Grid, well_cond) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        cond = well_cond if sd.dim == 1 else "dir"

        domain_sides = self.domain_boundary_sides(sd)
        # Define boundary condition on faces
        return pp.BoundaryCondition(sd, domain_sides.top + domain_sides.bottom, cond)

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type(sd, "neu")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries,
        with a constant value of 0 unless fluid's reference pressure is changed.

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        value = self.fluid.convert_units(
            self.params.get("well_flux", -1), "kg * m ^ 3 * s ^ -1"
        )

        vals_loc = np.zeros(boundary_grid.num_cells)
        if boundary_grid.dim == 0:
            domain_sides = self.domain_boundary_sides(boundary_grid)
            # Inflow for the top boundary of the well.
            vals_loc[domain_sides.top] = value
        return vals_loc

    def bc_type_mobrho(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type(sd, "dir")

    def bc_type_enthalpy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for enthalpy.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type(sd, "dir")

    def bc_type_fourier(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Fourier flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type(sd, "neu")


class TailoredPermeability(pp.constitutive_laws.CubicLawPermeability):
    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        This function is an extension of the CubicLawPermeability class which includes
        well permeability.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability values.

        """
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        matrix = [sd for sd in subdomains if sd.dim == self.nd]
        fractures_and_intersections: list[pp.Grid] = [
            sd
            for sd in subdomains
            if sd.dim < self.nd and ("parent_well_index" not in sd.tags)
        ]
        wells = [sd for sd in subdomains if "parent_well_index" in sd.tags]

        permeability = (
            projection.cell_prolongation(matrix) @ self.matrix_permeability(matrix)
            + projection.cell_prolongation(fractures_and_intersections)
            @ self.cubic_law_permeability(fractures_and_intersections)
            + projection.cell_prolongation(wells) @ self.well_permeability(wells)
        )
        return permeability

    def well_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability values.

        """
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_ad_array(1, size, name="well permeability")
        return permeability


class FlowModel(
    OneVerticalWell,
    setup_utils.OrthogonalFractures3d,
    BoundaryConditionsWellSetup,
    TailoredPermeability,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    pass


def test_incompressible_pressure_values():
    """One central vertical well, one horizontal fracture.

    The well is placed such that the pressure at the fracture is log distributed, up to
    the effect of the low-permeable matrix.

    """
    params = {
        # Set impermeable matrix
        "material_constants": {
            "solid": pp.SolidConstants({"permeability": 1e-6 / 4, "well_radius": 0.01})
        },
        # Use only the horizontal fracture of OrthogonalFractures3d
        "fracture_indices": [2],
    }

    setup = FlowModel(params)
    pp.run_time_dependent_model(setup, params)
    # Check that the matrix pressure is close to linear in z
    matrix = setup.mdg.subdomains(dim=3)[0]
    matrix_pressure = setup.pressure([matrix]).evaluate(setup.equation_system).val
    dist = np.absolute(matrix.cell_centers[2, :] - 0.5)
    p_range = np.max(matrix_pressure) - np.min(matrix_pressure)
    expected_p = p_range * (0.5 - dist) / 0.5
    diff = expected_p - matrix_pressure
    # This is a very loose tolerance, but the pressure is not exactly linear due to the
    # the fracture pressure distribution and grid effects.
    assert np.max(np.abs(diff / p_range)) < 1e-1
    # With a matrix permeability of 1/4e-6, the pressure drop for a unit flow rate is
    # (1/2**2) / 1/4e-6 = 1e6: 1/2 for splitting flow in two directions (top and
    # bottom), and 1/2 for the distance from the fracture to the boundary.
    assert np.isclose(np.max(matrix_pressure), 1e6, rtol=1e-1)
    # In the fracture, check that the pressure is log distributed
    fracs = setup.mdg.subdomains(dim=2)
    fracture_pressure = setup.pressure(fracs).evaluate(setup.equation_system).val
    sd = fracs[0]
    injection_cell = sd.closest_cell(np.atleast_2d([0.5, 0.5, 0.5]).T)
    # Check that the injection cell is the one with the highest pressure
    assert np.argmax(fracture_pressure) == injection_cell
    pt = sd.cell_centers[:, injection_cell]
    dist = pp.distances.point_pointset(pt, sd.cell_centers)
    min_p = 1e6
    scale_dist = np.exp(-np.max(dist))
    perm = 1e-3
    expected_p = min_p + 1 / (4 * np.pi * perm) * np.log(dist / scale_dist)
    assert np.isclose(np.min(fracture_pressure), min_p, rtol=1e-2)
    wells = setup.mdg.subdomains(dim=0)
    well_pressure = setup.pressure(wells).evaluate(setup.equation_system).val

    # Check that the pressure drop from the well to the fracture is as expected The
    # Peacmann well model is: u = 2 * pi * k * h * (p_fracture - p_well) / ( ln(r_e /
    # r_well) + S) We assume r_e = 0.20 * meas(injection_cell). h is the aperture, and
    # the well radius is 0.01 and skin factor 0. With a permeability of 0.1^2/12
    # and unit flow rate, the pressure drop is p_fracture - p_well = 1 / (2 * pi * 1e-3
    # * 0.1) * ln(0.2 / 0.1)
    k = 1e-2 / 12
    r_e = np.sqrt(sd.cell_volumes[injection_cell]) * 0.2
    dp_expected = 1 / (2 * np.pi * k * 0.1) * np.log(r_e / 0.01)
    dp = well_pressure[0] - fracture_pressure[injection_cell]
    assert np.isclose(dp, dp_expected, rtol=1e-4)


class THModel(
    OneVerticalWell,
    setup_utils.OrthogonalFractures3d,
    BoundaryConditionsWellSetup,
    TailoredPermeability,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    pass


def test_energy_conservation():
    """One central vertical well, one horizontal fracture.

    The well is placed such that the pressure at the fracture is log distributed, up to
    the effect of the low-permeable matrix.

    """
    # We want low pressures, to ensure energy is not dominated by -p in fluid part.
    dt = 1e-4
    params = {
        # Set impermeable matrix
        "material_constants": {
            "solid": pp.SolidConstants(
                {
                    "specific_heat_capacity": 1e0,  # Ensure energy stays in domain.
                    "well_radius": 0.02,
                    "residual_aperture": 1.0,  # Ensure high permeability in fracture.
                    "thermal_conductivity": 1e-6,
                    "permeability": 1e4,  # Reduce pressure effect
                    "normal_permeability": 1e4,
                }
            ),
            "fluid": pp.FluidConstants(
                {
                    "specific_heat_capacity": 1e0,
                    "thermal_conductivity": 1e-6,
                    "normal_thermal_conductivity": 1e-6,
                }
            ),
        },
        # Use only the horizontal fracture of OrthogonalFractures3d
        "fracture_indices": [2],
        "time_manager": pp.TimeManager(schedule=[0, dt], dt_init=dt, constant_dt=True),
        "grid_type": "cartesian",
    }

    setup = THModel(params)
    pp.run_time_dependent_model(setup, params)
    # Check that the total enthalpy equals the injected one.
    in_val = 1e7
    u_expected = in_val * dt

    sds = setup.mdg.subdomains()
    u = setup.volume_integral(setup.total_internal_energy(sds), sds, 1)
    h_f = setup.fluid_enthalpy(sds) * setup.porosity(sds)
    h_s = setup.solid_enthalpy(sds) * (pp.ad.Scalar(1) - setup.porosity(sds))
    h = setup.volume_integral(h_f + h_s, sds, 1)
    u_val = np.sum(u.evaluate(setup.equation_system).val)
    h_val = np.sum(h.evaluate(setup.equation_system).val)
    assert np.isclose(u_val, u_expected, rtol=1e-3)
    # u and h should be close with our setup
    assert np.isclose(u_val, h_val, rtol=1e-3)
    well_intf = setup.mdg.interfaces(codim=2)
    well_enthalpy_flux = (
        setup.well_enthalpy_flux(well_intf).evaluate(setup.equation_system).val
    )
    well_fluid_flux = setup.well_flux(well_intf).evaluate(setup.equation_system).val
    h_well = (
        setup.fluid_enthalpy(setup.mdg.subdomains(dim=0))
        .evaluate(setup.equation_system)
        .val
    )
    # The well enthalpy flux should be equal to well enthalpy times well fluid flux
    assert np.isclose(well_enthalpy_flux, h_well * well_fluid_flux, rtol=1e-10)
    # Well fluid flux should equal the injected fluid. Minus for convention of interface
    # fluxes from higher to lower domain.
    assert np.isclose(well_fluid_flux, -1, rtol=1e-10)


class Poromechanics(
    OneVerticalWell,
    setup_utils.OrthogonalFractures3d,
    BoundaryConditionsWellSetup,
    pp.poromechanics.Poromechanics,
):
    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.solid.convert_units(1, "m")
        h = 0.5 * ls
        mesh_sizes = {
            "cell_size": h,
        }
        return mesh_sizes


def test_poromechanics():
    """Test that the poromechanics model runs without errors."""
    # These parameters hopefully yield a relatively easy problem
    params = {
        "fracture_indices": [2],
        "well_flux": -1e-2,
    }
    setup = Poromechanics(params)
    pp.run_time_dependent_model(setup, {})


class Thermoporomechanics(
    OneVerticalWell,
    setup_utils.OrthogonalFractures3d,
    BoundaryConditionsWellSetup,
    pp.thermoporomechanics.Thermoporomechanics,
):
    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.solid.convert_units(1, "m")
        h = 0.5 * ls
        mesh_sizes = {
            "cell_size": h,
        }
        return mesh_sizes


def test_thermoporomechanics():
    """Test that the thermoporomechanics model runs without errors."""
    # These parameters hopefully yield a relatively easy problem
    params = {
        "fracture_indices": [2],
        "well_flux": -1e-2,
        "well_enthalpy": 1e0,
    }

    setup = Thermoporomechanics(params)
    pp.run_time_dependent_model(setup, {})
