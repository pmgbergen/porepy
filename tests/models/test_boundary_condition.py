"""This file is testing the functionality of `pp.BoundaryConditionMixin`.

"""

from typing import Callable, Sequence

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.models import MassBalance as MassBalance_
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)


class CustomBoundaryCondition(pp.BoundaryConditionMixin):
    """We define a custom dummy boundary condition.

    Neumann values are explicitly set, they are time dependent.
    Dirichlet values are equal to density on a boundary grid.

    """

    custom_bc_neumann_key = "custom_bc_neumann"

    fluid_density: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def update_all_boundary_conditions(self) -> None:
        super().update_all_boundary_conditions()

        self.update_boundary_condition(
            name=self.custom_bc_neumann_key, function=self.bc_values_neumann
        )

    def bc_values_neumann(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Returns values on the whole boundary. We implicitly rely on the filter that
        sets zeros at the cells related to Dirichlet condition.

        Note: the values are time dependent.

        """
        t = self.time_manager.time
        return np.arange(boundary_grid.num_cells) * boundary_grid.parent.dim * t

    def bc_type_dummy(self, subdomain: pp.Grid) -> pp.BoundaryCondition:
        """The north boundary is Dirichlet, the remainder is Neumann."""
        sides = self.domain_boundary_sides(subdomain)
        return pp.BoundaryCondition(sd=subdomain, faces=sides.north, cond="dir")

    def create_dummy_ad_boundary_condition(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        op = lambda bgs: self.create_boundary_operator(
            name=self.custom_bc_neumann_key, domains=bgs
        )
        return self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.fluid_density,
            neumann_operator=op,
            robin_operator=op,
            bc_type=self.bc_type_dummy,
            name="boundary_condition_dummy",
            dim=1,
        )


class MassBalance(CustomBoundaryCondition, MassBalance_):
    pass


@pytest.mark.parametrize("t_end", [2, 3])
def test_boundary_condition_mixin(t_end: int):
    """We create a custom boundary condition operator and test that:
    1) The values are set correctly.
    2) Dirichlet values do not intersect Neumann values due to the filters.
    3) Previous timestep values are set correctly for the time dependent Neumann.

    """
    setup = MassBalance()
    setup.time_manager.dt = 1
    setup.time_manager.time_final = t_end
    pp.run_time_dependent_model(setup, params={})

    subdomains = setup.mdg.subdomains()

    for sd in subdomains:
        bc_type = setup.bc_type_dummy(sd)
        bc_operator = setup.create_dummy_ad_boundary_condition([sd])
        bc_val = bc_operator.value(setup.equation_system)

        # Testing the Dirichlet values. They should be equal to the fluid density.
        expected_val = setup.fluid.density()
        assert np.allclose(bc_val[bc_type.is_dir], expected_val)
        assert not np.allclose(bc_val[bc_type.is_neu], expected_val)

        # Testing the Neumann values.
        bg = setup.mdg.subdomain_to_boundary_grid(sd)
        assert bg is not None
        expected_val = np.arange(bg.num_cells) * bg.parent.dim * t_end
        # Projecting the expected value to the subdomain.
        expected_val = bg.projection().T @ expected_val
        assert np.allclose(bc_val[bc_type.is_neu], expected_val[bc_type.is_neu])

        # Testing previous timestep.
        bc_val_prev_ts = bc_operator.previous_timestep().value(setup.equation_system)
        expected_val = np.arange(bg.num_cells) * bg.parent.dim * (t_end - 1)
        # Projecting the expected value to the subdomain.
        expected_val = bg.projection().T @ expected_val
        assert np.allclose(bc_val_prev_ts[bc_type.is_neu], expected_val[bc_type.is_neu])


"""Here follows mixins related to testing of Robin limit cases, and eventually the test itself. """


class BCValuesLeftSide:
    """Boundary values for primary variables on west and east boundary.

    Used for:
    * Momentum balance
    * Mass and energy balance.

    """

    def not_dir_inds(self, g):
        bounds = self.domain_boundary_sides(g)
        return bounds.north + bounds.south

    def dir_inds(self, g):
        bounds = self.domain_boundary_sides(g)
        return bounds.west + bounds.east

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns displacement values in the x-direction of the west and east
        boundary."""
        values = np.zeros((self.nd, bg.num_cells))
        values[0, self.dir_inds(bg)] = 42
        return values.ravel("F")

    def _bc_values_scalar(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns scalar values on the west boundary."""
        values = np.zeros(bg.num_cells)
        values[self.dir_inds(bg)] = 42
        return values

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns pressure values on the west boundary."""
        return self._bc_values_scalar(bg=bg)

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns temperature values on the west boundary."""
        return self._bc_values_scalar(bg=bg)


class BCRobin:
    """Set Dirichlet and Robin for momentum balance and mass and energy balance.

    Sets Dirichlet on the west and east boundaries, and Robin on the top and bottom
    boundaries. The value of the Robin weight is determined from the parameter "alpha"
    in the params dictionary. This class also sets Robin boundary values.

    This class is common for all the model classes that enters into testing Robin limit
    cases.

    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Sets Robin conditions for the test setup."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "rob")
        bc.is_rob[:, self.dir_inds(sd)] = False
        bc.is_dir[:, self.dir_inds(sd)] = True

        alpha = self.params["alpha"]

        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        bc.robin_weight = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F") * alpha
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on the west boundary, and Robin with alpha = 0 on
        all other boundaries.

        """
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.all_bf, "rob")
        bc.is_rob[self.dir_inds(sd)] = False
        bc.is_dir[self.dir_inds(sd)] = True

        bc.robin_weight = np.ones(sd.num_faces) * self.params["alpha"]
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)


class BCNeumannReference:
    """Set Dirichlet and Neumann for momentum balance and mass and energy balance.

    Sets Dirichlet on the west and east boundaries, and Neumann on top and bottom.

    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Assigns Neumann and Dirichlet boundaries for the Neumann reference setup."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "neu")
        bc.is_neu[:, self.dir_inds(sd)] = False
        bc.is_dir[:, self.dir_inds(sd)] = True
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on the west boundary, and Neumann on all other
        boundaries.

        """

        bc = pp.BoundaryCondition(sd, self.dir_inds(sd), "dir")
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)


class BCValuesFlux:
    def bc_values_fourier_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns Fourier flux values on top, bottom and right boundary."""
        values = np.zeros((bg.num_cells))
        values[self.not_dir_inds(bg)] = 24
        return values.ravel("F")

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns Darcy flux values on top, bottom and right boundary."""
        values = np.zeros((bg.num_cells))
        values[self.not_dir_inds(bg)] = 24
        return values.ravel("F")

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns stress values on the non-Dirichlet boundaries for the test setup."""
        values = np.zeros((self.nd, bg.num_cells))
        inds = self.not_dir_inds(bg)
        values[0, inds] = 24
        return values.ravel("F")


class BCDirichletReference:
    """Set all Dirichlet boundaries for momentum balance and mass and energy balance."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Assigns Dirichlet boundaries on all domain boundary sides."""
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "dir")
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on all boundaries.

        """
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, bounds.all_bf, "dir")
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns displacement values in the x-direction of the top and bottom boundary
        sides."""
        values = super().bc_values_displacement(bg=bg)
        values = values.reshape((self.nd, bg.num_cells), order="F")
        alpha = self.params["alpha"]
        inds = self.not_dir_inds(bg)
        volumes = bg.cell_volumes[inds]
        values[0, inds] = 24 / (volumes * alpha)
        return values.ravel("F")

    def _bc_values_scalar(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Set the values for scalar fields.

        Parameters:
            bg: Boundary grid.

        Returns:
            np.ndarray: Boundary values.

        """
        # Call super to get values for Dirichlet boundaries.
        values = super()._bc_values_scalar(bg)
        # Set values for non-Dirichlet boundaries. Scale by alpha and volume to
        # counteract the scaling implicit in Robin conditions.
        volumes = bg.cell_volumes[self.not_dir_inds(bg)]
        values[self.not_dir_inds(bg)] = 24 / (volumes * self.params["alpha"])
        return values


class CommonMassEnergyBalance(
    SquareDomainOrthogonalFractures,
    BCValuesLeftSide,
    BCValuesFlux,
    pp.models.mass_and_energy_balance.MassAndEnergyBalance,
):
    """Base mass and energy balance setup.

    The setup in this class is common for the reference class for mass and energy
    balance and for the "test" class for mass and energy balance. The "test" class is
    the class which represents a problem setup with Robin boundaries.

    """


class MassAndEnergyBalanceRobin(BCRobin, CommonMassEnergyBalance):
    """Mass and energy balance with Robin conditions on top, bottom and right boundary."""


class CommonMomentumBalance(
    SquareDomainOrthogonalFractures,
    BCValuesLeftSide,
    BCValuesFlux,
    pp.models.momentum_balance.MomentumBalance,
):
    """Base momentum balance setup.

    The setup in this class is common for the reference class for momentum balance and
    for the "test" class for momentum balance. The "test" class is the class which
    represents a problem setup with Robin boundaries.

    """


class MomentumBalanceRobin(BCRobin, CommonMomentumBalance):
    """Momentum balance with Robin conditions on top and bottom boundary."""


def run_model(balance_class, alpha):
    params = {
        "times_to_export": [],
        "fracture_indices": [],
        "meshing_arguments": {"cell_size": 0.5},
    }

    params["alpha"] = alpha
    instance = balance_class(params)
    pp.run_time_dependent_model(instance, params)
    sd = instance.mdg.subdomains(dim=2)[0]

    try:
        displacement = instance.displacement([sd]).value(instance.equation_system)
        return {"displacement": displacement}
    except AttributeError:
        pressure = instance.pressure([sd]).value(instance.equation_system)
        temperature = instance.temperature([sd]).value(instance.equation_system)
        return {"temperature": temperature, "pressure": pressure}


# Parameterize the test function with the necessary balance types and conditions
@pytest.mark.parametrize(
    "rob_class, reference_class, alpha",
    [
        (MomentumBalanceRobin, CommonMomentumBalance, 0),
        (MassAndEnergyBalanceRobin, CommonMassEnergyBalance, 0),
        (MassAndEnergyBalanceRobin, CommonMassEnergyBalance, 1e8),
        (MomentumBalanceRobin, CommonMomentumBalance, 1e8),
    ],
)
def test_robin_limit_case(rob_class, reference_class, alpha):
    """Test Robin limit cases.

    The Robin conditions are implemented on the form: sigma * n + alpha * u = G. That
    means that setting Robin conditions with alpha = 0 should correspond to setting
    Neumann conditions. For large alpha (alpha -> \infty), the Robin conditions should
    correspond to Dirichlet conditions.

    We test this for momentum balance and mass and energy balance.

    Common for all model setups is that all of them have Dirichlet conditions on the
    west an east boundaries.

    The model class setups with documentation are further up in this document.

    """
    if alpha > 0:
        reference_bc_class = BCDirichletReference
    elif alpha == 0:
        reference_bc_class = BCNeumannReference

    class LocalReference(reference_bc_class, reference_class):
        """Reference class with the correct reference boundary types."""

    rob_results = run_model(rob_class, alpha)
    reference_results = run_model(LocalReference, alpha)

    assert all(
        np.allclose(rob_results[key], reference_results[key], atol=1e-7)
        for key in rob_results.keys()
    )
