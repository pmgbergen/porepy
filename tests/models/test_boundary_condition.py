"""This file is testing the functionality of `pp.BoundaryConditionMixin`."""

from typing import Sequence

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.applications.test_utils.models import MassBalance as MassBalance_
from porepy.models.momentum_balance import MomentumBalance


class CustomBoundaryCondition(pp.PorePyModel):
    """We define a custom dummy boundary condition.

    Neumann values are explicitly set, they are time dependent.
    Dirichlet values are equal to density on a boundary grid.

    """

    custom_bc_neumann_key = "custom_bc_neumann"

    def update_all_boundary_conditions(self) -> None:
        super().update_all_boundary_conditions()

        self.update_boundary_condition(
            name=self.custom_bc_neumann_key, function=self.bc_values_neumann
        )

    def bc_values_neumann(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Returns values on the whole boundary. We implicitly rely on the filter that
        sets zeros at the cells related to Dirichlet condition.

        Note: the values are time dependent.

        """
        t = self.time_manager.time
        return np.arange(bg.num_cells) * bg.parent.dim * t

    def bc_type_dummy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """The north boundary is Dirichlet, the remainder is Neumann."""
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd=sd, faces=domain_sides.north, cond="dir")

    def create_dummy_ad_boundary_condition(
        self, subdomains: Sequence[pp.Grid]
    ) -> pp.ad.Operator:
        op = lambda bgs: self.create_boundary_operator(
            name=self.custom_bc_neumann_key, domains=bgs
        )
        return self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.fluid.density,
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
    model = MassBalance(
        {
            "times_to_export": [],  # Suppress output for tests
        }
    )
    model.time_manager.dt = 1
    model.time_manager.time_final = t_end
    pp.run_time_dependent_model(model)

    subdomains = model.mdg.subdomains()

    for sd in subdomains:
        bc_type = model.bc_type_dummy(sd)
        bc_operator = model.create_dummy_ad_boundary_condition([sd])
        bc_val = model.equation_system.evaluate(bc_operator)

        # Testing the Dirichlet values. They should be equal to the fluid density.
        expected_val = model.fluid.reference_component.density
        assert np.allclose(bc_val[bc_type.is_dir], expected_val)
        assert not np.allclose(bc_val[bc_type.is_neu], expected_val)

        # Testing the Neumann values.
        bg = model.mdg.subdomain_to_boundary_grid(sd)
        assert bg is not None
        expected_val = np.arange(bg.num_cells) * bg.parent.dim * t_end
        # Projecting the expected value to the subdomain.
        expected_val = bg.projection().T @ expected_val
        assert np.allclose(bc_val[bc_type.is_neu], expected_val[bc_type.is_neu])

        # Testing previous timestep.
        bc_val_prev_ts = model.equation_system.evaluate(bc_operator.previous_timestep())
        expected_val = np.arange(bg.num_cells) * bg.parent.dim * (t_end - 1)
        # Projecting the expected value to the subdomain.
        expected_val = bg.projection().T @ expected_val
        assert np.allclose(bc_val_prev_ts[bc_type.is_neu], expected_val[bc_type.is_neu])


"""Here follows mixins related to testing of Robin limit cases, and eventually the test itself. """


class BCValuesDirichletIndices(pp.PorePyModel):
    """Boundary values for primary variables on Dirichlet boundaries.

    Used for:
    * Momentum balance
    * Mass and energy balance.

    """

    def rob_inds(self, sd) -> np.ndarray:
        """Indices for the non-Dirichlet boundaries for test.

        The Robin limit case test tests Robin approximating either Dirichlet or Neumann.
        All test models have Dirichlet on dir_inds (Dirichlet index) boundaries, and
        Robin approximating Dirichlet or Neumann on the remaining ones. This method
        returns the indices of the north and south boundaries, which are the Dirichlet
        indices.

        """
        domain_sides = self.domain_boundary_sides(sd)
        return domain_sides.north + domain_sides.south

    def dir_inds(self, sd) -> np.ndarray:
        """Indices for the Dirichlet boundaries for test.

        The Robin limit case test tests Robin approximating either Dirichlet or Neumann.
        All test models have Dirichlet on dir_inds (Dirichlet index) boundaries, and
        Robin approximating Dirichlet or Neumann on the remaining ones. This method
        returns the indices of the west and east boundaries, which are the Robin
        indices.

        """
        domain_sides = self.domain_boundary_sides(sd)
        return domain_sides.west + domain_sides.east

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns displacement values in the x-direction of the Dirichlet boundaries."""
        values = np.zeros((self.nd, bg.num_cells))
        values[0, self.dir_inds(bg)] = 42
        return values.ravel("F")

    def _bc_values_scalar(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns scalar values on the Dirichlet boundaries."""
        values = np.zeros(bg.num_cells)
        values[self.dir_inds(bg)] = 42
        return values

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns pressure boundary values."""
        return self._bc_values_scalar(bg=bg)

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns temperature boundary values."""
        return self._bc_values_scalar(bg=bg)


class BCRobin(pp.PorePyModel):
    """Set Dirichlet and Robin for momentum balance and mass and energy balance.

    Sets Dirichlet on dir_inds-boundaries, and Robin on the remaining ones. The value of
    the Robin weight is determined from the parameter "alpha" in the params dictionary.
    This class also sets Robin boundary values.

    This class is common for all the test classes that enters into testing Robin limit
    cases.

    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Sets Robin and Dirichlet conditions.

        Sets Dirichlet boundary condition type on the Dirichlet index-boundaries and
        Robin on all others.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "rob")
        bc.is_rob[:, self.dir_inds(sd)] = False
        bc.is_dir[:, self.dir_inds(sd)] = True

        alpha = self.params["alpha"]

        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        bc.robin_weight = (
            np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), order="F") * alpha
        )
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        Sets Dirichlet boundary condition type on the Dirichlet index-boundaries and
        Robin on all others.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.all_bf, "rob")
        bc.is_rob[self.dir_inds(sd)] = False
        bc.is_dir[self.dir_inds(sd)] = True

        bc.robin_weight = np.ones(sd.num_faces) * self.params["alpha"]
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)


class BCNeumannReference(pp.PorePyModel):
    """Set Dirichlet and Neumann for momentum balance and mass and energy balance."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Assigns Neumann and Dirichlet boundaries for the Neumann case."""
        bc = pp.BoundaryConditionVectorial(sd, self.dir_inds(sd), "dir")
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on the Dirichlet index boundaries, and Neumann on
        all others.

        """

        bc = pp.BoundaryCondition(sd, self.dir_inds(sd), "dir")
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self._bc_type_scalar(sd=sd)


class BCValuesFlux(pp.PorePyModel):
    def bc_values_fourier_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns Fourier flux values on Robin index boundaries."""
        return self._bc_values_scalar_flux(bg)

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns Darcy flux values on Robin index boundaries."""
        return self._bc_values_scalar_flux(bg)

    def _bc_values_scalar_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((bg.num_cells))
        val = 24
        if self.params["alpha"] > 0:  # Robin-Dirichlet
            # The flux value here will be the value of the Robin condition and not seen
            # in the Dirichlet reference case. We need to multiply with the cell volume
            # and the alpha value to account for Robin being interpreted as an
            # integrated flux (volume) and being compared to alpha * u, since the Robin
            # condition is on the form sigma * n + alpha * u = G and the first term is
            # negligible for large alpha.
            volumes = bg.cell_volumes[self.rob_inds(bg)]
            val *= volumes * self.params["alpha"]
        values[self.rob_inds(bg)] = val
        return values.ravel("F")

    def bc_values_stress(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns stress values on the non-Dirichlet boundaries."""
        values = np.zeros((self.nd, bg.num_cells))
        val = 24
        if self.params["alpha"] > 0:  # Robin-Dirichlet
            # The flux value here will be the value of the Robin condition and not seen
            # in the Dirichlet reference case. We need to multiply with the cell volume
            # and the alpha value to account for Robin being interpreted as an
            # integrated flux (volume) and being compared to alpha * u, since the Robin
            # condition is on the form sigma * n + alpha * u = G and the first term is
            # negligible for large alpha.
            volumes = bg.cell_volumes[self.rob_inds(bg)]
            val *= volumes * self.params["alpha"]
        values[0, self.rob_inds(bg)] = val
        return values.ravel("F")


class BCDirichletReference(pp.PorePyModel):
    """Set all Dirichlet boundaries for momentum balance and mass and energy balance."""

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Assigns Dirichlet boundaries on all domain boundary sides."""
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "dir")
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on all boundaries.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns displacement values in the x-direction of the Robin index
        boundaries."""
        values = super().bc_values_displacement(bg=bg)
        values = values.reshape((self.nd, bg.num_cells), order="F")
        inds = self.rob_inds(bg)
        values[0, inds] = 24
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
        values[self.rob_inds(bg)] = 24
        return values


class CommonMassEnergyBalance(
    SquareDomainOrthogonalFractures,
    BCValuesDirichletIndices,
    BCValuesFlux,
    pp.MassAndEnergyBalance,
):
    """Base mass and energy balance model.

    The model in this class is common for the reference class for mass and energy
    balance and for the "test" class for mass and energy balance. The "test" class is
    the class which represents a problem model with Robin boundaries.

    """


class MassAndEnergyBalanceRobin(BCRobin, CommonMassEnergyBalance):
    """Mass and energy balance with Robin and Dirichlet conditions.

    The methods dir_inds and rob_inds determine which boundaries are Dirichlet and which
    are Robin.

    """


class CommonMomentumBalance(
    SquareDomainOrthogonalFractures,
    BCValuesDirichletIndices,
    BCValuesFlux,
    MomentumBalance,
):
    """Base momentum balance model.

    The model in this class is common for the reference class for momentum balance and
    for the "test" class for momentum balance. The "test" class is the class which
    represents a problem model with Robin boundaries.

    """


class MomentumBalanceRobin(BCRobin, CommonMomentumBalance):
    """Momentum balance with Robin and Dirichlet conditions.

    The methods dir_inds and rob_inds determine which boundaries are Dirichlet and which
    are Robin.

    """


def run_model(model_class: type[pp.PorePyModel], alpha: float) -> dict[str, np.ndarray]:
    params = {
        "times_to_export": [],
        "fracture_indices": [],
        "meshing_arguments": {"cell_size": 0.5},
        "times_to_export": [],  # Suppress output for tests
    }

    params["alpha"] = alpha
    model = model_class(params)
    pp.run_time_dependent_model(model)
    sd = model.mdg.subdomains(dim=2)[0]

    if isinstance(model, MomentumBalance):
        displacement = model.equation_system.evaluate(model.displacement([sd]))
        return {"displacement": displacement}
    elif isinstance(model, pp.MassAndEnergyBalance):
        pressure = model.equation_system.evaluate(model.pressure([sd]))
        temperature = model.equation_system.evaluate(model.temperature([sd]))
        return {"temperature": temperature, "pressure": pressure}


# Parameterize the test function with the necessary balance types and conditions
@pytest.mark.parametrize(
    "model_class, reference_model_class, alpha",
    [
        (MomentumBalanceRobin, CommonMomentumBalance, 0),
        (MassAndEnergyBalanceRobin, CommonMassEnergyBalance, 0),
        (MassAndEnergyBalanceRobin, CommonMassEnergyBalance, 1e8),
        (MomentumBalanceRobin, CommonMomentumBalance, 1e8),
    ],
)
def test_robin_limit_case(
    model_class: type[pp.PorePyModel],
    reference_model_class: type[pp.PorePyModel],
    alpha: float,
):
    """Test Robin limit cases.

    The Robin conditions are implemented on the form: sigma * n + alpha * u = G. That
    means that setting Robin conditions with alpha = 0 should correspond to setting
    Neumann conditions. For large alpha (alpha -> \infty), the Robin conditions should
    correspond to Dirichlet conditions.

    We test this for momentum balance and mass and energy balance.

    Common for all models is that the have Dirichlet conditions on the boundaries
    returned by the method dir_inds.

    The model classes with documentation are further up in this document.

    """
    if alpha > 0:
        reference_bc_class = BCDirichletReference
    elif alpha == 0:
        reference_bc_class = BCNeumannReference

    class LocalReferenceModel(reference_bc_class, reference_model_class):
        """Reference class with the correct reference boundary types."""

    rob_results = run_model(model_class, alpha)
    reference_results = run_model(LocalReferenceModel, alpha)

    assert all(
        np.allclose(rob_results[key], reference_results[key], atol=1e-7)
        for key in rob_results.keys()
    )
