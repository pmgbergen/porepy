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


class BCValues:
    """Set boundary values for momentum, mass, and mass and energy balance."""

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns displacement values in the x-direction of the west boundary."""
        values = np.ones((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)
        values[0][bounds.west] += np.ones(len(values[0][bounds.west]))
        return values.ravel("F")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns pressure values on the west boundary."""
        values = np.ones(bg.num_cells)
        bounds = self.domain_boundary_sides(bg)
        values[bounds.west] += np.ones(len(values[bounds.west]))
        return values

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Assigns temperature values on the west boundary."""
        values = np.ones(bg.num_cells)
        bounds = self.domain_boundary_sides(bg)
        values[bounds.west] += np.ones(len(values[bounds.west]))
        return values


class BCRobDir:
    """Set Dirichlet and Robin for momentum, mass, and mass and energy balance.

    Sets Dirichlet on the west boundary, and Robin with alpha = 0 on all other
    boundaries.

    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "rob",
        )
        bc.is_rob[:, bounds.west] = False
        bc.is_dir[:, bounds.west] = True

        r_w = np.tile(np.eye(sd.dim), (1, sd.num_faces))
        bc.robin_weight = np.reshape(r_w, (sd.dim, sd.dim, sd.num_faces), "F") * 0
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on the west boundary, and Robin with alpha = 0 on
        all other boundaries.

        """
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "rob",
        )
        bc.is_rob[bounds.west] = False
        bc.is_dir[bounds.west] = True

        bc.robin_weight = np.zeros(sd.num_faces)
        return bc

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_scalar(sd=sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_scalar(sd=sd)


class BCDirNeu(BCRobDir):
    """Set Dirichlet and Neumann for momentum, mass, and mass and energy balance.

    Sets Dirichlet on the west boundary, and Neumann on all other boundaries.

    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, bounds.all_bf, "neu")
        bc.is_neu[:, bounds.west] = False
        bc.is_dir[:, bounds.west] = True
        return bc

    def _bc_type_scalar(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Helper function for setting boundary conditions on scalar fields.

        The function sets Dirichlet on the west boundary, and Neumann on all other
        boundaries.

        """

        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryCondition(
            sd,
            bounds.north + bounds.south + bounds.east + bounds.west,
            "neu",
        )
        bc.is_neu[bounds.west] = False
        bc.is_dir[bounds.west] = True
        return bc


class MassBalanceNeu(
    SquareDomainOrthogonalFractures,
    BCValues,
    BCDirNeu,
    pp.models.fluid_mass_balance.SinglePhaseFlow,
): ...


class MassBalanceRob(BCRobDir, MassBalanceNeu): ...


class MassAndEnergyBalanceNeu(
    SquareDomainOrthogonalFractures,
    BCValues,
    BCDirNeu,
    pp.models.mass_and_energy_balance.MassAndEnergyBalance,
): ...


class MassAndEnergyBalanceRob(BCRobDir, MassAndEnergyBalanceNeu): ...


class MomentumBalanceNeu(
    SquareDomainOrthogonalFractures,
    BCValues,
    BCDirNeu,
    pp.models.momentum_balance.MomentumBalance,
): ...


class MomentumBalanceRob(BCRobDir, MomentumBalanceNeu): ...


@pytest.fixture()
def run_models():
    params = {
        "times_to_export": [],
        "fracture_indices": [],
        "meshing_arguments": {"cell_size": 0.5},
    }
    models = {}

    def run_model(balance_class):
        instance = balance_class(params)
        pp.run_time_dependent_model(instance, params)
        sd = instance.mdg.subdomains(dim=2)[0]

        if isinstance(instance, (MassBalanceRob, MassBalanceNeu)):
            pressure = instance.pressure([sd]).value(instance.equation_system)
            return {"pressure": pressure}
        elif isinstance(instance, (MomentumBalanceRob, MomentumBalanceNeu)):
            displacement = instance.displacement([sd]).value(instance.equation_system)
            return {"displacement": displacement}
        elif isinstance(instance, (MassAndEnergyBalanceRob, MassAndEnergyBalanceNeu)):
            temperature = instance.temperature([sd]).value(instance.equation_system)
            return {"temperature": temperature}

    models["mass_balance"] = {
        "rob": run_model(MassBalanceRob),
        "neu": run_model(MassBalanceNeu),
    }

    models["momentum_balance"] = {
        "rob": run_model(MomentumBalanceRob),
        "neu": run_model(MomentumBalanceNeu),
    }

    models["mass_and_energy_balance"] = {
        "rob": run_model(MassAndEnergyBalanceRob),
        "neu": run_model(MassAndEnergyBalanceNeu),
    }
    return models


@pytest.mark.parametrize(
    "balance_type", ["mass_balance", "momentum_balance", "mass_and_energy_balance"]
)
def test_robin_limit_case(run_models, balance_type):
    """Test that Robin conditions are equivalent to Neumann with Robin weight = 0.

    The Robin conditions are implemented on the form: sigma * n + alpha * u = G. That
    means that setting Robin conditions with alpha = 0 should correspond exactly to
    setting Neumann conditions.

    We test this for mass balance, mass and energy balance, and momentum balance.

    Common for all model setups is that they have one Dirichlet condition to introduce
    some driving forces to the system.

    The model class setups with documentation are further up in this document.

    """
    rob_results = run_models[balance_type]["rob"]
    neu_results = run_models[balance_type]["neu"]

    assert all(
        np.allclose(rob_results[key], neu_results[key]) for key in rob_results.keys()
    )
