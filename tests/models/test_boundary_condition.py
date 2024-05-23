"""This file is testing the functionality of `pp.BoundaryConditionMixin`.

"""
from typing import Callable, Sequence

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.models import MassBalance as MassBalance_


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
        return self._combine_boundary_operators(
            subdomains=subdomains,
            dirichlet_operator=self.fluid_density,
            neumann_operator=lambda bgs: self.create_boundary_operator(
                name=self.custom_bc_neumann_key, domains=bgs
            ),
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
        bc_val_prev_ts = bc_operator.at_previous_timestep().value(setup.equation_system)
        expected_val = np.arange(bg.num_cells) * bg.parent.dim * (t_end - 1)
        # Projecting the expected value to the subdomain.
        expected_val = bg.projection().T @ expected_val
        assert np.allclose(bc_val_prev_ts[bc_type.is_neu], expected_val[bc_type.is_neu])
