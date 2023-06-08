"""This file contains testing that targeted rediscretization of models yields same
results as full discretization.

Targeted rediscretization is a feature that allows rediscretization of a subset of
discretizations defined in the nonlinear_discretization property. The list is accessed
by :meth:`porepy.models.solution_strategy.SolutionStrategy.rediscretize`  called at the
beginning of a nonlinear iteration.

"""
from __future__ import annotations

import copy

import numpy as np
import pytest

import porepy as pp

from . import setup_utils


class RediscretizationTest:
    """Class to short-circuit the solution strategy to a single iteration.

    The class is used as a mixin which partially replaces the SolutionStrategy class.

    Relevant parts of simulation flow:
    1. Discretize the problem
    2. Assemble the linear system
    3. Solve the linear system
    4. Update the state variables
    5. Check convergence first time (expected not to pass)
    6. Rediscretize at the beginning of the next iteration
    7. Assemble the linear system
    8. Check convergence second time (passes by hard-coding)

    Then, quit the simulation and compare stored linear systems.

    """

    def check_convergence(self, *args, **kwargs):
        if self._nonlinear_iteration > 1:
            return 0.0, True, False
        else:
            # Call to super is okay here, since the full model used in the tests is
            # known to have a method of this name.
            return super().check_convergence(*args, **kwargs)

    def rediscretize(self):
        if self.params["full_rediscretization"]:
            return super().discretize()
        else:
            return super().rediscretize()

    def assemble_linear_system(self) -> None:
        """Store all assembled linear systems for later comparison."""
        super().assemble_linear_system()
        if not hasattr(self, "stored_linear_system"):
            self.stored_linear_system = []
        self.stored_linear_system.append(copy.deepcopy(self.linear_system))


# Non-trivial solution achieved through BCs.
class BCs(setup_utils.BoundaryConditionsMassAndEnergyDirNorthSouth):
    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries. We
        set the nonzero random pressure on the north boundary, and 0 on the south
        boundary.

        Parameters:
            subdomains: List of subdomains for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        vals = []
        if len(subdomains) == 0:
            return pp.ad.DenseArray(np.zeros(0), name="bc_values_darcy")
        for sd in subdomains:
            domain_sides = self.domain_boundary_sides(sd)
            vals_loc = np.zeros(sd.num_faces)
            np.random.seed(0)
            vals_loc[domain_sides.north] = np.random.rand(domain_sides.north.sum())
            vals.append(vals_loc)
        return pp.wrap_as_ad_array(np.hstack(vals), name="bc_values_darcy")


# No need to test momentum balance, as it contains no discretizations that need
# rediscretization.
model_classes = [
    setup_utils._add_mixin(BCs, setup_utils.MassBalance),
    setup_utils._add_mixin(BCs, setup_utils.MassAndEnergyBalance),
    setup_utils._add_mixin(BCs, setup_utils.Poromechanics),
    setup_utils._add_mixin(BCs, setup_utils.Thermoporomechanics),
]


@pytest.mark.parametrize("model_class", model_classes)
def test_targeted_rediscretization(model_class):
    """Test that targeted rediscretization yields same results as full discretization."""
    params_full = {
        "full_rediscretization": True,
        "fracture_indices": [0, 1],
        "cartesian": True,
        # Make flow problem non-linear:
        "material_constants": {"fluid": pp.FluidConstants({"compressibility": 1})},
    }
    # Finalize the model class by adding the rediscretization mixin.
    rediscretization_model_class = setup_utils._add_mixin(
        RediscretizationTest, model_class
    )
    # A model object with full rediscretization.
    model_full = rediscretization_model_class(params_full)
    pp.run_time_dependent_model(model_full, params_full)

    # A model object with targeted rediscretization.
    params_targeted = params_full.copy()
    params_targeted["full_rediscretization"] = False
    # Set up the model.
    model_targeted = rediscretization_model_class(params_targeted)
    pp.run_time_dependent_model(model_targeted, params_targeted)

    # Check that the linear systems are the same.
    assert len(model_full.stored_linear_system) == 2
    assert len(model_targeted.stored_linear_system) == 2
    for i in range(len(model_full.stored_linear_system)):
        A_full, b_full = model_full.stored_linear_system[i]
        A_targeted, b_targeted = model_targeted.stored_linear_system[i]

        # Convert to dense array to ensure the matrices are identical.
        assert np.allclose(A_full.A, A_targeted.A)
        assert np.allclose(b_full, b_targeted)

    # Check that the discretization matrix changes between iterations. Without this
    # check, missing rediscretization may go unnoticed.
    tol = 1e-2
    diff = model_full.stored_linear_system[0][0] - model_full.stored_linear_system[1][0]
    assert np.linalg.norm(diff.todense()) > tol
