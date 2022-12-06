"""Tests for model variables.

"""
import numpy as np
import pytest

import porepy as pp
from porepy.models.constitutive_laws import ad_wrapper

from .setup_utils import MomentumBalance


class BoundaryConditionsDirNorthSouth:
    """Overload the boundary condition.

    Dirichlet values on the north and south boundaries. The values are set to zero by
    default, but can be changed by setting the parameters north_displacement and
    south_displacement.

    """

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary regions
        _, _, _, north, south, *_ = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, north + south, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """
        Not sure where this one should reside. Note that we could remove the
        grid_operator BC and DirBC, probably also ParameterArray/Matrix (unless needed
        to get rid of pp.ad.Discretization. I don't see how it would be, though).
        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        # Define boundary regions
        values = []
        for sd in subdomains:
            _, _, _, north, south, *_ = self.domain_boundary_sides(sd)
            val_loc = np.zeros((self.nd, sd.num_faces))
            val_loc[1, north] = self.params.get("north_displacement", 0)
            val_loc[1, south] = self.params.get("south_displacement", 0)
            values.append(val_loc.ravel("F"))
        return ad_wrapper(np.hstack(values), True, name="bc_values_mechnics")


class LinearModel(BoundaryConditionsDirNorthSouth, MomentumBalance):
    pass


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({}, 0),
        ({}, -0.1),
        ({"porosity": 0.5}, 0.2),
    ],
)
def test_2d_single_fracture(solid_vals, north_displacement):
    """Test that the solution is qualitatively sound.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.

    """
    # Instantiate constants and store in params.
    solid = pp.SolidConstants(solid_vals)
    params = {
        "suppress_export": True,  # Suppress output for tests
        "material_constants": {"solid": solid},
        "north_displacement": north_displacement,
    }

    # Create model and run simulation
    setup = LinearModel(params)
    pp.run_time_dependent_model(setup, params)

    # Check that the pressure is linear
    sd = setup.mdg.subdomains(dim=setup.nd)[0]
    var = setup.equation_system.get_variables([setup.displacement_variable], [sd])
    vals = setup.equation_system.get_variable_values(var)
    if np.isclose(north_displacement, 0):
        assert np.allclose(vals, 0)
    else:
        if north_displacement < 0:
            # Boundary displacement is negative, so the y displacement should be
            # negative
            assert np.all(np.sign(vals[setup.nd - 1 :: setup.nd]) < 0)

            # Check that x displacement has the same sign as north_displacement for
            # x<0.5, and the opposite sign for x>0.5. To see why this makes sense, think
            # through what happens around the symmetry line of x=0.5 when pulling or
            # pushing the top (north) boundary.
            tol = 1e-10
            left = sd.cell_centers[0] < setup.box["xmax"] / 2 - tol
            right = sd.cell_centers[0] > setup.box["xmax"] / 2 + tol
            assert np.all(
                np.sign(vals[:: setup.nd][left]) == np.sign(north_displacement)
            )
            assert np.all(
                np.sign(vals[:: setup.nd][right]) == -np.sign(north_displacement)
            )
        else:
            # Check that y displacement is positive in top half of domain
            top = sd.cell_centers[1] > 0.5
            assert np.all(
                np.sign(vals[setup.nd - 1 :: setup.nd][top])
                == np.sign(north_displacement)
            )
            # Fracture cuts the domain in half, so the bottom half should be undispalced
            bottom = sd.cell_centers[1] < 0.5
            assert np.allclose(vals[setup.nd - 1 :: setup.nd][bottom], 0)
            # No displacement in x direction
            assert np.allclose(vals[:: setup.nd], 0)

    # Check that the displacement jump and traction are as expected
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)
    jump = setup.displacement_jump(sd_frac).evaluate(setup.equation_system).val
    traction = setup.contact_traction(sd_frac).evaluate(setup.equation_system).val
    if north_displacement > 0:
        # Normal component of displacement jump should be positive
        assert np.all(jump[setup.nd - 1 :: setup.nd] > 0)
        # Traction should be zero
        assert np.allclose(traction, 0)
    else:
        # Displacement jump should be zero
        assert np.all(np.isclose(jump, 0))
        # Normal traction should be non-positive. Zero if north_displacement is zero.,
        if north_displacement < 0:
            assert np.all(traction[setup.nd - 1 :: setup.nd] <= 0)
        else:
            assert np.allclose(traction, 0)
