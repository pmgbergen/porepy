"""Tests for poromechanics.

The hardcoded fracture gap value of 0.042 ensures positive aperture for all simulation.
This is needed to avoid degenerate mass balance equation in fracture.

TODO: Clean up.
"""
import numpy as np
import pytest

import porepy as pp

from .setup_utils import PoromechanicsCombined
from .test_momentum_balance import BoundaryConditionsDirNorthSouth

p0 = 0.0


class LinearModel(BoundaryConditionsDirNorthSouth, PoromechanicsCombined):
    def bc_values_darcy(self, subdomains: list[pp.Grid]):
        values = []
        for sd in subdomains:
            all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(
                sd
            )
            vals = np.zeros(sd.num_faces)
            vals[all_bf] = p0
            values.append(vals)
        return pp.constitutive_laws.ad_wrapper(
            np.concatenate(values), True, name="bc_values_darcy"
        )

    def initial_condition(self):
        super().initial_condition()
        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        top_cells = sd.cell_centers[1] > 0.5
        vals = np.zeros((self.nd, sd.num_cells))
        vals[1, top_cells] = 0.042
        self.equation_system.set_variable_values(
            vals.ravel("F"),
            [self.displacement_variable],
            to_state=True,
            to_iterate=True,
        )
        # Find mortar cells on the top boundary
        intf, data = self.mdg.interfaces(return_data=True)[0]
        # Identify by normal vector in sd_primary
        faces_primary = intf.primary_to_mortar_int().tocsr().indices
        switcher = pp.grid_utils.switch_sign_if_inwards_normal(
            sd,
            self.nd,
            faces_primary,
        )
        nc = sum([sd.num_cells for sd in self.mdg.subdomains()])
        ones = np.ones(nc)
        self.equation_system.set_variable_values(
            p0 * ones, [self.pressure_variable], to_iterate=True, to_state=True
        )
        normals = (switcher * sd.face_normals[: sd.dim].ravel("F")).reshape(
            sd.dim, -1, order="F"
        )
        intf_normals = normals[:, faces_primary]
        top_cells = intf_normals[1, :] < 0

        # Set mortar displacement to zero on bottom and 0.042 on top
        vals = np.zeros((self.nd, intf.num_cells))
        vals[1, top_cells] = 0.042
        self.equation_system.set_variable_values(
            vals.ravel("F"),
            [self.interface_displacement_variable],
            to_state=True,
            to_iterate=True,
        )

        # Bc values for stress

        _, _, _, north, south, *_ = self.domain_boundary_sides(sd)
        val_loc = np.zeros((self.nd, sd.num_faces))
        val_loc[1, north] = 0.042
        sd_data[pp.STATE].update({"bc_values_mechanics": val_loc.ravel("F")})
        val_loc[1, north] = self.params.get("north_displacement", 0)
        sd_data[pp.STATE][pp.ITERATE].update(
            {"bc_values_mechanics": val_loc.ravel("F")}
        )

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
        return pp.ad.TimeDependentArray("bc_values_mechanics", subdomains)

    def reference_pressure(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference pressure.

        Parameters:
            subdomains: List of subdomains.

            Returns:
                Operator representing the reference pressure.

        TODO: Confirm that this is the right place for this method. # IS: Definitely not
        a Material. Most closely related to the constitutive laws. # Perhaps create a
        reference values class that is a mixin to the constitutive laws? # Could have
        values in the init and methods returning operators just as # this method.
        """
        p_ref = self.fluid.convert_units(p0, "Pa")
        size = sum([sd.num_cells for sd in subdomains])
        return pp.constitutive_laws.ad_wrapper(
            p_ref, True, size, name="reference_pressure"
        )


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({}, 0.042),
        ({}, -0.1),
        ({"porosity": 0.5}, 0.1),
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
    solid_vals["fracture_gap"] = 0.042
    solid_vals["biot_coefficient"] = 0.8
    # solid_vals["permeability"] = 1e-5
    fluid_vals = {"compressibility": 1}
    solid = pp.SolidConstants(solid_vals)
    fluid = pp.FluidConstants(fluid_vals)

    params = {
        "suppress_export": True,  # Suppress output for tests
        "material_constants": {"solid": solid, "fluid": fluid},
        "north_displacement": north_displacement,
        "max_iterations": 50,
    }

    # Create model and run simulation
    setup = LinearModel(params)
    pp.run_time_dependent_model(setup, params)

    # Check that the pressure is linear
    sd = setup.mdg.subdomains(dim=setup.nd)[0]
    u_var = setup.equation_system.get_variables([setup.displacement_variable], [sd])
    u_vals = setup.equation_system.get_variable_values(u_var).reshape(
        setup.nd, -1, order="F"
    )
    p_var = setup.equation_system.get_variables(
        [setup.pressure_variable], setup.mdg.subdomains()
    )
    p_vals = setup.equation_system.get_variable_values(p_var)
    print(p_vals)

    top = sd.cell_centers[1] > 0.5
    bottom = sd.cell_centers[1] < 0.5
    tol = 1e-10
    if np.isclose(north_displacement, 0.042):

        assert np.allclose(u_vals[:, bottom], 0)
        # Zero x and nonzero y displacement in top
        assert np.allclose(u_vals[0, top], 0)
        assert np.allclose(u_vals[1, top], 0.042)
        # Zero displacement relative to initial value implies zero pressure
        assert np.allclose(p_vals, 0)
    elif north_displacement < tol:
        # Boundary displacement is negative, so the y displacement should be
        # negative
        assert np.all(u_vals[1] < 0)

        # Check that x displacement has the same sign as north_displacement for
        # x<0.5, and the opposite sign for x>0.5. To see why this makes sense, think
        # through what happens around the symmetry line of x=0.5 when pulling or
        # pushing the top (north) boundary.
        left = sd.cell_centers[0] < 0.5
        assert np.all(u_vals[0, left] < 0)
        right = sd.cell_centers[0] > 0.5
        assert np.all(u_vals[0, right] > 0)
        # Compression implies pressure increase
        assert np.all(p_vals > -tol)
    else:
        # Check that y displacement is positive in top half of domain
        assert np.all(u_vals[1, top] > 0)
        # Fracture cuts the domain in half, so the bottom half is only affected by
        # the pressure, which is negative, and thus the displacement should be
        # expansive, i.e. positive
        assert np.all(u_vals[1, bottom] > 0)
        # Same argument as for y in bottom half
        # assert np.allclose(u_vals[:: setup.nd], 0)
        # Expansion implies pressure reduction
        assert np.all(p_vals < tol)

    # Check that the displacement jump and traction are as expected
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)
    jump = (
        setup.displacement_jump(sd_frac)
        .evaluate(setup.equation_system)
        .val.reshape(setup.nd, -1, order="F")
    )
    traction = (
        setup.contact_traction(sd_frac)
        .evaluate(setup.equation_system)
        .val.reshape(setup.nd, -1, order="F")
    )
    if north_displacement > 0.042:
        # Normal component of displacement jump should be positive
        assert np.all(jump[1] > -tol)
        # Traction should be zero
        assert np.allclose(traction, 0)
    else:
        # Displacement jump should be equal to initial displacement
        assert np.allclose(jump[0], 0.0)
        assert np.allclose(jump[1], 0.042)
        # Normal traction should be non-positive. Zero if north_displacement is zero.,
        if north_displacement < 0.042:
            assert np.all(traction[setup.nd - 1 :: setup.nd] <= tol)
        else:
            assert np.allclose(traction, 0)
