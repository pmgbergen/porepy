#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for contact mechanics with pressure coupling.

We have the full Biot equations in the matrix, and mass conservation and contact
conditions in the non-intersecting fracture(s). For the contact mechanical part of this
test, please refer to test_contact_mechanics.
"""
import numpy as np
import scipy.sparse as sps
import unittest
import scipy.sparse.linalg as spla

import porepy as pp
import porepy.models.contact_mechanics_biot_model as model


class TestContactMechanicsBiot(unittest.TestCase):
    def _solve(self, setup):
        model.run_biot(setup)
        gb = setup.gb

        nd = gb.dim_max()

        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d_m = gb.edge_props((g1, g2))
        d_1 = gb.node_props(g1)

        mg = d_m["mortar_grid"]

        u_mortar = d_m[pp.STATE][setup.mortar_displacement_variable]
        contact_force = d_1[pp.STATE][setup.contact_traction_variable]
        fracture_pressure = d_1[pp.STATE][setup.scalar_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_slave_avg(nd=nd) * mg.sign_of_mortar_sides(nd=nd) * u_mortar
        )
        projection = d_m["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((2, -1), order="F")

        contact_force = contact_force.reshape((2, -1), order="F")

        return u_mortar_local_decomposed, contact_force, fracture_pressure

    def test_pull_north_positive_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=0.001
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))

    def test_pull_south_positive_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=-0.001, ux_north=0, uy_north=0
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))

    def test_push_north_zero_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=-0.001
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))

    def test_push_south_zero_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0.001, ux_north=0, uy_north=0
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))

    def test_positive_fracture_pressure_positive_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=0, source_value=0.001
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Fracture pressure is positive
        self.assertTrue(np.all(fracture_pressure > 1e-7))


def distribute_iterate(
    assembler, setup, values, mortar_displacement_variable, contact_traction_variable
):
    """ Update the previous iterate of the mortar displacement and contact traction,
    and obtain current matrix displacement iterate.

    Method is a tailored copy from assembler.distribute_variable.
    """
    dof = np.cumsum(np.append(0, np.asarray(assembler.full_dof)))
    var_name = setup.displacement_variable

    for pair, bi in assembler.block_dof.items():
        g = pair[0]
        name = pair[1]
        # Avoid edges
        if not isinstance(g, pp.Grid):
            if name == mortar_displacement_variable:
                mortar_u = values[dof[bi] : dof[bi + 1]]
                data = setup.gb.edge_props(g)
                data[pp.STATE]["previous_iterate"][
                    mortar_displacement_variable
                ] = mortar_u
            continue
        # Only interested in highest dimension
        if g.dim < setup.gb.dim_max():
            if name == contact_traction_variable:
                contact = values[dof[bi] : dof[bi + 1]]
                data = setup.gb.node_props(g)
                data[pp.STATE]["previous_iterate"][contact_traction_variable] = contact

            continue
        # Only need the displacement
        if name != var_name:
            continue

        u = values[dof[bi] : dof[bi + 1]]
    return u


class SetupContactMechanicsBiot(model.ContactMechanicsBiot):
    def __init__(self, ux_south, uy_south, ux_north, uy_north, source_value=0):
        mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }
        super().__init__(mesh_args, folder_name="dummy")
        self.ux_south = ux_south
        self.uy_south = uy_south
        self.ux_north = ux_north
        self.uy_north = uy_north
        self.scalar_source_value = source_value

    def create_grid(self, rotate_fracture=False):
        """
        Method that creates and returns the GridBucket of a 2D domain with six
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """
        if rotate_fracture:
            self.frac_pts = np.array([[0.7, 0.3], [0.3, 0.7]])
        else:
            self.frac_pts = np.array([[0.3, 0.7], [0.5, 0.5]])
        frac_edges = np.array([[0], [1]])

        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = gb.dim_max()

    def source(self, g, key, t=0):
        if key == self.mechanics_parameter_key:
            values = super().source(g, key, t)
        elif key == self.scalar_parameter_key:
            if g.dim == self.Nd:
                values = np.zeros(g.num_cells)
            else:
                values = self.scalar_source_value * np.ones(g.num_cells)
        else:
            raise ValueError("No BC values implemented for keyword " + str(key))
        return values

    def bc_type(self, g, key, t=0):
        _, _, _, north, south, _, _ = self.domain_boundary_sides(g)
        if key == self.mechanics_parameter_key:
            bc = pp.BoundaryConditionVectorial(g, north + south, "dir")
        elif key == self.scalar_parameter_key:
            # Define boundary condition on faces
            bc = pp.BoundaryCondition(g, north + south, "dir")
        else:
            raise ValueError("No BCs implemented for keyword " + str(key))
        return bc

    def bc_values(self, g, key, t=0):
        # Set the boundary values
        if key == self.mechanics_parameter_key:
            _, _, _, north, south, _, _ = self.domain_boundary_sides(g)
            values = np.zeros((g.dim, g.num_faces))
            values[0, south] = self.ux_south
            values[1, south] = self.uy_south
            values[0, north] = self.ux_north
            values[1, north] = self.uy_north
            values = values.ravel("F")
        elif key == self.scalar_parameter_key:
            values = np.zeros(g.num_faces)
        else:
            raise ValueError("No BC values implemented for keyword " + str(key))
        return values


if __name__ == "__main__":
    unittest.main()
