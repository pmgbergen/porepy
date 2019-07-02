"""
Various integration tests for contact mechanics.
"""
import numpy as np
import unittest
import scipy.sparse.linalg as spla

import porepy as pp


class TestContactMechanics(unittest.TestCase):
    def _solve(self, model):
        pp.models.contact_mechanics.run_mechanics(model)
        gb = model.gb

        nd = gb.dim_max()

        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d_m = gb.edge_props((g1, g2))
        d_1 = gb.node_props(g1)

        mg = d_m["mortar_grid"]

        u_mortar = d_m[pp.STATE][model.mortar_displacement_variable]
        contact_force = d_1[pp.STATE][model.contact_traction_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_slave_avg(nd=nd) * mg.sign_of_mortar_sides(nd=nd) * u_mortar
        )
        projection = d_m["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((2, -1), order="F")

        contact_force = contact_force.reshape((2, -1), order="F")

        return u_mortar_local_decomposed, contact_force

    def test_pull_top_positive_opening(self):

        model = SetupContactMechanics(ux_bottom=0, uy_bottom=0, ux_top=0, uy_top=0.001)

        u_mortar, contact_force = self._solve(model)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

    def test_pull_bottom_positive_opening(self):

        model = SetupContactMechanics(ux_bottom=0, uy_bottom=-0.001, ux_top=0, uy_top=0)

        u_mortar, contact_force = self._solve(model)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] < 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

    def test_push_top_zero_opening(self):

        model = SetupContactMechanics(ux_bottom=0, uy_bottom=0, ux_top=0, uy_top=-0.001)

        u_mortar, contact_force = self._solve(model)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

    def test_push_bottom_zero_opening(self):

        model = SetupContactMechanics(ux_bottom=0, uy_bottom=0.001, ux_top=0, uy_top=0)

        u_mortar, contact_force = self._solve(model)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))


class SetupContactMechanics(pp.models.contact_mechanics.ContactMechanics):
    def __init__(self, ux_bottom, uy_bottom, ux_top, uy_top):
        mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }
        super().__init__(mesh_args, folder_name="dummy")
        self.ux_bottom = ux_bottom
        self.uy_bottom = uy_bottom
        self.ux_top = ux_top
        self.uy_top = uy_top

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

    def set_parameters(self):
        """
        Set the parameters for the simulation. The stress is given in GPa.
        """
        gb = self.gb

        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                rock = pp.Granite()
                lam = rock.LAMBDA * np.ones(g.num_cells) / pp.GIGA
                mu = rock.MU * np.ones(g.num_cells) / pp.GIGA

                k = pp.FourthOrderTensor(g.dim, mu, lam)

                # Define boundary regions
                top = g.face_centers[g.dim - 1] > np.max(g.nodes[1]) - 1e-9
                bot = g.face_centers[g.dim - 1] < np.min(g.nodes[1]) + 1e-9

                # Define boundary condition on sub_faces
                bc = pp.BoundaryConditionVectorial(g, top + bot, "dir")
                frac_face = g.tags["fracture_faces"]
                bc.is_neu[:, frac_face] = False
                bc.is_dir[:, frac_face] = True

                # Set the boundary values
                u_bc = np.zeros((g.dim, g.num_faces))

                u_bc[0, bot] = self.ux_bottom
                u_bc[1, bot] = self.uy_bottom
                u_bc[0, top] = self.ux_top
                u_bc[1, top] = self.uy_top

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": u_bc.ravel("F"),
                        "source": 0,
                        "fourth_order_tensor": k,
                    },
                )

            elif g.dim == 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction},
                )

        for e, d in gb.edges():
            mg = d["mortar_grid"]

            pp.initialize_data(mg, d, self.mechanics_parameter_key, {})


if __name__ == "__main__":
    unittest.main()
