"""
This is a setup class for solving linear elasticity with contact between the fractures.

The setup handles parameters, variables and discretizations. Default (unitary-like)
parameters are set. A "run script" function for setting up the class and solving the
nonlinear contact mechanics problem is also provided.
"""
import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist
import porepy as pp


class ContactMechanics:
    def __init__(self, mesh_args, folder_name):
        self.mesh_args = mesh_args
        self.folder_name = folder_name

        # Variables
        self.displacement_variable = "u"
        self.mortar_displacement_variable = "mortar_u"
        self.contact_traction_variable = "contact_traction"

        # Keyword
        self.mechanics_parameter_key = "mechanics"

        # Terms of the equations
        self.friction_coupling_term = "fracture_force_balance"

    def create_grid(self):
        """
        Method that creates a GridBucket of a 2D domain with one fracture and sets
        projections to local coordinates for all fractures.

        The method requires the following attribute:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            frac_pts (np.array): Nd x (number of fracture points), the coordinates of
                the fracture endpoints.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
            gb (pp.GridBucket): The produced grid bucket.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.
        """
        # List the fracture points
        self.frac_pts = np.array([[0.2, 0.8], [0.5, 0.5]])
        # Each column defines one fracture
        frac_edges = np.array([[0], [1]])
        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()

    def domain_boundary_sides(self, g):
        """
        Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries.
        """
        tol = 1e-10
        box = self.box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        north = g.face_centers[1] > box["ymax"] - tol
        south = g.face_centers[1] < box["ymin"] + tol
        if self.Nd == 2:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom

    def bc_type(self, g):
        all_bf, *_ = self.domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, all_bf, "dir")
        return bc

    def bc_values(self, g):
        # Values for all Nd components, facewise
        values = np.zeros((self.Nd, g.num_faces))
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    def source(self, g):
        return 0

    def set_parameters(self):
        """
        Set the parameters for the simulation.
        """
        gb = self.gb

        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                lam = np.ones(g.num_cells)
                mu = np.ones(g.num_cells)
                C = pp.FourthOrderTensor(g.dim, mu, lam)

                # Define boundary condition
                bc = self.bc_type(g)
                # Default internal BC is Neumann. We change to Dirichlet for the contact
                # problem. I.e., the mortar variable represents the displacement on the
                # fracture faces.
                frac_face = g.tags["fracture_faces"]
                bc.is_neu[:, frac_face] = False
                bc.is_dir[:, frac_face] = True
                # BC and source values
                bc_val = self.bc_values(g)
                source_val = self.source(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": source_val,
                        "fourth_order_tensor": C,
                    },
                )

            elif g.dim == self.Nd - 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction},
                )
        # Should we keep this, @EK?
        for e, d in gb.edges():
            mg = d["mortar_grid"]

            # Parameters for the surface diffusion.
            mu = 1
            lmbda = 1

            pp.initialize_data(
                mg, d, self.mechanics_parameter_key, {"mu": mu, "lambda": lmbda}
            )

    def assign_variables(self):
        """
        Assign variables to the nodes and edges of the grid bucket.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self.Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.displacement_variable: {"cells": self.Nd}
                }
            elif g.dim == self.Nd - 1:
                d[pp.PRIMARY_VARIABLES] = {
                    self.contact_traction_variable: {"cells": self.Nd}
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {}

        for e, d in gb.edges():

            if e[0].dim == self.Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.mortar_displacement_variable: {"cells": self.Nd}
                }

            else:
                d[pp.PRIMARY_VARIABLES] = {}

    def assign_discretizations(self):
        """
        Assign discretizations to the nodes and edges of the grid bucket.
        """
        # For the Nd domain we solve linear elasticity with mpsa.
        Nd = self.Nd
        gb = self.gb
        mpsa = pp.Mpsa(self.mechanics_parameter_key)
        # We need a void discretization for the contact traction variable defined on
        # the fractures.
        empty_discr = pp.VoidDiscretization(self.mechanics_parameter_key, ndof_cell=Nd)

        for g, d in gb:
            if g.dim == Nd:
                d[pp.DISCRETIZATION] = {self.displacement_variable: {"mpsa": mpsa}}
            elif g.dim == Nd - 1:
                d[pp.DISCRETIZATION] = {
                    self.contact_traction_variable: {"empty": empty_discr}
                }

        # Define the contact condition on the mortar grid
        coloumb = pp.ColoumbContact(self.mechanics_parameter_key, Nd)
        contact = pp.PrimalContactCoupling(self.mechanics_parameter_key, mpsa, coloumb)

        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)
            if g_h.dim == Nd:
                d[pp.COUPLING_DISCRETIZATION] = {
                    self.friction_coupling_term: {
                        g_h: (self.displacement_variable, "mpsa"),
                        g_l: (self.contact_traction_variable, "empty"),
                        (g_h, g_l): (self.mortar_displacement_variable, contact),
                    }
                }

    def initial_condition(self):
        """
        Initial guess for Newton iteration.
        """

        for g, d in self.gb:
            if g.dim == self.Nd:
                # Initialize displacement variable
                state = {self.displacement_variable: np.zeros(g.num_cells * self.Nd)}

            elif g.dim == self.Nd - 1:
                # Initialize contact variable
                traction = np.vstack(
                    (np.zeros((g.dim, g.num_cells)), -1 * np.ones(g.num_cells))
                ).ravel(order="F")
                state = {"previous_iterate": {self.contact_traction_variable: traction}}
            else:
                state = {}
            pp.set_state(d, state)

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]

            if mg.dim == self.Nd - 1:
                size = mg.num_cells * self.Nd
                state = {
                    "previous_iterate": {
                        self.mortar_displacement_variable: np.zeros(size)
                    }
                }
                pp.set_state(d, state)

    def extract_iterate(self, assembler, solution_vector):
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE]["previous_iterate"] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            assembler (pp.Assembler): assembler for self.gb.
            solution_vector (np.array): solution vector for the current iterate.

        Returns:
            (np.array): displacement solution vector for the Nd grid.
        """
        dof = np.cumsum(np.append(0, np.asarray(assembler.full_dof)))

        for pair, bi in assembler.block_dof.items():
            g = pair[0]
            name = pair[1]
            # Identify edges, and update the mortar displacement iterate
            if isinstance(g, tuple):
                if name == self.mortar_displacement_variable:
                    mortar_u = solution_vector[dof[bi] : dof[bi + 1]]
                    data = self.gb.edge_props(g)
                    data[pp.STATE]["previous_iterate"][
                        self.mortar_displacement_variable
                    ] = mortar_u
                continue
            else:
                # g is a node (not edge)

                # For the fractures, update the contact force
                if g.dim < self.gb.dim_max():
                    if name == self.contact_traction_variable:
                        contact = solution_vector[dof[bi] : dof[bi + 1]]
                        data = self.gb.node_props(g)
                        data[pp.STATE]["previous_iterate"][
                            self.contact_traction_variable
                        ] = contact

                else:
                    # Only need the displacements for Nd
                    if name != self.displacement_variable:
                        continue
                    u = solution_vector[dof[bi] : dof[bi + 1]]
        return u

    def reconstruct_local_displacement_jump(self, data_edge):
        """
        Reconstruct the displacement jump in local coordinates.

        Args:
            data_edge (dictionary): The dictionary on the gb edge. Should contain
                - a mortar grid
                - a projection, obtained by calling
                pp.contact_conditions.set_projections(self.gb)
        Returns:
            (np.array): ambient_dim x g_l.num_cells. First 1-2 dimensions are in the
            tangential direction of the fracture, last dimension is normal.
        """
        mg = data_edge["mortar_grid"]
        mortar_u = data_edge[pp.STATE][self.mortar_displacement_variable]
        displacement_jump_global_coord = (
            mg.mortar_to_slave_avg(nd=self.Nd)
            * mg.sign_of_mortar_sides(nd=self.Nd)
            * mortar_u
        )
        projection = data_edge["tangential_normal_projection"]
        # Rotated displacement jumps. these are in the local coordinates, on
        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        return u_mortar_local.reshape((self.Nd, -1), order="F")

    def _set_friction_coefficient(self, g):

        nodes = g.nodes

        tips = nodes[:, [0, -1]]

        fc = g.cell_centers
        D = cdist(fc.T, tips.T)
        D = np.min(D, axis=1)
        R = 200
        beta = 10
        friction_coefficient = 0.5 * (1 + beta * np.exp(-R * D ** 2))
        #        friction_coefficient = 0.5 * np.ones(g.num_cells)
        return friction_coefficient


def run_mechanics(setup):
    """
    Function for solving linear elasticity with a non-linear Coulomb contact.

    In addition to the standard parameters for mpsa we also require the following
    under the mechanics keyword (returned from setup.set_parameters):
        'friction_coeff' : The coefficient of friction
        'c' : The numerical parameter in the non-linear complementary function.

    Arguments:
        setup: A setup class with methods:
                create_grid(): Create and return the grid bucket
                set_parameters(): assigns data to grid bucket.
                assign_variables(): assigns variables on grid bucket nodes and edges.
                assign_discretizations(): assigns discretizations on grid bucket nodes
                and edges.
                initial_condition(): Returns initial guess for 'u' and 'lam'.
            and attributes:
                folder_name: returns a string. The data from the simulation will be
                written to the file 'folder_name/' + setup.out_name and the vtk files to
                'res_plot/' + setup.out_name
    """
    # Define mixed-dimensional grid. Avoid overwriting existing gb.
    if "gb" in setup.__dict__:
        gb = setup.gb
    else:
        gb = setup.create_grid()
        gb = setup.gb

    # Pick up grid of highest dimension - there should be a single one of these
    g_max = gb.grids_of_dimension(setup.Nd)[0]
    # Set simulation parameters and assign variables and discretizations
    setup.set_parameters()
    setup.initial_condition()
    setup.assign_variables()
    setup.assign_discretizations()

    # Set up assembler and discretize
    assembler = pp.Assembler(gb)
    assembler.discretize()

    # Prepare for iteration

    u0 = gb.node_props(g_max)[pp.STATE][setup.displacement_variable]
    errors = []

    counter_newton = 0
    converged_newton = False
    max_newton = 15

    viz = pp.Exporter(g_max, name="mechanics", folder=setup.folder_name)

    while counter_newton <= max_newton and not converged_newton:
        print("Newton iteration number: ", counter_newton, "/", max_newton)

        counter_newton += 1
        # Re-discretize the nonlinear term
        assembler.discretize(term_filter=setup.friction_coupling_term)

        # Assemble and solve
        A, b = assembler.assemble_matrix_rhs()
        #        if gb.num_cells() > 6e4: #4
        #            sol = solvers.amg(gb, A, b)
        #        else:
        sol = sps.linalg.spsolve(A, b)

        # Obtain the current iterate for the displacement, and distribute the current
        # iterates for mortar displacements and contact traction.
        u1 = setup.extract_iterate(assembler, sol)

        viz.write_vtk({"ux": u1[::2], "uy": u1[1::2]})

        # Calculate the error
        solution_norm = l2_norm_cell(g_max, u1)
        iterate_difference = l2_norm_cell(g_max, u1, u0)

        # The if is intended to avoid division through zero
        if solution_norm < 1e-12 and iterate_difference < 1e-12:
            converged_newton = True
            error = np.sum((u1 - u0) ** 2)
        else:
            if iterate_difference / solution_norm < 1e-10:
                converged_newton = True
            error = np.sum((u1 - u0) ** 2) / np.sum(u1 ** 2)

        print("Error: ", error)
        errors.append(error)
        # Prepare for next iteration
        u0 = u1

    if counter_newton > max_newton and not converged_newton:
        raise ValueError("Newton iterations did not converge")
    assembler.distribute_variable(sol)


def l2_norm_cell(g, u, uref=None):
    """
    Compute the cell volume weighted norm of a vector-valued cellwise quantity.

    Args:
        g (pp.Grid)
        u (np.array): Vector-valued function.
    """
    if uref is None:
        norm = np.reshape(u ** 2, (g.dim, g.num_cells), order="F") * g.cell_volumes
    else:
        norm = (
            np.reshape((u - uref) ** 2, (g.dim, g.num_cells), order="F")
            * g.cell_volumes
        )
    return np.sum(norm)
