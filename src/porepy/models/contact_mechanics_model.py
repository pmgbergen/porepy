"""
This is a setup class for solving linear elasticity with contact between the fractures.

The setup handles parameters, variables and discretizations. Default (unitary-like)
parameters are set. A "run script" function for setting up the class and solving the
nonlinear contact mechanics problem is also provided.

NOTE: This module should be considered an experimental feature, which may
undergo major changes on little notice.

"""
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp
import porepy.models.abstract_model

# Module-wide logger
logger = logging.getLogger(__name__)


class ContactMechanics(porepy.models.abstract_model.AbstractModel):
    """This is a shell class for contact mechanics problems.

    Setting up such problems requires a lot of boilerplate definitions of variables,
    parameters and discretizations. This class is intended to provide a standardized
    setup, with all discretizations in place and reasonable parameter and boundary
    values. The intended use is to inherit from this class, and do the necessary
    modifications and specifications for the problem to be fully defined. The minimal
    adjustment needed is to specify the method create_grid().

    Attributes:
        displacement_variable (str): Name assigned to the displacement variable in the
            highest-dimensional subdomain. Will be used throughout the simulations,
            including in Paraview export.

        mortar_displacement_variable (str): Name assigned to the displacement variable
            on the fracture walls. Will be used throughout the simulations, including in
            Paraview export.
        contact_traction_variable (str): Name assigned to the variable for contact
            forces in the fracture. Will be used throughout the simulations, including
            in Paraview export.

        mechanics_parameter_key (str): Keyword used to define parameters and
            discretizations for the mechanics problem.
        friction_coupling_term (str): Keyword used to define paramaters and
            discretizations for the friction problem.

        params (dict): Dictionary of parameters used to control the solution procedure.
        viz_folder_name (str): Folder for visualization export.
        gb (pp.GridBucket): Mixed-dimensional grid. Should be set by a method
            create_grid which should be provided by the user.
        convergence_status (bool): Whether the non-linear iterations has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'

    All attributes are given natural values at initialization of the class.

    """

    @pp.time_logger
    def __init__(self, params: Optional[Dict] = None):

        # Variables
        self.displacement_variable: str = "u"
        self.mortar_displacement_variable: str = "mortar_u"
        self.contact_traction_variable: str = "contact_traction"

        # Keyword
        self.mechanics_parameter_key: str = "mechanics"

        # Terms of the equations
        self.friction_coupling_term: str = "fracture_force_balance"

        # Solver parameters
        if params is None:
            self.params = {}
            self.viz_folder_name: str = "contact_mechanics_viz"
        else:
            self.params = params
            self.viz_folder_name = params.get("folder_name", "contact_mechanics_viz")

        # Initialize grid bucket
        self.gb: pp.GridBucket = None  # type: ignore
        self.box: Dict[str, float] = None  # type: ignore

        # Set a convergence study. Not sure if a boolean is sufficient, or whether
        # we should have an enum here.
        self.convergence_status: bool = False
        self.linear_solver: str = "direct"

        self._iteration: int = 0

    # Public methods

    @pp.time_logger
    def create_grid(self) -> None:
        """Create a (fractured) domain in 2D or 3D, with projections to local
        coordinates set for all fractures.

        The method requires the following attribute, which is stored in self.params:
            mesh_args (dict): Containing the mesh sizes.

        The method assigns the following attributes to self:
            gb (pp.GridBucket): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
            Nd (int): The dimension of the matrix, i.e., the highest dimension in the
                grid bucket.

        After self.gb is set, the method should also call

            pp.contact_conditions.set_projections(self.gb)

        """
        raise NotImplementedError(
            "Method for grid creation should be implemented in a subclass"
        )

    @pp.time_logger
    def get_state_vector(self) -> np.ndarray:
        """Get a vector of the current state of the variables; with the same ordering
            as in the assembler.

        Returns:
            np.array: The current state, as stored in the GridBucket.

        """
        size = self.assembler.num_dof()
        state = np.zeros(size)
        for g, var in self.assembler.block_dof.keys():
            # Index of
            ind = self.assembler.dof_ind(g, var)

            if isinstance(g, tuple):
                values = self.gb.edge_props(g)[pp.STATE][var]
            else:
                values = self.gb.node_props(g)[pp.STATE][var]
            state[ind] = values

        return state

    @pp.time_logger
    def before_newton_loop(self):
        """Will be run before entering a Newton loop.
        Discretize time-dependent quantities etc.
        """
        self.convergence_status = False
        self._iteration = 0

    @pp.time_logger
    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        filt = pp.assembler_filters.ListFilter(term_list=[self.friction_coupling_term])
        self.assembler.discretize(filt=filt)

    @pp.time_logger
    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE][pp.ITERATE] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        self._iteration += 1
        self._update_iterate(solution_vector)

    @pp.time_logger
    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.assembler.distribute_variable(solution)
        self.convergence_status = True

    @pp.time_logger
    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._is_nonlinear_problem():
            raise ValueError("Newton iterations did not converge")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    @pp.time_logger
    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: Dict[str, Any],
    ) -> Tuple[float, bool, bool]:
        g_max = self._nd_grid()

        if not self._is_nonlinear_problem():
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = np.any(np.isnan(solution))
            converged = not diverged
            error = np.nan if diverged else 0
            return error, converged, diverged

        mech_dof = self.assembler.dof_ind(g_max, self.displacement_variable)

        # Also find indices for the contact variables
        contact_dof = np.array([], dtype=np.int)
        for e, _ in self.gb.edges():
            if e[0].dim == self._Nd:
                contact_dof = np.hstack(
                    (
                        contact_dof,
                        self.assembler.dof_ind(e[1], self.contact_traction_variable),
                    )
                )

        # Pick out the solution from current, previous iterates, as well as the
        # initial guess.
        u_mech_now = solution[mech_dof]
        u_mech_prev = prev_solution[mech_dof]
        u_mech_init = init_solution[mech_dof]

        contact_now = solution[contact_dof]
        contact_prev = prev_solution[contact_dof]
        contact_init = init_solution[contact_dof]

        # Calculate errors
        difference_in_iterates_mech = np.sum((u_mech_now - u_mech_prev) ** 2)
        difference_from_init_mech = np.sum((u_mech_now - u_mech_init) ** 2)

        contact_norm = np.sum(contact_now ** 2)
        difference_in_iterates_contact = np.sum((contact_now - contact_prev) ** 2)
        difference_from_init_contact = np.sum((contact_now - contact_init) ** 2)

        tol_convergence: float = nl_params["nl_convergence_tol"]
        # Not sure how to use the divergence criterion
        # tol_divergence = nl_params["nl_divergence_tol"]

        converged = False
        diverged = False

        # Check absolute convergence criterion
        if difference_in_iterates_mech < tol_convergence:
            converged = True
            error_mech = difference_in_iterates_mech

        else:
            # Check relative convergence criterion
            if (
                difference_in_iterates_mech
                < tol_convergence * difference_from_init_mech
            ):
                converged = True
            error_mech = difference_in_iterates_mech / difference_from_init_mech

        # The if is intended to avoid division through zero
        if contact_norm < 1e-10 and difference_in_iterates_contact < 1e-10:
            # converged = True
            error_contact = difference_in_iterates_contact
        else:
            error_contact = (
                difference_in_iterates_contact / difference_from_init_contact
            )

        logger.info("Error in contact force is {}".format(error_contact))
        logger.info("Error in matrix displacement is {}".format(error_mech))

        return error_mech, converged, diverged

    @pp.time_logger
    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:

        A, b = self.assembler.assemble_matrix_rhs()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min"
            + f" {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )
        if self.linear_solver == "direct":
            return spla.spsolve(A, b)
        else:
            raise NotImplementedError("Not that far yet")

    @pp.time_logger
    def reconstruct_local_displacement_jump(
        self,
        data_edge: Dict,
        projection: pp.TangentialNormalProjection,
        from_iterate: bool = True,
    ) -> np.ndarray:
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
        if from_iterate:
            mortar_u = data_edge[pp.STATE][pp.ITERATE][
                self.mortar_displacement_variable
            ]
        else:
            mortar_u = data_edge[pp.STATE][self.mortar_displacement_variable]
        displacement_jump_global_coord = (
            mg.mortar_to_secondary_avg(nd=self._Nd)
            * mg.sign_of_mortar_sides(nd=self._Nd)
            * mortar_u
        )
        # Rotated displacement jumps. these are in the local coordinates, on
        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        return u_mortar_local.reshape((self._Nd, -1), order="F")

    @pp.time_logger
    def reconstruct_stress(self, previous_iterate: bool = False) -> None:
        """
        Compute the stress in the highest-dimensional grid based on the displacement
        states in that grid, adjacent interfaces and global boundary conditions.

        The stress is stored in the data dictionary of the highest-dimensional grid,
        in [pp.STATE]['stress'].

        Parameters:
            previous_iterate (boolean, optional): If True, use values from previous
                iteration to compute the stress. Defaults to False.

        """
        # Highest-dimensional grid and its data
        g = self._nd_grid()
        d = self.gb.node_props(g)

        # Pick the relevant displacemnet field
        if previous_iterate:
            u = d[pp.STATE][pp.ITERATE][self.displacement_variable]
        else:
            u = d[pp.STATE][self.displacement_variable]

        matrix_dictionary: Dict[str, sps.spmatrix] = d[pp.DISCRETIZATION_MATRICES][
            self.mechanics_parameter_key
        ]

        # Make a discretization object to get hold of the right keys to access the
        # matrix_dictionary
        mpsa = pp.Mpsa(self.mechanics_parameter_key)
        # Stress contribution from internal cell center displacements
        stress: np.ndarray = matrix_dictionary[mpsa.stress_matrix_key] * u

        # Contributions from global boundary conditions
        bound_stress_discr: sps.spmatrix = matrix_dictionary[
            mpsa.bound_stress_matrix_key
        ]
        global_bc_val: np.ndarray = d[pp.PARAMETERS][self.mechanics_parameter_key][
            "bc_values"
        ]
        stress += bound_stress_discr * global_bc_val

        # Contributions from the mortar displacement variables
        for e, d_e in self.gb.edges():
            # Only contributions from interfaces to the highest dimensional grid
            mg: pp.MortarGrid = d_e["mortar_grid"]
            if mg.dim == self._Nd - 1:
                if previous_iterate:
                    u_e: np.ndarray = d_e[pp.STATE][pp.ITERATE][
                        self.mortar_displacement_variable
                    ]
                else:
                    u_e = d_e[pp.STATE][self.mortar_displacement_variable]

                stress += (
                    bound_stress_discr * mg.mortar_to_primary_avg(nd=self._Nd) * u_e
                )

        d[pp.STATE]["stress"] = stress

    @pp.time_logger
    def prepare_simulation(self) -> None:
        """Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, export for visualization, linear solvers etc.

        """
        self.create_grid()
        self._Nd = self.gb.dim_max()
        self._set_parameters()
        self._assign_variables()
        self._assign_discretizations()
        self._initial_condition()
        self._discretize()
        self._initialize_linear_solver()

        g_max = self._nd_grid()
        self.viz = pp.Exporter(
            g_max, file_name="mechanics", folder_name=self.viz_folder_name
        )

    @pp.time_logger
    def after_simulation(self) -> None:
        """Called after a time-dependent problem"""
        pass

    # Methods for populating the model etc.

    @pp.time_logger
    def _set_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        gb = self.gb

        for g, d in gb:
            if g.dim == self._Nd:
                # Rock parameters
                lam = np.ones(g.num_cells)
                mu = np.ones(g.num_cells)
                C = pp.FourthOrderTensor(mu, lam)

                # Define boundary condition
                bc = self._bc_type(g)

                # BC and source values
                bc_val = self._bc_values(g)
                source_val = self._source(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": source_val,
                        "fourth_order_tensor": C,
                        # "max_memory": 7e7,
                    },
                )

            elif g.dim == self._Nd - 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction},
                )
        for _, d in gb.edges():
            mg = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)

    @pp.time_logger
    def _nd_grid(self) -> pp.Grid:
        """Get the grid of the highest dimension. Assumes self.gb is set."""
        return self.gb.grids_of_dimension(self._Nd)[0]

    @pp.time_logger
    def _domain_boundary_sides(
        self, g: pp.Grid
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
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
        if self._Nd == 2:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom

    @pp.time_logger
    def _bc_type(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions: Dirichlet on all global boundaries,
        Dirichlet also on fracture faces.
        """
        all_bf = g.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(g, all_bf, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    @pp.time_logger
    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Set homogeneous conditions on all boundary faces."""
        # Values for all Nd components, facewise
        values = np.zeros((self._Nd, g.num_faces))
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    @pp.time_logger
    def _source(self, g: pp.Grid) -> np.ndarray:
        """"""
        return np.zeros(self._Nd * g.num_cells)

    @pp.time_logger
    def _assign_variables(self) -> None:
        """
        Assign variables to the nodes and edges of the grid bucket.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self._Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.displacement_variable: {"cells": self._Nd}
                }
            elif g.dim == self._Nd - 1:
                d[pp.PRIMARY_VARIABLES] = {
                    self.contact_traction_variable: {"cells": self._Nd}
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {}

        for e, d in gb.edges():

            if e[0].dim == self._Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.mortar_displacement_variable: {"cells": self._Nd}
                }

            else:
                d[pp.PRIMARY_VARIABLES] = {}

    @pp.time_logger
    def _assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        """
        # For the Nd domain we solve linear elasticity with mpsa.
        Nd = self._Nd
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
        coloumb = pp.ColoumbContact(self.mechanics_parameter_key, Nd, mpsa)
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

    # Methods for discretization, numerical issues etc.

    @pp.time_logger
    def _initial_condition(self):
        """Set initial guess for the variables.

        The displacement is set to zero in the Nd-domain, and at the fracture interfaces
        The displacement jump is thereby also zero.

        The contact pressure is set to zero in the tangential direction,
        and -1 (that is, in contact) in the normal direction.

        """

        for g, d in self.gb:
            if g.dim == self._Nd:
                # Initialize displacement variable
                state = {self.displacement_variable: np.zeros(g.num_cells * self._Nd)}

            elif g.dim == self._Nd - 1:
                # Initialize contact variable
                traction = np.vstack(
                    (np.zeros((g.dim, g.num_cells)), -1 * np.ones(g.num_cells))
                ).ravel(order="F")
                state = {
                    self.contact_traction_variable: traction,
                }
                pp.set_iterate(d, {self.contact_traction_variable: traction})
            else:
                state = {}
            pp.set_state(d, state)

        for _, d in self.gb.edges():
            mg = d["mortar_grid"]

            if mg.dim == self._Nd - 1:
                size = mg.num_cells * self._Nd
                state = {self.mortar_displacement_variable: np.zeros(size)}
                iterate = {self.mortar_displacement_variable: np.zeros(size)}
                pp.set_iterate(d, iterate)
            else:
                state = {}
            pp.set_state(d, state)

    @pp.time_logger
    def _update_iterate(self, solution_vector: np.ndarray) -> None:
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE][pp.ITERATE] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            assembler (pp.Assembler): assembler for self.gb.
            solution_vector (np.array): solution vector for the current iterate.

        """
        assembler = self.assembler
        variable_names = []
        for pair in assembler.block_dof.keys():
            variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(assembler.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in assembler.block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue
                if isinstance(g, tuple):
                    # This is really an edge
                    if name == self.mortar_displacement_variable:
                        mortar_u = solution_vector[dof[bi] : dof[bi + 1]]
                        data = self.gb.edge_props(g)
                        data[pp.STATE][pp.ITERATE][
                            self.mortar_displacement_variable
                        ] = mortar_u.copy()
                else:
                    data = self.gb.node_props(g)

                    # g is a node (not edge)

                    # For the fractures, update the contact force
                    if g.dim < self._Nd:
                        if name == self.contact_traction_variable:
                            contact = solution_vector[dof[bi] : dof[bi + 1]]
                            data = self.gb.node_props(g)
                            data[pp.STATE][pp.ITERATE][
                                self.contact_traction_variable
                            ] = contact.copy()

    @pp.time_logger
    def _set_friction_coefficient(self, g: pp.Grid) -> np.ndarray:
        """The friction coefficient is uniform, and equal to 1."""
        return np.ones(g.num_cells)

    @pp.time_logger
    def _discretize(self) -> None:
        """Discretize all terms"""

        if not hasattr(self, "assembler"):
            self.assembler = pp.Assembler(self.gb)

        tic = time.time()
        logger.info("Discretize")
        self.assembler.discretize()
        logger.info("Done. Elapsed time {}".format(time.time() - tic))

    @pp.time_logger
    def _initialize_linear_solver(self) -> None:
        solver: str = self.params.get("linear_solver", "direct")
        if solver == "direct":
            """In theory, it should be possible to instruct SuperLU to reuse the
            symbolic factorization from one iteration to the next. However, it seems
            the scipy wrapper around SuperLU has not implemented the necessary
            functionality, as discussed in

                https://github.com/scipy/scipy/issues/8227

            We will therefore pass here, and pay the price of long computation times.

            """
            self.linear_solver = "direct"

        else:
            raise ValueError(f"Unknown linear solver {solver}")

    @pp.time_logger
    def _is_nonlinear_problem(self) -> bool:
        """
        If there is no fracture, the problem is usually linear.
        Overwrite this function if e.g. parameter nonlinearities are included.
        """
        return self.gb.dim_min() < self._Nd
