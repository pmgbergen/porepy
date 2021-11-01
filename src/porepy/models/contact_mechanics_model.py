"""
This is a setup class for solving linear elasticity with contact mechanics governing
the relative motion of fracture surfaces.

The setup handles parameters, variables and discretizations. Default (unitary-like)
parameters are set.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp
from porepy.models.abstract_model import AbstractModel

# Module-wide logger
logger = logging.getLogger(__name__)
module_sections = ["models", "numerics"]


class ContactMechanics(AbstractModel):
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
            Default parameters are set in AbstractModel
        gb (pp.GridBucket): Mixed-dimensional grid. Should be set by a method
            create_grid which should be provided by the user.
        convergence_status (bool): Whether the non-linear iterations has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'

    All attributes are given natural values at initialization of the class.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, params: Optional[Dict] = None):

        super().__init__(params)

        # Variables
        self.displacement_variable: str = "u"
        self.mortar_displacement_variable: str = "mortar_u"
        self.contact_traction_variable: str = "contact_traction"

        # Keyword
        self.mechanics_parameter_key: str = "mechanics"

        # Terms of the equations
        self.friction_coupling_term: str = "fracture_force_balance"

    ## Public methods
    @pp.time_logger(sections=module_sections)
    def get_state_vector(self, use_iterate: bool = False) -> np.ndarray:
        """Get a vector of the current state of the variables; with the same ordering
            as in the assembler.

        Returns:
            np.array: The current state, as stored in the GridBucket.

        """
        return self.dof_manager.assemble_variable(from_iterate=use_iterate)

    @pp.time_logger(sections=module_sections)
    def before_newton_loop(self):
        """Will be run before entering a Newton loop.
        Discretize time-dependent quantities etc.
        """
        self.convergence_status = False
        self._nonlinear_iteration = 0

    @pp.time_logger(sections=module_sections)
    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        if self._use_ad:
            self._eq_manager.equations["contact"].discretize(self.gb)
        else:
            filt = pp.assembler_filters.ListFilter(
                term_list=[self.friction_coupling_term]
            )
            self.assembler.discretize(filt=filt)

    @pp.time_logger(sections=module_sections)
    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """
        Extract parts of the solution for current iterate.

        The iterate solutions in d[pp.STATE][pp.ITERATE] are updated for the
        mortar displacements and contact traction are updated.
        Method is a tailored copy from assembler.distribute_variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        self._nonlinear_iteration += 1
        self.dof_manager.distribute_variable(
            values=solution_vector, additive=self._use_ad, to_iterate=True
        )

    @pp.time_logger(sections=module_sections)
    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._use_ad:
            # Fetch iterate solution, which was updated in after_newton_iteration
            solution = self.dof_manager.assemble_variable(from_iterate=True)
            # Distribute to pp.STATE
            self.dof_manager.distribute_variable(values=solution, additive=False)
        else:
            self.assembler.distribute_variable(solution)
        self.convergence_status = True

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: Dict[str, Any],
    ) -> Tuple[float, bool, bool]:
        """
        Check whether the solution has converged by comparing values from the two
        most recent iterations.

        Taylored implementation if AD is not used. Else, the generic check in
        AbstractModel is used.

        Parameters:
            solution (array): solution of current iteration.
            solution (array): solution of previous iteration.
            solution (array): initial solution (or from beginning of time step).
            nl_params (dictionary): assumed to have the key nl_convergence_tol whose
                value is a float.
        """
        if self._use_ad or not self._is_nonlinear_problem():
            return super().check_convergence(
                solution, prev_solution, init_solution, nl_params
            )

        g_max = self._nd_grid()
        mech_dof = self.dof_manager.grid_and_variable_to_dofs(
            g_max, self.displacement_variable
        )

        # Also find indices for the contact variables
        contact_dof = np.array([], dtype=int)
        for e, _ in self.gb.edges():
            if e[0].dim == self._Nd:
                contact_dof = np.hstack(
                    (
                        contact_dof,
                        self.dof_manager.grid_and_variable_to_dofs(
                            e[1], self.contact_traction_variable
                        ),
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
        diverged = False  # type: ignore

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

    @pp.time_logger(sections=module_sections)
    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:

        if self._use_ad:
            A, b = self._eq_manager.assemble()
        else:
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

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def prepare_simulation(self) -> None:
        """Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, export for visualization, linear solvers etc.

        """
        self.create_grid()
        self._Nd = self.gb.dim_max()
        self._set_parameters()
        self._assign_variables()
        self._assign_discretizations()
        # Once we have defined all discretizations, it's time to instantiate an
        # equation manager (needs to know which terms it should treat)
        if not self._use_ad:
            self.assembler: pp.Assembler = pp.Assembler(self.gb, self.dof_manager)

        self._initial_condition()
        self._discretize()
        self._initialize_linear_solver()

        g_max = self._nd_grid()
        self.viz = pp.Exporter(
            g_max,
            file_name=self.params["file_name"],
            folder_name=self.params["folder_name"],
        )

    @pp.time_logger(sections=module_sections)
    def after_simulation(self) -> None:
        """Called after a time-dependent problem"""
        pass

    # Methods for populating the model etc.

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def _nd_grid(self) -> pp.Grid:
        """Get the grid of the highest dimension. Assumes self.gb is set."""
        return self.gb.grids_of_dimension(self._Nd)[0]

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Set homogeneous conditions on all boundary faces."""
        # Values for all Nd components, facewise
        values = np.zeros((self._Nd, g.num_faces))
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    @pp.time_logger(sections=module_sections)
    def _source(self, g: pp.Grid) -> np.ndarray:
        """"""
        return np.zeros(self._Nd * g.num_cells)

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def _assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        """

        # For the Nd domain we solve linear elasticity with mpsa.
        Nd = self._Nd
        gb = self.gb
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(gb)
        if not self._use_ad:
            mpsa = pp.Mpsa(self.mechanics_parameter_key)
            # We need a void discretization for the contact traction variable defined on
            # the fractures.
            empty_discr = pp.VoidDiscretization(
                self.mechanics_parameter_key, ndof_cell=Nd
            )

            for g, d in gb:
                if g.dim == Nd:
                    d[pp.DISCRETIZATION] = {self.displacement_variable: {"mpsa": mpsa}}
                elif g.dim == Nd - 1:
                    d[pp.DISCRETIZATION] = {
                        self.contact_traction_variable: {"empty": empty_discr}
                    }

            # Define the contact condition on the mortar grid
            coloumb = pp.ColoumbContact(self.mechanics_parameter_key, Nd, mpsa)
            contact = pp.PrimalContactCoupling(
                self.mechanics_parameter_key, mpsa, coloumb
            )

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

        else:
            eq_manager = pp.ad.EquationManager(gb, self.dof_manager)

            g_primary: pp.Grid = gb.grids_of_dimension(Nd)[0]
            g_frac: List[pp.Grid] = gb.grids_of_dimension(Nd - 1).tolist()
            num_frac_cells = np.sum([g.num_cells for g in g_frac])

            if len(gb.grids_of_dimension(Nd)) != 1:
                raise NotImplementedError("This will require further work")
            grid_list = [g for g, _ in gb]
            edge_list = [(g_primary, g) for g in g_frac]

            mortar_proj = pp.ad.MortarProjections(
                grids=grid_list, edges=edge_list, gb=gb, nd=self._Nd
            )
            subdomain_proj = pp.ad.SubdomainProjections(grids=grid_list, nd=self._Nd)
            tangential_proj_lists = [
                gb.node_props(
                    gf, "tangential_normal_projection"
                ).project_tangential_normal(gf.num_cells)
                for gf in g_frac
            ]
            tangential_proj = pp.ad.Matrix(sps.block_diag(tangential_proj_lists))

            mpsa_ad = mpsa_ad = pp.ad.MpsaAd(self.mechanics_parameter_key, [g_primary])
            bc_ad = pp.ad.BoundaryCondition(
                self.mechanics_parameter_key, grids=[g_primary]
            )
            div = pp.ad.Divergence(grids=[g_primary], dim=g_primary.dim)

            # Primary variables on Ad form
            u = eq_manager.variable(g_primary, self.displacement_variable)
            u_mortar = eq_manager.merge_variables(
                [(e, self.mortar_displacement_variable) for e in edge_list]
            )
            contact_force = eq_manager.merge_variables(
                [(g, self.contact_traction_variable) for g in g_frac]
            )

            u_mortar_prev = u_mortar.previous_timestep()

            # Stress in g_h
            stress = (
                mpsa_ad.stress * u
                + mpsa_ad.bound_stress * bc_ad
                + mpsa_ad.bound_stress
                * subdomain_proj.face_restriction([g_primary])
                * mortar_proj.mortar_to_primary_avg
                * u_mortar
            )

            # momentum balance equation in g_h
            momentum_eq = div * stress

            coloumb_ad = pp.ad.ColoumbContactAd(
                self.mechanics_parameter_key, edges=edge_list
            )
            jump_rotate = (
                tangential_proj
                * subdomain_proj.cell_restriction(g_frac)
                * mortar_proj.mortar_to_secondary_avg
                * mortar_proj.sign_of_mortar_sides
            )

            # Contact conditions
            jump_discr = coloumb_ad.displacement * jump_rotate * u_mortar
            tmp = np.ones(num_frac_cells * self._Nd)
            tmp[self._Nd - 1 :: self._Nd] = 0
            exclude_normal = pp.ad.Matrix(
                sps.dia_matrix((tmp, 0), shape=(tmp.size, tmp.size))
            )
            # Rhs of contact conditions
            rhs = (
                coloumb_ad.rhs
                + exclude_normal * coloumb_ad.displacement * jump_rotate * u_mortar_prev
            )
            contact_eq = coloumb_ad.traction * contact_force + jump_discr - rhs

            # Force balance
            mat = None
            for _, d in gb.edges():
                mg: pp.MortarGrid = d["mortar_grid"]
                if mg.dim < self._Nd - 1:
                    continue

                faces_on_fracture_surface = mg.primary_to_mortar_int().tocsr().indices
                m = pp.grid_utils.switch_sign_if_inwards_normal(
                    g_primary, self._Nd, faces_on_fracture_surface
                )
                if mat is None:
                    mat = m
                else:
                    mat += m

            sign_switcher = pp.ad.Matrix(mat)

            # Contact from primary grid and mortar displacements (via primary grid)
            contact_from_primary_mortar = (
                mortar_proj.primary_to_mortar_int
                * subdomain_proj.face_prolongation([g_primary])
                * sign_switcher
                * stress
            )
            contact_from_secondary = (
                mortar_proj.sign_of_mortar_sides
                * mortar_proj.secondary_to_mortar_int
                * subdomain_proj.cell_prolongation(g_frac)
                * tangential_proj.transpose()
                * contact_force
            )
            force_balance_eq = contact_from_primary_mortar + contact_from_secondary

            eq_manager.equations.update(
                {
                    "momentum": momentum_eq,
                    "contact": contact_eq,
                    "force_balance": force_balance_eq,
                }
            )

            self._eq_manager = eq_manager

    @pp.time_logger(sections=module_sections)
    def _set_friction_coefficient(self, g: pp.Grid) -> np.ndarray:
        """The friction coefficient is uniform, and equal to 1."""
        return np.ones(g.num_cells)

    # Methods for discretization, numerical issues etc.
    @pp.time_logger(sections=module_sections)
    def _discretize(self) -> None:
        """Discretize all terms"""
        tic = time.time()
        logger.info("Discretize")
        if self._use_ad:
            self._eq_manager.discretize(self.gb)
        else:
            self.assembler.discretize()
        logger.info("Done. Elapsed time {}".format(time.time() - tic))

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def _is_nonlinear_problem(self) -> bool:
        """
        If there is no fracture, the problem is usually linear.
        Overwrite this function if e.g. parameter nonlinearities are included.
        """
        return self.gb.dim_min() < self._Nd
