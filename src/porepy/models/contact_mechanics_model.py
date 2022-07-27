"""
This is a setup class for solving linear elasticity with contact mechanics governing
the relative motion of fracture surfaces.

The setup handles parameters, variables and discretizations. Default (unitary-like)
parameters are set.

For a description of the mathematical model, see e.g. the tutorials and the PhD thesis
of Hüeber (2008): https://elib.uni-stuttgart.de/handle/11682/4854
"""
from __future__ import annotations

import logging
import time
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import porepy as pp
from porepy.models.abstract_model import AbstractModel

# Module-wide logger
logger = logging.getLogger(__name__)


class ContactMechanicsAdObjects:
    """Storage class for ad related objects.

    Stored objects include variables, compound ad operators and projections.
    """

    displacement: pp.ad.Variable
    interface_displacement: pp.ad.Variable
    contact_force: pp.ad.Variable

    contact_traction: pp.ad.Operator

    local_fracture_coord_transformation: pp.ad.Matrix
    local_fracture_coord_transformation_normal: pp.ad.Matrix
    subdomain_projections_vector: pp.ad.SubdomainProjections
    tangential_component_frac: pp.ad.Matrix
    normal_component_frac: pp.ad.Matrix
    normal_to_tangential_frac: pp.ad.Matrix
    internal_boundary_vector_to_outwards: pp.ad.Matrix
    mortar_projections_vector: pp.ad.MortarProjections


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
            including in ParaView export.
        mortar_displacement_variable (str): Name assigned to the displacement variable
            on the fracture walls. Will be used throughout the simulations, including in
            ParaView export.
        contact_traction_variable (str): Name assigned to the variable for contact
            forces in the fracture. Will be used throughout the simulations, including
            in ParaView export.
        mechanics_parameter_key (str): Keyword used to define parameters and
            discretizations for the mechanics problem.
        friction_coupling_term (str): Keyword used to define parameters and
            discretizations for the friction problem.
        params (dict): Dictionary of parameters used to control the solution procedure.
            Default parameters are set in AbstractModel
        mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. Should be set by a method
            create_grid which should be provided by the user.
        convergence_status (bool): Whether the non-linear iterations has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'

    All attributes are given natural values at initialization of the class.

    """

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

    def before_newton_loop(self):
        """Will be run before entering a Newton loop.
        Discretize time-dependent quantities etc.
        """
        self.convergence_status = False
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear terms
        if not self._use_ad:
            filt = pp.assembler_filters.ListFilter(
                term_list=[self.friction_coupling_term]
            )
            self.assembler.discretize(filt=filt)

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

        Tailored implementation if AD is not used. Else, the generic check in
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

        # The default convergence check cannot be applied. Do a manual check instead.
        sd_max = self._nd_subdomain()
        mech_dof = self.dof_manager.grid_and_variable_to_dofs(
            sd_max, self.displacement_variable
        )

        # Also find indices for the contact variables
        contact_dof = np.array([], dtype=int)
        u_j_dof = np.array([], dtype=int)

        for intf in self.mdg.interfaces(dim=self.nd - 1):
            _, sd_secondary = self.mdg.interface_to_subdomain_pair(intf)

            contact_dof = np.hstack(
                (
                    contact_dof,
                    self.dof_manager.grid_and_variable_to_dofs(
                        sd_secondary, self.contact_traction_variable
                    ),
                )
            )
            u_j_dof = np.hstack(
                (
                    u_j_dof,
                    self.dof_manager.grid_and_variable_to_dofs(
                        intf, self.mortar_displacement_variable
                    ),
                )
            )

        # Pick out the solution from current, previous iterates, as well as the
        # initial guess.
        u_mech_now = solution[mech_dof]
        u_mech_prev = prev_solution[mech_dof]
        u_mech_init = init_solution[mech_dof]

        u_j_now = solution[u_j_dof]
        u_j_prev = prev_solution[u_j_dof]
        u_j_init = init_solution[u_j_dof]

        contact_now = solution[contact_dof]
        contact_prev = prev_solution[contact_dof]
        contact_init = init_solution[contact_dof]

        # Calculate errors
        difference_in_iterates_mech = np.sum((u_mech_now - u_mech_prev) ** 2)
        difference_from_init_mech = np.sum((u_mech_now - u_mech_init) ** 2)

        u_j_norm = np.sum(u_j_now**2)
        difference_in_iterates_j = np.sum((u_j_now - u_j_prev) ** 2)
        difference_from_init_j = np.sum((u_j_now - u_j_init) ** 2)

        contact_norm = np.sum(contact_now**2)
        difference_in_iterates_contact = np.sum((contact_now - contact_prev) ** 2)
        difference_from_init_contact = np.sum((contact_now - contact_init) ** 2)

        tol_convergence: float = nl_params["nl_convergence_tol"]
        # Not sure how to use the divergence criterion
        # tol_divergence = nl_params["nl_divergence_tol"]

        converged_mech, converged_j, converged_contact = False, False, False
        diverged = False  # type: ignore

        # Check absolute convergence criterion
        if difference_in_iterates_mech < tol_convergence:
            converged_mech = True
            error_mech = difference_in_iterates_mech

        else:
            # Check relative convergence criterion
            if (
                difference_in_iterates_mech
                < tol_convergence * difference_from_init_mech
            ):
                converged_mech = True
            error_mech = difference_in_iterates_mech / difference_from_init_mech

        # The if is intended to avoid division through zero
        if contact_norm < 1e-10 and difference_in_iterates_contact < 1e-10:
            converged_contact = True
            error_contact = difference_in_iterates_contact
        else:
            error_contact = (
                difference_in_iterates_contact / difference_from_init_contact
            )

            if (
                difference_in_iterates_contact
                < tol_convergence * difference_from_init_contact
            ):
                converged_contact = True

        # The if is intended to avoid division through zero
        if u_j_norm < 1e-10 and difference_in_iterates_j < 1e-10:
            converged_j = True
            error_j = difference_in_iterates_j
        else:
            error_j = difference_in_iterates_j / difference_from_init_j
            if difference_in_iterates_j < tol_convergence * difference_from_init_j:
                converged_j = True
        converged = converged_mech and converged_j and converged_contact
        logger.info("Error in contact force is {}".format(error_contact))
        logger.info("Error in matrix displacement is {}".format(error_mech))
        error = error_mech + error_j + error_contact

        return error, converged, diverged

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:

        if self._use_ad:
            A, b = self._eq_manager.assemble()
        else:
            A, b = self.assembler.assemble_matrix_rhs()  # type: ignore
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min"
            + f" {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )
        if self.linear_solver == "direct":
            return spla.spsolve(A, b)
        elif self.linear_solver == "pypardiso":
            try:
                import pypardiso  # type: ignore

                return pypardiso.spsolve(A, b)
            except ImportError:
                warnings.warn(
                    "Could not import pypardiso, defaulting to scipy.linalg.spsolve"
                )
                return spla.spsolve(A, b)
        else:
            raise NotImplementedError("Not that far yet")

    def reconstruct_local_displacement_jump(
        self,
        intf: pp.MortarGrid,
        projection: pp.TangentialNormalProjection,
        from_iterate: bool = True,
    ) -> np.ndarray:
        """
        Reconstruct the displacement jump in local coordinates.

        Args:
            intf represented by a mortar grid, expected to have the projection
            obtained by calling pp.contact_conditions.set_projections(self.mdg)
        Returns:
            (np.array): ambient_dim x sd_l.num_cells. First 1-2 dimensions are in the
            tangential direction of the fracture, last dimension is normal.
        """
        interface_data = self.mdg.interface_data(intf)
        if from_iterate:
            mortar_u = interface_data[pp.STATE][pp.ITERATE][
                self.mortar_displacement_variable
            ]
        else:
            mortar_u = interface_data[pp.STATE][self.mortar_displacement_variable]
        displacement_jump_global_coord = (
            intf.mortar_to_secondary_avg(nd=self.nd)
            * intf.sign_of_mortar_sides(nd=self.nd)
            * mortar_u
        )
        # Rotated displacement jumps. these are in the local coordinates, on
        project_to_local = projection.project_tangential_normal(int(intf.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        return u_mortar_local.reshape((self.nd, -1), order="F")

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
        sd = self._nd_subdomain()
        data = self.mdg.subdomain_data(sd)

        # Pick the relevant displacement field
        if previous_iterate:
            u = data[pp.STATE][pp.ITERATE][self.displacement_variable]
        else:
            u = data[pp.STATE][self.displacement_variable]

        matrix_dictionary: Dict[str, sps.spmatrix] = data[pp.DISCRETIZATION_MATRICES][
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
        global_bc_val: np.ndarray = data[pp.PARAMETERS][self.mechanics_parameter_key][
            "bc_values"
        ]
        stress += bound_stress_discr * global_bc_val

        # Contributions from the mortar displacement variables
        for intf, intf_data in self.mdg.interfaces(return_data=True):
            # Only contributions from interfaces to the highest dimensional grid
            if intf.dim == self.nd - 1:
                if previous_iterate:
                    u_intf: np.ndarray = intf_data[pp.STATE][pp.ITERATE][
                        self.mortar_displacement_variable
                    ]
                else:
                    u_intf = intf_data[pp.STATE][self.mortar_displacement_variable]

                stress += (
                    bound_stress_discr * intf.mortar_to_primary_avg(nd=self.nd) * u_intf
                )

        data[pp.STATE]["stress"] = stress

    def prepare_simulation(self) -> None:
        """Is run prior to a time-stepping scheme. Use this to initialize
        discretizations, export for visualization, linear solvers etc.

        """
        self.create_grid()
        self.nd = self.mdg.dim_max()
        self._assign_variables()
        if self._use_ad:
            self._assign_ad_variables()
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(self.mdg)
        self._initial_condition()
        self._set_parameters()
        self._assign_discretizations()
        # Once we have defined all discretizations, it's time to instantiate an
        # equation manager (needs to know which terms it should treat)
        if not self._use_ad:
            self.assembler: pp.Assembler = pp.Assembler(self.mdg, self.dof_manager)

        self._discretize()
        self._initialize_linear_solver()  # TODO: Purge method from standard models?

        self.exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"],
            folder_name=self.params["folder_name"],
        )

    def _initial_condition(self):
        """Set initial guess for the variables.

        The displacement is set to zero in the Nd-domain, and at the fracture interfaces
        The displacement jump is thereby also zero.

        The contact pressure is set to zero in the tangential direction,
        and -1 (that is, in contact) in the normal direction.

        """
        # Zero for displacement and initial bc values for Biot
        super()._initial_condition()

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd - 1:
                # Contact as initial guess. Ensure traction is consistent with
                # zero jump, which follows from the default zeros set for all
                # variables, specifically interface displacement, by super method.
                vals = np.zeros((self.nd, sd.num_cells))
                vals[-1] = -1
                vals = vals.ravel("F")
                data[pp.STATE].update({self.contact_traction_variable: vals})

                data[pp.STATE][pp.ITERATE].update(
                    {self.contact_traction_variable: vals.copy()}
                )

    def after_simulation(self) -> None:
        """Called after a time-dependent problem"""
        pass

    # Methods for populating the model etc.

    def _set_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                pp.initialize_data(
                    sd,
                    data,
                    self.mechanics_parameter_key,
                    {
                        "bc": self._bc_type(sd),
                        "bc_values": self._bc_values(sd),
                        "source": self._body_force(sd),
                        "fourth_order_tensor": self._stiffness_tensor(sd),
                    },
                )

            elif sd.dim == self.nd - 1:
                c_num_n, c_num_t = self._numerical_constants(sd)
                pp.initialize_data(
                    sd,
                    data,
                    self.mechanics_parameter_key,
                    {
                        "friction_coefficient": self._friction_coefficient(sd),
                        "initial_gap": self._initial_gap(sd),
                        "dilation_angle": self._dilation_angle(sd),
                        "c_num": c_num_n,  # Non-ad uses single constant
                        "c_num_normal": c_num_n,  # Ad allows for two
                        "c_num_tangential": c_num_t,
                    },
                )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(intf, data, self.mechanics_parameter_key)

    def _nd_subdomain(self) -> pp.Grid:
        """Get the grid of the highest dimension. Assumes self.mdg is set."""
        return self.mdg.subdomains(dim=self.nd)[0]

    def _bc_type(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Define type of boundary conditions: Dirichlet on all global boundaries,
        Dirichlet also on fracture faces.


        Args:
            sd: Subdomain grid.

        Returns:
            bc: Boundary condition representation.

        """
        all_bf = sd.get_boundary_faces()
        bc = pp.BoundaryConditionVectorial(sd, all_bf, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = sd.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        """Set homogeneous conditions on all boundary faces."""
        # Values for all Nd components, face-wise
        values = np.zeros((self.nd, sd.num_faces))
        # Reshape according to PorePy convention
        values = values.ravel("F")
        return values

    def _body_force(self, sd: pp.Grid) -> np.ndarray:
        """Body force parameter.

        If the source term represents gravity in the y (2d) or z (3d) direction,
        use:
            vals = np.zeros((self,nd, sd.num_cells))
            vals[-1] = density * pp.GRAVITY_ACCELERATION * sd.cell_volumes
            return vals.ravel("F")

        Parameters
        ----------
        sd : pp.Grid
            Subdomain, usually the matrix.

        Returns
        -------
        np.ndarray
            Integrated source values, shape self.nd * g.num_cells.

        """
        vals = np.zeros(self.nd * sd.num_cells)
        return vals

    def _dilation_angle(self, sd: pp.Grid) -> np.ndarray:
        """Dilation angle parameter.


        Parameters
        ----------
        sd : pp.Grid
            Fracture subdomain grid.

        Returns
        -------
        vals : np.ndarray
            Cell-wise dilation angle.

        """
        vals: np.ndarray = np.zeros(sd.num_cells)
        return vals

    def _initial_gap(self, sd: pp.Grid) -> np.ndarray:
        """Initial gap value.

        Distance between the surfaces of a fracture when in mechanical contact in
        the unperturbed state.
        Parameters
        ----------
        sd : pp.Grid
            Fracture subdomain grid.

        Returns
        -------
        vals : np.ndarray
            Cell-wise gap value.

        """
        vals: np.ndarray = np.zeros(sd.num_cells)
        return vals

    def _friction_coefficient(self, sd: pp.Grid) -> np.ndarray:
        """Friction coefficient parameter.

        The friction coefficient is uniform and equal to 1.
        Parameters
        ----------
        sd : pp.Grid
            Fracture subdomain grid.

        Returns
        -------
        np.ndarray
            Cell-wise friction coefficient.

        """
        return np.ones(sd.num_cells)

    def _numerical_constants(self, sd: pp.Grid) -> Tuple[np.ndarray, np.ndarray]:
        """Numerical constants for contact mechanics.

        Unitary and homogeneous default values.

        The constants enter sums of traction and displacement jump of the type
            T_i + c * u_i   for i in {n, tau}

        Parameters
        ----------
        sd : pp.Grid
            Subdomain grid for which the constants are returned.

        Returns
        -------
        c_num_n : np.ndarray (sd.num_cells)
            Numerical constant for normal complimentary function.
        c_num_t: np.ndarray (sd.num_cells * sd.dim)
            Numerical constant for tangential complimentary function.

        """
        c_num_n = np.ones(sd.num_cells)
        # Expand to tangential Nd-1 vector
        tangential_vals = np.ones(sd.num_cells)
        c_num_t = np.kron(tangential_vals, np.ones(sd.dim))
        return c_num_n, c_num_t

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        """Fourth order stress tensor.


        Parameters
        ----------
        sd : pp.Grid
            Matrix grid.

        Returns
        -------
        pp.FourthOrderTensor
            Cell-wise representation of the stress tensor.

        """
        # Rock parameters
        lam = np.ones(sd.num_cells)
        mu = np.ones(sd.num_cells)
        return pp.FourthOrderTensor(mu, lam)

    def _assign_variables(self) -> None:
        """
        Assign variables to the subdomains and interfaces of the mixed-dimensional grid.
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == self.nd:
                data[pp.PRIMARY_VARIABLES] = {
                    self.displacement_variable: {"cells": self.nd}
                }
            elif sd.dim == self.nd - 1:
                data[pp.PRIMARY_VARIABLES] = {
                    self.contact_traction_variable: {"cells": self.nd}
                }
            else:
                data[pp.PRIMARY_VARIABLES] = {}

        for intf, data in self.mdg.interfaces(return_data=True):
            if intf.dim == self.nd - 1:
                data[pp.PRIMARY_VARIABLES] = {
                    self.mortar_displacement_variable: {"cells": self.nd}
                }

            else:
                data[pp.PRIMARY_VARIABLES] = {}

    def _assign_ad_variables(self) -> None:
        """Assign variables to self._ad

        Raises
        ------
        NotImplementedError
            If the md_grid contains multiple nd subdomains.

        Assigns the following attributes to self._ad:
            displacement: Primary variable in the nd subdomain.
            interface_displacement: Primary variables on interfaces of
                dimension nd - 1
            contact_force: Primary variable on all fracture subdomains
            contact_traction: Secondary variable on all fracture subdomains.
                This traction is scaled with the inverse of the elastic moduli
                of the matrix.


        Returns
        -------
        None

        """

        # Ad variables
        mdg, nd = self.mdg, self.nd
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(mdg)
        self._eq_manager = pp.ad.EquationManager(mdg, self.dof_manager)
        sd_primary: pp.Grid = self._nd_subdomain()
        fracture_subdomains: List[pp.Grid] = mdg.subdomains(dim=nd - 1)
        self._num_frac_cells = np.sum([sd.num_cells for sd in fracture_subdomains])

        if len(mdg.subdomains(dim=nd)) != 1:
            # Some parts of the code have been partially prepared for
            # this case, other parts require adjustment.
            raise NotImplementedError("This will require further work")

        matrix_fracture_interfaces = [
            mdg.subdomain_pair_to_interface((sd_primary, sd))
            for sd in fracture_subdomains
        ]

        self._set_ad_objects()
        self._set_ad_projections()

        # Primary variables on Ad form
        self._ad.displacement = self._eq_manager.variable(
            sd_primary, self.displacement_variable
        )
        self._ad.interface_displacement = self._eq_manager.merge_variables(
            [
                (intf, self.mortar_displacement_variable)
                for intf in matrix_fracture_interfaces
            ]
        )
        self._ad.contact_force = self._eq_manager.merge_variables(
            [(sd, self.contact_traction_variable) for sd in fracture_subdomains]
        )
        discr = pp.ad.ContactTractionAd(
            self.mechanics_parameter_key,
            matrix_fracture_interfaces,
            fracture_subdomains,
        )
        self._ad.contact_traction = discr.traction_scaling * self._ad.contact_force

    def _set_ad_objects(self) -> None:
        """Sets the storage class self._ad


        Returns
        -------
        None

        """
        self._ad = ContactMechanicsAdObjects()

    def _assign_discretizations(self) -> None:
        """Assign discretizations to subdomains and interfaces."""

        # For the Nd domain we solve linear elasticity with mpsa.
        nd = self.nd
        mdg = self.mdg
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(mdg)
        if not self._use_ad:
            mpsa = pp.Mpsa(self.mechanics_parameter_key)
            # We need a void discretization for the contact traction variable defined on
            # the fractures.
            empty_discr = pp.VoidDiscretization(
                self.mechanics_parameter_key, ndof_cell=nd
            )

            for sd, data in mdg.subdomains(return_data=True):
                if sd.dim == nd:
                    data[pp.DISCRETIZATION] = {
                        self.displacement_variable: {"mpsa": mpsa}
                    }
                elif sd.dim == nd - 1:
                    data[pp.DISCRETIZATION] = {
                        self.contact_traction_variable: {"empty": empty_discr}
                    }

            # Define the contact condition on the mortar grid
            coloumb = pp.ColoumbContact(self.mechanics_parameter_key, nd, mpsa)
            contact = pp.PrimalContactCoupling(
                self.mechanics_parameter_key, mpsa, coloumb
            )

            for intf, intf_data in mdg.interfaces(return_data=True):
                sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
                if sd_primary.dim == nd:
                    intf_data[pp.COUPLING_DISCRETIZATION] = {
                        self.friction_coupling_term: {
                            sd_primary: (self.displacement_variable, "mpsa"),
                            sd_secondary: (self.contact_traction_variable, "empty"),
                            intf: (self.mortar_displacement_variable, contact),
                        }
                    }

        else:
            self._assign_equations()

    def _assign_equations(self):
        """Assign equations to self._eq_manager.

        The ad variables are set by a previous call to _assign_ad_variables and
        accessed through self._ad.*variable_name*

        The following equations are assigned to the equation manager:
            "momentum" in the nd subdomain
            "contact_mechanics_normal" in all fracture subdomains
            "contact_mechanics_tangential" in all fracture subdomains
            "force_balance" at the matrix-fracture interfaces

        Returns
        -------
        None.

        """
        mdg, nd = self.mdg, self.nd

        sd_nd: pp.Grid = self._nd_subdomain()

        fracture_subdomains: List[pp.Grid] = mdg.subdomains(dim=nd - 1)
        self._num_frac_cells = np.sum([g.num_cells for g in fracture_subdomains])

        matrix_fracture_interfaces = [
            intf for intf in mdg.subdomain_to_interfaces(sd_nd) if intf.codim == 1
        ]

        # Projections between subdomains, rotations etc. must be wrapped into
        # ad objects
        self._set_ad_projections()

        # Construct equations
        momentum_eq: pp.ad.Operator = self._momentum_balance_equation([sd_nd])
        contact_n: pp.ad.Operator = self._contact_mechanics_normal_equation(
            fracture_subdomains
        )
        contact_t: pp.ad.Operator = self._contact_mechanics_tangential_equation(
            fracture_subdomains
        )
        force_balance_eq: pp.ad.Operator = self._force_balance_equation(
            [sd_nd],
            fracture_subdomains,
            matrix_fracture_interfaces,
        )
        # Assign equations to manager
        self._eq_manager.name_and_assign_equations(
            {
                "momentum": momentum_eq,
                "contact_mechanics_normal": contact_n,
                "contact_mechanics_tangential": contact_t,
                "force_balance": force_balance_eq,
            },
        )

    def _set_ad_projections(
        self,
    ) -> None:
        """
        Sets projection and rotation matrices.


        The following attributes are set to self._ad:
            local_fracture_coord_transformation: Concatenated transformation matrices from
                global to local (fracture plane) coordinates
            mortar_projections_vector: Projections between matrix and fracture grids
                and their interface/mortar grids.
            subdomain_projections_vector: Prolongation and restriction operators for
                all grids with Nd components for each cell or face.
            normal_component_frac: Extract the normal component (scalar) of a
                Nd-dimensional fracture cell vector.
            tangential_component_frac: Extract the tangential component (Nd-1 vector)
                of a Nd-dimensional fracture cell vector.
            normal_to_tangential_frac: Prolong from scalar to (Nd-1) cell-wise values.
            internal_boundary_vector_to_outwards: Switch sign/direction of a nd
                vector at internal boundary faces of the matrix grid, i.e. corresponding
                to all matrix-fracture interfaces. Entries are 1 and -1, size is
                (ndgrid.num_faces * self.nd) ** 2

        Returns
        -------
        None

        """
        ad = self._ad

        subdomains: List[pp.Grid] = [sd for sd in self.mdg.subdomains()]
        fracture_subdomains: List[pp.Grid] = self.mdg.subdomains(dim=self.nd - 1)
        fracture_matrix_interfaces: List[pp.MortarGrid] = [
            self.mdg.subdomain_pair_to_interface((self._nd_subdomain(), sd))
            for sd in fracture_subdomains
        ]

        # Store number of fracture cells
        self._num_frac_cells = int(np.sum([sd.num_cells for sd in fracture_subdomains]))

        # Special treatment in case no fractures are present
        if len(fracture_subdomains) == 0:
            matrix_attributes = [
                "local_fracture_coord_transformation",
                "normal_component_frac",
                "tangential_component_frac",
                "normal_to_tangential_frac",
                "internal_boundary_vector_to_outwards",
            ]
            for attribute_name in matrix_attributes:
                setattr(ad, attribute_name, pp.ad.Matrix(sps.csr_matrix((0, 0))))
            setattr(
                ad,
                "internal_boundary_vector_to_outwards",
                sps.csr_matrix(
                    (
                        self._nd_subdomain().num_faces * self.nd,
                        self._nd_subdomain().num_faces * self.nd,
                    )
                ),
            )
        else:
            # Global to local coordinate transformation
            local_coord_proj_list = [
                self.mdg.subdomain_data(sd)[
                    "tangential_normal_projection"
                ].project_tangential_normal(sd.num_cells)
                for sd in fracture_subdomains
            ]
            ad.local_fracture_coord_transformation = pp.ad.Matrix(
                sps.block_diag(local_coord_proj_list)
            )
            # Matrices for extracting normal and tangential components on the fractures
            n_frac_c_dim: int = int(self._num_frac_cells * self.nd)
            n_frac_c_t: int = int(self._num_frac_cells * (self.nd - 1))
            normal_rows: np.ndarray = np.arange(self._num_frac_cells)
            normal_cols: np.ndarray = np.arange(self.nd - 1, n_frac_c_dim, self.nd)
            normal_data: np.ndarray = np.ones(self._num_frac_cells)

            tangential_cols: np.ndarray = np.setdiff1d(
                np.arange(n_frac_c_dim), normal_cols
            )
            tangential_rows: np.ndarray = np.arange(n_frac_c_t)
            tangential_data: np.ndarray = np.ones(n_frac_c_t)

            ad.normal_component_frac = pp.ad.Matrix(
                sps.csr_matrix(
                    (normal_data, (normal_rows, normal_cols)),
                    shape=(self._num_frac_cells, n_frac_c_dim),
                )
            )
            ad.tangential_component_frac = pp.ad.Matrix(
                sps.csr_matrix(
                    (tangential_data, (tangential_rows, tangential_cols)),
                    shape=(n_frac_c_t, n_frac_c_dim),
                )
            )

            # Indices in local fracture bases.
            # There are n_frac_c normal components of local vectors and n_frac_c_t
            # tangential components. The Matrix constructed below expands from the former to
            # the latter, so that we e.g. can compare a tangential vector V_t and the
            # prolonged quantity normal_to_tangential * V_n
            local_inds_t: np.ndarray = np.arange(n_frac_c_t)
            local_inds_n: np.ndarray = np.kron(
                np.arange(self._num_frac_cells), np.ones(self.nd - 1)
            )

            n2t_matrix = sps.csr_matrix(
                (np.ones(n_frac_c_t), (local_inds_t, np.int32(local_inds_n))),
                shape=(n_frac_c_t, self._num_frac_cells),
            )
            ad.normal_to_tangential_frac = pp.ad.Matrix(n2t_matrix)

            # Sign switcher accounting for normal direction of faces on internal boundaries of
            # matrix grid (corresponding to matrix-fracture interfaces).
            outwards_mat = None
            for intf in fracture_matrix_interfaces:
                faces_on_fracture_surface = intf.primary_to_mortar_int().tocsr().indices
                switcher_int = pp.grid_utils.switch_sign_if_inwards_normal(
                    self._nd_subdomain(), self.nd, faces_on_fracture_surface
                )
                if outwards_mat is None:
                    outwards_mat = switcher_int
                else:
                    outwards_mat += switcher_int

            ad.internal_boundary_vector_to_outwards = pp.ad.Matrix(outwards_mat)

        # Projections between interfaces and subdomains
        ad.mortar_projections_vector = pp.ad.MortarProjections(
            subdomains=subdomains,
            interfaces=fracture_matrix_interfaces,
            mdg=self.mdg,
            nd=self.nd,
        )

        # Prolongation and restriction between subdomain subsets and full subdomain list
        ad.subdomain_projections_vector = pp.ad.SubdomainProjections(
            subdomains=subdomains, nd=self.nd
        )

    def _displacement_jump(
        self, fracture_subdomains: List[pp.Grid], previous_timestep=False
    ) -> pp.ad.Operator:
        """Construct AD Operator representing jump of interface displacement.

        Args:
            fracture_subdomains
                List of fracture subdomains.

        Returns:
            rotated_jumps
                Operator representing jump across all fractures.

        """
        ad = self._ad
        if previous_timestep:
            interface_displacement = ad.interface_displacement.previous_timestep()
        else:
            interface_displacement = ad.interface_displacement
        rotated_jumps: pp.ad.Operator = (
            ad.local_fracture_coord_transformation
            * ad.subdomain_projections_vector.cell_restriction(fracture_subdomains)
            * ad.mortar_projections_vector.mortar_to_secondary_avg
            * ad.mortar_projections_vector.sign_of_mortar_sides
            * interface_displacement
        )
        rotated_jumps.set_name("Rotated displacement jump")
        return rotated_jumps

    def _contact_mechanics_normal_equation(
        self,
        fracture_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Contact mechanics equation for the normal constraints.

        Parameters
        ----------
        fracture_subdomains : List[pp.Grid]
            List of fracture subdomains.

        Returns
        -------
        equation : pp.ad.Operator
            Contact mechanics equation for the normal constraints.

        """
        numerical_c_n = pp.ad.ParameterMatrix(
            self.mechanics_parameter_key,
            array_keyword="c_num_normal",
            subdomains=fracture_subdomains,
        )

        T_n: pp.ad.Operator = self._ad.normal_component_frac * self._ad.contact_traction

        MaxAd = pp.ad.Function(pp.ad.maximum, "max_function")
        zeros_frac = pp.ad.Array(np.zeros(self._num_frac_cells))
        u_n: pp.ad.Operator = self._ad.normal_component_frac * self._displacement_jump(
            fracture_subdomains
        )
        equation: pp.ad.Operator = T_n + MaxAd(
            (-1) * T_n - numerical_c_n * (u_n - self._gap(fracture_subdomains)),
            zeros_frac,
        )
        return equation

    def _friction_bound(
        self,
        fracture_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Ad operator representing the friction bound.

        Parameters
        ----------
        fracture_subdomains : List[pp.Grid]
            List of fracture subdomains.

        Returns
        -------
        bound : pp.ad.Operator
            Friction bound operator.

        Note:
            Hueeber and Berge use
            (-1) * friction_coefficient * (T_n + c_n * (u_n - gap))
            The argument is that with this choice, "the Newton-type iteration
            automatically takes the form of an active set method" (Hueber's
            thesis p. 104). Since we abandon the sets in the ad-based
             implementation, the simpler form more closely related to the
             Coulomb friction law is preferred.
        """
        friction_coefficient = pp.ad.ParameterMatrix(
            self.mechanics_parameter_key,
            array_keyword="friction_coefficient",
            subdomains=fracture_subdomains,
        )
        ad = self._ad
        T_n: pp.ad.Operator = ad.normal_component_frac * ad.contact_traction
        bound: pp.ad.Operator = (-1) * friction_coefficient * T_n
        bound.set_name("friction_bound")
        return bound

    def _gap(
        self,
        fracture_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:
        """Gap function.

        The gap function includes an initial (constant) value and shear dilation.
        It depends linearly on the norm of tangential displacement jump:
            g = g_0 + tan(dilation_angle) * norm([[u]]_t)

        Parameters
        ----------
        fracture_subdomains : List[pp.Grid]
            List of fracture subdomains.

        Returns
        -------
        gap : pp.ad.Operator
            Gap function representing the distance between the fracture
            interfaces when in mechanical contact.

        """
        initial_gap: pp.ad.Operator = pp.ad.ParameterArray(
            self.mechanics_parameter_key,
            array_keyword="initial_gap",
            subdomains=fracture_subdomains,
        )
        angle: pp.ad.Operator = pp.ad.ParameterArray(
            self.mechanics_parameter_key,
            array_keyword="dilation_angle",
            subdomains=fracture_subdomains,
        )
        Norm = pp.ad.Function(
            partial(pp.ad.functions.l2_norm, self.nd - 1), "norm_function"
        )
        Tan = pp.ad.Function(pp.ad.functions.tan, "tan_function")
        shear_dilation: pp.ad.Operator = Tan(angle) * Norm(
            self._ad.tangential_component_frac
            * self._displacement_jump(fracture_subdomains)
        )

        gap = initial_gap + shear_dilation
        gap.set_name("gap_with_shear_dilation")
        return gap

    def _contact_mechanics_tangential_equation(
        self,
        fracture_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:
        """
        Contact mechanics equation for the tangential constraints.

        The function reads
            C_t = max(b_p, ||T_t+c_t u_t||) T_t - max(0, b_p) (T_t+c_t u_t)
        with u being displacement jump increments, t denoting tangential
        component and b_p the friction bound.

        For b_p = 0, the equation C_t = 0 does not in itself imply T_t = 0,
        which is what the contact conditions require. The case is handled
        through the use of a characteristic function.

        Parameters
        ----------
        fracture_subdomains : List[pp.Grid]
            List of fracture subdomains.

        Returns
        -------
        complementary_eq : pp.ad.Operator
            Contact mechanics equation for the tangential constraints.

        """
        ad = self._ad

        # Parameter and constants
        numerical_c_t = pp.ad.ParameterMatrix(
            self.mechanics_parameter_key,
            array_keyword="c_num_tangential",
            subdomains=fracture_subdomains,
        )
        ones_frac = pp.ad.Array(np.ones(self._num_frac_cells * (self.nd - 1)))
        zeros_frac = pp.ad.Array(np.zeros(self._num_frac_cells))

        # Functions
        MaxAd = pp.ad.Function(pp.ad.maximum, "max_function")
        NormAd = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        tol = 1e-5
        Characteristic = pp.ad.Function(
            partial(pp.ad.functions.characteristic_function, tol),
            "characteristic_function_for_zero_normal_traction",
        )

        # Variables
        T_t = ad.tangential_component_frac * ad.contact_traction
        u_t_prime = ad.tangential_component_frac * (
            self._displacement_jump(fracture_subdomains)
            - self._displacement_jump(fracture_subdomains, previous_timestep=True)
        )
        u_t_prime.set_name("u_tau_increment")

        # Combine the above into expressions that enter the equation
        tangential_sum = T_t + numerical_c_t * u_t_prime

        norm_tangential_sum = NormAd(tangential_sum)
        norm_tangential_sum.set_name("norm_tangential")

        b_p = MaxAd(self._friction_bound(fracture_subdomains), zeros_frac)
        b_p.set_name("bp")

        bp_tang = (ad.normal_to_tangential_frac * b_p) * tangential_sum

        maxbp_abs = ad.normal_to_tangential_frac * MaxAd(b_p, norm_tangential_sum)
        characteristic: pp.ad.Operator = ad.normal_to_tangential_frac * Characteristic(
            b_p
        )
        characteristic.set_name("characteristic_function")

        # Compose the equation itself.
        # The last term handles the case bound=0, in which case T_t = 0 cannot
        # be deduced from the standard version of the complementary function
        # (i.e. without the characteristic function). Filter out the other terms
        # in this case to improve convergence
        complementary_eq: pp.ad.Operator = (ones_frac - characteristic) * (
            bp_tang - maxbp_abs * T_t
        ) + characteristic * (T_t)
        return complementary_eq

    def _momentum_balance_equation(
        self,
        matrix_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:
        """Define the momentum balance equation in the matrix subdomain.

        Parameters
        ----------
        matrix_subdomains : List[pp.Grid]
            List of N-dimensional subdomains, usually with a single entry.

        Returns
        -------
        momentum_eq : pp.ad.Operator
            The momentum balance equation.

        """

        div = pp.ad.Divergence(subdomains=matrix_subdomains, dim=self.nd)

        stress = self._stress(matrix_subdomains)
        # Source term typically represents gravity
        source_term = pp.ad.ParameterArray(
            self.mechanics_parameter_key,
            "source",
            matrix_subdomains,
        )
        momentum_eq: pp.ad.Operator = div * stress - source_term
        return momentum_eq

    def _stress(
        self,
        matrix_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:
        """Ad representation of mechanical stress.

        Parameters
        ----------
        matrix_subdomains : List[pp.Grid]
            List of N-dimensional subdomains, usually with a single entry.

        Returns
        -------
        stress : pp.ad.Operator
            Stress operator.

        """
        ad = self._ad
        discr = pp.ad.MpsaAd(self.mechanics_parameter_key, matrix_subdomains)
        bc = pp.ad.BoundaryCondition(
            self.mechanics_parameter_key, subdomains=matrix_subdomains
        )
        stress = (
            discr.stress * ad.displacement
            + discr.bound_stress * bc
            + discr.bound_stress
            * ad.subdomain_projections_vector.face_restriction(matrix_subdomains)
            * ad.mortar_projections_vector.mortar_to_primary_avg
            * ad.interface_displacement
        )
        # Name operator. "Mechanical" is intended as a distinction from e.g.
        # "thermal stress" (another component of THM stress) and
        # "poromechanical stress" (combination of mechanical and pressure)
        stress.set_name("mechanical_stress")
        return stress

    def _force_balance_equation(
        self,
        matrix_subdomains: List[pp.Grid],
        fracture_subdomains: List[pp.Grid],
        interfaces: List[pp.MortarGrid],
    ) -> pp.ad.Operator:
        """Force balance equation at matrix-fracture interfaces.

        Args:
            matrix_subdomains : List[pp.Grid]
                Matrix subdomains, expected to be of length 1.
            fracture_subdomains: List[pp.Grid].
                Fracture subdomains.
            interfaces: List[pp.MortarGrid]

        Returns:
            Operator representing the force balance equation.

        """
        ad = self._ad
        if len(matrix_subdomains) > 1:
            raise NotImplementedError("Implementation assumes a single Nd grid")

        # Contact traction from primary grid and mortar displacements (via primary grid)
        contact_from_primary_mortar = (
            ad.mortar_projections_vector.primary_to_mortar_int
            * ad.subdomain_projections_vector.face_prolongation(matrix_subdomains)
            * ad.internal_boundary_vector_to_outwards
            * self._stress(matrix_subdomains)
        )
        # Traction from the actual contact force.
        contact_from_secondary = (
            ad.mortar_projections_vector.sign_of_mortar_sides
            * ad.mortar_projections_vector.secondary_to_mortar_int
            * ad.subdomain_projections_vector.cell_prolongation(fracture_subdomains)
            * ad.local_fracture_coord_transformation.transpose()
            * ad.contact_force
        )
        force_balance_eq: pp.ad.Operator = (
            contact_from_primary_mortar + contact_from_secondary
        )
        return force_balance_eq

    # Methods for discretization, numerical issues etc.

    def _discretize(self) -> None:
        """Discretize all terms"""
        tic = time.time()
        logger.info("Discretize")
        if self._use_ad:
            self._eq_manager.discretize(self.mdg)
        else:
            self.assembler.discretize()
        logger.info("Done. Elapsed time {}".format(time.time() - tic))

    def _initialize_linear_solver(self) -> None:
        solver = self.linear_solver

        if solver == "direct":
            """In theory, it should be possible to instruct SuperLU to reuse the
            symbolic factorization from one iteration to the next. However, it seems
            the scipy wrapper around SuperLU has not implemented the necessary
            functionality, as discussed in

                https://github.com/scipy/scipy/issues/8227

            We will therefore pass here, and pay the price of long computation times.
            """
            self.linear_solver = "direct"
        elif solver == "pypardiso":
            self.linear_solver = "pypardiso"
        else:
            raise ValueError(f"Unknown linear solver {solver}")

    def _is_nonlinear_problem(self) -> bool:
        """
        If there is no fracture, the problem is usually linear.
        Overwrite this function if e.g. parameter nonlinearities are included.
        """
        return self.mdg.dim_min() < self.nd
