"""
This is a setup class for solving the Biot equations with contact mechanics at the fractures.

The class ContactMechanicsBiot inherits from ContactMechanics, which is a model for
the purely mechanical problem with contact conditions on the fractures. Here, we
expand to a model where the displacement solution is coupled to a scalar variable, e.g.
pressure (Biot equations) or temperature. Parameters, variables and discretizations are
set in the model class, and the problem may be solved using run_biot.

NOTE: This module should be considered an experimental feature, which may
undergo major changes on little notice.

"""
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.models.contact_mechanics_model as contact_model
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations

# Module-wide logger
logger = logging.getLogger(__name__)
module_sections = ["models", "numerics"]


class ContactMechanicsBiot(contact_model.ContactMechanics):
    """This is a shell class for poro-elastic contact mechanics problems.

    Setting up such problems requires a lot of boilerplate definitions of variables,
    parameters and discretizations. This class is intended to provide a standardized
    setup, with all discretizations in place and reasonable parameter and boundary
    values. The intended use is to inherit from this class, and do the necessary
    modifications and specifications for the problem to be fully defined. The minimal
    adjustment needed is to specify the method create_grid().

    Attributes:
        time (float): Current time.
        time_step (float): Size of an individual time step
        end_time (float): Time at which the simulation should stop.

        displacement_variable (str): Name assigned to the displacement variable in the
            highest-dimensional subdomain. Will be used throughout the simulations,
            including in Paraview export.
        mortar_displacement_variable (str): Name assigned to the displacement variable
            on the fracture walls. Will be used throughout the simulations, including in
            Paraview export.
        contact_traction_variable (str): Name assigned to the variable for contact
            forces in the fracture. Will be used throughout the simulations, including
            in Paraview export.
        scalar_variable (str): Name assigned to the scalar variable (say, temperature
            or pressure). Will be used throughout the simulations, including
            in Paraview export.
        mortar scalar_variable (str): Name assigned to the interface scalar variable
            representing flux between grids. Will be used throughout the simulations,
            including in Paraview export.

        mechanics_parameter_key (str): Keyword used to define parameters and
            discretizations for the mechanics problem.
        scalar_parameter_key (str): Keyword used to define parameters and
            discretizations for the flow problem.

        params (dict): Dictionary of parameters used to control the solution procedure.
        viz_folder_name (str): Folder for visualization export.
        gb (pp.GridBucket): Mixed-dimensional grid. Should be set by a method
            create_grid which should be provided by the user.
        convergence_status (bool): Whether the non-linear iterations has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'
        scalar_scale (float): Scaling coefficient for the scalar variable. Can be used
            to get comparable size of the mechanical and flow problem.
        scalar_scale (float): Scaling coefficient for the vector variable. Can be used
            to get comparable size of the mechanical and flow problem.
        subtract_fracture_pressure (bool): If True (default) the scalar variable will be
            interpreted as a pressure, and contribute a force to the fracture walls from
            the lower-dimensional grid.

    Except from the grid, all attributes are given natural values at initialization of
    the class.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params)

        # Time
        self.time: float = 0
        self.time_step: float = 1
        self.end_time: float = 1

        # Temperature
        self.scalar_variable: str = "p"
        self.mortar_scalar_variable: str = "mortar_" + self.scalar_variable
        self.scalar_coupling_term: str = "robin_" + self.scalar_variable
        self.scalar_parameter_key: str = "flow"

        # Scaling coefficients
        self.scalar_scale: float = 1
        self.length_scale: float = 1

        # Whether or not to subtract the fracture pressure contribution for the contact
        # traction. This should be done if the scalar variable is pressure, but not for
        # temperature. See assign_discretizations
        self.subtract_fracture_pressure: bool = True

    @pp.time_logger(sections=module_sections)
    def before_newton_loop(self) -> None:
        """Will be run before entering a Newton loop.
        E.g.
           Discretize time-dependent quantities etc.
           Update time-dependent parameters (captured by assembly).
        """
        self._set_parameters()

    @pp.time_logger(sections=module_sections)
    def before_newton_iteration(self) -> None:
        # Re-discretize the nonlinear term
        filt = pp.assembler_filters.ListFilter(term_list=[self.friction_coupling_term])
        if self._use_ad:
            self._eq_manager.equations[1].discretize(self.gb)
        else:
            self.assembler.discretize(filt=filt)

    @pp.time_logger(sections=module_sections)
    def after_newton_iteration(self, solution: np.ndarray) -> None:
        self._update_iterate(solution)

    @pp.time_logger(sections=module_sections)
    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)
        self._save_mechanical_bc_values()

    @pp.time_logger(sections=module_sections)
    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        raise ValueError("Newton iterations did not converge")

    @pp.time_logger(sections=module_sections)
    def reconstruct_stress(self, previous_iterate: bool = False) -> None:
        """
        Compute the stress in the highest-dimensional grid based on the displacement
        and pressure states in that grid, adjacent interfaces and global boundary
        conditions.

        The stress is stored in the data dictionary of the highest-dimensional grid,
        in [pp.STATE]['stress'].

        Parameters:
            previous_iterate (boolean, optional): If True, use values from previous
                iteration to compute the stress. Defaults to False.

        """
        # First the mechanical part of the stress
        super().reconstruct_stress(previous_iterate)

        g = self._nd_grid()
        d = self.gb.node_props(g)

        matrix_dictionary: Dict[str, sps.spmatrix] = d[pp.DISCRETIZATION_MATRICES][
            self.mechanics_parameter_key
        ]
        mpsa = pp.Biot(self.mechanics_parameter_key)
        if previous_iterate:
            p = d[pp.STATE][pp.ITERATE][self.scalar_variable]
        else:
            p = d[pp.STATE][self.scalar_variable]

        # Stress contribution from the scalar variable
        d[pp.STATE]["stress"] += matrix_dictionary[mpsa.grad_p_matrix_key] * p

        # Is it correct there is no contribution from the global boundary conditions?

    # Methods for setting parametrs etc.

    @pp.time_logger(sections=module_sections)
    def _set_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        self._set_scalar_parameters()
        self._set_mechanics_parameters()

    @pp.time_logger(sections=module_sections)
    def _set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self._Nd:
                # Rock parameters
                lam = np.ones(g.num_cells) / self.scalar_scale
                mu = np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                # Define boundary condition
                bc = self._bc_type_mechanics(g)
                # BC and source values
                bc_val = self._bc_values_mechanics(g)
                source_val = self._source_mechanics(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_val,
                        "source": source_val,
                        "fourth_order_tensor": C,
                        "time_step": self.time_step,
                        "biot_alpha": self._biot_alpha(g),
                        "p_reference": np.zeros(g.num_cells),
                    },
                )

            elif g.dim == self._Nd - 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction, "time_step": self.time_step},
                )

        for _, d in gb.edges():
            mg: pp.MortarGrid = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)

    @pp.time_logger(sections=module_sections)
    def _set_scalar_parameters(self) -> None:
        tensor_scale = self.scalar_scale / self.length_scale ** 2
        kappa = 1 * tensor_scale
        mass_weight = 1 * self.scalar_scale
        for g, d in self.gb:
            bc = self._bc_type_scalar(g)
            bc_values = self._bc_values_scalar(g)
            source_values = self._source_scalar(g)

            specific_volume = self._specific_volume(g)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            alpha = self._biot_alpha(g)
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight * specific_volume,
                    "biot_alpha": alpha,
                    "source": source_values,
                    "second_order_tensor": diffusivity,
                    "time_step": self.time_step,
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            a_l = self._aperture(g_l)
            # Take trace of and then project specific volumes from g_h
            v_h = (
                mg.primary_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self._specific_volume(g_h)
            )
            # Division by a/2 may be thought of as taking the gradient in the normal
            # direction of the fracture.
            normal_diffusivity = kappa * 2 / (mg.secondary_to_mortar_avg() * a_l)
            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_h
            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {"normal_diffusivity": normal_diffusivity},
            )

    @pp.time_logger(sections=module_sections)
    def _bc_type_mechanics(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        # Use parent class method for mechanics
        return super()._bc_type(g)

    @pp.time_logger(sections=module_sections)
    def _bc_type_scalar(self, g: pp.Grid) -> pp.BoundaryCondition:
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    @pp.time_logger(sections=module_sections)
    def _bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        """
        Note that Dirichlet values should be divided by length_scale, and Neumann values
        by scalar_scale.
        """
        # Set the boundary values
        return super()._bc_values(g)

    @pp.time_logger(sections=module_sections)
    def _bc_values_scalar(self, g: pp.Grid) -> np.ndarray:
        """
        Note that Dirichlet values should be divided by scalar_scale.
        """
        return np.zeros(g.num_faces)

    @pp.time_logger(sections=module_sections)
    def _source_mechanics(self, g: pp.Grid) -> np.ndarray:
        return super()._source(g)

    @pp.time_logger(sections=module_sections)
    def _source_scalar(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_cells)

    @pp.time_logger(sections=module_sections)
    def _biot_alpha(self, g: pp.Grid) -> float:
        return 1

    @pp.time_logger(sections=module_sections)
    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of co-dimension 2 and 3.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self._Nd:
            aperture *= 0.1
        return aperture

    @pp.time_logger(sections=module_sections)
    def _specific_volume(self, g: pp.Grid) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in codimensions 2 and 3.
        """
        a = self._aperture(g)
        return np.power(a, self._Nd - g.dim)

    @pp.time_logger(sections=module_sections)
    def _assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether or not to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        if not self._use_ad:

            # Shorthand
            key_s, key_m = self.scalar_parameter_key, self.mechanics_parameter_key
            var_s, var_d = self.scalar_variable, self.displacement_variable

            # Define discretization
            # For the Nd domain we solve linear elasticity with mpsa.
            mpsa = pp.Mpsa(key_m)
            empty_discr = pp.VoidDiscretization(key_m, ndof_cell=self._Nd)
            # Scalar discretizations (all dimensions)
            diff_disc_s = IE_discretizations.ImplicitMpfa(key_s)
            mass_disc_s = IE_discretizations.ImplicitMassMatrix(key_s, var_s)
            source_disc_s = pp.ScalarSource(key_s)
            # Coupling discretizations
            # All dimensions
            div_u_disc = pp.DivU(
                key_m,
                key_s,
                variable=var_d,
                mortar_variable=self.mortar_displacement_variable,
            )
            # Nd
            grad_p_disc = pp.GradP(key_m)
            stabilization_disc_s = pp.BiotStabilization(key_s, var_s)

            # Assign node discretizations
            for g, d in self.gb:
                if g.dim == self._Nd:
                    d[pp.DISCRETIZATION] = {
                        var_d: {"mpsa": mpsa},
                        var_s: {
                            "diffusion": diff_disc_s,
                            "mass": mass_disc_s,
                            "stabilization": stabilization_disc_s,
                            "source": source_disc_s,
                        },
                        var_d + "_" + var_s: {"grad_p": grad_p_disc},
                        var_s + "_" + var_d: {"div_u": div_u_disc},
                    }

                elif g.dim == self._Nd - 1:
                    d[pp.DISCRETIZATION] = {
                        self.contact_traction_variable: {"empty": empty_discr},
                        var_s: {
                            "diffusion": diff_disc_s,
                            "mass": mass_disc_s,
                            "source": source_disc_s,
                        },
                    }
                else:
                    d[pp.DISCRETIZATION] = {
                        var_s: {
                            "diffusion": diff_disc_s,
                            "mass": mass_disc_s,
                            "source": source_disc_s,
                        }
                    }

            # Define edge discretizations for the mortar grid
            contact_law = pp.ColoumbContact(
                self.mechanics_parameter_key, self._Nd, mpsa
            )
            contact_discr = pp.PrimalContactCoupling(
                self.mechanics_parameter_key, mpsa, contact_law
            )
            # Account for the mortar displacements effect on scalar balance in the matrix,
            # as an internal boundary contribution, fracture, aperture changes appear as a
            # source contribution.
            div_u_coupling = pp.DivUCoupling(
                self.displacement_variable, div_u_disc, div_u_disc
            )
            # Account for the pressure contributions to the force balance on the fracture
            # (see contact_discr).
            # This discretization needs the keyword used to store the grad p discretization:
            grad_p_key = key_m
            matrix_scalar_to_force_balance = pp.MatrixScalarToForceBalance(
                grad_p_key, mass_disc_s, mass_disc_s
            )
            if self.subtract_fracture_pressure:
                fracture_scalar_to_force_balance = pp.FractureScalarToForceBalance(
                    mass_disc_s, mass_disc_s
                )

            for e, d in self.gb.edges():
                g_l, g_h = self.gb.nodes_of_edge(e)

                if g_h.dim == self._Nd:
                    d[pp.COUPLING_DISCRETIZATION] = {
                        self.friction_coupling_term: {
                            g_h: (var_d, "mpsa"),
                            g_l: (self.contact_traction_variable, "empty"),
                            (g_h, g_l): (
                                self.mortar_displacement_variable,
                                contact_discr,
                            ),
                        },
                        self.scalar_coupling_term: {
                            g_h: (var_s, "diffusion"),
                            g_l: (var_s, "diffusion"),
                            e: (
                                self.mortar_scalar_variable,
                                pp.RobinCoupling(key_s, diff_disc_s),
                            ),
                        },
                        "div_u_coupling": {
                            g_h: (
                                var_s,
                                "mass",
                            ),  # This is really the div_u, but this is not implemented
                            g_l: (var_s, "mass"),
                            e: (self.mortar_displacement_variable, div_u_coupling),
                        },
                        "matrix_scalar_to_force_balance": {
                            g_h: (var_s, "mass"),
                            g_l: (var_s, "mass"),
                            e: (
                                self.mortar_displacement_variable,
                                matrix_scalar_to_force_balance,
                            ),
                        },
                    }
                    if self.subtract_fracture_pressure:
                        d[pp.COUPLING_DISCRETIZATION].update(
                            {
                                "fracture_scalar_to_force_balance": {
                                    g_h: (var_s, "mass"),
                                    g_l: (var_s, "mass"),
                                    e: (
                                        self.mortar_displacement_variable,
                                        fracture_scalar_to_force_balance,
                                    ),
                                }
                            }
                        )
                else:
                    d[pp.COUPLING_DISCRETIZATION] = {
                        self.scalar_coupling_term: {
                            g_h: (var_s, "diffusion"),
                            g_l: (var_s, "diffusion"),
                            e: (
                                self.mortar_scalar_variable,
                                pp.RobinCoupling(key_s, diff_disc_s),
                            ),
                        }
                    }

        else:

            gb = self.gb
            Nd = self._Nd
            dof_manager = pp.DofManager(gb)
            eq_manager = pp.ad.EquationManager(gb, dof_manager)

            g_primary = gb.grids_of_dimension(Nd)[0]
            g_frac = gb.grids_of_dimension(Nd - 1).tolist()

            grid_list = [
                g_primary,
                *g_frac,
                *gb.grids_of_dimension(Nd - 2),
                *gb.grids_of_dimension(Nd - 3),
            ]

            if len(gb.grids_of_dimension(Nd)) != 1:
                raise NotImplementedError("This will require further work")

            edge_list_highest = [(g_primary, g) for g in g_frac]
            edge_list = [e for e, _ in gb.edges()]

            mortar_proj_scalar = pp.ad.MortarProjections(edges=edge_list, gb=gb, nd=1)
            mortar_proj_vector = pp.ad.MortarProjections(
                edges=edge_list_highest, gb=gb, nd=self._Nd
            )
            subdomain_proj_scalar = pp.ad.SubdomainProjections(gb=gb)
            subdomain_proj_vector = pp.ad.SubdomainProjections(gb=gb, nd=self._Nd)

            tangential_normal_proj_list = []
            normal_proj_list = []
            for gf in g_frac:
                proj = gb.node_props(gf, "tangential_normal_projection")
                tangential_normal_proj_list.append(
                    proj.project_tangential_normal(gf.num_cells)
                )
                normal_proj_list.append(proj.project_normal(gf.num_cells))

            tangential_normal_proj = pp.ad.Matrix(
                sps.block_diag(tangential_normal_proj_list)
            )
            normal_proj = pp.ad.Matrix(sps.block_diag(normal_proj_list))

            # Ad representation of discretizations
            mpsa_ad = pp.ad.BiotAd(self.mechanics_parameter_key, g_primary)
            grad_p_ad = pp.ad.GradPAd(self.mechanics_parameter_key, g_primary)

            mpfa_ad = pp.ad.MpfaAd(self.scalar_parameter_key, grid_list)
            mass_ad = pp.ad.MassMatrixAd(self.scalar_parameter_key, grid_list)
            robin_ad = pp.ad.RobinCouplingAd(self.scalar_parameter_key, edge_list)

            div_u_ad = pp.ad.DivUAd(
                self.mechanics_parameter_key,
                grids=g_primary,
                mat_dict_keyword=self.scalar_parameter_key,
            )
            stab_biot_ad = pp.ad.BiotStabilizationAd(
                self.scalar_parameter_key, g_primary
            )

            coloumb_ad = pp.ad.ColoumbContactAd(
                self.mechanics_parameter_key, edge_list_highest
            )

            bc_ad = pp.ad.BoundaryCondition(
                self.mechanics_parameter_key, grids=[g_primary]
            )
            div_vector = pp.ad.Divergence(grids=[g_primary], dim=g_primary.dim)

            # Primary variables on Ad form
            u = eq_manager.variable(g_primary, self.displacement_variable)
            u_mortar = eq_manager.merge_variables(
                [(e, self.mortar_displacement_variable) for e in edge_list_highest]
            )
            contact_force = eq_manager.merge_variables(
                [(g, self.contact_traction_variable) for g in g_frac]
            )
            p = eq_manager.merge_variables(
                [(g, self.scalar_variable) for g in grid_list]
            )
            mortar_flux = eq_manager.merge_variables(
                [(e, self.mortar_scalar_variable) for e in edge_list]
            )

            u_prev = u.previous_timestep()
            u_mortar_prev = u_mortar.previous_timestep()
            p_prev = p.previous_timestep()

            # Reference pressure, corresponding to an initial stress free state
            p_reference = pp.ad.ParameterArray(
                param_keyword=self.mechanics_parameter_key,
                array_keyword="p_reference",
                grids=[g_primary],
                gb=gb,
            )

            # Stress in g_h
            stress = (
                mpsa_ad.stress * u
                + mpsa_ad.bound_stress * bc_ad
                + mpsa_ad.bound_stress
                * subdomain_proj_vector.face_restriction(g_primary)
                * mortar_proj_vector.mortar_to_primary_avg
                * u_mortar
                + grad_p_ad.grad_p
                * subdomain_proj_scalar.cell_restriction(g_primary)
                * p
                # The reference pressure is only defined on g_primary, thus there is no need
                # for a subdomain projection.
                - grad_p_ad.grad_p * p_reference
            )

            momentum_eq = pp.ad.Expression(
                div_vector * stress, dof_manager, "momentuum", grid_order=[g_primary]
            )

            jump = (
                subdomain_proj_vector.cell_restriction(g_frac)
                * mortar_proj_vector.mortar_to_secondary_avg
                * mortar_proj_vector.sign_of_mortar_sides
            )
            jump_rotate = tangential_normal_proj * jump

            # Contact conditions
            num_frac_cells = np.sum([g.num_cells for g in g_frac])

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
            contact_conditions = coloumb_ad.traction * contact_force + jump_discr - rhs
            contact_eq = pp.ad.Expression(
                contact_conditions, dof_manager, "contact", grid_order=g_frac
            )

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
                mortar_proj_vector.primary_to_mortar_int
                * subdomain_proj_vector.face_prolongation(g_primary)
                * sign_switcher
                * stress
            )
            contact_from_secondary = (
                mortar_proj_vector.sign_of_mortar_sides
                * mortar_proj_vector.secondary_to_mortar_int
                * subdomain_proj_vector.cell_prolongation(g_frac)
                * tangential_normal_proj.transpose()
                * contact_force
            )
            if self.subtract_fracture_pressure:
                # This gives an error because of -=

                mat = []

                for e in edge_list_highest:
                    mg = gb.edge_props(e, "mortar_grid")

                    faces_on_fracture_surface = (
                        mg.primary_to_mortar_int().tocsr().indices
                    )
                    sgn, _ = g_primary.signs_and_cells_of_boundary_faces(
                        faces_on_fracture_surface
                    )
                    fracture_normals = g_primary.face_normals[
                        : self._Nd, faces_on_fracture_surface
                    ]
                    outwards_fracture_normals = sgn * fracture_normals

                    data = outwards_fracture_normals.ravel("F")
                    row = np.arange(g_primary.dim * mg.num_cells)
                    col = np.tile(np.arange(mg.num_cells), (g_primary.dim, 1)).ravel(
                        "F"
                    )
                    n_dot_I = sps.csc_matrix((data, (row, col)))
                    # i) The scalar contribution to the contact stress is mapped to
                    # the mortar grid  and multiplied by n \dot I, with n being the
                    # outwards normals on the two sides.
                    # Note that by using different normals for the two sides, we do not need to
                    # adjust the secondary pressure with the corresponding signs by applying
                    # sign_of_mortar_sides as done in PrimalContactCoupling.
                    # iii) The contribution should be subtracted so that we balance the primary
                    # forces by
                    # T_contact - n dot I p,
                    # hence the minus.
                    mat.append(n_dot_I * mg.secondary_to_mortar_int(nd=1))
                # May need to do this as for tangential projections, additive that is
                normal_matrix = pp.ad.Matrix(sps.block_diag(mat))

                p_frac = subdomain_proj_scalar.cell_restriction(g_frac) * p

                contact_from_secondary2 = normal_matrix * p_frac
            force_balance_eq = pp.ad.Expression(
                contact_from_primary_mortar
                + contact_from_secondary
                + contact_from_secondary2,
                dof_manager,
                "force_balance",
                grid_order=edge_list_highest,
            )

            bc_val_scalar = pp.ad.BoundaryCondition(
                self.scalar_parameter_key, grid_list
            )

            div_scalar = pp.ad.Divergence(grids=grid_list)

            dt = self.time_step

            # FIXME: Need bc for div_u term, including previous time step
            accumulation_primary = (
                div_u_ad.div_u * (u - u_prev)
                + stab_biot_ad.stabilization
                * subdomain_proj_scalar.cell_restriction(g_primary)
                * (p - p_prev)
                + div_u_ad.bound_div_u
                * subdomain_proj_vector.face_restriction(g_primary)
                * mortar_proj_vector.mortar_to_primary_int
                * (u_mortar - u_mortar_prev)
            )

            # Accumulation term on the fractures.
            frac_vol = np.hstack([g.cell_volumes for g in g_frac])
            vol_mat = pp.ad.Matrix(
                sps.dia_matrix((frac_vol, 0), shape=(num_frac_cells, num_frac_cells))
            )
            accumulation_fracs = (
                vol_mat * normal_proj * jump * (u_mortar - u_mortar_prev)
            )

            accumulation_all = mass_ad.mass * (p - p_prev)

            flux = (
                mpfa_ad.flux * p
                + mpfa_ad.bound_flux * bc_val_scalar
                + mpfa_ad.bound_flux
                * mortar_proj_scalar.mortar_to_primary_int
                * mortar_flux
            )
            flow_md = (
                dt
                * (  # Time scaling of flux terms, both inter-dimensional and from
                    # the higher dimension
                    div_scalar * flux
                    - mortar_proj_scalar.mortar_to_secondary_int * mortar_flux
                )
                + accumulation_all
                + subdomain_proj_scalar.cell_prolongation(g_primary)
                * accumulation_primary
                + subdomain_proj_scalar.cell_prolongation(g_frac) * accumulation_fracs
            )

            interface_flow_eq = robin_ad.mortar_scaling * (
                mortar_proj_scalar.primary_to_mortar_avg
                * mpfa_ad.bound_pressure_cell
                * p
                + mortar_proj_scalar.primary_to_mortar_avg
                * mpfa_ad.bound_pressure_face
                * (
                    mortar_proj_scalar.mortar_to_primary_int * mortar_flux
                    + bc_val_scalar
                )
                - mortar_proj_scalar.secondary_to_mortar_avg * p
                + robin_ad.mortar_discr * mortar_flux
            )

            flow_eq = pp.ad.Expression(
                flow_md, dof_manager, "flow on nodes", grid_order=grid_list
            )
            interface_eq = pp.ad.Expression(
                interface_flow_eq,
                dof_manager,
                "flow on interface",
                grid_order=edge_list,
            )

            eq_manager.equations += [
                momentum_eq,
                contact_eq,
                force_balance_eq,
                flow_eq,
                interface_eq,
            ]
            self._eq_manager = eq_manager

    @pp.time_logger(sections=module_sections)
    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        # First for the nodes
        for g, d in self.gb:
            if g.dim == self._Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.displacement_variable: {"cells": self._Nd},
                    self.scalar_variable: {"cells": 1},
                }
            elif g.dim == self._Nd - 1:
                d[pp.PRIMARY_VARIABLES] = {
                    self.contact_traction_variable: {"cells": self._Nd},
                    self.scalar_variable: {"cells": 1},
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {self.scalar_variable: {"cells": 1}}

        # Then for the edges
        for e, d in self.gb.edges():
            _, g_h = self.gb.nodes_of_edge(e)

            if g_h.dim == self._Nd:
                d[pp.PRIMARY_VARIABLES] = {
                    self.mortar_displacement_variable: {"cells": self._Nd},
                    self.mortar_scalar_variable: {"cells": 1},
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {self.mortar_scalar_variable: {"cells": 1}}

    @pp.time_logger(sections=module_sections)
    def _initial_condition(self) -> None:
        """
        Initial guess for Newton iteration, scalar variable and bc_values (for time
        discretization).
        """
        super()._initial_condition()

        for g, d in self.gb:
            # Initial value for the scalar variable.
            initial_scalar_value = np.zeros(g.num_cells)
            d[pp.STATE].update({self.scalar_variable: initial_scalar_value})

            d[pp.STATE][pp.ITERATE].update(
                {self.scalar_variable: initial_scalar_value.copy()}
            )
            if g.dim == self._Nd:
                bc_values = self._bc_values_mechanics(g)
                mech_dict = {"bc_values": bc_values}
                d[pp.STATE].update({self.mechanics_parameter_key: mech_dict})

        for _, d in self.gb.edges():
            mg = d["mortar_grid"]
            initial_value = np.zeros(mg.num_cells)
            d[pp.STATE][self.mortar_scalar_variable] = initial_value
            d[pp.STATE][pp.ITERATE][self.mortar_scalar_variable] = initial_value.copy()

    def _update_iterate(self, solution: np.ndarray) -> None:
        super()._update_iterate(solution)

        cumulative = self._use_ad

        dof_manager = self.dof_manager
        variable_names = []
        for pair in dof_manager.block_dof.keys():
            variable_names.append(pair[1])

        dof = np.cumsum(np.append(0, np.asarray(dof_manager.full_dof)))

        for var_name in set(variable_names):
            for pair, bi in dof_manager.block_dof.items():
                g = pair[0]
                name = pair[1]
                if name != var_name:
                    continue

                local_sol = solution[dof[bi] : dof[bi + 1]].copy()

                if isinstance(g, tuple):
                    # This is really an edge
                    if name == self.mortar_scalar_variable:
                        data = self.gb.edge_props(g)
                        if cumulative:
                            data[pp.STATE][pp.ITERATE][name] += local_sol
                        else:
                            data[pp.STATE][pp.ITERATE][name] = local_sol
                else:  # This is a node
                    if name == self.scalar_variable:
                        data = self.gb.node_props(g)
                        if cumulative:
                            data[pp.STATE][pp.ITERATE][name] += local_sol
                        else:
                            data[pp.STATE][pp.ITERATE][name] = local_sol

    @pp.time_logger(sections=module_sections)
    def _save_mechanical_bc_values(self) -> None:
        """
        The div_u term uses the mechanical bc values for both current and previous time
        step. In the case of time dependent bc values, these must be updated. As this
        is very easy to overlook, we do it by default.
        """
        key = self.mechanics_parameter_key
        g = self.gb.grids_of_dimension(self._Nd)[0]
        d = self.gb.node_props(g)
        d[pp.STATE][key]["bc_values"] = d[pp.PARAMETERS][key]["bc_values"].copy()

    # Methods for discretization etc.

    @pp.time_logger(sections=module_sections)
    def _discretize(self) -> None:
        """Discretize all terms"""
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(self.gb)

        if not hasattr(self, "assembler"):
            self.assembler = pp.Assembler(self.gb, self.dof_manager)

        g_max = self.gb.grids_of_dimension(self._Nd)[0]

        tic = time.time()
        logger.info("Discretize")

        if self._use_ad:
            self._eq_manager.discretize(self.gb)
        else:
            # Discretization is a bit cumbersome, as the Biot discetization removes the
            # one-to-one correspondence between discretization objects and blocks in
            # the matrix.
            # First, Discretize with the biot class
            self._discretize_biot()

            # Next, discretize term on the matrix grid not covered by the Biot discretization,
            # i.e. the diffusion, mass and source terms
            filt = pp.assembler_filters.ListFilter(
                grid_list=[g_max], term_list=["source", "mass", "diffusion"]
            )
            self.assembler.discretize(filt=filt)

            # Build a list of all edges, and all couplings
            edge_list: List[
                Union[
                    Tuple[pp.Grid, pp.Grid],
                    Tuple[pp.Grid, pp.Grid, Tuple[pp.Grid, pp.Grid]],
                ]
            ] = []
            for e, _ in self.gb.edges():
                edge_list.append(e)
                edge_list.append((e[0], e[1], e))
            if len(edge_list) > 0:
                filt = pp.assembler_filters.ListFilter(grid_list=edge_list)  # type: ignore
                self.assembler.discretize(filt=filt)

            # Finally, discretize terms on the lower-dimensional grids. This can be done
            # in the traditional way, as there is no Biot discretization here.
            for dim in range(0, self._Nd):
                grid_list = self.gb.grids_of_dimension(dim)
                if len(grid_list) > 0:
                    filt = pp.assembler_filters.ListFilter(grid_list=grid_list)
                    self.assembler.discretize(filt=filt)

        logger.info("Done. Elapsed time {}".format(time.time() - tic))

    @pp.time_logger(sections=module_sections)
    def _initialize_linear_solver(self) -> None:

        solver = self.params.get("linear_solver", "direct")

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
            raise ValueError("unknown linear solver " + solver)

    @pp.time_logger(sections=module_sections)
    def _discretize_biot(self, update_after_geometry_change: bool = False) -> None:
        """
        To save computational time, the full Biot equation (without contact mechanics)
        is discretized once. This is to avoid computing the same terms multiple times.
        """
        g = self._nd_grid()
        d = self.gb.node_props(g)
        biot = pp.Biot(
            mechanics_keyword=self.mechanics_parameter_key,
            flow_keyword=self.scalar_parameter_key,
            vector_variable=self.displacement_variable,
            scalar_variable=self.scalar_variable,
        )
        if update_after_geometry_change:
            # This is primary indented for rediscretization after fracture propagation.
            biot.update_discretization(g, d)
        else:
            biot.discretize(g, d)
