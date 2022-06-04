"""
This is a setup class for solving the Biot equations with contact mechanics at the fractures.

The class ContactMechanicsBiot inherits from ContactMechanics, which is a model for
the purely mechanical problem with contact conditions on the fractures. Here, we
expand to a model where the displacement solution is coupled to a scalar variable, e.g.
pressure (Biot equations) or temperature. Parameters, variables and discretizations are
set in the model class, and the problem may be solved using run_biot.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations

# Module-wide logger
logger = logging.getLogger(__name__)
module_sections = ["models", "numerics"]


class ContactMechanicsBiotAdObjects(
    pp.models.contact_mechanics_model.ContactMechanicsAdObjects
):
    """Storage class for ad related objects.

    Stored objects include variables, compound ad operators and projections.
    """

    pressure: pp.ad.Variable
    interface_flux: pp.ad.Variable
    subdomain_projections_scalar: pp.ad.SubdomainProjections
    mortar_projections_scalar: pp.ad.MortarProjections
    time_step: pp.ad.Scalar
    flux_discretization: Union[pp.ad.MpfaAd, pp.ad.TpfaAd]


class ContactMechanicsBiot(pp.ContactMechanics):
    """This is a shell class for poroelastic contact mechanics problems.

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
        convergence_status (bool): Whether the non-linear iterations have converged.
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

    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params)

        # Time
        self.time: float = 0
        self.time_step: float = self.params.get("time_step", 1.0)
        self.end_time: float = self.params.get("end_time", 1.0)

        # Temperature
        self.scalar_variable: str = "p"
        self.mortar_scalar_variable: str = "mortar_" + self.scalar_variable
        self.scalar_coupling_term: str = "robin_" + self.scalar_variable
        self.scalar_parameter_key: str = "flow"

        # Scaling coefficients
        self.scalar_scale: float = 1.0
        self.length_scale: float = 1.0

        # Whether to subtract the fracture pressure contribution for the contact
        # traction. This should be done if the scalar variable is pressure, but not for
        # temperature. See assign_discretizations
        self.subtract_fracture_pressure: bool = True

    def before_newton_loop(self) -> None:
        """Will be run before entering a Newton loop.
        E.g.
           Discretize time-dependent quantities etc.
           Update time-dependent parameters (captured by assembly).
        """
        super().before_newton_loop()
        self._set_parameters()

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        super().after_newton_convergence(solution, errors, iteration_counter)
        self._save_mechanical_bc_values()

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

    # Methods for setting parameters etc.

    def _set_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        self._set_scalar_parameters()
        self._set_mechanics_parameters()

    def _set_mechanics_parameters(self) -> None:
        """
        Set the parameters for the simulation.
        """
        super()._set_parameters()
        gb = self.gb
        for g, d in gb:
            if g.dim == self._Nd:
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": self._bc_type_mechanics(g),
                        "bc_values": self._bc_values_mechanics(g),
                        "time_step": self.time_step,
                        "biot_alpha": self._biot_alpha(g),
                        "p_reference": self._reference_scalar(g),
                    },
                )

            elif g.dim == self._Nd - 1:
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "time_step": self.time_step,
                        "mass_weight": np.ones(g.num_cells),
                    },
                )

        for _, d in gb.edges():
            mg: pp.MortarGrid = d["mortar_grid"]
            pp.initialize_data(mg, d, self.mechanics_parameter_key)

    def _set_scalar_parameters(self) -> None:
        tensor_scale = self.scalar_scale / self.length_scale**2
        mass_weight = 1 * self.scalar_scale
        for g, d in self.gb:
            specific_volume = self._specific_volume(g)
            kappa = self._permeability(g) / self._viscosity(g) * tensor_scale
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            alpha = self._biot_alpha(g)
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": self._bc_type_scalar(g),
                    "bc_values": self._bc_values_scalar(g),
                    "mass_weight": mass_weight * specific_volume,
                    "biot_alpha": alpha,
                    "source": self._source_scalar(g),
                    "second_order_tensor": diffusivity,
                    "time_step": self.time_step,
                    "vector_source": self._vector_source(g),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )
        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            if mg.codim == 2:
                continue
            a_l = self._aperture(g_l)
            # Take trace of and then project specific volumes from g_h
            v_h = (
                mg.primary_to_mortar_avg()
                * np.abs(g_h.cell_faces)
                * self._specific_volume(g_h)
            )
            # Division by a/2 may be thought of as taking the gradient in the normal
            # direction of the fracture.
            kappa_l = self._permeability(g_l) / self._viscosity(g_l)
            normal_diffusivity = mg.secondary_to_mortar_avg() * (kappa_l * 2 / a_l)
            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_h
            pp.initialize_data(
                e,
                data_edge,
                self.scalar_parameter_key,
                {
                    "normal_diffusivity": normal_diffusivity,
                    "vector_source": np.zeros(mg.num_cells * self.gb.dim_max()),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )

    def _bc_type_mechanics(self, g: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Use parent class method for mechanics

        Parameters
        ----------
        g : pp.Grid
            Subdomain grid.

        Returns
        -------
        pp.BoundaryConditionVectorial
            Boundary condition representation.

        """
        return super()._bc_type(g)

    def _bc_type_scalar(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet condition on all external boundaries

        Parameters
        ----------
        g : pp.Grid
            Subdomain grid.

        Returns
        -------
        pp.BoundaryCondition
            Boundary condition representation.

        """
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _bc_values_mechanics(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous values on all boundaries

        Note that Dirichlet values should be divided by length_scale, and Neumann values
        by scalar_scale.

        Parameters
        ----------
        g : pp.Grid
            Subdomain grid.

        Returns
        -------
        np.ndarray (nd x #faces)
            Boundary condition values.

        """
        # Set the boundary values
        return super()._bc_values(g)

    def _bc_values_scalar(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous values on all boundaries

        Note that Dirichlet values should be divided by scalar_scale.

        Parameters
        ----------
        g : pp.Grid
            Subdomain grid.

        Returns
        -------
        np.ndarray (#faces)
            Boundary condition values.

        """
        return np.zeros(g.num_faces)

    def _stiffness_tensor(self, g: pp.Grid) -> pp.FourthOrderTensor:
        """Stress tensor parameter, unitary Lame parameters.


        Parameters
        ----------
        g : pp.Grid
            Matrix grid.

        Returns
        -------
        pp.FourthOrderTensor
            Representation of the stress tensor.

        """
        # Rock parameters
        lam = np.ones(g.num_cells) / self.scalar_scale
        mu = np.ones(g.num_cells) / self.scalar_scale
        return pp.FourthOrderTensor(mu, lam)

    def _source_scalar(self, g: pp.Grid) -> np.ndarray:
        """Zero source term.

        Units: m^3 / s
        """
        return np.zeros(g.num_cells)

    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Unitary permeability.

        Units: m^2
        """
        return np.ones(g.num_cells)

    def _viscosity(self, g: pp.Grid) -> np.ndarray:
        """Unitary viscosity.

        Units: kg / m / s = Pa s
        """
        return np.ones(g.num_cells)

    def _vector_source(self, g: pp.Grid) -> np.ndarray:
        """Zero vector source (gravity).

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * fluid_density
        """
        vals = np.zeros((self.gb.dim_max(), g.num_cells))
        return vals.ravel("F")

    def _reference_scalar(self, g: pp.Grid) -> np.ndarray:
        """Reference scalar value.

        Used for the scalar (pressure) contribution to stress.
        Parameters
        ----------
        g : pp.Grid
            Matrix grid.

        Returns
        -------
        np.ndarray
            Reference scalar value.

        """
        return np.zeros(g.num_cells)

    def _biot_alpha(self, g: pp.Grid) -> Union[float, np.ndarray]:
        if self._use_ad:
            return 1.0 * np.ones(g.num_cells)
        else:
            return 1.0

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

    def _specific_volume(self, g: pp.Grid) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically, equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in codimensions 2 and 3.
        """
        a = self._aperture(g)
        return np.power(a, self._Nd - g.dim)

    def _assign_discretizations(self) -> None:
        """
        Assign discretizations to the nodes and edges of the grid bucket.

        Note the attribute subtract_fracture_pressure: Indicates whether to
        subtract the fracture pressure contribution for the contact traction. This
        should not be done if the scalar variable is temperature.
        """
        if not hasattr(self, "dof_manager"):
            self.dof_manager = pp.DofManager(self.gb)
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
                if d["mortar_grid"].codim == 2:
                    continue
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
            self._assign_equations()

    def _assign_equations(self):
        """Assign equations for mixed-dimensional flow and deformation.

        The following equations are assigned to the equation manager:
            "momentum" in the nd subdomain
            "contact_mechanics_normal" in all fracture subdomains
            "contact_mechanics_tangential" in all fracture subdomains
            "force_balance" at the matrix-fracture interfaces

            "subdomain_flow" in all subdomains
            "interface_flow" on all interfaces of codimension 1

        Returns
        -------
        None.

        """
        # The parent assigns momentum and force balance and two contact mechanics equations.
        # These are constructed by submethods. The former two,
        # _momentum_balance_equation and _force_balance_equation, are modified
        # in this class to account for pressure effects.
        super()._assign_equations()

        # Now, assign the two flow equations not present in the parent model.
        subdomains: List[pp.Grid] = [g for g, _ in self.gb]

        interfaces = [e for e, d in self.gb.edges() if d["mortar_grid"].codim == 1]

        # Construct equations
        subdomain_flow_eq: pp.ad.Operator = self._subdomain_flow_equation(subdomains)
        interface_flow_eq: pp.ad.Operator = self._interface_flow_equation(interfaces)
        # Assign equations to manager
        self._eq_manager.name_and_assign_equations(
            {
                "subdomain_flow": subdomain_flow_eq,
                "interface_flow": interface_flow_eq,
            },
        )

    def _set_ad_projections(
        self,
    ) -> None:
        """
        Sets projection and rotation matrices.


        The following attributes are assigned to self._ad in addition to those
        set by the parent class:
            subdomain_projections_scalar
            local_fracture_coord_transformation_normal

        Returns
        -------
        None

        """
        super()._set_ad_projections()
        gb, ad = self.gb, self._ad
        subdomains: List[pp.Grid] = [g for g, _ in gb]
        fracture_subdomains: List[pp.Grid] = gb.grids_of_dimension(
            self._Nd - 1
        ).tolist()
        ad.subdomain_projections_scalar = pp.ad.SubdomainProjections(grids=subdomains)
        interfaces = [e for e, d in gb.edges() if d["mortar_grid"].codim == 1]
        ad.mortar_projections_scalar = pp.ad.MortarProjections(
            grids=subdomains, edges=interfaces, gb=gb, nd=1
        )

        normal_proj_list = []
        if len(fracture_subdomains) > 0:
            for gf in fracture_subdomains:
                proj = self.gb.node_props(gf, "tangential_normal_projection")
                normal_proj_list.append(proj.project_normal(gf.num_cells))
            normal_proj = pp.ad.Matrix(sps.block_diag(normal_proj_list))
        else:
            # In the case of no fractures, empty matrices are needed.
            normal_proj = pp.ad.Matrix(sps.csr_matrix((0, 0)))

        ad.local_fracture_coord_transformation_normal = normal_proj

    def _force_balance_equation(
        self,
        matrix_subdomains: List[pp.Grid],
        fracture_subdomains: List[pp.Grid],
        interfaces: List[Tuple[pp.Grid, pp.Grid]],
    ) -> pp.ad.Operator:
        """Force balance equation on fracture subdomains.

        Parameters
        ----------
        matrix_subdomains: List[pp.Grid]
            Matrix grids. Normally, only a single matrix grid is used
        fracture_subdomains: List[pp.Grid].
            Fracture grids.
        interfaces: List[Tuple[pp.Grid, pp.Grid]]
            Matrix-fracture interfaces.

        Returns
        -------
        force_balance_eq : pp.ad.Operator
            Force balance equation with contact stress and pressure contribution.

        Implementation note:
            The fracture pressure mapping involves flipping of normals as is done in
            the super method. Reuse of internal_boundary_vector_to_outwards would be
            preferable, but was deemed to lead to too complicated projections between
            subdomains and interfaces.
        """
        # The force balance equation for the contact mechanics without fluid pressure
        # with a term representing the fluid pressure in the fracture.

        # First the pure elastic force balance.
        eq = super()._force_balance_equation(
            matrix_subdomains, fracture_subdomains, interfaces
        )

        if not self.subtract_fracture_pressure:
            # If the scalar variable does not have the interpretation of a pressure,
            # no modifications of the equation is needed.
            return eq

        # The fracture pressure force takes the form
        #   normal_vector * I p
        # where I is the identity matrix.

        g_primary = matrix_subdomains[0]
        mat = []

        # Build up matrices containing the n-dot-I products for each interface.
        for e in interfaces:
            # It is assumed that the list interfaces contains only matrix-fracture
            # interfaces, thus no need to filter on dimensions

            mg = self.gb.edge_props(e, "mortar_grid")
            assert isinstance(mg, pp.MortarGrid)  # Appease mypy

            # Find the normal vectors of faces in g_primary on the boundary of this
            # interface.
            faces_on_fracture_boundary = mg.primary_to_mortar_int().tocsr().indices
            internal_boundary_normals = g_primary.face_normals[
                : self._Nd, faces_on_fracture_boundary
            ]
            # Also get sign of the faces.
            sgn, _ = g_primary.signs_and_cells_of_boundary_faces(
                faces_on_fracture_boundary
            )
            # Normal vector. Scale with sgn so that the normals point out of all cells
            outwards_fracture_normals = sgn * internal_boundary_normals

            # Matrix representation of n_dot_I for this interface
            data = outwards_fracture_normals.ravel("F")
            row = np.arange(self._Nd * mg.num_cells)
            col = np.tile(np.arange(mg.num_cells), (self._Nd, 1)).ravel("F")
            n_dot_I = sps.csc_matrix((data, (row, col)))
            # The scalar contribution to the contact stress is mapped to the mortar grid
            # and multiplied by n \dot I, with n being the outwards normals on the two
            # sides.
            mat.append(n_dot_I * mg.secondary_to_mortar_int(nd=1))

        if len(interfaces) == 0:
            mat = [sps.csr_matrix((0, 0))]

        # Block matrix version covering all interfaces.
        normal_matrix = pp.ad.Matrix(sps.block_diag(mat))

        # Ad variable representing pressure on all fracture subdomains.
        p_frac = (
            self._ad.subdomain_projections_scalar.cell_restriction(fracture_subdomains)
            * self._ad.pressure
        )

        # Add n_dot_I * p to the force balance.
        fracture_pressure = normal_matrix * p_frac
        force_balance_equation: pp.ad.Operator = eq + fracture_pressure
        return force_balance_equation

    def _subdomain_flow_equation(self, subdomains: List[pp.Grid]):
        """Mass balance equation for slightly compressible flow in a deformable medium.

        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains on which the equation is defined.

        Returns
        -------
        eq : pp.ad.Operator
            The equation on AD form.

        """

        ad = self._ad
        g_frac: List[pp.Grid] = self.gb.grids_of_dimension(self._Nd - 1).tolist()
        mass_discr = pp.ad.MassMatrixAd(self.scalar_parameter_key, subdomains)

        # Flow parameters

        # Facilitate updates of dt. self.time_step_ad.time_step._value must be updated
        # if time steps are changed.
        ad.time_step = pp.ad.Scalar(self.time_step, "time step")

        # Source term for the flow equation.
        flow_source = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="source",
            grids=subdomains,
        )

        div_scalar = pp.ad.Divergence(grids=subdomains)
        # Terms relating to the mechanics-to-flow coupling.
        biot_accumulation_primary = self._biot_terms_flow([self._nd_grid()])

        # Accumulation term specific to the fractures due to fracture volume changes.
        accumulation_fracs = self._volume_change(g_frac)

        # Accumulation due to compressibility. Applies to all subdomains
        accumulation_all = mass_discr.mass * (
            ad.pressure - ad.pressure.previous_timestep()
        )

        flux = self._fluid_flux(subdomains)

        eq = (
            ad.time_step
            * (  # Time scaling of flux terms, both inter-dimensional and from
                # the higher dimension
                div_scalar * flux
                - ad.mortar_projections_scalar.mortar_to_secondary_int
                * ad.interface_flux
            )
            + accumulation_all
            + ad.subdomain_projections_scalar.cell_prolongation([self._nd_grid()])
            * biot_accumulation_primary
            - ad.subdomain_projections_scalar.cell_prolongation(g_frac)
            * accumulation_fracs
            - flow_source
        )
        return eq

    def _interface_flow_equation(self, interfaces: List[Tuple[pp.Grid, pp.Grid]]):
        """Equation for interface fluxes.

        Parameters
        ----------
        interfaces : List[Tuple[pp.Grid, pp.Grid]]
            List of interfaces for which a flow equation should be constructed.

        Returns
        -------
        interface_flow_eq : pp.ad.Operator
            The interface equation on ad form.

        """
        # Interface equation: \lambda = -\kappa (p_l - p_h)
        # Robin_ad.mortar_discr represents -\kappa. The involved term is
        # reconstruction of p_h on internal boundary, which has contributions
        # from cell center pressure, external boundary and interface flux
        # on internal boundaries (including those corresponding to "other"
        # fractures).

        # Create list of subdomains. Ensure matrix grid is present so that bc
        # and vector_source_subdomains are consistent with ad.flux_discretization
        subdomains = [self._nd_grid()]
        for interface in interfaces:
            for sd in interface:
                if sd not in subdomains:
                    subdomains.append(sd)

        ad = self._ad
        flux_discr = ad.flux_discretization
        interface_discr = pp.ad.RobinCouplingAd(self.scalar_parameter_key, interfaces)

        bc = pp.ad.ParameterArray(
            self.scalar_parameter_key,
            array_keyword="bc_values",
            grids=subdomains,
        )

        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="vector_source",
            grids=subdomains,
        )
        vector_source_interfaces = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="vector_source",
            edges=interfaces,
        )
        # Construct primary (higher-dimensional) pressure
        # IMPLEMENTATION NOTE: this could possibly do with a sub-method
        p_primary = (
            flux_discr.bound_pressure_cell * ad.pressure
            + flux_discr.bound_pressure_face
            * ad.mortar_projections_scalar.mortar_to_primary_int
            * ad.interface_flux
            + flux_discr.bound_pressure_face * bc
            + flux_discr.vector_source * vector_source_subdomains
        )
        # Project the two pressures to the interface and equate with \lambda
        interface_flow_eq: pp.ad.Operator = (
            interface_discr.mortar_discr
            * (
                ad.mortar_projections_scalar.primary_to_mortar_avg * p_primary
                - ad.mortar_projections_scalar.secondary_to_mortar_avg * ad.pressure
                + interface_discr.mortar_vector_source * vector_source_interfaces
            )
            + ad.interface_flux
        )
        return interface_flow_eq

    def _biot_terms_flow(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        """Biot terms, div(u) and stabilization.

        Parameters
        ----------
        subdomains : List[pp.Grid]
            Matrix subdomains, expected to have length=1.

        Returns
        -------
        biot_terms : pp.ad.Operator
            Ad operator representing d/dt div(u) and stabilization terms of the
            Biot flow equation in the matrix.

        """
        ad = self._ad
        div_u_discr = pp.ad.DivUAd(
            self.mechanics_parameter_key,
            grids=subdomains,
            mat_dict_keyword=self.scalar_parameter_key,
        )
        stabilization_discr = pp.ad.BiotStabilizationAd(
            self.scalar_parameter_key, subdomains
        )
        biot_alpha = pp.ad.ParameterMatrix(
            self.scalar_parameter_key,
            array_keyword="biot_alpha",
            grids=subdomains,
        )
        # Boundary conditions for the mechanics, on this and the previous time step.
        bc_mech = pp.ad.ParameterArray(
            self.mechanics_parameter_key,
            array_keyword="bc_values",
            grids=subdomains,
        )
        bc_mech_previous = pp.ad.ParameterArray(
            self.mechanics_parameter_key,
            array_keyword="bc_values_previous_timestep",
            grids=subdomains,
        )
        # The "div_u" really represents the time increment d/dt div(u), thus
        # all contributions are defined on differences between current and previous
        # state. There are three components: matrix, external boundary and
        # internal boundary (fractures). The last term requires projection of
        # displacements from interfaces
        matrix_div_u: pp.ad.Operator = div_u_discr.div_u * (
            ad.displacement - ad.displacement.previous_timestep()
        )
        external_boundary_div_u: pp.ad.Operator = div_u_discr.bound_div_u * (
            bc_mech - bc_mech_previous
        )
        internal_boundary_div_u: pp.ad.Operator = (
            div_u_discr.bound_div_u
            * ad.subdomain_projections_vector.face_restriction(subdomains)
            * ad.mortar_projections_vector.mortar_to_primary_int
            * (
                ad.interface_displacement
                - ad.interface_displacement.previous_timestep()
            )
        )
        div_u_terms: pp.ad.Operator = biot_alpha * (
            matrix_div_u + external_boundary_div_u + internal_boundary_div_u
        )
        div_u_terms.set_name("div_u")

        # The stabilization term is also defined on a time increment, but only
        # considers the matrix subdomain and no boundary contributions.
        stabilization_term: pp.ad.Operator = (
            stabilization_discr.stabilization
            * ad.subdomain_projections_scalar.cell_restriction(subdomains)
            * (ad.pressure - ad.pressure.previous_timestep())
        )
        stabilization_term.set_name("Biot stabilization")
        biot_terms: pp.ad.Operator = div_u_terms + stabilization_term
        return biot_terms

    def _volume_change(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        """Volume change term for fracture subdomains

        Volume change terms for intersections are neglected in this implementation.

        Parameters
        ----------
        subdomains : List[pp.Grid]
            Fracture subdomains.

        Returns
        -------
        volume_change : pp.ad.Operator
            Volume change term for the fracture subdomains as an ad operator.

        TODO: Extend to intersections
        """
        ad = self._ad
        interface_displacement_prev = ad.interface_displacement.previous_timestep()
        interface_displacement = ad.interface_displacement

        # Change (in time) of the interface jump
        rotated_jumps: pp.ad.Operator = (
            ad.subdomain_projections_vector.cell_restriction(subdomains)
            * ad.mortar_projections_vector.mortar_to_secondary_avg
            * ad.mortar_projections_vector.sign_of_mortar_sides
            * (interface_displacement - interface_displacement_prev)
        )

        discr = pp.ad.MassMatrixAd(self.mechanics_parameter_key, subdomains)
        # Neglects intersections
        volume_change: pp.ad.Operator = (
            discr.mass * self._ad.normal_component_frac * rotated_jumps
        )
        volume_change.set_name("Volume change")
        return volume_change

    def _fluid_flux(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
        """Fluid flux.


        Parameters
        ----------
        subdomains : List[pp.Grid]
            Subdomains for which fluid fluxes are defined, normally all.

        Returns
        -------
        flux : pp.ad.Operator
            Flux on ad form.

        Note:
            The ad flux discretization used here is stored for consistency with
            self._interface_flow_equation, where self._ad.flux_discretization
            is applied.
        """
        bc = pp.ad.ParameterArray(
            self.scalar_parameter_key,
            array_keyword="bc_values",
            grids=subdomains,
        )
        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=self.scalar_parameter_key,
            array_keyword="vector_source",
            grids=subdomains,
        )

        flux_discr = pp.ad.MpfaAd(self.scalar_parameter_key, subdomains)
        # Store to ensure consistency in interface flux
        self._ad.flux_discretization = flux_discr
        flux: pp.ad.Operator = (
            flux_discr.flux * self._ad.pressure
            + flux_discr.bound_flux * bc
            + flux_discr.bound_flux
            * self._ad.mortar_projections_scalar.mortar_to_primary_int
            * self._ad.interface_flux
            + flux_discr.vector_source * vector_source_subdomains
        )
        flux.set_name("Fluid flux")
        return flux

    def _stress(
        self,
        matrix_subdomains: List[pp.Grid],
    ) -> pp.ad.Operator:
        """Ad representation of poromechanical stress.


        Parameters
        ----------
        matrix_subdomains : List[pp.Grid]
            List of N-dimensional grids, usually with a single entry.

        Returns
        -------
        stress : pp.ad.Operator
            Stress operator.

        """

        mechanical_stress = super()._stress(matrix_subdomains)
        discr = pp.ad.BiotAd(self.mechanics_parameter_key, matrix_subdomains)
        p_reference = pp.ad.ParameterArray(
            param_keyword=self.mechanics_parameter_key,
            array_keyword="p_reference",
            grids=matrix_subdomains,
        )
        pressure: pp.ad.Operator = (
            discr.grad_p
            * self._ad.subdomain_projections_scalar.cell_restriction(matrix_subdomains)
            * self._ad.pressure
            # The reference pressure is only defined on g_primary, thus there is no need
            # for a subdomain projection.
            - discr.grad_p * p_reference
        )
        pressure.set_name("pressure_stress")  # better name needed
        stress: pp.ad.Operator = mechanical_stress + pressure
        stress.set_name("poromechanical_stress")
        return stress

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        super()._assign_variables()
        # First for the nodes
        for g, d in self.gb:
            if g.dim == self._Nd:
                d[pp.PRIMARY_VARIABLES].update(
                    {
                        self.scalar_variable: {"cells": 1},
                    }
                )
            else:
                d[pp.PRIMARY_VARIABLES].update({self.scalar_variable: {"cells": 1}})

        # Then for the edges
        for e, d in self.gb.edges():
            _, g_h = self.gb.nodes_of_edge(e)

            if d["mortar_grid"].codim == 1:
                d[pp.PRIMARY_VARIABLES].update(
                    {self.mortar_scalar_variable: {"cells": 1}}
                )

    def _assign_ad_variables(self) -> None:
        """Assign variables to self._ad


        Assigns the following attributes to self._ad in addition to those set by
        the parent class:
            pressure: primary variable in all subdomains.
            interface_flux: Primary variable on interfaces of codimension 1 (usually
                all interfaces).

        Returns
        -------
        None

        """
        super()._assign_ad_variables()
        subdomains: List[pp.Grid] = [g for g, _ in self.gb]

        interfaces = [e for e, d in self.gb.edges() if d["mortar_grid"].codim == 1]

        # Primary variables on Ad form
        self._ad.pressure = self._eq_manager.merge_variables(
            [(g, self.scalar_variable) for g in subdomains]
        )
        self._ad.interface_flux = self._eq_manager.merge_variables(
            [(e, self.mortar_scalar_variable) for e in interfaces]
        )

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
            prev_solution (array): solution of previous iteration.
            init_solution (array): initial solution (or from beginning of time step).
            nl_params (dictionary): assumed to have the key nl_convergence_tol whose
                value is a float.
        """
        if self._use_ad or not self._is_nonlinear_problem():
            return super().check_convergence(
                solution, prev_solution, init_solution, nl_params
            )
        error_super, converged_super, diverged_super = super().check_convergence(
            solution, prev_solution, init_solution, nl_params
        )
        p_dof = np.array([], dtype=int)
        for g, _ in self.gb:
            p_dof = np.hstack(
                (
                    p_dof,
                    self.dof_manager.grid_and_variable_to_dofs(g, self.scalar_variable),
                )
            )

        # Pick out the solution from current, previous iterates, as well as the
        # initial guess.
        p_now = solution[p_dof]
        p_prev = prev_solution[p_dof]
        p_init = init_solution[p_dof]

        # Calculate errors
        difference_in_iterates = np.sum((p_now - p_prev) ** 2)
        difference_from_init = np.sum((p_now - p_init) ** 2)

        tol_convergence: float = nl_params["nl_convergence_tol"]

        converged_p = False
        diverged_p = False  # type: ignore

        # Check absolute convergence criterion
        if difference_in_iterates < tol_convergence:
            converged_p = True
            error_p = difference_in_iterates
        else:
            # Check relative convergence criterion
            if difference_in_iterates < tol_convergence * difference_from_init:
                converged_p = True
            error_p = difference_in_iterates / difference_from_init

        converged = converged_p and converged_super
        diverged = diverged_p or diverged_super
        logger.info("Error in pressure is {}".format(error_p))
        error = error_super + error_p

        return error, converged, diverged

    def _initial_condition(self) -> None:
        """
        Initial guess for Newton iteration, scalar variable and bc_values (for time
        discretization).
        """
        super()._initial_condition()
        g = self._nd_grid()
        d = self.gb.node_props(g)
        params = d[pp.PARAMETERS][self.mechanics_parameter_key]
        params["bc_values_previous_timestep"] = self._bc_values_mechanics(g)

    def _save_mechanical_bc_values(self) -> None:
        """
        The div_u term uses the mechanical bc values for both current and previous time
        step. In the case of time dependent bc values, these must be updated. As this
        is very easy to overlook, we do it by default.
        """
        key = self.mechanics_parameter_key
        g = self._nd_grid()
        d = self.gb.node_props(g)
        d[pp.PARAMETERS][key]["bc_values_previous_timestep"] = d[pp.PARAMETERS][key][
            "bc_values"
        ].copy()

    # Methods for discretization etc.

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
            # Discretization is a bit cumbersome, as the Biot discretization removes the
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

    def _set_ad_objects(self) -> None:
        """Sets the storage class self._ad


        Returns
        -------
        None

        """
        self._ad = ContactMechanicsBiotAdObjects()
