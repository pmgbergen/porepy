"""This module contains an implementation of a base model for incompressible flow problems.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class _AdVariables:
    pressure: pp.ad.Variable
    mortar_flux: pp.ad.Variable


class IncompressibleFlow(pp.models.abstract_model.AbstractModel):
    """This is a shell class for single-phase incompressible flow problems.

    This class is intended to provide a standardized setup, with all discretizations
    in place and reasonable parameter and boundary values. The intended use is to
    inherit from this class, and do the necessary modifications and specifications
    for the problem to be fully defined. The minimal adjustment needed is to
    specify the method create_grid(). The class also serves as parent for other
    model classes (CompressibleFlow).

    Public attributes:
        variable (str): Name assigned to the pressure variable in the
            highest-dimensional subdomain. Will be used throughout the simulations,
            including in Paraview export. The default variable name is 'p'.
        mortar_variable (str): Name assigned to the flux variable on the interfaces.
            Will be used throughout the simulations, including in Paraview export.
            The default mortar variable name is 'mortar_p'.
        parameter_key (str): Keyword used to define parameters and discretizations.
        params (dict): Dictionary of parameters used to control the solution procedure.
            Some frequently used entries are file and folder names for export,
            mesh sizes...
        gb (pp.GridBucket): Mixed-dimensional grid. Should be set by a method
            create_grid which should be provided by the user.
        convergence_status (bool): Whether the non-linear iterations has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'

    All attributes are given natural values at initialization of the class.

    The implementation assumes use of AD.
    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params)
        # Variables
        self.variable: str = "p"
        self.mortar_variable: str = "mortar_" + self.variable
        self.parameter_key: str = "flow"
        self._use_ad = True
        self._ad = _AdVariables()

    def prepare_simulation(self) -> None:
        self.create_grid()
        # Exporter initialization must be done after grid creation.
        self.exporter = pp.Exporter(
            self.gb, self.params["file_name"], folder_name=self.params["folder_name"]
        )
        self._set_parameters()
        self._assign_variables()

        self._create_dof_and_eq_manager()
        self._create_ad_variables()

        self._assign_discretizations()
        self._initial_condition()

        self._export()
        self._discretize()

    def _set_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameters fields of the data dictionaries are updated for all
        subdomains and edges (of codimension 1).
        """
        for g, d in self.gb:
            bc = self._bc_type(g)
            bc_values = self._bc_values(g)

            source_values = self._source(g)

            specific_volume = self._specific_volume(g)

            kappa = self._permeability(g) / self._viscosity(g)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(g.num_cells)
            )

            gravity = self._vector_source(g)

            pp.initialize_data(
                g,
                d,
                self.parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "source": source_values,
                    "second_order_tensor": diffusivity,
                    "vector_source": gravity.ravel("F"),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for e, data_edge in self.gb.edges():
            g_l, g_h = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            if mg.codim == 2:
                # Codim 2 (well type) interfaces are on the user's own responsibility
                # and must be handled in run scripts or by subclassing.
                continue

            a_l = self._aperture(g_l)
            # Take trace of and then project specific volumes from g_h
            trace = np.abs(g_h.cell_faces)
            v_h = mg.primary_to_mortar_avg() * trace * self._specific_volume(g_h)
            # Division by a/2 may be thought of as taking the gradient in the normal
            # direction of the fracture.
            kappa_l = self._permeability(g_l) / self._viscosity(g_l)
            normal_diffusivity = mg.secondary_to_mortar_avg() * (kappa_l * 2 / a_l)
            # The interface flux is to match fluxes across faces of g_h,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_h

            # Vector source/gravity zero by default
            gravity = self._vector_source(mg)
            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.parameter_key,
                {
                    "normal_diffusivity": normal_diffusivity,
                    "vector_source": gravity.ravel("F"),
                    "ambient_dimension": self.gb.dim_max(),
                },
            )

    def _bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries."""
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values.

        Units:
            Dirichlet conditions: Pa = kg / m^1 / s^2
            Neumann conditions: m^3 / s
        """
        return np.zeros(g.num_faces)

    def _source(self, g: pp.Grid) -> np.ndarray:
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
        return vals

    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self.gb.dim_max():
            aperture *= 0.1
        return aperture

    def _specific_volume(self, g: pp.Grid) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in dimension 1 and 0.
        """
        a = self._aperture(g)
        return np.power(a, self._nd_grid().dim - g.dim)

    def _assign_variables(self) -> None:
        """
        Assign primary variables to the nodes and edges of the grid bucket.
        """
        # First for the nodes
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES] = {
                self.variable: {"cells": 1},
            }
        # Then for the edges
        for e, d in self.gb.edges():
            if d["mortar_grid"].codim == 2:
                continue
            else:
                d[pp.PRIMARY_VARIABLES] = {
                    self.mortar_variable: {"cells": 1},
                }

    def _create_dof_and_eq_manager(self) -> None:
        """Create a dof_magaer and eq_maganer based on a mixed-dimensional grid"""
        self.dof_manager = pp.DofManager(self.gb)
        self._eq_manager = pp.ad.EquationManager(self.gb, self.dof_manager)

    def _create_ad_variables(self) -> None:
        """Create the merged variables for potential and mortar flux"""

        grid_list = [g for g, _ in self.gb.nodes()]
        # Make a list of interfaces, but only for couplings one dimension apart (e.g. not for
        # couplings that involve wells)
        # Make a list of interfaces, but only for couplings one dimension apart (e.g. not for
        # couplings that involve wells)
        edge_list = [e for e, d in self.gb.edges() if d["mortar_grid"].codim < 2]
        self._ad.pressure = self._eq_manager.merge_variables(
            [(g, self.variable) for g in grid_list]
        )
        self._ad.mortar_flux = self._eq_manager.merge_variables(
            [(e, self.mortar_variable) for e in edge_list]
        )

    def _assign_discretizations(self) -> None:
        """Define equations through discretizations.

        Assigns a Laplace/Darcy problem discretized using Mpfa on all subdomains with
        Neumann conditions on all internal boundaries. On edges of co-dimension one,
        interface fluxes are related to higher- and lower-dimensional pressures using
        the RobinCoupling.

        Gravity is included, but may be set to 0 through assignment of the vector_source
        parameter.
        """

        grid_list = [g for g, _ in self.gb.nodes()]
        self.grid_list = grid_list
        if len(self.gb.grids_of_dimension(self.gb.dim_max())) != 1:
            raise NotImplementedError("This will require further work")

        edge_list = [e for e, d in self.gb.edges() if d["mortar_grid"].codim < 2]

        mortar_proj = pp.ad.MortarProjections(
            edges=edge_list, grids=grid_list, gb=self.gb, nd=1
        )

        # Ad representation of discretizations
        flow_ad = pp.ad.MpfaAd(self.parameter_key, grid_list)
        robin_ad = pp.ad.RobinCouplingAd(self.parameter_key, edge_list)

        div = pp.ad.Divergence(grids=grid_list)

        # Ad variables
        p = self._ad.pressure
        mortar_flux = self._ad.mortar_flux

        # Ad parameters
        vector_source_grids = pp.ad.ParameterArray(
            param_keyword=self.parameter_key,
            array_keyword="vector_source",
            grids=grid_list,
        )
        vector_source_edges = pp.ad.ParameterArray(
            param_keyword=self.parameter_key,
            array_keyword="vector_source",
            edges=edge_list,
        )
        bc_val = pp.ad.BoundaryCondition(self.parameter_key, grid_list)
        source = pp.ad.ParameterArray(
            param_keyword=self.parameter_key,
            array_keyword="source",
            grids=grid_list,
        )

        # Ad equations
        flux = (
            flow_ad.flux * p
            + flow_ad.bound_flux * bc_val
            + flow_ad.bound_flux * mortar_proj.mortar_to_primary_int * mortar_flux
            + flow_ad.vector_source * vector_source_grids
        )
        subdomain_flow_eq = (
            div * flux - mortar_proj.mortar_to_secondary_int * mortar_flux - source
        )

        # Interface equation: \lambda = -\kappa (p_l - p_h)
        # Robin_ad.mortar_discr represents -\kappa. The involved term is
        # reconstruction of p_h on internal boundary, which has contributions
        # from cell center pressure, external boundary and interface flux
        # on internal boundaries (including those corresponding to "other"
        # fractures).
        p_primary = (
            flow_ad.bound_pressure_cell * p
            + flow_ad.bound_pressure_face
            * mortar_proj.mortar_to_primary_int
            * mortar_flux
            + flow_ad.bound_pressure_face * bc_val
            + flow_ad.vector_source * vector_source_grids
        )
        # Project the two pressures to the interface and equate with \lambda
        interface_flow_eq = (
            robin_ad.mortar_discr
            * (
                mortar_proj.primary_to_mortar_avg * p_primary
                - mortar_proj.secondary_to_mortar_avg * p
                + robin_ad.mortar_vector_source * vector_source_edges
            )
            + mortar_flux
        )
        subdomain_flow_eq.set_name("flow on nodes")
        interface_flow_eq.set_name("flow on interfaces")

        # Add to the equation list:
        self._eq_manager.equations.update(
            {
                "subdomain_flow": subdomain_flow_eq,
                "interface_flow": interface_flow_eq,
            }
        )

    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """Use a direct solver for the linear system."""
        A, b = self._eq_manager.assemble()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min"
            + f" {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."
        )
        tic = time.time()
        x = sps.linalg.spsolve(A, b)
        logger.info("Solved linear system in {} seconds".format(time.time() - tic))
        return x

    def _discretize(self) -> None:
        """Discretize all terms"""
        tic = time.time()
        self._eq_manager.discretize(self.gb)
        logger.info("Discretized in {} seconds".format(time.time() - tic))

    def after_newton_iteration(self, solution_vector: np.ndarray) -> None:
        """
        Scatters the solution vector for current iterate.

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

        solution = self.dof_manager.assemble_variable(from_iterate=True)
        self.dof_manager.distribute_variable(values=solution, additive=False)
        self.convergence_status = True
        self._export()

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu([self.variable])

    def _is_nonlinear_problem(self):
        return False

    ## Methods required by AbstractModel but irrelevant for static problems:
    def before_newton_loop(self):
        self._nonlinear_iteration = 0

    def after_simulation(self):
        pass
