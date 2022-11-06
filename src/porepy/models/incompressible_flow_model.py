"""This module contains an implementation of a base model for incompressible flow problems.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class _AdVariables:
    pressure: pp.ad.Variable
    mortar_flux: pp.ad.Variable
    mortar_proj: pp.ad.MortarProjections
    flux_discretization: Union[pp.ad.MpfaAd, pp.ad.TpfaAd]
    subdomains: List[pp.Grid]


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
            including in ParaView export. The default variable name is 'p'.
        mortar_variable (str): Name assigned to the flux variable on the interfaces.
            Will be used throughout the simulations, including in ParaView export.
            The default mortar variable name is 'mortar_p'.
        parameter_key (str): Keyword used to define parameters and discretizations.
        params (dict): Dictionary of parameters used to control the solution procedure.
            Some frequently used entries are file and folder names for export,
           mesh sizes...
        mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. Should be set by a method
            create_grid which should be provided by the user.
        convergence_status (bool): Whether the non-linear iteration has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'
        exporter (pp.Exporter): Used for writing files for visualization.

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
        self.exporter: pp.Exporter

    def prepare_simulation(self) -> None:
        self.create_grid()
        # Exporter initialization must be done after grid creation.
        self.exporter = pp.Exporter(
            self.mdg,
            self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
        )

        self._assign_variables()
        self._create_dof_and_eq_manager()
        self._create_ad_variables()
        self._initial_condition()

        self._set_parameters()

        self._assign_equations()

        self._export()
        self._discretize()
        self._initialize_linear_solver()

    def _set_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameter fields of the data dictionaries are updated for all
        subdomains and interfaces (of codimension 1).
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            bc = self._bc_type(sd)
            bc_values = self._bc_values(sd)

            source_values = self._source(sd)

            specific_volume = self._specific_volume(sd)

            kappa = self._permeability(sd) / self._viscosity(sd)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(sd.num_cells)
            )

            gravity = self._vector_source(sd)

            pp.initialize_data(
                sd,
                data,
                self.parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "source": source_values,
                    "second_order_tensor": diffusivity,
                    "vector_source": gravity.ravel("F"),
                    "ambient_dimension": self.mdg.dim_max(),
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for intf, intf_data in self.mdg.interfaces(return_data=True):
            sd_primary, sd_secondary = self.mdg.interface_to_subdomain_pair(intf)
            if intf.codim == 2:
                # Co-dimension 2 (well type) interfaces are on the user's own responsibility
                # and must be handled in run scripts or by subclassing.
                continue

            a_secondary = self._aperture(sd_secondary)
            # Take trace of and then project specific volumes from sd_primary
            trace = np.abs(sd_primary.cell_faces)
            v_primary = (
                intf.primary_to_mortar_avg() * trace * self._specific_volume(sd_primary)
            )
            # Division by a/2 may be thought of as taking the gradient in the normal
            # direction of the fracture.
            kappa_l = self._permeability(sd_secondary) / self._viscosity(sd_secondary)
            normal_diffusivity = intf.secondary_to_mortar_avg() * (
                kappa_l * 2 / a_secondary
            )
            # The interface flux is to match fluxes across faces of sd_primary,
            # and therefore need to be weighted by the corresponding
            # specific volumes
            normal_diffusivity *= v_primary

            # Vector source/gravity zero by default
            gravity = self._vector_source(intf)
            pp.initialize_data(
                intf,
                intf_data,
                self.parameter_key,
                {
                    "normal_diffusivity": normal_diffusivity,
                    "vector_source": gravity.ravel("F"),
                    "ambient_dimension": self.mdg.dim_max(),
                    "darcy_flux": np.zeros(intf.num_cells),
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

    def _vector_source(self, g: Union[pp.Grid, pp.MortarGrid]) -> np.ndarray:
        """Zero vector source (gravity).

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * fluid_density
        """
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        return vals

    def _aperture(self, g: pp.Grid) -> np.ndarray:
        """
        Aperture is a characteristic thickness of a cell, with units [m].
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.
        See also specific_volume.
        """
        aperture = np.ones(g.num_cells)
        if g.dim < self.mdg.dim_max():
            aperture *= 0.1
        return aperture

    def _specific_volume(self, g: pp.Grid) -> np.ndarray:
        """
        The specific volume of a cell accounts for the dimension reduction and has
        dimensions [m^(Nd - d)].
        Typically, equals 1 in Nd, the aperture in codimension 1 and the square/cube
        of aperture in dimension 1 and 0.
        """
        a = self._aperture(g)
        return np.power(a, self._nd_subdomain().dim - g.dim)

    def _assign_variables(self) -> None:
        """
        Assign primary variables to subdomains and interfaces of the mixed-dimensional grid.
        """
        for _, data in self.mdg.subdomains(return_data=True):
            data[pp.PRIMARY_VARIABLES] = {
                self.variable: {"cells": 1},
            }
        for intf, data in self.mdg.interfaces(return_data=True):
            if intf.codim == 2:
                continue
            else:
                data[pp.PRIMARY_VARIABLES] = {
                    self.mortar_variable: {"cells": 1},
                }

    def _create_dof_and_eq_manager(self) -> None:
        """Create a dof_manager and eq_manager based on a mixed-dimensional grid"""
        self.dof_manager = pp.DofManager(self.mdg)
        self._eq_manager = pp.ad.EquationManager(self.mdg, self.dof_manager)

    def _create_ad_variables(self) -> None:
        """Create the mixed-dimensional variables for potential and mortar flux"""

        self._ad.pressure = self._eq_manager.merge_variables(
            [(sd, self.variable) for sd in self.mdg.subdomains()]
        )
        self._ad.mortar_flux = self._eq_manager.merge_variables(
            [
                (intf, self.mortar_variable)
                for intf in self.mdg.interfaces()
                if intf.codim < 2
            ]
        )

    def _assign_equations(self) -> None:
        """Define equations.

        Assigns a Laplace/Darcy problem discretized using Mpfa on all subdomains with
        Neumann conditions on all internal boundaries. On interfaces of co-dimension one,
        interface fluxes are related to higher- and lower-dimensional pressures using
        the RobinCoupling.

        Gravity is included, but may be set to 0 through assignment of the vector_source
        parameter.
        """

        subdomains = [sd for sd in self.mdg.subdomains()]
        self._ad.subdomains = subdomains
        if len(list(self.mdg.subdomains(dim=self.mdg.dim_max()))) != 1:
            raise NotImplementedError("This will require further work")

        interfaces = [intf for intf in self.mdg.interfaces() if intf.codim < 2]

        self._ad.mortar_proj = pp.ad.MortarProjections(
            interfaces=interfaces, subdomains=subdomains, mdg=self.mdg, dim=1
        )

        # Ad representation of discretizations
        robin_ad = pp.ad.RobinCouplingAd(self.parameter_key, interfaces)

        div = pp.ad.Divergence(subdomains=subdomains)

        # Ad variables
        p = self._ad.pressure
        mortar_flux = self._ad.mortar_flux

        # Ad parameters
        vector_source_grids = pp.ad.ParameterArray(
            param_keyword=self.parameter_key,
            array_keyword="vector_source",
            subdomains=subdomains,
        )
        vector_source_interfaces = pp.ad.ParameterArray(
            param_keyword=self.parameter_key,
            array_keyword="vector_source",
            interfaces=interfaces,
        )
        bc_val = pp.ad.BoundaryCondition(self.parameter_key, subdomains)
        source = pp.ad.ParameterArray(
            param_keyword=self.parameter_key,
            array_keyword="source",
            subdomains=subdomains,
        )

        # Ad equations
        subdomain_flow_eq = (
            div * self._flux(subdomains)
            - self._ad.mortar_proj.mortar_to_secondary_int * mortar_flux
            - source
        )

        # Interface equation: \lambda = -\kappa (p_l - p_h)
        # Robin_ad.mortar_discr represents -\kappa. The involved term is
        # reconstruction of p_h on internal boundary, which has contributions
        # from cell center pressure, external boundary and interface flux
        # on internal boundaries (including those corresponding to "other"
        # fractures).
        flux_discr = self._ad.flux_discretization
        p_primary = (
            flux_discr.bound_pressure_cell * p
            + flux_discr.bound_pressure_face
            * self._ad.mortar_proj.mortar_to_primary_int
            * mortar_flux
            + flux_discr.bound_pressure_face * bc_val
            + flux_discr.vector_source * vector_source_grids
        )
        # Project the two pressures to the interface and equate with \lambda
        interface_flow_eq = (
            robin_ad.mortar_discr
            * (
                self._ad.mortar_proj.primary_to_mortar_avg * p_primary
                - self._ad.mortar_proj.secondary_to_mortar_avg * p
                + robin_ad.mortar_vector_source * vector_source_interfaces
            )
            + mortar_flux
        )
        subdomain_flow_eq.set_name("flow on subdomains")
        interface_flow_eq.set_name("flow on interfaces")

        # Add to the equation list:
        self._eq_manager.equations.update(
            {
                "subdomain_flow": subdomain_flow_eq,
                "interface_flow": interface_flow_eq,
            }
        )

    def _flux(self, subdomains: List[pp.Grid]) -> pp.ad.Operator:
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
            self._interface_flow_equations, where self._ad.flux_discretization
            is applied.
        """
        bc = pp.ad.ParameterArray(
            self.parameter_key,
            array_keyword="bc_values",
            subdomains=subdomains,
        )
        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=self.parameter_key,
            array_keyword="vector_source",
            subdomains=subdomains,
        )

        flux_discr = pp.ad.MpfaAd(self.parameter_key, subdomains)
        # Store to ensure consistency in interface flux
        self._ad.flux_discretization = flux_discr
        flux: pp.ad.Operator = (
            flux_discr.flux * self._ad.pressure
            + flux_discr.bound_flux * bc
            + flux_discr.bound_flux
            * self._ad.mortar_proj.mortar_to_primary_int
            * self._ad.mortar_flux
            + flux_discr.vector_source * vector_source_subdomains
        )
        flux.set_name("Fluid flux")
        return flux

    def _discretize(self) -> None:
        """Discretize all terms"""
        tic = time.time()
        self._eq_manager.discretize(self.mdg)
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
