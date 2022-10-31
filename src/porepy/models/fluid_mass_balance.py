"""
Class types:
    Generic ScalarBalanceEquation
    Specific MassBalanceEquations defines subdomain and interface equations through the
        terms entering. Darcy type interface relation is assumed.
    Specific ConstitutiveEquations and
    specific SolutionStrategy for both incompressible and compressible case.

Notes:
    Apertures and specific volumes are not included.

    Refactoring needed for constitutive equations. Modularisation and moving to the library.

    Upwind for the mobility of the fluid flux is not complete.

"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import constitutive_laws
import numpy as np
from constitutive_laws import ad_wrapper
from geometry import ModelGeometry

import porepy as pp

logger = logging.getLogger(__name__)


class ScalarBalanceEquation:
    """Generic class for scalar balance equations on the form

    d_t(accumulation) + div(flux) - source = 0

    All terms need to be specified in order to define an equation.
    """

    def balance_equation(
        self,
        subdomains: list[pp.Grid],
        accumulation: pp.ad.Operator,
        flux: pp.ad.Operator,
        source: pp.ad.Operator,
    ) -> pp.ad.Operator:
        """Define the balance equation.

        Args:
            subdomains: List of grids on which the equation is defined.
            accumulation: Accumulation term.
            flux: Flux term.
            source: Source term.

        Returns:
            Operator representing the balance equation.
        """

        dt = pp.ad.time_derivatives.dt
        div = pp.ad.Divergence(subdomains)
        return dt(accumulation) + div * flux - source

    def volume_integral(
        self,
        integrand: pp.ad.Operator,
        grids: list[pp.GridLike],
    ) -> pp.ad.Operator:
        """Numerical volume integral over subdomain or interface cells.

        Includes cell volumes and specific volume.
        FIXME: Decide whether to use this on source terms.

        Args:
            integrand: Operator to be integrated.
            grids: List of subdomain or interface grids over which to integrate.

        Returns:
            Operator representing the integral.
        """
        geometry = pp.ad.Geometry(grids, nd=self.nd)
        return geometry.cell_volumes * self.specific_volume(grids) * integrand


class MassBalanceEquations(ScalarBalanceEquation):
    """Mixed-dimensional mass balance equation.

    Balance equation for all subdomains and Darcy-type flux relation on all interfaces of
    codimension one.

    FIXME: Well equations? Low priority.

    """

    def set_equations(self):
        subdomains = self.mdg.subdomains()
        interfaces = self.mdg.interfaces()
        sd_eq = self.subdomain_mass_balance_equation(subdomains)
        intf_eq = self.interface_fluid_flux_equation(interfaces)
        self.system_manager.set_equation(sd_eq, (subdomains, "cells", 1))
        self.system_manager.set_equation(intf_eq, (interfaces, "cells", 1))

    def subdomain_mass_balance_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        accumulation = self.fluid_mass(subdomains)
        flux = self.fluid_flux(subdomains)
        source = self.fluid_source(subdomains)
        return self.balance_equation(subdomains, accumulation, flux, source)

    def fluid_mass(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:

        density = self.fluid_density(subdomains) * self.porosity(subdomains)
        mass = self.volume_integral(density, subdomains)
        mass.set_name("fluid_mass")
        return mass

    def fluid_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        flux = self.face_mobility(subdomains) * self.darcy_flux(subdomains)
        flux.set_name("fluid_flux")
        return flux

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid source term.

        Args:
            subdomains: List of subdomains.

        Returns:
            Operator representing the source term.
        """
        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros(num_cells)
        source = pp.ad.Array(vals, "fluid_source")
        return source


class ConstitutiveEquationsIncompressibleFlow(
    constit_library.DarcyFlux, constit_library.DimensionReduction
):
    """Constitutive equations for incompressible flow.

    .. note::
        We should consider modularising and moving to constit_library.
    """

    def face_mobility(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # interfaces = self.subdomains_to_interfaces(subdomains)
        # projection = pp.ad.MortarProjections(subdomains, interfaces, dim=1)

        discr = self.mobility_discretization(subdomains)
        cell_mobility = self.fluid_density(subdomains) / self.viscosity(subdomains)
        # FIXME: complete with BCs etc. Nontrivial!
        flux: pp.ad.Operator = discr.upwind * cell_mobility
        flux.set_name("face_mobility")
        return flux

    def mobility_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Discretization:
        return pp.ad.UpwindAd(self.flow_discretization_parameter_key, subdomains)

    def interface_mobility_discretization(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.Discretization:
        """

        Args:
            interfaces:

        Returns:

        """
        return pp.ad.UpwindCouplingAd(
            self.flow_discretization_parameter_key, interfaces
        )

    def interface_fluid_flux_equation(self, interfaces: list[pp.MortarGrid]):
        subdomains = self.interfaces_to_subdomains(interfaces)
        projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces, dim=1)

        p_trace = self.pressure_trace(self, subdomains)
        p = self.pressure(subdomains)
        interface_geometry = pp.ad.Geometry(interfaces, matrices=["cell_volumes"])
        # Project the two pressures to the interface and equate with \lambda
        eq = self.interface_fluid_flux(
            interfaces
        ) - interface_geometry.cell_volumes * self.normal_diffusivity(interfaces) * (
            projection.primary_to_mortar_avg * p_trace
            - projection.mortar_projection_scalar.secondary_to_mortar_avg * p
            # FIXME: The plan is to remove RoubinCoupling. That requires alternative
            #  implementation of the below
            # + robin_ad.mortar_vector_source * vector_source_interfaces
        )
        return eq

    def bc_values_flow(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """
        Not sure where this one should reside.
        Note that we could remove the grid_operator BC and DirBC, probably also
        ParameterArray/Matrix (unless needed to get rid of pp.ad.Discretization. I don't see
        how it would be, though).
        Args:
            subdomains:

        Returns:

        """
        num_faces = sum([sd.num_faces for sd in subdomains])
        return ad_wrapper(0, True, num_faces, "bc_vals_flow")


class ConstitutiveEquationsCompressibleFlow(
    constit_library.FluidDensityFromPressure, ConstitutiveEquationsIncompressibleFlow
):
    """Resolution order is important:
    Left to right, i.e., DensityFromPressure mixin's method is used when calling
    self.fluid_density
    """

    pass


class VariablesSinglePhaseFlow:
    """
    Creates necessary variables (pressure, interface flux) and provides getter methods for
    these and their reference values.
    Getters construct merged variables on the fly, and can be called on any subset of the
    grids where the variable is defined. Setter method (assig_variables), however, must
    create on all grids where the variable is to be used.

    .. note::
        Awaiting Veljko's more convenient SystemManager, some old implementation is kept.
    """

    def _assign_variables(self) -> None:
        """
        Assign primary variables to subdomains and interfaces of the mixed-dimensional grid.
        Old implementation awaiting SystemManager

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

    def pressure(self, subdomains):
        p = self._eq_manager.merge_variables([(sd, self.variable) for sd in subdomains])
        # Veljko: p = self.system_manager.merged_variable(subdomains, "pressure")
        return p

    def interface_fluid_flux(
        self, interfaces: list[pp.MortarGrid]
    ) -> pp.ad.MergedVariable:
        flux = self._eq_manager.merge_variables(
            [(intf, self.mortar_variable) for intf in interfaces if intf.codim < 2]
        )
        # Veljko: flux = self.system_manager.merged_variable(subdomains, self.mortar_variable)
        return flux

    def reference_pressure(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        num_cells = sum([sd.num_cells for sd in subdomains])
        return ad_wrapper(self.fluid.PRESSURE, True, num_cells, "reference_pressure")


class SolutionStrategyIncompressibleFlow(pp.models.abstract_model.AbstractModel):
    """This is whatever is left of pp.IncompressibleFlow.

    At some point, this will be refined to be a more sophisticated (modularised) solution
    strategy class.
    More refactoring may be beneficial.

    This is *not* a full-scale model (in the old sense), but must be mixed with
    balance equations, constitutive laws etc. See user_examples.

    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params)
        # Variables
        self.variable: str = "p"
        self.mortar_variable: str = "mortar_" + self.variable
        self.flow_discretization_parameter_key: str = "flow"
        self.exporter: pp.Exporter

        # Place initialization stuff to be moved to abstract below.
        self.units = params.get("units", pp.Units())

    def prepare_simulation(self) -> None:
        self.set_geometry()
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
        self._initial_condition()
        # New: Set material components. Could be moved to init:
        self.set_materials()
        # New: renamed from _set_parameters
        self.set_discretization_parameters()

        self.set_equations()

        self._export()
        self._discretize()
        self._initialize_linear_solver()

    def set_materials(self):
        """Sketch approach of setting materials. Works for now.

        Should probably go in AbstractModel.
        May want to use more refined approach (setter method, protect attribute names...)
        FIXME: Move to AbstractModel/AbstractSolutionStrategy.
        """
        for name, material in self.params["materials"].items():
            assert issubclass(material, pp.models.materials.Material)
            setattr(self, name, material(self.units))

    def initial_condition(self) -> None:
        """New formulation requires darcy flux (the flux is "advective" with mobilities
        included).

        Returns:

        """
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                sd,
                data,
                self.flow_parameter_keyword,
                {"darcy_flux": np.zeros(sd.num_faces)},
            )
        for intf, data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                data,
                self.flow_parameter_keyword,
                {"darcy_flux": np.zeros(intf.num_faces)},
            )

    def set_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameter fields of the data dictionaries are updated for all
        subdomains and interfaces (of codimension 1).
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            bc = self.bc_type_flow(sd)

            specific_volume = self.grid_specific_volume(sd)

            kappa = self.permeability(sd)
            diffusivity = pp.SecondOrderTensor(
                kappa * specific_volume * np.ones(sd.num_cells)
            )

            pp.initialize_data(
                sd,
                data,
                self.flow_discretization_parameter_key,
                {
                    "bc": bc,
                    "second_order_tensor": diffusivity,
                    #                    "darcy_flux": self.darcy_flux(sd),
                    "ambient_dimension": self.nd,
                },
            )

        # Assign diffusivity in the normal direction of the fractures.
        for intf, intf_data in self.mdg.interfaces(return_data=True):
            pp.initialize_data(
                intf,
                intf_data,
                self.flow_discretization_parameter_key,
                {
                    "ambient_dimension": self.nd,
                    "darcy_flux": self.darcy_flux(intf),
                },
            )

    def bc_type_flow(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.
        FIXME: Refactor?
        """
        # Define boundary regions
        all_bf, *_ = self._domain_boundary_sides(g)
        # Define boundary condition on faces
        return pp.BoundaryCondition(g, all_bf, "dir")

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

    def _create_dof_and_eq_manager(self) -> None:
        """Create a dof_manager and eq_manager based on a mixed-dimensional grid"""
        self.dof_manager = pp.DofManager(self.mdg)
        self._eq_manager = pp.ad.EquationManager(self.mdg, self.dof_manager)

    def _discretize(self) -> None:
        """Discretize all terms"""
        tic = time.time()
        self._eq_manager.discretize(self.mdg)
        logger.info("Discretized in {} seconds".format(time.time() - tic))

    def before_newton_iteration(self):
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            self.flow_discretization_parameter_key,
            self.flow_discretization_parameter_key,
            lam_name=self.mortar_variable,
        )
        # FIXME: Rediscretize upwind.

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


class IncompressibleCombined(
    ModelGeometry,
    MassBalanceEquations,
    ConstitutiveEquationsIncompressibleFlow,
    VariablesSinglePhaseFlow,
    SolutionStrategyIncompressibleFlow,
):
    """Demonstration of how to combine in a class which can be used with
    pp.run_stationary_problem (once cleanup has been done).
    """

    pass


"""
Compressible flow below.

Note on time dependency: I'm tempted to suggest assigning time_manager to stationary models
and partially remove the distinction with transient ones.
"""


class SolutionStrategyCompressibleFlow(SolutionStrategyIncompressibleFlow):
    """This class extends the Incompressible flow model by including a
    cumulative term expressed through pressure and a constant compressibility
    coefficient. For a full documentation refer to the parent class.

    The simulation starts at time t=0.



    Attributes:
        time_manager: Time-stepping control manager.

    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        """
        Parameters:
            params (dict): Dictionary of parameters used to control the solution procedure.
                Some frequently used entries are file and folder names for export,
                mesh sizes...
        """
        if params is None:
            params = {}
        super().__init__(params)

        # Time manager
        self.time_manager = params.get(
            "time_manager",
            pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True),
        )

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu([self.variable], time_dependent=True)

    def after_simulation(self):
        self.exporter.write_pvd()
