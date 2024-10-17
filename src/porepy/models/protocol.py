"""Contains the protocols that declare the methods and attributes that must be present
in an instance of a PorePy model. The proper implementations of these methods can be
found in various classes within the models directiory.

"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol, Sequence

import numpy as np
import scipy.sparse as sps

# Conditional importing ensures that the protocols do not mess with the runtime
# definitions, i.e., the protocol empty method is accidentally called as a proper method
# and returns None.
if not TYPE_CHECKING:
    # This branch is accessed in python runtime.
    class PorePyModel(Protocol):
        """This is an empty placeholder of the protocol, used mainly for type hints."""

else:
    # This branch is accessed by mypy and linters.

    import porepy as pp

    class ModelGeometryProtocol(Protocol):
        """This protocol provides the declarations of the methods and the properties,
        typically defined in ModelGeometry."""

        fracture_network: pp.fracture_network
        """Representation of fracture network including intersections."""

        well_network: pp.WellNetwork3d
        """Well network."""

        mdg: pp.MixedDimensionalGrid
        """Mixed-dimensional grid.

        Set by the method :meth:`set_md_grid`.

        """

        nd: int
        """Ambient dimension of the problem.

        Set by the method :meth:`set_geometry`

        """

        @property
        def domain(self) -> pp.Domain:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        @property
        def fractures(self) -> list[pp.LineFracture] | list[pp.PlaneFracture]:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def set_geometry(self) -> None:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def is_well(self, grid: pp.Grid | pp.MortarGrid) -> bool:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def meshing_arguments(self) -> dict[str, float]:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def meshing_kwargs(self) -> dict:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def subdomains_to_interfaces(
            self, subdomains: list[pp.Grid], codims: list[int]
        ) -> list[pp.MortarGrid]:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def interfaces_to_subdomains(
            self, interfaces: list[pp.MortarGrid]
        ) -> list[pp.Grid]:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def subdomains_to_boundary_grids(
            self, subdomains: Sequence[pp.Grid]
        ) -> Sequence[pp.BoundaryGrid]:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def wrap_grid_attribute(
            self,
            grids: Sequence[pp.GridLike],
            attr: str,
            *,
            dim: int,
        ) -> pp.ad.DenseArray:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def basis(
            self, grids: Sequence[pp.GridLike], dim: int
        ) -> list[pp.ad.SparseArray]:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def e_i(
            self, grids: Sequence[pp.GridLike], *, i: int, dim: int
        ) -> pp.ad.SparseArray:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.


        def tangential_component(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def normal_component(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def subdomain_projections(self, dim: int) -> pp.ad.SubdomainProjections:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def domain_boundary_sides(
            self, domain: pp.Grid | pp.BoundaryGrid, tol: Optional[float] = 1e-10
        ) -> pp.domain.DomainSides:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def internal_boundary_normal_to_outwards(
            self,
            subdomains: list[pp.Grid],
            *,
            dim: int,
        ) -> pp.ad.Operator:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def outwards_internal_boundary_normals(
            self,
            interfaces: list[pp.MortarGrid],
            *,
            unitary: bool,
        ) -> pp.ad.Operator:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def specific_volume(
            self, grids: list[pp.Grid] | list[pp.MortarGrid]
        ) -> pp.ad.Operator:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.


        def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Aperture [m].

            Aperture is a characteristic thickness of a cell, with units [m]. It's value
            is 1 in matrix, thickness of fractures and "side length" of cross-sectional
            area/volume (or "specific volume") for intersections of dimension 1 and 0.

            See also:
                :meth:specific_volume.

            Parameters:
                subdomains: List of subdomain grids.

            Returns:
                Ad operator representing the aperture for each cell in each subdomain.

            """

        def isotropic_second_order_tensor(
            self, subdomains: list[pp.Grid], permeability: pp.ad.Operator
        ) -> pp.ad.Operator:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

    class SolutionStrategyProtocol(Protocol):
        """This protocol provides the declarations of the methods and the properties,
        typically defined in SolutionStrategy."""

        convergence_status: bool
        """Whether the non-linear iteration has converged."""

        equation_system: pp.ad.EquationSystem
        """Equation system manager.

        Will be set by :meth:`set_equation_system_manager`.

        """
        linear_system: tuple[sps.spmatrix, np.ndarray]
        """The linear system to be solved in each iteration of the non-linear solver.

        The tuple contains the sparse matrix and the right hand side residual vector.

        """
        params: dict
        """Dictionary of parameters."""
        units: pp.Units
        """Units of the model.

        See also :meth:`set_units`.

        """
        fluid: pp.FluidConstants
        """Fluid constants.

        See also :meth:`set_materials`.

        """
        solid: pp.SolidConstants
        """Solid constants.

        See also :meth:`set_materials`.

        """
        time_manager: pp.TimeManager
        """Time manager for the simulation."""
        restart_options: dict
        """Restart options. The template is provided in `SolutionStrategy.__init__`."""
        ad_time_step: pp.ad.Scalar
        """Time step as an automatic differentiation scalar."""
        nonlinear_solver_statistics: pp.SolverStatistics
        """Solver statistics for the nonlinear solver."""

        @property
        def time_step_indices(self) -> np.ndarray:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        @property
        def iterate_indices(self) -> np.ndarray:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def _is_time_dependent(self) -> bool:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.


    class VariableProtocol(Protocol):
        """This protocol provides the declarations of the methods and the properties,
        typically defined in VariableMixin."""

        def perturbation_from_reference(self, variable_name: str, grids: list[pp.Grid]):
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def create_variables(self) -> None:
            """Assign primary variables to subdomains and interfaces of the mixed-
            dimensional grid."""

    class BoundaryConditionProtocol(Protocol):
        """This protocol provides declarations of methods and properties related to
        boundary conditions.

        """

        def create_boundary_operator(
            self, name: str, domains: Sequence[pp.BoundaryGrid]
        ) -> pp.ad.TimeDependentDenseArray:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def _combine_boundary_operators(
            self,
            subdomains: Sequence[pp.Grid],
            dirichlet_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            neumann_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
            robin_operator: Optional[
                None | Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator]
            ],
            bc_type: Callable[[pp.Grid], pp.BoundaryCondition],
            name: str,
            dim: int = 1,
        ) -> pp.ad.Operator:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def update_all_boundary_conditions(self) -> None:
            """This method is called before a new time step to set the values of the
            boundary conditions.

            Note:
                One can use the convenience method `update_boundary_condition` for each
                boundary condition value.

            """

    class EquationProtocol(Protocol):
        """This protocol provides declarations of methods and properties related to
        equations.

        """

        def volume_integral(
            self,
            integrand: pp.ad.Operator,
            grids: list[pp.Grid] | list[pp.MortarGrid],
            dim: int,
        ) -> pp.ad.Operator:
            """"""
            # AUTODOC: The implementation's docstring will fetch here automatically.

        def set_equations(self) -> None:
            """Set equations for the subdomains and interfaces."""

    class DataSavingProtocol(Protocol):
        """This protocol provides the declarations of the methods and the properties,
        typically defined in DataSavingMixin."""

        exporter: pp.Exporter
        """Exporter for visualization."""

        def save_data_time_step(self) -> None:
            """Export the model state at a given time step and log time.

            The options for exporting times are:
                * `None`: All time steps are exported
                * `list`: Export if time is in the list. If the list is empty, then no
                times are exported.

            In addition, save the solver statistics to file if the option is set.

            """

        def initialize_data_saving(self) -> None:
            """Initialize data saving.

            This method is called by :meth:`prepare_simulation` to initialize the
            exporter and any other data saving functionality (e.g., empty data
            containers to be appended in :meth:`save_data_time_step`).

            In addition, set path for storing solver statistics data to file for each
            time step.

            """

        def load_data_from_vtu(
            self,
            vtu_files: Path | list[Path],
            time_index: int,
            times_file: Optional[Path] = None,
            keys: Optional[str | list[str]] = None,
            **kwargs,
        ) -> None:
            """Initialize data in the model by reading from a pvd file.

            Parameters:
                vtu_files: path(s) to vtu file(s).
                keys: keywords addressing cell data to be transferred. If 'None', the
                    mixed-dimensional grid is checked for keywords corresponding to
                    primary variables identified through pp.TIME_STEP_SOLUTIONS.
                keyword arguments: see documentation of
                    :meth:`porepy.viz.exporter.Exporter.import_state_from_vtu`

            Raises:
                ValueError: if incompatible file type provided.

            """

        def load_data_from_pvd(
            self,
            pvd_file: Path,
            is_mdg_pvd: bool = False,
            times_file: Optional[Path] = None,
            keys: Optional[str | list[str]] = None,
        ) -> None:
            """Initialize data in the model by reading from a pvd file.

            Parameters:
                pvd_file: path to pvd file with exported vtu files.
                is_mdg_pvd: flag controlling whether pvd file is a mdg file, i.e.,
                    generated with Exporter._export_mdg_pvd() or Exporter.write_pvd().
                times_file: path to json file storing history of time and time step
                    size.
                keys: keywords addressing cell data to be transferred. If 'None', the
                    mixed-dimensional grid is checked for keywords corresponding to
                    primary variables identified through pp.TIME_STEP_SOLUTIONS.

            Raises:
                ValueError: if incompatible file type provided.

            """

    class PorePyModel(
        BoundaryConditionProtocol,
        EquationProtocol,
        VariableProtocol,
        ModelGeometryProtocol,
        DataSavingProtocol,
        SolutionStrategyProtocol,
        Protocol,
    ):
        """This protocol declares the core, physics-agnostic functionality of
        a PorePy model.

        The main purpose of the protocol is to provide type hints for countless model
        mixins, so mypy can properly verify the inter-mixin method calls, and an IDE
        such as VSCode can properly autocomplete these hints. You must either inherit
        from this class or provide it as a type annotation for the PorePy model object.

        Note:
            This can also be considered the list of the functionality which must be
            implemented by an instanciated model, although it does not verify it in
            runtime, since it is not an abstract base class.

        """
