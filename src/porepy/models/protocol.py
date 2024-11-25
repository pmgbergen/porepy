"""Contains the protocols that declare the methods and attributes that must be present
in an instance of a PorePy model. The proper implementations of these methods can be
found in various classes within the models directiory.

Note that the protocol framework is accessed by static type checkes only!

Warning:
    For developers:

    Do not bring the ``typing.Protocol`` class in any form into the mixin framework
    of PorePy! Use it exclusively in ``if``-sections for ``typing.TYPE_CHECKING``.

    Protocols use ``__slot__`` which leads to unforeseeable behaviour when combined
    with multiple inheritance and mixing.

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
    # NOTE See Warning in module docstring before attempting anything here.
    class PorePyModel:
        """This is an empty placeholder of the protocol, used mainly for type hints."""

    class CompositionalFlowModelProtocol:
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
            """Domain of the problem."""

        @property
        def fractures(self) -> list[pp.LineFracture] | list[pp.PlaneFracture]:
            """Fractures of the problem."""

        def set_geometry(self) -> None:
            """Define geometry and create a mixed-dimensional grid.

            The default values provided in set_domain, set_fractures, grid_type and
            meshing_arguments produce a 2d unit square domain with no fractures and a
            four Cartesian cells.

            """

        def set_well_network(self) -> None:
            """Assign well network class :attr:`well_network`."""

        def is_well(self, grid: pp.Grid | pp.MortarGrid) -> bool:
            """Check if a subdomain is a well.

            Parameters:
                sd: Subdomain to check.

            Returns:
                True if the subdomain is a well, False otherwise.

            """

        def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
            """Grid type for the mixed-dimensional grid.

            Returns:
                Grid type for the mixed-dimensional grid.

            """

        def meshing_arguments(self) -> dict[str, float]:
            """Meshing arguments for mixed-dimensional grid generation.

            Returns:
                Meshing arguments compatible with
                :meth:`~porepy.grids.mdg_generation.create_mdg`.

            """

        def meshing_kwargs(self) -> dict:
            """Keyword arguments for md-grid creation.

            Returns:
                Keyword arguments compatible with pp.create_mdg() method.

            """

        def subdomains_to_interfaces(
            self, subdomains: list[pp.Grid], codims: list[int]
        ) -> list[pp.MortarGrid]:
            """Interfaces neighbouring any of the subdomains.

            Parameters:
                subdomains: Subdomains for which to find interfaces.
                codims: Codimension of interfaces to return. The common option is [1],
                    i.e. only interfaces between subdomains one dimension apart.

            Returns:
                Unique list of all interfaces neighboring any of the subdomains.
                Interfaces are sorted according to their index, as defined by the
                mixed-dimensional grid.

            """

        def interfaces_to_subdomains(
            self, interfaces: list[pp.MortarGrid]
        ) -> list[pp.Grid]:
            """Subdomain neighbours of interfaces.

            Parameters:
                interfaces: List of interfaces for which to find subdomains.

            Returns:
                Unique list of all subdomains neighbouring any of the interfaces. The
                subdomains are sorted according to their index as defined by the
                mixed-dimensional grid.

            """

        def subdomains_to_boundary_grids(
            self, subdomains: Sequence[pp.Grid]
        ) -> Sequence[pp.BoundaryGrid]:
            """Boundary grids of subdomains.

            This is a 1-1 mapping between subdomains and their boundary grids. No
            sorting is performed.

            Parameters:
                subdomains: List of subdomains for which to find boundary grids.

            Returns:
                List of boundary grids associated with the provided subdomains.

            """

        def wrap_grid_attribute(
            self,
            grids: Sequence[pp.GridLike],
            attr: str,
            *,
            dim: int,
        ) -> pp.ad.DenseArray:
            """Wrap a grid attribute as an ad matrix.

            Parameters:
                grids: List of grids on which the property is defined.
                attr: Grid attribute to wrap. The attribute should be a ndarray and will
                    be flattened if it is not already one-dimensional.
                dim: Dimensions to include for vector attributes. Intended use is to
                    limit the number of dimensions for a vector attribute, e.g. to
                    exclude the z-component of a vector attribute in 2d, to achieve
                    compatibility with code which is explicitly 2d (e.g. fv
                    discretizations).

            Returns:
                class:`porepy.numerics.ad.DenseArray`: `(shape=(dim *
                    num_cells_in_grids,))`

                    The property wrapped as a single ad vector. The values are arranged
                    according to the order of the grids in the list, optionally
                    flattened if the attribute is a vector.

            Raises:
                ValueError: If one of the grids does not have the attribute.
                ValueError: If the attribute is not an ndarray.

            """

        def basis(
            self, grids: Sequence[pp.GridLike], dim: int
        ) -> list[pp.ad.SparseArray]:
            """Return a cell-wise basis for all subdomains.

            The basis is represented as a list of matrices, each of which represents a
            basis function. The individual matrices have shape ``Nc * dim, Nc`` where
            ``Nc`` is the total number of cells in the subdomains.

            Examples:
                To extend a cell-wise scalar to a vector field, use
                ``sum([e_i for e_i in basis(subdomains)])``. To restrict to a vector in
                the tangential direction only, use
                ``sum([e_i for e_i in basis(subdomains, dim=nd-1)])``

            See also:
                :meth:`e_i` for the construction of a single basis function.
                :meth:`normal_component` for the construction of a restriction to the
                    normal component of a vector only.
                :meth:`tangential_component` for the construction of a restriction to
                    the tangential component of a vector only.

            Parameters:
                grids: List of grids on which the basis is defined.
                dim: Dimension of the basis.

            Returns:
                List of pp.ad.SparseArrayArray, each of which represents a basis
                function.

            """

        def e_i(
            self, grids: Sequence[pp.GridLike], *, i: int, dim: int
        ) -> pp.ad.SparseArray:
            """Return a cell-wise basis function in a specified dimension.

            It is assumed that the grids are embedded in a space of dimension dim and
            aligned with the coordinate axes, that is, the reference space of the grid.
            Moreover, the grid is assumed to be planar.

            Example:
                For a grid with two cells, and with `i=1` and `dim=3`, the returned
                basis will be (after conversion to a numpy array)
                .. code-block:: python
                    array([[0., 0.],
                        [1., 0.],
                        [0., 0.],
                        [0., 0.],
                        [0., 1.],
                        [0., 0.]])

            See also:
                :meth:`basis` for the construction of a full basis.

            Parameters:
                grids: List of grids on which the basis vector is defined.
                i: Index of the basis function. Note: Counts from 0.
                dim: Dimension of the functions.

            Returns:
                pp.ad.SparseArray: Ad representation of a matrix with the basis
                functions as columns.

            Raises:
                ValueError: If i is larger than dim - 1.

            """

        def tangential_component(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
            """Compute the tangential component of a vector field.

            The tangential space is defined according to the local coordinates of the
            subdomains, with the tangential space defined by the first `self.nd`
            components of the cell-wise vector. It is assumed that the components of the
            vector are stored with a dimension-major ordering (the dimension varies
            fastest).

            Parameters:
                subdomains: List of grids on which the vector field is defined.

            Returns:
                Operator extracting tangential component of the vector field and
                expressing it in tangential basis.

            """

        def normal_component(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
            """Compute the normal component of a vector field.

            The normal space is defined according to the local coordinates of the
            subdomains, with the normal space defined by final component, e.g., number
            `self.nd-1` (zero offset). of the cell-wise vector. It is assumed that the
            components of a vector are stored with a dimension-major ordering (the
            dimension varies fastest).

            See also:
                :meth:`e_i` for the definition of the basis functions.
                :meth:`tangential_component` for the definition of the tangential space.

            Parameters:
                subdomains: List of grids on which the vector field is defined.

            Returns:
                Matrix extracting normal component of the vector field and expressing it
                in normal basis. The size of the matrix is `(Nc, Nc * self.nd)`, where
                `Nc` is the total number of cells in the subdomains.

            """

        def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
            """Ad wrapper around tangential_normal_projections for fractures.

            Parameters:
                subdomains: List of subdomains for which to compute the local
                coordinates.

            Returns:
                Local coordinates as a pp.ad.SparseArray.

            """

        def subdomain_projections(self, dim: int) -> pp.ad.SubdomainProjections:
            """Return the projection operators for all subdomains in md-grid.

            The projection operators restrict or prolong a dim-dimensional quantity
            from the full set of subdomains to any subset. Projection operators are
            constructed once and then stored. If you need to use projection operators
            based on a different set of subdomains, please construct them yourself.
            Alternatively, compose a projection from subset A to subset B as
                P_A_to_B = P_full_to_B * P_A_to_full.

            Parameters:
                dim: Dimension of the quantities to be projected.

            Returns:
                proj: Projection operator.

            """

        def domain_boundary_sides(
            self, domain: pp.Grid | pp.BoundaryGrid, tol: Optional[float] = 1e-10
        ) -> pp.domain.DomainSides:
            """Obtain indices of the faces lying on the sides of the domain boundaries.

            The method is primarily intended for box-shaped domains. However, it can
            also be applied to non-box-shaped domains (e.g., domains with perturbed
            boundary nodes) provided `tol` is tuned accordingly.

            Parameters:
                domain: Subdomain or boundary grid.
                tol: Tolerance used to determine whether a face center lies on a
                    boundary side.

            Returns:
                NamedTuple containing the domain boundary sides. Available attributes
                are:

                    - all_bf (np.ndarray of int): indices of the boundary faces.
                    - east (np.ndarray of bool): flags of the faces lying on the East
                        side.
                    - west (np.ndarray of bool): flags of the faces lying on the West
                        side.
                    - north (np.ndarray of bool): flags of the faces lying on the North
                        side.
                    - south (np.ndarray of bool): flags of the faces lying on the South
                        side.
                    - top (np.ndarray of bool): flags of the faces lying on the Top
                        side.
                    - bottom (np.ndarray of bool): flags of the faces lying on Bottom
                        side.

            Examples:

                .. code:: python

                    model = pp.SinglePhaseFlow({})
                    model.prepare_simulation()
                    sd = model.mdg.subdomains()[0]
                    sides = model.domain_boundary_sides(sd)
                    # Access north faces using index or name is equivalent:
                    north_by_index = sides[3]
                    north_by_name = sides.north
                    assert all(north_by_index == north_by_name)

            """

        def internal_boundary_normal_to_outwards(
            self,
            subdomains: list[pp.Grid],
            *,
            dim: int,
        ) -> pp.ad.Operator:
            """Obtain a vector for flipping normal vectors on internal boundaries.

            For a list of subdomains, check if the normal vector on internal boundaries
            point into the internal interface (i.e., away from the fracture), and if so,
            flip the normal vector. The flipping takes the form of an operator that
            multiplies the normal vectors of all faces on fractures, leaves internal
            faces (internal to the subdomain proper, that is) unchanged, but flips the
            relevant normal vectors on subdomain faces that are part of an internal
            boundary.

            Currently, this is a helper method for the computation of outward normals in
            :meth:`outwards_internal_boundary_normals`. Other usage is allowed, but one
            is adviced to carefully consider subdomain lists when combining this with
            other operators.

            Parameters:
                subdomains: List of subdomains.

            Returns:
                Operator with flipped signs if normal vector points inwards.

            """

        def outwards_internal_boundary_normals(
            self,
            interfaces: list[pp.MortarGrid],
            *,
            unitary: bool,
        ) -> pp.ad.Operator:
            """Compute outward normal vectors on internal boundaries.

            Parameters:
                interfaces: List of interfaces.
                unitary: If True, return unit vectors, i.e. normalize by face area.

            Returns:
                Operator computing outward normal vectors on internal boundaries; in
                effect, this is a matrix. Evaluated shape `(num_intf_cells * dim,
                num_intf_cells * dim)`.

            """

        def specific_volume(
            self, grids: list[pp.Grid] | list[pp.MortarGrid]
        ) -> pp.ad.Operator:
            """Specific volume [m^(nd-d)].

            For subdomains, the specific volume is the cross-sectional area/volume of the
            cell, i.e. aperture to the power :math`nd-dim`. For interfaces, the specific
            volume is inherited from the higher-dimensional subdomain neighbor.

            See also:
                :meth:aperture.

            Parameters:
                subdomains: List of subdomain or interface grids.

            Returns:
                Specific volume for each cell.

            """

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
            """Isotropic permeability [m^2].

            Parameters:
                permeability: Permeability, scalar per cell.

            Returns:
                3d isotropic permeability, with nonzero values on the diagonal and zero
                values elsewhere. K is a second order tensor having 3^2 entries per cell,
                represented as an array of length 9*nc. The values are ordered as
                    Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz

            """

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
        """Units of the model provided in ``params['units']``."""
        reference_variable_values: pp.ReferenceVariableValues
        """The model reference values for variables, converted to simulation
        :attr:`units`.

        Reference values can be provided through ``params['reference_values']``.

        """
        solid: pp.SolidConstants
        """Solid constants. Can be provided through
        ``params['material_constants']['solid']``.

        See also :meth:`set_materials`.

        """
        numerical: pp.NumericalConstants
        """Numerical constants. Can be provided through
        ``params['material_constants']['numerical']``.

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
            """Indices for storing time step solutions.

            Index 0 corresponds to the most recent time step with the know solution, 1 -
            to the previous time step, etc.

            Returns:
                An array of the indices of which time step solutions will be stored,
                counting from 0. Defaults to storing the most recently computed solution
                only.

            """

        @property
        def iterate_indices(self) -> np.ndarray:
            """Indices for storing iterate solutions.

            Returns:
                An array of the indices of which iterate solutions will be stored.

            """

        def _is_time_dependent(self) -> bool:
            """Specifies whether the Model problem is time-dependent.

            Returns:
                bool: True if the problem is time-dependent, False otherwise.

            """

    class FluidProtocol(Protocol):
        """This protocol provides declarations of methods defined in the
        :class:`~porepy.compositional.compositional_mixins.FluidMixin`."""

        fluid: pp.Fluid[pp.FluidComponent, pp.Phase[pp.FluidComponent]]
        """Fluid object.

        See also :meth:`create_fluid`.

        """

        def create_fluid(self) -> None:
            """Create the :attr:`fluid` object based on the default or user-provided
            context of components and phases.

            See also:
                :meth:`~porepy.compositional.compositional_mixins.FluidMixin.
                get_components`

                :meth:`~porepy.compositional.compositional_mixins.FluidMixin.
                get_phase_configuration` respectively.

            """

        def assign_thermodynamic_properties_to_phases(self) -> None:
            """Assigns callable properties to the dynamic phase objects, after the
            fluid and all variables are defined.

            See also:
                :meth:`~porepy.compositional.compositional_mixins.FluidMixin.
                assign_thermodynamic_properties_to_phases`.

            """

        def dependencies_of_phase_properties(
            self, phase: pp.Phase
        ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
            """Returns the Callables representing variables, on which the thermodynamic
            properties of phases depend.

            For a more detailed explanation, see :meth:`~porepy.compositional.
            compositional_mixins.dependencies_of_phase_properties.`

            """

    class VariableProtocol(Protocol):
        """This protocol provides the declarations of the methods and the properties,
        typically defined in VariableMixin."""

        def perturbation_from_reference(
            self, variable_name: str, grids: list[pp.Grid]
        ) -> pp.ad.Operator:
            """Perturbation of some quantity ``name`` from its reference value.

            The parameter ``name`` should be the name of a mixed-in method, returning an
            AD operator for given ``grids``.

            ``name`` should also be defined in the model's :attr:`reference_values`.

            This method calls the model method with given ``name`` on given ``grids`` to
            create an operator ``A``. It then fetches the respective reference value and
            wraps it into an AD scalar ``A_0``. The return value is an operator ``A - A_0``.

            Parameters:
                name: Name of the quantity to be perturbed from a reference value.
                grids: List of subdomain or interface grids on which the quantity is
                    defined.

            Returns:
                Operator for the perturbation.

            """

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
            """Creates an operator on boundary grids.

            Parameters:
                name: Name of the variable or operator to be represented on the
                    boundary.
                domains: A sequence of boundary grids on which the operator is defined.

            Raises:
                ValueError: If the passed sequence of domains does not consist entirely
                    of boundary grid.

            Returns:
                An operator of given name representing value on given sequence of
                boundary grids. Can possibly be time-dependent.

            """

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
            """Creates an operator representing Dirichlet, Neumann and Robin boundary
            conditions and projects it to the subdomains from boundary grids.

            Parameters:
                subdomains: List of subdomains.
                dirichlet_operator: Function that returns the Dirichlet boundary
                    condition operator.
                neumann_operator: Function that returns the Neumann boundary condition
                    operator.
                robin_operator: Function that returns the Robin boundary condition
                    operator. Expected to be None for e.g. advective fluxes.
                dim: Dimension of the equation. Defaults to 1.
                name: Name of the resulting operator. Must be unique for an operator.

            Returns:
                Boundary condition representation operator.

            """

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
            """Numerical volume integral over subdomain or interface cells.

            Includes cell volumes and specific volume.

            Parameters:
                integrand: Operator for the integrand. Assumed to be a cell-wise scalar
                    or vector quantity, cf. :code:`dim` argument.
                grids: List of subdomains or interfaces to be integrated over.
                dim: Spatial dimension of the integrand. dim = 1 for scalar problems,
                    dim > 1 for vector problems.

            Returns:
                Operator for the volume integral.

            Raises:
                ValueError: If the grids are not all subdomains or all interfaces.

            """

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
        FluidProtocol,
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

    class CompositionalFlowModelProtocol(Protocol):
        """Protocol declaring a collection of mixed-in methods specific to the
        compositional flow setting."""

        @property
        def primary_variable_names(self) -> list[str]:
            """Returns a list of primary variables, which in the basic set-up consist
            of

            1. pressure,
            2. overall fractions,
            3. tracer fractions,
            4. specific fluid enthalpy.

            Primary variable names are used to define the primary block in the Schur
            elimination in the solution strategy.

            Implemented in :meth:`~porepy.models.compositional_flow.VariablesCF.
            primary_variables`.

            """

        @property
        def primary_equation_names(self) -> list[str]:
            """Returns the list of primary equation, consisting of

            1. pressure equation,
            2. energy balance equation,
            3. mass balance equations per fluid component,
            4. transport equations per solute in compounds in the fluid.

            Note:
                Interface equations, which are non-local equations since they relate
                interface variables and respective subdomain variables on some subdomain
                cells, are not included.

                This might have an effect on the Schur complement in the solution
                strategy.

            Implemented in :meth:`~porepy.models.compositional_flow.PrimaryEquationsCF.
            primary_equation_names`.

            """

        @property
        def _is_ref_phase_eliminated(self) -> bool:
            """Helper property to access the model parameters and check if the
            reference phase is eliminated. Default value is True.

            Implemented in :meth:`~porepy.compositional.compositional_mixins.
            _MixtureDOFHandler._is_ref_phase_eliminated`.

            """

        @property
        def _is_ref_comp_eliminated(self) -> bool:
            """Helper property to access the model parameters and check if the
            reference component is eliminated. Default value is True.

            Implemented in :meth:`~porepy.compositional.compositional_mixins.
            _MixtureDOFHandler._is_ref_comp_eliminated`.

            """
