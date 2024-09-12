from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol, Sequence, Union

import numpy as np
import scipy.sparse as sps

# The following is the way to avoid circular importing.
if TYPE_CHECKING:
    import porepy as pp
# TODO: It is either this, which requires to annotate all the porepy types with
# quotation marks, or:
# if not TYPE_CHECKING:

#     class ModelGeometryProtocol(Protocol):
#         pass

# else:

#     import porepy as pp

#     class ModelGeometryProtocol(Protocol):
#         # correct definition
#
#  I'm not sure what is prettier.
# Maybe the latter is preferable because it ensures that the protocol does not mess with
# the runtime definitions


class ModelGeometryProtocol(Protocol):
    """This class provides geometry related methods and information for a simulation
    model."""

    fracture_network: "pp.fracture_network"
    """Representation of fracture network including intersections."""

    well_network: "pp.WellNetwork3d"
    """Well network."""

    mdg: "pp.MixedDimensionalGrid"
    """Mixed-dimensional grid. Set by the method :meth:`set_md_grid`."""

    nd: int
    """Ambient dimension of the problem. Set by the method :meth:`set_geometry`"""

    @property
    def domain(self) -> "pp.Domain":
        """Domain of the problem."""

    @property
    def fractures(self) -> list["pp.LineFracture"] | list["pp.PlaneFracture"]:
        """Fractures of the problem."""

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid.

        The default values provided in set_domain, set_fractures, grid_type and
        meshing_arguments produce a 2d unit square domain with no fractures and a four
        Cartesian cells.

        """

    def is_well(self, grid: "pp.Grid | pp.MortarGrid") -> bool:
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
        self, subdomains: list["pp.Grid"], codims: list[int]
    ) -> list["pp.MortarGrid"]:
        """Interfaces neighbouring any of the subdomains.

        Parameters:
            subdomains: Subdomains for which to find interfaces.
            codims: Codimension of interfaces to return. The common option is [1], i.e.
                only interfaces between subdomains one dimension apart.

        Returns:
            Unique list of all interfaces neighboring any of the subdomains. Interfaces
            are sorted according to their index, as defined by the mixed-dimensional
            grid.

        """

    def interfaces_to_subdomains(
        self, interfaces: list["pp.MortarGrid"]
    ) -> list["pp.Grid"]:
        """Subdomain neighbours of interfaces.

        Parameters:
            interfaces: List of interfaces for which to find subdomains.

        Returns:
            Unique list of all subdomains neighbouring any of the interfaces. The
            subdomains are sorted according to their index as defined by the
            mixed-dimensional grid.

        """

    def subdomains_to_boundary_grids(
        self, subdomains: Sequence["pp.Grid"]
    ) -> Sequence["pp.BoundaryGrid"]:
        """Boundary grids of subdomains.

        Parameters:
            subdomains: List of subdomains for which to find boundary grids.

        Returns:
            List of boundary grids associated with the provided subdomains.

        """

    def wrap_grid_attribute(
        self,
        grids: Sequence["pp.GridLike"],
        attr: str,
        *,
        dim: int,
    ) -> "pp.ad.DenseArray":
        """Wrap a grid attribute as an ad matrix.

        Parameters:
            grids: List of grids on which the property is defined.
            attr: Grid attribute to wrap. The attribute should be a ndarray and will be
                flattened if it is not already one dimensional.
            dim: Dimensions to include for vector attributes. Intended use is to
                limit the number of dimensions for a vector attribute, e.g. to exclude
                the z-component of a vector attribute in 2d, to achieve compatibility
                with code which is explicitly 2d (e.g. fv discretizations).

        Returns:
            class:`porepy.numerics.ad.DenseArray`: `(shape=(dim * num_cells_in_grids,))`

                The property wrapped as a single ad vector. The values are arranged
                according to the order of the grids in the list, optionally flattened if
                the attribute is a vector.

        Raises:
            ValueError: If one of the grids does not have the attribute.
            ValueError: If the attribute is not a ndarray.

        """

    def basis(
        self, grids: Sequence["pp.GridLike"], dim: int
    ) -> list["pp.ad.SparseArray"]:
        """Return a cell-wise basis for all subdomains.

        The basis is represented as a list of matrices, each of which represents a
        basis function. The individual matrices have shape ``Nc * dim, Nc`` where ``Nc``
        is the total number of cells in the subdomains.

        Examples:
            To extend a cell-wise scalar to a vector field, use
            ``sum([e_i for e_i in basis(subdomains)])``. To restrict to a vector in the
            tangential direction only, use
            ``sum([e_i for e_i in basis(subdomains, dim=nd-1)])``

        See also:
            :meth:`e_i` for the construction of a single basis function.
            :meth:`normal_component` for the construction of a restriction to the
                normal component of a vector only.
            :meth:`tangential_component` for the construction of a restriction to the
                tangential component of a vector only.

        Parameters:
            grids: List of grids on which the basis is defined.
            dim: Dimension of the basis.

        Returns:
            List of pp.ad.SparseArrayArray, each of which represents a basis function.

        """

    def e_i(
        self, grids: Sequence["pp.GridLike"], *, i: int, dim: int
    ) -> "pp.ad.SparseArray":
        """Return a cell-wise basis function in a specified dimension.

        It is assumed that the grids are embedded in a space of dimension dim and
        aligned with the coordinate axes, that is, the reference space of the grid.
        Moreover, the grid is assumed to be planar.

        Example:
            For a grid with two cells, and with `i=1` and `dim=3`, the returned basis
            will be (after conversion to a numpy array)
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
            pp.ad.SparseArray: Ad representation of a matrix with the basis functions as
            columns.

        Raises:
            ValueError: If i is larger than dim.

        """

    def tangential_component(self, subdomains: list["pp.Grid"]) -> "pp.ad.Operator":
        """Compute the tangential component of a vector field.

        The tangential space is defined according to the local coordinates of the
        subdomains, with the tangential space defined by the first `self.nd` components
        of the cell-wise vector. It is assumed that the components of the vector are
        stored with a dimension-major ordering (the dimension varies fastest).

        Parameters:
            subdomains: List of grids on which the vector field is defined.

        Returns:
            Operator extracting tangential component of the vector field and expressing
            it in tangential basis.

        """

    def normal_component(self, subdomains: list["pp.Grid"]) -> "pp.ad.SparseArray":
        """Compute the normal component of a vector field.

        The normal space is defined according to the local coordinates of the
        subdomains, with the normal space defined by final component, e.g., number
        `self.nd-1` (zero offset). of the cell-wise vector. It is assumed that the
        components of a vector are stored with a dimension-major ordering (the dimension
        varies fastest).

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

    def local_coordinates(self, subdomains: list["pp.Grid"]) -> "pp.ad.SparseArray":
        """Ad wrapper around tangential_normal_projections for fractures.

        Parameters:
            subdomains: List of subdomains for which to compute the local coordinates.

        Returns:
            Local coordinates as a pp.ad.SparseArray.

        """

    def subdomain_projections(self, dim: int) -> "pp.ad.SubdomainProjections":
        """Return the projection operators for all subdomains in md-grid.

        The projection operators restrict or prolong a dim-dimensional quantity from the
        full set of subdomains to any subset. Projection operators are constructed once
        and then stored. If you need to use projection operators based on a different
        set of subdomains, please construct them yourself. Alternatively, compose a
        projection from subset A to subset B as
            P_A_to_B = P_full_to_B * P_A_to_full.

        Parameters:
            dim: Dimension of the quantities to be projected.

        Returns:
            proj: Projection operator.

        """

    def domain_boundary_sides(
        self, domain: "pp.Grid | pp.BoundaryGrid", tol: Optional[float] = 1e-10
    ) -> "pp.domain.DomainSides":
        """Obtain indices of the faces lying on the sides of the domain boundaries.

        The method is primarily intended for box-shaped domains. However, it can also be
        applied to non-box-shaped domains (e.g., domains with perturbed boundary nodes)
        provided `tol` is tuned accordingly.

        Parameters:
            domain: Subdomain or boundary grid.
            tol: Tolerance used to determine whether a face center lies on a boundary
                side.

        Returns:
            NamedTuple containing the domain boundary sides. Available attributes are:

                - all_bf (np.ndarray of int): indices of the boundary faces.
                - east (np.ndarray of bool): flags of the faces lying on the East side.
                - west (np.ndarray of bool): flags of the faces lying on the West side.
                - north (np.ndarray of bool): flags of the faces lying on the North
                    side.
                - south (np.ndarray of bool): flags of the faces lying on the South
                    side.
                - top (np.ndarray of bool): flags of the faces lying on the Top side.
                - bottom (np.ndarray of bool): flags of the faces lying on Bottom side.

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
        subdomains: list["pp.Grid"],
        *,
        dim: int,
    ) -> "pp.ad.Operator":
        """Obtain a vector for flipping normal vectors on internal boundaries.

        For a list of subdomains, check if the normal vector on internal boundaries
        point into the internal interface (e.g., into the fracture), and if so, flip the
        normal vector. The flipping takes the form of an operator that multiplies the
        normal vectors of all faces on fractures, leaves internal faces (internal to the
        subdomain proper, that is) unchanged, but flips the relevant normal vectors on
        subdomain faces that are part of an internal boundary.

        Currently, this is a helper method for the computation of outward normals in
        :meth:`outwards_internal_boundary_normals`. Other usage is allowed, but one
        is adviced to carefully consider subdomain lists when combining this with other
        operators.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator with flipped signs if normal vector points inwards.

        """

    def outwards_internal_boundary_normals(
        self,
        interfaces: list["pp.MortarGrid"],
        *,
        unitary: bool,
    ) -> "pp.ad.Operator":
        """Compute outward normal vectors on internal boundaries.

        Parameters:
            interfaces: List of interfaces.
            unitary: If True, return unit vectors, i.e. normalize by face area.

        Returns:
            Operator computing outward normal vectors on internal boundaries; in effect,
            this is a matrix. Evaluated shape `(num_intf_cells * dim,
            num_intf_cells * dim)`.

        """


class SolutionStrategyProtocol(Protocol):
    """This is a class that specifies methods that a model must implement to
    be compatible with the linearization and time stepping methods.

    """

    convergence_status: bool
    """Whether the non-linear iteration has converged."""

    equation_system: "pp.ad.EquationSystem"
    """Equation system manager. Will be set by :meth:`set_equation_system_manager`.

    """
    linear_system: tuple[sps.spmatrix, np.ndarray]
    """The linear system to be solved in each iteration of the non-linear solver.
    The tuple contains the sparse matrix and the right hand side residual vector.

    """
    params: dict
    """Dictionary of parameters."""
    units: "pp.Units"
    """Units of the model. See also :meth:`set_units`."""
    fluid: "pp.FluidConstants"
    """Fluid constants. See also :meth:`set_materials`."""
    solid: "pp.SolidConstants"
    """Solid constants. See also :meth:`set_materials`."""
    time_manager: "pp.TimeManager"
    """Time manager for the simulation."""
    restart_options: dict
    """Restart options for restart from pvd as expected restart routines within
    :class:`~porepy.viz.data_saving_model_mixin.DataSavingMixin` The template is
    provided in `SolutionStrategy.__init__`.

    """
    ad_time_step: "pp.ad.Scalar"
    """Time step as an automatic differentiation scalar."""
    nonlinear_solver_statistics: "pp.SolverStatistics"
    """Solver statistics for the nonlinear solver."""

    @property
    def time_step_indices(self) -> np.ndarray:
        """Indices for storing time step solutions.

        Note:
            (Previous) Time step indices should start with 1.

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


class VariableProtocol(Protocol):
    def perturbation_from_reference(
        self, variable_name: str, grids: "pp.GridLikeSequence"
    ):
        """Perturbation of a variable from its reference value.

        The parameter :code:`variable_name` should be the name of a variable so that
        :code:`self.variable_name()` and `self.reference_variable_name()` are valid
        calls. These methods will be provided by mixin classes; normally this will be a
        subclass of :class:`VariableMixin`.

        The returned operator will be of the form
        :code:`self.variable_name(grids) - self.reference_variable_name(grids)`.

        Parameters:
            variable_name: Name of the variable.
            grids: List of subdomain or interface grids on which the variable is
                defined.

        Returns:
            Operator for the perturbation.

        """

    def create_variables(self) -> None:
        """Assign primary variables to subdomains and interfaces of the
        mixed-dimensional grid.

        """


class BoundaryConditionProtocol(Protocol):
    """Mixin class for boundary conditions.

    This class is intended to be used together with the other model classes providing
    generic functionality for boundary conditions.

    """

    def create_boundary_operator(
        self, name: str, domains: Sequence["pp.BoundaryGrid"]
    ) -> "pp.ad.TimeDependentDenseArray":
        """
        Parameters:
            name: Name of the variable or operator to be represented on the boundary.
            domains: A sequence of boundary grids on which the operator is defined.

        Raises:
            ValueError: If the passed sequence of domains does not consist entirely
                of instances of boundary grid.

        Returns:
            An operator of given name representing time-dependent value on given
            sequence of boundary grids.

        """

    def _combine_boundary_operators(
        self,
        subdomains: Sequence["pp.Grid"],
        dirichlet_operator: Callable[[Sequence["pp.BoundaryGrid"]], "pp.ad.Operator"],
        neumann_operator: Callable[[Sequence["pp.BoundaryGrid"]], "pp.ad.Operator"],
        robin_operator: Optional[
            Union[None, Callable[[Sequence["pp.BoundaryGrid"]], "pp.ad.Operator"]]
        ],
        bc_type: Callable[["pp.Grid"], "pp.AbstractBoundaryCondition"],
        name: str,
        dim: int = 1,
    ) -> "pp.ad.Operator":
        """Creates an operator representing Dirichlet, Neumann and Robin boundary
        conditions and projects it to the subdomains from boundary grids.

        Parameters:
            subdomains: List of subdomains.
            dirichlet_operator: Function that returns the Dirichlet boundary condition
                operator.
            neumann_operator: Function that returns the Neumann boundary condition
                operator.
            robin_operator: Function that returns the Robin boundary condition operator.
                Expected to be None for e.g. advective fluxes.
            dim: Dimension of the equation. Defaults to 1.
            name: Name of the resulting operator. Must be unique for an operator.

        Returns:
            Boundary condition representation operator.

        """

    def update_all_boundary_conditions(self) -> None:
        """This method is called before a new time step to set the values of the
        boundary conditions.

        This implementation updates only the filters for Dirichlet and Neumann
        values. The specific boundary condition values should be updated in
        overrides by models.

        Note:
            One can use the convenience method `update_boundary_condition` for each
            boundary condition value.

        """


class EquationProtocol(Protocol):
    """Generic class for vector balance equations.

    In the only known use case, the balance equation is the momentum balance equation,

        d_t(momentum) + div(stress) - source = 0,

    with momentum frequently being zero. All terms need to be specified in order to
    define an equation.

    """

    def volume_integral(
        self,
        integrand: "pp.ad.Operator",
        grids: Union[list["pp.Grid"], list["pp.MortarGrid"]],
        dim: int,
    ) -> "pp.ad.Operator":
        """Numerical volume integral over subdomain or interface cells.

        Includes cell volumes and specific volume.

        Parameters:
            integrand: Operator for the integrand. Assumed to be a cell-wise scalar or
                vector quantity, cf. :code:`dim` argument.
            grids: List of subdomains or interfaces to be integrated over.
            dim: Spatial dimension of the integrand. dim = 1 for scalar problems, dim >
                1 for vector problems.

        Returns:
            Operator for the volume integral.

        Raises:
            ValueError: If the grids are not all subdomains or all interfaces.

        """

    def set_equations(self) -> None:
        """Set equations for the subdomains and interfaces."""


class DataSavingProtocol(Protocol):
    """Class for saving data from a simulation model.

    Contract with other classes:
        The model should/may call save_data_time_step() at the end of each time step.

    """

    exporter: "pp.Exporter"
    """Exporter for visualization."""

    def save_data_time_step(self) -> None:
        """Export the model state at a given time step, and log time.
        The options for exporting times are:
            * None: All time steps are exported
            * list: Export if time is in the list. If the list is empty, then no times
            are exported.

        In addition, save the solver statistics to file if the option is set.

        """

    def initialize_data_saving(self) -> None:
        """Initialize data saving.

        This method is called by :meth:`prepare_simulation` to initialize the exporter,
        and any other data saving functionality (e.g., empty data containers to be
        appended in :meth:`save_data_time_step`).

        In addition, set path for storing solver statistics data to file for each time
        step.

        """

    def load_data_from_vtu(
        self,
        vtu_files: Union[Path, list[Path]],
        time_index: int,
        times_file: Optional[Path] = None,
        keys: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ) -> None:
        """Initialize data in the model by reading from a pvd file.

        Parameters:
            vtu_files: path(s) to vtu file(s)
            keys: keywords addressing cell data to be transferred. If 'None', the
                mixed-dimensional grid is checked for keywords corresponding to primary
                variables identified through pp.TIME_STEP_SOLUTIONS.
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
        keys: Optional[Union[str, list[str]]] = None,
    ) -> None:
        """Initialize data in the model by reading from a pvd file.

        Parameters:
            pvd_file: path to pvd file with exported vtu files.
            is_mdg_pvd: flag controlling whether pvd file is a mdg file, i.e., generated
                with Exporter._export_mdg_pvd() or Exporter.write_pvd().
            times_file: path to json file storing history of time and time step size.
            keys: keywords addressing cell data to be transferred. If 'None', the
                mixed-dimensional grid is checked for keywords corresponding to primary
                variables identified through pp.TIME_STEP_SOLUTIONS.

        Raises:
            ValueError: if incompatible file type provided.

        """

class SolutionStrategyMBProtocol(Protocol):
    """This is a class that specifies methods that a momentum balance model must
    implement to be compatible with the linearization and time stepping methods.

    """
    displacement_variable: str = "u"
    """Name of the displacement variable."""

    interface_displacement_variable: str = "u_interface"
    """Name of the displacement variable on fracture-matrix interfaces."""

    contact_traction_variable: str = "t"
    """Name of the contact traction variable."""

    # Discretization
    stress_keyword: str = "mechanics"
    """Keyword for stress term.

    Used to access discretization parameters and store discretization matrices.

    """

    def contact_mechanics_numerical_constant(
        self, subdomains: list["pp.Grid"]
    ) -> "pp.ad.Operator":
        """Numerical constant for the contact problem [m^-1].

        A physical interpretation of this constant is a characteristic length of
        the fracture, as it appears as a scaling of displacement jumps when
        comparing to nondimensionalized contact tractions.

        Parameters:
            subdomains: List of subdomains. Only the first is used.

        Returns:
            c_num: Numerical constant.

        """


    def contact_mechanics_open_state_characteristic(
        self, subdomains: list["pp.Grid"]
    ) -> "pp.ad.Operator":
        r"""Characteristic function used in the tangential contact mechanics relation.
        Can be interpreted as an indicator of the fracture cells in the open state.
        Used to make the problem well-posed in the case b_p is zero.

        The function reads
        .. math::
            \begin{equation}
            \text{characteristic} =
            \begin{cases}
                1 & \\text{if }~~ |b_p| < tol  \\
                0 & \\text{otherwise.}
            \end{cases}
            \end{equation}
        or simply `1 if (abs(b_p) < tol) else 0`

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            characteristic: Characteristic function.

        """

class BoundaryConditionMBProtocol(Protocol):
    def bc_type_mechanics(self, sd: "pp.Grid") -> "pp.BoundaryConditionVectorial":
        """Define type of boundary conditions.

        Parameters:
            sd: Subdomain grid.

        Returns:
            Boundary condition representation. Dirichlet on all global boundaries,
            Dirichlet also on fracture faces.

        """

    def combine_boundary_operators_mechanical_stress(
        self, subdomains: list["pp.Grid"]
    ) -> "pp.ad.Operator":
        """Combine mechanical stress boundary operators.

        Note that the default Robin operator is the same as that of Neumann. Override
        this method to define and assign another boundary operator of your choice. The
        new operator should then be passed as an argument to the
        _combine_boundary_operators method, just like self.mechanical_stress is passed
        to robin_operator in the default setup.

        Parameters:
            subdomains: List of the subdomains whose boundary operators are to be
                combined.

        Returns:
            The combined mechanical stress boundary operator.

        """

class VariablesMBProtocol(Protocol):
    def displacement(self, domains: "pp.SubdomainsOrBoundaries") -> "pp.ad.Operator":
        """Displacement in the matrix.

        Parameters:
            domains: List of subdomains or interface grids where the displacement is
                defined. Should be the matrix subdomains.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the subdomains is not equal to the ambient
                dimension of the problem.
            ValueError: If the method is called on a mixture of grids and boundary
                grids

        """

    def interface_displacement(self, interfaces: list["pp.MortarGrid"]) -> "pp.ad.Operator":
        """Displacement on fracture-matrix interfaces.

        Parameters:
            interfaces: List of interface grids where the displacement is defined.
                Should be between the matrix and fractures.

        Returns:
            Variable for the displacement.

        Raises:
            ValueError: If the dimension of the interfaces is not equal to the ambient
                dimension of the problem minus one.

        """

    def contact_traction(self, subdomains: list["pp.Grid"]) -> "pp.ad.Operator":
        """Fracture contact traction [-].

        Parameters:
            subdomains: List of subdomains where the contact traction is defined. Should
                be of co-dimension one, i.e. fractures.

        Returns:
            Variable for nondimensionalized fracture contact traction.

        """

    def displacement_jump(self, subdomains: list["pp.Grid"]) -> "pp.ad.Operator":
        """Displacement jump on fracture-matrix interfaces.

        Parameters:
            subdomains: List of subdomains where the displacement jump is defined.
                Should be a fracture subdomain.

        Returns:
            Operator for the displacement jump.

        Raises:
             AssertionError: If the subdomains are not fractures, i.e. have dimension
                `nd - 1`.

        """

class ConstitutiveLawsMBProtocol(Protocol):

    def mechanical_stress(self, domains: "pp.SubdomainsOrBoundaries") -> "pp.ad.Operator":
        """Linear elastic mechanical stress.

        .. note::
            The below discretization assumes the stress is discretized with a Mpsa
            finite volume discretization. Other discretizations may be possible, but are
            not available in PorePy at the moment, and would likely require changes to
            this method.

        Parameters:
            grids: List of subdomains or boundary grids. If subdomains, should be of
                co-dimension 0.

        Raises:
            ValueError: If any grid is not of co-dimension 0.
            ValueError: If any the method is called with a mixture of subdomains and
                boundary grids.

        Returns:
            Ad operator representing the mechanical stress on the faces of the grids.

        """

    def fracture_stress(self, interfaces: list["pp.MortarGrid"]) -> "pp.ad.Operator":
        """Fracture stress on interfaces [Pa]."""


    def friction_bound(self, subdomains: list["pp.Grid"]) -> "pp.ad.Operator":
        """Friction bound [-].

        Dimensionless, since fracture deformation equations consider non-dimensional
        tractions. In this class, the bound is given by

        .. math::
            - F t_n

        where :math:`F` is the friction coefficient and :math:`t_n` is the normal
        component of the contact traction. TODO: Rename class to CoulombFrictionBound?

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise friction bound operator [-].

        """

    def fracture_gap(self, subdomains: list["pp.Grid"]) -> "pp.ad.Operator":
        """Fracture gap [m].

        Parameters:
            subdomains: List of subdomains where the gap is defined.

        Raises:
            ValueError: If the reference fracture gap is smaller than the maximum
                fracture closure. This can lead to negative openings from the
                Barton-Bandis model.

        Returns:
            Cell-wise fracture gap operator.

        """

    def stiffness_tensor(self, subdomain: "pp.Grid") -> "pp.FourthOrderTensor":
        """Stiffness tensor [Pa].

        Parameters:
            subdomain: Subdomain where the stiffness tensor is defined.

        Returns:
            Cell-wise stiffness tensor in SI units.

        """

    def characteristic_displacement(self, subdomains: list["pp.Grid"]) -> \
            "pp.ad.Operator":
        """Characteristic displacement [m].

        The value is fetched from the solid constants. See also the method
        :meth:`characteristic_contact_traction` and its documentation.

        Parameters:
            subdomains: List of subdomains where the characteristic displacement is
                defined.

        Returns:
            Scalar operator representing the characteristic displacement.

        """

    def characteristic_contact_traction(
        self, subdomains: list["pp.Grid"]
    ) -> "pp.ad.Operator":
        """Characteristic traction [Pa].

        The value is computed from the solid constants and the characteristic
        displacement. Inversion of this relationship, i.e.,
        u_char=u_char(t_char), can be done in a mixin overriding the
        characteristic sizes. This may be beneficial if the characteristic
        traction is easier to estimate than the characteristic displacement.

        Parameters:
            subdomains: List of subdomains where the characteristic traction is defined.

        Returns:
            Scalar operator representing the characteristic traction.

        """

    def gravity_force(
        self,
        grids: Union[list["pp.Grid"], list["pp.MortarGrid"]],
        material: Literal["fluid", "solid"],
    ) -> "pp.ad.Operator":
        """Gravity force term on either subdomains or interfaces.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.
            material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector representing the gravity force.

        """

    def stress(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Stress operator.

        Parameters:
            subdomains: List of subdomains where the stress is defined.

        Returns:
            Operator for the stress.

        """

class MomentumBalanceProtocol(SolutionStrategyMBProtocol,
                              VariablesMBProtocol,
                              ConstitutiveLawsMBProtocol,
                              BoundaryConditionMBProtocol,
                              Protocol):
    """Protocol for the momentum balance model"""

class PorePyModel(
    BoundaryConditionProtocol,
    EquationProtocol,
    VariableProtocol,
    ModelGeometryProtocol,
    DataSavingProtocol,
    SolutionStrategyProtocol,
    Protocol,
):
    """This is a protocol meant for subclassing. Its parents are not meant for
    subclassing (TODO)"""


# TODO: ALL DOCSTRINGS HERE
# TODO: Try to remove all reduntant type: ignore
# Specific mixins should(?) annotate on the class level what attributes do they define

# 1. locality of definitions (this mixin relies only on geometry)
# 2. one or multiple protocols?
# 3. where to put the docstrings?
# 4. We need to inherit.


class PorousMediaProtocol(Protocol):
    def porosity(self, subdomains: "list[pp.Grid]") -> "pp.ad.Operator":
        """Porous media porosity.

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            The porosity represented as an Ad operator [-].

        """

    def reference_porosity(self, subdomains: list["pp.Grid"]) -> "pp.ad.Operator":
        """Reference porosity.

        Parameters:
            subdomains: List of subdomains where the reference porosity is defined.

        Returns:
            Reference porosity operator.

        """

    def gravity_force(
        self,
        grids: Union[list["pp.Grid"], list["pp.MortarGrid"]],
        material: Literal["fluid", "solid"],
    ) -> "pp.ad.Operator":
        """Gravity force term on either subdomains or interfaces.

        Parameters:
            grids: List of subdomain or interface grids where the vector source is
                defined.
            material: Name of the material. Could be either "fluid" or "solid".

        Returns:
            Cell-wise nd-vector representing the gravity force.

        """


class PressureProtocol(Protocol):
    pressure_variable: str
    """Name of the pressure variable."""

    def pressure(self, domains: "pp.SubdomainsOrBoundaries") -> "pp.ad.Operator":
        """Pressure term. Either a primary variable if subdomains are provided a
        boundary condition operator if boundary grids are provided.

        Parameters:
            domains: List of subdomains or boundary grids.

        Raises:
            ValueError: If the grids are not all subdomains or all boundary grids.

        Returns:
            Operator representing the pressure [Pa].

        """

    def reference_pressure(self, subdomains: "list[pp.Grid]") -> "pp.ad.Operator":
        """Reference pressure.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the reference pressure [Pa].

        """


class AdvectiveFluxProtocol(PressureProtocol, Protocol):
    def advective_flux(
        self,
        subdomains: "list[pp.Grid]",
        advected_entity: "pp.ad.Operator",
        discr: "pp.ad.UpwindAd",
        bc_values: "pp.ad.Operator",
        interface_flux: Optional[
            Callable[[list["pp.MortarGrid"]], "pp.ad.Operator"]
        ] = None,
    ) -> "pp.ad.Operator":
        """An operator represetning the advective flux on subdomains.

        .. note::
            The implementation assumes that the advective flux is discretized using a
            standard upwind discretization. Other discretizations may be possible, but
            this has not been considered.

        Parameters:
            subdomains: List of subdomains.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.
            bc_values: Boundary conditions for the advective flux.
            interface_flux: Interface flux operator/variable. If subdomains have no
                neighboring interfaces, this argument can be omitted.

        Returns:
            Operator representing the advective flux.

        """

    def interface_advective_flux(
        self,
        interfaces: list["pp.MortarGrid"],
        advected_entity: "pp.ad.Operator",
        discr: "pp.ad.UpwindCouplingAd",
    ) -> "pp.ad.Operator":
        """An operator represetning the advective flux on interfaces.

        Parameters:
            interfaces: List of interface grids.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.

        Returns:
            Operator representing the advective flux on the interfaces.

        """

    def well_advective_flux(
        self,
        interfaces: list["pp.MortarGrid"],
        advected_entity: "pp.ad.Operator",
        discr: "pp.ad.UpwindCouplingAd",
    ) -> "pp.ad.Operator":
        """An operator represetning the advective flux on interfaces.

        Parameters:
            interfaces: List of interface grids.
            advected_entity: Operator representing the advected entity.
            discr: Discretization of the advective flux.

        Returns:
            Operator representing the advective flux on the interfaces.

        """


class DarcyFluxProtocol(PressureProtocol, Protocol):
    def darcy_flux(self, domains: "pp.SubdomainsOrBoundaries") -> "pp.ad.Operator":
        """Discretization of Darcy's law.

        Parameters:
            domains: List of domains where the Darcy flux is defined.

        Raises:
            ValueError if the domains are a mixture of grids and boundary grids.

        Returns:
            Face-wise Darcy flux in cubic meters per second.

        """

    def interface_darcy_flux(
        self, interfaces: "list[pp.MortarGrid]"
    ) -> "pp.ad.MixedDimensionalVariable":
        """Interface Darcy flux.

        Integrated over faces in the mortar grid.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the interface Darcy flux [kg * m^2 * s^-2].

        """

    def well_flux(
        self, interfaces: "list[pp.MortarGrid]"
    ) -> "pp.ad.MixedDimensionalVariable":
        """Variable for the volumetric well flux.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Variable representing the Darcy-like well flux [kg * m^2 * s^-2].

        """

    def permeability(self, subdomains: "list[pp.Grid]") -> "pp.ad.Operator":
        """Permeability [m^2].

        The permeability is quantity which enters the discretized equations in a form
        that cannot be differentiated by Ad (this is at least true for a subset of the
        relevant discretizations). For this reason, the permeability is not returned as
        an Ad operator, but as a numpy array, to be wrapped as a SecondOrderTensor and
        passed as a discretization parameter.

        Parameters:
            subdomains: Subdomains where the permeability is defined.

        Returns:
            Cell-wise permeability tensor.

        """

    def normal_permeability(
        self, interfaces: list["pp.MortarGrid"]
    ) -> "pp.ad.Operator":
        """Normal permeability [m^2].

        Contrary to the permeability, the normal permeability typically enters into the
        discrete equations in a form that can be differentiated by Ad. For this reason,
        the normal permeability is returned as an Ad operator.

        Parameters:
            interfaces: List of interface grids. Not used in this implementation, but
                included for compatibility with other implementations.

        Returns:
            Scalar normal permeability on the interfaces between subdomains, represented
            as an Ad operator. The value is picked from the solid constants.

        """

    def combine_boundary_operators_darcy_flux(
        self, subdomains: list["pp.Grid"]
    ) -> "pp.ad.Operator":
        """Combine Darcy flux boundary operators.

        Note that the default Robin operator is the same as that of Neumann. Override
        this method to define and assign another boundary operator of your choice. The
        new operator should then be passed as an argument to the
        _combine_boundary_operators method, just like self.darcy_flux is passed to
        robin_operator in the default setup.

        Parameters:
            subdomains: List of the subdomains whose boundary operators are to be
                combined.

        Returns:
            The combined Darcy flux boundary operator.

        """


class FluidMassBalanceProtocol(
    PorousMediaProtocol, AdvectiveFluxProtocol, DarcyFluxProtocol, Protocol
):
    bc_data_fluid_flux_key: str
    bc_data_darcy_flux_key: str
    """Name of the boundary data for the Neuman boundary condition."""
    interface_darcy_flux_variable: str
    """Name of the primary variable representing the Darcy flux on interfaces of
    codimension one."""

    well_flux_variable: str
    """Name of the primary variable representing the well flux on interfaces of
    codimension two."""

    mobility_keyword: str
    """Keyword for mobility factor.

    Used to access discretization parameters and store discretization matrices.

    """
    darcy_keyword: str
    """Keyword for Darcy flux term.

    Used to access discretization parameters and store discretization matrices.

    """

    def fluid_density(
        self, subdomains: "pp.SubdomainsOrBoundaries"
    ) -> "pp.ad.Operator":
        """Fluid density.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Fluid density as a function of pressure [kg * m^-3].

        """

    def mobility(self, subdomains: "pp.SubdomainsOrBoundaries") -> "pp.ad.Operator":
        """Mobility of the fluid flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator representing the mobility [m * s * kg^-1].

        """

    def mobility_discretization(self, subdomains: "list[pp.Grid]") -> "pp.ad.UpwindAd":
        """Discretization of the fluid mobility factor.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Discretization of the fluid mobility.

        """

    def bc_type_fluid_flux(self, sd: "pp.Grid") -> "pp.BoundaryCondition":
        """Boundary conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned.

        """

    def interface_darcy_flux_equation(
        self, interfaces: list["pp.MortarGrid"]
    ) -> "pp.ad.Operator":
        """Darcy flux on interfaces.

        The units of the Darcy flux are [m^2 Pa / s], see note in :meth:`darcy_flux`.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Operator representing the Darcy flux equation on the interfaces.

        """

    def interface_mobility_discretization(
        self, interfaces: list["pp.MortarGrid"]
    ) -> "pp.ad.UpwindCouplingAd":
        """Discretization of the interface mobility.

        Parameters:
            interfaces: List of interface grids.

        Returns:
            Discretization for the interface mobility.

        """

    def bc_type_darcy_flux(self, sd: "pp.Grid") -> "pp.BoundaryCondition":
        """Boundary conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object. Per default Dirichlet-type BC are assigned,
            requiring pressure values on the bonudary.

        """

    def fluid_viscosity(
        self, subdomains: "pp.SubdomainsOrBoundaries"
    ) -> "pp.ad.Operator":
        """Fluid viscosity .

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Operator for fluid viscosity [Pa * s], represented as an Ad operator.

        """


class TensorProtocol(Protocol):
    def isotropic_second_order_tensor(
        self, subdomains: list["pp.Grid"], permeability: "pp.ad.Operator"
    ) -> "pp.ad.Operator":
        """Isotropic permeability [m^2].

        Parameters:
            permeability: Permeability, scalar per cell.

        Returns:
            3d isotropic permeability, with nonzero values on the diagonal and zero
            values elsewhere. K is a second order tensor having 3^2 entries per cell,
            represented as an array of length 9*nc. The values are ordered as
                Kxx, Kxy, Kxz, Kyx, Kyy, Kyz, Kzx, Kzy, Kzz

        """

    def operator_to_SecondOrderTensor(
        self,
        sd: "pp.Grid",
        operator: "pp.ad.Operator",
        fallback_value: "pp.number",
    ) -> "pp.SecondOrderTensor":
        """Convert Ad operator to PorePy tensor representation.

        Parameters:
            sd: Subdomain where the operator is defined.
            operator: Operator to convert.

        Returns:
            SecondOrderTensor representation of the operator.

        """


class MixedDimensionalProtocol(Protocol):
    def specific_volume(
        self, grids: Union[list["pp.Grid"], list["pp.MortarGrid"]]
    ) -> "pp.ad.Operator":
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

    def aperture(self, subdomains: list["pp.Grid"]) -> "pp.ad.Operator":
        """Aperture [m].

        Aperture is a characteristic thickness of a cell, with units [m]. It's value is
        1 in matrix, thickness of fractures and "side length" of cross-sectional
        area/volume (or "specific volume") for intersections of dimension 1 and 0.

        See also:
            :meth:specific_volume.

        Parameters:
            subdomains: List of subdomain grids.

        Returns:
            Ad operator representing the aperture for each cell in each subdomain.

        """
