from typing import Literal, Optional, Protocol, Sequence

import porepy as pp


class ModelGeometryProtocol(Protocol):
    """This class provides geometry related methods and information for a simulation
    model."""

    fracture_network: pp.fracture_network
    """Representation of fracture network including intersections."""

    well_network: pp.WellNetwork3d
    """Well network."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid. Set by the method :meth:`set_md_grid`."""

    @property
    def nd(self) -> int:
        """Ambient dimension of the problem. Set by the method :meth:`set_geometry`"""

    @property
    def units(self) -> pp.Units:
        """Unit system."""

    params: dict

    @property
    def domain(self) -> pp.Domain:
        """Domain of the problem."""

    @property
    def fractures(self) -> list[pp.LineFracture] | list[pp.PlaneFracture]:
        """Fractures of the problem."""

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid.

        The default values provided in set_domain, set_fractures, grid_type and
        meshing_arguments produce a 2d unit square domain with no fractures and a four
        Cartesian cells.

        """

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
            codims: Codimension of interfaces to return. The common option is [1], i.e.
                only interfaces between subdomains one dimension apart.

        Returns:
            Unique list of all interfaces neighboring any of the subdomains. Interfaces
            are sorted according to their index, as defined by the mixed-dimensional
            grid.

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
            attr: Grid attribute to wrap. The attribute should be a ndarray and will be
                flattened if it is not already one dimensional.
            dim: Dimensions to include for vector attributes. Intended use is to
                limit the number of dimensions for a vector attribute, e.g. to exclude
                the z-component of a vector attribute in 2d, to achieve compatibility
                with code which is explicitly 2d (e.g. fv discretizations).

        Returns:
            class:`porepy.numerics.ad.DenseArray`: ``(shape=(dim * num_cells_in_grids,))``

                The property wrapped as a single ad vector. The values are arranged
                according to the order of the grids in the list, optionally flattened if
                the attribute is a vector.

        Raises:
            ValueError: If one of the grids does not have the attribute.
            ValueError: If the attribute is not a ndarray.

        """

    def basis(self, grids: Sequence[pp.GridLike], dim: int) -> list[pp.ad.SparseArray]:
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
        self, grids: Sequence[pp.GridLike], *, i: int, dim: int
    ) -> pp.ad.SparseArray:
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

    def tangential_component(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
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

    def normal_component(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
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

    def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
        """Ad wrapper around tangential_normal_projections for fractures.

        Parameters:
            subdomains: List of subdomains for which to compute the local coordinates.

        Returns:
            Local coordinates as a pp.ad.SparseArray.

        """

    def subdomain_projections(self, dim: int) -> pp.ad.SubdomainProjections:
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
        self, domain: pp.Grid | pp.BoundaryGrid, tol: Optional[float] = 1e-10
    ) -> pp.domain.DomainSides:
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
                - north (np.ndarray of bool): flags of the faces lying on the North side.
                - south (np.ndarray of bool): flags of the faces lying on the South side.
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
        subdomains: list[pp.Grid],
        *,
        dim: int,
    ) -> pp.ad.Operator:
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
        interfaces: list[pp.MortarGrid],
        *,
        unitary: bool,
    ) -> pp.ad.Operator:
        """Compute outward normal vectors on internal boundaries.

        Parameters:
            interfaces: List of interfaces.
            unitary: If True, return unit vectors, i.e. normalize by face area.

        Returns:
            Operator computing outward normal vectors on internal boundaries; in effect,
            this is a matrix. Evaluated shape `(num_intf_cells * dim,
            num_intf_cells * dim)`.

        """


class PorePyModel(ModelGeometryProtocol, Protocol):
    """This is a protocol meant for subclassing (TODO)"""
