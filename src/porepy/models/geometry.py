"""Geometry definition for simulation setup.

"""

from __future__ import annotations

import copy
from typing import Literal, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.fracs.fracture_network_3d import FractureNetwork3d


class ModelGeometry:
    """This class provides geometry related methods and information for a simulation
    model."""

    # Define attributes to be assigned later
    fracture_network: pp.fracture_network
    """Representation of fracture network including intersections."""
    well_network: pp.WellNetwork3d
    """Well network."""
    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid. Set by the method :meth:`set_md_grid`."""
    nd: int
    """Ambient dimension of the problem. Set by the method :meth:`set_geometry`"""
    units: pp.Units
    """Unit system."""
    params: dict
    """Parameters for the model."""
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities.
    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    """

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid.

        The default values provided in set_domain, set_fractures, grid_type and
        meshing_arguments produce a 2d unit square domain with no fractures and a four
        Cartesian cells.

        """
        # Create the geometry through domain amd fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)
        # Create a mixed-dimensional grid.
        self.mdg = pp.create_mdg(
            self.grid_type(),
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)

    @property
    def domain(self) -> pp.Domain:
        """Domain of the problem."""
        return self._domain

    def set_domain(self) -> None:
        """Set domain of the problem.

        Defaults to a 2d unit square domain.
        Override this method to define a geometry with a different domain.

        """
        self._domain = nd_cube_domain(2, self.solid.convert_units(1.0, "m"))

    @property
    def fractures(self) -> Union[list[pp.LineFracture], list[pp.PlaneFracture]]:
        """List of fractures in the fracture network."""
        return self._fractures

    def set_fractures(self) -> None:
        """Set fractures in the fracture network.

        Override this method to define a geometry with fractures.

        """
        self._fractures: list = []

    def set_well_network(self) -> None:
        """Assign well network class."""
        self.well_network = pp.WellNetwork3d(domain=self._domain)

    def is_well(self, grid: pp.Grid | pp.MortarGrid) -> bool:
        """Check if a subdomain is a well.

        Parameters:
            sd: Subdomain to check.

        Returns:
            True if the subdomain is a well, False otherwise.

        """
        if isinstance(grid, pp.Grid):
            return getattr(grid, "well_num", -1) >= 0
        elif isinstance(grid, pp.MortarGrid):
            return False
        else:
            raise ValueError("Unknown grid type.")

    def grid_type(self) -> Literal["simplex", "cartesian", "tensor_grid"]:
        """Grid type for the mixed-dimensional grid.

        Returns:
            Grid type for the mixed-dimensional grid.

        """
        return self.params.get("grid_type", "cartesian")

    def meshing_arguments(self) -> dict[str, float]:
        """Meshing arguments for mixed-dimensional grid generation.

        Returns:
            Meshing arguments compatible with
            :meth:`~porepy.grids.mdg_generation.create_mdg`.

        """
        # Default value of 1/2, scaled by the length unit.
        cell_size = self.solid.convert_units(0.5, "m")
        default_meshing_args: dict[str, float] = {"cell_size": cell_size}
        # If meshing arguments are provided in the params, they should already be
        # scaled by the length unit.
        return self.params.get("meshing_arguments", default_meshing_args)

    def meshing_kwargs(self) -> dict:
        """Keyword arguments for md-grid creation.

        Returns:
            Keyword arguments compatible with pp.create_mdg() method.

        """
        meshing_kwargs = self.params.get("meshing_kwargs", None)
        if meshing_kwargs is None:
            meshing_kwargs = {}
        return meshing_kwargs

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
        # Initialize list of interfaces, build it up one subdomain at a time.
        interfaces: list[pp.MortarGrid] = []
        for sd in subdomains:
            for intf in self.mdg.subdomain_to_interfaces(sd):
                if intf not in interfaces and intf.codim in codims:
                    interfaces.append(intf)
        return self.mdg.sort_interfaces(interfaces)

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
        subdomains: list[pp.Grid] = []
        for interface in interfaces:
            for sd in self.mdg.interface_to_subdomain_pair(interface):
                if sd not in subdomains:
                    subdomains.append(sd)
        return self.mdg.sort_subdomains(subdomains)

    def subdomains_to_boundary_grids(
        self, subdomains: Sequence[pp.Grid]
    ) -> Sequence[pp.BoundaryGrid]:
        """Boundary grids of subdomains.

        Parameters:
            subdomains: List of subdomains for which to find boundary grids.

        Returns:
            List of boundary grids associated with the provided subdomains.

        """
        boundary_grids = [self.mdg.subdomain_to_boundary_grid(sd) for sd in subdomains]
        return [bg for bg in boundary_grids if bg is not None]

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
        # NOTE: The enforcement of keyword-only arguments, combined with this class
        # being used as a mixin with other classes (thus this function represented as a
        # Callable in the other classes) does not make mypy happy. The problem seems to
        # be that a method specified as Callable must be called exactly as the type
        # specification, thus when this method is called with arguments
        #
        #   self.wrap_grid_attribute(..., dim=some_integer, ...)
        #
        # mypy will react on the difference between the type specification (that did not
        # include the dim argument) and the actual call. We also tried adding the *
        # (indicating the start of keyword-only in the type specification), but while
        # this made mypy happy, it is not vald syntax. The only viable solution (save
        # from using typing protocols, which we really do not want to do, there are
        # enough classes and inheritance in the mixin combination as it is) seems to be
        # to add a # type: ignore[call-arg] comment where the method is called. By only
        # ignoring call-args problems, we limit the risk of silencing other errors that
        # mypy might find.

        if len(grids) > 0:
            # Check that all grids have the sought after attribute. We could have
            # avoided this loop by surrounding the getattr with a try-except, but this
            # would have given a more convoluted code.
            if not all(hasattr(g, attr) for g in grids):
                raise ValueError(f"Grids do not have attribute {attr}")
            # Check that the attribute is a ndarray
            if not all(isinstance(getattr(g, attr), np.ndarray) for g in grids):
                raise ValueError(f"Attribute {attr} is not a ndarray")

            # NOTE: We do not rule out the combination of subdomains and interfaces
            # in the same list. There should be no chance of errors here, and although
            # such a case seems to EK at the moment to be a bit of an edge case, there
            # is no reason to rule it out.

            if dim is None:
                # Default to all dimensions
                vals = np.hstack([getattr(g, attr).ravel("F") for g in grids])
            else:
                # Only include the first dim dimensions
                # We need to force the array to be 2d here, in case the dimension
                # argument is given for a non-vector attribute like cell_volumes.
                vals = np.hstack(
                    [np.atleast_2d(getattr(g, attr))[:dim].ravel("F") for g in grids]
                )
        else:
            # For an empty list of grids, return an empty matrix
            vals = np.zeros(0)

        array = pp.ad.DenseArray(vals)
        array.set_name(f"Array wrapping attribute {attr} on {len(grids)} grids.")
        return array

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
        # NOTE: See self.wrap_grid_attribute for comments on typing when this method
        # is used as a mixin, and the need to add type-ignore[call-arg] on use of this
        # method.

        # Collect the basis functions for each dimension
        basis: list[pp.ad.SparseArray] = []
        for i in range(dim):
            basis.append(self.e_i(grids, i=i, dim=dim))
        # Stack the basis functions horizontally
        return basis

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
        # NOTE: See self.wrap_grid_attribute for comments on typing when this method
        # is used as a mixin, and the need to add type-ignore[call-arg] on use of this
        # method.

        # TODO: Should we expand this to grids not aligned with the coordinate axes, and
        # possibly unify with ``porepy.utils.projections.TangentialNormalProjection``?
        # This is not a priority for the moment, though.

        if dim is None:
            dim = self.nd

        # Sanity checks
        if i >= dim:
            raise ValueError("Basis function index out of range")

        # Construct a single vector, and later stack it to a matrix
        # Collect the basis functions for each dimension
        e_i = np.zeros((dim, 1))
        e_i[i] = 1
        # Expand to cell-wise column vectors.
        num_cells = sum([g.num_cells for g in grids])
        # Expand to a matrix.
        mat = sps.kron(sps.eye(num_cells), e_i)
        return pp.ad.SparseArray(mat)

    # Local basis related methods
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
        # We first need an inner product (or dot product), i.e. extract the tangential
        # component of the cell-wise vector v to be transformed. Then we want to express
        # it in the tangential basis. The two operations are combined in a single
        # operator composed right to left: v will be hit by first e_i.T (row vector) and
        # secondly t_i (column vector). Ignore mypy keyword argument error.
        op: pp.ad.Operator = pp.ad.sum_operator_list(
            [
                self.e_i(subdomains, i=i, dim=self.nd - 1)  # type: ignore[arg-type]
                @ self.e_i(subdomains, i=i, dim=self.nd).T
                for i in range(self.nd - 1)
            ]
        )
        op.set_name("tangential_component")
        return op

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
        # Create the basis function for the normal component (which is known to be the
        # last component).
        e_n = self.e_i(subdomains, i=self.nd - 1, dim=self.nd)
        e_n.set_name("normal_component")
        return e_n.T

    def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
        """Ad wrapper around tangential_normal_projections for fractures.

        Parameters:
            subdomains: List of subdomains for which to compute the local coordinates.

        Returns:
            Local coordinates as a pp.ad.SparseArray.

        """
        # TODO: If we ever implement a mapping to reference space for all subdomains,
        # the present method should be revisited.

        # For now, assert all subdomains are fractures, i.e. dim == nd - 1.
        # TODO: Extend to all subdomains, not only codimension 1?
        assert all([sd.dim == self.nd - 1 for sd in subdomains])
        if len(subdomains) > 0:
            # Compute the local coordinates for each subdomain. For this, we use the
            # preset tangential_normal_projection attribute of the subdomains.
            local_coord_proj_list = [
                self.mdg.subdomain_data(sd)[
                    "tangential_normal_projection"
                ].project_tangential_normal(sd.num_cells)
                for sd in subdomains
            ]
            local_coord_proj = sps.block_diag(local_coord_proj_list)
        else:
            # Also treat no subdomains
            local_coord_proj = sps.csr_matrix((0, 0))
        return pp.ad.SparseArray(local_coord_proj)

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
        name = f"_subdomain_proj_of_dimension_{dim}"
        if hasattr(self, name):
            proj = getattr(self, name)
        else:
            proj = pp.ad.SubdomainProjections(self.mdg.subdomains(), dim)
            setattr(self, name, proj)
        return proj

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
        if isinstance(domain, pp.Grid):
            # bc_type_* methods ... require working with subdomains

            face_centers = domain.face_centers
            num_faces = domain.num_faces
            all_bf = domain.get_boundary_faces()
        elif isinstance(domain, pp.BoundaryGrid):
            # Cells of the boundary grid are faces of the parent subdomain.
            face_centers = domain.cell_centers
            num_faces = domain.num_cells
            all_bf = np.arange(num_faces)
        else:
            raise ValueError(
                "Domain must be either Grid or BoundaryGrid. Provided:", domain
            )

        # Get domain boundary sides
        box = copy.deepcopy(self.domain.bounding_box)

        east = np.abs(box["xmax"] - face_centers[0]) <= tol
        west = np.abs(box["xmin"] - face_centers[0]) <= tol
        if self.mdg.dim_max() == 1:
            north = np.zeros(num_faces, dtype=bool)
            south = north.copy()
        else:
            north = np.abs(box["ymax"] - face_centers[1]) <= tol
            south = np.abs(box["ymin"] - face_centers[1]) <= tol
        if self.mdg.dim_max() < 3:
            top = np.zeros(num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = np.abs(box["zmax"] - face_centers[2]) <= tol
            bottom = np.abs(box["zmin"] - face_centers[2]) <= tol

        # Create a namedtuple to store the arrays
        domain_sides = pp.domain.DomainSides(
            all_bf, east, west, north, south, top, bottom
        )

        return domain_sides

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
        if len(subdomains) == 0:
            # Special case if no interfaces.
            sign_flipper = pp.ad.SparseArray(sps.csr_matrix((0, 0)))
        else:
            # There is already a method to construct a switcher matrix in grid_utils,
            # so we use that. Loop over all subdomains, construct a local switcher
            # matrix and store it. The mixed-dimensional version can then be constructed
            # by block diagonal concatenation.
            # NOTE: While it is somewhat unfortunate to have the method in grid_utils,
            # since this leads to a more nested code, it also means we can use the
            # method outside the Ad framework. For the time being, we keep it.
            matrices = []
            for sd in subdomains:
                # Use the tagging of fracture surfaces to identify the faces on internal
                # boundaries.
                faces_on_fracture_surface = np.where(sd.tags["fracture_faces"])[0]
                switcher_int = pp.grid_utils.switch_sign_if_inwards_normal(
                    sd, dim, faces_on_fracture_surface
                )
                matrices.append(switcher_int)

            # Construct the block diagonal matrix.
            sign_flipper = pp.ad.SparseArray(sps.block_diag(matrices).tocsr())
        sign_flipper.set_name("Flip_normal_vectors")
        return sign_flipper

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
        # NOTE: See self.wrap_grid_attribute for comments on typing when this method
        # is used as a mixin, and the need to add type-ignore[call-arg] on use of this
        # method.

        if len(interfaces) == 0:
            # Special case if no interfaces.
            return pp.ad.DenseArray(np.zeros(0))

        # Main ingredients: Normal vectors for primary subdomains for each interface,
        # and a switcher matrix to flip the sign if the normal vector points inwards.
        # The first is constructed herein, the second is a method of this class.

        # Since the normal vectors are stored on the primary subdomains, but are to be
        # computed on the interfaces, we need mortar projections.

        # Get hold of the primary subdomains, i.e. the higher-dimensional neighbors of
        # the interfaces.
        primary_subdomains: list[pp.Grid] = []
        for intf in interfaces:
            primary_subdomains.append(self.mdg.interface_to_subdomain_pair(intf)[0])

        # Projection operator between the subdomains and interfaces. The projection is
        # constructed to only consider the higher-dimensional subdomains.
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, primary_subdomains, interfaces, dim=self.nd
        )
        # Ignore mypy complaint about unexpected keyword arguments.
        primary_face_normals = self.wrap_grid_attribute(
            primary_subdomains, "face_normals", dim=self.nd  # type: ignore[call-arg]
        )
        # Account for sign of boundary face normals. This will give a matrix with a
        # shape equal to the total number of faces in all primary subdomains.
        # Ignore mypy complaint about unexpected keyword arguments.
        flip = self.internal_boundary_normal_to_outwards(
            primary_subdomains, dim=self.nd  # type: ignore[call-arg]
        )
        # Flip the normal vectors. Unravelled from the right: Restrict from faces on all
        # subdomains to the primary ones, multiply with the face normals, flip the
        # signs, and project back up to all subdomains.
        flipped_normals = flip @ primary_face_normals
        # Project to mortar grid, as a mapping from mortar to the subdomains and back
        # again. If we are to use cell_volumes from interfaces to normalize, projection
        # must logically be integration, not average. This also means that the normals
        # have length equal to cell_volume on mortar grids, by analogy to face_area for
        # subdomains.
        outwards_normals = mortar_projection.primary_to_mortar_int @ flipped_normals
        outwards_normals.set_name("outwards_internal_boundary_normals")

        # Normalize by face area if requested.
        if unitary:
            # 1 over cell volumes on the interfaces
            # Ignore mypy complaint about unexpected keyword arguments.
            cell_volumes_inv = pp.ad.Scalar(1) / self.wrap_grid_attribute(
                interfaces, "cell_volumes", dim=self.nd  # type: ignore[call-arg]
            )

            # Expand cell volumes to nd by multiplying from left by e_i and summing
            # over all dimensions.
            cell_volumes_inv_nd = pp.ad.sum_operator_list(
                [e @ cell_volumes_inv for e in self.basis(interfaces, self.nd)]
            )
            # Scale normals.
            outwards_normals = cell_volumes_inv_nd * outwards_normals
            outwards_normals.set_name("unitary_outwards_internal_boundary_normals")

        return outwards_normals
