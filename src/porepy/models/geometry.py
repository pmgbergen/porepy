"""Geometry definition for simulation model."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Literal, Optional, Sequence, Union, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from dataclasses import dataclass,field
import math
from porepy.fracs.wells_3d import _add_interface

logger = logging.getLogger(__name__)


class ModelGeometry(pp.PorePyModel):
    """This class provides geometry related methods and information for a simulation
    model."""

    _domain: pp.Domain
    _fractures: list

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid.

        The default values provided in set_domain, set_fractures, grid_type and
        meshing_arguments produce a 2d unit square domain with no fractures and a
        four Cartesian cells.

        """
        # Create the geometry through domain amd fracture set.
        self.set_domain()
        self.set_fractures()
        # Create a fracture network and a mixed-dimensional grid.
        self.create_fracture_network()
        self.create_mdg()

        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        # Set up well network and add wells to the mixed-dimensional grid.
        self.set_well_network()
        self.add_wells_to_mdg()

    @property
    def domain(self) -> pp.Domain:
        """Domain of the problem."""
        return self._domain

    def set_domain(self) -> None:
        """Set domain of the problem.

        Defaults to a 2d unit square domain.
        Override this method to define a geometry with a different domain.

        """
        self._domain = nd_cube_domain(2, self.units.convert_units(1.0, "m"))

    @property
    def fractures(self) -> Union[list[pp.LineFracture], list[pp.PlaneFracture]]:
        """Fractures of the problem."""
        return self._fractures

    def set_fractures(self) -> None:
        """Set fractures in the fracture network.

        Override this method to define a geometry with fractures.

        """
        self._fractures = []

    def create_fracture_network(self) -> None:
        """Set the fracture network from the fractures and domain."""
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

    def create_mdg(self) -> None:
        """Set the mixed-dimensional grid from the domain, fracture network and meshing
        arguments.
        """
        self.mdg = pp.create_mdg(
            self.grid_type(),
            self.meshing_arguments(),
            self.fracture_network,
            **self.meshing_kwargs(),
        )

    def set_well_network(self) -> None:
        """Assign well network class."""
        self.well_network = pp.WellNetwork3d(domain=self._domain)

    def add_wells_to_mdg(self) -> None:
        """Add wells to the mixed-dimensional grid."""
        if len(self.well_network.wells) > 0:
            # Compute intersections.
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)

    def is_well_grid(self, grid: pp.Grid | pp.MortarGrid) -> bool:
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
        cell_size = self.units.convert_units(0.5, "m")
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

    def depth(self, points: np.ndarray) -> np.ndarray:
        """Compute depth of points.

        Parameters:
            points: Array of points where depth is to be calculated. The nd-1 coordinate
                is assumed to be the depth coordinate, with larger values indicating
                larger depth. Shape: (N, num_points), with N >= nd.

        Returns:
            Depth values for the provided points.

        """
        top_depth = self.params.get("stratigraphy", {}).get("top_depth", 0.0)
        key = "zmax" if self.nd == 3 else "ymax"
        return top_depth + self.domain.bounding_box[key] - points[self.nd - 1, :]

    @pp.ad.cached_method
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
        # Initialize list of interfaces, build it up one subdomain at a time.
        interfaces: list[pp.MortarGrid] = []
        for sd in subdomains:
            for intf in self.mdg.subdomain_to_interfaces(sd):
                if intf not in interfaces and intf.codim in codims:
                    interfaces.append(intf)
        return self.mdg.sort_interfaces(interfaces)

    @pp.ad.cached_method
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

        This is a 1-1 mapping between subdomains and their boundary grids. No
        sorting is performed.

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

    def basis(self, grids: Sequence[pp.GridLike], dim: int) -> list[pp.ad.Projection]:
        """Return a cell-wise basis for all subdomains.

        The basis is represented as a list of projections, each of which represents a
        basis function. The individiual basis functions can be represented as a
        projection matrix, of shape ``Nc * dim, Nc`` where ``Nc`` is the total number of
        cells in the subdomains.

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
            List of pp.ad.SparseArray, each of which represents a basis
            function.

        """
        # Collect the basis functions for each dimension.
        basis: list[pp.ad.Projection] = []
        for i in range(dim):
            basis.append(self.e_i(grids, i=i, dim=dim))
        # Stack the basis functions horizontally.
        return basis

    @pp.ad.cached_method
    def e_i(
        self, grids: Sequence[pp.GridLike], *, i: int, dim: int
    ) -> pp.ad.Projection:
        """Return a cell-wise basis function in a specified dimension.

        It is assumed that the grids are embedded in a space of dimension dim and
        aligned with the coordinate axes, that is, the reference space of the grid.
        Moreover, the grid is assumed to be planar.

        Example:
            For a grid with two cells, and with `i=1` and `dim=3`, the returned basis
            will be a Projection that is equivalent to applying the following projection
            matrix:
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
            Ad projection that represents a basis function.

        Raises:
            ValueError: If i is larger than dim - 1.

        """
        if dim is None:
            dim = self.nd

        # Sanity checks
        if i >= dim:
            raise ValueError("Basis function index out of range")

        # Expand to cell-wise column vectors.
        num_cells = sum([g.num_cells for g in grids])
        range_ind = np.arange(i, dim * num_cells, dim)

        slicer = pp.ad.Projection(
            domain_indices=np.arange(num_cells),
            range_indices=range_ind,
            range_size=num_cells * dim,
            domain_size=num_cells,
        )

        return slicer

    # Local basis related methods
    @pp.ad.cached_method
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
        # We first need an inner product (or dot product), i.e. extract the tangential
        # component of the cell-wise vector v to be transformed. Then we want to express
        # it in the tangential basis. The two operations are combined in a single
        # operator composed right to left: v will be hit by first e_i.T (row vector) and
        # secondly t_i (column vector).
        op: pp.ad.Operator = pp.ad.sum_projection_list(
            [
                self.e_i(subdomains, i=i, dim=self.nd - 1)
                @ self.e_i(subdomains, i=i, dim=self.nd).T
                for i in range(self.nd - 1)
            ]
        )
        op.set_name("tangential_component")
        return op

    @pp.ad.cached_method
    def normal_component(self, subdomains: list[pp.Grid]) -> pp.ad.Projection:
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
            Projection extracting normal component of the vector field and expressing it
            in normal basis. The size of the projection is `(Nc, Nc * self.nd)`, where
            `Nc` is the total number of cells in the subdomains.

        """
        # Create the basis function for the normal component (which is known to be the
        # last component).
        e_n = self.e_i(subdomains, i=self.nd - 1, dim=self.nd)
        e_n.set_name("normal_component")
        return e_n.T

    def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.SparseArray:
        """Ad wrapper around tangential_normal_projections for fractures.

        The method constructs a projection from global to local coordinates for a list
        of subdomains. The local coordinates are defined by the tangential and normal
        directions of the subdomains, as defined by their tangential_normal_projection
        attribute.

        The inverse of this projection can be used to map quantities from local to
        global coordinates. It can be constructed by transposing the projection returned
        by this method.

        Parameters:
            subdomains: List of subdomains for which to compute the local coordinates.

        Returns:
            Projection from global to local coordinates as a pp.ad.SparseArray.

        """
        # For now, assert all subdomains are fractures, i.e. dim == nd - 1.
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
            local_coord_proj = pp.matrix_operations.csc_matrix_from_sparse_blocks(
                local_coord_proj_list
            )
        else:
            # Also treat no subdomains.
            local_coord_proj = sps.csr_matrix((0, 0))
        return pp.ad.SparseArray(local_coord_proj)

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
                domain_sides = model.domain_boundary_sides(sd)
                # Access north faces using index or name is equivalent:
                north_by_index = domain_sides[3]
                north_by_name = domain_sides.north
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
            sign_flipper = pp.ad.SparseArray(
                pp.matrix_operations.sparse_dia_from_sparse_blocks(matrices)
            )
        sign_flipper.set_name("Flip_normal_vectors")
        return sign_flipper

    @pp.ad.cached_method
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
        primary_face_normals = self.wrap_grid_attribute(
            primary_subdomains, "face_normals", dim=self.nd
        )
        # Account for sign of boundary face normals. This will give a matrix with a
        # shape equal to the total number of faces in all primary subdomains.
        flip = self.internal_boundary_normal_to_outwards(
            primary_subdomains, dim=self.nd
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
        outwards_normals = mortar_projection.primary_to_mortar_int() @ flipped_normals
        outwards_normals.set_name("outwards_internal_boundary_normals")

        # Normalize by face area if requested.
        if unitary:
            # 1 over cell volumes on the interfaces
            cell_volumes_inv = pp.ad.Scalar(1) / self.wrap_grid_attribute(
                interfaces, "cell_volumes", dim=self.nd
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

    @pp.ad.cached_method
    def subdomains_to_well_interfaces(
        self,
        subdomains: list[pp.Grid],
        well_types: tuple[str, ...] = ("injection_well", "production_well"),
    ) -> list[pp.MortarGrid]:
        """Return interfaces connected to tagged well subdomains.

        This helper is intended for models where wells are represented by tagged
        subdomains, possibly with non-standard geometric codimension such as 3D-0D.
        Unlike ``subdomains_to_interfaces(..., codims=[...])``, this function does
        not filter by codimension, but instead identifies wells explicitly from
        subdomain tags.

        Parameters:
            subdomains: Subdomains for which to find neighbouring well interfaces.
            well_types: Tags identifying well subdomains.

        Returns:
            Unique list of all interfaces neighbouring tagged well subdomains.
            Interfaces are sorted according to their index in the mixed-dimensional grid.
        """
        interfaces: list[pp.MortarGrid] = []

        for sd in subdomains:
            if not any(tag in sd.tags for tag in well_types):
                continue

            for intf in self.mdg.subdomain_to_interfaces(sd):
                if intf not in interfaces:
                    interfaces.append(intf)

        return self.mdg.sort_interfaces(interfaces)





class LoadGeometryMixin(pp.PorePyModel):
    """Provide functionality to store and load the full model geometry from files.

    The intended workflow is as follows:

    Example:
        class NewModelClass(LoadGeometryMixin, YourBaseModelClass):
            ...

        model = NewModelClass(params)
        # Mesh and save msh and fracture network files once.
        model.create_and_export_geometry()

        # Geometry is loaded from msh and fracture network files in subsequent runs.
        for i in range(num_runs):
            pp.run_time_dependent_model(model)

    """

    def set_geometry(self) -> None:
        """Load and set model geometry from ``msh``, ``geo``, and ``csv`` files that
        contain a mesh and information about fractures.

        The following attributes are set after running this method:
        - ``self.mdg: pp.MixedDimensionalGrid``
        - ``self._domain: pp.Domain``
        - ``self.fracture_network: pp.FractureNetwork2D`` (or 3D) if a ``csv`` file is
            provided. Otherwise an empty network.

        Warning:
            Elliptic fracture networks are not yet supported.

        """
        # Paths to geometry files. The structure is how `self.fracture_network.mesh` and
        # `self.create_and_export_geometry` create them.
        file_name = Path(self.meshing_kwargs()["file_name"])
        folder_path = file_name.parent
        msh_path = (folder_path / file_name.stem).with_suffix(".msh")
        geo_path = (folder_path / file_name.stem).with_suffix(".geo_unrolled")
        fracture_network_path = folder_path / self.meshing_kwargs()["csv_file_name"]

        # Check whether the msh or geo file exists. If used as in the docstring example,
        # both exist and the msh file is used to avoid remeshing unnecessarily.
        if not msh_path.exists():
            if not geo_path.exists():
                raise ValueError(
                    "Either 'folder_path` / `file_name.msh` or 'folder_path` /"
                    + " `file_name.geo_unrolled` must exist."
                )
            else:
                logger.info(f"msh file not found, remeshing from {geo_path}.")
                gmsh_path = geo_path
        else:
            logger.info(
                f"msh file found, loading mixed-dimensional grid from {msh_path}."
            )
            if geo_path.exists():
                logger.info(f"Both msh and geo files found. Ignoring {geo_path}.")
            gmsh_path = msh_path

        # Set file permissions. This turned out to be important for GH actions.
        msh_path.chmod(777)
        fracture_network_path.chmod(777)

        # Load mixed-dimensional grid from geo or msh file.
        self.fracture_network = pp.fracture_importer.network_from_csv(
            fracture_network_path
        )
        # TODO (part of GH 1576): Replace with fracture_network.dim, which is introduced
        # in that PR.
        self.nd = (
            2
            if isinstance(
                self.fracture_network, pp.fracs.fracture_network_2d.FractureNetwork2d
            )
            else 3
        )

        self.mdg = pp.fracture_importer.dfm_from_gmsh(gmsh_path, dim=self.nd)

        # Obtain domain and fracture list directly from the fracture network.
        self._domain = cast(pp.Domain, self.fracture_network.domain)
        self._fractures = self.fracture_network.fractures

        # Create projection between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        # Create well network and mesh.
        self.set_well_network()
        self.add_wells_to_mdg()

    def create_and_export_geometry(self, set_geometry_class=ModelGeometry) -> None:
        """Export mesh and fracture network to ``msh``, ``geo``, and ``csv`` files.

        Parameters:
            folder_path: Path to folder where files are to be stored.
            set_geometry_class: Class whose ``set_geometry`` method is to be used for
                meshing and storing the ``msh`` and ``geo`` files. To be overridden,
                e.g., if (parts of) the geometry is (are) loaded from a gmsh file
                instead of created within PorePy. Default is
                :class:`~porepy.models.geometry.ModelGeometry`.

        """
        # Explicitely call the ``set_geometry`` method of the provided class. msh and
        # geo files are saved after meshing, since
        # ``self.meshing_kwargs["write_geo"]`` is True and
        # ``self.meshing_kwargs["clear_gmsh"]`` is False.
        set_geometry_class.set_geometry(self)  # type: ignore[attr-defined]

        # In addition, save the fracture network.
        folder_path = Path(self.meshing_kwargs()["file_name"]).parent
        csv_file_name = self.meshing_kwargs()["csv_file_name"]
        fracture_network_path = folder_path / csv_file_name
        self.fracture_network.to_csv(fracture_network_path)

    def meshing_kwargs(self) -> dict:
        """Provide default meshing kwargs for storing and loading `mdg` and the fracture
        network.

        The following keyword arguments are added if not already provided by
        :meth:`~porepy.models.geometry.ModelGeometry.meshing_kwargs`:
        - ``file_name: str``: Name of the mesh and geo file (without extension). Default
            is "gmsh_frac_file.msh".
        TODO PvS: Consider including a default folder (same as default folder for
        exporting).
        - ``write_geo: bool``: Whether to keep the geo file after meshing. Default is
            True.
        - ``clear_gmsh: bool``: Whether to remove the msh file after meshing. Default is
            False.
        - ``csv_file_name: str``: Name of the fracture network csv file. Default is
            "fracture_network.csv".
        Returns:
            Keyword arguments compatible with :meth:`~porepy.create_mdg()`.

        """
        # Add kwargs related to storing the geometry files to the meshing kwargs of
        # ``ModelGeometry``.
        default_meshing_kwargs = {
            # The first three kwargs are passed to ``fracture_network.mesh``.
            "file_name": "gmsh_frac_file.msh",  # Name of the output mesh and geo files.
            "write_geo": True,  # Keep the geo file after meshing.
            "clear_gmsh": False,  # Keep the msh file after meshing.
            # The latter two are used by `LoadGeometryMixin.set_geometry`.
            "csv_file_name": "fracture_network.csv",
        }
        meshing_kwargs = super().meshing_kwargs()  # type: ignore[safe-super]
        default_meshing_kwargs.update(meshing_kwargs)

        return default_meshing_kwargs




@dataclass(frozen=True)
class LayerDepthSpec:
    """User-facing geological layer definition.

    Depth is positive downward.
    Only the bottom depth is specified for each layer.
    """
    name: str
    bottom_depth: float


@dataclass(frozen=True)
class ResolvedLayer:
    """Layer with explicit top and bottom depths."""

    name: str
    top_depth: float
    bottom_depth: float
    porosity: float
    permeability: float

    @property
    def thickness(self) -> float:
        return self.bottom_depth - self.top_depth
    
@dataclass(frozen=True)
class ZLayer:

    # Vertical coordinate convention:
    # z = deepest_bottom_depth - depth
    # Hence z = 0 at the bottom of the stratigraphy, and z increases upward.
    name: str
    z_top: float
    z_bottom: float

    @property
    def thickness(self) -> float:
        return self.z_top - self.z_bottom
    




@dataclass(frozen=True)
class GlobalCellSize:
    dx: float
    dy: float
    dz: float

    def __post_init__(self) -> None:
        for name, value in (("dx", self.dx), ("dy", self.dy), ("dz", self.dz)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")



@dataclass(frozen=True)
class FineRectangle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    dx: float
    dy: float

    def __post_init__(self) -> None:
        if self.xmax <= self.xmin:
            raise ValueError("xmax must be greater than xmin.")
        if self.ymax <= self.ymin:
            raise ValueError("ymax must be greater than ymin.")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("Rectangle dx and dy must be positive.")



@dataclass(frozen=True)
class MeshingSpec:
    global_cell_size: GlobalCellSize
    fine_rectangles: tuple[FineRectangle, ...] = ()
    layer_dz: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for layer_name, dz in self.layer_dz.items():
            if dz <= 0:
                raise ValueError(
                    f"dz for layer '{layer_name}' must be positive, got {dz}."
                )

    def dz_for_layer(self, layer_name: str) -> float:
        return self.layer_dz.get(layer_name, self.global_cell_size.dz)



@dataclass(frozen=True)
class AxisRefinement:
    """Local refinement interval on one axis."""

    start: float
    end: float
    d: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(
                f"Expected end > start, got start={self.start}, end={self.end}."
            )
        if self.d <= 0:
            raise ValueError(f"Expected positive spacing, got d={self.d}.")






def layers_to_negative_z(
    resolved_layers: tuple[ResolvedLayer, ...],
) -> tuple[ZLayer, ...]:
    if not resolved_layers:
        raise ValueError("No resolved layers provided.")

    deepest_bottom = resolved_layers[-1].bottom_depth

    z_layers: list[ZLayer] = []
    for layer in resolved_layers:
        z_top = -layer.top_depth
        z_bottom = -layer.bottom_depth

        z_layers.append(
            ZLayer(
                name=layer.name,
                z_top=z_top,
                z_bottom=z_bottom,
            )
        )

    return tuple(z_layers)



def make_interval_points(start: float, end: float, d: float) -> np.ndarray:
    """Create coordinates for one interval, including both endpoints.

    Parameters:
        start: Interval start.
        end: Interval end.
        d: Target spacing.

    Returns:
        1D array of coordinates from start to end, inclusive.
    """
    if end <= start:
        raise ValueError(f"Expected end > start, got start={start}, end={end}.")
    if d <= 0:
        raise ValueError(f"Expected positive spacing, got d={d}.")

    n = math.ceil((end - start) / d)
    return np.linspace(start, end, n + 1)


def make_z_coordinates(
    z_layers: tuple[ZLayer, ...],
    refinement: MeshingSpec,
) -> np.ndarray:
    """Build full vertical TensorGrid coordinates from all layers.

    Layers are meshed one by one and then concatenated.
    Shared interface points are included only once.
    """
    if not z_layers:
        raise ValueError("No z-layers provided.")

    z_arrays: list[np.ndarray] = []

    for i, layer in enumerate(reversed(z_layers)):
        # reversed(z_layers): bottom layer first, top layer last
        dz = refinement.dz_for_layer(layer.name)

        layer_points = make_interval_points(
            start=layer.z_bottom,
            end=layer.z_top,
            d=dz,
        )

        if i > 0:
            # Drop first point to avoid duplicating the shared interface
            layer_points = layer_points[1:]

        z_arrays.append(layer_points)

    z_pts = np.concatenate(z_arrays)

    return z_pts


def make_axis_coordinates(
    start: float,
    end: float,
    default_d: float,
    refinements: tuple[AxisRefinement, ...] = (),
) -> np.ndarray:
    """Build coordinates on one axis with optional local refinement."""
    if end <= start:
        raise ValueError(f"Expected end > start, got start={start}, end={end}.")
    if default_d <= 0:
        raise ValueError(f"Expected positive default_d, got {default_d}.")

    clipped_refinements: list[AxisRefinement] = []
    for ref in refinements:
        clipped_start = max(start, ref.start)
        clipped_end = min(end, ref.end)

        if clipped_end <= clipped_start:
            continue

        clipped_refinements.append(
            AxisRefinement(
                start=clipped_start,
                end=clipped_end,
                d=ref.d,
            )
        )

    breakpoints = {start, end}
    for ref in clipped_refinements:
        breakpoints.add(ref.start)
        breakpoints.add(ref.end)

    sorted_points = sorted(breakpoints)

    segments: list[np.ndarray] = []

    for i in range(len(sorted_points) - 1):
        seg_start = sorted_points[i]
        seg_end = sorted_points[i + 1]

        segment_d = default_d
        covering_ds = [
            ref.d
            for ref in clipped_refinements
            if ref.start <= seg_start and seg_end <= ref.end
        ]
        if covering_ds:
            segment_d = min(covering_ds)

        seg_points = make_interval_points(
            start=seg_start,
            end=seg_end,
            d=segment_d,
        )

        if i > 0:
            seg_points = seg_points[1:]

        segments.append(seg_points)

    return np.concatenate(segments)


def x_refinements_from_rectangles(
    fine_rectangles: tuple[FineRectangle, ...],
) -> tuple[AxisRefinement, ...]:
    """Project rectangular refinements onto the x-axis."""
    return tuple(
        AxisRefinement(
            start=rect.xmin,
            end=rect.xmax,
            d=rect.dx,
        )
        for rect in fine_rectangles
    )

def y_refinements_from_rectangles(
    fine_rectangles: tuple[FineRectangle, ...],
) -> tuple[AxisRefinement, ...]:
    """Project rectangular refinements onto the y-axis."""
    return tuple(
        AxisRefinement(
            start=rect.ymin,
            end=rect.ymax,
            d=rect.dy,
        )
        for rect in fine_rectangles
    )

def make_x_coordinates(box: dict[str, float], meshing: MeshingSpec) -> np.ndarray:
    """Build x-coordinates from meshing settings."""
    return make_axis_coordinates(
        start=box["xmin"],
        end=box["xmax"],
        default_d=meshing.global_cell_size.dx,
        refinements=x_refinements_from_rectangles(meshing.fine_rectangles),
    )

def make_y_coordinates(box: dict[str, float], meshing: MeshingSpec) -> np.ndarray:
    """Build y-coordinates from meshing settings."""
    return make_axis_coordinates(
        start=box["ymin"],
        end=box["ymax"],
        default_d=meshing.global_cell_size.dy,
        refinements=y_refinements_from_rectangles(meshing.fine_rectangles),
    )




class ReservoirGeometry(pp.PorePyModel):

    def build_meshing_spec(self) -> MeshingSpec:
        """Parse self.params into internal MeshingSpec."""

        meshing_config = self.params.get("meshing_config", {})
        meshing_arguments = self.params.get("meshing_arguments", {})

        cell_size = meshing_config.get("cell_size")
        if cell_size is None:
            cell_size = meshing_arguments.get("cell_size")

        # ---- global sizes ----
        dx = meshing_config.get("cell_size_x", cell_size)
        dy = meshing_config.get("cell_size_y", cell_size)
        dz = meshing_config.get("cell_size_z", cell_size)

        if dx is None or dy is None or dz is None:
            raise ValueError("Missing global cell size (cell_size or cell_size_x/y/z).")

        global_cell_size = GlobalCellSize(dx=dx, dy=dy, dz=dz)

        # ---- layer dz ----
        layer_dz = meshing_config.get("layer_cell_size_z", {})

        # ---- rectangles ----
        rectangles_input = meshing_config.get("fine_rectangles", [])

        fine_rectangles = tuple(
            FineRectangle(
                xmin=rect["xmin"],
                xmax=rect["xmax"],
                ymin=rect["ymin"],
                ymax=rect["ymax"],
                dx=rect.get("cell_size_x", dx),
                dy=rect.get("cell_size_y", dy),
            )
            for rect in rectangles_input
        )

        return MeshingSpec(
            global_cell_size=global_cell_size,
            fine_rectangles=fine_rectangles,
            layer_dz=layer_dz,
        )



    def resolve_layers(self) -> tuple[ResolvedLayer, ...]:
        """Resolve layer definitions from self.params."""

        stratigraphy = self.params.get("stratigraphy")
        if stratigraphy is None:
            #generate default single layer if no stratigraphy provided
            top_depth = self.params.get("stratigraphy", {}).get("top_depth", 0.0)
            a, b, c = self.params.get("domain_size", (10, 10, 10))  # [m]
            bottom_depth = top_depth + c
            return     (
    ResolvedLayer(
        name="default_layer",
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        porosity=self.solid.porosity,
        permeability=self.solid.permeability,
    ),
)





        layers = stratigraphy.get("layers", [])
        if not layers:
            raise ValueError("No layers provided.")

        if "top_depth" not in stratigraphy:
            raise ValueError("Missing 'top_depth' in stratigraphy.")

        current_top = stratigraphy["top_depth"]

        resolved: list[ResolvedLayer] = []
        seen_names: set[str] = set()

        for layer in layers:
            name = layer["name"]
            bottom_depth = layer["bottom_depth"]

            if name in seen_names:
                raise ValueError(f"Duplicate layer name: {name}")
            seen_names.add(name)

            if bottom_depth <= current_top:
                raise ValueError(
                    f"Layer '{name}' has bottom_depth={bottom_depth}, "
                    f"which must be deeper than top_depth={current_top}."
                )

            porosity = layer.get("porosity")
            permeability = layer.get("permeability")
            resolved.append(
                ResolvedLayer(
                    name=name,
                    top_depth=current_top,
                    bottom_depth=bottom_depth,
                    porosity=porosity,
                    permeability=permeability,
                )
            )

            current_top = bottom_depth

        return tuple(resolved)




    def set_domain(self) -> None:

        ls = self.units.convert_units(1, "m")  # length scaling
        a, b, c = self.params.get("domain_size", (10, 10, 10))  # [m]
        stratigraphy = self.params.get("stratigraphy", {})
        top_depth = stratigraphy.get("top_depth", 0.0)
        bottom_depth = top_depth + c



        domain = pp.Domain({"xmin": 0.0, "xmax": a * ls, "ymin": 0.0, "ymax": b * ls, "zmin": -bottom_depth, "zmax": -top_depth})
        self._domain = domain



    def meshing_arguments(self) -> dict:

        meshing_spec = self.build_meshing_spec()

        layers= self.resolve_layers()

        # ---- consistency check: layer thickness vs domain thickness ----
        if layers:
            total_layer_thickness = layers[-1].bottom_depth - layers[0].top_depth

            key_max = "zmax"
            key_min = "zmin"

            domain_thickness = (
                self.domain.bounding_box[key_max]
                - self.domain.bounding_box[key_min]
            )

            assert np.isclose(
                total_layer_thickness, domain_thickness
            ), (
                f"Layer thickness ({total_layer_thickness}) does not match "
                f"domain thickness ({domain_thickness})."
            )



        z_layers = layers_to_negative_z(layers)

        z_pts = make_z_coordinates(z_layers, meshing_spec)

        box = self.domain.bounding_box
        x_pts = make_x_coordinates(box, meshing_spec)
        y_pts = make_y_coordinates(box, meshing_spec)

        ls = self.units.convert_units(1, "m")  # length scaling

        meshing_config = self.params.get("meshing_config", {})
        meshing_arguments = self.params.get("meshing_arguments", {})

        cell_size = meshing_config.get("cell_size")
        if cell_size is None:
            cell_size = meshing_arguments.get("cell_size")



        mesh_sizes = {
            # Cartesian: 2 by 8 cells.
            "x_pts": x_pts*ls,
            "y_pts": y_pts*ls,
            "z_pts": z_pts*ls,
        }
        if cell_size is not None:
            mesh_sizes["cell_size"] = cell_size*ls

        print("Mesh sizes:", mesh_sizes)
        return mesh_sizes

    def grid_type(self) -> str:
        """Grid type for the mixed-dimensional grid.

        Returns:
            Grid type for the mixed-dimensional grid.

        """
        return self.params.get("grid_type", "tensor_grid")



class PointWellGeometry(pp.PorePyModel):

    def set_geometry(self):
        super().set_geometry()

        def add_well_if_given(param_name, well_name):
            coordinates = self.params.get(param_name)

            if coordinates is None:
                return

            point = np.array(coordinates, dtype=float)

            if point.size == 0:
                return

            self._add_well(point, 0, well_name)

        add_well_if_given("injection_coordinates", "injection")
        add_well_if_given("production_coordinates", "production")





    def _add_well(
        self,
        point: np.ndarray,
        well_index: int,
        well_type: Literal["injection", "production"],
    ) -> None:
        """Helper method to construct a well in 3D as a PointGrid and add respective
        interface.

        Parameters:
            point: Point in space representing well.
            well_index: Assigned number for well of type ``well_type``.
            well_type: Label to add a tag to the point grid labelng as injector or
            producer.

        """
        matrix = self.mdg.subdomains(dim=self.nd)[0]
        assert isinstance(point, np.ndarray)
        p: np.ndarray
        if point.shape == (2,):
            p = np.zeros(3)
            p[:2] = point
        elif point.shape == (3,):
            p = point
        else:
            raise ValueError(
                f"Point for well {(well_type, well_index)} must be 1D array of length "
                + "2 or 3."
            )

        sd_0d = pp.PointGrid(self.units.convert_units(p, "m"))
        # Tag for processing of equations.
        sd_0d.tags[f"{well_type}_well"] = well_index
        sd_0d.compute_geometry()

        self.mdg.add_subdomains(sd_0d)

        # Motivated by wells_3d.py#L828
        cell_matrix = matrix.closest_cell(sd_0d.cell_centers)
        cell_well = np.array([0], dtype=int)
        cell_cell_map = sps.coo_matrix(
            (np.ones(1, dtype=bool), (cell_well, cell_matrix)),
            shape=(sd_0d.num_cells, matrix.num_cells),
        )

        _add_interface(0, matrix, sd_0d, self.mdg, cell_cell_map)
