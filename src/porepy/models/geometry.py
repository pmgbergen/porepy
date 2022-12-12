"""Geometry definition for simulation setup.

"""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class ModelGeometry:
    """This class provides geometry related methods and information for a simulation
    model."""

    # Define attributes to be assigned later
    fracture_network: Union[pp.FractureNetwork2d, pp.FractureNetwork3d]
    """Representation of fracture network including intersections."""
    well_network: pp.WellNetwork3d
    """Well network."""
    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""
    box: dict[str, float]
    """Box-shaped domain. FIXME: change to "domain"? """
    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    
    units: pp.Units
    """Unit system."""

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid."""
        # Create fracture network and mixed-dimensional grid
        self.set_fracture_network()
        self.set_md_grid()
        self.nd: int = self.mdg.dim_max()
        # If fractures are present, it is advised to call
        pp.contact_conditions.set_projections(self.mdg)

    def set_fracture_network(self) -> None:
        """Assign fracture network class."""
        self.fracture_network = pp.FractureNetwork2d()

    def mesh_arguments(self) -> dict:
        """Mesh arguments for md-grid creation.

        Returns:
            mesh_args: Dictionary of meshing arguments compatible with
                FractureNetwork.mesh() method.

        """
        mesh_args: dict[str, float] = {}
        return mesh_args

    def set_md_grid(self) -> None:
        """Create the mixed-dimensional grid.

        A unit square grid with no fractures is assigned by default if
        self.fracture_network contains no fractures. Otherwise, the network's mesh
        method is used.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.

        """

        if self.fracture_network.num_frac() == 0:
            # Length scale:
            ls = 1 / self.units.m
            # Mono-dimensional grid by default
            phys_dims = np.array([1, 1]) * ls
            n_cells = np.array([1, 1])
            self.box = pp.geometry.bounding_box.from_points(
                np.array([[0, 0], phys_dims]).T
            )
            g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
            g.compute_geometry()
            self.mdg = pp.meshing.subdomains_to_mdg([[g]])
        else:
            self.mdg = self.fracture_network.mesh(self.mesh_arguments())
            self.box = self.fracture_network.domain

    def subdomains_to_interfaces(
        self, subdomains: list[pp.Grid], codims: Optional[list] = None
    ) -> list[pp.MortarGrid]:
        """Interfaces neighbouring any of the subdomains.

        Parameters:
            subdomains: Subdomains for which to find interfaces.
            codims: Codimension of interfaces to return. Defaults to [1], i.e.
                only interfaces between subdomains one dimension apart.

        Returns:
            Unique list of all interfaces neighboring any of the subdomains. Interfaces
            are sorted according to their index, as defined by the mixed-dimensional
            grid.

        """
        if codims is None:
            codims = [1]

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

    def wrap_grid_attribute(
        self,
        grids: Sequence[pp.GridLike],
        attr: str,
        dim: Optional[int] = None,
        inverse: bool = False,
    ) -> pp.ad.Matrix:
        """Wrap a grid attribute as an ad matrix.

        Parameters:
            grids: List of grids on which the property is defined.
            attr: Grid attribute to wrap. The attribute should be a ndarray and will be
                flattened if it is not already one dimensional.
            dim: Dimensions to include for vector attributes. Intended use is to
                limit the number of dimensions for a vector attribute, e.g. to exclude
                the z-component of a vector attribute in 2d, to acieve compatibility
                with code which is explicitly 2d (e.g. fv discretizations).
            inverse: If True, the inverse of the attribute will be wrapped. This is a
                hack around the fact that the Ad framework does not support division.
                FIXME: Remove when ad supports division.

        Returns:
            The property wrapped as an ad matrix, with the wrapped attribute on the
            diagonal.

        Raises:
            ValueError: If one of the grids does not have the attribute.
            ValueError: If the attribute is not a ndarray.

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
            if inverse:
                vals = 1 / vals
            mat = sps.diags(vals)
        else:
            # For an empty list of grids, return an empty matrix
            mat = sps.csr_matrix((0, 0))

        ad_matrix = pp.ad.Matrix(mat)
        ad_matrix.set_name(f"Matrix wrapping attribute {attr} on {len(grids)} grids")
        return ad_matrix

    def basis(self, grids: Sequence[pp.GridLike], dim: int = None) -> np.ndarray:
        """Return a cell-wise basis for all subdomains.

        Parameters:
            grids: List of grids on which the basis is defined.
            dim: Dimension of the base. Defaults to self.nd.

        Returns:
            Array (dim) of pp.ad.Matrix, each of which represents a basis function.

        """
        if dim is None:
            dim = self.nd

        assert dim <= self.nd, "Basis functions of higher dimension than the md grid"
        # Collect the basis functions for each dimension
        basis = []
        for i in range(dim):
            basis.append(self.e_i(grids, i, dim))
        # Stack the basis functions horizontally
        return np.hstack(basis)

    def e_i(
        self, grids: Sequence[pp.GridLike], i: int, dim: int = None
    ) -> pp.ad.Matrix:
        """Return a cell-wise basis function.

        It is assumed that the grids are embedded in a space of dimension dim and
        aligned with the coordinate axes, that is, the reference space of the grid.
        Moreover, the grid is assumed to be planar.

        Parameters:
            grids: List of grids on which the basis vector is defined.
            dim: Dimension of the functions.
            i: Index of the basis function. Note: Counts from 0.

        Returns:
            pp.ad.Matrix: Ad representation of a matrix with the basis functions as
                columns.

        """
        # TODO: Should we expand this to grids not aligned with the coordinate axes, and
        # possibly unify with ``porepy.utils.projections.TangentialNormalProjection``?

        if dim is None:
            dim = self.nd

        # Sanity checks
        assert dim <= self.nd, "Basis functions of higher dimension than the md grid"
        assert i < dim, "Basis function index out of range"

        # Construct a single vector, and later stack it to a matrix
        # Collect the basis functions for each dimension
        e_i = np.zeros(dim).reshape(-1, 1)
        e_i[i] = 1
        # Expand to cell-wise column vectors.
        num_cells = sum([g.num_cells for g in grids])
        # Expand to a matrix.
        mat = sps.kron(sps.eye(num_cells), e_i)
        return pp.ad.Matrix(mat)

    def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        """Ad wrapper around tangential_normal_projections for fractures.

        TODO: Extend to all subdomains, not only codimension 1?
        TODO: Revise this method if we every assign a reference mapping attribute to
        all grids.

        Parameters:
            subdomains: List of subdomains for which to compute the local coordinates.

        Returns:
            Local coordinates as a pp.ad.Matrix.

        """
        # For now, assert all subdomains are fractures, i.e. dim == nd - 1
        assert all([sd.dim == self.nd - 1 for sd in subdomains])
        if len(subdomains) > 0:
            local_coord_proj_list = [
                self.mdg.subdomain_data(sd)[
                    "tangential_normal_projection"
                ].project_tangential_normal(sd.num_cells)
                for sd in subdomains
            ]
            local_coord_proj = sps.block_diag(local_coord_proj_list)
        else:
            local_coord_proj = sps.csr_matrix((0, 0))
        return pp.ad.Matrix(local_coord_proj)

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
        self, g: pp.Grid
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Obtain indices of grid faces on the different parts of the domain boundary.

        It is assumed the domain is box shaped.

        TODO: Update this from develop before merging.
        Parameters:
            g: Grid for which to obtain the boundary faces.

        """
        tol = 1e-10
        box = self.box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        if self.nd == 1:
            north = np.zeros(g.num_faces, dtype=bool)
            south = north.copy()
        else:
            north = g.face_centers[1] > box["ymax"] - tol
            south = g.face_centers[1] < box["ymin"] + tol
        if self.nd < 3:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom

    # Local basis related methods
    def tangential_component(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Compute the tangential component of a vector field.

        The tangential space is defined according to the local coordinates of the
        subdomains.

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
        # secondly t_i (column vector).
        op: pp.ad.Operator = sum(
            [
                self.e_i(subdomains, i, self.nd - 1)
                * self.e_i(subdomains, i, self.nd).T
                for i in range(self.nd - 1)
            ]
        )
        op.set_name("tangential_component")
        return op

    def normal_component(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Compute the normal component of a vector field.

        Parameters:
            subdomains: List of grids on which the vector field is defined.

        Returns:
            Operator extracting normal component of the vector field and expressing it
            in normal basis.
        """
        e_n = self.e_i(subdomains, self.nd - 1, self.nd)
        e_n.set_name("normal_component")
        return e_n.T

    def internal_boundary_normal_to_outwards(
        self,
        subdomains: list[pp.Grid],
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
            sign_flipper = pp.ad.Matrix(sps.csr_matrix((0, 0)))
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
            sign_flipper = pp.ad.Matrix(sps.block_diag(matrices).tocsr())
        sign_flipper.set_name("Flip_normal_vectors")
        return sign_flipper

    def outwards_internal_boundary_normals(
        self,
        interfaces: list[pp.MortarGrid],
        unitary: Optional[bool] = False,
    ) -> pp.ad.Operator:
        """Compute outward normal vectors on internal boundaries.

        Args:
            interfaces: List of interfaces.
            unitary: If True, return unit vectors, i.e. normalize by face area.

        Returns:
            Operator computing outward normal vectors on internal boundaries. Evaluated
            shape `(num_intf_cells * dim, num_intf_cells * dim)`.

        """
        if len(interfaces) == 0:
            # Special case if no interfaces.
            mat = sps.csr_matrix((0, 0))
            return pp.ad.Matrix(mat)

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
        # Face normals on the primary subdomains
        primary_face_normals = self.wrap_grid_attribute(
            primary_subdomains, "face_normals", dim=self.nd
        )
        # Account for sign of boundary face normals. This will give a matrix with a
        # shape equal to the total number of faces in all primary subdomains.
        flip = self.internal_boundary_normal_to_outwards(primary_subdomains, self.nd)
        # Flip the normal vectors. Unravelled from the right: Restrict from faces on all
        # subdomains to the primary ones, multiply with the face normals, flip the
        # signs, and project back up to all subdomains.
        flipped_normals = flip * primary_face_normals
        # Project to mortar grid, as a mapping from mortar to the subdomains and back
        # again.
        outwards_normals = (
            mortar_projection.primary_to_mortar_avg
            * flipped_normals
            * mortar_projection.mortar_to_primary_avg
        )
        outwards_normals.set_name("outwards_internal_boundary_normals")

        # Normalize by face area if requested.
        if unitary:
            # 1 over cell volumes on the interfaces
            cell_volumes_inv = self.wrap_grid_attribute(
                interfaces, "cell_volumes", inverse=True
            )

            # Expand cell volumes to nd.
            # EK: It should be possible to do this in a better, less opaque, way. A
            # Kronecker product comes to mind, but this will require an extension of the
            # Ad matrix.
            cell_volumes_inv_nd = sum(
                [e * cell_volumes_inv * e.T for e in self.basis(interfaces, self.nd)]
            )
            # Scale normals.
            outwards_normals = cell_volumes_inv_nd * outwards_normals
            outwards_normals.set_name("unitary_outwards_internal_boundary_normals")
        return outwards_normals
