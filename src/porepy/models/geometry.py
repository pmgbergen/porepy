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
    box: dict
    """Box-shaped domain. FIXME: change to "domain"? """
    nd: int
    """Ambient dimension."""

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
            FractureNetwork.mesh()
                method.

        """
        mesh_args = dict()
        return mesh_args

    def set_md_grid(self) -> None:
        """Create the mixed-dimensional grid.

        A unit square grid with no fractures is assigned by default if
        self.fracture_network contains no fractures. Otherwise, the network's mesh
        method is used.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced grid bucket. box (dict): The
            bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """

        if self.fracture_network.num_frac() == 0:
            # Mono-dimensional grid by default
            phys_dims = np.array([1, 1])
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

        Args:
            subdomains: Subdomains for which to find interfaces.
            codims: Codimension of interfaces to return. Defaults to [1], i.e.
                only interfaces between one dimension apart.

        Returns:
            list[pp.MortarGrid]: Unique, sorted list of interfaces.
        """
        if codims is None:
            codims = [1]
        interfaces = list()
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
            Unique sorted list of all subdomains neighbouring any of the interfaces.

        """
        subdomains = list()
        for interface in interfaces:
            for sd in self.mdg.interface_to_subdomain_pair(interface):
                if sd not in subdomains:
                    subdomains.append(sd)
        return self.mdg.sort_subdomains(subdomains)

    def wrap_grid_attribute(
        self,
        grids: list[pp.GridLike],
        attr: str,
        dim: Optional[int] = None,
        inverse: Optional[bool] = False,
    ) -> pp.ad.Matrix:
        """Wrap a grid attribute as an ad matrix.

        Parameters:
            grids: List of grids on which the property is defined.
            attr: Grid attribute to wrap. The attribute should be a ndarray and will be
                flattened if it is not already a vector.
            dim: Dimensions to include for vector attributes. Intended use is to
                limit the number of dimensions for a vector attribute, e.g. to exclude
                the z-component of a vector attribute in 2d, to acieve compatibility
                with code which is explicitly 2d (e.g. fv discretizations).
            inverse: If True, the inverse of the attribute will be wrapped. This is a
                hack around the fact that the ad framework does not support division.
                FIXME: Remove when ad supports division.

        Returns:
            ad_matrix: The property wrapped as an ad matrix.

        TODO: Test the method (and other methods in this class).

        """
        if len(grids) > 0:
            if dim is None:
                vals = np.hstack([getattr(g, attr).ravel("F") for g in grids])
            else:
                # Only include the first dim dimensions
                vals = np.hstack([getattr(g, attr)[:dim].ravel("F") for g in grids])
            if inverse:
                vals = 1 / vals
            mat = sps.diags(vals)
        else:
            mat = sps.csr_matrix((0, 0))
        ad_matrix = pp.ad.Matrix(mat)
        return ad_matrix

    def basis(self, grids: Sequence[pp.GridLike], dim: int = None) -> np.ndarray:
        """Return a cell-wise basis for all subdomains.

        Parameters:
            grids: List of grids on which the basis is defined. dim: Dimension of the
            base. Defaults to self.nd.

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

    def e_i(self, grids: list[pp.GridLike], i: int, dim: int = None) -> np.ndarray:
        """Return a cell-wise basis function.

        Parameters:
            grids: List of grids on which the basis vector is defined. dim (int):
            Dimension of the functions. i (int): Index of the basis function. Note:
            Counts from 0.

        Returns:
            pp.ad.Matrix: Ad representation of a matrix with the basis functions as
                columns.

        """
        if dim is None:
            dim = self.nd
        assert dim <= self.nd, "Basis functions of higher dimension than the md grid"
        assert i < dim, "Basis function index out of range"
        # Collect the basis functions for each dimension
        e_i = np.zeros(dim).reshape(-1, 1)
        e_i[i] = 1
        # expand to cell-wise column vectors.
        num_cells = sum([g.num_cells for g in grids])
        mat = sps.kron(sps.eye(num_cells), e_i)
        return pp.ad.Matrix(mat)

    def local_coordinates(self, subdomains: list[pp.Grid]) -> pp.ad.Matrix:
        """Ad wrapper around tangential_normal_projections for fractures.

        TODO: Extend to all subdomains.

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

    def subdomain_projections(self, dim: int):
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
        name = f"_subdomain_proj_{dim}"
        if hasattr(self, name):
            proj = getattr(self, name)
        else:
            proj = pp.ad.SubdomainProjections(self.mdg.subdomains(), dim)
            setattr(self, name, proj)
        return proj

    def domain_boundary_sides(
        self, g: pp.Grid
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries. It is assumed the domain is box shaped.

        TODO: Update this from develop before merging.
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
    def tangential_component(self, grids: list[pp.Grid]) -> pp.ad.Operator:
        """Compute the tangential component of a vector field.

        Parameters:
            grids: List of grids on which the vector field is defined.

        Returns:
            tangential: Operator extracting tangential component of the vector field and
            expressing it in tangential basis.
        """
        # We first need an inner product (or dot product), i.e. extract the tangential
        # component of the cell-wise vector v to be transformed. Then we want to express
        # it in the tangential basis. The two operations are combined in a single
        # operator composed right to left: v will be hit by first e_i.T (row vector) and
        # secondly t_i (column vector).
        op = sum(
            [
                self.e_i(grids, i, self.nd - 1) * self.e_i(grids, i, self.nd).T
                for i in range(self.nd - 1)
            ]
        )
        op.set_name("tangential_component")
        return op

    def normal_component(self, grids: list[pp.Grid]) -> pp.ad.Operator:
        """Compute the normal component of a vector field.

        Parameters:
            grids: List of grids on which the vector field is defined.

        Returns:
            normal: Operator extracting normal component of the vector field and
            expressing it in normal basis.
        """
        e_n = self.e_i(grids, self.nd - 1, self.nd)
        e_n.set_name("normal_component")
        return e_n.T

    def internal_boundary_normal_to_outwards(
        self,
        interfaces: list[pp.MortarGrid],
        dim: int,
    ) -> pp.ad.Operator:
        """Flip sign if normal vector points inwards.

        Currently, this is a helper method for the computation of outward normals in
        :meth:`outwards_internal_boundary_normals`. Other usage is allowed, but one
        is adviced to carefully consider subdomain lists when combining this with other
        operators.

        Parameters:
            interfaces: List of interfaces.

        Returns:
            Operator with flipped signs if normal vector points inwards.

        """
        if len(interfaces) == 0:
            sign_flipper = sps.csr_matrix((0, 0))
        else:
            # Two loops are required to be able to prolong each matrix created in the
            # first loop to the full set of subdomains in the second loop.
            matrices = []
            subdomains = []
            for intf in interfaces:
                # Extracting matrix for each interface should in theory allow for multiple
                # matrix subdomains, but this is not tested.
                matrix_subdomain = self.mdg.interface_to_subdomain_pair(intf)[0]
                faces_on_fracture_surface = intf.primary_to_mortar_int().tocsr().indices
                switcher_int = pp.grid_utils.switch_sign_if_inwards_normal(
                    matrix_subdomain, dim, faces_on_fracture_surface
                )
                matrices.append(switcher_int)
                subdomains.append(matrix_subdomain)
            projection = pp.ad.SubdomainProjections(subdomains, dim)
            sign_flipper: pp.ad.Operator = None
            for m, sd in zip(matrices, subdomains):
                m_loc = (
                    projection.face_prolongation([sd])
                    * pp.ad.Matrix(m)
                    * projection.face_restriction([sd])
                )
                if sign_flipper is None:
                    sign_flipper = m_loc
                else:
                    sign_flipper += m_loc

        return sign_flipper

    def outwards_internal_boundary_normals(
        self,
        interfaces: list[pp.MortarGrid],
        unitary: Optional[bool] = False,
        dim: Optional[int] = None,
    ) -> pp.ad.Operator:
        """Compute outward normal vectors on internal boundaries.

        Args:
            interfaces: List of interfaces.
            unitary: If True, return unit vectors, i.e. normalize by face area.
            dim: Dimension of the problem. Defaults to self.nd.

        Returns:
            Operator computing outward normal vectors on internal boundaries. Evaluated
            shape `(num_intf_cells * dim, num_intf_cells * dim)`.

        """
        if len(interfaces) == 0:
            mat = sps.csr_matrix((0, 0))
            return pp.ad.Matrix(mat)
        if dim is None:
            dim = self.nd

        # Main ingredients: Normal vectors for primary subdomains for each interface,
        # and a switcher matrix to flip the sign if the normal vector points inwards.
        # The first is constructed herein, the second is a method of this class.
        # A fair bit of juggling with subdomain lists is needed to distinguish between
        # all subdomains and the subdomains of the primary side of the interface, which
        # are the ones expected by the switcher matrix method.
        # TODO: Consider to let the switcher method operate on the full list of
        # subdomains (and not just the primary ones), and let it return empty entries
        # where appropriate (i.e. for the secondary subdomains).
        subdomains = self.interfaces_to_subdomains(interfaces)
        sd_projection = pp.ad.SubdomainProjections(subdomains, dim)
        primary_subdomains = []
        for intf in interfaces:
            # Extracting matrix for each interface should in theory allow for multiple
            # matrix subdomains, but this is not tested. See also
            # :meth:`internal_boundary_normal_to_outwards`.
            primary_subdomains.append(self.mdg.interface_to_subdomain_pair(intf)[0])
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=dim
        )
        primary_face_normals = self.wrap_grid_attribute(
            primary_subdomains, "face_normals", dim=dim
        )
        # Account for sign of boundary face normals
        flip = self.internal_boundary_normal_to_outwards(interfaces, dim)
        flipped_normals = (
            sd_projection.face_prolongation(primary_subdomains)
            * flip
            * primary_face_normals
            * sd_projection.face_restriction(primary_subdomains)
        )
        # Project to mortar grid
        outwards_normals = (
            mortar_projection.primary_to_mortar_avg
            * flipped_normals
            * mortar_projection.mortar_to_primary_avg
        )
        outwards_normals.set_name("outwards_internal_boundary_normals")

        # Normalize by face area if requested.
        if unitary:
            cell_volumes_inv = self.wrap_grid_attribute(
                interfaces, "cell_volumes", inverse=True
            )

            # Expand cell volumes to nd.
            cell_volumes_inv_nd = sum(
                [e * cell_volumes_inv * e.T for e in self.basis(interfaces, dim=dim)]
            )
            # Scale normals.
            outwards_normals = cell_volumes_inv_nd * outwards_normals
            outwards_normals.set_name("unitary_outwards_internal_boundary_normals")
        return outwards_normals
