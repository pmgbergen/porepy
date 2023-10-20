"""
Geometric projections related to the tangential and normal spaces of a set of
vectors.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TangentialNormalProjection:
    """Represent a set of projections into tangent and normal vectors.

    The spaces are defined by the normal vector (see __init__ documentation).
    The basis for the tangential space is arbitrary (arbitrary direction in 2d,
    rotation angle in 3d). The basis for the tangential is stored in the attribute
    tangential_basis.

    Attributes:
        num_vecs (int): Number of tangent/normal spaces represented by this object.
        dim (int): Dimension of the ambient space.
        tangential_basis (np.array, size: dim x dim-1 x num_vec): Basis vectors for the
            tangential space.
        projection (np.array, size dim x dim x num_vecs): Projection matrices onto the
            tangential and normal space. The first dim-1 rows represent projection to the
            tangential space, the final row is the normal component.
        normals (np.array, size dim x num_vecs): Unit normal vectors.

    """

    def __init__(self, normals, dim=None):
        if dim is None:
            dim = normals.shape[0]

        # Normalize vectors
        normals = normals / np.linalg.norm(normals, axis=0)

        self.num_vecs = normals.shape[1]
        self.dim = dim

        # Compute normal and tangential basis
        basis, normal = self._decompose_vector(normals)

        basis = basis.reshape((dim, dim, self.num_vecs))
        self.tangential_basis = basis[:, :-1, :]

        # The projection is found by inverting the basis vectors
        self.projection = self._invert_3d_matrix(basis)
        self.normals = normal

    ## Methods for genertation of projection matrices

    def project_tangential_normal(self, num=None):
        """Define a projection matrix to decompose a matrix into tangential
        and normal components.

        The intended usage is to decompose a grid-based vector variable into the
        tangent and normal spaces of the grid, with the tacit understanding that there is
        a single normal vector shared for all the cells (or faces) in the grid.

        The method can also create projection matrix based on unequal normal vectors.
        One projection will be generated per column in self.normal. To activate
        this behavior, set num=None.

        Parameters:
            num (int, optional): Number of (equal) projections to be generated.
                Will correspond to the number of cells / faces in the grid.
                The projection matrix will have num * self.dim columns. If not
                specified (default), one projection will be generated per vector in
                self.normals.
                NOTE: If self.num_vecs > 1, but num is not None, only the first
                given normal vector will be used to generate the tangential space.

        Returns:
            scipy.sparse.csc_matrix: Projection matrix, structure as a block
                diagonal matrix, with block size dim x dim.
                For each block, the first dim-1 rows projects onto the tangent
                space, the final row projects onto the normal space.
                size: ((self.dim * num) x (self.dim * num). If num is not None,
                size: ((self.dim * num_vecs) x (self.dim * num_vecs)

        """
        if num is None:
            num = self.projection.shape[-1]
            data = np.array(
                [self.projection[:, :, i].ravel("f") for i in range(num)]
            ).ravel()
        else:
            data = np.tile(self.projection[:, :, 0].ravel(order="F"), num)

        mat = pp.matrix_operations.csc_matrix_from_blocks(data, self.dim, num)

        return mat

    def project_tangential(self, num=None):
        """Define a projection matrix of a specific size onto the tangent space.

        The intended usage is to project a grid-based vector variable onto the
        tangent space of the grid, with the tacit understanding that there is
        a single normal vector shared for all the cells (or faces) in the grid.

        The method can also create projection matrix based on unequal normal vectors.
        One projection will be generated per column in self.normal. To activate
        this behavior, set num=None.

        Parameters:
            num (int, optional): Number of (equal) projections to be generated.
                Will correspond to the number of cells / faces in the grid.
                The projection matrix will have num * self.dim columns. If not
                specified (default), one projection will be generated per vector in
                self.normals.
                NOTE: If self.num_vecs > 1, but num is not None, only the first
                given normal vector will be used to generate the tangential space.

        Returns:
            scipy.sparse.csc_matrix: Tangential projection matrix, structure as a block
                diagonal matrix. The first (dim-1) x dim block projects onto the first
                tangent space, etc.
                size: ((self.dim - 1) * num) x (self.dim * num). If num is not None,
                size: ((self.dim - 1) * num_vecs) x (self.dim * num_vecs)

        """
        # Construct the full projection matrix - tangential and normal
        full_projection = self.project_tangential_normal(num)

        # Find type and size of projection.
        if num is None:
            num = self.num_vecs
        size_proj = self.dim * num

        # Generate restriction matrix to the tangential space only
        rows = np.arange(num * (self.dim - 1))
        cols = np.setdiff1d(
            np.arange(size_proj), np.arange(self.dim - 1, size_proj, self.dim)
        )
        data = np.ones_like(rows)
        remove_normal_components = sps.csc_matrix(
            (data, (rows, cols)), shape=(rows.size, size_proj)
        )

        # Return the restricted matrix.
        return remove_normal_components * full_projection

    def project_normal(self, num=None):
        """Define a projection matrix of a specific size onto the normal space.

        The projection is from global coordinates to the local coordinates defined

        The method can also create projection matrix based on unequal normal vectors.
        One projection will be generated per column in self.normal. To activate
        this behavior, set num=None.

        Parameters:
            num (int, optional): Number of (equal) projections to be generated.
                The projection matrix will have num * self.dim columns. If not
                specified (default), one projection will be generated per vector in
                self.normals.
                NOTE: If self.num_vecs > 1, but num is not None, only the first
                given normal vector will be used to generate the normal space.

        Returns:
            scipy.sparse.csc_matrix: Tangential projection matrix, structure as a block
                diagonal matrix. The first 1 x dim block projects onto the first
                tangent space, etc.
                size: num x (self.dim * num). If num is not None.
                size: num_vecs x (self.dim * num_vecs) els.

        """
        # Generate full projection matrix
        full_projection = self.project_tangential_normal(num)

        # Find mode and size of projection
        if num is None:
            num = self.num_vecs

        size_proj = self.dim * num

        # Construct restriction matrix to normal space.
        rows = np.arange(num)
        cols = np.arange(self.dim - 1, size_proj, self.dim)
        data = np.ones_like(rows)
        remove_tangential_components = sps.csc_matrix(
            (data, (rows, cols)), shape=(rows.size, size_proj)
        )

        # Return the restricted matrix
        return remove_tangential_components * full_projection

    def local_projection(self, ind=None):
        """Get the local projection matrix (reference)

        Parameters:
            ind (int, optional): Index (referring to the order of the normal vectors
                provided to __init__) of the basis to return. Defaults to the first one.

        Returns:
            np.array (self.dim x self.dim): Local projection matrix. Multiplication
                gives projection to the tangential space (first self.dim - 1 rows)
                and normal space (last)

        """
        if ind is None:
            ind = 0
        return self.projection[:, :, ind]

    ### Helper functions below
    def _decompose_vector(self, nc):
        if self.dim == 2:
            # Normalize the normal vector, just to be sure
            normal = nc / np.linalg.norm(nc, axis=0)
            # The tanegntial vector in 2d is deterministic up to an arbitrary sign. We
            # choose the sign such that the vector points in the positive x-direction.
            tc1 = np.zeros_like(normal)
            negative_n1 = normal[1] < 0
            tc1[:, negative_n1] = np.vstack(
                [-normal[1, negative_n1], normal[0, negative_n1]]
            )
            positive_n1 = np.logical_not(negative_n1)
            tc1[:, positive_n1] = np.vstack(
                [normal[1, positive_n1], -normal[0, positive_n1]]
            )

            basis = np.hstack([tc1, normal])
        else:  # self.dim == 3
            # Normalize the normal vector, just to be sure
            normal = nc / np.linalg.norm(nc, axis=0)
            # In 3d, there is some freedom in choosing the tangential vectors, but it is
            # important that the definition minimize the risk for poorly conditioned
            # computations (implementer note to self: Exactly what this means is
            # unclear, but we have had spurious numerical issues that were traced back
            # to a previous implementation using random vectors within the tangent
            # plane. What caused this was never clear, but we do not want to take
            # chances here). We choose one tangential vector to lie in the plane
            # orthorgonal to the maximum direction of the normal vector (ex: a normal
            # vector of [1, 3, 2] would give a first tangent vector in the xz-plane).
            # The second tangent vector is the cross product of the normal and the first
            # tangent vector.

            # Find the maximum direction of the normal vector for each column (each
            # individual normal vector)
            max_dim = np.argmax(np.abs(normal), axis=0)
            # Tangent vectors, to be filled in
            tc1 = np.zeros_like(nc)
            tc2 = np.zeros_like(nc)

            # Loop over the dimensions, fill in all columns (individual tangent vectors)
            # that have their maximum along this dimension.
            for i in range(self.dim):
                # Columns to be filled in
                hit = max_dim == i
                # The other dimensions
                other_dim = np.setdiff1d(np.arange(self.dim), i)

                # NOTE: We could try to assign a positive sign to the first tangent, as
                # is done in 2d, but the benefit is less clear in 3d. Also, the special
                # case of a normal vector aligned with a coordinate direction, where a
                # positive direction could be most useful, is covered just below.
                tc1[other_dim[0], hit] = -normal[other_dim[1], hit]
                tc1[other_dim[1], hit] = normal[other_dim[0], hit]

                # If the normal vector is aligned with one of the coordinate directions,
                # the tangent assigned above will be zero. In this case, we assign a
                # positive value along the first dimension in other_dim. In practice
                # (because of the definition of other_dim), this means the tangent
                # vector points along the x-axis if the normal vector is aligned with
                # the y- or z-axis, and along the y-axis if the normal vector is
                # aligned with the x-axis.
                aligned_with_axis = np.logical_and(
                    hit, np.linalg.norm(normal[other_dim], axis=0) < 1e-8
                )
                tc1[other_dim[0], aligned_with_axis] = 1

            # Normalize the first tangent vectors
            tc1 = tc1 / np.linalg.norm(tc1, axis=0)
            # The second tangent vector is the cross product of the normal and the
            # first tangent vector.
            tc2 = np.cross(nc, tc1, axis=0)
            # Normalize the second tangent vectors
            tc2 = tc2 / np.linalg.norm(tc2, axis=0)

            # Define the matrix of tangent and normal vectors
            basis = np.hstack([tc1, tc2, normal])

        return basis, normal

    def _invert_3d_matrix(self, M):
        """
        Find the inverse of the (m,m,k) 3D ndArray M. The inverse is interpreted as the
        2d inverse of M[:, :, i] for i = 0...k

        Parameters:
        M: (m, m, k) ndArray

        Returns:
        M_inv: Inverse of M
        """
        M_inv = np.zeros(M.shape)
        for i in range(M.shape[-1]):
            M_inv[:, :, i] = np.linalg.inv(M[:, :, i])
        return M_inv


def set_local_coordinate_projections(
    mdg: pp.MixedDimensionalGrid, interfaces: Optional[list[pp.MortarGrid]] = None
) -> None:
    """Define a local coordinate system, and projection matrices, for all
    grids of co-dimension 1.

    Parameters:
        mdg: Mixed-dimensional grid.
        interfaces: List of MortarGrids. If not provided, all interfaces of co-dimension
            1 will be considered.

    The function adds one item to the data dictionary of all MixedDimensionalGrid edges
    that neighbors a co-dimension 1 grid, defined as:
        key: Literal["tangential_normal_projection"],
        value: pp.TangentialNormalProjection object providing projection to the surface
            of the lower-dimensional grid.

    Note that grids of co-dimension 2 and higher are ignored in this construction,
    as we do not plan to do contact mechanics on these objects.

    It is assumed that the surface is planar.

    """
    if interfaces is None:
        interfaces = mdg.interfaces(dim=mdg.dim_max() - 1)

    # Information on the vector normal to the surface is not available directly
    # from the surface grid (it could be constructed from the surface geometry,
    # which spans the tangential plane). We instead get the normal vector from
    # the adjacent higher dimensional grid.
    # We therefore access the grids via the edges of the mixed-dimensional grid.
    for intf in interfaces:
        # Only consider edges where the lower-dimensional neighbor is of co-dimension 1
        if not intf.dim == (mdg.dim_max() - 1):
            continue

        # Neighboring grids
        sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)

        # Find faces of the higher dimensional grid that coincide with the mortar
        # grid. Go via the primary to mortar projection.
        # Convert matrix to csr, then the relevant face indices are found from
        # the (column) indices.
        faces_on_surface = intf.primary_to_mortar_int().tocsr().indices

        # Find out whether the boundary faces have outwards pointing normal vectors
        # Negative sign implies that the normal vector points inwards.
        sgn, _ = sd_primary.signs_and_cells_of_boundary_faces(faces_on_surface)

        # Unit normal vector.
        unit_normal = sd_primary.face_normals[: sd_primary.dim] / sd_primary.face_areas
        # Ensure all normal vectors on the relevant surface points outwards.
        unit_normal[:, faces_on_surface] *= sgn

        # Now we need to pick out *one*  normal vector of the higher dimensional grid.

        # which coincides with this mortar grid, so we kill off all entries for the
        # "other" side:
        unit_normal[:, intf._ind_face_on_other_side] = 0

        # Project to the mortar and then to the fracture.
        outwards_unit_vector_mortar = intf.primary_to_mortar_int().dot(unit_normal.T).T
        normal_lower = (
            intf.mortar_to_secondary_int().dot(outwards_unit_vector_mortar.T).T
        )

        # NOTE: The normal vector is based on the first cell in the mortar grid,
        # and will be pointing from that cell towards the other side of the
        # mortar grid. This defines the positive direction in the normal direction.
        # Although a simpler implementation seems to be possible, going via the
        # first element in faces_on_surface, there is no guarantee that this will
        # give us a face on the positive (or negative) side, hence the more general
        # approach is preferred.
        #
        # NOTE: The basis for the tangential direction is determined by the
        # construction internally in TangentialNormalProjection.
        projection = pp.TangentialNormalProjection(normal_lower)

        d_l = mdg.subdomain_data(sd_secondary)
        # Store the projection operator in the lower-dimensional data.
        d_l["tangential_normal_projection"] = projection
