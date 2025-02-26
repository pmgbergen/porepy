"""Projections into tangent and normal spaces for a local coordinate system.

The projections are represented by the class TangentialNormalProjection.

The module also contains a function ``set_local_coordinate_projections`` that sets
projection matrices for all grids of co-dimension 1 in a mixed-dimensional grid.

"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TangentialNormalProjection:
    """Represent a set of projections into tangent and normal vectors.

    The spaces are defined by the normal vector (see __init__ documentation).

    Parameters:
        normals: ``shape=(dim, num_vecs)``

            Normal vectors that define the local coordinate systems.

    """

    def __init__(self, normals: np.ndarray) -> None:
        # Normalize vectors
        normals = normals / np.linalg.norm(normals, axis=0)

        self.num_vecs: int = normals.shape[1]
        """Number of normal vectors represented by this object."""
        self.dim: int = normals.shape[0]
        """Dimension of the ambient space."""

        # Compute normal and tangential basis
        basis, normal = self._construct_local_basis(normals)

        basis = basis.reshape((self.dim, self.dim, self.num_vecs))

        # The projection is found by inverting the basis vectors
        self._projection = self._invert_3d_matrix(basis)
        """Projection matrix to the local coordinate system."""

        self.normals = normal
        """Normal vectors, ``shape=(dim, num_vecs)``."""

    ## Methods for genertation of projection matrices

    def project_tangential_normal(self, num: Optional[int] = None) -> sps.spmatrix:
        """Define a projection matrix to decompose a matrix into tangential
        and normal components.

        The intended usage is to decompose a vector defined in the global coordinate
        system into the tangent and normal spaces of a local coordinate system. The
        projection is constructed either by repeating the projection matrix obtained
        from the first (referring to the order of the normal vectors provided at
        initialization) normal vector ``num`` times (if ``num`` is not None), or by
        stacking the projection matrices for each normal vector that was provided at
        initialization (if ``num`` is None). Note that in the former option, only the
        first normal vector is used to define the normal space.

        Parameters:
            num: Number of projections to be generated. See above for details.

        Returns:
            Projection matrix as a block diagonal matrix, with block size
                ``dim x dim``. For each block, the first ``dim - 1`` rows project onto
                the tangent space, the final row projects onto the normal space.

        """
        if num is None:
            num = self._projection.shape[-1]
            data = np.array(
                [self._projection[:, :, i].ravel("F") for i in range(num)]
            ).ravel()
        else:
            data = np.tile(self._projection[:, :, 0].ravel(order="F"), num)

        mat = pp.matrix_operations.csc_matrix_from_dense_blocks(data, self.dim, num)

        return mat

    def project_tangential(self, num: Optional[int] = None) -> sps.spmatrix:
        """Define a projection matrix of a specific size onto the tangential space.

        The intended usage is to project a vector defined in the global coordinate
        system into the tangent space of a local coordinate system. The projection is
        constructed either, by repeating the projection matrix obtained from the first
        (referring to the order of the normal vectors provided at initialization) normal
        vector ``num`` times (if ``num`` is not None), or by stacking the projection
        matrices for each normal vector that was provided at initialization (if ``num``
        is None). Note that in the former option, only the first normal vector is used
        to define the normal space.

        Parameters:
            num: Number of projections to be generated. See above for details.

        Returns:
            Tangential projection matrix, structured as a block diagonal matrix.

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

    def project_normal(self, num: Optional[int] = None) -> sps.spmatrix:
        """Define a projection matrix of a specific size onto the normal space.

        The intended usage is to project a vector defined in the global coordinate
        system into the normal space of a local coordinate system. The projection is
        constructed either, by repeating the projection matrix obtained from the first
        (referring to the order of the normal vectors provided at initialization) normal
        vector ``num`` times (if ``num`` is not None), or by stacking the projection
        matrices for each normal vector that was provided at initialization (if ``num``
        is None). Note that in the former option, only the first normal vector is used
        to define the normal space.

        Parameters:
            num: Number of projections to be generated. See above for details.

        Returns:
            Normal projection matrix, structured as a block diagonal matrix.

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

    ### Helper functions below
    def _construct_local_basis(
        self, normal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct a local basis for the tangential space and the normal space from a
        set of normal vectors.

        Parameters:
            normal: Normal vectors ``shape=(dim, num_vecs)``.

        Returns:
            Basis for the tangential and normal space, and the normal vector itself.

            np.ndarray: ``shape=(dim, dim, num_vecs)``. The first dim-1 columns are
                the basis for the tangential space, the final column is the normal
                vector.

            np.ndarray: ``shape=(dim, num_vecs)``. The normal vector.

        """

        if self.dim == 2:
            # Normalize the normal vector, just to be sure
            normal = normal / np.linalg.norm(normal, axis=0)
            # The tangential vector in 2d is deterministic up to an arbitrary sign. We
            # choose the sign such that the vector points in the positive x-direction.
            tc1 = np.zeros_like(normal)
            negative_n1 = normal[1] < 0
            tc1[:, negative_n1] = np.vstack(
                [-normal[1, negative_n1], normal[0, negative_n1]]
            )
            positive_n1 = normal[1] > 0
            tc1[:, positive_n1] = np.vstack(
                [normal[1, positive_n1], -normal[0, positive_n1]]
            )
            # If the normal vector is aligned with the x-axis, we assign a positive
            # value along the y-axis.
            equal_zero = normal[1] == 0
            tc1[1, equal_zero] = 1

            basis = np.hstack([tc1, normal])
        else:  # self.dim == 3
            # Normalize the normal vector, just to be sure
            normal = normal / np.linalg.norm(normal, axis=0)
            # In 3d, there is some freedom in choosing the tangential vectors, but it is
            # important that the definition minimize the risk for poorly conditioned
            # computations (implementer note to self: Exactly what this means is
            # unclear, but we have had spurious numerical issues that were traced back
            # to a previous implementation using random vectors within the tangent
            # plane. What caused this was never clear, but we do not want to take
            # chances here). We choose one tangential vector to lie in the plane
            # orthorgonal to the maximum direction of the normal vector (ex: a normal
            # vector of [1, 3, 2] has maximum component in the y direction and thus gives
            # a first tangent vector in the xz-plane). The second tangent vector is the
            # cross product of the normal and the first tangent vector.

            # Find the maximum direction of the normal vector for each column (each
            # individual normal vector)
            max_dim = np.argmax(np.abs(normal), axis=0)
            # Tangent vectors, to be filled in
            tc1 = np.zeros_like(normal)
            tc2 = np.zeros_like(normal)

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
            tc2 = np.cross(normal, tc1, axis=0)
            # Normalize the second tangent vectors
            tc2 = tc2 / np.linalg.norm(tc2, axis=0)

            # Define the matrix of tangent and normal vectors
            basis = np.hstack([tc1, tc2, normal])

        return basis, normal

    def _invert_3d_matrix(self, M: np.ndarray) -> np.ndarray:
        """Find the inverse of the ``(m,m,k)`` 3D array ``M``.

        The inverse is interpreted as the 2d inverse of ``M[:, :, i]`` for ``i = 0...k``

        Parameters:
            M: Array with ``shape=(m, m, k)``.

        Returns:
            Inverse of ``M``.

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

    The function adds one item to the data dictionary of all MixedDimensionalGrid edges
    that neighbors a co-dimension 1 grid, defined as:
        key: Literal["tangential_normal_projection"],
        value: pp.TangentialNormalProjection object providing projection to the surface
            of the lower-dimensional grid.

    Note that grids of co-dimension 2 and higher are ignored in this construction,
    as we do not plan to do contact mechanics on these objects.

    It is assumed that the surface is planar.

    Parameters:
        mdg: Mixed-dimensional grid.
        interfaces: List of MortarGrids. If not provided, all interfaces of co-dimension
            1 will be considered.

    """
    if interfaces is None:
        interfaces = mdg.interfaces(dim=mdg.dim_max() - 1)

    # Information on the vector normal to the surface is not available directly from the
    # surface grid (it could be constructed from the surface geometry, which spans the
    # tangential plane). We instead get the normal vector from the adjacent higher
    # dimensional grid. We therefore access the grids via the edges of the
    # mixed-dimensional grid.
    for intf in interfaces:
        # Only consider edges where the lower-dimensional neighbor is of co-dimension 1
        if not intf.dim == (mdg.dim_max() - 1):
            continue

        # Neighboring grids
        sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)

        # Find faces of the higher dimensional grid that coincide with the mortar grid.
        # Go via the primary to mortar projection. Convert matrix to csr, then the
        # relevant face indices are found from the (column) indices.
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

        # NOTE: The normal vector is based on the first cell in the mortar grid, and
        # will be pointing from that cell towards the other side of the mortar grid.
        # This defines the positive direction in the normal direction. Although a
        # simpler implementation seems to be possible, going via the first element in
        # faces_on_surface, there is no guarantee that this will give us a face on the
        # positive (or negative) side, hence the more general approach is preferred.
        #
        # NOTE: The basis for the tangential direction is determined by the construction
        # internally in TangentialNormalProjection.
        projection = pp.TangentialNormalProjection(normal_lower)

        d_l = mdg.subdomain_data(sd_secondary)
        # Store the projection operator in the lower-dimensional data.
        d_l["tangential_normal_projection"] = projection


def sides_of_fracture(
    intf: pp.MortarGrid, sd_primary: pp.Grid, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Identify the top and bottom sides of the interface based on a direction vector.

    The positive side is defined as the one where the outwards normal vectors of the
    matrix point in the direction of the direction vector. The negative side is defined
    as the one having outwards normal vectors pointing in the opposite direction as the
    direction vector.

    Usage note: The third return value is used to identify whether the positive side is the
    first side of the mortar grid. This is important e.g. when considering the jump
    across the interface, which is defined as the second side minus the first side.
    Thus, if the negative side is the first side, the jump is the bottom side (relative
    to the direction vector) minus the top side, implying that a negative jump (in
    global coordinates) is a tensile opening.

    The current implementation assumes that the interface represents a planar surface.

    Parameters:
        intf: Interface where the sides are to be identified.
        sd_primary: Subdomain of the primary grid.
        direction: Vector used to identify the top side. Shape is ``(3,)``,  ``(3,1)`` or
            ``(3, intf.num_cells)``. The former two will be broadcasted. The latter in
            theory allows for different direction vectors for each cell, but this is not
            tested.

    Returns:
        Tuple of two arrays and a bool. The first containing the indices of the positive
        side, and the second containing the indices of the negative side. The third
        element is a boolean indicating if the positive side is the first side of the
        mortar grid.

    """
    # PorePy grid coordinates are 3d regardless of the dimension of the grid.
    coord_dim = 3

    # Compute outwards normals in the matrix and project to the interface.
    faces_on_fracture_surface = np.where(sd_primary.tags["fracture_faces"])[0]
    switcher_int = pp.grid_utils.switch_sign_if_inwards_normal(
        sd_primary, coord_dim, faces_on_fracture_surface
    )
    normal_primary = switcher_int @ sd_primary.face_normals.ravel(order="F")
    normal_intf = (intf.primary_to_mortar_avg(coord_dim) @ normal_primary).reshape(
        (coord_dim, -1), order="F"
    )
    # Identify the top side of the interface using the inner product with the direction
    # vector.
    inner = np.sum(normal_intf * direction.reshape(coord_dim, -1), axis=0)
    if np.allclose(inner, 0):
        raise ValueError("The direction vector is orthogonal to the normal vectors.")
    negative_side = np.where(inner < 0)[0]
    positive_side = np.where(inner >= 0)[0]
    positive_side_first = None
    # Compare with sides of the mortar grid.

    for i, (proj, _) in enumerate(intf.project_to_side_grids()):
        # The projection matrix is a block diagonal matrix, where each block projects
        # from the mortar grid cells of the side grid in question. Thus, the nonzero
        # column indices identify the mortar cells on this side.
        proj_inds = proj.nonzero()[1]
        if np.allclose(positive_side, proj_inds):
            positive_side_first = i == 0
        else:
            assert np.allclose(negative_side, proj_inds)
    if positive_side_first is None:
        # This should not happen for planar surfaces (and possibly other underlying
        # assumptions).
        raise ValueError("Could not identify the top side as the first or second side.")
    return positive_side, negative_side, positive_side_first
