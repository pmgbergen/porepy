"""
Geometric projections related to the tangential and normal spaces of a set of
vectors.
"""
import numpy as np
import scipy.sparse as sps

import porepy as pp

module_sections = ["matrix", "numerics", "discretization"]


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
        normal (np.array, size dim x num_vecs): Unit normal vectors.

    """

    @pp.time_logger(sections=module_sections)
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
    @pp.time_logger(sections=module_sections)
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

        mat = pp.utils.sparse_mat.csc_matrix_from_blocks(data, self.dim, num)

        return mat

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def project_normal(self, num=None):
        """Define a projection matrix of a specific size onto the normal space.

        The intended usage is to project a grid-based vector variable onto the
        normal space of the grid, with the tacit understanding that there is
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

    @pp.time_logger(sections=module_sections)
    def local_projection(self, ind=None):
        """Get the local projection matrix (refe)

        Paremeters:
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

    @pp.time_logger(sections=module_sections)
    def _decompose_vector(self, nc):
        if self.dim == 3:
            t1 = np.random.rand(self.dim, 1) * np.ones(self.num_vecs)
            t2 = np.random.rand(self.dim, 1) * np.ones(self.num_vecs)
            normal, tc1, tc2 = self._gram_schmidt(nc, t1, t2)
            basis = np.hstack([tc1, tc2, normal])
        else:
            t1 = np.random.rand(self.dim, 1) * np.ones(self.num_vecs)
            normal, tc1 = self._gram_schmidt(nc, t1)
            basis = np.hstack([tc1, normal])
        return basis, normal

    @pp.time_logger(sections=module_sections)
    def _gram_schmidt(self, u1, u2, u3=None):
        """
        Perform a Gram Schmidt procedure for the vectors u1, u2 and u3 to obtain a set of
        orhtogonal vectors.

        Parameters:
            u1: ndArray
            u2: ndArray
            u3: ndArray

        Returns:
            u1': ndArray u1 / ||u1||
            u2': ndarray (u2 - u2*u1 * u1) / ||u2||
            u3': (optional) ndArray (u3 - u3*u2' - u3*u1')/||u3||
        """
        u1 = u1 / np.sqrt(np.sum(u1 ** 2, axis=0))

        u2 = u2 - np.sum(u2 * u1, axis=0) * u1
        u2 = u2 / np.sqrt(np.sum(u2 ** 2, axis=0))

        if u3 is None:
            return u1, u2
        u3 = u3 - np.sum(u3 * u1, axis=0) * u1 - np.sum(u3 * u2, axis=0) * u2
        u3 = u3 / np.sqrt(np.sum(u3 ** 2, axis=0))

        return u1, u2, u3

    @pp.time_logger(sections=module_sections)
    def _invert_3d_matrix(self, M):
        """
        Find the inverse of the (m,m,k) 3D ndArray M. The inverse is intrepreted as the
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
