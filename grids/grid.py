# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:55:43 2016

@author: keile
"""
from __future__ import division
import numpy as np
import itertools
from enum import Enum
from scipy import sparse as sps

from utils import matrix_compression, half_space, mcolon

from compgeom import basics as cg
from compgeom.sort_points import sort_point_pairs

class FaceTag(np.uint8, Enum):
    NONE = 0
    BOUNDARY = 1
    FRACTURE = 2
#   NEXT = 4
    WHOLE = np.iinfo(type(NONE)).max

class Grid(object):
    """
    Parent class for all grids.

    The grid stores topological information, as well as geometric
    information (after a call to self.compute_geometry().

    As of yet, there is no structure for tags (face or cell) is the grid.
    This will be introduced later.

    Attributes:
        Comes in two classes. Topologogical information, defined at
        construction time:

        dim (int): dimension. Should be 0 or 1 or 2 or 3
        nodes (np.ndarray): node coordinates. size: dim x num_nodes
        face_nodes (sps.csc-matrix): Face-node relationships. Matrix size:
            num_faces x num_cells. To use compute_geometry() later, he field
            face_nodes.indices should store the nodes of each face sorted.
        cell_faces (sps.csc-matrix): Cell-face relationships. Matrix size:
            num_faces x num_cells. Matrix elements have value +-1, where +
            corresponds to the face normal vector being outwards.
        name (list): Information on the formation of the grid, such as the
            constructor, computations of geometry etc.
        num_nodes (int): Number of nodes in the grid
        num_faces (int): Number of faces in the grid
        num_cells (int): Number of cells in the grid

        ---
        Geometric information, available after compute_geometry() has been
        called on the object:

        face_areas (np.ndarray): Areas of all faces
        face_centers (np.ndarray): Centers of all faces. Dimensions dim x
            num_faces
        face_normals (np.ndarray): Normal vectors of all faces. Dimensions
            dim x num_faces. See also cell_faces.
        cell_centers (np.ndarray): Centers of all cells. Dimensions dim x
            num_cells
        cell_volumes (np.ndarray): Volumes of all cells

    """

    def __init__(self, dim, nodes, face_nodes, cell_faces, name):
        """Initialize the grid

        See class documentation for further description of parameters.

        Parameters
        ----------
        dim (int): grid dimension
        nodes (np.ndarray): node coordinates.
        face_nodes (sps.csc_matrix): Face-node relations.
        cell_faces (sps.csc_matrix): Cell-face relations
        name (str): Name of grid
        """
        assert dim>=0 and dim<=3
        self.dim = dim
        self.nodes = nodes
        self.cell_faces = cell_faces
        self.face_nodes = face_nodes

        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]

        # Infer bookkeeping from size of parameters
        self.num_nodes = nodes.shape[1]
        self.num_faces = face_nodes.shape[1]
        self.num_cells = cell_faces.shape[1]

        # Add tag for the boundary faces
        self.face_tags = np.tile(FaceTag.NONE, self.num_faces)
        self.update_boundary_face_tag()

    def copy(self):
        """
        Create a deep copy of the grid.

        Returns:
            grid: A deep copy of self

        """
        h = Grid(self.dim, self.nodes.copy(), self.face_nodes.copy(),
                 self.cell_faces.copy(), self.name)
        if hasattr(self, 'cell_volumes'):
            h.cell_volumes = self.cell_volumes.copy()
        if hasattr(self, 'cell_centers'):
            h.cell_centers = self.cell_centers.copy()
        if hasattr(self, 'face_centers'):
            h.face_centers = self.face_centers.copy()
        if hasattr(self, 'face_normals'):
            h.face_normals = self.face_normals.copy()
        if hasattr(self, 'face_areas'):
            h.face_areas = self.face_areas.copy()
        if hasattr(self, 'face_tags'):
            h.face_tags = self.face_tags.copy()
        return h

    def __repr__(self):
        """
        Implementation of __repr__

        NOTE: Should be revisited - not sure if this is the optimal information
        to provide here.
        """
        s = 'Grid with history ' + ', '.join(self.name) + '\n'
        s = s + 'Number of cells ' + str(self.num_cells) + '\n'
        s = s + 'Number of faces ' + str(self.num_faces) + '\n'
        s = s + 'Number of nodes ' + str(self.num_nodes) + '\n'
        return s


    def compute_geometry(self, is_embedded=False, is_starshaped=False):
        """Compute geometric quantities for the grid.

        This method initializes class variables describing the grid
        geometry, see class documentation for details.

        The method could have been called from the constructor, however,
        in cases where the grid is modified after the initial construction (
        say, grid refinement), this may lead to costly, unnecessary
        computations.
        """

        self.name.append('Compute geometry')

        if self.dim == 0:
            self.__compute_geometry_0d()
        elif self.dim == 1:
            self.__compute_geometry_1d()
        elif self.dim == 2:
            self.__compute_geometry_2d(is_embedded, is_starshaped)
        else:
            self.__compute_geometry_3d()

    def __compute_geometry_0d(self):
        "Compute 0D geometry"
        self.face_areas = np.ones(1)
        self.face_centers = self.nodes
        self.face_normals = np.zeros(3) # not well-defined

        self.cell_volumes = np.ones(1)
        self.cell_centers = self.nodes

    def __compute_geometry_1d(self):
        "Compute 1D geometry"

        xn = self.nodes

        self.face_areas = np.ones(self.num_faces)

        fn = self.face_nodes.indices
        n = fn.size
        self.face_centers = xn[:,fn]

        self.face_normals = np.tile(cg.compute_tangent(xn), (n,1)).T

        cf = self.cell_faces.indices
        xf1 = self.face_centers[:, cf[::2]]
        xf2 = self.face_centers[:, cf[1::2]]

        self.cell_volumes = np.linalg.norm(xf1 - xf2, axis=0)
        self.cell_centers = 0.5*(xf1 + xf2)

        # Ensure that normal vector direction corresponds with sign convention
        # in self.cellFaces
        def nrm(u):
            return np.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])

        [fi, ci, val] = sps.find(self.cell_faces)
        _, idx = np.unique(fi, return_index=True)
        sgn = val[idx]
        fc = self.face_centers[:, fi[idx]]
        cc = self.cell_centers[:, ci[idx]]
        v = fc - cc
        # Prolong the vector from cell to face center in the direction of the
        # normal vector. If the prolonged vector is shorter, the normal should
        # flipped
        vn = v + nrm(v) * self.face_normals[:, fi[idx]] * 0.001
        flip = np.logical_or(np.logical_and(nrm(v) > nrm(vn), sgn > 0),
                             np.logical_and(nrm(v) < nrm(vn), sgn < 0))
        self.face_normals[:, flip] *= -1

    def __compute_geometry_2d(self, is_embedded, is_starshaped):
        "Compute 2D geometry, with method motivated by similar MRST function"

        xn = self.nodes

        if is_embedded:
            R = cg.project_plane_matrix(xn)
            xn = np.dot(R, xn)

        fn = self.face_nodes.indices
        edge1 = fn[::2]
        edge2 = fn[1::2]

        xe1 = xn[:, edge1]
        xe2 = xn[:, edge2]

        edge_length_x = xe2[0] - xe1[0]
        edge_length_y = xe2[1] - xe1[1]
        edge_length_z = xe2[2] - xe1[2]
        self.face_areas = np.sqrt(np.power(edge_length_x, 2) +
                                  np.power(edge_length_y, 2) +
                                  np.power(edge_length_z, 2))
        self.face_centers = 0.5 * (xe1 + xe2)
        n = edge_length_z.shape[0]
        self.face_normals = np.vstack((edge_length_y, -edge_length_x, np.zeros(n)))

        cell_faces, cellno = self.cell_faces.nonzero()
        num_cell_faces = np.bincount(cellno)

        cx = np.bincount(cellno, weights=self.face_centers[0, cell_faces])
        cy = np.bincount(cellno, weights=self.face_centers[1, cell_faces])
        cz = np.bincount(cellno, weights=self.face_centers[2, cell_faces])
        cell_centers = np.vstack((cx, cy, cz)) / num_cell_faces

        a = xe1[:, cell_faces] - cell_centers[:, cellno]
        b = xe2[:, cell_faces] - cell_centers[:, cellno]

        sub_volumes = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
        self.cell_volumes = np.bincount(cellno, weights=sub_volumes)

        sub_centroids = (cell_centers[:, cellno] + 2 *
                         self.face_centers[:, cell_faces]) / 3

        ccx = np.bincount(cellno, weights=sub_volumes * sub_centroids[0])
        ccy = np.bincount(cellno, weights=sub_volumes * sub_centroids[1])
        ccz = np.bincount(cellno, weights=sub_volumes * sub_centroids[2])

        self.cell_centers = np.vstack((ccx, ccy, ccz)) / self.cell_volumes

        if is_starshaped:
            faces, _, _ = sps.find(self.cell_faces)
            nodes, _, _ = sps.find(self.face_nodes)

            for c in np.arange(self.num_cells):
                loc = slice(self.cell_faces.indptr[c],
                            self.cell_faces.indptr[c+1])
                faces_loc = faces[loc]
                loc_n = self.face_nodes.indptr[faces_loc]
                ordering = sort_point_pairs(np.array([nodes[loc_n],
                            nodes[loc_n+1]]), ordering=True)[1].astype(np.float)
                normal = np.multiply( 1-2.*ordering, np.divide(
                    self.face_normals[:,faces_loc], self.face_areas[faces_loc]))

                x0 = xn[:,nodes[loc_n]]
                x1 = xn[:,nodes[loc_n+1]]
                coords = np.concatenate((x0,x1),axis=1)
                if not np.all(half_space.half_space_int(normal, x0, coords)):
                    center = half_space.half_space_pt(normal, x0, coords)
                    center_tile = np.tile(center, (faces_loc.size,1)).T
                    a = xe1[:,faces_loc] - center_tile
                    b = xe2[:,faces_loc] - center_tile
                    sub_volumes = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
                    self.cell_centers[:,c] = center
                    self.cell_volumes[c] = np.sum(sub_volumes)

        # Ensure that normal vector direction corresponds with sign convention
        # in self.cellFaces
        def nrm(u):
            return np.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])

        [fi, ci, val] = sps.find(self.cell_faces)
        _, idx = np.unique(fi, return_index=True)
        sgn = val[idx]
        fc = self.face_centers[:, fi[idx]]
        cc = self.cell_centers[:, ci[idx]]
        v = fc - cc
        # Prolong the vector from cell to face center in the direction of the
        # normal vector. If the prolonged vector is shorter, the normal should
        # flipped
        vn = v + nrm(v) * self.face_normals[:, fi[idx]] * 0.001
        flip = np.logical_or(np.logical_and(nrm(v) > nrm(vn), sgn > 0),
                             np.logical_and(nrm(v) < nrm(vn), sgn < 0))
        self.face_normals[:, flip] *= -1

        if is_embedded:
            self.face_normals = np.dot(R.T, self.face_normals)
            self.face_centers = np.dot(R.T, self.face_centers)
            self.cell_centers = np.dot(R.T, self.cell_centers)

    def __compute_geometry_3d(self):
        """
        Helper function to compute geometry for 3D grids

        The implementation is motivated by the similar MRST function.

        NOTE: The function is very long, and could have been broken up into
        parts (face and cell computations are an obvious solution).

        """
        print('Not finished yet. Use with caution')
        xn = self.nodes
        num_face_nodes = self.face_nodes.nnz
        face_node_ptr = self.face_nodes.indptr

        num_nodes_per_face = face_node_ptr[1:] - face_node_ptr[:-1]

        # Face-node relationships. Note that the elements here will also
        # serve as a representation of an edge along the face (face_nodes[i]
        #  represents the edge running from face_nodes[i] to face_nodes[i+1])
        face_nodes = self.face_nodes.indices
        # For each node, index of its parent face
        face_node_ind = matrix_compression.rldecode(np.arange(
            self.num_faces), num_nodes_per_face)

        # Index of next node on the edge list. Note that this assumes the
        # elements in face_nodes is stored in an ordered fasion
        next_node = np.arange(num_face_nodes) + 1
        # Close loops, for face i, the next node is the first of face i
        next_node[face_node_ptr[1:]-1] = face_node_ptr[:-1]

        # Mapping from cells to faces
        edge_2_face = sps.coo_matrix((np.ones(num_face_nodes),
                                      (np.arange(num_face_nodes),
                                       face_node_ind))).tocsc()

        # Define temporary face center as the mean of the face nodes
        tmp_face_center = xn[:, face_nodes] * edge_2_face / num_nodes_per_face
        # Associate this value with all the edge of this face
        tmp_face_center = edge_2_face * tmp_face_center.transpose()

        # Vector along each edge
        along_edge = xn[:, face_nodes[next_node]] - xn[:, face_nodes]
        # Vector from face center to start node of each edge
        face_2_node = tmp_face_center.transpose() - xn[:, face_nodes]

        # Assign a normal vector with this edge, by taking the cross product
        # between face_2_node and along_edge
        # Divide by two to ensure that the normal vector has length equal to
        # the area of the face triangle (by properties of cross product)
        sub_normals = np.vstack((along_edge[1] * face_2_node[2] -
                                 along_edge[2] * face_2_node[1],
                                 along_edge[2] * face_2_node[0] -
                                 along_edge[0] * face_2_node[2],
                                 along_edge[0] * face_2_node[1] -
                                 along_edge[1] * face_2_node[0],
                                 )) / 2

        def nrm(v):
            return np.sqrt(np.sum(v*v, axis=0))

        # Calculate area of sub-face associated with each edge - note that
        # the sub-normals are area weighted
        sub_areas = nrm(sub_normals)

        # Centers of sub-faces are given by the centroid coordinates,
        # e.g. the mean coordinate of the edge endpoints and the temporary
        # face center
        sub_centroids = (xn[:, face_nodes] + xn[:, face_nodes[next_node]]
                         + tmp_face_center.transpose()) / 3

        # Face normals are given as the sum of the sub-components
        face_normals = sub_normals * edge_2_face
        # Similar with face areas
        face_areas = edge_2_face.transpose() * sub_areas

        # Test whether the sub-normals are pointing in the same direction as
        # the main normal: Distribute the main normal onto the edges,
        # and take scalar product by element-wise multiplication with
        # sub-normals, and sum over the components (axis=0).
        # NOTE: There should be a built-in function for this in numpy?
        sub_normals_sign = np.sign(np.sum(sub_normals * (edge_2_face *
                                          face_normals.transpose()).transpose(),
                                          axis=0))

        # Finally, face centers are the area weighted means of centroids of
        # the sub-faces
        face_centers = sub_areas * sub_centroids * edge_2_face / face_areas

        # .. and we're done with the faces. Store information
        self.face_centers = face_centers
        self.face_normals = face_normals
        self.face_areas = face_areas

        # Cells

        # Temporary cell center coordinates as the mean of the face center
        # coordinates. The cells are divided into sub-tetrahedra (
        # corresponding to triangular sub-faces above), with the temporary
        # cell center as the final node

        # Mapping from edges to cells. Take absolute value of cell_faces,
        # since the elementsn are signed (contains the divergence)
        # Note that edge_2_cell will cotain more elements than edge_2_face,
        # since the former will count internal faces twice (one for each
        # adjacent cell)
        edge_2_cell = edge_2_face * np.abs(self.cell_faces)
        # Sort indices to avoid messing up the mappingsn later
        edge_2_cell.sort_indices()

        # Obtain relations between edges, faces and cells, in the form of
        # index lists. Each element in the list corresponds to an edge seen
        # from a cell (e.g. edges on internal faces are seen twice).

        # Cell numbers are obtained from the columns in edge_2_cell.
        cell_numbers = matrix_compression.rldecode(np.arange(self.num_cells),
                                                   np.diff(edge_2_cell.indptr))
        # Edge numbers from the rows. Here it is crucial that the indices
        # are sorted
        edge_numbers = edge_2_cell.indices
        # Face numbers are obtained from the face-node relations (with the
        # nodes doubling as representation of edges)
        face_numbers = face_node_ind[edge_numbers]

        # Number of edges per cell
        num_cell_edges = edge_2_cell.indptr[1:] - edge_2_cell.indptr[:-1]

        def bincount_nd(arr, weights):
            """ Utility function to sum vector quantities by np.bincount. We
            could probably have used np.apply_along_axis, but I could not
            make it work.

            Intended use: Map sub-cell centroids to a quantity for the cell.
            """
            dim = weights.shape[0]
            sz = arr.max() + 1

            count = np.zeros((dim, sz))
            for iter1 in range(dim):
                count[iter1] = np.bincount(arr, weights=weights[iter1],
                                           minlength=sz)
            return count

        # First estimate of cell centers as the mean of its faces' centers
        # Divide by num_cell_edges here since all edges bring in their faces
        tmp_cell_centers = bincount_nd(cell_numbers,
                                       face_centers[:, face_numbers]
                                       / num_cell_edges[cell_numbers])

        # Distance from the temporary cell center to the sub-centroids (of
        # the tetrahedra associated with each edge)
        dist_cellcenter_subface = sub_centroids[:, edge_numbers] \
                                  - tmp_cell_centers[:, cell_numbers]

        # Get sign of normal vectors, seen from all faces.
        # Make sure we get a numpy ndarray, and not a matrix (.A), and that
        # the array is 1D (squeeze)
        orientation = np.squeeze(self.cell_faces[face_numbers, cell_numbers].A)

        # Get outwards pointing sub-normals for all sub-faces: We need to
        # account for both the orientation of the face, and the orientation
        # of sub-faces relative to faces.
        outer_normals = sub_normals[:, edge_numbers] \
                        * orientation * sub_normals_sign[edge_numbers]

        # Volumes of tetrahedra are now given by the dot product between the
        #  outer normal (which is area weighted, and thus represent the base
        #  of the tet), with the distancance from temporary cell center (the
        # dot product gives the hight).
        tet_volumes = np.sum(dist_cellcenter_subface * outer_normals,
                             axis=0) / 3
        assert np.all(tet_volumes > 0)  # On the fly test
        # The cell volumes are now found by summing sub-tetrahedra
        cell_volumes = np.bincount(cell_numbers, weights=tet_volumes)
        tri_centroids = 3 / 4 * dist_cellcenter_subface

        # Compute a correction to the temporary cell center, by a volume
        # weighted sum of the sub-tetrahedra
        rel_centroid = bincount_nd(cell_numbers, tet_volumes * tri_centroids) \
                       / cell_volumes
        cell_centers = tmp_cell_centers + rel_centroid

        # ... and we're done
        self.cell_centers = cell_centers
        self.cell_volumes = cell_volumes

    def cell_nodes(self):
        mat = (self.face_nodes * np.abs(self.cell_faces) *
               sps.eye(self.num_cells)) > 0
        return mat

    def num_cell_nodes(self):
        return self.cell_nodes().sum(axis=0).A.ravel(1)


    def get_internal_nodes(self):
        """
        Get internal nodes id of the grid.

        Returns:
            np.ndarray (1D), index of internal nodes.

        """
        return np.setdiff1d(np.arange(self.num_nodes), self.get_boundary_nodes(),
                            assume_unique=True)


    def get_internal_faces(self):
        """
        Get internal faces id of the grid

        Returns:
            np.ndarray (1d), index of internal faces.

        """
        return self.indices(self.has_not_face_tag(FaceTag.BOUNDARY))

    def get_boundary_faces(self):
        """
        Get boundary faces id of the grid

        Returns:
            np.ndarray (1d), index of boundary faces

        """
        return self.indices(self.has_face_tag(FaceTag.BOUNDARY))

    def get_boundary_nodes(self):
        """
        Get nodes on the boundary

        Returns:
            np.ndarray (1d), index of nodes on the boundary

        """
        b_faces = self.get_boundary_faces()
        first = self.face_nodes.indptr[b_faces]
        second = self.face_nodes.indptr[b_faces+1]-1
        return np.unique(self.face_nodes.indices[mcolon.mcolon(first, second)])

    def update_boundary_face_tag(self):
        bd_faces = np.argwhere(np.abs(self.cell_faces).sum(axis=1).A.ravel(1)
                                                                  == 1).ravel(1)
        self.add_face_tag(bd_faces, FaceTag.BOUNDARY)

    def cell_diameters(self, cn = None):
        """
        Compute the cell diameters.

        Parameters:
            cn (optional): cell nodes map, previously already computed.
            Otherwise a call to self.cell_nodes is provided.

        Returns:
            np.array, num_cells: values of the cell diameter for each cell
        """

        def comb( n ): return np.fromiter(itertools.chain.from_iterable( \
                                          itertools.combinations(n,2)), \
                                          n.dtype).reshape((2,-1), order='F')

        def diam( n ): return np.amax( np.linalg.norm( self.nodes[:, n[0,:]] - \
                                                       self.nodes[:, n[1,:]], \
                                                       axis=0 ) )

        if cn is None: cn = self.cell_nodes()
        return np.array([ diam(comb(cn.indices[cn.indptr[c]:cn.indptr[c+1]])) \
                                          for c in np.arange(self.num_cells) ] )


    def cell_face_as_dense(self):
        """
        Obtain the cell-face relation in the from of two rows, rather than a
        sparse matrix. This alterative format can be useful in some cases.

        Each column in the array corresponds to a face, and the elements in
        that column refers to cell indices. The value -1 signifies a boundary.
        The normal vector of the face points from the first to the second row.

        Returns:
            np.ndarray, 2 x num_faces: Array representation of face-cell
                relations
        """
        n = self.cell_faces.tocsr()
        d = np.diff(n.indptr)
        rows = matrix_compression.rldecode(np.arange(d.size), d)
        # Increase the data by one to distinguish cell indices from boundary
        # cells
        data = n.indices + 1
        cols = ((n.data + 1)/2).astype('i')
        neighs = sps.coo_matrix((data, (rows, cols))).todense()
        # Subtract 1 to get back to real cell indices
        neighs -= 1
        neighs = neighs.transpose().A.astype('int')
        # Finally, we need to switch order of rows to get normal vectors
        # pointing from first to second row.
        return neighs[::-1]

    def cell_connection_map(self):
        """
        Get a matrix representation of cell-cell connections, as defined by
        two cells sharing a face.

        Returns:
            scipy.sparse.csr_matrix, size num_cells * num_cells: Boolean
                matrix, element (i,j) is true if cells i and j share a face.
                The matrix is thus symmetric.
        """

        # Create a copy of the cell-face relation, so that we can modify it at will
        cell_faces = self.cell_faces.copy()

        # Direction of normal vector does not matter here, only 0s and 1s
        cell_faces.data = np.abs(cell_faces.data)

        # Find connection between cells via the cell-face map
        c2c = cell_faces.transpose() * cell_faces
        # Only care about absolute values
        c2c.data = np.clip(c2c.data, 0, 1).astype('bool')

        return c2c

    def add_face_tag(self, f, tag):
        self.face_tags[f] = np.bitwise_or(self.face_tags[f], tag)

    def remove_face_tag(self, f, tag):
        self.face_tags[f] = np.bitwise_and(self.face_tags[f], np.bitwise_not(tag))

    def has_face_tag(self, tag):
        return np.bitwise_and(self.face_tags, tag).astype(np.bool)

    def has_not_face_tag(self, tag):
        return self.has_face_tag(np.bitwise_not(tag))

    def has_only_face_tag(self, tag):
        return self.face_tags == tag

    def indices(self, true_false):
        return np.argwhere(true_false).ravel(1)
