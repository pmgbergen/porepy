# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:55:43 2016

@author: keile
"""

import numpy as np
from enum import Enum
from scipy import sparse as sps

from utils import matrix_compression


class GridType(Enum):
    """
    Enumeration to define types of grids. Not quite sure what I want to use
    them for; right now the primary motivation is to test which type of grid
    this is.
    Possible future usage could assign dimension to the fields, etc.
    """
    triangle = 1
    cartesian_2D = 2
    tensor_2D = 3


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

        dim (int): dimension. Should be 2 or 3
        nodes (np.ndarray): node coordinates. size: dim x num_nodes
        face_nodes (sps.csc-matrix): Face-node relationships. Matrix size:
            num_faces x num_cells.
        cell_faces (sps.csc-matrix): Cell-face relationships. Matrix size:
            num_faces x num_cells. Matrix elements have value +-1, where +
            corresponds to the face normal vector being outwards.
        name (str): Placeholder field to give information on the grid. Will
            be changed to something meaningful in the future
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
        self.dim = dim
        self.nodes = nodes
        self.cell_faces = cell_faces
        self.face_nodes = face_nodes
        self.name = name

        # Infer bookkeeping from size of parameters
        self.num_nodes = nodes.shape[1]
        self.num_faces = face_nodes.shape[1]
        self.num_cells = cell_faces.shape[1]

    def compute_geometry(self):
        """Compute geometric quantities for the grid.

        This method initializes class variables describing the grid
        geometry, see class documentation for details.

        The method could have been called from the constructor, however,
        in cases where the grid is modified after the initial construction (
        say, grid refinement), this may lead to costly, unnecessary
        computations.
        """
        if self.dim == 2:
            self.__compute_geometry_2d()
        else:
            self.__compute_geometry_3d()

    def __compute_geometry_2d(self):
        "Compute 2D geometry, with method motivated by similar MRST function"

        xn = self.nodes

        fn = self.face_nodes.indices
        edge1 = fn[::2]
        edge2 = fn[1::2]

        xe1 = xn[:, edge1]
        xe2 = xn[:, edge2]

        edge_length_x = xe2[0] - xe1[0]
        edge_length_y = xe2[1] - xe1[1]
        self.face_areas = np.sqrt(np.power(edge_length_x, 2) +
                                  np.power(edge_length_y, 2))
        self.face_centers = 0.5 * (xe1 + xe2)
        self.face_normals = np.vstack((edge_length_y, -edge_length_x))

        cell_faces, cellno = self.cell_faces.nonzero()

        num_cell_faces = np.bincount(cellno)

        cx = np.bincount(cellno, weights=self.face_centers[0, cell_faces])
        cy = np.bincount(cellno, weights=self.face_centers[1, cell_faces])
        cell_centers = np.vstack((cx, cy)) / num_cell_faces

        a = xe1[:, cell_faces] - cell_centers[:, cellno]
        b = xe2[:, cell_faces] - cell_centers[:, cellno]

        sub_volumes = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
        self.cell_volumes = np.bincount(cellno, weights=sub_volumes)
        
        sub_centroids = (cell_centers[:, cellno] + 2 *
                         self.face_centers[:, cell_faces]) / 3

        ccx = np.bincount(cellno, weights=sub_volumes * sub_centroids[0])
        ccy = np.bincount(cellno, weights=sub_volumes * sub_centroids[1])

        self.cell_centers = np.vstack((ccx, ccy)) / self.cell_volumes

        # Ensure that normal vector direction corresponds with sign convention
        # in self.cellFaces
        def nrm(u):
            return np.sqrt(u[0]*u[0] + u[1]*u[1])

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

    def __compute_geometry_3d(self):
        """
        Helper function to compute geometry for 3D grids

        The implementation is motivated by the similar MRST function
        """
        print('Not finished yet. Use with caution')
        xn = self.nodes
        num_face_nodes = self.face_nodes.nnz
        face_node_ptr = self.face_nodes.indptr

        num_nodes_per_face = face_node_ptr[1:] - face_node_ptr[:-1]

        face_nodes = self.face_nodes.indices
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

        # Calculate area of sub-face associated with each edge
        sub_areas = nrm(sub_normals)
        assert np.all(sub_areas > 0)  # This really cannot fail because of
        # properties of the norm

        # Centers of sub-faces are given by the centroid coordinates,
        # e.g. the mean coordinate of the edge endpoints and the temporary
        # face center
        sub_centroids = (xn[:, face_nodes] + xn[:, face_nodes[next_node]]
                         + tmp_face_center.transpose()) / 3
        # Face normals are given as the sum of the sub-components
        face_normals = edge_2_face.transpose() * sub_normals.transpose()
        # Similar with face areas
        face_areas = edge_2_face.transpose() * sub_areas

        # Consistency check - this will only work if all sub-normals are
        # pointing in the same direction, but this they should. Maybe this
        # is a test of the ordering of the nodes in face_nodes?
        # assert np.isclose(nrm(face_normals), face_areas).all()

        sub_normals_sign = np.sign(np.sum(sub_normals * (edge_2_face *
                                       face_normals).transpose(), axis=0))

        # Finally, face centers are the area weighted means of centroids of
        # the sub-faces
        face_centers = sub_areas * sub_centroids * edge_2_face / face_areas

        # .. and we're done with the faces. Store information
        self.face_centers = face_centers
        self.face_normals = face_normals
        self.face_areas = face_areas

        # Cells
        cell_volumes = np.zeros(self.num_cells)
        cell_centers = np.zeros((self.dim, self.num_cells))

        num_edges = edge_2_face.shape[0]
        cell_face_mat = np.abs(self.cell_faces)
        faceptr = self.cell_faces.indptr
        cell_faces = self.cell_faces.indices
        num_cell_faces = faceptr[1:] - faceptr[:-1]

        # Temporary cell center coordinates as the mean of the face center
        # coordinates. The cells are divided into sub-tetrahedra (
        # corresponding to triangular sub-faces above), with the temporary
        # cell center as the final node
        tmp_cell_centers = (face_centers * cell_face_mat) / num_cell_faces

        def vec_2_diag_mat(vec):
            num_elem = vec.shape[0]
            return sps.dia_matrix((vec, 0), shape=(num_elem, num_elem))

        # Vectors from
        cx_edge = edge_2_face * cell_face_mat * \
            sps.dia_matrix((tmp_cell_centers[0], 0),
                           shape=(self.num_cells, self.num_cells))
        cy_edge = edge_2_face * cell_face_mat * \
            sps.dia_matrix((tmp_cell_centers[1], 0),
                           shape=(self.num_cells, self.num_cells))
        cz_edge = edge_2_face * cell_face_mat * \
            sps.dia_matrix((tmp_cell_centers[2], 0),
                           shape=(self.num_cells, self.num_cells))

        edge_2_cell = edge_2_face * cell_face_mat

        sub_centroids_cellwise_x = edge_2_cell.transpose() * \
            sps.dia_matrix((sub_centroids[0], 0), shape=(num_edges, num_edges))
        sub_centroids_cellwise_y = edge_2_cell.transpose() * \
            sps.dia_matrix((sub_centroids[1], 0), shape=(num_edges, num_edges))
        sub_centroids_cellwise_z = edge_2_cell.transpose() * \
            sps.dia_matrix((sub_centroids[2], 0), shape=(num_edges, num_edges))

        orient = edge_2_face * self.cell_faces * vec_2_diag_mat(np.ones(
            self.num_cells))
        # Something is wrong with orientation - Sunday night
        orientation = edge_2_face * self.cell_faces * np.ones(self.num_cells)
            # sps.dia_matrix((np.ones(self.num_cells), 0),
            #                shape=(self.num_cells, self.num_cells))
        outer_normals_x = sub_normals[0] * orientation
        outer_normals_y = sub_normals[1] * orientation
        outer_normals_z = sub_normals[2] * orientation

        on_x = orient.transpose() * vec_2_diag_mat(sub_normals[0])
        on_y = orient.transpose() * vec_2_diag_mat(sub_normals[1])
        on_z = orient.transpose() * vec_2_diag_mat(sub_normals[2])




        cell_center_2_edge_x = sub_centroids_cellwise_x.transpose() - cx_edge
        cell_center_2_edge_y = sub_centroids_cellwise_y.transpose() - cy_edge
        cell_center_2_edge_z = sub_centroids_cellwise_z.transpose() - cz_edge

        tvx = on_x * cell_center_2_edge_x
        tvy = on_y * cell_center_2_edge_y
        tvz = on_z * cell_center_2_edge_z

        tet_volumes = (cell_center_2_edge_x.transpose() * outer_normals_x
                       + cell_center_2_edge_y.transpose() * outer_normals_y
                       + cell_center_2_edge_z.transpose() * outer_normals_z
                       ) / 3
        tet_centers = 3/4 * np.vstack((cell_center_2_edge_x.sum(axis=1),
                                       cell_center_2_edge_y.sum(axis=1),
                                       cell_center_2_edge_z.sum(axis=1)
                                       ))

        # Accumulate edge quantities to faces by np.bincount,
        # using edge_2_mat.indices as weights, but not sure if we should
        # sort them first


        # dist_subcentroid_tmpcc_x = edge_cell_mask.multiply((tmp_cell_centers[
        #                                                       0] - cx_edge))
        # dist_subcentroid_tmpcc_y = edge_cell_mask.multiply((tmp_cell_centers[
        #                                                       1] - cy_edge))
        # dist_subcentroid_tmpcc_z = edge_cell_mask.multiply((tmp_cell_centers[
        #                                                       2] - cz_edge))
        # dist_subcentroid_tmpcc_y = tmp_cell_centers[1] - cy_edge
        # dist_subcentroid_tmpcc_z = tmp_cell_centers[2] - cz_edge

        e2c = edge_2_cell * vec_2_diag_mat(np.ones(self.num_cells))

        cell_numbers = matrix_compression.rldecode(np.arange(self.num_cells),
                                                   np.diff(e2c.indptr))
        edge_numbers = e2c.indices
        face_numbers = face_node_ind[edge_numbers]

        num_cell_edges = edge_2_cell.indptr[1:] - edge_2_cell.indptr[:-1]

        def bincount_nd(arr, weights=None):
            dim = weights.shape[0]
            sz = arr.max() + 1

            if weights is None:
                weights = np.ones((dim, arr.shape[1]))

            count = np.zeros((dim, sz))
            for iter1 in range(dim):
                count[iter1] = np.bincount(arr, weights=weights[iter1],
                                           minlength=sz)
            return count

        tmp_cell_centers = bincount_nd(cell_numbers,
                                       weights=face_centers[:, face_numbers]
                                               / num_cell_edges[cell_numbers])

        dist_cell_edge = sub_centroids[:, edge_numbers] - \
                         tmp_cell_centers[:, cell_numbers]

        orientation = self.cell_faces[face_numbers, cell_numbers].A

        outer_normals = sub_normals[:, edge_numbers] * \
            np.squeeze(orientation * sub_normals_sign[edge_numbers])

        tri_volumes = np.sum(dist_cell_edge * outer_normals, axis=0) / 3
        assert np.all(tri_volumes > 0)
        cell_volumes = np.bincount(cell_numbers, weights=tri_volumes)
        tri_centroids = 3 / 4 * dist_cell_edge

        rel_centroid = bincount_nd(cell_numbers,
                                   weights=tri_volumes * tri_centroids) / \
                       cell_volumes
        cell_centers = tmp_cell_centers + rel_centroid

        self.cell_centers = cell_centers
        self.cell_volumes = cell_volumes

        a = 2


    def cell_nodes(self):
        mat = (self.face_nodes * np.abs(self.cell_faces) *
               sps.eye(self.num_cells)) > 0
        return mat

    def num_cell_nodes(self):
        return self.cell_nodes().sum(axis=0).A.ravel(1)
