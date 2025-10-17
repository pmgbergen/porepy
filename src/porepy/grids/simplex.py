"""Module containing classes for simplex grids.

.. rubric:: Acknowledgement

The implementation of structured grids is in practice a translation of the corresponding
functions found in the `Matlab Reservoir Simulation Toolbox (MRST)
<www.sintef.no/projectweb/mrst/>`_ developed by SINTEF ICT.

"""

from typing import Optional
from warnings import warn

import numpy as np
import scipy.sparse as sps
import scipy.spatial

import porepy as pp
from porepy.grids.grid import Grid


class TriangleGrid(Grid):
    """Class representation of a general triangular grid.

    If no triangulation is provided, Delaunay will be applied.

    Note:
        Triangular grids are by definition 2D.

    Example:

        >>> p = np.random.rand(2, 10)
        >>> tri = scipy.spatial.Delaunay(p.transpose()).simplices
        >>> g = TriangleGrid(p, tri.transpose())

    Parameters:
        p: ``shape=(2, num_nodes)``

            Cloud of point coordinates.
        tri: ``shape=(3, num_cells), default=None``

            Cell-node connections. If None, a Delaunay triangulation will be applied.
            The ordering of nodes in each cell is assumed to be counter-clockwise.
        name: ``default=None``

            Name of the grid. If None, ``'TriangleGrid'`` will be assigned.

    """

    def __init__(
        self,
        p: np.ndarray,
        tri: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        self.dim = 2

        if tri is None:
            triangulation = scipy.spatial.Delaunay(p.transpose())
            tri = triangulation.simplices
            tri = tri.transpose()

        if name is None:
            name = "TriangleGrid"

        num_nodes = p.shape[1]

        # Add a zero z-coordinate.
        if p.shape[0] == 2:
            nodes = np.vstack((p, np.zeros(num_nodes)))
        else:
            nodes = p

        assert num_nodes > 2  # Check of transposes of point array

        # Tabulate the nodes in [first, second, third] faces of each triangle in
        # counterclockwise order.
        cell_wise_face_nodes = np.hstack(
            (tri[[0, 1]], tri[[1, 2]], tri[[2, 0]])
        ).transpose()

        # The cell-face orientation is positive if it coincides with the face
        # orientation from low to high node index
        cf_data = np.sign(cell_wise_face_nodes[:, 1] - cell_wise_face_nodes[:, 0])

        # Uniquify the face-nodes (match the faces on two neighboring cells). Sort of
        # each row, so that faces with the same nodes but different orientation are
        # recognized as the same face. We cannot use the result from np.unique directly,
        # since the sorted face-nodes will have a different ordering than the original.
        # Hence, get a mapping to the unique faces and construct the face-node relation
        # using this mapping. Also return the mapping back to the original ordering,
        # this will give us the cell-face relation.
        _, face_node_mapping, cell_face_mapping = np.unique(
            np.sort(cell_wise_face_nodes, axis=1),
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        face_nodes = cell_wise_face_nodes[face_node_mapping]

        # Check that the orientation of the faces is consistent, in that the neighboring
        # cells of each face have different signs in the cell-face relation. If not,
        # we flip the sign of the last occurrence of the face in question.
        cf_weights = np.bincount(cell_face_mapping, weights=cf_data)
        inconsistent_orientation = np.where(np.abs(cf_weights) > 1)[0]
        for ind in inconsistent_orientation:
            # In principle, we can flip any of the two occurences of the face. Pick the
            # last one, somewhat arbitrarily, it is not clear to EK that this matters.
            hit = np.where(cell_face_mapping == ind)[0][-1]
            cf_data[hit] *= -1

        num_faces = face_nodes.shape[0]
        num_cells = tri.shape[1]

        num_nodes_per_face = 2
        face_nodes = face_nodes.ravel("C")
        indptr = np.hstack(
            (
                np.arange(0, num_nodes_per_face * num_faces, num_nodes_per_face),
                num_nodes_per_face * num_faces,
            )
        )
        data = np.ones(face_nodes.shape, dtype=bool)
        face_nodes = sps.csc_matrix(
            (data, face_nodes, indptr), shape=(num_nodes, num_faces)
        )

        # Cell-face relation. This can be constructed from the mapping back to the
        # cell-wise face-node relation, recalling that the cell-nodes were stacked so
        # that the faces of all first cells came first, etc.
        num_faces_per_cell = 3
        # Reshape and ravel in Fortran order to get the faces of the first cells first.
        cell_face_indices = cell_face_mapping.reshape(
            num_faces_per_cell, num_cells
        ).ravel("F")
        cf_data = cf_data.reshape(num_faces_per_cell, num_cells).ravel("F").astype(int)
        indptr = np.hstack(
            (
                np.arange(0, num_faces_per_cell * num_cells, num_faces_per_cell),
                num_faces_per_cell * num_cells,
            )
        )
        cell_faces = sps.csc_matrix(
            (cf_data, cell_face_indices, indptr), shape=(num_faces, num_cells)
        )

        super().__init__(2, nodes, face_nodes, cell_faces, name)

    def cell_node_matrix(self) -> np.ndarray:
        """
        Returns:
            An array with ``shape=(num_cells, 3)``, representing the cell-to-node map.

        """
        msg = "This functionality is deprecated and will be removed in a future version"
        warn(msg, DeprecationWarning)

        # Absolute value needed since cellFaces can be negative
        cn = self.face_nodes * np.abs(self.cell_faces) * sps.eye(self.num_cells)
        row, col = cn.nonzero()
        scol = np.argsort(col)

        # Consistency check
        assert np.all(np.bincount(col) == (self.dim + 1))

        return row[scol].reshape(self.num_cells, 3)


class StructuredTriangleGrid(TriangleGrid):
    """Class for a structured triangular grids, composed of squares divided
    into two.

    Example:
        Grid on the unit cube.

        >>> nx = np.array([2, 3])
        >>> physdims = np.ones(2)
        >>> g = simplex.StructuredTriangleGrid(nx, physdims)

    Parameters:
        nx: ``shape=(2,)``

            Number of cells in each direction of the underlying Cartesian grid.
        physdims: ``shape=(2,), default=None``

            Domain size. If None, ``nx`` is used, thus Cartesian cells are unit squares.
        name: ``default=None``

            Name of the grid. If None, ``'StructuredTriangleGrid'`` will be assigned.

    """

    def __init__(
        self,
        nx: np.ndarray,
        physdims: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        nx = np.asarray(nx)
        assert nx.size == 2

        if name is None:
            name = "StructuredTriangleGrid"

        if physdims is None:
            physdims = nx
        else:
            physdims = np.asarray(physdims)
            assert physdims.size == 2

        x = np.linspace(0, physdims[0], nx[0] + 1)
        y = np.linspace(0, physdims[1], nx[1] + 1)

        # Node coordinates
        x_coord, y_coord = np.meshgrid(x, y)
        p = np.vstack((x_coord.ravel(order="C"), y_coord.ravel(order="C")))

        # Define nodes of the first row of cells.
        tmp_ind = np.arange(0, nx[0])
        ind_1 = tmp_ind  # Lower left node in quad
        ind_2 = tmp_ind + 1  # Lower right node
        ind_3 = nx[0] + 2 + tmp_ind  # Upper right node
        ind_4 = nx[0] + 1 + tmp_ind  # Upper left node

        # The first triangle is defined by (i1, i2, i3), the next by (i1, i3, i4). Stack
        # these vertically, and reshape so that the first quad is split into cells 0 and
        # 1 and so on
        tri_base = np.vstack((ind_1, ind_2, ind_3, ind_1, ind_3, ind_4)).reshape(
            (3, -1), order="F"
        )
        # Initialize array of triangles. For the moment, we will append the cells here,
        # but we do know how many cells there are in advance, so pre-allocation is
        # possible if this turns out to be a bottleneck
        tri = tri_base

        # Loop over all remaining rows in the y-direction.
        for iter1 in range(nx[1].astype(int) - 1):
            # The node numbers are increased by nx[0] + 1 for each row
            tri = np.hstack((tri, tri_base + (iter1 + 1) * (nx[0] + 1)))

        super().__init__(p, tri, name=name)


class TetrahedralGrid(Grid):
    """Class for Tetrahedral grids.

    If the cells are not provided, a Delaunay tesselation will be
    constructed.

    Parameters:
        p: ``shape=(3, num_points)``

            Coordinates of vertices.
        tet: ``shape=(4, num_tet), default=None``

            Cell vertices. If None, a Delaunay triangulation will be performed.
        name: ``default=None``

            Name of grid type. If None, ``'TetrahedralGrid'`` will be assigned.

    """

    def __init__(
        self,
        p: np.ndarray,
        tet: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        self.dim = 3

        if tet is None:
            # Transform points to column vector (scipy.Delaunay requires this format).
            tesselation = scipy.spatial.Delaunay(p.transpose())
            tet = tesselation.simplices.transpose()

        if name is None:
            name = "TetrahedralGrid"

        num_nodes = p.shape[1]

        nodes = p
        assert num_nodes > 3  # Check of transposes of point array.

        num_cells = tet.shape[1]

        # As a preparatory step to construct the face-node and cell-face relations,
        # permute the nodes for all tetrahedra. After this step, the nodes in each cell
        # will be ordered so that, if taking the cross product of the vector from node 0
        # to node 1 and the vector from node 0 to node 2, this will point in the
        # opposite direction to the vector from node 0 to node 3 (to see this, read the
        # code in _permute_nodes carefully). In EK's understanding, the point is to get
        # a systematic ordering of the nodes, including a system that lets us deal with
        # the two cells sharing a face in a consistent manner (see construction of the
        # cell-face relation below).
        tet = self._permute_nodes(p, tet)
        # This is apparently needed to appease mypy.
        assert tet is not None

        # Define face-nodes so that the first column contains fn of cell 0, etc. Due to
        # the permutation of the nodes in the previous step, and the order in which the
        # nodes are listed in the definition of face_nodes, the nodes of each face are
        # ordered so that the normal vector formed by the cross product of the vector from
        # node 0 to node 1 and the vector from node 0 to node 2 points points in the
        # same direction as the vector from node 0 to node 3 (to see this, draw an
        # example and verify). This implies that the two cells sharing a face will have
        # that face represented with opposite ordering of the nodes.
        face_nodes = np.vstack(
            (tet[[1, 0, 2]], tet[[0, 1, 3]], tet[[2, 0, 3]], tet[[1, 2, 3]])
        )
        # Reshape face-nodes into a 3x 4*num_cells-matrix, with the four first columns
        # belonging to cell 0.
        face_nodes = face_nodes.reshape((3, 4 * num_cells), order="F")
        sort_ind = np.squeeze(np.argsort(face_nodes, axis=0))

        # Now find the unique face-nodes, by comparing columns in the sorted array.
        # Internal faces will be found twice, once  for ecah cell, while external faces
        # only occur once. The second returned value gives the index of the cells which
        # the face belongs to. Do unique on an array with sorted columns, so that faces
        # with the same nodes but with different ordering are recognized as the same
        # face.
        face_nodes, cell_faces = np.unique(
            np.sort(face_nodes, axis=0), axis=1, return_inverse=True
        )
        # Numpy may return cell-faces as a 2d array, so we need to ravel it.
        cell_faces = cell_faces.ravel(order="F")

        num_faces = face_nodes.shape[1]
        # Construct the face-node relation. Each face has three nodes.
        num_nodes_per_face = 3
        face_nodes = face_nodes.ravel(order="F")
        indptr = np.hstack(
            (
                np.arange(0, num_nodes_per_face * num_faces, num_nodes_per_face),
                num_nodes_per_face * num_faces,
            )
        )
        data = np.ones(face_nodes.shape, dtype=bool)
        face_nodes = sps.csc_matrix(
            (data, face_nodes, indptr), shape=(num_nodes, num_faces)
        )

        # Cell-face relation. Index pointers are straightforward, since we know that
        # each cell has exactly four faces.
        num_faces_per_cell = 4
        indptr = np.hstack(
            (
                np.arange(0, num_faces_per_cell * num_cells, num_faces_per_cell),
                num_faces_per_cell * num_cells,
            )
        )
        # The data should be +1 or -1, depending on the orientation of the face relative
        # to the cell. Due to the ordering of the nodes in each face (see construction
        # above), the orientation is different for the two cells sharing a face. Hence
        # the sorting permutation is different for the two cells, and  it turns out (try
        # and see) that one of the cells will have a cyclic permutation of (0, 1, 2) as
        # the sorting permutation, while the other cell will have a permutation that can
        # be obtained by swapping two elements in (0, 1, 2). The former can be
        # identified by checking if the difference between two consecutive elements in
        # the sorting permutation is 1. For these cells, we set the sign to -1, for the
        # others +1.
        data = np.ones(cell_faces.shape, dtype=int)
        sgn_change = np.where(np.any(np.diff(sort_ind, axis=0) == 1, axis=0))[0]
        data[sgn_change] = -1
        cell_faces = sps.csc_matrix(
            (data, cell_faces, indptr), shape=(num_faces, num_cells)
        )

        super().__init__(3, nodes, face_nodes, cell_faces, name)

    def _permute_nodes(self, p: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Auxiliary function to permute the points in ``p`` according to the tetrahedra
        they belong to, given by ``t``.

        """
        v = self._triple_product(p, t)
        permute = np.where(v > 0)[0]
        if t.ndim == 1:
            if permute[0]:
                t[:2] = t[1::-1]
        else:
            t[:2, permute] = t[1::-1, permute]
        return t

    def _triple_product(self, p: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Auxiliary function to perform the triple product between the first column
        in ``p`` and the remaining columns``.

        Parameters:
            p: ``shape=(3, num_points)``

                Point cloud, where each column represents a point in 3D.
            t: An index array to slice ``p`` column-wise before computing the triple
                product

        Returns:
            An array with ``shape=(3, n)``, where ``n`` is determined by the length of
            ``t`` -1.

        """
        px = p[0]
        py = p[1]
        pz = p[2]

        x = px[t]
        y = py[t]
        z = pz[t]

        dx = x[1:] - x[0]
        dy = y[1:] - y[0]
        dz = z[1:] - z[0]

        cross_x = dy[0] * dz[1] - dy[1] * dz[0]
        cross_y = dz[0] * dx[1] - dz[1] * dx[0]
        cross_z = dx[0] * dy[1] - dx[1] * dy[0]

        return dx[2] * cross_x + dy[2] * cross_y + dz[2] * cross_z


class StructuredTetrahedralGrid(TetrahedralGrid):
    """Class for a structured tetrahedral grids, composed of Cartesian cells divided
    into two.

    Example:
        Grid on the unit cube.

        >>> nx = np.array([2, 3])
        >>> physdims = np.ones(2)
        >>> g = simplex.StructuredTriangleGrid(nx, physdims)

    Parameters:
        nx: ``shape=(2,)``

            Number of cells in each direction of the underlying Cartesian grid.
        physdims: ``shape=(2,), default=None``

            Domain size. If None, ``nx`` is used, thus Cartesian cells are unit cubes.
        name: ``default=None``

            Name of the grid. If None, ``'StructuredTetrahedralGrid'`` will be assigned.

    """

    def __init__(
        self,
        nx: np.ndarray,
        physdims: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        if name is None:
            name = "StructuredTetrahedralGrid"

        nx = np.asarray(nx).astype(int)
        assert nx.size == 3

        if physdims is None:
            physdims = nx
        else:
            physdims = np.asarray(physdims)
            assert physdims.size == 3

        x = np.linspace(0, physdims[0], nx[0] + 1)
        y = np.linspace(0, physdims[1], nx[1] + 1)
        z = np.linspace(0, physdims[2], nx[2] + 1)

        # Node coordinates
        y_coord, x_coord, z_coord = np.meshgrid(y, x, z)
        p = np.vstack(
            (
                x_coord.ravel(order="F"),
                y_coord.ravel(order="F"),
                z_coord.ravel(order="F"),
            )
        )

        # Define nodes of the first row of cells.
        tmp_ind = np.arange(0, nx[0])
        ind_1 = tmp_ind  # Lower left node in quad
        ind_2 = tmp_ind + 1  # Lower right node
        ind_3 = nx[0] + 1 + tmp_ind  # Upper left node
        ind_4 = nx[0] + 2 + tmp_ind  # Upper right node

        nxy = (nx[0] + 1) * (nx[1] + 1)
        ind_5 = ind_1 + nxy
        ind_6 = ind_2 + nxy
        ind_7 = ind_3 + nxy
        ind_8 = ind_4 + nxy

        tet_base = np.vstack(
            (
                ind_1,
                ind_2,
                ind_3,
                ind_5,
                ind_2,
                ind_3,
                ind_5,
                ind_7,
                ind_2,
                ind_5,
                ind_6,
                ind_7,
                ind_2,
                ind_3,
                ind_4,
                ind_7,
                ind_2,
                ind_4,
                ind_6,
                ind_7,
                ind_4,
                ind_6,
                ind_7,
                ind_8,
            )
        ).reshape((4, -1), order="F")
        # Initialize array of triangles. For the moment, we will append the cells here,
        # but we do know how many cells there are in advance, so pre-allocation is
        # possible if this turns out to be a bottleneck.

        # Loop over all remaining rows in the y-direction.
        for iter2 in range(nx[2].astype(int)):
            for iter1 in range(nx[1].astype(int)):
                increment = iter2 * nxy + iter1 * (nx[0] + 1)
                if iter2 == 0 and iter1 == 0:
                    tet = tet_base + increment
                else:
                    tet = np.hstack((tet, tet_base + increment))

        super().__init__(p, tet=tet, name=name)
