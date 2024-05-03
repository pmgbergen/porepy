"""Module containing classes for simplex grids.

.. rubric:: Acknowledgement

The implementation of structured grids is in practice a translation of the corresponding
functions found in the `Matlab Reservoir Simulation Toolbox (MRST)
<www.sintef.no/projectweb/mrst/>`_ developed by SINTEF ICT.

"""

from typing import Optional

import numpy as np
import scipy.sparse as sps
import scipy.spatial.qhull

from porepy.grids.grid import Grid
from porepy.utils import accumarray, setmembership


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
            triangulation = scipy.spatial.qhull.Delaunay(p.transpose())
            tri = triangulation.simplices
            tri = tri.transpose()

        if name is None:
            name = "TriangleGrid"

        num_nodes = p.shape[1]

        # Add a zero z-coordinate
        if p.shape[0] == 2:
            nodes = np.vstack((p, np.zeros(num_nodes)))
        else:
            nodes = p

        assert num_nodes > 2  # Check of transposes of point array

        # Tabulate the nodes in [first, second, third] faces of each triangle in
        # counterclockwise order
        face_nodes = np.hstack((tri[[0, 1]], tri[[1, 2]], tri[[2, 0]])).transpose()

        # The cell-face orientation is positive if it coincides with the face
        # orientation from low to high node index
        cf_data = np.sign(face_nodes[:, 1] - face_nodes[:, 0])

        # Face node relations. Each face is oriented from low to high node index.
        face_nodes.sort(axis=1)
        face_nodes, _, cell_faces = setmembership.unique_rows(face_nodes)

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

        # Cell face relation
        num_faces_per_cell = 3
        cell_faces = cell_faces.reshape(num_faces_per_cell, num_cells).ravel("F")
        cf_data = cf_data.reshape(num_faces_per_cell, num_cells).ravel("F").astype(int)

        indptr = np.hstack(
            (
                np.arange(0, num_faces_per_cell * num_cells, num_faces_per_cell),
                num_faces_per_cell * num_cells,
            )
        )
        cell_faces = sps.csc_matrix(
            (cf_data, cell_faces, indptr), shape=(num_faces, num_cells)
        )

        super().__init__(2, nodes, face_nodes, cell_faces, name)

    def cell_node_matrix(self) -> np.ndarray:
        """
        Returns:
            An array with ``shape=(num_cells, 3)``, representing the cell-to-node map.

        """

        # Absolute value needed since cellFaces can be negative
        cn = self.face_nodes * np.abs(self.cell_faces) * sps.eye(self.num_cells)
        row, col = cn.nonzero()
        scol = np.argsort(col)

        # Consistency check
        assert np.all(accumarray.accum(col, np.ones(col.size)) == (self.dim + 1))

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

        # Transform points to column vector if necessary (scipy.Delaunay requires this
        # format)
        if tet is None:
            tesselation = scipy.spatial.qhull.Delaunay(p.transpose())
            tet = tesselation.simplices.transpose()

        if name is None:
            name = "TetrahedralGrid"

        num_nodes = p.shape[1]

        nodes = p
        assert num_nodes > 3  # Check of transposes of point array

        num_cells = tet.shape[1]
        tet = self._permute_nodes(p, tet)
        # This is apparently needed to appease mypy
        assert tet is not None

        # Define face-nodes so that the first column contains fn of cell 0, etc.
        face_nodes = np.vstack(
            (tet[[1, 0, 2]], tet[[0, 1, 3]], tet[[2, 0, 3]], tet[[1, 2, 3]])
        )
        # Reshape face-nodes into a 3x 4*num_cells-matrix, with the four first columns
        # belonging to cell 0.
        face_nodes = face_nodes.reshape((3, 4 * num_cells), order="F")
        sort_ind = np.squeeze(np.argsort(face_nodes, axis=0))
        face_nodes_sorted = np.sort(face_nodes, axis=0)
        face_nodes, _, cell_faces = setmembership.unique_columns_tol(face_nodes_sorted)

        num_faces = face_nodes.shape[1]

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

        # Cell face relation
        num_faces_per_cell = 4
        indptr = np.hstack(
            (
                np.arange(0, num_faces_per_cell * num_cells, num_faces_per_cell),
                num_faces_per_cell * num_cells,
            )
        )
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
