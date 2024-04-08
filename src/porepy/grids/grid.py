"""Module containing the parent class for all grids.

See documentation of class :class:`Grid` for further details.

.. rubric:: Acknowledgements
    The data structure for the grid is inspired by that used in the
    `Matlab Reservoir Simulation Toolbox (MRST) <www.sintef.no/projectweb/mrst/>`_
    developed by SINTEF ICT. Some of the methods, in particular
    :meth:`~Grid.compute_geometry` and its subfunctions is to a large degree
    translations of the corresponding functions in MRST as they were defined around
    2016.

"""

from __future__ import annotations

import copy
import itertools
import warnings
from itertools import count
from typing import Any, Optional, Union

import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data
from porepy.utils import mcolon, tags


class Grid:
    """Parent class for all grids.

    The grid stores topological information, as well as geometric information. Geometric
    information requires calling :meth:`compute_geometry` to be initialized.

    Note:
        As of yet, there is no structure for tags (face or cell) in the grid. This may
        be introduced later.

    Parameters:
        dim: Grid dimension.
        nodes: ``shape=(ambient_dimension, num_nodes)``

            Node coordinates, where ``ambient_dimension`` is the dimension of the grid.
        face_nodes: ``shape=(num_nodes, num_faces)``

            A map from faces to respective nodes spanning the face.
        cell_faces: ``shape=(num_faces, num_cells)``

            A map from cells to faces bordering the respective cell.
        name: Name of grid.
        history: ``default=None``

            Information on the formation of the grid.
        external_tags: ``default=None``

            External tags for nodes and grids. Will be added to :attr:`~tags`.

    """

    _counter = count(0)
    """Counter of instantiated grids. See :meth:`__new__` and :meth:`id`."""
    __id: int
    """Name-mangled reference to assigned ID."""

    def __new__(cls, *args, **kwargs) -> Grid:
        """Make object and set ID by forwarding :attr:`_counter`."""

        obj = object.__new__(cls)
        obj.__id = next(cls._counter)
        return obj

    def __init__(
        self,
        dim: int,
        nodes: np.ndarray[Any, np.dtype[np.float64]],
        face_nodes: sps.csc_matrix,
        cell_faces: sps.csc_matrix,
        name: str,
        history: Optional[Union[str, list[str]]] = None,
        external_tags: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        if not (dim >= 0 and dim <= 3):
            raise ValueError("A grid has to be of dimension 0, 1, 2, or 3.")

        self.dim: int = dim
        """Grid dimension. Should be in ``{0, 1, 2, 3}``."""

        self.nodes: np.ndarray = nodes
        """An array with ``shape=(ambient_dimension, num_nodes)`` containing node
        coordinates column-wise."""

        # Force topological information to be stored as integers. The known subclasses
        # of Grid all have integer values in these arrays, so this is a safeguard aimed
        # at third-party code.
        cell_faces.data = cell_faces.data.astype(int)
        face_nodes.data = face_nodes.data.astype(int)

        self.cell_faces: sps.csc_matrix = cell_faces
        """An array with ``shape=(num_faces, num_cells)`` representing the map from
        cells to faces bordering respective cell.

        Matrix elements have value +-1, where + corresponds to the face normal vector
        being outwards.

        """
        self.face_nodes: sps.csc_matrix = face_nodes
        """An array with ``shape=(num_nodes, num_faces)`` representing the map from
        faces to nodes spanning respective face.

        Assumes the nodes of each face are ordered according to the right-hand rule.

        Note:
            To use :meth:`compute_geometry` later, the field
            ``face_nodes.indices`` should store the nodes of each face sorted.
            ``face_nodes.indices[face_nodes.indptr[i]:face_nodes.indptr[i+1]]``
            are the nodes of face i, which should be ordered counter-clockwise.

            By counter-clockwise we mean as seen from cell ``cell_faces[i,:] == -1``.

            Equivalently the nodes will be clockwise as seen from cell
            ``cell_faces[i,:] == 1``.

            Note that operations on the face_nodes matrix
            (such as converting it to a csr-matrix) may change the ordering of
            the nodes (``face_nodes.indices``),
            which will break :meth:`compute_geometry`.

        """

        self.name: str = name
        """Name assigned to this grid."""

        self.history: list[str]
        """Information on the formation of the grid, such as the
        constructor, computations of geometry etc.

        """

        if history is None:
            self.history = []
        elif isinstance(history, list):
            self.history = history
        else:  # history is str
            self.history = [history]

        # Infer bookkeeping from size of parameters
        self.num_nodes: int = nodes.shape[1]
        """Number of nodes in the grid."""
        self.num_faces: int = face_nodes.shape[1]
        """Number of faces in the grid."""
        self.num_cells: int = cell_faces.shape[1]
        """Number of cells in the grid."""

        # NOTE: Variables that are only relevant for some grids. Use with caution.
        self.frac_num: int = -1
        """Index of the fracture the grid corresponds to. Take
        value ``(0, 1, ...)`` if the grid corresponds to a fracture, -1 if not.

        """

        self.parent_cell_ind: np.ndarray = np.arange(self.num_cells)
        """Index of parent the cell in the parent grid for grids that have refined
        sub-grids or are sub-grids of larger grids.

        Defaults to a mapping to its own index with ``shape=(num_cells,)``.

        """

        self.global_point_ind: np.ndarray = np.arange(self.num_nodes)
        """An array with ``shape=(num_nodes,)`` containing indices of each point,
        assigned during processing of mixed-dimensional grids created by gmsh.

        Used to identify points that are geometrically equal, though on different grids.

        Could potentially be used to identify such geometrically equal points at a
        later stage, but there is no guarantee that this will work.

        """

        self._physical_name_index: int = -1
        """Used to keep track of processing of grids generated by gmsh."""

        self.well_num: int = -1
        """Index of the well associated to the grid. Takes a value in
        ``(0, 1, ..)`` if the grid corresponds to a well, -1 if not.

        """

        self.periodic_face_map: np.ndarray
        """Index of periodic boundary faces,
        ``(shape=(2, num_periodic_faces), dtype=int)``.

        Face index ``periodic_face_map[0, i]`` is periodic with
        face index ``periodic_face_map[1, i]``.
        This attribute is set with :meth:`set_periodic_map`.

        """

        self.frac_pairs: np.ndarray = np.array([[]], dtype=int)
        """Indices of faces that are geometrically coinciding, but
        lay on different side of a lower-dimensional grid.

        """

        # Add tag for the boundary faces
        self.tags: dict[str, Any]
        """Tags allow to mark subdomains of interest.

        The default tags are used to mark faces or nodes as fracture, tips and domain
        boundaries.
        User tags can be provided in the constructor or added later.

        """
        if external_tags is None:
            self.tags = {}
            self.initiate_face_tags()
            self.update_boundary_face_tag()

            # Add tag for the boundary nodes
            self.initiate_node_tags()
            self.update_boundary_node_tag()
        else:
            self.tags = external_tags
            self._check_tags()

        # NOTE: These attributes are defined in compute_geometry.
        self.face_areas: np.ndarray
        """Areas of all faces ``(shape=(num_cells,))``.
        Available after calling :meth:`~compute_geometry`.

        """
        self.face_centers: np.ndarray
        """Centers of all faces. ``(shape=(ambient_dimension, num_faces))``.
        Available after calling :meth:`~compute_geometry`.

        """
        self.face_normals: np.ndarray
        """An array containing column-wise normal vectors of all faces with
        ``shape=(ambient_dimenaion, num_faces)``.

        See also :attr:`cell_faces`.

        Available after calling :meth:`compute_geometry`.

        """
        self.cell_centers: np.ndarray
        """An array containing column-wise the centers of all cells with
        ``shape=(ambient_dimension, num_cells)``.

        Available after calling :meth:`~compute_geometry`.

        """
        self.cell_volumes: np.ndarray
        """An array containing column-wise the volumes per cell with
        ``shape=(num_cells,)``.

        Available after calling :meth:`~compute_geometry`.

        """

    @property
    def id(self) -> int:
        """Grid ID.

        The returned attribute must not be changed. This may severely compromise other
        parts of the code, such as sorting in md grids.

        The attribute is set in :meth:`__new__`.
        This avoids calls to ``super().__init__`` in child classes.

        """
        return self.__id

    def copy(self) -> pp.Grid:
        """Create a new instance with some attributes deep-copied from the grid.

        Returns:
            A deep copy of ``self``. Some predefined attributes are also copied.

        """
        # Instantiating a new object gives it a unique id (see __new__)
        h = Grid(
            self.dim,
            self.nodes.copy(),
            self.face_nodes.copy(),
            self.cell_faces.copy(),
            name=self.name,
            history=self.history,
        )
        copy_attributes = [
            "cell_volumes",
            "cell_centers",
            "face_centers",
            "face_normals",
            "face_areas",
            "tags",
            "periodic_face_map",
        ]
        for attr in copy_attributes:
            if hasattr(self, attr):
                setattr(h, attr, copy.deepcopy(getattr(self, attr)))

        return h

    def __repr__(self) -> str:
        """Returns a string representation of the grid including topological information
        ."""
        s = f"Grid with name {self.name} and id {self.id}" + "\n"
        s = "Grid history: " + ", ".join(self.history) + "\n"
        s += "Number of cells " + str(self.num_cells) + "\n"
        s += "Number of faces " + str(self.num_faces) + "\n"
        s += "Number of nodes " + str(self.num_nodes) + "\n"
        s += "Dimension " + str(self.dim)
        return s

    def __str__(self) -> str:
        """Returns a simplified string representation including the given name and some
        topological information."""
        s = str()

        # Special treatment of point grids.
        if "PointGrid" in self.name:
            s = "Point grid.\n"
            n = self.nodes
            s += "Coordinate: (" + str(n[0]) + ", " + str(n[1])
            s += ", " + str(n[2]) + ")\n"
            return s

        # More or less uniform treatment of the types of grids.
        if "CartGrid" in self.name:
            s = "Cartesian grid in " + str(self.dim) + " dimensions.\n"
        elif "TensorGrid" in self.name:
            s = "Tensor grid in " + str(self.dim) + " dimensions.\n"
        elif "StructuredTriangleGrid" in self.name:
            s = "Structured triangular grid.\n"
        elif "TriangleGrid" in self.name:
            s = "Triangular grid. \n"
        elif "StructuredTetrahedralGrid" in self.name:
            s = "Structured tetrahedral grid.\n"
        elif "TetrahedralGrid" in self.name:
            s = "Tetrahedral grid.\n"
        else:
            s = " ".join(self.name) + "\n"
        s = s + "Number of cells " + str(self.num_cells) + "\n"
        s = s + "Number of faces " + str(self.num_faces) + "\n"
        s = s + "Number of nodes " + str(self.num_nodes) + "\n"

        return s

    def compute_geometry(self) -> None:
        """Compute geometric quantities for the grid.

        The method could have been called from the constructor, however, in cases where
        the grid is modified after the initial construction ( say, grid refinement),
        this may lead to costly, unnecessary computations.

        Computes the face areas, face centers, face normals and cell volumes.

        """

        self.history.append("Compute geometry")

        if self.dim == 0:
            self._compute_geometry_0d()
        elif self.dim == 1:
            self._compute_geometry_1d()
        elif self.dim == 2:
            self._compute_geometry_2d()
        else:
            self._compute_geometry_3d()

    def _compute_geometry_0d(self) -> None:
        """Compute 0D geometry"""
        self.face_areas = np.zeros(0)
        self.face_centers = self.nodes
        self.face_normals = np.zeros((3, 0))  # not well-defined

        # Force cell volume to have data type float, so that mypy does not get confused
        # for higher-dimensional grids.
        self.cell_volumes = np.ones(self.num_cells, dtype=float)
        if not hasattr(self, "cell_centers"):
            raise ValueError("Can not compute geometry of 0d grid without cell centers")
        # Here, we should assign the cell centers, however this does nothing:
        # self.cell_centers = self.cell_centers

    def _compute_geometry_1d(self) -> None:
        """Auxiliary function to compute the geometry for 1D grids."""

        self.face_areas = np.ones(self.num_faces)

        fn = self.face_nodes.indices
        n = fn.size
        self.face_centers = self.nodes[:, fn]

        self.face_normals = np.tile(
            pp.map_geometry.compute_tangent(self.nodes), (n, 1)
        ).T

        cf = self.cell_faces.indices
        xf1 = self.face_centers[:, cf[::2]]
        xf2 = self.face_centers[:, cf[1::2]]

        self.cell_volumes = np.linalg.norm(xf1 - xf2, axis=0)
        self.cell_centers = 0.5 * (xf1 + xf2)

        # Ensure that normal vector direction corresponds with sign convention
        # in self.cellFaces

        def nrm(u):
            return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

        fi, ci, val = sparse_array_to_row_col_data(self.cell_faces)
        _, idx = np.unique(fi, return_index=True)
        sgn = val[idx]
        fc = self.face_centers[:, fi[idx]]
        cc = self.cell_centers[:, ci[idx]]
        v = fc - cc
        # Prolong the vector from cell to face center in the direction of the
        # normal vector. If the prolonged vector is shorter, the normal should
        # be flipped
        vn = v + nrm(v) * self.face_normals[:, fi[idx]] * 0.001
        flip = np.logical_or(
            np.logical_and(nrm(v) > nrm(vn), sgn > 0),
            np.logical_and(nrm(v) < nrm(vn), sgn < 0),
        )
        self.face_normals[:, flip] *= -1

    def _compute_geometry_2d(self) -> None:
        """Auxiliary function to compute the geometry for 2D grids.

        We assume that:
        - either the cell_faces and face_nodes are consistently oriented
        - or that the grid is composed of convex cells.
        """

        # Each face is determined by a start and end node, the tangent is given by
        # the x_end - x_start. The face normal is a 90 degree clock-wise rotation
        # of the tangent.

        # Define an oriented face to nodes mapping, the orientation is determined by the
        # ordering in self.face_nodes.indices. The start node gets a -1 and the end node
        # a +1.
        fn_orient = sps.csc_matrix(self.face_nodes, dtype=int, copy=True)
        fn_orient.data = -np.power(-1, np.arange(fn_orient.data.size))

        # Consistency check: For each cell, the nodes should occur twice in the
        # face-node relation: Once as a start node and once as an end node. Summed over
        # all faces of the cell, the result should be zero.
        is_oriented = (fn_orient * self.cell_faces).nnz == 0
        if not is_oriented:
            # The assumptions underlying the computation for general cells is broken.
            # Fall back to a legacy implementation which is only valid for convex cells.
            warnings.warn(
                "Orientations in face_nodes and cell_faces are inconsistent. "
                "Fall back on an implementation that assumes all cells are convex."
            )

        # Compute the tangent vectors and use them to compute face attributes
        tangent = self.nodes * fn_orient
        self.face_areas = np.sqrt(np.square(tangent).sum(axis=0))
        self.face_centers = 0.5 * self.nodes * np.abs(fn_orient)

        # Compute the temporary cell centers as average of the cell nodes
        faceno, cellno, cf_orient = sparse_array_to_row_col_data(self.cell_faces)
        cx = np.bincount(cellno, weights=self.face_centers[0, faceno])
        cy = np.bincount(cellno, weights=self.face_centers[1, faceno])
        cz = np.bincount(cellno, weights=self.face_centers[2, faceno])
        temp_cell_centers = np.vstack((cx, cy, cz)) / np.bincount(cellno)

        # Create sub-simplexes based on triplets, each consisting of a cell center and
        # the start and end of a face. Compute the vectors that are normal to the
        # sub-simplex and whose length is the area.
        subsimplex_heights = self.face_centers[:, faceno] - temp_cell_centers[:, cellno]
        # Use a cross product to get the area of the sub-simplex.
        subsimplex_normals = 0.5 * np.cross(
            subsimplex_heights, cf_orient * tangent[:, faceno], axis=0
        )

        # Construct the unit normal of the grid as planar object
        if is_oriented:
            plane_normal = subsimplex_normals.sum(axis=1)
            plane_normal /= np.linalg.norm(plane_normal)
        else:
            plane_normal = pp.map_geometry.compute_normal(self.nodes)

        # Compute the face normals by rotating the tangent according to the orientation
        # of the plane
        self.face_normals = np.cross(tangent, plane_normal, axis=0)

        # Compute the signed volumes of sub-simplexes. Positive values indicate that
        # cell_faces and face_nodes are consistently oriented; in practice, this means
        # that nodes that are oriented counter clock-wise give positive values.
        subsimplex_volumes = np.dot(plane_normal, subsimplex_normals)

        # In case of inconsistent orientation, the sub-simplex volumes and normals need
        # to be corrected.
        if not is_oriented:
            # The volume is still correct, but it may be negative. Fix this.
            subsimplex_volumes = np.abs(subsimplex_volumes)

            # We flip the normal if the inner product between the height (face_center -
            # cell_center) and the face normal is different from what is expected from
            # the cell-face relation (as contained in cf_orient).
            flip = (
                cf_orient
                * np.sum(subsimplex_heights * self.face_normals[:, faceno], axis=0)
            ) < 0
            # Gather the information of whether to flip for the two sides of each face.
            # Under the assumption that the grid is convex, the two sides should yield
            # the same decision. For a non-convex cell, the two sides may (will?) not
            # agree on whether to flip, and the decision is essentially arbitrary.
            flip = np.bincount(faceno, weights=flip).astype(bool)
            self.face_normals[:, flip] *= -1

        # Compute the cell volumes by adding all relevant sub-simplex volumes.
        self.cell_volumes = np.bincount(cellno, weights=subsimplex_volumes)

        # Sanity check on the cell_volumes
        assert np.all(self.cell_volumes >= 0)

        # Compute cells centroids as weighted average of the sub-simplex centroids
        sub_centroids = (
            temp_cell_centers[:, cellno] + 2 * self.face_centers[:, faceno]
        ) / 3
        ccx = np.bincount(cellno, weights=subsimplex_volumes * sub_centroids[0])
        ccy = np.bincount(cellno, weights=subsimplex_volumes * sub_centroids[1])
        ccz = np.bincount(cellno, weights=subsimplex_volumes * sub_centroids[2])

        self.cell_centers = np.vstack((ccx, ccy, ccz)) / self.cell_volumes

    def _compute_geometry_3d(self) -> None:
        """Auxiliary function to compute the geometry for 3D grids.

        The implementation is motivated by the similar MRST function.

        Note:
            The function is very long, and could have been broken up into
            parts (face and cell computations are an obvious solution).

        """
        num_face_nodes = self.face_nodes.nnz
        face_node_ptr = self.face_nodes.indptr

        num_nodes_per_face = face_node_ptr[1:] - face_node_ptr[:-1]

        # Face-node relationships. Note that the elements here will also serve as a
        # representation of an edge along the face (face_nodes[i] represents the edge
        # running from face_nodes[i] to face_nodes[i+1]).
        face_nodes = self.face_nodes.indices
        # For each node, index of its parent face
        face_node_ind = pp.matrix_operations.rldecode(
            np.arange(self.num_faces), num_nodes_per_face
        )

        # Index of next node on the edge list. Note that this assumes the elements in
        # face_nodes is stored in an ordered fashion
        next_node = np.arange(num_face_nodes) + 1
        # Close loops, for face i, the next node is the first of face i
        next_node[face_node_ptr[1:] - 1] = face_node_ptr[:-1]

        # Mapping from cells to faces
        edge_2_face = sps.coo_matrix(
            (np.ones(num_face_nodes), (np.arange(num_face_nodes), face_node_ind))
        ).tocsc()

        # Define temporary face center as the mean of the face nodes
        tmp_face_center = self.nodes[:, face_nodes] * edge_2_face / num_nodes_per_face
        # Associate this value with all the edge of this face
        tmp_face_center = edge_2_face * tmp_face_center.transpose()

        # Vector along each edge
        along_edge = self.nodes[:, face_nodes[next_node]] - self.nodes[:, face_nodes]
        # Vector from face center to start node of each edge
        face_2_node = tmp_face_center.transpose() - self.nodes[:, face_nodes]

        # Assign a normal vector with this edge, by taking the cross product between
        # along_edge and face_2_node. Divide by two to ensure that the normal vector has
        # length equal to the area of the face triangle (by properties of cross product)
        sub_normals = (
            np.vstack(
                (
                    along_edge[1] * face_2_node[2] - along_edge[2] * face_2_node[1],
                    along_edge[2] * face_2_node[0] - along_edge[0] * face_2_node[2],
                    along_edge[0] * face_2_node[1] - along_edge[1] * face_2_node[0],
                )
            )
            / 2
        )

        def nrm(v):
            return np.sqrt(np.sum(v * v, axis=0))

        # Calculate area of sub-face associated with each edge - note that the
        # sub-normals are area weighted
        sub_areas = nrm(sub_normals)

        # Centers of sub-faces are given by the centroid coordinates, e.g. the mean
        # coordinate of the edge endpoints and the temporary face center
        sub_centroids = (
            self.nodes[:, face_nodes]
            + self.nodes[:, face_nodes[next_node]]
            + tmp_face_center.transpose()
        ) / 3

        # Face normals are given as the sum of the sub-components
        face_normals = sub_normals * edge_2_face
        # Similar with face areas
        face_areas = edge_2_face.transpose() * sub_areas

        # Test whether the sub-normals are pointing in the same direction as the main
        # normal: Distribute the main normal onto the edges, and take scalar product by
        # element-wise multiplication with sub-normals, and sum over the components
        # (axis=0).
        # NOTE: There should be a built-in function for this in numpy?
        sub_normals_sign = np.sign(
            np.sum(
                sub_normals * (edge_2_face * face_normals.transpose()).transpose(),
                axis=0,
            )
        )

        # Finally, face centers are the area weighted means of centroids of
        # the sub-faces
        face_centers = sub_areas * sub_centroids * edge_2_face / face_areas

        # .. and we're done with the faces. Store information
        self.face_centers = face_centers
        self.face_normals = face_normals
        self.face_areas = face_areas

        # Cells

        # Temporary cell center coordinates as the mean of the face center coordinates.
        # The cells are divided into sub-tetrahedra ( corresponding to triangular
        # sub-faces above), with the temporary cell center as the final node

        # Mapping from edges to cells. Take absolute value of cell_faces, since the
        # elements are signed (contains the divergence). Note that edge_2_cell will
        # contain more elements than edge_2_face, since the former will count internal
        # faces twice (one for each adjacent cell)
        edge_2_cell = edge_2_face * np.abs(self.cell_faces)
        # Sort indices to avoid messing up the mappings later
        edge_2_cell.sort_indices()

        # Obtain relations between edges, faces and cells, in the form of index lists.
        # Each element in the list corresponds to an edge seen from a cell (e.g. edges
        # on internal faces are seen twice).

        # Cell numbers are obtained from the columns in edge_2_cell.
        cell_numbers = pp.matrix_operations.rldecode(
            np.arange(self.num_cells), np.diff(edge_2_cell.indptr)
        )
        # Edge numbers from the rows. Here it is crucial that the indices are sorted
        edge_numbers = edge_2_cell.indices
        # Face numbers are obtained from the face-node relations (with the nodes
        # doubling as representation of edges)
        face_numbers = face_node_ind[edge_numbers]

        # Number of edges per cell
        num_cell_edges = edge_2_cell.indptr[1:] - edge_2_cell.indptr[:-1]

        def bincount_nd(arr, weights):
            """Utility function to sum vector quantities by np.bincount. We
            could probably have used np.apply_along_axis, but I could not
            make it work.

            Intended use: Map sub-cell centroids to a quantity for the cell.
            """
            dim = weights.shape[0]
            sz = arr.max() + 1

            count = np.zeros((dim, sz))
            for iter1 in range(dim):
                count[iter1] = np.bincount(arr, weights=weights[iter1], minlength=sz)
            return count

        # First estimate of cell centers as the mean of its faces' centers Divide by
        # num_cell_edges here since all edges bring in their faces
        tmp_cell_centers = bincount_nd(
            cell_numbers, face_centers[:, face_numbers] / num_cell_edges[cell_numbers]
        )

        # Distance from the temporary cell center to the sub-centroids (of
        # the tetrahedra associated with each edge)
        dist_cellcenter_subface = (
            sub_centroids[:, edge_numbers] - tmp_cell_centers[:, cell_numbers]
        )

        # Get sign of normal vectors, seen from all faces. Make sure we get a numpy
        # ndarray, and not a matrix (.A), and that the array is 1D (squeeze)
        orientation = np.squeeze(self.cell_faces[face_numbers, cell_numbers].A)

        # Get outwards pointing sub-normals for all sub-faces: We need to account for
        # both the orientation of the face, and the orientation of sub-faces relative to
        # faces.
        outer_normals = (
            sub_normals[:, edge_numbers] * orientation * sub_normals_sign[edge_numbers]
        )

        # Volumes of tetrahedra are now given by the dot product between the outer
        #  normal (which is area weighted, and thus represent the base of the tet), with
        #  the distance from temporary cell center (the dot product gives the height).
        tet_volumes = np.sum(dist_cellcenter_subface * outer_normals, axis=0) / 3

        # Sometimes the sub-tet volumes can have a volume of numerical zero. Why this is
        # so is not clear, but for the moment, we allow for a slightly negative value.
        if not np.all(tet_volumes > -1e-12):  # On the fly test
            raise ValueError("Some tetrahedra have negative volume")

        # The cell volumes are now found by summing sub-tetrahedra
        cell_volumes = np.bincount(cell_numbers, weights=tet_volumes)
        tri_centroids = 3 / 4 * dist_cellcenter_subface

        # Compute a correction to the temporary cell center, by a volume weighted sum of
        # the sub-tetrahedra
        rel_centroid = (
            bincount_nd(cell_numbers, tet_volumes * tri_centroids) / cell_volumes
        )
        cell_centers = tmp_cell_centers + rel_centroid

        # ... and we're done
        self.cell_centers = cell_centers
        self.cell_volumes = cell_volumes

    def cell_nodes(self) -> sps.csc_matrix:
        """Obtain mapping between cells and nodes.

        Returns:
            An array with ``shape=(num_nodes, num_cells)`` representing the mapping from
            cells to nodes spanning respective cell.

            The value 1 indicates a connection between a cell and node column-wise.

        """
        mat = (self.face_nodes * np.abs(self.cell_faces)) > 0
        return mat

    def num_cell_nodes(self) -> np.ndarray:
        """
        Returns:
            An array with ``shape=(num_cells,)`` containing the number of nodes per
            cell.

        """
        return self.cell_nodes().sum(axis=0).A.ravel("F")

    def get_internal_nodes(self) -> np.ndarray:
        """
        Returns:
            An array with ``shape=(num_internal_nodes,)`` containing the indices of
            internal nodes.

        """
        internal_nodes = np.setdiff1d(
            np.arange(self.num_nodes), self.get_boundary_nodes(), assume_unique=True
        )
        return internal_nodes

    def get_all_boundary_faces(self) -> np.ndarray:
        """
        Returns:
            An array with ``shape=(num_boundary_faces,)`` containing the indices of
            all faces tagged as either fractures, domain boundary or tip.

        """
        return self._indices(tags.all_face_tags(self.tags))

    def get_all_boundary_nodes(self) -> np.ndarray:
        """
        Returns:
            An array with Indices of all boundary nodes ``shape=(num_boundary_nodes,)``
            containing the indices of all faces tagged as either fractures,
            domain boundary or tip.

        """
        return self._indices(tags.all_node_tags(self.tags))

    def get_boundary_faces(self) -> np.ndarray:
        """
        Returns:
            An array with ``shape=(n,)`` containing the indices of all faces tagged as
            domain boundary.

        """
        return self._indices(self.tags["domain_boundary_faces"])

    def get_internal_faces(self) -> np.ndarray:
        """
        Returns:
            An array with ``shape=(num_internal_faces,)`` containing indices of internal
            faces.

        """
        return np.setdiff1d(
            np.arange(self.num_faces), self.get_all_boundary_faces(), assume_unique=True
        )

    def get_boundary_nodes(self) -> np.ndarray:
        """
        Returns:
            An array with ``shape=(n,)`` containing indices of all domain
            boundary nodes.

        """
        return self._indices(self.tags["domain_boundary_nodes"])

    def update_boundary_face_tag(self) -> None:
        """Tags faces on the boundary of the grid with boundary tag."""
        zeros = np.zeros(self.num_faces, dtype=bool)
        self.tags["domain_boundary_faces"] = zeros
        if self.dim > 0:  # by default no 0d grid at the boundary of the domain
            bd_faces = np.argwhere(np.diff(self.cell_faces.tocsr().indptr) == 1).ravel()
            self.tags["domain_boundary_faces"][bd_faces] = True

    def set_periodic_map(self, periodic_face_map: np.ndarray) -> None:
        """Sets the index map between periodic boundary faces.

        The mapping assumes a one to one mapping between the periodic boundary faces
        (i.e., matching faces).

        Note:
            This method changes the attribute ``self.tags["domain_boundary_faces"]``.
            The domain boundary tags are set to ``False`` for all faces
            in ``periodic_face_map``.

        Parameters:
            periodic_face_map: ``shape=(2, num_periodic_faces), dtype=int``

                Defines the periodic faces.
                Face index ``periodic_face_map[0, i]`` is periodic with
                face index ``periodic_face_map[1, i]``.
                The given map is stored to the attribute :attr:`periodic_face_map`.

        Raises:
            ValueError: If ``periodic_face_map`` is of wrong shape or contains negative
                values.

        """
        if periodic_face_map.shape[0] != 2:
            raise ValueError("dimension 0 of periodic_face_map must be of size 2")
        if np.max(periodic_face_map) > self.num_faces:
            raise ValueError("periodic face number larger than number of faces")
        if np.min(periodic_face_map) < 0:
            raise ValueError("periodic face number cannot be negative")

        self.periodic_face_map = periodic_face_map
        self.tags["domain_boundary_faces"][self.periodic_face_map.ravel("C")] = False

    def update_boundary_node_tag(self) -> None:
        """Tags nodes on the boundary of the grid with boundary tag."""

        mask = {
            "domain_boundary_faces": "domain_boundary_nodes",
            "fracture_faces": "fracture_nodes",
            "tip_faces": "tip_nodes",
        }
        zeros = np.zeros(self.num_nodes, dtype=bool)

        for face_tag, node_tag in mask.items():
            self.tags[node_tag] = zeros.copy()
            faces = np.where(self.tags[face_tag])[0]
            if faces.size > 0:
                first = self.face_nodes.indptr[faces]
                second = self.face_nodes.indptr[faces + 1]
                nodes = self.face_nodes.indices[mcolon.mcolon(first, second)]
                self.tags[node_tag][nodes] = True

    def cell_diameters(self, cn: Optional[sps.spmatrix] = None) -> np.ndarray:
        """Computes the cell diameters.

        Parameters:
            cn: ``default=None``
                Cell-to-nodes map, already computed previously.
                If None, a call to :meth:`cell_nodes` is provided.

        Returns:
            Values of the cell diameter for each cell, ``(shape=(num_cells))``.

            If the dimension of the grid is zero, returns 0.

        """
        if self.dim == 0:
            return np.zeros(1)

        def comb(n):
            return np.fromiter(
                itertools.chain.from_iterable(itertools.combinations(n, 2)), n.dtype
            ).reshape((2, -1), order="F")

        def diam(n):
            return np.amax(
                np.linalg.norm(self.nodes[:, n[0, :]] - self.nodes[:, n[1, :]], axis=0)
            )

        if cn is None:
            cn = self.cell_nodes()
        return np.array(
            [
                diam(comb(cn.indices[cn.indptr[c] : cn.indptr[c + 1]]))
                for c in np.arange(self.num_cells)
            ]
        )

    def cell_face_as_dense(self) -> np.ndarray:
        """Obtain the cell-face relation in the form of two rows, rather than a
        sparse matrix.

        This alternative format can be useful in some cases.

        Each column in the array corresponds to a face, and the elements in that column
        refers to cell indices. The value -1 signifies a boundary. The normal vector of
        the face points from the first to the second row.

        Returns:
            Array representation of face-cell relations with ``shape=(2, num_faces)``.

        """
        if self.num_faces == 0:
            return np.zeros((0, 2))
        n = self.cell_faces.tocsr()
        d = np.diff(n.indptr)
        rows = pp.matrix_operations.rldecode(np.arange(d.size), d)
        # Increase the data by one to distinguish cell indices from boundary
        # cells
        data = n.indices + 1
        cols = ((n.data + 1) / 2).astype(int)
        neighs = sps.coo_matrix((data, (rows, cols))).todense()
        # Subtract 1 to get back to real cell indices
        neighs -= 1
        neighs = neighs.transpose().A.astype(int)
        # Finally, we need to switch order of rows to get normal vectors
        # pointing from first to second row.
        return neighs[::-1]

    def cell_connection_map(self) -> sps.csr_matrix:
        """Get a matrix representation of cell-cell connections, as defined by
        two cells sharing a face.

        Returns:
            A sparse matrix with ``(shape=(num_cells, num_cells), dtype=bool)``.

            Element ``(i,j)`` is True if cells ``i`` and ``j`` share a face.
            The matrix is thus symmetric.

        """

        # Create a copy of the cell-face relation, so that we can modify it at
        # will
        cell_faces = self.cell_faces.copy()

        # Direction of normal vector does not matter here, only 0s and 1s
        cell_faces.data = np.abs(cell_faces.data)

        # Find connection between cells via the cell-face map
        c2c = cell_faces.transpose() * cell_faces
        # Only care about absolute values
        c2c.data = np.clip(c2c.data, 0, 1).astype("bool")

        return c2c

    def signs_and_cells_of_boundary_faces(
        self, faces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the direction of the normal vector (inward or outwards from a cell)
        and the cell neighbor of **boundary** faces.

        Parameters:
            faces: ``shape=(n,)``

                Indices of ``n`` faces that you want to know the sign for. The faces
                must be boundary faces.

        Raises:
            ValueError: If a target face is internal.

        Returns:
            A 2-tuple containing

            :obj:`~numpy.ndarray`:
                ``shape=(n,)``

                The sign of the faces. Will be +1 if the face normal vector points out
                of the cell, -1 if the normal vector is pointing inwards.

            :obj:`~numpy.ndarray`:
                ``shape=(n,)``

                For each face, index of the cell next to the boundary.

        """

        IA = np.argsort(faces)
        IC = np.argsort(IA)

        fi, ci, sgn = sparse_array_to_row_col_data(self.cell_faces[faces[IA], :])
        if fi.size != faces.size:
            raise ValueError("sign of internal faces does not make sense")

        fi_sorted = np.argsort(fi)
        sgn, ci = sgn[fi_sorted], ci[fi_sorted]
        sgn, ci = sgn[IC], ci[IC]
        return sgn, ci

    def closest_cell(
        self, p: np.ndarray, return_distance: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """For a set of points, find closest cell by cell center.

        If several centers have the same distance, one of them will be returned.

        For ``dim < 3``, no checks are made if the point is in the plane / line
        of the grid.

        Parameters:
            p: ``shape=(3, n)``

                Coordinates of ``n`` points. If ``p.shape[0] < 3``,
                additional points will be treated as zeros.
            return_distance: A flag indicating whether the distances should be returned
                as well.

        Returns:
            An array with ``(shape=(n,), dtype=int)`` containing for each point the
            index of the cell with center closest to the point.

            If ``return_distance`` is True, returns a 2-tuple, where the second array
            contains the distances to respective centers for each point.


        """
        p = np.atleast_2d(p)
        if p.shape[0] < 3:
            z = np.zeros((3 - p.shape[0], p.shape[1]))
            p = np.vstack((p, z))

        def min_dist(pts):
            c = self.cell_centers
            d = np.sum(np.power(c - pts, 2), axis=0)
            min_id = np.argmin(d)
            return min_id, np.sqrt(d[min_id])

        ci = np.empty(p.shape[1], dtype=int)
        di = np.empty(p.shape[1])
        for i in range(p.shape[1]):
            ci[i], di[i] = min_dist(p[:, i].reshape((3, -1)))

        if return_distance:
            return ci, di
        else:
            return ci

    def initiate_face_tags(self) -> None:
        """Create zero arrays for the standard face tags and update :attr:`tags`."""
        keys = tags.standard_face_tags()
        values = [np.zeros(self.num_faces, dtype=bool) for _ in keys]
        tags.add_tags(self, dict(zip(keys, values)))

    def initiate_node_tags(self) -> None:
        """Create zero arrays for the standard node tags and update :attr:`tags`."""
        keys = tags.standard_node_tags()
        values = [np.zeros(self.num_nodes, dtype=bool) for _ in keys]
        tags.add_tags(self, dict(zip(keys, values)))

    def _check_tags(self) -> None:
        """Check if all the standard tags are specified in :attr:`tags`,
        and the tag arrays have correct sizes.

        Raises:
            ValueError: If any inconsistency among tags is found.

        """
        for key in tags.standard_node_tags():
            if key not in self.tags:
                raise ValueError(f"The tag key {key} must be specified")
            value: np.ndarray = self.tags[key]
            if not value.size == self.num_nodes:
                raise ValueError(f"Wrong size of value for tag {key}")

        for key in tags.standard_face_tags():
            if key not in self.tags:
                raise ValueError(f"The tag key {key} must be specified")
            value = self.tags[key]
            if not value.size == self.num_faces:
                raise ValueError(f"Wrong size of value for tag {key}")

    @staticmethod
    def _indices(true_false: np.ndarray) -> np.ndarray:
        """Auxiliary function for :obj:`~numpy.argwhere` with ``ravel('F')."""
        return np.argwhere(true_false).ravel("F")
