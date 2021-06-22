""" Module containing the parent class for all grids.

See documentation of the Grid class for further details.

Acknowledgements:
    The data structure for the grid is inspired by that used in the Matlab
    Reservoir Simulation Toolbox (MRST) developed by SINTEF ICT, see
    www.sintef.no/projectweb/mrst/ . Some of the methods, in particular
    compute_geometry() and its subfunctions is to a large degree translations
    of the corresponding functions in MRST.

"""
import itertools
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.utils import matrix_compression, mcolon, tags

module_sections = ["grids", "gridding"]


class Grid:
    """
    Parent class for all grids.

    The grid stores topological information, as well as geometric
    information (after a call to self.compute_geometry().

    As of yet, there is no structure for tags (face or cell) is the grid.
    This will be introduced later.

    Attributes:
        Comes in tthree classes. Topologogical information, defined at
        construction time:

        dim (int): dimension. Should be 0 or 1 or 2 or 3
        nodes (np.ndarray): node coordinates. size: dim x num_nodes
        face_nodes (sps.csc-matrix): Face-node relationships. Matrix size:
            num_faces x num_cells. To use compute_geometry() later, he field
            face_nodes.indices should store the nodes of each face sorted.
            For more information, see information on compute_geometry()
            below.
        cell_faces (sps.csc-matrix): Cell-face relationships. Matrix size:
            num_faces x num_cells. Matrix elements have value +-1, where +
            corresponds to the face normal vector being outwards.
        name (list): Information on the formation of the grid, such as the
            constructor, computations of geometry etc.
        num_nodes (int): Number of nodes in the grid
        num_faces (int): Number of faces in the grid
        num_cells (int): Number of cells in the grid

        ---

        Geometric information, obtained by call to compute_geometry():
        Assumes the nodes of each face is ordered according to the right
        hand rule.
        face_nodes.indices[face_nodes.indptr[i]:face_nodes.indptr[i+1]]
        are the nodes of face i, which should be ordered counter-clockwise.
        By counter-clockwise we mean as seen from cell cell_faces[i,:] == -1.
        Equivalently the nodes will be clockwise as seen from cell
        cell_faces[i,:] == 1. Note that operations on the face_nodes matrix
        (such as converting it to a csr-matrix) may change the ordering of
        the nodes (face_nodes.indices), which will break compute_geometry().
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

        ----

        Other fieds: These may only be assigned to certain grids, use with
        caution:

        frac_num (int): Index of the fracture the grid corresponds to. Take
            value (0, 1, ...) if the grid corresponds to a fracture, -1 if not.
        parent_cell_ind (np.ndarray): For grids that have refined or are subgrids
            of larger grids, index of parent the cell in the parent grid.
            Defaults to a mapping to its own index.
        global_point_ind (np.ndarray): Index of each point, assigned during processing
            of mixed-dimensional grids created by gmsh. Used to identify points that
            are geometrically equal, though on different grids. Could potentially be
            used to identify such geometrically equal points at a later stage, but
            there is no guarantee that this will work.
        _physical_name_index (int): Used to keep track of processing of grids generated
            by gmsh.
        frac_pairs (np.ndarray): indices of faces that are geometrically coinciding, but
            lay on different side of a lower-dimensional grid.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(
        self,
        dim: int,
        nodes: np.ndarray,
        face_nodes: sps.csc_matrix,
        cell_faces: sps.csc_matrix,
        name: Union[List[str], str],
        external_tags: Dict[str, np.ndarray] = None,
    ) -> None:
        """Initialize the grid

        See class documentation for further description of parameters.

        Parameters
        ----------
        dim (int): grid dimension
        nodes (np.ndarray): node coordinates.
        face_nodes (sps.csc_matrix): Face-node relations.
        cell_faces (sps.csc_matrix): Cell-face relations
        name (str): Name of grid
        tags (dict): Tags for nodes and grids. Will be constructed if not provided.
        """
        if not (dim >= 0 and dim <= 3):
            raise ValueError("A grid has to be 0, 1, 2, or 3.")

        self.dim: int = dim
        self.nodes: np.ndarray = nodes
        self.cell_faces: sps.csc_matrix = cell_faces
        self.face_nodes: sps.csc_matrix = face_nodes

        if isinstance(name, list):
            self.name: List[str] = name
        else:
            self.name = [name]

        # Infer bookkeeping from size of parameters
        self.num_nodes: int = nodes.shape[1]
        self.num_faces: int = face_nodes.shape[1]
        self.num_cells: int = cell_faces.shape[1]

        # NOTE: Variables that are only relevant for some grids.
        # Use with caution.
        self.frac_num: int = -1
        self.parent_cell_ind: np.ndarray = np.arange(self.num_cells)
        self.global_point_ind: np.ndarray = np.arange(self.num_nodes)
        self._physical_name_index: int = -1

        self.frac_pairs: np.ndarray = np.array([[]], dtype=int)

        # Add tag for the boundary faces
        if external_tags is None:
            self.tags: Dict[str, np.ndarray] = {}
            self.initiate_face_tags()
            self.update_boundary_face_tag()

            # Add tag for the boundary nodes
            self.initiate_node_tags()
            self.update_boundary_node_tag()
        else:
            self.tags = external_tags
            self._check_tags()

    @pp.time_logger(sections=module_sections)
    def copy(self):
        """
        Create a deep copy of the grid.

        Returns:
            grid: A deep copy of self. All attributes will also be copied.

        """
        h = Grid(
            self.dim,
            self.nodes.copy(),
            self.face_nodes.copy(),
            self.cell_faces.copy(),
            self.name,
        )
        if hasattr(self, "cell_volumes"):
            h.cell_volumes = self.cell_volumes.copy()
        if hasattr(self, "cell_centers"):
            h.cell_centers = self.cell_centers.copy()
        if hasattr(self, "face_centers"):
            h.face_centers = self.face_centers.copy()
        if hasattr(self, "face_normals"):
            h.face_normals = self.face_normals.copy()
        if hasattr(self, "face_areas"):
            h.face_areas = self.face_areas.copy()
        if hasattr(self, "tags"):
            h.tags = self.tags.copy()
        if hasattr(self, "periodic_face_map"):
            h.periodic_face_map = self.periodic_face_map.copy()
        return h

    @pp.time_logger(sections=module_sections)
    def __repr__(self) -> str:
        """
        Implementation of __repr__

        """
        s = "Grid with history " + ", ".join(self.name) + "\n"
        s = s + "Number of cells " + str(self.num_cells) + "\n"
        s = s + "Number of faces " + str(self.num_faces) + "\n"
        s = s + "Number of nodes " + str(self.num_nodes) + "\n"
        s += "Dimension " + str(self.dim)
        return s

    @pp.time_logger(sections=module_sections)
    def __str__(self) -> str:
        """Implementation of __str__"""
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

    @pp.time_logger(sections=module_sections)
    def compute_geometry(self) -> None:
        """Compute geometric quantities for the grid.

        This method initializes class variables describing the grid
        geometry, see class documentation for details.

        The method could have been called from the constructor, however,
        in cases where the grid is modified after the initial construction (
        say, grid refinement), this may lead to costly, unnecessary
        computations.
        """

        self.name.append("Compute geometry")

        if self.dim == 0:
            self._compute_geometry_0d()
        elif self.dim == 1:
            self._compute_geometry_1d()
        elif self.dim == 2:
            self._compute_geometry_2d()
        else:
            self._compute_geometry_3d()

    @pp.time_logger(sections=module_sections)
    def _compute_geometry_0d(self) -> None:
        "Compute 0D geometry"
        self.face_areas = np.zeros(0)
        self.face_centers = self.nodes
        self.face_normals = np.zeros((3, 0))  # not well-defined

        self.cell_volumes = np.ones(self.num_cells)
        if not hasattr(self, "cell_centers"):
            raise ValueError("Can not compute geometry of 0d grid without cell centers")
        # Here, we should assign the cell centers, however this does nothing:
        # self.cell_centers = self.cell_centers

    @pp.time_logger(sections=module_sections)
    def _compute_geometry_1d(self) -> None:
        "Compute 1D geometry"

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
        @pp.time_logger(sections=module_sections)
        def nrm(u):
            return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

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
        flip = np.logical_or(
            np.logical_and(nrm(v) > nrm(vn), sgn > 0),
            np.logical_and(nrm(v) < nrm(vn), sgn < 0),
        )
        self.face_normals[:, flip] *= -1

    @pp.time_logger(sections=module_sections)
    def _compute_geometry_2d(self) -> None:
        "Compute 2D geometry, with method motivated by similar MRST function"

        R = pp.map_geometry.project_plane_matrix(self.nodes, check_planar=False)
        self.nodes = np.dot(R, self.nodes)

        fn = self.face_nodes.indices
        edge1 = fn[::2]
        edge2 = fn[1::2]

        xe1 = self.nodes[:, edge1]
        xe2 = self.nodes[:, edge2]

        edge_length_x = xe2[0] - xe1[0]
        edge_length_y = xe2[1] - xe1[1]
        edge_length_z = xe2[2] - xe1[2]
        self.face_areas = np.sqrt(
            np.power(edge_length_x, 2)
            + np.power(edge_length_y, 2)
            + np.power(edge_length_z, 2)
        )
        self.face_centers = 0.5 * (xe1 + xe2)
        n = edge_length_z.shape[0]
        self.face_normals = np.vstack((edge_length_y, -edge_length_x, np.zeros(n)))

        cell_faces, cellno = self.cell_faces.nonzero()
        cx = np.bincount(cellno, weights=self.face_centers[0, cell_faces])
        cy = np.bincount(cellno, weights=self.face_centers[1, cell_faces])
        cz = np.bincount(cellno, weights=self.face_centers[2, cell_faces])
        self.cell_centers = np.vstack((cx, cy, cz)) / np.bincount(cellno)

        a = xe1[:, cell_faces] - self.cell_centers[:, cellno]
        b = xe2[:, cell_faces] - self.cell_centers[:, cellno]

        sub_volumes = 0.5 * np.abs(a[0] * b[1] - a[1] * b[0])
        self.cell_volumes = np.bincount(cellno, weights=sub_volumes)

        sub_centroids = (
            self.cell_centers[:, cellno] + 2 * self.face_centers[:, cell_faces]
        ) / 3

        ccx = np.bincount(cellno, weights=sub_volumes * sub_centroids[0])
        ccy = np.bincount(cellno, weights=sub_volumes * sub_centroids[1])
        ccz = np.bincount(cellno, weights=sub_volumes * sub_centroids[2])

        self.cell_centers = np.vstack((ccx, ccy, ccz)) / self.cell_volumes

        # Ensure that normal vector direction corresponds with sign convention
        # in self.cellFaces
        @pp.time_logger(sections=module_sections)
        def nrm(u):
            return np.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

        [fi, ci, val] = sps.find(self.cell_faces)
        _, idx = np.unique(fi, return_index=True)
        sgn = val[idx]
        fc = self.face_centers[:, fi[idx]]
        cc = self.cell_centers[:, ci[idx]]
        v = fc - cc
        # Prolong the vector from cell to face center in the direction of the
        # normal vector. If the prolonged vector is shorter, the normal should
        # flipped
        vn = (
            v
            + nrm(v) * self.face_normals[:, fi[idx]] / self.face_areas[fi[idx]] * 0.001
        )
        flip = np.logical_or(
            np.logical_and(nrm(v) > nrm(vn), sgn > 0),
            np.logical_and(nrm(v) < nrm(vn), sgn < 0),
        )
        self.face_normals[:, flip] *= -1

        self.nodes = np.dot(R.T, self.nodes)
        self.face_normals = np.dot(R.T, self.face_normals)
        self.face_centers = np.dot(R.T, self.face_centers)
        self.cell_centers = np.dot(R.T, self.cell_centers)

    @pp.time_logger(sections=module_sections)
    def _compute_geometry_3d(self):
        """
        Helper function to compute geometry for 3D grids

        The implementation is motivated by the similar MRST function.

        NOTE: The function is very long, and could have been broken up into
        parts (face and cell computations are an obvious solution).

        """
        num_face_nodes = self.face_nodes.nnz
        face_node_ptr = self.face_nodes.indptr

        num_nodes_per_face = face_node_ptr[1:] - face_node_ptr[:-1]

        # Face-node relationships. Note that the elements here will also
        # serve as a representation of an edge along the face (face_nodes[i]
        #  represents the edge running from face_nodes[i] to face_nodes[i+1])
        face_nodes = self.face_nodes.indices
        # For each node, index of its parent face
        face_node_ind = matrix_compression.rldecode(
            np.arange(self.num_faces), num_nodes_per_face
        )

        # Index of next node on the edge list. Note that this assumes the
        # elements in face_nodes is stored in an ordered fasion
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

        # Assign a normal vector with this edge, by taking the cross product
        # between along_edge and face_2_node
        # Divide by two to ensure that the normal vector has length equal to
        # the area of the face triangle (by properties of cross product)
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

        @pp.time_logger(sections=module_sections)
        def nrm(v):
            return np.sqrt(np.sum(v * v, axis=0))

        # Calculate area of sub-face associated with each edge - note that
        # the sub-normals are area weighted
        sub_areas = nrm(sub_normals)

        # Centers of sub-faces are given by the centroid coordinates,
        # e.g. the mean coordinate of the edge endpoints and the temporary
        # face center
        sub_centroids = (
            self.nodes[:, face_nodes]
            + self.nodes[:, face_nodes[next_node]]
            + tmp_face_center.transpose()
        ) / 3

        # Face normals are given as the sum of the sub-components
        face_normals = sub_normals * edge_2_face
        # Similar with face areas
        face_areas = edge_2_face.transpose() * sub_areas

        # Test whether the sub-normals are pointing in the same direction as
        # the main normal: Distribute the main normal onto the edges,
        # and take scalar product by element-wise multiplication with
        # sub-normals, and sum over the components (axis=0).
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

        # Temporary cell center coordinates as the mean of the face center
        # coordinates. The cells are divided into sub-tetrahedra (
        # corresponding to triangular sub-faces above), with the temporary
        # cell center as the final node

        # Mapping from edges to cells. Take absolute value of cell_faces,
        # since the elements are signed (contains the divergence).
        # Note that edge_2_cell will contain more elements than edge_2_face,
        # since the former will count internal faces twice (one for each
        # adjacent cell)
        edge_2_cell = edge_2_face * np.abs(self.cell_faces)
        # Sort indices to avoid messing up the mappings later
        edge_2_cell.sort_indices()

        # Obtain relations between edges, faces and cells, in the form of
        # index lists. Each element in the list corresponds to an edge seen
        # from a cell (e.g. edges on internal faces are seen twice).

        # Cell numbers are obtained from the columns in edge_2_cell.
        cell_numbers = matrix_compression.rldecode(
            np.arange(self.num_cells), np.diff(edge_2_cell.indptr)
        )
        # Edge numbers from the rows. Here it is crucial that the indices
        # are sorted
        edge_numbers = edge_2_cell.indices
        # Face numbers are obtained from the face-node relations (with the
        # nodes doubling as representation of edges)
        face_numbers = face_node_ind[edge_numbers]

        # Number of edges per cell
        num_cell_edges = edge_2_cell.indptr[1:] - edge_2_cell.indptr[:-1]

        @pp.time_logger(sections=module_sections)
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

        # First estimate of cell centers as the mean of its faces' centers
        # Divide by num_cell_edges here since all edges bring in their faces
        tmp_cell_centers = bincount_nd(
            cell_numbers, face_centers[:, face_numbers] / num_cell_edges[cell_numbers]
        )

        # Distance from the temporary cell center to the sub-centroids (of
        # the tetrahedra associated with each edge)
        dist_cellcenter_subface = (
            sub_centroids[:, edge_numbers] - tmp_cell_centers[:, cell_numbers]
        )

        # Get sign of normal vectors, seen from all faces.
        # Make sure we get a numpy ndarray, and not a matrix (.A), and that
        # the array is 1D (squeeze)
        orientation = np.squeeze(self.cell_faces[face_numbers, cell_numbers].A)

        # Get outwards pointing sub-normals for all sub-faces: We need to
        # account for both the orientation of the face, and the orientation
        # of sub-faces relative to faces.
        outer_normals = (
            sub_normals[:, edge_numbers] * orientation * sub_normals_sign[edge_numbers]
        )

        # Volumes of tetrahedra are now given by the dot product between the
        #  outer normal (which is area weighted, and thus represent the base
        #  of the tet), with the distancance from temporary cell center (the
        # dot product gives the hight).
        tet_volumes = np.sum(dist_cellcenter_subface * outer_normals, axis=0) / 3

        # Sometimes the sub-tet volumes can have a volume of numerical zero.
        # Why this is so is not clear, but for the moment, we allow for a
        # slightly negative value.
        if not np.all(tet_volumes > -1e-12):  # On the fly test
            raise ValueError("Some tets have negative volume")

        # The cell volumes are now found by summing sub-tetrahedra
        cell_volumes = np.bincount(cell_numbers, weights=tet_volumes)
        tri_centroids = 3 / 4 * dist_cellcenter_subface

        # Compute a correction to the temporary cell center, by a volume
        # weighted sum of the sub-tetrahedra
        rel_centroid = (
            bincount_nd(cell_numbers, tet_volumes * tri_centroids) / cell_volumes
        )
        cell_centers = tmp_cell_centers + rel_centroid

        # ... and we're done
        self.cell_centers = cell_centers
        self.cell_volumes = cell_volumes

    @pp.time_logger(sections=module_sections)
    def cell_nodes(self) -> sps.csc_matrix:
        """
        Obtain mapping between cells and nodes.

        Returns:
            sps.csc_matrix, size num_nodes x num_cells: Value 1 indicates a
                connection between cell and node.

        """
        mat = (self.face_nodes * np.abs(self.cell_faces)) > 0
        return mat

    @pp.time_logger(sections=module_sections)
    def num_cell_nodes(self) -> np.ndarray:
        """Number of nodes per cell.

        Returns:
            np.ndarray, size num_cells: Number of nodes per cell.

        """
        return self.cell_nodes().sum(axis=0).A.ravel("F")

    @pp.time_logger(sections=module_sections)
    def get_internal_nodes(self) -> np.ndarray:
        """
        Get internal nodes id of the grid.

        Returns:
            np.ndarray (1D), index of internal nodes.

        """
        internal_nodes = np.setdiff1d(
            np.arange(self.num_nodes), self.get_boundary_nodes(), assume_unique=True
        )
        return internal_nodes

    @pp.time_logger(sections=module_sections)
    def get_all_boundary_faces(self) -> np.ndarray:
        """
        Get indices of all faces tagged as either fractures, domain boundary or
        tip.
        """
        return self._indices(tags.all_face_tags(self.tags))

    @pp.time_logger(sections=module_sections)
    def get_all_boundary_nodes(self) -> np.ndarray:
        """
        Get indices of all nodes tagged as either fractures, domain boundary or
        tip.
        """
        return self._indices(tags.all_node_tags(self.tags))

    @pp.time_logger(sections=module_sections)
    def get_boundary_faces(self) -> np.ndarray:
        """
        Get indices of all faces tagged as domain boundary.
        """
        return self._indices(self.tags["domain_boundary_faces"])

    @pp.time_logger(sections=module_sections)
    def get_internal_faces(self) -> np.ndarray:
        """
        Get internal faces id of the grid

        Returns:
            np.ndarray (1d), index of internal faces.

        """
        return np.setdiff1d(
            np.arange(self.num_faces), self.get_all_boundary_faces(), assume_unique=True
        )

    @pp.time_logger(sections=module_sections)
    def get_boundary_nodes(self) -> np.ndarray:
        """
        Get nodes on the boundary

        Returns:
            np.ndarray (1d), index of nodes on the boundary

        """
        return self._indices(self.tags["domain_boundary_nodes"])

    @pp.time_logger(sections=module_sections)
    def update_boundary_face_tag(self) -> None:
        """Tag faces on the boundary of the grid with boundary tag."""
        zeros = np.zeros(self.num_faces, dtype=bool)
        self.tags["domain_boundary_faces"] = zeros
        if self.dim > 0:  # by default no 0d grid at the boundary of the domain
            bd_faces = np.argwhere(np.diff(self.cell_faces.tocsr().indptr) == 1).ravel()
            self.tags["domain_boundary_faces"][bd_faces] = True

    @pp.time_logger(sections=module_sections)
    def set_periodic_map(self, periodic_face_map: np.ndarray) -> None:
        """
        Set the index map between periodic boundary faces. The mapping assumes
        a one to one mapping between the periodic boundary faces (i.e., matching
        faces).

        Parameters:
        periodic_face_map (np.ndarray, int, 2 x # periodic faces): Defines the periodic
            faces. Face index periodic_face_map[0, i] is periodic with face index
            periodic_face_map[1, i]. The given map is stored to the attribute periodic_face_map

        New attributes:
        periodic_face_map (np.ndarray, int, 2 x # periodic faces): See periodic_face_map
            in Parameters.

        Changes attributes:
        tags["domain_boundary_faces"]: The domain boundary tags are set to False
            for all faces in periodic_face_map.
        """
        if periodic_face_map.shape[0] != 2:
            raise ValueError("dimension 0 of periodic_face_map must be of size 2")
        if np.max(periodic_face_map) > self.num_faces:
            raise ValueError("periodic face number larger than number of faces")
        if np.min(periodic_face_map) < 0:
            raise ValueError("periodic face number cannot be negative")

        self.periodic_face_map = periodic_face_map
        self.tags["domain_boundary_faces"][self.periodic_face_map.ravel("C")] = False

    @pp.time_logger(sections=module_sections)
    def update_boundary_node_tag(self) -> None:
        """Tag nodes on the boundary of the grid with boundary tag."""

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

    @pp.time_logger(sections=module_sections)
    def cell_diameters(self, cn: sps.spmatrix = None) -> np.ndarray:
        """
        Compute the cell diameters. If self.dim == 0, return 0

        Parameters:
            cn (optional): cell nodes map, previously already computed.
            Otherwise a call to self.cell_nodes is provided.

        Returns:
            np.array, num_cells: values of the cell diameter for each cell

        """
        if self.dim == 0:
            return np.zeros(1)

        @pp.time_logger(sections=module_sections)
        def comb(n):
            return np.fromiter(
                itertools.chain.from_iterable(itertools.combinations(n, 2)), n.dtype
            ).reshape((2, -1), order="F")

        @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def cell_face_as_dense(self) -> np.ndarray:
        """
        Obtain the cell-face relation in the from of two rows, rather than a
        sparse matrix. This alternative format can be useful in some cases.

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
        cols = ((n.data + 1) / 2).astype(int)
        neighs = sps.coo_matrix((data, (rows, cols))).todense()
        # Subtract 1 to get back to real cell indices
        neighs -= 1
        neighs = neighs.transpose().A.astype(int)
        # Finally, we need to switch order of rows to get normal vectors
        # pointing from first to second row.
        return neighs[::-1]

    @pp.time_logger(sections=module_sections)
    def cell_connection_map(self) -> sps.csr_matrix:
        """
        Get a matrix representation of cell-cell connections, as defined by
        two cells sharing a face.

        Returns:
            scipy.sparse.csr_matrix, size num_cells * num_cells: Boolean
                matrix, element (i,j) is true if cells i and j share a face.
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

    @pp.time_logger(sections=module_sections)
    def signs_and_cells_of_boundary_faces(
        self, faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the direction of the normal vector (inward or outwards from a cell)
        and the cell neighbour of _boundary_ faces.

        Parameters:
            faces: (ndarray) indices of faces that you want to know the sign for. The
                faces must be boundary faces.

        Returns:
            (ndarray) the sign of the faces

        Raises:
            ValueError if a target face is internal.

        """

        IA = np.argsort(faces)
        IC = np.argsort(IA)

        fi, ci, sgn = sps.find(self.cell_faces[faces[IA], :])
        if fi.size != faces.size:
            raise ValueError("sign of internal faces does not make sense")

        fi_sorted = np.argsort(fi)
        sgn, ci = sgn[fi_sorted], ci[fi_sorted]
        sgn, ci = sgn[IC], ci[IC]
        return sgn, ci

    @pp.time_logger(sections=module_sections)
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the bounding box of the grid.

        Returns:
            np.array (size 3): Minimum node coordinates in each direction.
            np.array (size 3): Maximum node coordinates in each direction.

        """
        if self.dim == 0:
            coords = self.cell_centers
        else:
            coords = self.nodes
        return np.amin(coords, axis=1), np.amax(coords, axis=1)

    @pp.time_logger(sections=module_sections)
    def closest_cell(
        self, p: np.ndarray, return_distance: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """For a set of points, find closest cell by cell center.

        If several centers have the same distance, one of them will be
        returned.

        For dim < 3, no checks are made if the point is in the plane / line
        of the grid.

        Parameters:
            p (np.ndarray, 3xn): Point coordinates. If p.shape[0] < 3,
                additional points will be treated as zeros.

        Returns:
            np.ndarray of ints: For each point, index of the cell with center
                closest to the point.
        """
        p = np.atleast_2d(p)
        if p.shape[0] < 3:
            z = np.zeros((3 - p.shape[0], p.shape[1]))
            p = np.vstack((p, z))

        @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def initiate_face_tags(self) -> None:
        keys = tags.standard_face_tags()
        values = [np.zeros(self.num_faces, dtype=bool) for _ in keys]
        tags.add_tags(self, dict(zip(keys, values)))

    @pp.time_logger(sections=module_sections)
    def initiate_node_tags(self) -> None:
        keys = tags.standard_node_tags()
        values = [np.zeros(self.num_nodes, dtype=bool) for _ in keys]
        tags.add_tags(self, dict(zip(keys, values)))

    @pp.time_logger(sections=module_sections)
    def _check_tags(self) -> None:
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
    @pp.time_logger(sections=module_sections)
    def _indices(true_false: np.ndarray) -> np.ndarray:
        """Shorthand for np.argwhere."""
        return np.argwhere(true_false).ravel("F")
