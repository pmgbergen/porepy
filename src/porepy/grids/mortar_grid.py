""" Module containing the class for the mortar grid.
"""
import warnings
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse as sps

import porepy as pp

# Module level constants, used to define sides of a mortar grid.
# This is in essence an Enum, but that led to trouble in pickling a GridBucket.
NONE_SIDE = 0
LEFT_SIDE = 1
RIGHT_SIDE = 2
WHOLE_SIDE = np.iinfo(type(NONE_SIDE)).max


class MortarGrid:
    """
    Parent class for mortar grids it contains two grids representing the left
    and right part of the mortar grid and the weighted mapping from the higher
    dimensional grid (as set of faces) to the mortar grids and from the lower
    dimensional grid (as set of cells) to the mortar grids. The two mortar grids
    can be different.

    Attributes:

        dim (int): dimension. Should be 0 or 1 or 2.
        side_grids (dictionary of Grid): grid for each side. The key is an integer
            with value {0, 1, 2}, and the value is a Grid.
        sides (array of integers with values in {0, 1, 2}): ordering of the sides.
        _high_to_mortar_int (sps.csc-matrix): Face-cell relationships between the
            high dimensional grid and the mortar grids. Matrix size:
            num_faces x num_cells. In the beginning we assume matching grids,
            but it can be modified by calling refine_mortar(). The matrix
            elements represent the ratio between the geometrical objects.
        _slave_to_mortar_int (sps.csc-matrix): Cell-cell relationships between the
            mortar grids and the low dimensional grid. Matrix size:
            num_cells x num_cells. Matrix elements represent the ratio between
            the geometrical objects.
        name (list): Information on the formation of the grid, such as the
            constructor, computations of geometry etc.
        tol (double): Tolerance use when matching grids during update of mortar or
            master / slave grids.

    """

    def __init__(
        self,
        dim: int,
        side_grids: Dict[int, pp.Grid],
        face_cells: sps.spmatrix,
        name: Union[str, List[str]] = "",
        face_duplicate_ind: Optional[np.ndarray] = None,
        tol: float = 1e-6,
    ):
        """Initialize the mortar grid

        See class documentation for further description of parameters.

        If faces the higher-dimensional grid is split along the mortar grid (e.g. room
        is made for a fracture grid), it is assumed that the extra faces are added
        to the end of the face list. That is, for face pairs {(a1, b1), (a2, b2), ...}
        max(a_i) should be less than min(b_j).
        NOTE: This behaviour can be overridden by providing indices of the extra faces
        in the parameter face_duplicate_ind.

        Parameters:
            dim (int): grid dimension
            side_grids (dictionary of Grid): grid on each side.
            face_cells (sps.csc_matrix): Cell-face relations between the higher
                dimensional grid and the lower dimensional grid.
            name (str): Name of the grid. Can also be used to set various information on
                the grid.
            face_duplicate_ind (np.ndarray, optional): Which faces should be considered
                duplicates, and mapped to the second of the side_grids. If not provided,
                duplicate faces will be inferred from the indices of the faces. Will
                only be used if len(side_Grids) == 2.
            tol (double, optional): Tolerance used in geometric computations. Defaults
                to 1e-6.

        """

        if dim == 3:
            raise ValueError("A mortar grid cannot be 3d")
        if not np.all([g.dim == dim for g in side_grids.values()]):
            raise ValueError("All the mortar grids have to have the same dimension")

        self.dim = dim
        self.side_grids: Dict[int, pp.Grid] = side_grids.copy()
        self.sides: np.ndarray = np.array(list(self.side_grids.keys()))

        if not (self.num_sides() == 1 or self.num_sides() == 2):
            raise ValueError("The number of sides have to be 1 or 2")

        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]
        self.name.append("mortar_grid")
        self.tol = tol

        # easy access attributes with a fixed ordering of the side grids
        self.num_cells: np.ndarray = np.sum(
            [g.num_cells for g in self.side_grids.values()], dtype=np.int
        )
        self.cell_volumes: np.ndarray = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )
        self.cell_centers: np.ndarray = np.hstack(
            [g.cell_centers for g in self.side_grids.values()]
        )

        # face_cells mapping from the higher dimensional grid to the mortar grid
        # also here we assume that, in the beginning the mortar grids are equal
        # to the co-dimensional grid. If this assumption is not satisfied we
        # need to change the following lines

        # Creation of the high_to_mortar_int, basically we start from the face_cells
        # map and we split the relation
        # low_dimensional_cell -> 2 high_dimensional_face
        # as
        # low_dimensional_cell -> high_dimensional_face
        # The mapping consider the cell ids of the second mortar grid shifted by
        # the g.num_cells of the first grid. We keep this convention through the
        # implementation. The ordering is given by sides or the keys of
        # side_grids.

        # Number of cells in the first mortar grid
        num_cells = list(self.side_grids.values())[0].num_cells
        cells, faces, data = sps.find(face_cells)
        if self.num_sides() == 2:

            # Depending on the numbering of the faces, some work may be needed to place
            # oposing mortar cells on each side of the lower-dimensional grid.
            # At the end of this if-else, the first num_cells are on one side, the
            # rest is on the other side.
            if face_duplicate_ind is None:
                # This is a tacit assumption on the numbering scheme for split faces,
                # all faces on one side of the mortar grid should be indexed first,
                # the their duplicate on the other side of the fracture.
                cells_on_second_side = faces > np.median(faces)
                cells[cells_on_second_side] += num_cells
            else:
                cells_on_second_side = np.in1d(faces, face_duplicate_ind)
                cells[cells_on_second_side] += num_cells

        shape = (num_cells * self.num_sides(), face_cells.shape[1])
        self._high_to_mortar_int: sps.spmatrix = sps.csc_matrix(
            (data.astype(np.float), (cells, faces)), shape=shape
        )

        # cell_cells mapping from the mortar grid to the lower dimensional grid.
        # It is composed by two identity matrices since we are assuming matching
        # grids here.
        identity = [[sps.identity(num_cells)]] * self.num_sides()
        self._slave_to_mortar_int: sps.spmatrix = sps.bmat(identity, format="csc")

    def __repr__(self) -> str:
        """
        Implementation of __repr__

        """
        s = (
            "Mortar grid with history "
            + ", ".join(self.name)
            + "\n"
            + "Dimension "
            + str(self.dim)
            + "\n"
            + f"Number of cells {self.num_cells}\n"
            + f"Number of sides {len(self.side_grids)}\n"
            + "Number of cells in lower-dimensional neighbor "
            + f"{self.mortar_to_low_int().shape[0]}\n"
            + "Number of faces in higher-dimensional neighbor "
            + f"{self.mortar_to_high_int().shape[0]}\n"
        )

        return s

    def __str__(self) -> str:
        """Implementation of __str__"""
        s = (
            "Mortar grid with history "
            + ", ".join(self.name)
            + "\n"
            + "Dimension "
            + str(self.dim)
            + "\n"
            + f"Number of cells {self.num_cells}"
        )
        return s

    def compute_geometry(self) -> None:
        """
        Compute the geometry of the mortar grids.
        We assume that they are not aligned with x (1d) or x, y (2d).
        """
        # Update the actual side grids
        for g in self.side_grids.values():
            g.compute_geometry()

        # Update the attributes
        self.num_cells = np.sum(
            [g.num_cells for g in self.side_grids.values()], dtype=np.int
        )
        self.cell_volumes = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )
        self.cell_centers = np.hstack(
            [g.cell_centers for g in self.side_grids.values()]
        )

    ### Methods to update the mortar grid, or the neighboring grids.

    def update_mortar(
        self, new_side_grids: Dict[int, pp.Grid], tol: float = None
    ) -> None:
        """
        Update the low_to_mortar_int and high_to_mortar_int maps when the mortar grids
        are changed.

        Parameter:
            side_matrix (dict): for each side (identified with values {0, 1, 2}, as
                used when this MortarGrid was defined) a matrix representing the
                new mapping between the old and new mortar grids.
            tol (double, optional): Tolerance used for matching the new and old grids.
                Defaults to self.tol.

        """
        if tol is None:
            tol = self.tol

        split_matrix = {}

        # For each side we compute the mapping between the old and the new mortar
        # grids, we store them in a dictionary with SideTag as key.
        for side, new_g in new_side_grids.items():
            g = self.side_grids[side]
            if g.dim != new_g.dim:
                raise ValueError("Grid dimension has to be the same")

            if g.dim == 0:
                # Nothing to do
                return
            elif g.dim == 1:
                split_matrix[side] = _split_matrix_1d(g, new_g, tol)
            elif g.dim == 2:
                split_matrix[side] = _split_matrix_2d(g, new_g, tol)
            else:
                # No 3d mortar grid
                raise ValueError

        # In the case of different side ordering between the input data and the
        # stored we need to remap it. The resulting matrix will be a block
        # diagonal matrix, where in each block we have the mapping between the
        # (relative to side) old grid and the new one.
        matrix = np.empty((self.num_sides(), self.num_sides()), dtype=np.object)

        # Loop on all the side grids, if not given an identity matrix is
        # considered
        for pos, (side, g) in enumerate(self.side_grids.items()):
            matrix[pos, pos] = split_matrix.get(side, sps.identity(g.num_cells))

        # Once the global matrix is constructed the new low_to_mortar_int and
        # high_to_mortar_int maps are updated.
        matrix = sps.bmat(matrix)
        self._slave_to_mortar_int = matrix * self._slave_to_mortar_int
        self._high_to_mortar_int = matrix * self._high_to_mortar_int

        # Update the side grids
        for side, g in new_side_grids.items():
            self.side_grids[side] = g.copy()

        # update the geometry
        self.compute_geometry()

        self._check_mappings()

    def update_slave(self, new_g: pp.Grid, tol: float = None) -> None:
        """
        Update the _slave_to_mortar_int map when the lower dimensional grid is changed.

        Parameter:
            new_g (pp.Grid): The new slave grid.
            tol (double, optional): Tolerance used for matching the new and old grids.
                Defaults to self.tol.

        """
        if tol is None:
            tol = self.tol

        split_matrix = {}

        # For each side we compute the mapping between the new lower dimensional
        # grid and the mortar grid, we store them in a dictionary with SideTag as key.
        for side, g in self.side_grids.items():
            if g.dim != new_g.dim:
                raise ValueError("Grid dimension has to be the same")

            if self.dim == 0:
                # Nothing to do
                return
            elif self.dim == 1:
                split_matrix[side] = _split_matrix_1d(g, new_g, tol).T
            elif self.dim == 2:
                split_matrix[side] = _split_matrix_2d(g, new_g, tol).T
            else:
                # No 3d mortar grid
                raise ValueError

        # In the case of different side ordering between the input data and the
        # stored we need to remap it. The resulting matrix will be a block
        # matrix, where in each block we have the mapping between the
        # (relative to side) the new grid and the mortar grid.
        matrix = np.empty((self.num_sides(), 1), dtype=np.object)

        for pos, (side, _) in enumerate(self.side_grids.items()):
            matrix[pos, 0] = split_matrix[side]

        # Update the low_to_mortar_int map. No need to update the high_to_mortar_int.
        self._slave_to_mortar_int = sps.bmat(matrix, format="csc")
        self._check_mappings()

    def update_master(self, g_new: pp.Grid, g_old: pp.Grid, tol: float = None):
        """
        Update the _slave_to_mortar_int map when the lower dimensional grid is changed.

        Parameter:
            g_new (pp.Grid): The new master grid.
            g_old (pp.Grid): The old master grid.
            tol (double, optional): Tolerance used for matching the new and old grids.
                Defaults to self.tol.

        """
        # TODO: Why is the signature of this method different from update_slave?
        if tol is None:
            tol = self.tol

        if self.dim == 0:

            # retrieve the old faces and the corresponding coordinates
            _, old_faces, _ = sps.find(self._high_to_mortar_int)
            old_nodes = g_old.face_centers[:, old_faces]

            # retrieve the boundary faces and the corresponding coordinates
            new_faces = g_new.get_all_boundary_faces()
            new_nodes = g_new.face_centers[:, new_faces]

            # we assume only one old node
            for i in range(1, old_nodes.shape[1]):
                is_same = (
                    pp.distances.point_pointset(old_nodes[:, 0], old_nodes[:, i]) < tol
                )
                if not is_same:
                    raise ValueError(
                        "0d->1d mappings must map to the same physical point"
                    )
            old_nodes = old_nodes[:, 0]
            mask = pp.distances.point_pointset(old_nodes, new_nodes) < tol
            new_faces = new_faces[mask]

            shape = (g_old.num_faces, g_new.num_faces)
            matrix_DIJ = (np.ones(old_faces.shape), (old_faces, new_faces))
            split_matrix = sps.csc_matrix(matrix_DIJ, shape=shape)

        elif self.dim == 1:
            # The case is conceptually similar to 0d, but quite a bit more technical
            split_matrix = pp.match_grids.match_grids_along_1d_mortar(
                self, g_new, g_old, tol
            )

        else:  # should be mg.dim == 2
            # It should be possible to use essentially the same approach as in 1d,
            # but this is not yet covered.
            raise NotImplementedError("Have not yet implemented this.")

        # Make a comment here
        self._high_to_mortar_int = self._high_to_mortar_int * split_matrix
        self._check_mappings()

    def num_sides(self) -> int:
        """
        Shortcut to compute the number of sides, it has to be 2 or 1.

        Return:
            Number of sides.

        """
        return len(self.side_grids)

    def project_to_side_grids(
        self,
    ) -> Generator[Tuple[sps.spmatrix, pp.Grid], None, None]:
        """Generator for the side grids (pp.Grid) representation of the mortar
        cells, and projection operators from the mortar cells, combining cells on all
        the sides, to the specific side grids.

        Yields:
            grid (pp.Grid): PorePy grid representing one of the sides of the
                mortar grid. Can be used for standard discretizations.
            proj (sps.csc_matrix): Projection from the mortar cells to this
                side grid.

        """
        counter = 0
        for grid in self.side_grids.values():
            nc = grid.num_cells
            rows = np.arange(nc)
            cols = rows + counter
            data = np.ones(nc)
            proj = sps.coo_matrix(
                (data, (rows, cols)), shape=(nc, self.num_cells)
            ).tocsc()

            counter += nc
            yield proj, grid

    ## Methods to construct projection matrices
    def high_to_mortar_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from faces of master to the mortar, by summing quantities
        from the master side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the master side are distributed to the mortar according to the overlap
        between a master face and generally several mortar cells.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with column sum unity.
                Size: g_master.num_faces x mortar_grid.num_cells.

        """
        return self._convert_to_vector_variable(self._high_to_mortar_int, nd)

    def slave_to_mortar_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from cells on the slave side to the mortar, by
        summing quantities from the slave side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the slave side are distributed to the mortar according to the overlap
        between a slave cell and generally several mortar cells.

        This mapping is intended for extensive properties, e.g. sources.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with column sum unity.
                Size: g_slave.num_cells x mortar_grid.num_cells.

        """
        return self._convert_to_vector_variable(self._slave_to_mortar_int, nd)

    def high_to_mortar_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from faces of master to the mortar, by averaging quantities
        from the master side.

        The projection matrix is scaled so that the row sum is unity, that is, values
        on the mortar side are computed as averages of values from the master side,
        according to the overlap between, general several, master faces and a mortar
        cell.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with row sum unity.
                Size: g_master.num_faces x mortar_grid.num_cells.

        """
        scaled_mat = self._row_sum_scaling_matrix(self._high_to_mortar_int)
        return self._convert_to_vector_variable(scaled_mat, nd)

    def slave_to_mortar_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from cells at the slave to the mortar, by averaging
        quantities from the slave side.

        The projection matrix is scaled so that the row sum is unity, that is, values
        on the mortar side are computed as averages of values from the slave side,
        according to the overlap between, generally several master cells and the mortar
        cells.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with row sum unity.
                Size: g_slave.num_cells x mortar_grid.num_cells.

        """
        scaled_mat = self._row_sum_scaling_matrix(self._slave_to_mortar_int)
        return self._convert_to_vector_variable(scaled_mat, nd)

    def _row_sum_scaling_matrix(self, mat):
        # Helper method to construct projection matrices.
        row_sum = mat.sum(axis=1).A.ravel()

        if np.all(row_sum == 1):
            # If only unit scalings, no need to do anything
            return mat

        # Profiling showed that scaling with a csc matrix is quicker than a diagonal
        # matrix. Savings both in construction (!) and multiplication.
        sz = row_sum.size
        indptr = np.arange(sz + 1)
        ind = np.arange(sz)
        scaling = sps.csc_matrix((1.0 / row_sum, ind, indptr), shape=(sz, sz))
        return scaling * mat

    # IMPLEMENTATION NOTE: The reverse projections, from mortar to master/slave are
    # found by taking transposes, and switching average and integration (since we are
    # changing which side we are taking the area relative to.

    def mortar_to_high_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to faces of master, by summing quantities
        from the mortar side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the mortar side are distributed to the master according to the overlap
        between a mortar cell and, generally several master faces.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with column sum unity.
                Size: mortar_grid.num_cells x g_master.num_faces.

        """
        return self._convert_to_vector_variable(self.high_to_mortar_avg().T, nd)

    def mortar_to_low_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to cells at the slave, by summing quantities
        from the mortar side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the mortar side are distributed to the slave according to the overlap
        between a mortar cell and, generally several slave cells.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with column sum unity.
                Size: mortar_grid.num_cells x g_slave_num_faces.

        """
        return self._convert_to_vector_variable(self.slave_to_mortar_avg().T, nd)

    def mortar_to_high_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to faces of master, by averaging
        quantities from the mortar side.

        The projection matrix is scaled so that the row sum is unity, that is, values
        on the master side are computed as averages of values from the mortar side,
        according to the overlap between, general several, mortar cell and a master
        face.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with row sum unity.
                Size: mortar_grid.num_cells x g_master.num_faces.

        """
        return self._convert_to_vector_variable(self.high_to_mortar_int().T, nd)

    def mortar_to_low_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to slave, by averaging quantities from the
        mortar side.

        The projection matrix is scaled so that the row sum is unity, that is, values
        on the slave side are computed as averages of values from the mortar side,
        according to the overlap between, general several, mortar cell and a slave
        cell.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with row sum unity.
                Size: mortar_grid.num_cells x g_slave.num_faces.

        """
        return self._convert_to_vector_variable(self.slave_to_mortar_int().T, nd)

    def _convert_to_vector_variable(
        self, matrix: sps.spmatrix, nd: int
    ) -> sps.spmatrix:
        """Convert the scalar projection to a vector quantity. If the prescribed
        dimension is 1 (default for all the above methods), the projection matrix
        will in effect not be altered.
        """
        if nd == 1:
            # No need to do expansion for 1d variables.
            return matrix
        else:
            return sps.kron(matrix, sps.eye(nd)).tocsc()

    def sign_of_mortar_sides(self, nd: int = 1) -> sps.spmatrix:
        """Assign positive or negative weight to the two sides of a mortar grid.

        This is needed e.g. to make projection operators into signed projections,
        for variables that have no particular defined sign conventions.

        This function defines a convention for what is a positive jump between
        the mortar sides.

        Example: Take the difference between right and left variables, and
        project to the slave grid by

            mortar_to_low_avg() * sign_of_mortar_sides()

        NOTE: The flux variables in flow and transport equations are defined as
        positive from master to slave. Hence the two sides have different
        conventions, and there is no need to adjust the signs further.

        IMPLEMENTATION NOTE: This method will probably not be meaningful if
        applied to mortar grids where the two side grids are non-matching.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.diag_matrix: Diagonal matrix with positive signs on variables
                belonging to the first of the side_grids.
                Size: mortar_grid.num_cells x mortar_grid.num_cells

        """
        nc = self.num_cells
        if self.num_sides() == 1:
            warnings.warn(
                "Is it really meaningful to ask for signs of a one sided mortar grid?"
            )
            return sps.dia_matrix((np.ones(nc * nd), 0), shape=(nd * nc, nd * nc))
        elif self.num_sides() == 2:
            # By the ordering of the mortar cells, we know that all cells on the one
            # side are put first, then the other side. Set + and - accordingly.
            # Implementation note: self.side_grids is a dictionary, not a list, thus
            # the indexing [1] and [2] (and not [0])
            data = np.hstack(
                (
                    -np.ones(self.side_grids[1].num_cells * nd),
                    np.ones(self.side_grids[2].num_cells * nd),
                )
            )
            return sps.dia_matrix((data, 0), shape=(nd * nc, nd * nc))

    def cell_diameters(self) -> np.ndarray:
        diams = np.empty(self.num_sides(), dtype=np.object)
        for pos, (_, g) in enumerate(self.side_grids.items()):
            diams[pos] = g.cell_diameters()
        return np.concatenate(diams).ravel()

    def _check_mappings(self, tol=1e-4) -> None:
        row_sum = self._high_to_mortar_int.sum(axis=1)
        if not (row_sum.min() > tol):
            raise ValueError("Check not satisfied for the master grid")

        row_sum = self._slave_to_mortar_int.sum(axis=1)
        if not (row_sum.min() > tol):
            raise ValueError("Check not satisfied for the slave grid")


class BoundaryMortar(MortarGrid):
    """
    Class for a mortar grid between two grids of the same dimension. This class
    inherits from the MortarGrid class, however, one should be carefull when using
    functions defined in MortarGrid and not BoundaryMortar as not all have been
    thested thoroughly.BoundaryMortar contains a mortar grid and the weighted
    mapping from the slave grid (as set of faces) to the mortar grid and from the
    master grid (as set of faces) to the mortar grid.

    Attributes:

        dim (int): dimension. Should be 0 or 1 or 2.
        side_grids (dictionary of Grid): Contains the mortar grid under the key
            "mortar_grid". Is included for consistency with MortarGrid
        sides (array of integers with values in {0, 1, 2}): ordering of the sides.
        slave_to_mortar_int (sps.csc-matrix): Face-cell relationships between the
            slave grid and the mortar grid. Matrix size:
            num_faces x num_cells. In the beginning we assume matching grids,
            but it can be modified by calling refine_mortar(). The matrix
            elements represent the ratio between the geometrical objects.
        high_to_mortar_int (sps.csc-matrix): face-cell relationships between
            master mortar grid and the mortar grid. Matrix size:
            num_faces x num_cells. Matrix elements represent the ratio between
            the geometrical objects.
        name (list): Information on the formation of the grid, such as the
            constructor, computations of geometry etc.

    """

    def __init__(
        self, dim: int, mortar_grid, master_slave: sps.spmatrix, name: str = ""
    ) -> None:
        """Initialize the mortar grid

        See class documentation for further description of parameters.
        The slave_to_mortar_int and high_to_mortar_int are identity mapping.

        Parameters
        ----------
        dim (int): grid dimension
        mortar_grid (pp.MortarGrid): mortar grid. It is assumed that there is a
            one to one mapping between the cells of the mortar grid and the
            faces of the slave grid given in master_slave
        master_slave (sps.csc_matrix): face-face relations between the slave
            grid and the master dimensional grid.
        name (str): Name of grid
        """

        if not dim >= 0 and dim < 3:
            raise ValueError("Mortar grid dimension must be 0, 1 or 2")
        if not mortar_grid.dim == dim:
            raise ValueError("Dimension of mortar grid does not match given dimension")

        self.dim = dim
        self.side_grids = {0: mortar_grid}
        self.sides = np.array(list(self.side_grids.keys()))

        if not (self.num_sides() == 1 or self.num_sides() == 2):
            raise ValueError("The number of sides have to be 1 or 2")

        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]
        self.name.append("mortar_grid")
        # master_slave is a mapping from the faces of the slave grid to the
        # faces of the master grid.
        # We assume that, in the beginning the mortar grids are equal
        # to the slave grid. If this assumption is not satisfied we
        # need to change the following lines
        slave_f, master_f, data = sps.find(master_slave)

        # It is assumed that the cells of the given mortar grid are ordered
        # by the face numbers of the slave side
        ix = np.argsort(slave_f)
        slave_f = slave_f[ix]
        master_f = master_f[ix]
        data = data[ix]

        # Define mappings
        cells = np.arange(slave_f.size)
        self.num_cells = cells.size
        if not self.num_cells == mortar_grid.num_cells:
            raise ValueError(
                """In the construction of Boundary mortar it is assumed
            to be a one to one mapping between the mortar grid and the contact faces of
            the slave grid"""
            )
        self.cell_volumes = mortar_grid.cell_volumes

        shape_master = (self.num_cells, master_slave.shape[1])
        shape_slave = (self.num_cells, master_slave.shape[0])
        self._high_to_mortar_int = sps.csc_matrix(
            (data.astype(np.float), (cells, master_f)), shape=shape_master
        )
        self._slave_to_mortar_int = sps.csc_matrix(
            (data.astype(np.float), (cells, slave_f)), shape=shape_slave
        )

    def __repr__(self) -> str:
        """
        Implementation of __repr__

        """
        s = (
            "Mortar grid with history "
            + ", ".join(self.name)
            + "\n"
            + "Dimension "
            + str(self.dim)
            + "\n"
            + "Face_cell mapping from the LEFT_SIDE grid to the mortar grid\n"
            + str(self.high_to_mortar_int)
            + "\n"
            + "Face_cell mapping from the SLAVE_SIDE grid to the mortar grid\n"
            + str(self.slave_to_mortar_int)
        )

        return s

    def __str__(self) -> str:
        """
        Implementation of __str__
        """
        s = str()

        s += "".join(
            [
                "Side " + str(s) + " with grid:\n" + str(g)
                for s, g in self.side_grids.items()
            ]
        )

        s += (
            "Mapping from the faces of the master_side grid to"
            + " the cells of the mortar grid. \nRows indicate the mortar"
            + " cell id, columns indicate the master_grid face id"
            + "\n"
            + str(self.high_to_mortar_int)
            + "\n"
            + "Mapping from the cells of the face of the slave_side grid"
            + "to the cells of the mortar grid. \nRows indicate the mortar"
            + " cell id, columns indicate the slave_grid face id"
            + "\n"
            + str(self.slave_to_mortar_int)
        )

        return s

    def num_sides(self) -> int:
        """
        Shortcut to compute the number of sides, it has to be 2 or 1.

        Return:
            Number of sides.
        """
        return len(self.side_grids)

    def compute_geometry(self) -> None:
        """
        Compute the geometry of the mortar grids.
        We assume that they are not aligned with x (1d) or x, y (2d).
        """
        for g in self.side_grids.values():
            g.compute_geometry()


# --- helper methods


def _split_matrix_1d(g_old: pp.Grid, g_new: pp.Grid, tol: float) -> sps.spmatrix:
    """
    By calling matching grid the function compute the cell mapping between two
    different grids.

    It is asumed that the two grids are aligned, with common start and
    endpoints. However, their nodes can be ordered in oposite directions.

    Parameters:
        g_old (Grid): the first (old) grid.
        g_new (Grid): the second (new) grid.
        tol (double): Tolerance in the matching of the grids

    Return:
        csr matrix: representing the cell mapping. The entries are the relative
            cell measure between the two grids.

    """
    weights, new_cells, old_cells = pp.match_grids.match_1d(g_new, g_old, tol)
    shape = (g_new.num_cells, g_old.num_cells)
    return sps.csr_matrix((weights, (new_cells, old_cells)), shape=shape)


def _split_matrix_2d(g_old: pp.Grid, g_new: pp.Grid, tol: float) -> sps.spmatrix:
    """
    By calling matching grid the function compute the cell mapping between two
    different grids.

    It is asumed that the two grids have common boundary.

    Parameters:
        g_old (Grid): the first (old) grid.
        g_new (Grid): the second (new) grid.
        tol (double): Tolerance in the matching of the grids

    Return:
        csr matrix: representing the cell mapping. The entries are the relative
            cell measure between the two grids.

    """
    weights, new_cells, old_cells = pp.match_grids.match_2d(g_new, g_old, tol)
    shape = (g_new.num_cells, g_old.num_cells)
    return sps.csr_matrix((weights, (new_cells, old_cells)), shape=shape)
