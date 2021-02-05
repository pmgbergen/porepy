""" Module containing the class for the mortar grid.
"""
import warnings
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse as sps

import porepy as pp

module_sections = ["grids", "gridding"]


class MortarSides(Enum):
    # Enum of constants used to identify the grids on each side of the mortar
    NONE_SIDE = 0
    LEFT_SIDE = 1
    RIGHT_SIDE = 2


class MortarGrid:
    """
    Parent class for mortar grids it contains two grids representing the left
    and right part of the mortar grid and the weighted mapping from the primary
    grid to the mortar grids and from the secondary grid to the mortar grids.
    The two mortar grids can be different. The primary grid is assumed to be one
    dimension higher than the mortar grids, while the secondary grid can either one
    dimension higher or the same dimension as the mortar grids.

    NOTE: The mortar class is mostly tested for the case when the secondary grid has
    the same dimension as the mortar grid. Especially, the updating of any grid
    should not be expected to work and will most likely throw an error.

    Attributes:

        dim (int): dimension. Should be 0 or 1 or 2.
        side_grids (dictionary of Grid): grid for each side. The key is an integer
            with value {0, 1, 2}, and the value is a Grid.
        sides (array of integers with values in {0, 1, 2}): ordering of the sides.
        _primary_to_mortar_int (sps.csc-matrix): Face-cell relationships between the
            primary (often higher-dimensional) grid and the mortar grids. Matrix size:
            num_faces x num_cells. In the beginning we assume matching grids,
            but it can be modified by calling refine_mortar(). The matrix
            elements represent the ratio between the geometrical objects.
        _secondary_to_mortar_int (sps.csc-matrix): Cell-cell relationships between the
            mortar grids and the secondary (often lower-dimensional) grid. Matrix size:
            num_cells_secondary x num_cells_mortar. Matrix elements represent the ratio
            between the geometrical objects.
        name (list): Information on the formation of the grid, such as the
            constructor, computations of geometry etc.
        tol (double): Tolerance use when matching grids during update of mortar or
            primary / secondary grids.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(
        self,
        dim: int,
        side_grids: Dict[MortarSides, pp.Grid],
        primary_secondary: sps.spmatrix = None,
        name: Union[str, List[str]] = "",
        face_duplicate_ind: Optional[np.ndarray] = None,
        tol: float = 1e-6,
    ):
        """Initialize the mortar grid

        See class documentation for further description of parameters.

        Parameters:
            dim (int): grid dimension
            side_grids (dictionary of Grid): grid on each side.
            primary_secondary (sps.csc_matrix): Cell-face relations between the higher
                dimensional grid and the lower dimensional grid. It is possible to not
                give the projection to create only the grid.
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
        self.side_grids: Dict[MortarSides, pp.Grid] = side_grids.copy()
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
        self.num_cells: int = np.sum(  # type: ignore
            [g.num_cells for g in self.side_grids.values()], dtype=int
        )
        self.cell_volumes: np.ndarray = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )
        self.cell_centers: np.ndarray = np.hstack(
            [g.cell_centers for g in self.side_grids.values()]
        )
        # Set projections
        if not (primary_secondary is None):
            self._init_projections(primary_secondary, face_duplicate_ind)

    @pp.time_logger(sections=module_sections)
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
            + f"{self.mortar_to_secondary_int().shape[0]}\n"
            + "Number of faces in higher-dimensional neighbor "
            + f"{self.mortar_to_primary_int().shape[0]}\n"
        )

        return s

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def compute_geometry(self) -> None:
        """
        Compute the geometry of the mortar grids.
        We assume that they are not aligned with x (1d) or x, y (2d).
        """
        # Update the actual side grids
        for g in self.side_grids.values():
            g.compute_geometry()

        # Update the attributes
        self.num_cells = np.sum(  # type: ignore
            [g.num_cells for g in self.side_grids.values()], dtype=int
        )
        self.cell_volumes = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )
        self.cell_centers = np.hstack(
            [g.cell_centers for g in self.side_grids.values()]
        )

    ### Methods to update the mortar grid, or the neighboring grids.

    @pp.time_logger(sections=module_sections)
    def update_mortar(
        self, new_side_grids: Dict[MortarSides, pp.Grid], tol: float = None
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
        matrix_blocks: np.ndarray = np.empty(
            (self.num_sides(), self.num_sides()), dtype=object
        )

        # Loop on all the side grids, if not given an identity matrix is
        # considered
        for pos, (side, g) in enumerate(self.side_grids.items()):
            matrix_blocks[pos, pos] = split_matrix.get(side, sps.identity(g.num_cells))

        # Once the global matrix is constructed the new low_to_mortar_int and
        # high_to_mortar_int maps are updated.
        matrix: sps.spmatrix = sps.bmat(matrix_blocks)
        self._secondary_to_mortar_int: sps.spmatrix = (
            matrix * self._secondary_to_mortar_int
        )
        self._primary_to_mortar_int: sps.spmatrix = matrix * self._primary_to_mortar_int

        # Update the side grids
        for side, g in new_side_grids.items():
            self.side_grids[side] = g.copy()

        # update the geometry
        self.compute_geometry()

        self._check_mappings()

    @pp.time_logger(sections=module_sections)
    def update_secondary(self, new_g: pp.Grid, tol: float = None) -> None:
        """
        Update the _secondary_to_mortar_int map when the lower dimensional grid is changed.

        Parameter:
            new_g (pp.Grid): The new secondary grid.
            tol (double, optional): Tolerance used for matching the new and old grids.
                Defaults to self.tol.

        """
        if tol is None:
            tol = self.tol

        if self.dim != new_g.dim:
            raise NotImplementedError(
                """update_ssecondary() is only implemented when the secondary
                grid has the same dimension as the mortar grid"""
            )

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
        matrix = np.empty((self.num_sides(), 1), dtype=object)

        for pos, (side, _) in enumerate(self.side_grids.items()):
            matrix[pos, 0] = split_matrix[side]

        # Update the low_to_mortar_int map. No need to update the high_to_mortar_int.
        self._secondary_to_mortar_int = sps.bmat(matrix, format="csc")
        self._check_mappings()

    @pp.time_logger(sections=module_sections)
    def update_primary(self, g_new: pp.Grid, g_old: pp.Grid, tol: float = None):
        """

        Update the _primary_to_mortar_int map when the primary (higher-dimensional) grid is
        changed.

        Parameter:
            g_new (pp.Grid): The new primary grid.
            g_old (pp.Grid): The old primary grid.
            tol (double, optional): Tolerance used for matching the new and old grids.
                Defaults to self.tol.
        """
        # IMPLEMENTATION NOTE: The signature of this method is different from
        # update_secondary(), since the latter must also take care of for the side grids.

        if tol is None:
            tol = self.tol

        if self.dim == 0:

            # retrieve the old faces and the corresponding coordinates
            _, old_faces, _ = sps.find(self._primary_to_mortar_int)
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
        self._primary_to_mortar_int = self._primary_to_mortar_int * split_matrix
        self._check_mappings()

    @pp.time_logger(sections=module_sections)
    def num_sides(self) -> int:
        """
        Shortcut to compute the number of sides, it has to be 2 or 1.

        Return:
            Number of sides.
        """
        return len(self.side_grids)

    @pp.time_logger(sections=module_sections)
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
    @pp.time_logger(sections=module_sections)
    def primary_to_mortar_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from faces of primary to the mortar, by summing quantities
        from the primary side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the primary side are distributed to the mortar according to the overlap
        between a primary face and generally several mortar cells.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with column sum unity.
                Size: g_primary.num_faces x mortar_grid.num_cells.

        """
        return self._convert_to_vector_variable(self._primary_to_mortar_int, nd)

    @pp.time_logger(sections=module_sections)
    def secondary_to_mortar_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from cells on the secondary side to the mortar, by
        summing quantities from the secondary side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the secondary side are distributed to the mortar according to the overlap
        between a secondary cell and generally several mortar cells.

        This mapping is intended for extensive properties, e.g. sources.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with column sum unity.
                Size: g_secondary.num_cells x mortar_grid.num_cells.

        """
        return self._convert_to_vector_variable(self._secondary_to_mortar_int, nd)

    @pp.time_logger(sections=module_sections)
    def primary_to_mortar_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from faces of primary to the mortar, by averaging quantities
        from the primary side.

        The projection matrix is scaled so that the row sum is unity, that is, values
        on the mortar side are computed as averages of values from the primary side,
        according to the overlap between, general several, primary faces and a mortar
        cell.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with row sum unity.
                Size: g_primary.num_faces x mortar_grid.num_cells.

        """
        scaled_mat = self._row_sum_scaling_matrix(self._primary_to_mortar_int)
        return self._convert_to_vector_variable(scaled_mat, nd)

    @pp.time_logger(sections=module_sections)
    def secondary_to_mortar_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from cells at the secondary to the mortar, by averaging
        quantities from the secondary side.

        The projection matrix is scaled so that the row sum is unity, that is, values
        on the mortar side are computed as averages of values from the secondary side,
        according to the overlap between, generally several primary cells and the mortar
        cells.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with row sum unity.
                Size: g_secondary.num_cells x mortar_grid.num_cells.
        """
        scaled_mat = self._row_sum_scaling_matrix(self._secondary_to_mortar_int)
        return self._convert_to_vector_variable(scaled_mat, nd)

    @pp.time_logger(sections=module_sections)
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

    # IMPLEMENTATION NOTE: The reverse projections, from mortar to primary/secondary are
    # found by taking transposes, and switching average and integration (since we are
    # changing which side we are taking the area relative to.

    @pp.time_logger(sections=module_sections)
    def mortar_to_primary_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to faces of primary, by summing quantities
        from the mortar side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the mortar side are distributed to the primary according to the overlap
        between a mortar cell and, generally several primary faces.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with column sum unity.
                Size: mortar_grid.num_cells x g_primary.num_faces.

        """
        return self._convert_to_vector_variable(self.primary_to_mortar_avg().T, nd)

    @pp.time_logger(sections=module_sections)
    def mortar_to_secondary_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to cells at the secondary, by summing quantities
        from the mortar side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the mortar side are distributed to the secondary according to the overlap
        between a mortar cell and, generally several secondary cells.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with column sum unity.
                Size: mortar_grid.num_cells x g_secondary.num_faces.


        """
        return self._convert_to_vector_variable(self.secondary_to_mortar_avg().T, nd)

    @pp.time_logger(sections=module_sections)
    def mortar_to_primary_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to faces of primary, by averaging
        quantities from the mortar side.

        The projection matrix is scaled so that the row sum is unity, that is, values
        on the primary side are computed as averages of values from the mortar side,
        according to the overlap between, general several, mortar cell and a primary
        face.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with row sum unity.
                Size: mortar_grid.num_cells x g_primary.num_faces.

        """
        return self._convert_to_vector_variable(self.primary_to_mortar_int().T, nd)

    @pp.time_logger(sections=module_sections)
    def mortar_to_secondary_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to secondary, by averaging quantities from the
        mortar side.

        The projection matrix is scaled so that the row sum is unity, that is, values
        on the secondary side are computed as averages of values from the mortar side,
        according to the overlap between, general several, mortar cell and a secondary
        cell.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd (int, optional): Spatial dimension of the projected quantity.
                Defaults to 1 (mapping for scalar quantities).

        Returns:
            sps.matrix: Projection matrix with row sum unity.
                Size: mortar_grid.num_cells x g_secondary.num_faces.

        """
        return self._convert_to_vector_variable(self.secondary_to_mortar_int().T, nd)

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def sign_of_mortar_sides(self, nd: int = 1) -> sps.spmatrix:
        """Assign positive or negative weight to the two sides of a mortar grid.

        This is needed e.g. to make projection operators into signed projections,
        for variables that have no particular defined sign conventions.

        This function defines a convention for what is a positive jump between
        the mortar sides.

        Example: Take the difference between right and left variables, and
        project to the secondary grid by

            mortar_to_secondary_avg() * sign_of_mortar_sides()

        NOTE: The flux variables in flow and transport equations are defined as
        positive from primary to secondary. Hence the two sides have different
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
            data = np.hstack(
                (
                    -np.ones(self.side_grids[MortarSides.LEFT_SIDE].num_cells * nd),
                    np.ones(self.side_grids[MortarSides.RIGHT_SIDE].num_cells * nd),
                )
            )
            return sps.dia_matrix((data, 0), shape=(nd * nc, nd * nc))

    @pp.time_logger(sections=module_sections)
    def cell_diameters(self) -> np.ndarray:
        diams = np.empty(self.num_sides(), dtype=object)
        for pos, (_, g) in enumerate(self.side_grids.items()):
            diams[pos] = g.cell_diameters()
        return np.concatenate(diams).ravel()

    @pp.time_logger(sections=module_sections)
    def _check_mappings(self, tol=1e-4) -> None:
        row_sum = self._primary_to_mortar_int.sum(axis=1)
        if not (row_sum.min() > tol):
            raise ValueError("Check not satisfied for the primary grid")

        row_sum = self._secondary_to_mortar_int.sum(axis=1)
        if not (row_sum.min() > tol):
            raise ValueError("Check not satisfied for the secondary grid")

    @pp.time_logger(sections=module_sections)
    def _init_projections(
        self,
        primary_secondary: sps.spmatrix,
        face_duplicate_ind: Optional[np.ndarray] = None,
    ):
        """Initialize projections from primary and secondary to mortar.

        Parameters:
        primary_secondary (sps.spmatrix): projection from the primary to the secondary.
            It is assumed that the primary, secondary and mortar grids are all matching.
        face_duplicate_ind (np.ndarray, optional): Which faces should be considered
                duplicates, and mapped to the second of the side_grids. If not provided,
                duplicate faces will be inferred from the indices of the faces. Will
                only be used if len(side_Grids) == 2.

        """
        # primary_secondary is a mapping from the cells (faces if secondary.dim == primary.dim)
        # of the secondary grid to the of the primary grid.
        secondary_f, primary_f, data = sps.find(primary_secondary)

        # If the face_duplicate_ind is given we have to reorder the primary face indices
        # such that the original faces comes first, then the duplicate faces.
        # If the face_duplicate_ind is not given, we then assume that the primary side faces
        # already have this ordering. If the grid is created using the pp.split_grid.py
        # module this should be the case.
        if self.num_sides() == 2 and (face_duplicate_ind is not None):
            is_second_side = np.in1d(primary_f, face_duplicate_ind)
            secondary_f = np.r_[
                secondary_f[~is_second_side], secondary_f[is_second_side]
            ]
            primary_f = np.r_[primary_f[~is_second_side], primary_f[is_second_side]]
            data = np.r_[data[~is_second_side], data[is_second_side]]

        # Store index of the faces on the 'other' side of the mortar grid.
        if self.num_sides() == 2:
            # After the above sorting, we know that the faces on the other side is in the
            # second half of primary_f, also if face_duplicate_ind is given.
            # ASSUMPTION: The mortar grids on the two sides should match each other, or
            # else, the below indexing is wrong. This also means that the size of primary_f
            # is an even number.
            sz = int(primary_f.size / 2)
            self._ind_face_on_other_side = primary_f[sz:]

        # We assumed that the cells of the given side grid(s) is(are) ordered
        # by the secondary side index. In other words: cell "n" of the side grid(s) should
        # correspond to the element with the n'th lowest index in secondary_f. We therefore
        # sort secondary_f to obtaint the side grid ordering. The primary faces should now be
        # sorted such that the left side comes first, then the right side. We use stable
        # sort to not mix up the ordering if there is two sides.
        ix = np.argsort(secondary_f, kind="stable")
        if self.num_sides() == 2:
            # If there are two sides we are in the case of a secondary grid of equal
            # dimension as the mortar grid. The mapping primary_secondary is then a mapping
            # from faces-cells, and we assume the higher dimensional grid is split and
            # there is exactly two primary faces mapping to each secondary cell. Check this:
            if not np.allclose(np.bincount(secondary_f), 2):
                raise ValueError(
                    """Each face in the higher dimensional grid must map to
                exactly two lower dimensional cells"""
                )
            # The mortar grid cells are ordered as first all cells on side 1 then all
            # cells on side 2. We there have to reorder ix to account for this:
            ix = np.reshape(ix, (2, -1), order="F").ravel("C")

        # Reorder mapping to fit with mortar cell ordering.
        secondary_f = secondary_f[ix]
        primary_f = primary_f[ix]
        data = data[ix]

        # Define mappings
        cells = np.arange(secondary_f.size)
        if not self.num_cells == cells.size:
            raise ValueError(
                """In the construction of MortarGrid it is assumed
            to be a one to one mapping between the mortar grid and the secondary mapping"""
            )

        shape_primary = (self.num_cells, primary_secondary.shape[1])
        shape_secondary = (self.num_cells, primary_secondary.shape[0])
        self._primary_to_mortar_int = sps.csc_matrix(
            (data.astype(float), (cells, primary_f)), shape=shape_primary
        )
        self._secondary_to_mortar_int = sps.csc_matrix(
            (data.astype(float), (cells, secondary_f)), shape=shape_secondary
        )


@pp.time_logger(sections=module_sections)
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


@pp.time_logger(sections=module_sections)
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
