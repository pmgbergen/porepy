"""A module containing the class for the mortar grid, a geometric representation of
interfaces between two subdomains in the mixed-dimensional sense.

"""

from __future__ import annotations

from enum import Enum
from itertools import count
from typing import Generator, Optional, Union

import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.numerics.linalg.matrix_operations import (
    sparse_array_to_row_col_data,
    sparse_kronecker_product,
)


class MortarSides(Enum):
    """A custom enumeration for the left, the right and empty side of an
    interface represented by a mortar grid."""

    # Enum of constants used to identify the grids on each side of the mortar
    NONE_SIDE = 0
    LEFT_SIDE = 1
    RIGHT_SIDE = 2


class MortarGrid:
    """Parent class for mortar grids.

    It contains two grids representing the left and right part of the mortar grid and
    the weighted mapping from the primary grid to the mortar grids and from the
    secondary grid to the mortar grids. The two mortar grids can be different. The
    primary grid is assumed to be one dimension higher than the mortar grids, while the
    secondary grid can either one dimension higher or the same dimension as the mortar
    grids.

    Note:
        The mortar class is mostly tested for the case when the secondary grid has the
        same dimension as the mortar grid. Especially, the updating of any grid should
        not be expected to work and will most likely throw an error.

    Parameters:
        dim: Grid dimension.
        side_grids: Grid on each side.
        primary_secondary: ``default=None``

            Cell-face relations between the higher dimensional grid and
            the lower dimensional grid. It is possible to not give the projection to
            create only the grid.
        codim: ``default=1``

            Dimension difference between the primary grid secondary grid.
        name: ``default=''``

            Name of the grid. Can also be used to set various information on the grid.
        face_duplicate_ind: ``default=None``

            Which faces should be considered duplicates, and
            mapped to the second of the ``side_grids``.

            If not provided, duplicate faces will be inferred from the indices of the
            faces.

            Will only be used if ``len(side_Grids)==2``.
        tol: ``default=1e-6``

            Tolerance used in geometric computations.

    Raises:
        ValueError: If ``dim==3`` , The mortar grid can not be three-dimensional.
        ValueError: If the mortar grids have different dimensions.
        ValueError: If the number of sides is not 1 or 2.
        ValueError: If ``face_duplicate_ind`` is not ``None`` and the co-dimension is 2.
            In this case there are no faces to duplicate.

    """

    _counter = count(0)
    """Counter of instantiated mortar grids. See :meth:`__new__` and :meth:`id`."""

    def __new__(cls, *args, **kwargs):
        """Make object and set ID by forwarding :attr:`_counter`."""

        obj = object.__new__(cls)
        obj.__id = next(cls._counter)
        return obj

    def __init__(
        self,
        dim: int,
        side_grids: dict[MortarSides, pp.Grid],
        primary_secondary: Optional[sps.spmatrix] = None,
        codim: int = 1,
        name: Union[str, list[str]] = "",
        face_duplicate_ind: Optional[np.ndarray] = None,
        tol: float = 1e-6,
    ) -> None:
        if dim == 3:
            raise ValueError("A mortar grid cannot be 3d")
        if not np.all([g.dim == dim for g in side_grids.values()]):
            raise ValueError("All the mortar grids have to have the same dimension")

        self.dim = dim
        """The ambient dimension of the mortar grid."""

        self.codim = codim
        """The co-dimension of the mortar grid."""

        self.side_grids: dict[MortarSides, pp.Grid] = side_grids.copy()
        """A dictionary containing for each enumeration :class:`MortarSides` the
        respective side grid."""

        self.sides: np.ndarray = np.array(list(self.side_grids.keys()))
        """An array containing the enumeration of each side grid
        (keys of :attr:`side_grids`)."""

        if not (self.num_sides() == 1 or self.num_sides() == 2):
            raise ValueError("The number of sides have to be 1 or 2")
        if face_duplicate_ind is not None and codim == 2:
            raise ValueError("Co-dimension 2 interfaces have no faces to duplicate")

        self.name: list[str]
        """The name of mortar grid stored as a list of relevant information."""
        if isinstance(name, list):
            self.name = name
        else:
            self.name = [name]
        self.name.append("mortar_grid")

        self.tol = tol
        """The tolerance given at instantiation."""

        # easy access attributes with a fixed ordering of the side grids
        self.num_cells: int = np.sum(  # type: ignore
            [g.num_cells for g in self.side_grids.values()], dtype=int
        )
        """Total number of cells in all side grids."""

        self.num_nodes: int = np.sum(  # type: ignore
            [g.num_nodes for g in self.side_grids.values()], dtype=int
        )
        """Total number of nodes in all side grids."""

        self.cell_volumes: np.ndarray = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )
        """Cell volumes of each cell in the side grids."""

        self.cell_centers: np.ndarray = np.hstack(
            [g.cell_centers for g in self.side_grids.values()]
        )
        """Cell centers of each cell in the side grids."""

        self.nodes: np.ndarray = np.hstack([g.nodes for g in self.side_grids.values()])
        """Nodes in the side grids."""

        # Set projections
        if not (primary_secondary is None):
            self._init_projections(primary_secondary, face_duplicate_ind)
            self._set_projections()

    @property
    def id(self):
        """Grid ID.

        The returned attribute must not be changed.
        This may severely compromise other parts of the code,
        such as sorting in md grids.

        The attribute is set in :meth:`__new__`.
        This avoids calls to the super constructor in child classes.

        """
        return self.__id

    def __repr__(self) -> str:
        """A string representation of the mortar grid including topological information
        and information about side grids."""

        s = (
            "Mortar grid with history "
            + ", ".join(self.name)
            + f" and id {self.id}.\n"
            + f"Dimension {self.dim} and codimension {self.codim}\n"
            + f"Number of cells {self.num_cells}\n"
            + f"Number of nodes {self.num_nodes}\n"
            + f"Number of sides {len(self.side_grids)}\n"
            + "Number of cells in lower-dimensional neighbor "
            + f"{self.mortar_to_secondary_int().shape[0]}\n"
        )
        if self.codim < 2:
            s += (
                "Number of faces in higher-dimensional neighbor "
                + f"{self.mortar_to_primary_int().shape[0]}\n"
            )
        else:
            s += (
                "Number of cells in higher-dimensional neighbor "
                + f"{self.mortar_to_primary_int().shape[0]}\n"
            )

        return s

    def __str__(self) -> str:
        """A simplified string representation of the mortar grid including the dimension
        and some other information."""

        s = (
            "Mortar grid with history "
            + ", ".join(self.name)
            + f" and id {self.id}.\n"
            + f"Dimension {self.dim} and codimension {self.codim}\n"
            + f"Number of cells {self.num_cells}"
            + f"Number of nodes {self.num_nodes}"
        )
        return s

    def compute_geometry(self) -> None:
        """Compute the geometry of the mortar grids.

        We assume that they are not aligned with the x-axis (1D) or the x-y-plane (2D).

        Performs calls to each grid's :meth:`~porepy.grids.grid.Grid.compute_geometry`.

        """
        # Update the actual side grids
        for g in self.side_grids.values():
            g.compute_geometry()

        # Update the attributes
        self.num_cells = np.sum(  # type: ignore
            [g.num_cells for g in self.side_grids.values()], dtype=int
        )
        self.num_nodes = np.sum(  # type: ignore
            [g.num_nodes for g in self.side_grids.values()], dtype=int
        )
        self.cell_volumes = np.hstack(
            [g.cell_volumes for g in self.side_grids.values()]
        )
        self.cell_centers = np.hstack(
            [g.cell_centers for g in self.side_grids.values()]
        )
        self.nodes = np.hstack([g.nodes for g in self.side_grids.values()])

    ### Methods to update the mortar grid, or the neighboring grids.

    def update_mortar(
        self, new_side_grids: dict[MortarSides, pp.Grid], tol: Optional[float] = None
    ) -> None:
        """Update the ``low_to_mortar_int`` and ``high_to_mortar_int`` maps when the
        mortar grids are changed.

        Parameters:
            new_side_grids: A dictionary containing for each side
                (identified with the enumeration :class:`MortarSides`) a matrix
                representing the new mapping between the old and new mortar grids.
            tol: ``default=None``

                Tolerance used for matching the new and old grids.

                If not provided, :attr:`tol` is used.

        Raises:
            ValueError: If the old and new mortar grids are not of the same dimension.
            ValueError: If the mortar grid is not of dimension 0,1 or 2.

        """
        if tol is None:
            tol = self.tol

        # Build mappings for integrated and averaged quantities separately.
        split_matrix_int, split_matrix_avg = {}, {}

        # For each side we compute the mapping between the old and the new mortar grids,
        # we store them in a dictionary with SideTag as key.
        for side, new_g in new_side_grids.items():
            g = self.side_grids[side]
            if g.dim != new_g.dim:
                raise ValueError("Grid dimension has to be the same")

            if g.dim == 0:
                # Nothing to do
                return
            elif g.dim == 1:
                # The mapping between grids will be left-multiplied by the existing
                # mapping from primary to mortar. Therefore, we construct mappings from
                # the old to the new grid.
                mat_avg = pp.match_grids.match_1d(new_g, g, tol, scaling="averaged")
                mat_int = pp.match_grids.match_1d(new_g, g, tol, scaling="integrated")
            elif g.dim == 2:
                mat_avg = pp.match_grids.match_2d(new_g, g, tol, scaling="averaged")
                mat_int = pp.match_grids.match_2d(new_g, g, tol, scaling="integrated")
            else:
                # No 3d mortar grid
                raise ValueError

            # Store values
            split_matrix_avg[side] = mat_avg
            split_matrix_int[side] = mat_int

        # In the case of different side ordering between the input data and the stored
        # we need to remap it. The resulting matrix will be a block diagonal matrix,
        # where in each block we have the mapping between the (relative to side) old
        # grid and the new one.
        matrix_blocks_avg: np.ndarray = np.empty(
            (self.num_sides(), self.num_sides()), dtype=object
        )
        matrix_blocks_int: np.ndarray = np.empty(
            (self.num_sides(), self.num_sides()), dtype=object
        )

        # Loop on all the side grids, if not given an identity matrix is
        # considered
        for pos, (side, g) in enumerate(self.side_grids.items()):
            matrix_blocks_avg[pos, pos] = split_matrix_avg.get(
                side, sps.identity(g.num_cells)
            )
            matrix_blocks_int[pos, pos] = split_matrix_int.get(
                side, sps.identity(g.num_cells)
            )

        # Once the global matrix is constructed the new primary_to_mortar_int and
        # secondary_to_mortar_int maps are updated.
        matrix_avg: sps.spmatrix = sps.bmat(matrix_blocks_avg)
        matrix_int: sps.spmatrix = sps.bmat(matrix_blocks_int)

        # We need to update mappings from both primary and secondary.
        # Use optimized storage to minimize memory consumption.
        self._primary_to_mortar_avg: sps.spmatrix = (
            pp.matrix_operations.optimized_compressed_storage(
                matrix_avg * self._primary_to_mortar_avg
            )
        )
        self._primary_to_mortar_int: sps.spmatrix = (
            pp.matrix_operations.optimized_compressed_storage(
                matrix_int * self._primary_to_mortar_int
            )
        )
        self._secondary_to_mortar_avg: sps.spmatrix = (
            pp.matrix_operations.optimized_compressed_storage(
                matrix_avg * self._secondary_to_mortar_avg
            )
        )
        self._secondary_to_mortar_int: sps.spmatrix = (
            pp.matrix_operations.optimized_compressed_storage(
                matrix_int * self._secondary_to_mortar_int
            )
        )

        # Also update the other mappings
        self._set_projections()

        # Update the side grids
        for side, g in new_side_grids.items():
            self.side_grids[side] = g.copy()

        # update the geometry
        self.compute_geometry()

        self._check_mappings()

    def update_secondary(self, new_g: pp.Grid, tol: Optional[float] = None) -> None:
        """Update the mappings between mortar and secondary grid when the latter is
        changed.

        Note:
            This function assumes that the secondary grid is only updated once: A change
            from matching to non-matching between the mortar and secondary grids is
            okay, but replacing a non-matching secondary grid with another one will not
            work.

        Parameters:
            new_g: The new secondary grid.
            tol: ``default=None``

                Tolerance used for matching the new and old grids.

                If not provided, :attr:`tol` is used.

        Raises:
            NotImplementedError: If the new secondary grid and the mortar grid are not
                of the same dimension.
            ValueError: If the old and new secondary grids are not of the same
                dimension.
            ValueError: If the mortar grid is not of dimension 0,1 or 2.

        """
        if tol is None:
            tol = self.tol

        if self.dim != new_g.dim:
            raise NotImplementedError(
                """update_secondary() is only implemented when the secondary
                grid has the same dimension as the mortar grid"""
            )

        # Build mappings for integrated and averaged quantities separately.
        split_matrix_avg, split_matrix_int = {}, {}

        # For each side we compute the mapping between the new lower dimensional grid
        # and the mortar grid, we store them in a dictionary with MortarSide as key.
        # IMPLEMENTATION NOTE: The loop, and some of the complexity below, is necessary
        # to allow for different grids on the mortar sides (if more than one). It is not
        # clear that our full implementation supports this (no obvious weak points), but
        # AFAIK, it has never been tested, but we will keep the generality for now.
        for side, g in self.side_grids.items():
            if g.dim != new_g.dim:
                raise ValueError("Grid dimension has to be the same")

            if self.dim == 0:
                # Nothing to do, all 0d grids are matching
                return
            elif self.dim == 1:
                # We need to map from the new to the old grid.
                # See below for ideas of how to allow for updating the secondary grid
                # more than once.
                mat_avg = pp.match_grids.match_1d(g, new_g, tol, scaling="averaged")
                mat_int = pp.match_grids.match_1d(g, new_g, tol, scaling="integrated")
            elif self.dim == 2:
                mat_avg = pp.match_grids.match_2d(g, new_g, tol, scaling="averaged")
                mat_int = pp.match_grids.match_2d(g, new_g, tol, scaling="integrated")
            else:
                # No 3d mortar grid
                raise ValueError

            split_matrix_avg[side] = mat_avg
            split_matrix_int[side] = mat_int
        # In the case of different side ordering between the input data and the stored
        # we need to remap it. The resulting matrix will be a block matrix, where in
        # each block we have the mapping between the (relative to side) the new grid and
        # the mortar grid.
        matrix_avg = np.empty((self.num_sides(), 1), dtype=object)
        matrix_int = np.empty((self.num_sides(), 1), dtype=object)

        for pos, (side, _) in enumerate(self.side_grids.items()):
            matrix_avg[pos] = split_matrix_avg[side]
            matrix_int[pos] = split_matrix_int[side]

        # IMPLEMENTATION NOTE: To allow for replacing the secondary grid more than once,
        # it is necessary to replace the line above into an update of the mapping from
        # secondary (not only overwriting as we do now). That should be possible, but
        # requires more thinking.
        self._secondary_to_mortar_avg = sps.bmat(matrix_avg, format="csc")
        self._secondary_to_mortar_int = sps.bmat(matrix_int, format="csc")

        # Update other mappings to and from secondary
        self._set_projections(primary=False)

        self._check_mappings()

    def update_primary(
        self, g_new: pp.Grid, g_old: pp.Grid, tol: Optional[float] = None
    ) -> None:
        """Update the ``_primary_to_mortar_int`` map when the primary
        (higher-dimensional) grid is changed.

        Parameters:
            g_new: The new primary grid.
            g_old: The old primary grid.
            tol: ``default=None``

                Tolerance used for matching the new and old grids.

                If not provided, :attr:`tol` is used.

        Raises:
            ValueError: For 0d mortar grids, if the faces of the old primary grid do
                not correspond to the same physical point.
            NotImplementedError: If the dimension of the mortar grid is >1 (this has not
                been implemented yet).

        """
        # IMPLEMENTATION NOTE: The signature of this method is different from
        # update_secondary(), since the latter must also take care of for the side
        # grids.

        if tol is None:
            tol = self.tol

        if self.dim == 0:
            # retrieve the old faces and the corresponding coordinates
            _, old_faces, _ = sparse_array_to_row_col_data(self._primary_to_mortar_int)
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
            split_matrix_int = sps.csc_matrix(matrix_DIJ, shape=shape)
            split_matrix_avg = split_matrix_int.copy()

        elif self.dim == 1:
            # The case is conceptually similar to 0d, but quite a bit more technical,
            # (see implementation of the called function).
            # Separate mappings for averaged and integrated quantities.
            split_matrix_avg = pp.match_grids.match_grids_along_1d_mortar(
                self, g_new, g_old, tol, scaling="averaged"
            )
            split_matrix_int = pp.match_grids.match_grids_along_1d_mortar(
                self, g_new, g_old, tol, scaling="integrated"
            )

        else:  # should be mg.dim == 2
            # It should be possible to use essentially the same approach as in 1d,
            # but this is not yet covered.
            raise NotImplementedError("Have not yet implemented this.")

        # Update mappings to and from the primary grid
        self._primary_to_mortar_int = self._primary_to_mortar_int * split_matrix_int
        self._primary_to_mortar_avg = self._primary_to_mortar_avg * split_matrix_avg

        self._set_projections(secondary=False)
        self._check_mappings()

    def num_sides(self) -> int:
        """Shortcut to compute the number of sides. It has to be 2 or 1.

        Returns:
            Number of sides.

        """
        return len(self.side_grids)

    def project_to_side_grids(
        self,
    ) -> Generator[tuple[sps.spmatrix, pp.Grid], None, None]:
        """Generator for the side grids and projection operators from the mortar cells,
        combining cells on all the sides, to the specific side grids.

        Yields:
            A 2-tuple containing

            :obj:`~scipy.sparse.csc_matrix`:
                Projection from the mortar cells to this side grid.

            :class:`~porepy.grids.grid.Grid`:
                PorePy grid representing one of the sides of the mortar grid. Can
                be used for standard discretizations.

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

    def primary_to_mortar_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from faces of primary to the mortar, by summing quantities
        from the primary side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the primary side are distributed to the mortar according to the overlap
        between a primary face and (in general several) mortar cells.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Projection matrix with column sum unity and
            ``shape=(nd*g_primary.num_faces, nd*mortar_grid.num_cells)``.

        """
        return sparse_kronecker_product(self._primary_to_mortar_int, nd)

    def secondary_to_mortar_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from cells on the secondary side to the mortar, by summing
        quantities from the secondary side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the secondary side are distributed to the mortar according to the overlap
        between a secondary cell and (in general several) mortar cells.

        This mapping is intended for extensive properties, e.g. sources.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Projection matrix with column sum unity and
            ``shape=(nd*g_secondary.num_cells, nd*mortar_grid.num_cells)``.

        """
        return sparse_kronecker_product(self._secondary_to_mortar_int, nd)

    def primary_to_mortar_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from faces of primary to the mortar, by averaging quantities
        from the primary side.

        The projection matrix is scaled so that the row sum is unity, that is, values on
        the mortar side are computed as averages of values from the primary side,
        according to the overlap between (in general several) primary faces and a mortar
        cell.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Projection matrix with row sum unity and
            ``shape=(nd*g_primary.num_faces, nd*mortar_grid.num_cells)``.

        """
        return sparse_kronecker_product(self._primary_to_mortar_avg, nd)

    def secondary_to_mortar_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from cells at the secondary to the mortar, by averaging
        quantities from the secondary side.

        The projection matrix is scaled so that the row sum is unity, that is, values on
        the mortar side are computed as averages of values from the secondary side,
        according to the overlap between (in general several) primary cells and the
        mortar cells.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Projection matrix with row sum unity and
            ``shape=(nd*g_secondary.num_cells, nd*mortar_grid.num_cells)``.

        """
        return sparse_kronecker_product(self._secondary_to_mortar_avg, nd)

    def mortar_to_primary_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to faces of primary, by summing quantities
        from the mortar side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the mortar side are distributed to the primary according to the overlap
        between a mortar cell and (in general several) primary faces.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Projection matrix with column sum unity and
            ``shape=(nd*mortar_grid.num_cells, nd*g_primary.num_faces)``.

        """
        return sparse_kronecker_product(self._mortar_to_primary_int, nd)

    def mortar_to_secondary_int(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to cells at the secondary, by summing
        quantities from the mortar side.

        The projection matrix is scaled so that the column sum is unity, that is, values
        on the mortar side are distributed to the secondary according to the overlap
        between a mortar cell and (in general several) secondary cells.

        This mapping is intended for extensive properties, e.g. fluxes.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Projection matrix with column sum unity and
            ``shape=(nd*mortar_grid.num_cells, nd*g_secondary.num_faces)``.


        """
        return sparse_kronecker_product(self._mortar_to_secondary_int, nd)

    def mortar_to_primary_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to faces of primary, by averaging quantities
        from the mortar side.

        The projection matrix is scaled so that the row sum is unity, that is, values on
        the primary side are computed as averages of values from the mortar side,
        according to the overlap between (in general several) mortar cells and a primary
        face.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Projection matrix with row sum unity and
            ``shape=(nd*mortar_grid.num_cells, nd*g_primary.num_faces)``.

        """
        return sparse_kronecker_product(self._mortar_to_primary_avg, nd)

    def mortar_to_secondary_avg(self, nd: int = 1) -> sps.spmatrix:
        """Project values from the mortar to secondary, by averaging quantities from the
        mortar side.

        The projection matrix is scaled so that the row sum is unity, that is, values on
        the secondary side are computed as averages of values from the mortar side,
        according to the overlap between (in general several) mortar cells and a
        secondary cell.

        This mapping is intended for intensive properties, e.g. pressures.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Projection matrix with row sum unity and
            ``shape=(nd*mortar_grid.num_cells, nd*g_secondary.num_faces)``.

        """
        return sparse_kronecker_product(self._mortar_to_secondary_avg, nd)

    def sign_of_mortar_sides(self, nd: int = 1) -> sps.spmatrix:
        """Assign positive or negative weight to the two sides of a mortar grid.

        This is needed e.g. to make projection operators into signed projections, for
        variables that have no particular defined sign conventions.

        This function defines a convention for what is a positive jump between the
        mortar sides.

        Example:

            Take the difference between right and left variables, and project to the
            secondary grid by

            >>> mortar_to_secondary_avg() * sign_of_mortar_sides()

        Notes:
            The flux variables in flow and transport equations are defined as positive
            from primary to secondary. Hence, the two sides have different conventions,
            and there is no need to adjust the signs further.

            This method will probably not be meaningful if applied to mortar grids where
            the two side grids are non-matching.

        Parameters:
            nd: ``default=1``

                Spatial dimension of the projected quantity. Defaults to 1
                (mapping for scalar quantities).

        Returns:
            Diagonal matrix with positive signs on variables belonging to the first of
            the side_grids and
            ``shape=(nd*mortar_grid.num_cells, nd*mortar_grid.num_cells)``.

        """
        nc = self.num_cells
        if self.num_sides() == 1:
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

    def cell_diameters(self) -> np.ndarray:
        """
        Returns:
            An array containing the diameters of each cell in the mortar grid and
            ``shape=(mortar_grid.num_cells,)``.

        """
        diams = np.empty(self.num_sides(), dtype=object)
        for pos, (_, g) in enumerate(self.side_grids.items()):
            diams[pos] = g.cell_diameters()
        return np.concatenate(diams).ravel()

    def _check_mappings(self, tol: float = 1e-4) -> None:
        """Check whether the tolerance for matching new and old grids is reached.

        Raises:
            ValueError: If the check is not satisfied for the primary grid.
            ValueError: If the check is not satisfied for the secondary grid.

        """
        row_sum = self._primary_to_mortar_int.sum(axis=1)
        if not (row_sum.min() > tol):
            raise ValueError("Check not satisfied for the primary grid")

        row_sum = self._secondary_to_mortar_int.sum(axis=1)
        if not (row_sum.min() > tol):
            raise ValueError("Check not satisfied for the secondary grid")

    def _init_projections(
        self,
        primary_secondary: sps.spmatrix,
        face_duplicate_ind: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize projections from primary and secondary to mortar.

        Parameters:
            primary_secondary: Projection from the primary to the secondary. It is
                assumed that the primary, secondary and mortar grids are all matching.
            face_duplicate_ind: ``default=None``

                Which faces should be considered duplicates,
                and mapped to the second of the side_grids.

                If not provided, duplicate
                faces will be inferred from the indices of the faces. Will only be used
                if ``len(side_Grids)==2``.

        Raises:
            ValueError: If there are two sides and a face of the primary grid is not
                mapped to exactly two lower dimensional cells.
            ValueError: If the mapping between the mortar grid and the secondary mapping
                is not one-to-one.

        """
        # primary_secondary is a mapping from the cells (faces if secondary.dim ==
        # primary.dim) of the secondary grid to the faces of the primary grid.
        secondary_f, primary_f, data = sparse_array_to_row_col_data(primary_secondary)

        # If the face_duplicate_ind is given we have to reorder the primary face indices
        # such that the original faces comes first, then the duplicate faces. If the
        # face_duplicate_ind is not given, we then assume that the primary side faces
        # already have this ordering. If the grid is created using the pp.split_grid.py
        # module this should be the case.
        if (
            self.num_sides() == 2
            and (face_duplicate_ind is not None)
            and self.codim < 2
        ):
            is_second_side = np.isin(primary_f, face_duplicate_ind)
            secondary_f = np.r_[
                secondary_f[~is_second_side], secondary_f[is_second_side]
            ]
            primary_f = np.r_[primary_f[~is_second_side], primary_f[is_second_side]]
            data = np.r_[data[~is_second_side], data[is_second_side]]

        # Store index of the faces on the 'other' side of the mortar grid.
        if self.num_sides() == 2:
            # After the above sorting, we know that the faces on the other side is in
            # the second half of primary_f, also if face_duplicate_ind is given.
            # ASSUMPTION: The mortar grids on the two sides should match each other, or
            # else, the below indexing is wrong. This also means that the size of
            # primary_f is an even number.
            sz = int(primary_f.size / 2)
            self._ind_face_on_other_side = primary_f[sz:]

        # We assumed that the cells of the given side grid(s) is(are) ordered by the
        # secondary side index. In other words: cell "n" of the side grid(s) should
        # correspond to the element with the n'th lowest index in secondary_f. We
        # therefore sort secondary_f to obtaint the side grid ordering. The primary
        # faces should now be sorted such that the left side comes first, then the right
        # side. We use stable sort to not mix up the ordering if there is two sides.
        ix = np.argsort(secondary_f, kind="stable")
        if self.num_sides() == 2 and self.codim < 2:
            # If there are two sides we are in the case of a secondary grid of equal
            # dimension as the mortar grid. The mapping primary_secondary is then a
            # mapping from faces-cells, and we assume the higher dimensional grid is
            # split and there are exactly two primary faces mapping to each secondary
            # cell. Check this:
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
            to be a one to one mapping between the mortar grid and the
            secondary mapping"""
            )

        shape_primary = (self.num_cells, primary_secondary.shape[1])
        shape_secondary = (self.num_cells, primary_secondary.shape[0])

        # IMPLEMENTATION NOTE: Use optimized storage to minimize memory consumption.
        self._primary_to_mortar_int = pp.matrix_operations.optimized_compressed_storage(
            sps.csc_matrix(
                (data.astype(float), (cells, primary_f)), shape=shape_primary
            )
        )
        self._primary_to_mortar_avg = self._primary_to_mortar_int.copy()

        self._secondary_to_mortar_int = (
            pp.matrix_operations.optimized_compressed_storage(
                sps.csc_matrix(
                    (data.astype(float), (cells, secondary_f)), shape=shape_secondary
                )
            )
        )
        self._secondary_to_mortar_avg = self._secondary_to_mortar_int.copy()

    def _set_projections(self, primary: bool = True, secondary: bool = True) -> None:
        """Set projections to and from primary from the current state of
        ``_primary_to_mortar_int`` and ``_secondary_to_mortar_int`` .

        """

        # IMPLEMENTATION NOTE: Use optimized storage to minimize memory consumption.
        if primary:
            self._mortar_to_primary_int = (
                pp.matrix_operations.optimized_compressed_storage(
                    self._primary_to_mortar_avg.T
                )
            )
            self._mortar_to_primary_avg = (
                pp.matrix_operations.optimized_compressed_storage(
                    self._primary_to_mortar_int.T
                )
            )

        if secondary:
            self._mortar_to_secondary_int = (
                pp.matrix_operations.optimized_compressed_storage(
                    self._secondary_to_mortar_avg.T
                )
            )
            self._mortar_to_secondary_avg = (
                pp.matrix_operations.optimized_compressed_storage(
                    self._secondary_to_mortar_int.T
                )
            )
