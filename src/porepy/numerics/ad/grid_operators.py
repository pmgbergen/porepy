""" Ad representation of grid-related quantities needed to write equations. The classes
defined here are mainly wrappers that constructs Ad matrices based on grid information.

"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .operators import Operator, SparseArray

__all__ = [
    "BoundaryProjection",
    "MortarProjections",
    "Divergence",
    "Trace",
    "SubdomainProjections",
]


class SubdomainProjections:
    """Wrapper class for generating projection to and from subdomains.

    One use case in when variables are defined on only some subdomains.

    The class should be used through the methods {cell, face}_{projection, restriction}.

    See also MortarProjections for projections to and from mortar subdomains.

    """

    def __init__(
        self,
        subdomains: Sequence[pp.Grid],
        dim: int = 1,
    ) -> None:
        """Construct subdomain restrictions and prolongations for a set of subdomains.

        The projections will be ordered according to the ordering in subdomains. It is
        critical that the same ordering is used by other operators.

        Parameters:
            subdomains: List of subdomains. The order of the subdomains in
                the list will establish the ordering of the subdomain projections.
            dim: Dimension of the quantities to be projected.

        """

        self._name = "SubdomainProjection"
        self.dim = dim
        self._is_scalar: bool = dim == 1

        # Uniquify the list of subdomains. There is no need to have the same subdomain
        # represented several times.
        if len(set(subdomains)) < len(subdomains):
            # The problem here is that the subdomain projections are stored in a dict,
            # with the subdomanis as keys. If the same subdomain is represented twice,
            # the first projection will be overwritten by the second, thus the order
            # of the subdomains will be lost. There is no easy way to handle this, the
            # only option is to fix the error on the caller side. An internal fix would
            # entail changing the storage format for the projection, potentially needing
            # a lot of memory.
            raise ValueError("Subdomains must be unique")

        self._num_grids: int = len(subdomains)

        # Store total number of faces and cells in the list of subdomains. This will be
        # needed to handle projections to and from empty lists (see usage below).
        self._tot_num_cells: int = sum([g.num_cells for g in subdomains])
        self._tot_num_faces: int = sum([g.num_faces for g in subdomains])

        self._cell_projection, self._face_projection = _subgrid_projections(
            subdomains, self.dim
        )

    def cell_restriction(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct restrictions from global to subdomain cell quantities.

        Parameters:
            subdomains: One or several subdomains to which the projection should apply.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represents the
            projection.

        """
        if not isinstance(subdomains, list):
            raise ValueError(self._error_message())

        if len(subdomains) > 0:
            # A key error will be raised if a grid in g is not known to
            # self._cell_projection
            # IMPLEMENTATION NOTE: Use csr format, since the number of rows can
            # be much less than the number of columns.
            mat = sps.bmat([[self._cell_projection[g].T] for g in subdomains]).tocsr()
        else:
            # If the grid list is empty, we project from the full set of cells to
            # nothing.
            mat = sps.csr_matrix((0, self._tot_num_cells * self.dim))
        return pp.ad.SparseArray(mat, name="CellRestriction")

    def cell_prolongation(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct prolongation from subdomain to global cell quantities.

        Parameters:
            subdomains: One or several subdomains to which the prolongation should apply.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represent the
            prolongation.

        """
        if not isinstance(subdomains, list):
            raise ValueError(self._error_message())
        if len(subdomains) > 0:
            # A key error will be raised if a grid in g is not known to
            # self._cell_projection
            # IMPLEMENTATION NOTE: Use csc format, since the number of columns can
            # be much less than the number of rows.
            mat = sps.bmat([[self._cell_projection[g] for g in subdomains]]).tocsc()
        else:
            # If the grid list is empty, we project from nothing to the full set of
            # cells
            mat = sps.csc_matrix((self._tot_num_cells * self.dim, 0))
        return pp.ad.SparseArray(mat, name="CellProlongation")

    def face_restriction(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct restrictions from global to subdomain face quantities.

        Parameters:
            subdomains: One or several subdomains to which
                the projection should apply.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represent the
                projection.

        """
        if len(subdomains) > 0:
            # A key error will be raised if a grid in subdomains is not known to
            # self._face_projection
            # IMPLEMENTATION NOTE: Use csr format, since the number of rows can
            # be much less than the number of columns.
            mat = sps.bmat([[self._face_projection[g].T] for g in subdomains]).tocsr()
        else:
            # If the grid list is empty, we project from the full set of faces to
            # nothing.
            mat = sps.csr_matrix((0, self._tot_num_faces * self.dim))
        return pp.ad.SparseArray(mat, name="FaceRestriction")

    def face_prolongation(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct prolongation from subdomain to global face quantities.

        Parameters:
            subdomains: One or several subdomains to which the prolongation should apply.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represent the
            prolongation.

        """
        if len(subdomains) > 0:
            # A key error will be raised if a grid in subdomains is not known to
            # self._face_projection
            # IMPLEMENTATION NOTE: Use csc format, since the number of columns can
            # be far smaller than the number of rows.
            mat = sps.bmat([[self._face_projection[g] for g in subdomains]]).tocsc()
        else:
            # If the grid list is empty, we project from nothing to the full set of
            # faces
            mat = sps.csc_matrix((self._tot_num_faces * self.dim, 0))
        return pp.ad.SparseArray(mat, name="FaceProlongation")

    def __repr__(self) -> str:
        s = (
            f"Restriction and prolongation operators for {self._num_grids}"
            " unique subdomains\n"
            f"Aimed at variables with dimension {self.dim}\n"
        )
        return s

    def _error_message(self):
        return "Argument should be a subdomain grid or a list of subdomain grids"


class MortarProjections:
    """Wrapper class to generate projections to and from MortarGrids.

    Attributes:
        mortar_to_primary_int (pp.ad.SparseArray): Matrix of projections from the mortar
            grid to the primary grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        mortar_to_primary_avg (pp.ad.SparseArray): Matrix of projections from the mortar
            grid to the primary grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        primary_to_mortar_int (pp.ad.SparseArray): Matrix of projections from the primary
            grid to the mortar grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        primary_to_mortar_avg (pp.ad.SparseArray): Matrix of projections from the primary
            grid to the mortar grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        mortar_to_secondary_int (pp.ad.SparseArray): Matrix of projections from the mortar
            grid to the secondary grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        mortar_to_secondary_avg (pp.ad.SparseArray): Matrix of projections from the mortar
            grid to the secondary grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        secondary_to_mortar_int (pp.ad.SparseArray): Matrix of projections from the secondary
            grid to the mortar grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        secondary_to_mortar_avg (pp.ad.SparseArray): Matrix of projections from the secondary
            grid to the mortar grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        sign_of_mortar_sides (pp.ad.SparseArray): Matrix representation that assigns signs
            to two mortar sides. Needed to implement a jump operator in contact
            mechanics.

    """

    def __init__(
        self,
        mdg: pp.MixedDimensionalGrid,
        subdomains: Sequence[pp.Grid],
        interfaces: Sequence[pp.MortarGrid],
        dim: int = 1,
    ) -> None:
        """Construct mortar projection object.

        The projections will be ordered according to the ordering in grids, or the order
        of the MixedDimensionalGrid iteration over grids. It is critical that the same
        ordering is used by other operators.

        Parameters:
            mdg: Mixed-dimensional grid.
            subdomains: List of grids for which the projections should apply. The order
                of the grids in the list establishes the ordering of the subdomain
                projections.
            interfaces: List of edges for which the projections should apply. The order
                of the grids in the list establishes the ordering of the subdomain
                projections.
            dim: Dimension of the quantities to be projected.

        """
        self._name = "MortarProjection"
        self._num_edges: int = len(interfaces)
        self.dim: int = dim

        # Initialize projections
        cell_projection, face_projection = _subgrid_projections(subdomains, self.dim)

        # IMPLEMENTATION NOTE:
        # sparse blocks are slow; it should be possible to do a right multiplication
        # of local-to-global mortar indices instead of the block.

        # Data structures for constructing the projection operators
        mortar_to_primary_int, mortar_to_primary_avg = [], []
        primary_to_mortar_int, primary_to_mortar_avg = [], []

        mortar_to_secondary_int, mortar_to_secondary_avg = [], []
        secondary_to_mortar_int, secondary_to_mortar_avg = [], []

        # The goal is to construct global projections between subdomains and mortar
        # subdomains. The construction takes two stages, and is different for
        # projections to and from the mortar grid: For projections from the mortar grid,
        # a mapping is first made from local mortar numbering global grid ordering. In
        # the second stage, the mappings from mortar are stacked to make a global
        # mapping. Projections to the mortar grid are made by first defining projections
        # from global grid numbering to local mortar subdomains, and then stack the
        # latter.

        # Special treatment is needed for the case of empty lists - see below
        # Helper function for that case:
        def zero_matrices(sz_mortar, sz_tot):
            m2g = pp.matrix_operations.optimized_compressed_storage(
                sps.csr_matrix((sz_tot, sz_mortar))
            )
            g2m = pp.matrix_operations.optimized_compressed_storage(
                sps.csr_matrix((sz_mortar, sz_tot))
            )
            return m2g, g2m

        if len(interfaces) > 0:
            for intf in interfaces:
                g_primary, g_secondary = mdg.interface_to_subdomain_pair(intf)
                assert isinstance(intf, pp.MortarGrid)  # Appease mypy
                if (
                    g_primary.dim != intf.dim + intf.codim
                ) or g_secondary.dim != intf.dim:
                    # This will correspond to DD of sorts; we could handle this
                    # by using cell_projections for g_primary and/or
                    # face_projection for g_secondary, depending on the exact
                    # configuration
                    raise NotImplementedError("Non-standard interface.")
                if g_primary in subdomains:
                    # Choose projection matrices for primary grid based on codimension
                    # of the interface.
                    primary_projection = (
                        face_projection[g_primary]
                        if intf.codim < 2
                        else cell_projection[g_primary]
                    )
                    # Create all projection matrices for this MortarGrid and append them
                    # to the list. The use of optimized storage is of importance here,
                    # since for small subdomain subdomains in problems with many cells
                    # in total, the projection matrices may have many more rows than
                    # columns, or opposite.

                    # Projections to primary
                    mortar_to_primary_int.append(
                        pp.matrix_operations.optimized_compressed_storage(
                            primary_projection * intf.mortar_to_primary_int(dim)
                        )
                    )
                    mortar_to_primary_avg.append(
                        pp.matrix_operations.optimized_compressed_storage(
                            primary_projection * intf.mortar_to_primary_avg(dim)
                        )
                    )

                    # Projections from primary
                    primary_to_mortar_int.append(
                        pp.matrix_operations.optimized_compressed_storage(
                            intf.primary_to_mortar_int(dim) * primary_projection.T
                        )
                    )
                    primary_to_mortar_avg.append(
                        pp.matrix_operations.optimized_compressed_storage(
                            intf.primary_to_mortar_avg(dim) * primary_projection.T
                        )
                    )
                else:
                    # The primary grid is not in the list of subdomains. All projections
                    # to the primary grid are zero. They should all have the correct
                    # dimensions, though. The size corresponding to the (missing)
                    # primary grid is always zero, and that of the mortar grid is always
                    # as above.
                    sz_mortar = intf.num_cells * dim
                    if len(subdomains) == 0:
                        # No subdomains provided
                        sz_tot = 0
                    else:
                        # We need the number of rows for the primary projection matrix.
                        # This equals total number of faces (or cells, if codim is > 1)
                        # in the subdomains times the dimension of the problem.
                        grid_entity = "num_faces" if intf.codim < 2 else "num_cells"
                        sz_tot = (
                            sum([getattr(sd, grid_entity) for sd in subdomains]) * dim
                        )
                    m2p, p2m = zero_matrices(sz_mortar, sz_tot)
                    mortar_to_primary_int.append(m2p)
                    mortar_to_primary_avg.append(m2p)
                    primary_to_mortar_int.append(p2m)
                    primary_to_mortar_avg.append(p2m)

                if g_secondary in subdomains:
                    # Projections to secondary
                    mortar_to_secondary_int.append(
                        pp.matrix_operations.optimized_compressed_storage(
                            cell_projection[g_secondary]
                            * intf.mortar_to_secondary_int(dim)
                        )
                    )
                    mortar_to_secondary_avg.append(
                        pp.matrix_operations.optimized_compressed_storage(
                            cell_projection[g_secondary]
                            * intf.mortar_to_secondary_avg(dim)
                        )
                    )

                    # Projections from secondary.

                    # IMPLEMENTATION NOTE: For some reason, forcing csr format here
                    # decreased the runtime with a factor of 5, while this was not
                    # important while creating the other projection matrices.
                    # Experimentation showed no similar pattern when flipping between
                    # csc and csr formats, so it probably just has to be in this way, or
                    # this was case-dependent behavior (the relevant test case was the
                    # field case in the 3d flow benchmark).
                    secondary_to_mortar_int.append(
                        pp.matrix_operations.optimized_compressed_storage(
                            intf.secondary_to_mortar_int(dim).tocsr()
                            * cell_projection[g_secondary].T
                        )
                    )
                    secondary_to_mortar_avg.append(
                        pp.matrix_operations.optimized_compressed_storage(
                            intf.secondary_to_mortar_avg(dim).tocsr()
                            * cell_projection[g_secondary].T
                        )
                    )
                else:
                    # The primary grid is not in the list of subdomains. All projections
                    # to the primary grid are zero. They should all have the correct
                    # dimensions, though. The size corresponding to the (missing)
                    # secondary grid is always zero, and that of the mortar grid is
                    # always as above.
                    sz_mortar = intf.num_cells * dim
                    if len(subdomains) == 0:
                        # No subdomains provided. The total size is zero.
                        sz_tot = 0
                    else:
                        # We need the number of rows for the primary projection matrix.
                        # This equals total number of faces in the subdomains
                        # times the dimension of the problem.
                        sz_tot = sum([sd.num_cells for sd in subdomains]) * dim
                    m2s, s2m = zero_matrices(sz_mortar, sz_tot)
                    mortar_to_secondary_int.append(m2s)
                    mortar_to_secondary_avg.append(m2s)
                    secondary_to_mortar_int.append(s2m)
                    secondary_to_mortar_avg.append(s2m)

        else:
            num_cells_lower_dimension = sum([g.num_cells for g in subdomains]) * dim
            num_faces_higher_dimension = sum([g.num_faces for g in subdomains]) * dim

            # Projections to and from the grid
            to_face = sps.csc_matrix((num_faces_higher_dimension, 0))
            from_face = sps.csr_matrix((0, num_faces_higher_dimension))
            to_cells = sps.csc_matrix((num_cells_lower_dimension, 0))
            from_cells = sps.csr_matrix((0, num_cells_lower_dimension))

            # Projections to primary
            mortar_to_primary_int.append(to_face)
            mortar_to_primary_avg.append(to_face)

            # Projections from primary
            primary_to_mortar_int.append(from_face)
            primary_to_mortar_avg.append(from_face)

            mortar_to_secondary_int.append(to_cells)
            mortar_to_secondary_avg.append(to_cells)

            secondary_to_mortar_int.append(from_cells)
            secondary_to_mortar_avg.append(from_cells)

        # Stack mappings from the mortar horizontally.
        # The projections are wrapped by a pp.ad.SparseArray to be compatible with the
        # requirements for processing of Ad operators.
        def bmat(matrices, name):
            # Create block matrix, convert it to optimized storage format
            block_matrix = pp.matrix_operations.optimized_compressed_storage(
                sps.bmat(matrices)
            )
            return SparseArray(block_matrix, name=name)

        self.mortar_to_primary_int = bmat(
            [mortar_to_primary_int], name="MortarToPrimaryInt"
        )
        self.mortar_to_primary_avg = bmat(
            [mortar_to_primary_avg], name="MortarToPrimaryAvg"
        )
        self.mortar_to_secondary_int = bmat(
            [mortar_to_secondary_int], name="MortarToSecondaryInt"
        )
        self.mortar_to_secondary_avg = bmat(
            [mortar_to_secondary_avg], name="MortarToSecondaryAvg"
        )

        # Vertical stacking of the projections
        self.primary_to_mortar_int = bmat(
            [[m] for m in primary_to_mortar_int], name="PrimaryToMortarInt"
        )
        self.primary_to_mortar_avg = bmat(
            [[m] for m in primary_to_mortar_avg], name="PrimaryToMortarAvg"
        )
        self.secondary_to_mortar_int = bmat(
            [[m] for m in secondary_to_mortar_int], name="SecondaryToMortarInt"
        )
        self.secondary_to_mortar_avg = bmat(
            [[m] for m in secondary_to_mortar_avg], name="SecondaryToMortarAvg"
        )

        # Also generate a merged version of MortarGrid.sign_of_mortar_sides:
        mats = []
        for intf in interfaces:
            assert isinstance(intf, pp.MortarGrid)  # Appease mypy
            mats.append(intf.sign_of_mortar_sides(dim))
        if len(interfaces) == 0:
            self.sign_of_mortar_sides = SparseArray(
                sps.bmat([[None]]), name="SignOfMortarSides"
            )
        else:
            self.sign_of_mortar_sides = SparseArray(
                sps.block_diag(mats), name="SignOfMortarSides"
            )

    def __repr__(self) -> str:
        s = (
            f"Mortar projection for {self._num_edges} interfaces\n"
            f"Aimed at variables with dimension {self.dim}\n"
            f"Projections to primary have dimensions {self.mortar_to_primary_avg.shape}\n"
            f"Projections to secondary have dimensions {self.mortar_to_secondary_avg.shape}\n"
        )
        return s


class BoundaryProjection:
    """A projection operator between boundary grids and subdomains."""

    def __init__(
        self, mdg: pp.MixedDimensionalGrid, subdomains: Sequence[pp.Grid], dim: int = 1
    ) -> None:
        _, face_projections = _subgrid_projections(subdomains, dim)

        # Size for the matrix, used for 0d subdomains.
        tot_num_faces = np.sum([sd.num_faces for sd in subdomains]) * dim

        mat = []
        for sd in subdomains:
            if sd.dim > 0:
                bg = mdg.subdomain_to_boundary_grid(sd)
                if bg is not None:
                    mat_loc = bg.projection(dim)
                    mat_loc = mat_loc * face_projections[sd].T
            else:
                # The subdomain has no faces, so the projection does not exist.
                mat_loc = sps.csr_matrix((0, tot_num_faces))
            mat.append(mat_loc)

        self._projection: sps.spmatrix
        """Projection from subdomain faces to boundary grid cells."""
        if len(mat) > 0:
            self._projection = sps.bmat([[m] for m in mat], format="csr")
        else:
            self._projection = sps.csr_matrix((0, 0))

    @property
    def subdomain_to_boundary(self) -> Operator:
        return SparseArray(self._projection, name="subdomains to boundaries projection")

    @property
    def boundary_to_subdomain(self) -> Operator:
        return SparseArray(
            self._projection.transpose().tocsc(),
            name="boundaries to subdomains projection",
        )


class Trace:
    """Wrapper class for Ad representations of trace operators and their inverse,
    that is, mappings between grid cells and faces.

    NOTE: The mapping will hit both boundary and interior faces, so the values
    to be mapped should be carefully filtered (e.g. by combining it with a
    mortar mapping).

    The mapping does not alter signs of variables, that is, the direction
    of face normal vectors is not accounted for.

    """

    def __init__(
        self,
        subdomains: list[pp.Grid],
        dim: int = 1,
        name: Optional[str] = None,
    ):
        """Construct trace operators and their inverse for a given set of subdomains.

        The operators will be ordered according to the ordering in subdomains. It is
        critical that the same ordering is used by other operators.

        Parameters:
            subdomains: List of grids. The order of the grids in the list sets the
                ordering of the trace operators.
            dim: Dimension of the quantities to be projected. Defaults to 1.
            name: Name of the operator. Default is None.

        """
        self.subdomains: list[pp.Grid] = subdomains
        self.dim: int = dim
        self._name: Optional[str] = name
        self._is_scalar: bool = dim == 1
        self._num_grids: int = len(subdomains)

        cell_projections, face_projections = _subgrid_projections(subdomains, self.dim)

        trace: list[sps.spmatrix] = []
        inv_trace: list[sps.spmatrix] = []

        if len(subdomains) > 0:
            for sd in subdomains:
                if self._is_scalar:
                    # TEMPORARY CONSTRUCT: Use the divergence operator as a trace.
                    # It would be better to define a dedicated function for this,
                    # perhaps in the grid itself.
                    div = np.abs(pp.fvutils.scalar_divergence(sd))

                    # Restrict global cell values to the local grid, use transpose of div
                    # to map cell values to faces.
                    trace.append(div.T * cell_projections[sd].T)
                    # Similarly restrict a global face quantity to the local grid, then
                    # map back to cells.
                    inv_trace.append(div * face_projections[sd].T)
                else:
                    raise NotImplementedError("kronecker")
        else:
            trace = [sps.csr_matrix((0, 0))]
            inv_trace = [sps.csr_matrix((0, 0))]
        # Stack both trace and inv_trace vertically to make them into mappings to
        # global quantities.
        # Wrap the stacked matrices into an Ad object
        self.trace = SparseArray(sps.bmat([[m] for m in trace]).tocsr())
        """ Matrix of trace projections from cells to faces."""
        self.inv_trace = SparseArray(sps.bmat([[m] for m in inv_trace]).tocsr())
        """ Matrix of inverse trace projections from faces to cells."""

    def __repr__(self) -> str:
        s = (
            f"Trace operator for {self._num_grids} subdomains\n"
            f"Aimed at variables with dimension {self.dim}\n"
            f"Projection from grid to mortar has dimensions {self.trace}\n"
        )
        return s

    def __str__(self) -> str:
        s = "Trace"
        if self._name is not None:
            s += f" named {self._name}"
        return s


class Divergence(Operator):
    """Wrapper class for Ad representations of divergence operators."""

    def _key(self) -> str:
        subdomain_ids = [sd.id for sd in self.subdomains]
        return f"(divergence, dim={self.dim}, subdomains={subdomain_ids})"

    def __init__(
        self,
        subdomains: list[pp.Grid],
        dim: int = 1,
        name: Optional[str] = None,
    ):
        """Construct divergence operators for a set of subdomains.

        The operators will be ordered according to the ordering in subdomains, or the
        order of the MixedDimensionalGrid iteration over subdomains. It is critical that
        the same ordering is used by other operators.

        IMPLEMENTATION NOTE: Only scalar quantities so far; vector operators will be
        added in due course.

        Parameters:
            subdomains: List of grids. The order of the subdomains in
                the list sets the ordering of the divergence operators.
            dim: Dimension of vector field. Defaults to 1.
            name: Name to be assigned to the operator. Default is None.

        """
        super().__init__(domains=subdomains, name=name)

        self.dim: int = dim

    def __repr__(self) -> str:
        s = (
            f"Divergence for vector field of size {self.dim}"
            f" defined on {len(self.subdomains)} subdomains\n"
        )

        num_faces = 0
        num_cells = 0
        for g in self.subdomains:
            num_faces += g.num_faces * self.dim
            num_cells += g.num_cells * self.dim
        s += f"The total size of the matrix is ({num_cells}, {num_faces}).\n"

        return s

    def __str__(self) -> str:
        s = "Divergence "
        if self._name is not None:
            s += f"named {self._name}"
        return s

    def parse(self, mdg: pp.MixedDimensionalGrid) -> sps.spmatrix:
        """Convert the Ad expression into a divergence operators on all relevant
        subdomains, represented as a sparse block matrix.

        Parameters:
            mdg: Not used, but needed for compatibility with the general parsing method
                for Operators.

        Returns:
            sps.spmatrix: Block matrix representation of a divergence operator on
            multiple subdomains.

        """
        if self.dim == 1:
            mat = [pp.fvutils.scalar_divergence(sd) for sd in self.subdomains]
        else:
            mat = [
                sps.kron(pp.fvutils.scalar_divergence(sd), sps.eye(self.dim))
                for sd in self.subdomains
            ]
        matrix = sps.block_diag(mat)
        return matrix


def _subgrid_projections(
    subdomains: Sequence[pp.Grid], dim: int
) -> tuple[dict[pp.Grid, sps.spmatrix], dict[pp.Grid, sps.spmatrix]]:
    """Construct prolongation matrices from individual subdomains to a set of subdomains.

    Parameters:
        subdomains: List of grids representing subdomains.
        dim: Dimension of the quantities to be projected. 1 corresponds to scalars, 2 to
            a vector of two components etc.

    Returns:
        cell_projection: Dictionary with the individual subdomains as keys and
            projection matrices for cell-based quantities as items.
        face_projection: Dictionary with the individual subdomains as keys and
        projection matrices for face-based quantities as items.

    The global cell and face numbering is set according to the order of the input
    subdomains.

    If the function is to be called with mortar or boundary grids, assign
    num_faces attributes (value 0).

    """
    face_projection: dict[pp.Grid, np.ndarray] = {}
    cell_projection: dict[pp.Grid, np.ndarray] = {}
    if len(subdomains) == 0:
        return cell_projection, face_projection

    tot_num_faces = np.sum([g.num_faces for g in subdomains]) * dim
    tot_num_cells = np.sum([g.num_cells for g in subdomains]) * dim

    face_offset = 0
    cell_offset = 0

    for sd in subdomains:
        cell_ind = cell_offset + pp.fvutils.expand_indices_nd(
            np.arange(sd.num_cells), dim
        )
        cell_sz = sd.num_cells * dim

        # Create matrix and convert to csc format, since the number of rows is (much)
        # higher than the number of columns.
        cell_projection[sd] = sps.coo_matrix(
            (np.ones(cell_sz), (cell_ind, np.arange(cell_sz))),
            shape=(tot_num_cells, cell_sz),
        ).tocsc()
        cell_offset = cell_ind[-1] + 1

        face_ind = face_offset + pp.fvutils.expand_indices_nd(
            np.arange(sd.num_faces), dim
        )
        face_sz, cell_sz = sd.num_faces * dim, sd.num_cells * dim
        face_projection[sd] = sps.coo_matrix(
            (np.ones(face_sz), (face_ind, np.arange(face_sz))),
            shape=(tot_num_faces, face_sz),
        ).tocsc()  # Again use csc storage, since num_cols < num_rows

        # Correct start of the numbering for the next grid
        if sd.dim > 0:
            # Point subdomains have no faces
            face_offset = face_ind[-1] + 1

    return cell_projection, face_projection
