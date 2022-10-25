""" Ad representation of grid-related quantities needed to write equations. The classes
defined here are mainly wrappers that constructs Ad matrices based on grid information.

"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .operators import Matrix, Operator

__all__ = [
    "MortarProjections",
    "Divergence",
    "BoundaryCondition",
    "Geometry",
    "Trace",
    "SubdomainProjections",
    "ParameterArray",
    "ParameterMatrix",
]


class SubdomainProjections(Operator):
    """Wrapper class for generating projection to and from subdomains.

    One use case in when variables are defined on only some subdomains.

    The class should be used through the methods {cell, face}_{projection, restriction}.

    See also MortarProjections for projections to and from mortar subdomains.

    """

    def __init__(
        self,
        subdomains: List[pp.Grid],
        dim: int = 1,
    ) -> None:
        """Construct subdomain restrictions and prolongations for a set of subdomains.

        The projections will be ordered according to the ordering in subdomains. It is critical
        that the same ordering is used by other operators.

        Parameters:
            subdomains (List of pp.Grid): List of subdomains. The order of the subdomains in
                the list will establish the ordering of the subdomain projections.
            dim (int, optional): Dimension of the quantities to be projected.

        """
        self._name = "SubdomainProjection"
        self.dim = dim
        self._is_scalar: bool = dim == 1

        self._num_grids: int = len(subdomains)

        # Store total number of faces and cells in the list of subdomains. This will be
        # needed to handle projections to and from empty lists (see usage below).
        self._tot_num_cells: int = sum([g.num_cells for g in subdomains])
        self._tot_num_faces: int = sum([g.num_faces for g in subdomains])

        self._cell_projection, self._face_projection = _subgrid_projections(
            subdomains, self.dim
        )

    def cell_restriction(self, subdomains: List[pp.Grid]) -> Matrix:
        """Construct restrictions from global to subdomain cell quantities.

        Parameters:
            subdomains (List of pp.Grid): One or several subdomains to which
                the projection should apply.

        Returns:
            pp.ad.Matrix: Matrix operator (in the Ad sense) that represents the
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
        return pp.ad.Matrix(
            mat,
            name="CellRestriction",
        )

    def cell_prolongation(self, subdomains: List[pp.Grid]) -> Matrix:
        """Construct prolongation from subdomain to global cell quantities.

        Parameters:
            subdomains (List of pp.Grid): One or several subdomains to which
                the prolongation should apply.

        Returns:
            pp.ad.Matrix: Matrix operator (in the Ad sense) that represent the
                prolongation.

        """
        if isinstance(subdomains, pp.Grid):
            return pp.ad.Matrix(
                self._cell_projection[subdomains], name="CellProlongation"
            )
        elif isinstance(subdomains, list):
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
            return pp.ad.Matrix(
                mat,
                name="CellProlongation",
            )
        else:
            raise ValueError(self._error_message())

    def face_restriction(self, subdomains: List[pp.Grid]) -> Matrix:
        """Construct restrictions from global to subdomain face quantities.

        Parameters:
            subdomains (List of pp.Grid): One or several subdomains to which
                the projection should apply.

        Returns:
            pp.ad.Matrix: Matrix operator (in the Ad sense) that represent the
                projection.

        """
        if isinstance(subdomains, pp.Grid):
            if "mortar_grid" in subdomains.name:
                raise ValueError("Argument should be a regular grid, not mortar grid")
            return pp.ad.Matrix(
                self._face_projection[subdomains].T, name="FaceRestriction"
            )
        elif isinstance(subdomains, list):
            if len(subdomains) > 0:
                # A key error will be raised if a grid in subdomains is not known to
                # self._face_projection
                # IMPLEMENTATION NOTE: Use csr format, since the number of rows can
                # be much less than the number of columns.
                mat = sps.bmat(
                    [[self._face_projection[g].T] for g in subdomains]
                ).tocsr()
            else:
                # If the grid list is empty, we project from the full set of faces to
                # nothing.
                mat = sps.csr_matrix((0, self._tot_num_faces * self.dim))
            return pp.ad.Matrix(
                mat,
                name="FaceRestriction",
            )
        else:
            raise ValueError(self._error_message())

    def face_prolongation(self, subdomains: List[pp.Grid]) -> Matrix:
        """Construct prolongation from subdomain to global face quantities.

        Parameters:
            subdomains (List of pp.Grid): One or several subdomains to which
                the prolongation should apply.

        Returns:
            pp.ad.Matrix: Matrix operator (in the Ad sense) that represent the
                prolongation.

        """
        if isinstance(subdomains, pp.Grid):
            return pp.ad.Matrix(
                self._face_projection[subdomains], name="FaceProlongation"
            )
        elif isinstance(subdomains, list):
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
            return pp.ad.Matrix(
                mat,
                name="FaceProlongation",
            )
        else:
            raise ValueError(self._error_message())

    def __repr__(self) -> str:
        s = (
            f"Restriction and prolongation operators for {self._num_grids} subdomains\n"
            f"Aimed at variables with dimension {self.dim}\n"
        )
        return s

    def _error_message(self):
        return "Argument should be a subdomain grid or a list of subdomain grids"


class MortarProjections(Operator):
    """Wrapper class to generate projections to and from MortarGrids.

    Attributes:
        mortar_to_primary_int (pp.ad.Matrix): Matrix of projections from the mortar
            grid to the primary grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        mortar_to_primary_avg (pp.ad.Matrix): Matrix of projections from the mortar
            grid to the primary grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        primary_to_mortar_int (pp.ad.Matrix): Matrix of projections from the primary
            grid to the mortar grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        primary_to_mortar_avg (pp.ad.Matrix): Matrix of projections from the primary
            grid to the mortar grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        mortar_to_secondary_int (pp.ad.Matrix): Matrix of projections from the mortar
            grid to the secondary grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        mortar_to_secondary_avg (pp.ad.Matrix): Matrix of projections from the mortar
            grid to the secondary grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        secondary_to_mortar_int (pp.ad.Matrix): Matrix of projections from the secondary
            grid to the mortar grid. Intended for extensive quantities (so fluxes).
            Represented as an Ad Matrix operator.
        secondary_to_mortar_avg (pp.ad.Matrix): Matrix of projections from the secondary
            grid to the mortar grid. Intended for intensive quantities (so pressures).
            Represented as an Ad Matrix operator.
        sign_of_mortar_sides (pp.Ad.Matrix): Matrix representation that assigns signs
            to two mortar sides. Needed to implement a jump operator in contact
            mechanics.

    """

    def __init__(
        self,
        mdg: pp.MixedDimensionalGrid,
        subdomains: List[pp.Grid],
        interfaces: List[pp.MortarGrid],
        dim: int = 1,
    ) -> None:
        """Construct mortar projection object.

        The projections will be ordered according to the ordering in grids, or the order
        of the MixedDimensionalGrid iteration over grids. It is critical that the same ordering
        is used by other operators.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid.
            subdomains (List of pp.Grid): List of grids for which the projections
                should apply. The order of the grids in the list establishes the ordering of
                the subdomain projections.
            interfaces (List of edges): List of edges for which the projections
                should apply. The order of the grids in the list establishes the ordering of
                the subdomain projections.
            dim (int, optional): Dimension of the quantities to be projected.

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

        # The goal is to construct global projections between subdomains and mortar subdomains.
        # The construction takes two stages, and is different for projections to and
        # from the mortar grid:
        # For projections from the mortar grid, a mapping is first made from local
        # mortar numbering global grid ordering. In the second stage, the mappings from
        # mortar are stacked to make a global mapping.
        # Projections to the mortar grid are made by first defining projections from
        # global grid numbering to local mortar subdomains, and then stack the latter.

        # Special treatment is needed for the case of empty lists - see below
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
                primary_projection = (
                    face_projection[g_primary]
                    if intf.codim < 2
                    else cell_projection[g_primary]
                )
                # Create all projection matrices for this MortarGrid and append them to
                # the list. The use of optimized storage is of importance here, since
                # for small subdomain subdomains in problems with many cells in total, the
                # projection matrices may have many more rows than columns, or opposite.

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

                # Projections to secondary
                mortar_to_secondary_int.append(
                    pp.matrix_operations.optimized_compressed_storage(
                        cell_projection[g_secondary] * intf.mortar_to_secondary_int(dim)
                    )
                )
                mortar_to_secondary_avg.append(
                    pp.matrix_operations.optimized_compressed_storage(
                        cell_projection[g_secondary] * intf.mortar_to_secondary_avg(dim)
                    )
                )

                # Projections from secondary.
                # IMPLEMENTATION NOTE: For some reason, forcing csr format here
                # decreased the runtime with a factor of 5, while this was not important
                # while creating the other projection matrices. Experimentation showed no
                # similar pattern when flipping between csc and csr formats, so it probably
                # just has to be in this way, or this was case-dependent behavior (the
                # relevant test case was the field case in the 3d flow benchmark).
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
            # FIXME: The assumption here is that a MixedDimensionalGrid with a single grid
            # (no fractures) has been constructed. In this case, the projection
            # to primary should have g.num_faces rows, while there are no
            # secondary subdomains to project to. If the mortar projection is constructed
            # for a different case (hard to imagine what, but who knows), it is not
            # clear what to do, so we'll raise an error.
            assert len(subdomains) == 1

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
        # The projections are wrapped by a pp.ad.Matrix to be compatible with the
        # requirements for processing of Ad operators.
        def bmat(matrices, name):
            # Create block matrix, convert it to optimized storage format
            block_matrix = pp.matrix_operations.optimized_compressed_storage(
                sps.bmat(matrices)
            )
            return Matrix(block_matrix, name=name)

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
            self.sign_of_mortar_sides = Matrix(
                sps.bmat([[None]]), name="SignOfMortarSides"
            )
        else:
            self.sign_of_mortar_sides = Matrix(
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


class Trace(Operator):
    """Wrapper class for Ad representations of trace operators and their inverse,
    that is, mappings between grid cells and faces.

    NOTE: The mapping will hit both boundary and interior faces, so the values
    to be mapped should be carefully filtered (e.g. by combining it with a
    mortar mapping).

    The mapping does not alter signs of variables, that is, the direction
    of face normal vectors is not accounted for.

    Attributes:
        trace (pp.ad.Matrix): Matrix of trace projections from cells to faces.
        inv_trace (pp.ad.Matrix): Matrix of trace projections from faces to cells.

    """

    def __init__(
        self,
        subdomains: List[pp.Grid],
        dim: int = 1,
        name: Optional[str] = None,
    ):
        """Construct trace operators and their inverse for a given set of subdomains.

        The operators will be ordered according to the ordering in subdomains. It is critical
        that the same ordering is used by other operators.

        Parameters:
            subdomains (List of pp.Grid): List of grids. The order of the grids in the list
                sets the ordering of the trace operators.
            dim (int, optional): Dimension of the quantities to be projected. Defaults to 1.
            name (str, optional): Name of the operator. Default is None.

        """
        super().__init__(name=name)

        self.grids = subdomains
        self.dim: int = dim
        self._is_scalar: bool = dim == 1
        self._num_grids: int = len(subdomains)

        cell_projections, face_projections = _subgrid_projections(subdomains, self.dim)

        trace: list[sps.spmatrix] = []
        inv_trace: list[sps.spmatrix] = []

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
        # Stack both trace and inv_trace vertically to make them into mappings to
        # global quantities.
        # Wrap the stacked matrices into an Ad object
        self.trace = Matrix(sps.bmat([[m] for m in trace]).tocsr())
        self.inv_trace = Matrix(sps.bmat([[m] for m in inv_trace]).tocsr())

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


class Geometry(Operator):
    """Wrapper class for Ad representations of grids.

    Attributes:
        cell_volumes (pp.ad.Matrix): Diagonal ad matrix of cell volumes.
        face_areas (pp.ad.Matrix):  Diagonal ad matrix of face areas.
        nd (int): Ambient/highest dimension of the mixed-dimensional grid.

    FIXME: Implement parse??
    """

    def __init__(self, subdomains: list[pp.Grid], nd: int, name: Optional[str] = None):
        """Construct concatenated grid operators for a given set of subdomains.

        The operators will be ordered according to the ordering in grids. It is critical
        that the same ordering is used by other operators.

        Parameters:
            subdomains (List of pp.Grid): List of grids. The order of the grids in the list
                sets the ordering of the geometry operators. Can be either subdomain (pp.Grid)
                or interface (pp.MortarGrid) grids.
            nd: ambient dimension.
            name (str, optional): Name of the operator. Default is None.

        Raises:
            AssertionError if nd is smaller than the dimension of any subdomain.
        """
        super().__init__(name=name)

        self.subdomains = subdomains
        for sd in subdomains:
            assert sd.dim <= nd

        self._num_grids: int = len(subdomains)
        self.nd = nd

        self.num_cells: int = sum([g.num_cells for g in subdomains])
        self.num_faces: int = sum([g.num_faces for g in subdomains])

        # Wrap the stacked matrices into Ad objects (could be extended to e.g. face normals)
        for field in ["cell_volumes", "face_areas"]:
            ad_matrix = Matrix(
                sps.diags(np.hstack([getattr(g, field) for g in subdomains]))
            )
            setattr(self, field, ad_matrix)

        def scalar_to_nd(size):
            """Expand matrix of size [N x M] to [nd*N x M].

            When left multiplied to a matrix A, each row of A is
            repeated nd times.

            Usage example: Scaling from traction to force is

                    force = (scalar_to_nd_face * face_areas) * traction

            FIXME: Refactor? We might also want a general kron wrapper.
            """
            rows: np.ndarray = np.arange(size * self.nd)
            cols: np.ndarray = np.kron(np.arange(size), np.ones(self.nd))
            data: np.ndarray = np.ones(size * self.nd)
            mat = sps.csc_matrix(
                (data, (rows, cols)),
                shape=(size * self.nd, size),
            )
            return pp.ad.Matrix(mat)

        self.scalar_to_nd_cell = scalar_to_nd(self.num_cells)
        self.scalar_to_nd_face = scalar_to_nd(self.num_faces)

    def __repr__(self) -> str:
        s = (
            f"Geometry operator for {self._num_grids} grids.\n"
            f"Ambient dimension is {self.nd}.\n"
        )
        return s

    def __str__(self) -> str:
        s = "Compound geometry"
        if self._name is not None:
            s += f" named {self._name}"
        return s


class Divergence(Operator):
    """Wrapper class for Ad representations of divergence operators."""

    def __init__(
        self,
        subdomains: List[pp.Grid],
        dim: int = 1,
        name: Optional[str] = None,
    ):
        """Construct divergence operators for a set of subdomains.

        The operators will be ordered according to the ordering in subdomains, or the order
        of the MixedDimensionalGrid iteration over subdomains. It is critical that the same
        ordering is used by other operators.

        IMPLEMENTATION NOTE: Only scalar quantities so far; vector operators will be
        added in due course.

        Parameters:
            subdomains (List of pp.Grid): List of grids. The order of the subdomains in
                the list sets the ordering of the divergence operators.
            dim (int, optional): Dimension of vector field. Defaults to 1.
            name (str, optional): Name to be assigned to the operator. Default is None.

        """
        super().__init__(name=name)
        self.subdomains = subdomains

        self.dim: int = dim
        self._set_tree(None)

    def __repr__(self) -> str:
        s = (
            f"divergence for vector field of size {self.dim}"
            f" defined on {len(self.subdomains)} subdomains\n"
        )

        nf = 0
        nc = 0
        for g in self.subdomains:
            nf += g.num_faces * self.dim
            nc += g.num_cells * self.dim
        s += f"The total size of the matrix is ({nc}, {nf})\n"

        return s

    def __str__(self) -> str:
        s = "Divergence "
        if self._name is not None:
            s += f"named {self._name}"
        return s

    def parse(self, mdg: pp.MixedDimensionalGrid) -> sps.spmatrix:
        """Convert the Ad expression into a divergence operators on all relevant subdomains,
        represented as a sparse block matrix.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Not used, but needed for compatibility with
                the general parsing method for Operators.

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


class BoundaryCondition(Operator):
    """Wrapper class for Ad representations of boundary conditions for a given keyword."""

    def __init__(
        self,
        keyword: str,
        subdomains: List[pp.Grid],
        name: Optional[str] = None,
    ):
        """Construct a wrapper for boundary conditions for a set of subdomains.

        The boundary values will be ordered according to the ordering in subdomains. It is
        critical that the same ordering is used by other operators.

        IMPLEMENTATION NOTE: Only scalar quantities so far; vector operators will be
        added in due course.

        FIXME: Consider merging with ParameterArray by initializing the latter with
            param_keyword = self.keyword, and array_keyword='bc_values'.

        Parameters:
            keyword (str): Keyword that should be used to access the data dictionary
                to get the relevant boundary conditions.
            subdomains (List of pp.Grid): List of subdomains. The order of the subdomains
                in the list sets the ordering of the boundary values.
            name (str, optional): Name to be assigned to the operator. Default is None.

        """
        super().__init__(name=name)
        self.keyword = keyword
        self.subdomains: List[pp.Grid] = subdomains
        # FIXME: Super sets tree. Purge?
        self._set_tree()

    def __repr__(self) -> str:
        s = f"Boundary Condition operator with keyword {self.keyword}\n"

        dims = np.zeros(4, dtype=int)
        for sd in self.subdomains:
            dims[sd.dim] += 1
        for d in range(3, -1, -1):
            if dims[d] > 0:
                s += f"{dims[d]} subdomains of dimension {d}\n"
        return s

    def __str__(self) -> str:
        return f"BC({self.keyword})"

    def parse(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """Convert the Ad expression into numerical values for the boundary conditions,
        in the form of an np.ndarray concatenated for all subdomains.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. The boundary condition
                will be taken from the data dictionaries with the relevant keyword.

        Returns:
            np.ndarray: Value of boundary conditions.

        """
        val = []
        for sd in self.subdomains:
            data = mdg.subdomain_data(sd)
            val.append(data[pp.PARAMETERS][self.keyword]["bc_values"])
        return np.hstack([v for v in val])


class DirBC(Operator):
    """Extract (scalar) Dirichlet BC from Boundary condition.
    This can be e.g. useful when applying AD functions to boundary data.

    FIXME: Purge if not used. Looks tailor made/like a hack to IS.
    """

    def __init__(
        self,
        bc,
        subdomains: List[pp.Grid],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self._bc = bc
        self.subdomains: List[pp.Grid] = subdomains
        if not (len(self.subdomains) == 1):
            raise RuntimeError("DirBc not implemented for more than one grid.")
        self._set_tree()

    def __repr__(self) -> str:
        return f"Dirichlet boundary data of size {self._bc.val.size}"

    def parse(self, mdg: pp.MixedDimensionalGrid):
        bc_val = self._bc.parse(mdg)  # TODO Is this done anyhow already?
        keyword = self._bc.keyword
        g = self.subdomains[0]
        data = mdg.subdomain_data(g)
        bc = data[pp.PARAMETERS][keyword]["bc"]
        is_dir = bc.is_dir
        is_not_dir = np.logical_not(is_dir)
        dir_bc_val = bc_val.copy()
        dir_bc_val[is_not_dir] = float("NaN")

        return dir_bc_val


class ParameterArray(Operator):
    """Extract an array from the parameter dictionaries for a given set of subdomains.

    Can be used to implement sources, and general arrays to be picked from the
    parameter array (and thereby could be changed during the simulation, without
    having to redefine the abstract Ad representation of the equations).

    """

    def __init__(
        self,
        param_keyword: str,
        array_keyword: str,
        subdomains: Optional[List[pp.Grid]] = None,
        interfaces: Optional[List[pp.MortarGrid]] = None,
        name: Optional[str] = None,
    ):
        """Construct a wrapper for parameter arrays for a set of subdomains.

        The values of the parameter will be ordered according to the ordering
        in subdomains, or the order of the MixedDimensionalGrid iteration over subdomains.
        It is critical that the same ordering is used by other operators.

        IMPLEMENTATION NOTE: This class only takes care of parameter arrays. For
            parameters which are (left) multiplied with other terms, use ParameterMatrix.

        Parameters:
            param_keyword (str): Keyword that should be used to access the data dictionary
                to get the relevant parameter dictionary (same way as discretizations
                pick out their parameters).
            subdomains (List of pp.Grid): List of subdomains. The order of the subdomains in
                the list establishes the ordering of the parameter values.
            interfaces (List of pp.MortarGrid): List of edges. The order of the edges in the
                list establishes the ordering of the parameter values.
            name (str, optional): Name to be assigned to the array. Default is None.

        Example:
            To get the source term for a flow equation initialize with param_keyword='flow',
            and array_keyword='source'.

        """
        super().__init__(name=name)
        # Check that at least one of subdomains and edges is given and set empty list
        # if only one is not given
        if subdomains is None:
            subdomains = []
            if interfaces is None:
                raise ValueError(
                    "ParameterArray needs at least a list of subdomains or a list of edges"
                )
        elif interfaces is None:
            interfaces = []
        self.param_keyword = param_keyword
        self.array_keyword = array_keyword
        self.subdomains: List[pp.Grid] = subdomains
        self.interfaces: List[pp.MortarGrid] = interfaces
        self._set_tree()

    def __repr__(self) -> str:
        s = (
            f"Will access the parameter with keyword {self.param_keyword}"
            f" and keyword {self.array_keyword}"
        )

        dims = np.zeros(4, dtype=int)
        for sd in self.subdomains:
            dims[sd.dim] += 1
        for d in range(3, -1, -1):
            if dims[d] > 0:
                s += f"{dims[d]} subdomains of dimension {d}\n"
        dims = np.zeros(4, dtype=int)
        for intf in self.interfaces:
            dims[intf.dim] += 1
        for d in range(3, -1, -1):
            if dims[d] > 0:
                s += f"""{dims[d]} interfaces of dimension {d}\n"""
        return s

    def __str__(self) -> str:
        return f"ParameterArray({self.param_keyword})({self.array_keyword})"

    def parse(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """Convert the Ad expression into numerical values for the scalar sources,
        in the form of an np.ndarray concatenated for all subdomains.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. The boundary condition
                will be taken from the data dictionaries with the relevant keyword.

        Returns:
            np.ndarray: Value of boundary conditions.

        """
        val = []
        for sd in self.subdomains:
            data = mdg.subdomain_data(sd)
            val.append(data[pp.PARAMETERS][self.param_keyword][self.array_keyword])
        for intf in self.interfaces:
            data = mdg.interface_data(intf)
            val.append(data[pp.PARAMETERS][self.param_keyword][self.array_keyword])
        if len(val) > 0:
            return np.hstack([v for v in val])
        else:
            return np.array([])


class ParameterMatrix(ParameterArray):
    """Extract a matrix from the parameter dictionaries for a given set of subdomains.

    Typical use: Parameters which are left multiplied with an ad expression. Note that
        array parameters are represented by one diagonal matrix for each grid.

    """

    def __str__(self) -> str:
        return f"ParameterMatrix({self.param_keyword})({self.array_keyword})"

    def parse(self, mdg: pp.MixedDimensionalGrid) -> np.ndarray:
        """Convert the Ad expression into numerical values for the scalar sources,
        in the form of an np.ndarray concatenated for all subdomains.

        Parameters:
            mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. The boundary condition
                will be taken from the data dictionaries with the relevant keyword.

        Returns:
            sps.spmatrix: Value of boundary conditions.

        """
        val = []
        for sd in self.subdomains:
            data = mdg.subdomain_data(sd)
            val.append(data[pp.PARAMETERS][self.param_keyword][self.array_keyword])
        for intf in self.interfaces:
            data = mdg.interface_data(intf)
            val.append(data[pp.PARAMETERS][self.param_keyword][self.array_keyword])
        if len(val) > 0:
            return sps.diags(np.hstack([v for v in val]))
        else:
            return sps.csr_matrix((0, 0))


# Helper methods below


def _subgrid_projections(
    subdomains: List[pp.Grid], dim: int
) -> Tuple[Dict[pp.Grid, sps.spmatrix], Dict[pp.Grid, sps.spmatrix]]:
    """Construct prolongation matrices from individual subdomains to a set of subdomains.

    Args:
        subdomains: List of grids representing subdomains.
        dim: Dimension of the quantities to be projected. 1 corresponds to scalars, 2 to a
            vector of two components etc.

    Returns:
        cell_projection: Dictionary with the individual subdomains as keys and projection
            matrices for cell-based quantities as items.
        face_projection: Dictionary with the individual subdomains as keys and projection
            matrices for face-based quantities as items.


    The global cell and face numbering is set according to the order of the
    input subdomains.

    """
    face_projection: Dict[pp.Grid, np.ndarray] = {}
    cell_projection: Dict[pp.Grid, np.ndarray] = {}
    are_interfaces = isinstance(subdomains[0], pp.MortarGrid)
    if not are_interfaces:
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

        if not are_interfaces:
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
