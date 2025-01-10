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

    The class should be used through the methods {cell, face}_{projection, restriction}.

    Parameters:
        subdomains: List of grids for which the projections should map to and from.
        dim: Dimension of the quantities to be mapped. Will typically be 1 (for scalar
            quantities) or Nd (the ambient dimension, for vector quantities).

    Raises:
        ValueError: If a subdomain occur more than once in the input list.

    See also:
        MortarProjections for projections to and from mortar subdomains.

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

    def cell_restriction(self, subdomains: list[pp.Grid]) -> SparseArray:
        """Construct restrictions from global to subdomain cell quantities.

        Parameters:
            subdomains: One or several subdomains to which the projection should apply.

        Raises:
            ValueError: If subdomains is not a list.

        Returns:
            pp.ad.SparseArray: Matrix operator (in the Ad sense) that represents the
                projection.

        """
        if not isinstance(subdomains, list):
            raise ValueError("Subdomains should be a list of grids")

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
            raise ValueError("Subdomains should be a list of grids")

        if len(subdomains) > 0:
            # A key error will be raised if a grid in g is not known to
            # self._cell_projection
            # IMPLEMENTATION NOTE: Use csc format, since the number of columns can
            # be much less than the number of rows.
            mat = sps.bmat([[self._cell_projection[g] for g in subdomains]]).tocsc()
        else:
            # If the grid list is empty, we project from nothing to the full set of
            # cells. CSC format is used for efficiency.
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
        if not isinstance(subdomains, list):
            raise ValueError("Subdomains should be a list of grids")

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
        if not isinstance(subdomains, list):
            raise ValueError("Subdomains should be a list of grids")

        if len(subdomains) > 0:
            # A key error will be raised if a grid in subdomains is not known to
            # self._face_projection
            # IMPLEMENTATION NOTE: Use csc format, since the number of columns can
            # be far smaller than the number of rows.
            mat = sps.bmat([[self._face_projection[g] for g in subdomains]]).tocsc()
        else:
            # If the grid list is empty, we project from nothing to the full set of
            # faces.
            mat = sps.csc_matrix((self._tot_num_faces * self.dim, 0))
        return pp.ad.SparseArray(mat, name="FaceProlongation")

    def __repr__(self) -> str:
        s = (
            f"Restriction and prolongation operators for {self._num_grids}"
            " unique subdomains\n"
            f"Aimed at variables with dimension {self.dim}\n"
        )
        return s


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


    def sign_of_mortar_sides(self) -> SparseArray:
        if len(interfaces) == 0:
            return SparseArray(
                sps.bmat([[None]]), name="SignOfMortarSides"
            )
        mats = []
        for intf in interfaces:
            assert isinstance(intf, pp.MortarGrid)  # Appease mypy
            mats.append(intf.sign_of_mortar_sides(dim))
        else:
            return SparseArray(
                sps.block_diag(mats), name="SignOfMortarSides"
            )

    def mortar_to_primary_int(self) -> Operator:

        face_projections = _face_projections(self._subdomains, self.dim)
        proj_mats = []
        for intf in self._interfaces:
            sd_primary, _ = self._mdg.interface_to_subdomain_pair(intf)
            if sd_primary in self._subdomains:
                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                            face_projections[sd_primary]
                            * intf.mortar_to_primary_int(self.dim)
                        )
                )
            else:
                # TODO: Optimized storage
                size = self.dim * sum([sd.num_faces for sd in self._subdomains])
                proj_mats.append(sps.csr_matrix((size, intf.num_cells * self.dim)))
        return self._bmat([proj_mats], name="MortarToPrimaryInt")

    
    def mortar_to_primary_avg(self) -> Operator:

        proj_mats = []
        face_projections = _face_projections(self._subdomains, self.dim)
        for intf in self._interfaces:
            sd_primary, _ = self._mdg.interface_to_subdomain_pair(intf)
            if sd_primary in self._subdomains:
                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                            face_projections[sd_primary]
                            * intf.mortar_to_primary_avg(self.dim)
                        )
                )
            else:
                size = self.dim * sum([sd.num_faces for sd in self._subdomains])
                proj_mats.append(sps.csr_matrix((size, intf.num_cells * self.dim)))
        return self._bmat([proj_mats], name="MortarToPrimaryAvg")

    def primary_to_mortar_int(self) -> Operator:

        proj_mats = []
        face_projections = _face_projections(self._subdomains, self.dim)
        for intf in self._interfaces:
            sd_primary, _ = self._mdg.interface_to_subdomain_pair(intf)
            if sd_primary in self._subdomains:
                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                            intf.primary_to_mortar_int(self.dim)
                            * face_projections[sd_primary].T
                        )
                )
            else:
                size = self.dim * sum([sd.num_faces for sd in self._subdomains])
                proj_mats.append(sps.csr_matrix((intf.num_cells * self.dim, size)))
        return self._bmat([[m] for m in proj_mats], name="PrimaryToMortarInt")
    
    def primary_to_mortar_avg(self) -> Operator:
        proj_mats = []
        face_projections = _face_projections(self._subdomains, self.dim)
        for intf in self._interfaces:
            sd_primary, _ = self._mdg.interface_to_subdomain_pair(intf)
            if sd_primary in self._subdomains:
                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                            intf.primary_to_mortar_avg(self.dim)
                            * face_projections[sd_primary].T
                        )
                )
            else:
                size = self.dim * sum([sd.num_faces for sd in self._subdomains])
                proj_mats.append(sps.csr_matrix((intf.num_cells * self.dim, size)))
        return self._bmat([[m] for m in proj_mats], name="PrimaryToMortarAvg")

    def mortar_to_secondary_int(self) -> Operator:
        proj_mats = []
        cell_projection = _cell_projections(self._subdomains, self.dim)

        for intf in self._interfaces:
            _, sd_secondary = self._mdg.interface_to_subdomain_pair(intf)
            if sd_secondary in self._subdomains:
                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                            cell_projection[sd_secondary]
                            * intf.mortar_to_secondary_int(self.dim)
                        )
                )
            else:
                size = self.dim * sum([sd.num_cells for sd in self._subdomains])
                proj_mats.append(sps.csr_matrix((size, intf.num_cells * self.dim)))

        if len(proj_mats) == 0:
            proj_mats.append(sps.csc_matrix((self.dim * sum([sd.num_cells for sd in self._subdomains]), 0)))

        return self._bmat([proj_mats], name="MortarToSecondaryInt")

    def mortar_to_secondary_avg(self) -> Operator:
        proj_mats = []
        cell_projection = _cell_projections(self._subdomains, self.dim)

        for intf in self._interfaces:
            _, sd_secondary = self._mdg.interface_to_subdomain_pair(intf)
            if sd_secondary in self._subdomains:
                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                            cell_projection[sd_secondary]
                            * intf.mortar_to_secondary_avg(self.dim)
                        )
                )
            else:
                size = self.dim * sum([sd.num_cells for sd in self._subdomains])
                proj_mats.append(sps.csr_matrix((size, intf.num_cells * self.dim)))

        if len(proj_mats) == 0:
            proj_mats.append(sps.csc_matrix((self.dim * sum([sd.num_cells for sd in self._subdomains]), 0)))

        return self._bmat([proj_mats], name="MortarToSecondaryAvg")

    def secondary_to_mortar_int(self) -> Operator:
        proj_mats = []
        cell_projection = _cell_projections(self._subdomains, self.dim)

        for intf in self._interfaces:
            _, sd_secondary = self._mdg.interface_to_subdomain_pair(intf)
            if sd_secondary in self._subdomains:
                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                            intf.secondary_to_mortar_int(self.dim)
                            *cell_projection[sd_secondary].T
                        )
                )
            else:
                size = self.dim * sum([sd.num_cells for sd in self._subdomains])
                proj_mats.append(sps.csr_matrix((intf.num_cells * self.dim, size)))

        if len(proj_mats) == 0:
            proj_mats.append(sps.csr_matrix((0, self.dim * sum([sd.num_cells for sd in self._subdomains]))))

        return self._bmat([[m] for m in proj_mats], name="SecondaryToMortarInt")

    def secondary_to_mortar_avg(self) -> Operator:
        proj_mats = []
        cell_projection = _cell_projections(self._subdomains, self.dim)

        for intf in self._interfaces:
            _, sd_secondary = self._mdg.interface_to_subdomain_pair(intf)
            if sd_secondary in self._subdomains:
                proj_mats.append(pp.matrix_operations.optimized_compressed_storage(
                            intf.secondary_to_mortar_avg(self.dim)
                            *cell_projection[sd_secondary].T
                        )
                )
            else:
                size = self.dim * sum([sd.num_cells for sd in self._subdomains])
                proj_mats.append(sps.csr_matrix((intf.num_cells * self.dim, size)))
        if len(proj_mats) == 0:
            proj_mats.append(sps.csr_matrix((0, self.dim * sum([sd.num_cells for sd in self._subdomains]))))

        return self._bmat([[m] for m in proj_mats], name="SecondaryToMortarAvg")

    def _bmat(self, matrices, name):
        # Create block matrix, convert it to optimized storage format.
        if len(matrices[0]) == 0:
            block_matrix = sps.csr_matrix((0, 0))
        else:
            block_matrix = pp.matrix_operations.optimized_compressed_storage(
                sps.bmat(matrices)
            )
        return SparseArray(block_matrix, name=name)                                    

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
        face_projections = _face_projections(subdomains, dim)

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

        cell_projections = _cell_projections(subdomains, self.dim)

        trace: list[sps.spmatrix] = []

        if len(subdomains) > 0:
            for sd in subdomains:
                if self._is_scalar:
                    # Local trace operator.
                    sd_trace = sd.trace(dim=self.dim)
                    # Restrict global cell values to the local grid.
                    trace.append(sd_trace * cell_projections[sd].T)

                else:
                    raise NotImplementedError("kronecker")
        else:
            trace = [sps.csr_matrix((0, 0))]
        # Stack trace vertically to make them into mappings to global quantities. Wrap
        # the stacked matrices into an AD object.
        self.trace = SparseArray(sps.bmat([[m] for m in trace]).tocsr())
        """ Matrix of trace projections from cells to faces."""

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
        mat = [sd.divergence(dim=self.dim) for sd in self.subdomains]
        matrix = sps.block_diag(mat)
        return matrix


def _cell_projections(subdomains: Sequence[pp.Grid], dim: int) -> dict[pp.Grid, sps.spmatrix]:
    """Construct prolongation matrices for cell-based quantities from individual subdomains to a set of subdomains.

    Parameters:
        subdomains: List of grids representing subdomains.
        dim: Dimension of the quantities to be projected. 1 corresponds to scalars, 2 to
            a vector of two components etc.

    Returns:
        cell_projection: Dictionary with the individual subdomains as keys and
            projection matrices for cell-based quantities as items.

    The global cell numbering is set according to the order of the input subdomains.

    """
    cell_projection: dict[pp.Grid, np.ndarray] = {}
    if len(subdomains) == 0:
        return cell_projection

    tot_num_cells = np.sum([g.num_cells for g in subdomains]) * dim
    cell_offset = 0

    for sd in subdomains:
        cell_ind = cell_offset + pp.fvutils.expand_indices_nd(np.arange(sd.num_cells), dim)
        cell_sz = sd.num_cells * dim

        cell_projection[sd] = sps.coo_matrix(
            (np.ones(cell_sz), (cell_ind, np.arange(cell_sz))),
            shape=(tot_num_cells, cell_sz),
        ).tocsc()
        cell_offset = cell_ind[-1] + 1

    return cell_projection

def _face_projections(subdomains: Sequence[pp.Grid], dim: int) -> dict[pp.Grid, sps.spmatrix]:
    """Construct prolongation matrices for face-based quantities from individual subdomains to a set of subdomains.

    Parameters:
        subdomains: List of grids representing subdomains.
        dim: Dimension of the quantities to be projected. 1 corresponds to scalars, 2 to
            a vector of two components etc.

    Returns:
        face_projection: Dictionary with the individual subdomains as keys and
        projection matrices for face-based quantities as items.

    The global face numbering is set according to the order of the input subdomains.

    """
    face_projection: dict[pp.Grid, np.ndarray] = {}
    if len(subdomains) == 0:
        return face_projection

    tot_num_faces = np.sum([g.num_faces for g in subdomains]) * dim
    face_offset = 0

    for sd in subdomains:
        face_ind = face_offset + pp.fvutils.expand_indices_nd(np.arange(sd.num_faces), dim)
        face_sz = sd.num_faces * dim

        face_projection[sd] = sps.coo_matrix(
            (np.ones(face_sz), (face_ind, np.arange(face_sz))),
            shape=(tot_num_faces, face_sz),
        ).tocsc()

        if sd.dim > 0:
            face_offset = face_ind[-1] + 1

    return face_projection
