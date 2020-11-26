from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps

from .operators import Operator, MergedOperator, Matrix


__all__ = [
    "MortarProjections",
    "Divergence",
    "BoundaryCondition",
    "Trace",
    "SubdomainProjections",
]


def _subgrid_projections(
    grids: List[pp.Grid], nd: int
) -> Tuple[Dict[pp.Grid, sps.spmatrix], Dict[pp.Grid, sps.spmatrix]]:
    """Construct prolongation matrices from individual grids to a set of grids.

    Matrices for both cells and faces are constructed.

    The global cell and face numbering is set according to the order of the
    input grids.

    """
    face_projection: Dict[pp.Grid, np.ndarray] = {}
    cell_projection: Dict[pp.Grid, np.ndarray] = {}

    tot_num_faces = np.sum([g.num_faces for g in grids])
    tot_num_cells = np.sum([g.num_cells for g in grids])

    face_offset = 0
    cell_offset = 0

    if nd != 1:
        raise NotImplementedError("Need vector version of projections. Kronecker")

    for g in grids:
        face_ind = face_offset + pp.fvutils.expand_indices_nd(
            np.arange(g.num_faces), nd
        )
        cell_ind = cell_offset + pp.fvutils.expand_indices_nd(
            np.arange(g.num_cells), nd
        )

        face_sz, cell_sz = g.num_faces * nd, g.num_cells * nd
        face_projection[g] = sps.coo_matrix(
            (np.ones(face_sz), (face_ind, np.arange(face_sz))),
            shape=(tot_num_faces, face_sz),
        ).tocsr()
        cell_projection[g] = sps.coo_matrix(
            (np.ones(cell_sz), (cell_ind, np.arange(cell_sz))),
            shape=(tot_num_cells, cell_sz),
        ).tocsr()

        # Correct start of the numbering for the next grid
        if g.dim > 0:
            face_offset = face_ind[-1] + 1
        cell_offset = cell_ind[-1] + 1

    return cell_projection, face_projection


class SubdomainProjections(Operator):
    def __init__(
        self,
        gb: Optional[List[pp.Grid]] = None,
        grids: Optional[List[pp.Grid]] = None,
        is_scalar: bool = True,
    ) -> None:
        if grids is None:
            if gb is None:
                raise ValueError(
                    "Trace needs either either a list of grids or a GridBucket"
                )
            grids = [g for g, _ in gb]

        self._is_scalar: bool = is_scalar
        if self._is_scalar:
            self._nd: int = 1
        else:
            self._nd = gb.dim_max()
        self._num_grids: int = len(grids)

        self._cell_projection, self._face_projection = _subgrid_projections(
            grids, self._nd
        )

    def cell_restriction(self, grids: Union[pp.Grid, List[pp.Grid]]) -> Matrix:
        if isinstance(grids, pp.Grid):
            return pp.ad.Matrix(self._cell_projection[grids].T)
        elif isinstance(grids, list):
            # A key error will be raised if a grid in g is not known to self._cell_projection
            return pp.ad.Matrix(
                sps.bmat([[self._cell_projection[g].T] for g in grids]).tocsr()
            )
        else:
            raise ValueError("Argument should be a grid or a list of grids")

    def cell_prolongation(self, grids: Union[pp.Grid, List[pp.Grid]]) -> Matrix:
        if isinstance(grids, pp.Grid):
            return pp.ad.Matrix(self._cell_projection[grids])
        elif isinstance(grids, list):
            # A key error will be raised if a grid in g is not known to self._cell_projection
            return pp.ad.Matrix(
                sps.bmat([self._cell_projection[g] for g in grids]).tocsr()
            )
        else:
            raise ValueError("Argument should be a grid or a list of grids")

    def face_restriction(self, grids: Union[pp.Grid, List[pp.Grid]]) -> Matrix:
        if isinstance(grids, pp.Grid):
            return pp.ad.Matrix(self._face_projection[grids].T)
        elif isinstance(grids, list):
            # A key error will be raised if a grid in g is not known to self._cell_projection
            return pp.ad.Matrix(
                sps.bmat([[self._face_projection[g].T] for g in grids]).tocsr()
            )
        else:
            raise ValueError("Argument should be a grid or a list of grids")

    def face_prolongation(self, grids: Union[pp.Grid, List[pp.Grid]]) -> Matrix:
        if isinstance(grids, pp.Grid):
            return pp.ad.Matrix(self._face_projection[grids])
        elif isinstance(grids, list):
            # A key error will be raised if a grid in g is not known to self._cell_projection
            return pp.ad.Matrix(
                sps.bmat([self._face_projection[g] for g in grids]).tocsr()
            )
        else:
            raise ValueError("Argument should be a grid or a list of grids")

    def __repr__(self) -> str:
        s = (
            f"Restriction and prolongation operators for {self._num_grids} grids\n"
            f"Aimed at variables with dimension {self._nd}\n"
        )
        return s


class MortarProjections(Operator):
    def __init__(
        self,
        gb: pp.GridBucket,
        grids: Optional[List[pp.Grid]] = None,
        edges: Optional[List[Tuple[pp.Grid, pp.Grid]]] = None,
        nd: int = 1,
    ):
        if grids is None:
            grids = [g for g, _ in gb.nodes()]
        if edges is None:
            edges = [e for e, _ in gb.edges()]

        self._num_edges: int = len(edges)
        self._nd: int = nd

        ## Initialize projections

        cell_projection, face_projection = _subgrid_projections(grids, self._nd)

        # sparse blocks are slow; it should be possible to do a right multiplication
        # of local-to-global mortar indices instead of the block.

        mortar_to_primary_int, mortar_to_primary_avg = [], []
        primary_to_mortar_int, primary_to_mortar_avg = [], []

        mortar_to_secondary_int, mortar_to_secondary_avg = [], []
        secondary_to_mortar_int, secondary_to_mortar_avg = [], []

        # The goal is to construct global projections between grids and mortar grids.
        # The construction takes two stages, and is different for projections to and
        # from the mortar grid:
        # For projections from the mortar grid, a mapping is first made from local
        # mortar numbering global grid ordering. In the second stage, the mappings from
        # mortar are stacked to make a global mapping.
        # Projections to the mortar grid are made by first defining projections from
        # global grid numbering to local mortar grids, and then stack the latter.

        for e in edges:
            g_primary, g_secondary = e
            mg: pp.MortarGrid = gb.edge_props(e, "mortar_grid")
            if (g_primary.dim != mg.dim + 1) or g_secondary.dim != mg.dim:
                # This will correspond to DD of sorts; we could handle this
                # by using cell_projections for g_primary and/or
                # face_projection for g_secondary, depending on the exact
                # configuration
                raise NotImplementedError("Non-standard interface.")

            # Projections to primary
            mortar_to_primary_int.append(
                face_projection[g_primary] * mg.mortar_to_primary_int(nd)
            )
            mortar_to_primary_avg.append(
                face_projection[g_primary] * mg.mortar_to_primary_avg(nd)
            )

            # Projections from primary
            primary_to_mortar_int.append(
                mg.primary_to_mortar_int(nd) * face_projection[g_primary].T
            )
            primary_to_mortar_avg.append(
                mg.primary_to_mortar_avg(nd) * face_projection[g_primary].T
            )

            mortar_to_secondary_int.append(
                cell_projection[g_secondary] * mg.mortar_to_secondary_int(nd)
            )
            mortar_to_secondary_avg.append(
                cell_projection[g_secondary] * mg.mortar_to_secondary_avg(nd)
            )

            secondary_to_mortar_int.append(
                mg.secondary_to_mortar_int(nd) * cell_projection[g_secondary].T
            )
            secondary_to_mortar_avg.append(
                mg.secondary_to_mortar_avg(nd) * cell_projection[g_secondary].T
            )

        # Stack mappings from the mortar horizontally.
        # The projections are wrapped by a pp.ad.Matrix to be compatible with the
        # requirements for processing of Ad operators.
        self.mortar_to_primary_int = Matrix(sps.bmat([mortar_to_primary_int]).tocsr())
        self.mortar_to_primary_avg = Matrix(sps.bmat([mortar_to_primary_avg]).tocsr())
        self.mortar_to_secondary_int = Matrix(
            sps.bmat([mortar_to_secondary_int]).tocsr()
        )
        self.mortar_to_secondary_avg = Matrix(
            sps.bmat([mortar_to_secondary_avg]).tocsr()
        )

        # Vertical stacking of the projections
        self.primary_to_mortar_int = Matrix(
            sps.bmat([[m] for m in primary_to_mortar_int]).tocsr()
        )
        self.primary_to_mortar_avg = Matrix(
            sps.bmat([[m] for m in primary_to_mortar_avg]).tocsr()
        )
        self.secondary_to_mortar_int = Matrix(
            sps.bmat([[m] for m in secondary_to_mortar_int]).tocsr()
        )
        self.secondary_to_mortar_avg = Matrix(
            sps.bmat([[m] for m in secondary_to_mortar_avg]).tocsr()
        )

    def __repr__(self) -> str:
        s = (
            f"Mortar projection for {self._num_edges} interfaces\n"
            f"Aimed at variables with dimension {self._nd}\n"
            f"Projections to primary have dimensions {self.mortar_to_primary_avg.shape}\n"
            f"Projections to secondary have dimensions {self.mortar_to_secondary_avg.shape}\n"
        )
        return s


class Trace(MergedOperator):
    """Mapping from grid faces to cell centers.

    The mapping will hit both boundary and interior faces, so the values
    to be mapped should be carefully filtered (e.g. by combining it with a
    mortar mapping).

    The mapping does not alter signs of variables, that is, the direction
    of face normal vectors is not accounted for.

    """

    def __init__(
        self,
        gb: Optional[List[pp.Grid]] = None,
        grids: Optional[List[pp.Grid]] = None,
        is_scalar: bool = True,
    ):

        if grids is None:
            if gb is None:
                raise ValueError(
                    "Trace needs either either a list of grids or a GridBucket"
                )
            grids = [g for g, _ in gb]

        self._is_scalar: bool = is_scalar
        if self._is_scalar:
            self._nd: int = 1
        else:
            self._nd = gb.dim_max()
        self._num_grids: int = len(grids)

        cell_projections, face_projections = _subgrid_projections(grids, self._nd)

        trace: sps.spmatrix = []
        inv_trace: sps.spmatrix = []

        for g in grids:
            if self._is_scalar:

                # TEMPORARY CONSTRUCT: Use the divergence operator as a trace.
                # It would be better to define a dedicated function for this,
                # perhaps in the grid itself.
                div = np.abs(pp.fvutils.scalar_divergence(g))

                # Restrict global cell values to the local grid, use transpose of div
                # to map cell values to faces.
                trace.append(div.T * cell_projections[g].T)
                # Similarly restrict a global face quantity to the local grid, then
                # map back to cells.
                inv_trace.append(div * face_projections[g].T)
            else:
                raise NotImplementedError("kronecker")

        # Stack both trace and inv_trace vertically to make them into mappings to
        # global quantities.
        # Wrap the stacked matrices into an Ad object
        self.trace = Matrix(sps.bmat([[m] for m in trace]).tocsr())
        self.inv_trace = Matrix(sps.bmat([[m] for m in inv_trace]).tocsr())

    def __repr__(self) -> str:
        s = (
            f"Trace operator for {self._num_grids} grids\n"
            f"Aimed at variables with dimension {self._nd}\n"
            f"Projection from grid to mortar has dimensions {self.trace}\n"
        )
        return s


class Divergence(MergedOperator):
    def __init__(self, grids, is_scalar=True):
        self.g = grids
        self.scalar = is_scalar
        self._set_tree(None)

    def __repr__(self) -> str:
        if self.scalar:
            s = "Scalar "
        else:
            s = "Vector "

        s += f"divergence defined on {len(self.g)} grids\n"

        nf = 0
        nc = 0
        for g in self.g:
            if self.scalar:
                nf += g.num_faces
                nc += g.num_cells
            else:
                # EK: the notion of vector divergence for grids of co-dimension >= 1
                # is not clear, but we ignore this here.
                nf += g.num_faces * g.dim
                nc += g.num_cells * g.dim

        s += f"The total size of the matrix is ({nc}, {nf})\n"

        return s


class BoundaryCondition(MergedOperator):
    def __init__(self, keyword, grids):
        self.keyword = keyword
        self.g = grids
        self._set_tree()

    def __repr__(self) -> str:
        s = f"Boundary Condition operator with keyword {self.keyword}\n"

        dims = np.zeros(4, dtype=np.int)
        for g in self.g:
            dims[g.dim] += 1

        for d in range(3, -1, -1):
            if dims[d] > 0:
                s += f"{dims[d]} grids of dimension {d}\n"

        return s
