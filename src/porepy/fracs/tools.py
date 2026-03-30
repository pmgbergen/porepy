"""This module contains technical tools used in the treatment of fractures.

This can be thought of as a module for backend utility functions, as opposed to the
frontend functions found in :mod:`~porepy.fracs.utils`.

"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp


def obtain_interdim_mappings(
    g: pp.Grid, fn: sps.spmatrix, n_per_face: int
) -> tuple[np.ndarray, np.ndarray]:
    """Finds mappings between faces in higher dimension and cells in the lower
    dimension.

    Parameters:
        g: Lower dimensional grid.
        fn: Face-node map of the higher-dimensional grid
            (see :data:`~porepy.grids.grid.Grid.face_nodes`).
        n_per_face: Number of nodes per face in the higher-dimensional grid.

    Returns:
        A 2-tuple containing

        :obj:`~numpy.ndarray`:
            An array containing indices of faces in the higher-dimensional grid that
            correspond to a cell in the lower-dimensional grid.
            The indexing is based on **all** cells in the lower-dimensional grid.
        :obj:`~numpy.ndarray`:
            Indices of the corresponding cells in the lower-dimensional grid.

    """
    if g.dim > 0:
        cn_loc = g.cell_nodes().indices.reshape((n_per_face, g.num_cells), order="F")
        cn = g.global_point_ind[cn_loc]
        cn = np.sort(cn, axis=0)
    else:
        cn = np.array([g.global_point_ind])
        # We also know that the higher-dimensional grid has faces of a single node.
        # This sometimes fails, so enforce it.
        if cn.ndim == 1:
            fn = fn.ravel()
    is_mem, cell_2_face = pp.array_operations.ismember_columns(
        cn.astype(np.int32), fn.astype(np.int32), sort=False
    )
    # An element in cell_2_face gives, for all cells in the lower-dimensional grid,
    # the index of the corresponding face in the higher-dimensional structure.
    if not (np.all(is_mem) or np.all(~is_mem)):
        warnings.warn(
            """Found inconsistency between cells and higher dimensional faces.
            Continuing, fingers crossed"""
        )
    low_dim_cell = np.where(is_mem)[0]
    return cell_2_face, low_dim_cell
