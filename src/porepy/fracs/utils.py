"""This module contains (frontend) utility functions related to fractures and their
meshing."""

from __future__ import annotations

import numpy as np

import porepy as pp


def pts_edges_to_linefractures(
    pts: np.ndarray, edges: np.ndarray
) -> list[pp.LineFracture]:
    """Convert points and edges into a list of line fractures.

    Parameters:
        pts: ``shape=(2, num_points)``

            2D coordinates of the start- and endpoints of the fractures.
        edges: ``shape=(2 + num_tags, num_fracs), dtype=int``

            Indices for the start- and endpoint of each fracture. Note, that one point
            in ``pts`` may be the start- and/or endpoint of multiple fractures.

            Additional rows are optional tags of the fractures. In the standard form,
            the third row (first row of tags) identifies the type of edges, referring
            to the numbering system in ``GmshInterfaceTags``. The second row of tags
            keeps track of the numbering of the edges (referring to the original
            order of the edges) in geometry processing like intersection removal.
            Additional tags can be assigned by the user.

    Returns:
        A list of line fractures resulting from above definition of points and edges.

    """
    fractures: list[pp.LineFracture] = []
    for start_index, end_index, *tags in edges.T:
        fractures.append(
            pp.LineFracture(
                np.array([pts[:, start_index], pts[:, end_index]]).T, tags=tags
            )
        )
    return fractures
