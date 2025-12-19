"""This module contains (frontend) utility functions related to fractures and their
meshing."""

from __future__ import annotations

import logging

import numpy as np

import porepy as pp

# Module level logger
logger = logging.getLogger(__name__)


def fracture_length_2d(pts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Find the length of 2D fracture traces.

    Parameters:
        pts: ``shape=(2, np)``

            Coordinates of 2D points (column-wise),
            where ``np`` is the number of points.
        edges: ``shape=(2, nf)``

            For ``nf`` (line) fractures, contains the (column) indices of start and
            end points in ``pts`` of respective fracture .

    Returns:
        An array of length ``nf``, containing the length of each fracture based on
        the Euclidean distance between start and end points.

    """
    start = pts[:, edges[0]]
    end = pts[:, edges[1]]

    length = np.sqrt(np.sum(np.power(end - start, 2), axis=0))
    return length


def uniquify_points(
    pts: np.ndarray, edges: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniquify a set of points by merging almost coinciding coordinates.

    Also update fractures, and remove edges that consist of a single point (either
    after the points were merged, or because the input was a point edge).

    Parameters:
        pts: ``shape=(nd, np)

            A point cloud in dimension ``nd`` containing ``np`` point and their
            coordinates column-wise.
        edges: ``shape=(n, nf)``

            An array containing indices of start- and endpoints of
            fractures, in the first and second row respectively,
            referring to columns in ``pts``.
            Additional rows representing fracture tags are also accepted.
        tol: Tolerance used for merging points.

    Returns:
        A 3-tuple containing

        :obj:`~numpy.ndarray`: ``shape=(nd, np_new)``

            The new point cloud consisting of ``np_new`` points after the merger.
        :obj:`~numpy.ndarray`: ``shape=(n, nf_new)``

            Updated indexation of edges with the same structure as ``edges``.
        :obj:`~numpy.ndarray`: ``shape=(1, m)``

            Indices (referring to input ``edges``) of fractures deleted as they
            effectively contained a single coordinate.

    """

    # uniquify points based on coordinates
    p_unique, _, o2n = pp.array_operations.uniquify_point_set(pts, tol=tol)
    # update edges
    e_unique_p = np.vstack((o2n[edges[:2]], edges[2:]))

    # Find edges that start and end in the same point, and delete them
    point_edge = np.where(np.diff(e_unique_p[:2], axis=0)[0] == 0)[0].ravel()
    e_unique = np.delete(e_unique_p, point_edge, axis=1)

    return p_unique, e_unique, point_edge


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
