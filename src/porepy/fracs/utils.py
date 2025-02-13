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
    p_unique, _, o2n = pp.utils.setmembership.unique_columns_tol(pts, tol=tol)
    # update edges
    e_unique_p = np.vstack((o2n[edges[:2]], edges[2:]))

    # Find edges that start and end in the same point, and delete them
    point_edge = np.where(np.diff(e_unique_p[:2], axis=0)[0] == 0)[0].ravel()
    e_unique = np.delete(e_unique_p, point_edge, axis=1)

    return p_unique, e_unique, point_edge


def linefractures_to_pts_edges(
    fractures: list[pp.LineFracture], tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of line fractures into arrays of the corresponding points and
    edges.

    The function loops over the points of the individual fractures and checks if the
    point is the start/end point (up to the given tolerance) of a previously checked
    fracture. If yes, the edge index links to the existing point. If no, the point is
    added to the points array.

    Parameters:
        fractures: List of line fractures.
        tol: ``default=1e-8``

            Absolute tolerance to decide if start-/endpoints of two different fractures
            are equal. The comparison is done element-wise.

    Returns:
        A 2-tuple containing

        :obj:`~numpy.ndarray`: ``(shape=(2, num_points))``

            Coordinates of the start- and endpoints of the fractures.
        :obj:`~numpy.ndarray`: ``shape=(2 + num_tags, len(fractures)), dtype=int``

            An array containing column-wise (per fracture) the indices for the start-
            and endpoint in the first two rows.

            Note that one point in ``pts`` may be the start- and/or endpoint of multiple
            fractures.

            Additional rows are optional tags of the fractures. In the standard form,
            the third row (first row of tags) identifies the type of edges, referring
            to the numbering system in ``GmshInterfaceTags``. The second row of tags
            keeps track of the numbering of the edges (referring to the original
            order of the edges) in geometry processing like intersection removal.
            Additional tags can be assigned by the user.

        When an empty list of fractures is passed, both arrays have shape ``(2, 0)``.

    """
    pts_list: list[np.ndarray] = []
    edges_list: list[np.ndarray] = []

    # Iterate through the fractures and list all start-/endpoints and the corresponding
    # edge indices.
    for frac in fractures:
        pt_indices: list[int] = []
        for point in frac.points():
            # Check if the point is already start-/endpoint of another fracture.
            compare_points = [
                np.allclose(point.squeeze(), x, atol=tol) for x in pts_list
            ]
            if not any(compare_points):
                pts_list.append(point.squeeze())
                pt_indices.append(len(pts_list) - 1)
            else:
                pt_indices.append(compare_points.index(True))
        # Sanity check that two points indices were added.
        assert len(pt_indices) == 2
        # Combine with tags of the fracture and store the full edge in a list.
        edges_list.append(np.concatenate([np.array(pt_indices), frac.tags]))

    # Transform the lists to two ``np.ndarrays`` (``pts`` and ``edges``).
    if pts_list:
        # ``np.stack`` requires a nonempty list.
        pts = np.stack(pts_list, axis=-1)
    else:
        pts = np.zeros([2, 0])
    # Before creating the ``edges`` array, determine the maximum number of tags.
    # -> This determines the shape of the ``edges`` array.
    max_edge_dim = max((np.shape(edge)[0] for edge in edges_list), default=2)
    # Initialize the ``edges`` array with ``-1``. This value indicates that each edge
    # has no tags. Fill in the first two rows with the fracture start-/endpoints and
    # the rest of the rows with tags where they exist. All other tags keep their
    # initial value of ``-1``, which is equal to the tag not existing. This seemingly
    # complicated procedure is done to ensure that the ``edges`` array is not ragged.
    edges = np.full((max_edge_dim, len(fractures)), -1, dtype=np.int32)
    for row_index, edge in enumerate(edges_list):
        edges[: edge.shape[0], row_index] = edge

    return pts, edges


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
