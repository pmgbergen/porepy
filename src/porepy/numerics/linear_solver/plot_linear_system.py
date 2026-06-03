"""This module contains routines for visualization of a block linear system. They are
meant to work with the `BlockLinearSystem` object.

"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import csr_matrix, spmatrix

if TYPE_CHECKING:
    from porepy.numerics.linear_solver import BlockLinearSystem


__all__ = ["spy", "color_spy", "matshow", "plot_max", "plot_vector"]


def spy(
    mat: spmatrix,
    show: bool = True,
    aspect: Literal["equal", "auto"] = "equal",
    marker: Optional[str] = None,
) -> None:
    if marker is None:
        marker = "+"
        if max(*mat.shape) > 300:
            marker = ","
    plt.spy(mat, marker=marker, markersize=4, color="black", aspect=aspect)
    if show:
        plt.show()


def color_spy(
    bmat: BlockLinearSystem,
    aspect: Literal["equal", "auto"] = "equal",
    show: bool = False,
    marker: Optional[str] = None,
    draw_marker: bool = True,
    color: bool = True,
    hatch: bool = True,
    alpha: float = 0.3,
) -> None:
    """Draws a sparse matrix stencil and colors the rows and columns to distinguish
    different groups.

    Parameters:
        bmat: The matrix to display.
        show: Whether to call `plt.show()`.
        aspect: Passed to `plt.spy()`. "auto" can be useful if the aspect ratio is
            too high. Otherwise, "equal" is better.
        marker: Which marker to use for the matrix stencil. Default value changes
            based on the matrix size.
        color: Whether to color the background.
        hatch: Whether to hatch the background.
        draw_marker: If not, only colors / hatches the background.
        alpha: Background transparency.

    """
    row_idx = [bmat.indexer.dofs_row[i] for i in bmat.indexer.enabled_groups_row]
    col_idx = [bmat.indexer.dofs_col[i] for i in bmat.indexer.enabled_groups_col]
    row_names, col_names = _enabled_group_names(bmat)

    if draw_marker:
        spy(bmat.mat, show=False, aspect=aspect, marker=marker)
    else:
        spy(csr_matrix(bmat.shape), show=False, aspect=aspect)

    row_sep = [0]
    for row in row_idx:
        if len(row) == 0:
            row_sep.append(row_sep[-1])
        else:
            row_sep.append(row[-1] + 1)
    row_sep = sorted(row_sep)

    col_sep = [0]
    for col in col_idx:
        if len(col) == 0:
            col_sep.append(col_sep[-1])
        else:
            col_sep.append(col[-1] + 1)
    col_sep = sorted(col_sep)

    if row_names is None:
        row_names = [str(i) for i in range(len(row_sep) - 1)]
    if col_names is None:
        col_names = [str(i) for i in range(len(col_sep) - 1)]

    hatch_types = itertools.cycle(["/", "\\"])

    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_names)):
        ystart, yend = row_sep[i : i + 2]
        row_label_pos.append(ystart + (yend - ystart) / 2)
        kwargs: dict[str, Any] = {}
        if color:
            kwargs["facecolor"] = f"C{i}"
        else:
            kwargs["fill"] = False
        if hatch:
            kwargs["hatch"] = next(hatch_types)
            kwargs["edgecolor"] = "red"

        plt.axhspan(ystart - 0.5, yend - 0.5, alpha=alpha, **kwargs)
    ax.yaxis.set_ticks(row_label_pos)
    ax.set_yticklabels(row_names, rotation=0)

    col_label_pos = []
    for i in range(len(col_names)):
        xstart, xend = col_sep[i : i + 2]
        col_label_pos.append(xstart + (xend - xstart) / 2)
        if color:
            kwargs["facecolor"] = f"C{i}"
        if hatch:
            kwargs["hatch"] = next(hatch_types)
        plt.axvspan(xstart - 0.5, xend - 0.5, alpha=alpha, **kwargs)
    ax.xaxis.set_ticks(col_label_pos)
    ax.set_xticklabels(col_names, rotation=0)

    if show:
        plt.show()


def matshow(
    mat: spmatrix,
    log: bool = True,
    show: bool = True,
    threshold: float = 1e-30,
    aspect: Literal["equal", "auto"] = "equal",
) -> None:
    """Displays the values of a sparse matrix. Converts it to a dense matrix, so
    should not be called for large matrices.

    Parameters:
        mat: The matrix to display.
        log: Whether to use the log scale.
        show: Whether to call `plt.show()`.
        threshold: Does not display the values with `abs(x) < threshold`.
        aspect: Passed to `plt.matshow()`. "auto" can be useful if the aspect ratio
        is too high. Otherwise, "equal" is better.

    """
    mat = mat.copy()
    try:
        mat = mat.toarray()
    except AttributeError:
        # If mat does not have a toarray() method, assume it is already a dense array.
        pass

    mat[abs(mat) < threshold] = np.nan
    if log:
        mat = np.log10(abs(mat))

    plt.matshow(mat, fignum=0, aspect=aspect)
    plt.colorbar()
    if show:
        plt.show()


def plot_max(bmat: BlockLinearSystem, annot: bool = True, mean: bool = False):
    """Displays a table, where each cell represents a matrix group. Useful to get the
    idea about the groups and which of them are not empty.

    Parameters:
        annot: Display the aggregated max / mean value for each cell.
        mean: Whether to show the mean value. Otherwise, shows the abs(max) value.

    """
    row_idx = [bmat.indexer.dofs_row[i] for i in bmat.indexer.enabled_groups_row]
    col_idx = [bmat.indexer.dofs_col[i] for i in bmat.indexer.enabled_groups_col]
    row_names, col_names = _enabled_group_names(bmat)
    data = []

    for row in row_idx:
        row_data = []
        for col in col_idx:
            ind_i, ind_j = np.meshgrid(row, col, sparse=True, indexing="ij", copy=False)
            submat = bmat.mat[ind_i, ind_j]
            if submat.data.size == 0:
                row_data.append(np.nan)
            else:
                if not mean:
                    row_data.append(abs(submat).max())
                else:
                    row_data.append(abs(submat).mean())
        data.append(row_data)

    ax = plt.gca()
    sns.heatmap(
        data=np.array(data),
        square=False,
        annot=annot,
        norm=LogNorm(),
        fmt=".1e",
        xticklabels=col_names,
        yticklabels=row_names,
        ax=ax,
        linewidths=0.01,
        linecolor="grey",
        cbar=False,
        cmap=sns.color_palette("coolwarm", as_cmap=True),
    )


def _enabled_group_names(bmat: BlockLinearSystem) -> tuple[list[str], list[str]]:
    return (
        [
            f"{i}: {bmat.indexer.group_names_row[i]}"
            for i in bmat.indexer.enabled_groups_row
        ],
        [
            f"{i}: {bmat.indexer.group_names_col[i]}"
            for i in bmat.indexer.enabled_groups_col
        ],
    )


def plot_vector(
    bmat: BlockLinearSystem,
    vec: np.ndarray,
    side: Literal["left", "right"],
    log: bool = True,
):
    """Displays a vector and colors the background to see different groups.

    Parameters:
        bmat: The current block linear system object.
        vec: The vector, in the same arrangement as the current block linear system.
        side: For Ax = b, use "left" for the vector b, and "right" for the vector x.
        log: Whether to use the log scale.

    """
    row_names, column_names = _enabled_group_names(bmat)

    if side == "left":
        group_names = row_names
        dofs = [bmat.indexer.dofs_row[i] for i in bmat.indexer.enabled_groups_row]
    elif side == "right":
        group_names = column_names
        dofs = [bmat.indexer.dofs_col[i] for i in bmat.indexer.enabled_groups_col]

    alpha = 0.3

    # this repeats the code of color_spy()
    sep = [0]
    for row in dofs:
        if len(row) == 0:
            sep.append(sep[-1])
        else:
            sep.append(row[-1] + 1)
    sep = sorted(sep)

    ax = plt.gca()
    row_label_pos = []
    for i in range(len(row_names)):
        ystart, yend = sep[i : i + 2]
        row_label_pos.append(ystart + (yend - ystart) / 2)
        plt.axvspan(ystart - 0.5, yend - 0.5, alpha=alpha, facecolor=f"C{i}")
    ax.xaxis.set_ticks(row_label_pos)
    ax.set_xticklabels(group_names, rotation=45)
    if log:
        vec = abs(vec)
        plt.yscale("log")

    plt.plot(vec)
