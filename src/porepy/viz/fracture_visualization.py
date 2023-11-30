"""
Visualization tools for fracture networks and wells. Plots 1d fractures as lines in a
2d domain using pyplot. Also plots wells as points.
"""

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import porepy as pp


def plot_fractures(
    pts: np.ndarray,
    edges: np.ndarray,
    domain: Optional[pp.Domain] = None,
    colortag: Optional[np.ndarray] = None,
    ax: Optional[mpl.axes.Axes] = None,
    **kwargs
) -> mpl.axes.Axes:
    """
    Plot 2d fractures as lines in a domain.

    The function is primarily intended for data exploration.

    Args:
        pts (np.ndarray, dims 2 x npt): Coordinates of the fracture endpoints.
        edges (np.ndarray, dims 2 x n_edges): Indices of fracture start and
            endpoints.
        domain (pp.Domain): Description of the domain. If not given, the domain is
            inferred from the bounding box associated with the fracture set.
        colortag (np.ndarray, dim n_edges, optional): Colorcoding for fractures
            (e.g. by fracture family). If provided, different colors will be
            asign to the different families. Defaults to all fractures being
            black.
        ax (matplotlib.axes.Axes, optional): If not given an axis, an axis will be created.
        kwargs (optional): Keyword arguments passed on to matplotlib.
            fig_id: figure id
            dpi:
            plot: Boolean flag determining whether the plot is drawn to canvas
            domain: Boolean flag determining whether the domain shall be plotted in red
                (or white)
            line_style: style of drawing lines
            pts_coord: Boolean flag determining whether the point coordinates are plotted
            axis_equal: Boolean flag determining whether both axes are treated equally
            axis: Boolean flag determining whether the axes are conforming with domain
            save: Boolean flag determining whether the figure is saved

    Returns:
        matplotlib.axes.Axes: The axis the fractures are plotted in

    Raises:
        ValueError if axis and figure id are provided both
        ValueError if keywords dpi and ax are provided both

    """
    # If not provided, determine the domain as bounding box
    if domain is None:
        domain = pp.Domain(pp.domain.bounding_box_of_point_cloud(pts))

    # If no axis is provided, construct one
    if ax is None:
        plt.figure(kwargs.get("fig_id", 1), dpi=kwargs.get("dpi", 100))
        ax = plt.axes()
        do_plot = kwargs.get("plot", True)  # To obtain legacy behaviour
    else:
        # Not sure if this should throw an error or just ignore the arguments:
        if kwargs.get("fig_id", None) is not None:
            raise ValueError("Cannot give both keyword argument 'fig_id' and 'ax'")
        elif kwargs.get("dpi", None) is not None:
            raise ValueError("Cannot give both keyword argument 'dpi' and 'ax'")
        do_plot = kwargs.get("plot", False)

    # Assign a color to each tag. We define these by RBG-values (simplest
    # option in pyplot).
    # For the moment, some RBG values are hard coded, do something more
    # intelligent if necessary.
    if colortag is None:
        tagmap = np.zeros(edges.shape[1], dtype="int")
        col = [(0.0, 0.0, 0.0)]
    else:
        utag, tagmap = np.unique(colortag, return_inverse=True)
        ntag = utag.size
        if ntag <= 3:
            col = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        elif ntag <= 6:
            col = [
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
            ]
        elif ntag <= 12:
            # https://www.rapidtables.com/web/color/RGB_Color.html
            col = [
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 1.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
                (0.5, 0.0, 0.0),
                (0.5, 0.5, 0.0),
                (0.0, 0.5, 0.0),
                (0.5, 0.0, 0.5),
                (0.0, 0.5, 0.5),
                (0.0, 0.0, 0.5),
            ]
        else:
            col = plt.get_cmap("tab20")(np.mod(utag, 20))

    # Fix the domain color
    if kwargs.get("domain", True):
        domain_color = "red"
    else:
        domain_color = "white"

    # Plot the domain
    ax.plot(
        [
            domain.bounding_box["xmin"],
            domain.bounding_box["xmax"],
            domain.bounding_box["xmax"],
            domain.bounding_box["xmin"],
            domain.bounding_box["xmin"],
        ],
        [
            domain.bounding_box["ymin"],
            domain.bounding_box["ymin"],
            domain.bounding_box["ymax"],
            domain.bounding_box["ymax"],
            domain.bounding_box["ymin"],
        ],
        "-",
        color=domain_color,
    )

    # Simple for-loop to draw one fracture after another. Not fancy, but it
    # serves its purpose.
    line_style = kwargs.get("line_style", "o-")
    for i in range(edges.shape[1]):
        ax.plot(
            [pts[0, edges[0, i]], pts[0, edges[1, i]]],
            [pts[1, edges[0, i]], pts[1, edges[1, i]]],
            line_style,
            color=col[tagmap[i]],
        )

    # Add point coordinates to the plot
    if kwargs.get("pts_coord", False):
        for i in range(pts.shape[1]):
            ax.text(
                pts[0, i], pts[1, i], "(" + str(pts[0, i]) + ", " + str(pts[1, i]) + ")"
            )

    # Set options for axes
    if kwargs.get("axis_equal", True):
        ax.axis("equal")
        ax.set_aspect("equal", adjustable="box")

    if kwargs.get("axis", "on") == "on":
        box_data: tuple[float, float, float, float] = (
            domain.bounding_box["xmin"],
            domain.bounding_box["xmax"],
            domain.bounding_box["ymin"],
            domain.bounding_box["ymax"],
        )
        ax.axis(box_data)

    else:
        ax.axis("off")

    # Finally set axis
    if do_plot:
        plt.show()
    if kwargs.get("save", None) is not None:
        plt.savefig(kwargs.get("save"), bbox_inches="tight", pad_inches=0.0)
        plt.close()

    return ax


def plot_wells(
    d: pp.Domain, w: np.ndarray, colortag: Optional[np.ndarray] = None, **kwargs
):
    """Plot 2d wells as points in a domain.

    The function is primarily intended for data exploration.

    Args:
        d (pp.Domain): Two-dimensional domain specification.
        w (np.ndarray, dims 2 x npt): Coordinates of the wells.
        colortag (np.ndarray, dim n_w, optional): Colorcoding for wells.
            If provided, different colors will be asign to the different wells.
            Defaults to all wells being black.
        kwargs: Keyword arguments passed on to matplotlib.
            plot: Boolean flag determining wether the figure is plotted

    Raises:
        NotImplementedError if more than 6 colors are requested

    """

    # Assign a color to each tag. We define these by RBG-values (simplest
    # option in pyplot).
    # For the moment, some RBG values are hard coded, do something more
    # intelligent if necessary.
    if colortag is None:
        tagmap = np.zeros(w.shape[1], dtype="int")
        col = [(0, 0, 0)]
    else:
        utag, tagmap = np.unique(colortag, return_inverse=True)
        ntag = utag.size
        if ntag <= 3:
            col = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        elif ntag < 6:
            col = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 0, 1)]
        else:
            raise NotImplementedError("Have not thought of more than six colors")

    # Setup figure and plot the domain
    plt.figure(kwargs.get("fig_id", 1))
    box_data: tuple[float, float, float, float] = (
        d.bounding_box["xmin"],
        d.bounding_box["xmax"],
        d.bounding_box["ymin"],
        d.bounding_box["ymax"],
    )
    plt.axis(box_data)
    plt.plot(
        [
            d.bounding_box["xmin"],
            d.bounding_box["xmax"],
            d.bounding_box["xmax"],
            d.bounding_box["xmin"],
            d.bounding_box["xmin"],
        ],
        [
            d.bounding_box["ymin"],
            d.bounding_box["ymin"],
            d.bounding_box["ymax"],
            d.bounding_box["ymax"],
            d.bounding_box["ymin"],
        ],
        "-",
        color="red",
    )

    # Simple for-loop to draw one well after another. Not fancy, but it
    # serves its purpose.
    for i, well in enumerate(w.T):
        plt.plot(*well, "o", color=col[tagmap[i]])

    # Finally draw the plot
    if kwargs.get("plot", True):
        plt.show()
