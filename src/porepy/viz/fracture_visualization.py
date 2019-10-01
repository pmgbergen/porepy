"""
Visualization tools for fracture networks. Plots 1d fractures as lines in a
2d domain using pyplot. Also plots wells as points.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_fractures(d, p, c, colortag=None, **kwargs):
    """
    Plot 2d fractures as lines in a domain.

    The function is primarily intended for data exploration.

    Parameters:
        d (dictionary): Domain size. Should contain fields xmin, xmax, ymin,
            ymax.
        p (np.ndarray, dims 2 x npt): Coordinates of the fracture endpoints.
        c (np.ndarray, dims 2 x n_edges): Indices of fracture start and
            endpoints.
        colortag (np.ndarray, dim n_edges, optional): Colorcoding for fractures
            (e.g. by fracture family). If provided, different colors will be
            asign to the different families. Defaults to all fractures being
            black.
        kwargs: Keyword arguments passed on to matplotlib.

    """

    # Assign a color to each tag. We define these by RBG-values (simplest
    # option in pyplot).
    # For the moment, some RBG values are hard coded, do something more
    # intelligent if necessary.
    if colortag is None:
        tagmap = np.zeros(c.shape[1], dtype="int")
        col = [(0, 0, 0)]
    else:
        utag, tagmap = np.unique(colortag, return_inverse=True)
        ntag = utag.size
        if ntag <= 3:
            col = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        elif ntag <= 6:
            col = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 0, 1)]
        elif ntag <= 12:
            # https://www.rapidtables.com/web/color/RGB_Color.html
            col = [
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, 0, 1),
                (0, 0, 1),
                (0.5, 0, 0),
                (0.5, 0.5, 0),
                (0, 0.5, 0),
                (0.5, 0, 0.5),
                (0, 0.5, 0.5),
                (0, 0, 0.5),
            ]
        else:
            raise NotImplementedError("Have not thought of more than twelwe colors")

    plt.figure(kwargs.get("fig_id", 1), dpi=kwargs.get("dpi", 100))

    if kwargs.get("domain", True):
        domain_color = "red"
    else:
        domain_color = "white"

    plt.plot(
        [d["xmin"], d["xmax"], d["xmax"], d["xmin"], d["xmin"]],
        [d["ymin"], d["ymin"], d["ymax"], d["ymax"], d["ymin"]],
        "-",
        color=domain_color,
    )

    # Simple for-loop to draw one fracture after another. Not fancy, but it
    # serves its purpose.
    line_style = kwargs.get("line_style", "o-")
    for i in range(c.shape[1]):
        plt.plot(
            [p[0, c[0, i]], p[0, c[1, i]]],
            [p[1, c[0, i]], p[1, c[1, i]]],
            line_style,
            color=col[tagmap[i]],
        )

    if kwargs.get("pts_coord", False):
        for i in range(p.shape[1]):
            plt.text(p[0, i], p[1, i], "(" + str(p[0, i]) + ", " + str(p[1, i]) + ")")

    if kwargs.get("axis_equal", True):
        plt.axis("equal")
        plt.gca().set_aspect("equal", adjustable="box")

    if kwargs.get("axis", "on") == "on":
        plt.axis([d["xmin"], d["xmax"], d["ymin"], d["ymax"]])
    else:
        plt.axis("off")

    # Finally set axis
    if kwargs.get("plot", True):
        plt.show()
    if kwargs.get("save", None) is not None:
        plt.savefig(kwargs.get("save"), bbox_inches="tight", pad_inches=0.0)
        plt.close()


def plot_wells(d, w, colortag=None, **kwargs):
    """
    Plot 2d wells as points in a domain.

    The function is primarily intended for data exploration.

    Parameters:
        d (dictionary): Domain size. Should contain fields xmin, xmax, ymin,
            ymax.
        w (np.ndarray, dims 2 x npt): Coordinates of the wells.
        colortag (np.ndarray, dim n_w, optional): Colorcoding for wells.
            If provided, different colors will be asign to the different wells.
            Defaults to all wells being black.
        kwargs: Keyword arguments passed on to matplotlib.

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

    plt.figure(kwargs.get("fig_id", 1))
    plt.axis([d["xmin"], d["xmax"], d["ymin"], d["ymax"]])
    plt.plot(
        [d["xmin"], d["xmax"], d["xmax"], d["xmin"], d["xmin"]],
        [d["ymin"], d["ymin"], d["ymax"], d["ymax"], d["ymin"]],
        "-",
        color="red",
    )

    # Simple for-loop to draw one well after another. Not fancy, but it
    # serves its purpose.
    for i, well in enumerate(w.T):
        plt.plot(*well, "o", color=col[tagmap[i]])

    # Finally set axis
    if kwargs.get("plot", True):
        plt.show()
