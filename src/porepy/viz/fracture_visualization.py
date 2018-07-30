# Various utility functions for  gridding

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
        elif ntag < 6:
            col = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 0, 1)]
        else:
            raise NotImplementedError("Have not thought of more than six colors")

    plt.figure(**kwargs)
    plt.axis([d["xmin"], d["xmax"], d["ymin"], d["ymax"]])
    plt.plot(
        [d["xmin"], d["xmax"], d["xmax"], d["xmin"], d["xmin"]],
        [d["ymin"], d["ymin"], d["ymax"], d["ymax"], d["ymin"]],
        "-",
        color="red",
    )

    # Simple for-loop to draw one fracture after another. Not fancy, but it
    # serves its purpose.
    for i in range(c.shape[1]):
        plt.plot(
            [p[0, c[0, i]], p[0, c[1, i]]],
            [p[1, c[0, i]], p[1, c[1, i]]],
            "o-",
            color=col[tagmap[i]],
        )
    # Finally set axis
    plt.show()
