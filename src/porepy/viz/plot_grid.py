"""
Module for PorePy's plotting functionality for (mixed-dimensional grids) built on matplotlib.

The functionality provided covers plotting of grids in 0 to 3 dimensions. Data may be
represented by cell-wise colors or cell- or face-wise vector arrows.
The module is quite useful for simple visualization purposes. For more advanced
visualization, especially in 3d, we recommend exporting the information to vtu using the
exporter module found in this folder.
"""
from __future__ import annotations

import string
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import porepy as pp


def plot_grid(
    grid: Union[pp.Grid, pp.MixedDimensionalGrid],
    cell_value: Optional[Union[np.ndarray, str]] = None,
    vector_value: Optional[Union[np.ndarray, str]] = None,
    info: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot the (possibly mixed-dimensional) grid with data in a 3d framework.

    It is possible to add the cell ids at the cells centers (info option 'c'),
    the face ids at the face centers (info option 'f'), the node ids at the node
    (info option 'n'), and the normal at the face (info option 'o'). If info is
    set to 'all' all the information are displayed.

    Args:
        grid (pp.Grid or pp.MixedDimensionalGrid): subdomain or mixed-dimensional grid.
        cell_value (str or array, optional) if g is a single grid then cell scalar field to be
            represented (only 1d and 2d). If g is a mixed-dimensional grid the name (key)
            of the scalar field.
        vector_value: (optional) if g is a single grid then vector scalar field to be
            represented (only 1d and 2d). If g is a mixed-dimensional grid the name (key)
            of the vector field.
        info: (optional) add extra information to the plot. C, F, and N add cell, face and
            node numbers, respectively. O gives a plot of the face normals. See the funtion
            add_info.
        kwargs (optional): Keyword arguments:
            alpha: transparency of cells (2d) and faces (3d)
            cells: boolean array with length number of cells. Only plot cells c
                where cells[c]=True. Not valid for a MixedDimensionalGrid.

    Example:
    # if grid is a single grid:
    cell_id = np.arange(grid.num_cells)
    plot_grid(grid, cell_value=cell_id, info="ncfo", alpha=0.75)

    # if grid is a mixed-dimensional grid
    plot_grid(grid, cell_value="cell_id", info="ncfo", alpha=0.75)

    """

    # Grid is a subdomain
    if isinstance(grid, pp.Grid):
        assert cell_value is None or isinstance(cell_value, np.ndarray)
        assert vector_value is None or isinstance(vector_value, np.ndarray)
        plot_sd(grid, cell_value, vector_value, info, **kwargs)

    # Grid is a mixed-dimensional grid
    if isinstance(grid, pp.MixedDimensionalGrid):
        assert cell_value is None or isinstance(cell_value, str)
        assert vector_value is None or isinstance(vector_value, str)
        plot_mdg(grid, cell_value, vector_value, info, **kwargs)


def save_img(
    name: str,
    grid: Union[pp.Grid, pp.MixedDimensionalGrid],
    cell_value: Optional[Union[np.ndarray, str]] = None,
    vector_value: Optional[Union[np.ndarray, str]] = None,
    info: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot and save the grid in a 3d framework.

    It is possible to add the cell ids at the cells centers (info option 'c'),
    the face ids at the face centers (info option 'f'), the node ids at the node
    (info option 'n'), and the normal at the face (info option 'o'). If info is
    set to 'all' all the informations are displayed.

    Args:
        name: the name of the file
        grid (pp.Grid or pp.MixedDimensionalGrid): subdomain or mixed-dimensional grid.
        cell_value (array or string, optional): if g is a single grid then cell scalar
            field to be represented (only 1d and 2d). If g is a mixed-dimensional grid
            the name (key) of the scalar field.
        vector_value (array or string, optional): if grid is a single grid then
            vector scalar field to be represented (only 1d and 2d). If grid is a
            mixed-dimensional grid the name (key) of the vector field.
        kwargs (optional): Keyword arguments:
            info: add extra information to the plot, see plot_grid.
            alpha: transparency of cells (2d) and faces (3d)

    Example:
    #  if grid is a single grid:
    cell_id = np.arange(grid.num_cells)
    save_img(grid, cell_value=cell_id, info="ncfo", alpha=0.75)

    #if grid is a mixed-dimensional grid
    save_img(grid, cell_value="cell_id", info="ncfo", alpha=0.75)

    """
    plot_grid(grid, cell_value, vector_value, info, **dict(kwargs, if_plot=False))
    plt.savefig(name, bbox_inches="tight", pad_inches=0)


def plot_sd(
    sd: pp.Grid,
    cell_value: Optional[np.ndarray],
    vector_value: Optional[np.ndarray],
    info: Optional[str],
    **kwargs,
) -> None:
    """
    Plot data on a subdomain and provided data.

    Args:
        sd (pp.Grid): Subdomain
        cell_value (np.ndarray): cell-wise scalar values, will be represented by the color
        of the cells.
        vector_value (np.ndarray): vector values, one 3d vector for each cell or for each
            face (see the _quiver function).
        info (str, optional): Which geometry information to display, see add_info.
        kwargs (optional): Keyword arguments:
            fig_size: Size of figure.
            fig_num: The number of the figure.
            color_map: Limits of the cell value color axis.
            if_plot: Boolean flag determining whether the plot is shown or not.
            plot_2d: Boolean flag determining wheter the plit is showed in 2d or 3d.
            pointsize: Size of points marking 0d grids.
            linewidth: Width of faces in 2d and edges in 3d.
            rgb: Color map weights. Defaults to [1, 0, 0].
            alpha: Transparency of the plot.
            cells: boolean array with length number of cells. Only plot cells c
                   where cells[c]=True
    """
    # Initialize figure with correct size.
    fig_size = kwargs.get("fig_size", None)
    fig_num = kwargs.get("fig_num", None)
    fig = plt.figure(num=fig_num, figsize=fig_size)

    # Initialize the corresponding axis
    ax: mpl.axes.Axes = fig.add_subplot(111)
    if not kwargs.get("plot_2d", False):
        ax = fig.add_subplot(111, projection="3d")

    # Set title and axis labels
    ax.set_title(kwargs.get("title", " ".join(sd.name)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if not kwargs.get("plot_2d", False):
        ax.set_zlabel("z")  # type: ignore[attr-defined]

    # Determine the color map (based on min and max values if not provided externally)
    if cell_value is not None and sd.dim != 3:
        if kwargs.get("color_map"):
            extr_value = kwargs["color_map"]
        else:
            extr_value = np.array([np.amin(cell_value), np.amax(cell_value)])
        kwargs["color_map"] = _color_map(extr_value)

    # Plot the grid and data
    _plot_sd_xd(sd, cell_value, vector_value, ax, **kwargs)

    # Determine min and max values of all grid nodes
    x, y, z = _lim(sd.nodes)
    # And set these values as limit of the axis.
    # In 2d, restrict the data, in 3d (default), do not.
    if kwargs.get("plot_2d", False):
        if not np.isclose(x[0], x[1]):
            ax.set_xlim(x)
        if not np.isclose(y[0], y[1]):
            ax.set_ylim(y)
    else:
        if not np.isclose(x[0], x[1]):
            ax.set_xlim3d(x)  # type: ignore[attr-defined]
        if not np.isclose(y[0], y[1]):
            ax.set_ylim3d(y)  # type: ignore[attr-defined]
        if not kwargs.get("plot_2d", False):
            ax.set_zlim3d(z)  # type: ignore[attr-defined]

    # Add info if provided.
    if info is not None:
        _add_info(sd, info, ax, **kwargs)

    # Add color map if provided.
    if kwargs.get("color_map"):
        fig.colorbar(kwargs["color_map"], ax=ax)

    # Draw and potentially show the plot.
    plt.draw()
    if kwargs.get("if_plot", True):
        plt.show()


def plot_mdg(
    mdg: pp.MixedDimensionalGrid,
    cell_value: Optional[str],
    vector_value: Optional[str],
    info: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Plot a mixed-dimensional grid and selected data.

    Args:
        mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid
        cell_value (str): key to scalar cell values, will be represented by the color
        of the cells.
        vector_value (str): key to vector cell or face values.
        info (str, optional): Which geometry information to display, see add_info.
        kwargs (optional): Keyword arguments:
            fig_size: Size of figure.
            fig_num: The number of the figure.
            color_map: Limits of the cell value color axis.
            rgb: Color map weights. Defaults to [1, 0, 0].
            if_plot: Boolean flag determining whether the plot is shown or not.
            pointsize: Size of points marking 0d grids.
            linewidth: Width of faces in 2d and edges in 3d.
            alpha: Transparency of the plot.
            cells: boolean array with length number of cells. Only plot cells c
                   where cells[c]=True
    """
    # Initialize figure with correct size
    fig_size = kwargs.get("fig_size", None)
    fig_num = kwargs.get("fig_num", None)
    fig = plt.figure(num=fig_num, figsize=fig_size)

    # Initialize the corresponding axis
    ax: mpl.axes.Axes = fig.add_subplot(111)
    if not kwargs.get("plot_2d", False):
        ax = fig.add_subplot(111, projection="3d")

    # Add title and axis labels
    title = kwargs.get("title", " ".join(mdg.name))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if not kwargs.get("plot_2d", False):
        ax.set_zlabel("z")  # type: ignore[attr-defined]

    # Define color map (based on min and max value of the cell value if none externally
    # provided)
    if cell_value is not None and mdg.dim_max() != 3:
        if kwargs.get("color_map"):
            extr_value = kwargs["color_map"]
        else:
            extr_value = np.array([np.inf, -np.inf])
            for _, sd_data in mdg.subdomains(return_data=True):
                values = pp.get_solution_values(
                    name=cell_value, data=sd_data, time_step_index=0
                )
                extr_value[0] = min(
                    np.amin(values),
                    extr_value[0],
                )
                extr_value[1] = max(
                    np.amax(values),
                    extr_value[1],
                )
        kwargs["color_map"] = _color_map(extr_value)

    # Plot each subdomain separately
    for index, (sd, sd_data) in enumerate(mdg.subdomains(return_data=True)):
        # Adjust rgb colors depending on the subdomain ordering
        kwargs["rgb"] = np.divide(kwargs.get("rgb", [1, 0, 0]), index + 1)
        # Plot the subdomain and data

        vector_value_array = (
            sd_data.get(pp.TIME_STEP_SOLUTIONS, {}).get(vector_value, {}).get(0, None)
        )
        if vector_value_array is not None:
            # The further algorithm requires the vector_value array of shape (3 x n).
            # Now, we have a 1D array.
            # Thus, we first reshape and then fill the remaining dimensions with zeros.
            vector_value_array = vector_value_array.reshape(
                (mdg.dim_max(), -1), order="F"
            )
            vector_value_array = np.vstack(
                [
                    vector_value_array,
                    np.zeros(
                        (3 - vector_value_array.shape[0], vector_value_array.shape[1])
                    ),
                ]
            )
        _plot_sd_xd(
            sd,
            sd_data.get(pp.TIME_STEP_SOLUTIONS, {}).get(cell_value, {}).get(0, None),
            vector_value_array,
            ax,
            **kwargs,
        )

    # Determine limits of axis based on the min and max values of grid coordinates
    # of all 1d, 2d, 3d subdomains
    val = np.array([_lim(sd.nodes) for sd in mdg.subdomains() if sd.dim > 0])

    x: tuple[float, float] = (np.amin(val[:, 0, :]), np.amax(val[:, 0, :]))
    y: tuple[float, float] = (np.amin(val[:, 1, :]), np.amax(val[:, 1, :]))
    z: tuple[float, float] = (np.amin(val[:, 2, :]), np.amax(val[:, 2, :]))

    # In 2d, restrict the data, in 3d (default), do not.
    if kwargs.get("plot_2d", False):
        if not np.isclose(x[0], x[1]):
            ax.set_xlim(x)
        if not np.isclose(y[0], y[1]):
            ax.set_ylim(y)
    else:
        if not np.isclose(x[0], x[1]):
            ax.set_xlim3d(x)  # type: ignore[attr-defined]
        if not np.isclose(y[0], y[1]):
            ax.set_ylim3d(y)  # type: ignore[attr-defined]
        if not kwargs.get("plot_2d", False):
            ax.set_zlim3d(z)  # type: ignore[attr-defined]

    # Add info if provided
    if info is not None:
        for sd in mdg.subdomains():
            _add_info(sd, info, ax, **kwargs)

    # Add color map if provided
    if kwargs.get("color_map"):
        fig.colorbar(kwargs["color_map"], ax=ax)

    # Draw and potentially show the plot.
    plt.draw()
    if kwargs.get("if_plot", True):
        plt.show()


# --------- Auxiliary functions


class _Arrow3D(FancyArrowPatch):
    """
    Arrow representation intended for visualization of vector quantities.
    """

    def __init__(
        self,
        xs: list[np.ndarray],
        ys: list[np.ndarray],
        zs: list[np.ndarray],
        *args,
        **kwargs,
    ):
        """
        Each arrow comes with head and tail points, denoting their start and end points.
        Provide the coordinates of both vertices as xs, ys and zs, each a list of the head
        and tail values.

        Args:
            xs (list of np.ndarray): two-element list of x-coordinates of head and tail points
            ys (list of np.ndarray): two-element list of y-coordinates of head and tail points
            zs (list of np.ndarray): two-element list of z-coordinates of head and tail points
            # FIXME be more precise
            args (optional): Arguments
            kwargs (optional): Keyword arguments for FancyArrowPatch

        """
        # Setup an empty arrow using the base class
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        # Store the coordinates
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """Auxiliary method, needed when using matplotlib>=3.5.
        See: https://github.com/matplotlib/matplotlib/issues/21688
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

    # FIXME: typing
    def draw(self, renderer) -> None:
        """Draw arrows using the given renderer.

        Args:
            renderer: Renderer
        """
        # Render the 3d coordinates of the arrows as preparation for plotting in the 2d plane.
        xs3d, ys3d, zs3d = self._verts3d
        axes_M = self.axes.M  # type: ignore[union-attr]
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, axes_M)
        # Extract the rendered positions in the 2d plotting plane
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        # Draw the arrows
        FancyArrowPatch.draw(self, renderer)


def _quiver(sd: pp.Grid, vector_value: np.ndarray, ax: mpl.axes.Axes, **kwargs) -> None:
    """
    Draws arrows representing vectors.

    Args:
        sd (pp.Grid): subdomain.
        vector_value (np.ndarray): 3 x n, where n equals either the number of faces or number
            of cells of the grids. Defines the starting point of the plotted vectors.
        ax (matplotlib axis): The axis to which the arrows will be added.
        kwargs (optional): Keyword arguments:
            vector_scale: Scale factor to adjust length of the vectors.
            linewidth: Width of the plotted vectors.

    Raises:
        ValueError if the shape of vector_value does not conform with the number of faces
        and cells.
    """
    # Define the origin of arrows as face or cell centers. For this,
    # use the shape of the vector values to determine whether the data is
    # implicitly assigned to faces or cells.
    if vector_value.shape[1] == sd.num_faces:
        where = sd.face_centers
    elif vector_value.shape[1] == sd.num_cells:
        where = sd.cell_centers
    else:
        raise ValueError

    # Allow for adjusting the length of data arrows.
    scale = kwargs.get("vector_scale", 1)

    # Determine the line width of each arrow
    linewidth = kwargs.get("linewidth", 1)

    # Define and draw all arrows.
    for v in np.arange(vector_value.shape[1]):
        # Define head and tail points, using the face or cell centers
        # and their prolongation by the data, incl. potential scaling.
        x = [where[0, v], where[0, v] + scale * vector_value[0, v]]
        y = [where[1, v], where[1, v] + scale * vector_value[1, v]]
        z = [where[2, v], where[2, v] + scale * vector_value[2, v]]

        # Define the 3d arrow
        a = _Arrow3D(
            x,
            y,
            z,
            mutation_scale=5,
            linewidth=linewidth,
            arrowstyle="-|>",
            color="k",
            zorder=np.inf,
        )

        # Add arrow to axis
        ax.add_artist(a)


def _plot_sd_xd(
    sd: pp.Grid,
    cell_value: Optional[np.ndarray],
    vector_value: Optional[np.ndarray],
    ax: mpl.axes.Axes,
    **kwargs,
) -> None:
    """
    Wrapper function to plot subdomain of arbitrary dimension. In 1d and 2d, the cell_value
    is represented by cell color. vector_value is shown as as arrows.

    Args:
        sd (pp.Grid): subdomain
        cell_value (np.ndarray): scalar cell data
        vector_value (np.ndarray, optional): vector cell or face data
        ax (matplotlib axis): axis
        kwargs (optional): Keyword arguments, see in the respective routines.
    """
    # Plot scalar data
    if sd.dim == 0:
        _plot_sd_0d(sd, ax, **kwargs)
    elif sd.dim == 1:
        _plot_sd_1d(sd, cell_value, ax, **kwargs)
    elif sd.dim == 2:
        _plot_sd_2d(sd, cell_value, ax, **kwargs)
    else:
        _plot_sd_3d(sd, ax, **kwargs)

    # Add vector data
    if vector_value is not None:
        _quiver(sd, vector_value, ax, **kwargs)


def _lim(
    nodes: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Extracts the x, y and z limits of a node array.

    Args:
        nodes (np.ndarray): 3d node array
    """
    # Determine min and max values for each compononent of all coordinates
    x: tuple[float, float] = (np.amin(nodes[0, :]), np.amax(nodes[0, :]))
    y: tuple[float, float] = (np.amin(nodes[1, :]), np.amax(nodes[1, :]))
    z: tuple[float, float] = (np.amin(nodes[2, :]), np.amax(nodes[2, :]))

    return x, y, z


def _color_map(
    extr_value: np.ndarray, cmap_type: str = "jet"
) -> "mpl.cm.ScalarMappable":
    """
    Constructs a color map and sets the extremal values of the value range.

    Args:
        extr_value (np.ndarray): two-element iterable object containing min and
            max values used as limits of the color map
        cmap_type (str): keyword
    """
    cmap = plt.get_cmap(cmap_type)
    scalar_map = mpl.cm.ScalarMappable(cmap=cmap)
    scalar_map.set_array(extr_value)
    scalar_map.set_clim(vmin=extr_value[0], vmax=extr_value[1])
    return scalar_map


def _add_info(sd: pp.Grid, info: str, ax: mpl.axes.Axes, **kwargs) -> None:
    """
    Adds information on numbering of geometry information of the grid g to ax.

    For each of the flags "C", "N" and "F" that are present in info, the cell, node and
    face numbers will be displayed at the corresponding cell centers, nodes and face
    centers, respectively. If "O" is present, the face normals are printed. If info is "ALL",
    all options are considered.

    Args:
        sd (pp.Grid): subdomain
        info (str): extra information to the plot. C, F, and N add cell, face and
            node numbers, respectively. O gives a plot of the face normals.
        ax (matplotlib axis): axis
        kwargs (optional): Keyword arguments used in _quiver.
    """

    def _disp(i: int, p: np.ndarray, c, m):
        """Add single scatter plot to ax.

        Args:
            i (int): integer identifier to be added at given position
            p (np.ndarray): position
            c: color
            m: marker
        """
        ax.scatter(*p, c=c, marker=m)  # type: ignore[misc]
        ax.text(*p, i)  # type: ignore[call-arg]

    def _disp_loop(v: np.ndarray, c, m):
        """Loop over disp.

        Args:
            v: positions
            c: color
            m: marker
        """
        [_disp(i, ic, c, m) for i, ic in enumerate(v.T)]

    # Convert info to upper case
    info = info.upper()
    # Consider all options if info is equal to 'all' (modulo case sensitivity)
    info = string.ascii_uppercase if info == "ALL" else info

    dim = 2 if kwargs.get("plot_2d", False) else 3

    # Display cell centers if "C" in info
    if "C" in info:
        _disp_loop(sd.cell_centers[:dim, :], "r", "o")
    # Display nodes if "N" in info
    if "N" in info:
        _disp_loop(sd.nodes[:dim, :], "b", "s")
    # Display face centers if "F" in info
    if "F" in info:
        _disp_loop(sd.face_centers[:dim, :], "y", "d")
    # Display face normals if "O" in info
    if "O" in info.upper() and sd.dim != 0:
        # Plot face normals. Scaling set to reduce interference with other information
        # and other face normals
        _quiver(sd, sd.face_normals * 0.4, ax, **kwargs)


def _plot_sd_0d(sd: pp.Grid, ax: mpl.axes.Axes, **kwargs) -> None:
    """
    Plot the 0d grid g as a circle on the axis ax.

    Args:
        sd (pp.Grid): 0d subdomain
        ax (matplotlib axes): axes
        kwargs (optional): Keyword arguments
            pointsize (float): defining the size of the marker
    """
    dim = 2 if kwargs.get("plot_2d", False) else 3
    ax.scatter(
        *sd.nodes[:dim, :], color="k", marker="o", s=kwargs.get("pointsize", 1)
    )  # type: ignore[misc]


def _plot_sd_1d(
    sd: pp.Grid, cell_value: Optional[np.ndarray], ax: mpl.axes.Axes, **kwargs
) -> None:
    """
    Plot the 1d grid g to the axis ax, with cell_value represented by the cell coloring.

    Args:
        sd (pp.Grid): 1d subdomain
        cell_value (np.ndarray): cell values
        ax (matplotlib axes): axes
        kwargs (optional): Keyword arguments
            color_map: Limits of the cell value color axis.
            alpha: Transparency of the plot.
            cells: boolean array with length number of cells. Only plot cells c
                   where cells[c]=True
    """
    # Fetch nodes and cells
    cell_nodes = sd.cell_nodes()
    nodes, cells, _ = sps.find(cell_nodes)

    # Define the coloring of cells. If a color map is provided, use that.
    # Otherwise, use a fixed color.
    if kwargs.get("color_map") and cell_value is not None:
        scalar_map = kwargs["color_map"]
        alpha = kwargs.get("alpha", 1)

        def _color_edge(value: float) -> str:
            return scalar_map.to_rgba(value, alpha)

    else:
        # Force cell_value to be an array (if None previously), and 0.
        cell_value = np.zeros(sd.num_cells)

        def _color_edge(value: float) -> str:
            return "k"

    # Make mypy happy
    assert isinstance(cell_value, np.ndarray)

    # Fetch mask defining which cells to plot
    cells = kwargs.get("cells", np.ones(sd.num_cells, dtype=bool))

    # Plot cells with coloring determined on the cell values
    for c in np.arange(sd.num_cells):
        # Apply mask
        if not cells[c]:
            continue
        # Determine the nodes of the cell
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
        ptsId = nodes[loc]
        pts = sd.nodes[:, ptsId]
        linewidth = kwargs.get("linewidth", 1)
        if kwargs.get("plot_2d", False):
            poly = PolyCollection([pts[:2, :].T], linewidth=linewidth)
            poly.set_edgecolor(_color_edge(cell_value[c]))
            ax.add_collection(poly)
        else:
            poly = Poly3DCollection([pts.T], linewidth=linewidth)
            poly.set_edgecolor(_color_edge(cell_value[c]))
            ax.add_collection3d(poly)  # type: ignore[attr-defined]


def _plot_sd_2d(
    sd: pp.Grid, cell_value: Optional[np.ndarray], ax: mpl.axes.Axes, **kwargs
):
    """
    Plot the 2d grid g to the axis ax, with cell_value represented by the cell coloring.

    Args:
        sd (pp.Grid): 2d subdomain
        cell_value (np.ndarray): cell values
        ax (matplotlib axes): axes
        kwargs (optional): Keyword arguments:
            color_map: Limits of the cell value color axis.
            linewidth: Width of faces in 2d and edges in 3d.
            rgb: Color map weights. Defaults to [1, 0, 0].
            alpha: Transparency of the plot.
            cells: boolean array with length number of cells. Only plot cells c
                   where cells[c]=True
    """
    faces, _, _ = sps.find(sd.cell_faces)
    nodes, _, _ = sps.find(sd.face_nodes)

    # Define the coloring of cells. If a color map is provided, use that.
    # Otherwise, use a fixed color.
    alpha = kwargs.get("alpha", 1)
    if kwargs.get("color_map") and cell_value is not None:
        scalar_map = kwargs["color_map"]

        def _color_face(value):
            return scalar_map.to_rgba(value, alpha)

    else:
        # Force cell_value to be an array (if None previously), and 0.
        cell_value = np.zeros(sd.num_cells)
        rgb = kwargs.get("rgb", [1, 0, 0])

        def _color_face(value):
            return np.r_[rgb, alpha]

    # Make mypy happy
    assert isinstance(cell_value, np.ndarray)

    # Fetch mask defining which cells to plot
    cells = kwargs.get("cells", np.ones(sd.num_cells, dtype=bool))

    # Plot cells with coloring determined on the cell values
    for c in np.arange(sd.num_cells):
        # Apply mask
        if not cells[c]:
            continue
        # Determine the faces of the cell
        loc_f = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
        faces_loc = faces[loc_f]
        # Determine the nodes of the fetched faces
        loc_n = sd.face_nodes.indptr[faces_loc]
        # Assign edges of the cell and sort them such that they form a circular chain
        pts_pairs = np.array([nodes[loc_n], nodes[loc_n + 1]])
        sorted_nodes, _ = pp.utils.sort_points.sort_point_pairs(pts_pairs)
        ordering = sorted_nodes[0, :]
        pts = sd.nodes[:, ordering]

        # Distinguish between 2d and 3d (relevant if the ambient dimension is 3).
        # In both cases, draw cells as polygons, fix the edge color, and color the
        # cell (a face for a 3d polygon)
        linewidth = kwargs.get("linewidth", 1)
        if kwargs.get("plot_2d", False):
            poly = PolyCollection([pts[:2].T], linewidth=linewidth)
            poly.set_edgecolor("k")
            poly.set_facecolor(_color_face(cell_value[c]))
            ax.add_collection(poly)
        else:
            poly = Poly3DCollection([pts.T], linewidth=linewidth)
            poly.set_edgecolor("k")
            poly.set_facecolors(_color_face(cell_value[c]))  # type: ignore[attr-defined]
            ax.add_collection3d(poly)  # type: ignore[attr-defined]

    # Define viewing angle in 3d
    if not kwargs.get("plot_2d", False):
        ax.view_init(90, -90)  # type: ignore[attr-defined]


def _plot_sd_3d(sd: pp.Grid, ax: mpl.axes.Axes, **kwargs) -> None:
    """
    Plot the 3d subdomain to the axis ax.

    Args:
        sd (pp.Grid): 3d subdomain
        ax (matplotlib axes): axes
        kwargs (optional): Keyword arguments:
    """
    faces_cells, cells, _ = sps.find(sd.cell_faces)
    nodes_faces, faces, _ = sps.find(sd.face_nodes)

    # Use trivial cell values (not relevant here)
    cell_value = np.zeros(sd.num_cells)

    # Define plotting options (colouring, transparency, line width)
    rgb = kwargs.get("rgb", [1, 0, 0])
    alpha = kwargs.get("alpha", 1)
    linewidth = kwargs.get("linewidth", 1)

    # Define the colouring of faces
    def _color_face(value: float) -> list:
        return np.r_[rgb, alpha]

    # Fetch mask defining which cells to plot
    cells = kwargs.get("cells", np.ones(sd.num_cells, dtype=bool))

    # Plot cells
    for c in np.arange(sd.num_cells):
        # Apply mask
        if not cells[c]:
            continue
        # Determine faces of cell
        loc_c = slice(sd.cell_faces.indptr[c], sd.cell_faces.indptr[c + 1])
        fs = faces_cells[loc_c]
        # Loop over all faces
        for f in fs:
            # Determine nodes formin a chain
            loc_f = slice(sd.face_nodes.indptr[f], sd.face_nodes.indptr[f + 1])
            ptsId = nodes_faces[loc_f]
            mask = pp.utils.sort_points.sort_point_plane(
                sd.nodes[:, ptsId], sd.face_centers[:, f], sd.face_normals[:, f]
            )
            pts = sd.nodes[:, ptsId[mask]]
            # Define and plot faces as polygons with fixed coloring
            poly = Poly3DCollection([pts.T], linewidth=linewidth)
            poly.set_edgecolor("k")
            poly.set_facecolors(_color_face(cell_value[c]))
            ax.add_collection3d(poly)  # type: ignore[attr-defined]
