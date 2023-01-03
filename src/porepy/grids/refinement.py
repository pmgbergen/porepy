"""This module contains various methods to refine a grid.

It furthermore contains classes to define sequences of refined grids, and a factory
class to generate sets of refined grids.

"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional, Union

import gmsh
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids.grid import Grid
from porepy.grids.simplex import TriangleGrid
from porepy.grids.structured import TensorGrid


def distort_grid_1d(
    g: pp.Grid, ratio: float = 0.1, fixed_nodes: Optional[np.ndarray] = None
) -> pp.Grid:
    """Randomly distort internal nodes in a 1d grid.

    The boundary nodes are left untouched, and the perturbations will not perturb the
    topology of the mesh.

    Parameters:
         g: The grid that will be perturbed. Modifications will happen in place.
         ratio: ``default=0.1``

            Perturbation ratio. A node can be moved at most half the
            distance in towards any of its neighboring nodes. The ratio will
            multiply the chosen distortion. Should be less than 1 to preserve grid
            topology.
         fixed_nodes: ``default=None``

            Index of nodes to keep fixed under distortion. Boundary
            nodes will always be fixed, even if not explicitly included as
            fixed_node.

    Returns:
         The grid, but with distorted nodes.

    """
    if fixed_nodes is None:
        fixed_nodes = np.array([0, g.num_nodes - 1], dtype=int)
    else:
        # Ensure that boundary nodes are also fixed
        fixed_nodes = np.hstack((fixed_nodes, np.array([0, g.num_nodes - 1])))
        fixed_nodes = np.unique(fixed_nodes).astype(int)

    g.compute_geometry()
    r = ratio * (0.5 - np.random.random(g.num_nodes - 2))
    r *= np.minimum(g.cell_volumes[:-1], g.cell_volumes[1:])
    direction = (g.nodes[:, -1] - g.nodes[:, 0]).reshape((-1, 1))
    nrm = np.linalg.norm(direction)
    g.nodes[:, 1:-1] += r * direction / nrm
    g.compute_geometry()
    return g


def refine_grid_1d(g: pp.Grid, ratio: int = 2) -> pp.Grid:
    """Refine the cells in a 1D grid.

    Parameters:
        g: The 1D grid that is to be refined.
        ratio: ``default=2``

            Refinement level.

    Returns:
        A new, more refined, grid.

    """

    # Implementation note: The main part of the function is the construction of the new
    # cell-face relation. Since the grid is 1d, nodes and faces are equivalent, and
    # notation used mostly refers to nodes instead of faces.

    # Cell-node relation
    cell_nodes = g.cell_nodes()
    nodes, cells, _ = sps.find(cell_nodes)

    # Every cell will contribute (ratio - 1) new nodes
    num_new_nodes = (ratio - 1) * g.num_cells + g.num_nodes
    x = np.zeros((3, num_new_nodes))
    # Coordinates for splitting of cells
    theta = np.arange(1, ratio) / float(ratio)
    pos = 0
    shift = 0

    # Array that indicates whether an item in the cell-node relation represents a node
    # not listed before (e.g. whether this is the first or second occurrence of the
    # cell)
    if_add = np.r_[1, np.ediff1d(cell_nodes.indices)].astype(bool)

    indices = np.empty(0, dtype=int)
    # Template array of node indices for refined cells
    ind = np.vstack((np.arange(ratio), np.arange(ratio) + 1)).flatten("F")
    nd = np.r_[np.diff(cell_nodes.indices)[1::2], 0]

    # Loop over all old cells and refine them.
    for c in np.arange(g.num_cells):
        # Find start and end nodes of the old cell
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
        start, end = cell_nodes.indices[loc]

        # Flags for whether this is the first occurrences of the nodes of the old cell.
        # If so, they should be added to the new node array
        if_add_loc = if_add[loc]

        # Local cell-node (thus cell-face) relations of the new grid
        indices = np.r_[indices, shift + ind]

        # Add coordinate of the startpoint to the node array if relevant
        if if_add_loc[0]:
            x[:, pos : (pos + 1)] = g.nodes[:, start, np.newaxis]
            pos += 1

        # Add coordinates of the internal nodes
        x[:, pos : (pos + ratio - 1)] = g.nodes[:, start, np.newaxis] * theta + g.nodes[
            :, end, np.newaxis
        ] * (1 - theta)
        pos += ratio - 1
        shift += ratio + (2 - np.sum(if_add_loc) * (1 - nd[c])) - nd[c]

        # Add coordinate to the endpoint, if relevant
        if if_add_loc[1]:
            x[:, pos : (pos + 1)] = g.nodes[:, end, np.newaxis]
            pos += 1

    # For 1d grids, there is a 1-1 relation between faces and nodes
    face_nodes = sps.identity(x.shape[1], format="csc")
    cell_faces = sps.csc_matrix(
        (
            np.ones(indices.size, dtype=bool),
            indices,
            np.arange(0, indices.size + 1, 2),
        )
    )
    g = Grid(1, x, face_nodes, cell_faces, "Refined 1d grid")
    g.compute_geometry()

    return g


def refine_triangle_grid(g: pp.TriangleGrid) -> tuple[pp.TriangleGrid, np.ndarray]:
    """Uniform refinement of triangular grids.

    All cells are split into four subcells by combining existing nodes and face centers.

    Note:
        It should be fairly straightforward to extend the function to 3D simplex grids
        as well. The loop over face combinations extends straightforwardly, but
        obtaining the node in the corner defined by faces may be a bit tricky.

    Parameters:
        g: The triangle grid that is to be refined.

    Returns:
        A 2-tuple containing

        :class:`~porepy.grids.simplex.TriangleGrid`:
            New grid, with ``nd+2`` times as many cells as ``g``.

        :obj:`~numpy.ndarray`: ``shape=(g.num_cells*(nd+2),)``

            Mapping from new to old cells.

    """
    # g needs to have face centers
    if not hasattr(g, "face_centers"):
        g.compute_geometry()
    nd = g.dim

    # Construct dense versions of face-node and cell-face maps.
    # This will crash if a non-simplex grid is provided.
    fn = g.face_nodes.indices.reshape((nd, g.num_faces), order="F")
    cf = g.cell_faces.indices.reshape((nd + 1, g.num_cells), order="F")

    new_nodes = np.hstack((g.nodes, g.face_centers))
    offset = g.num_nodes

    # Combinations of faces per cell
    binom = ((0, 1), (1, 2), (2, 0))

    # Holder for new tesselation.
    new_tri = np.empty(shape=(nd + 1, g.num_cells, nd + 2), dtype=int)

    # Loop over combinations
    for ti, b in enumerate(binom):
        # Find face-nodes of these faces. Each column corresponds to a single cell.
        # There should be one duplicate in each column (since this is 2D)
        loc_n = np.vstack((fn[:, cf[b[0]]], fn[:, cf[b[1]]]))
        # Find the duplicate: First sort along column, then take diff along column and
        # look for zero.
        # Implementation note: To extend to 3D, also require that np.gradient is zero
        # (that function should be able to do this).
        loc_n.sort(axis=0)
        equal = np.argwhere(np.diff(loc_n, axis=0) == 0)
        # equal is now 2xnum_cells. To pick out the right elements, consider the raveled
        # index, and construct the corresponding raveled array
        equal_n = loc_n.ravel()[
            np.ravel_multi_index(equal.T, loc_n.shape)
        ]  # type: ignore

        # Define node combination. Both nodes associated with a face have their offset
        # adjusted.
        new_tri[:, :, ti] = np.vstack((equal_n, offset + cf[b[0]], offset + cf[b[1]]))

    # Create final triangle by combining faces only
    new_tri[:, :, -1] = offset + cf

    # Reshape into 2d array
    new_tri = new_tri.reshape((nd + 1, (nd + 2) * g.num_cells))

    # The new grid inherits the history of the old one.
    history = g.history.copy()
    history.append("Refinement")

    # Also create a mapping from refined cells to parent cells
    parent = np.tile(np.arange(g.num_cells), g.dim + 2)

    new_grid = TriangleGrid(new_nodes, tri=new_tri, name=g.name)
    new_grid.history = history
    return new_grid, parent


def remesh_1d(g_old: pp.Grid, num_nodes: int, tol: float = 1e-6) -> pp.Grid:
    """Create a new 1d mesh covering the same domain as an old one.

    The new grid is equi-spaced, and there is no guarantee that the nodes in the old and
    new grids are coinciding. Use with care, in particular for grids with internal
    boundaries.

    Parameters:
        g_old: 1d grid to be replaced/remeshed.
        num_nodes: Number of nodes in the new grid.
        tol (optional): Tolerance used to compare node coordinates (for mapping of
            boundary conditions). Defaults to ``1e-6``.

    Returns:
        The new grid that covers the same domain as g_old.

    """

    # Create equi-spaced nodes covering the same domain as the old grid
    theta = np.linspace(0, 1, num_nodes)
    start, end = g_old.get_all_boundary_nodes()
    # Not sure why the new axis was necessary.
    nodes = g_old.nodes[:, start, np.newaxis] * theta + g_old.nodes[
        :, end, np.newaxis
    ] * (1.0 - theta)

    # Create the new grid, and assign nodes.
    g = TensorGrid(nodes[0, :])
    g.nodes = nodes
    g.compute_geometry()

    # Map the tags from the old grid to the new one
    # Normally the tags are given at faces/point that are fixed the 1d mesh
    # We use this assumption to proceed.
    for f_old in np.arange(g_old.num_faces):
        # Detect in the new grid which face is geometrically the same (upon a tolerance)
        # as in the old grid
        dist = pp.distances.point_pointset(g_old.face_centers[:, f_old], g.face_centers)
        f_new = np.where(dist < tol)[0]

        # If you find a match transfer all the tags from the face in the old grid to the
        # face in the new grid
        if f_new.size:
            if f_new.size != 1:
                raise ValueError(
                    "There cannot be more than one face, something went wrong"
                )
            for tag in pp.utils.tags.standard_face_tags():
                g.tags[tag][f_new] = g_old.tags[tag][f_old]

    g.update_boundary_node_tag()

    return g


class GridSequenceIterator:
    """Iterator for generating successively refined grids by the use of a grid factory.

    See Also:
        :class:`GridSequenceFactory`

    Parameters:
        factory: Factory class for generating set of refined grids.

    """

    def __init__(self, factory: GridSequenceFactory) -> None:
        self._factory = factory
        """The factory instance passed at instantiation."""
        self._counter: int = 0
        """A counter for the iterator."""

    def __next__(self) -> pp.MixedDimensionalGrid:
        """Choose the next grid in grid iterator.

        See also:
            1. `Documentation of Python built in __next__ function
            <https://docs.python.org/3/library/functions.html#next>`_

        Returns:
            The next refined mixed dimensional grid.

        Raises:
            StopIteration: When the counter parameter is larger than or equal to the set
            number of refinements.

        """
        if self._counter >= self._factory._num_refinements:
            self._factory.close()
            raise StopIteration()

        else:
            mdg: pp.MixedDimensionalGrid = self._factory._generate(self._counter)
            self._counter += 1
            return mdg


class GridSequenceFactory(abc.ABC):
    """Factory class to generate a set of refined grids.

    To define new refinement types, inherit from this class and override the
    :meth:`_generate` method.

    The class can be used in (for now) two ways:

    Either by nested or unstructured refinement.
    This is set by setting the parameter mode to ``'nested'`` or ``'unstructured'``
    respectively.

    The number of refinement is set by the parameter ``'num_refinements'``.

    Mesh sizes are set by the standard ``FractureNetwork.mesh()`` arguments
    ``mesh_size_fracs`` and ``mesh_size_bound``. These are set by the parameter
    ``mesh_param``.

    If the mode is ``'nested'``, ``mesh_param`` should be a single
    dictionary; subsequent mesh refinements are then set by nested refinement.

    If the mode is ``'unstructured'``,
    mesh_params should be a list, where each list element is a
    dictionary with mesh size parameters to be passed to the ``FractureNetwork``.

    Further argument, such as fractures that are really constraints, can be given by
    ``params['grid_params']``.
    These are passed on the ``FractureNetwork.mesh()``, essentially as ``**kwargs``.

    .. rubric:: Acknowledgements

    The design idea and the majority of the code was contributed to by Haakon Ervik.

    Parameters:
        network: Define the domain that is to be discretized.
        params: Parameter dictionary. See above for more details on admissible keywords.

            Note that entries for every keyword are **required** and will raise an error
            if not available.

    """

    def __init__(
        self, network: Union[pp.FractureNetwork2d, pp.FractureNetwork3d], params: dict
    ) -> None:
        self._network = network.copy()
        self._counter: int = 0
        self._set_parameters(params)

        self.dim: int
        """Dimension of the fracture network."""

        if isinstance(network, pp.FractureNetwork2d):
            self.dim = 2
        elif isinstance(network, pp.FractureNetwork3d):
            self.dim = 3
        else:
            raise ValueError("Invalid type for fracture network")

        if self._refinement_mode == "nested":
            self._prepare_nested()

    def __iter__(self) -> GridSequenceIterator:
        """Grid sequence iterator object.

        See also:
            1. `Documentation of Python built in __iter__ function
               <https://docs.python.org/3/library/functions.html#iter>`_

        Returns:
            The grid sequence iterator.

        """
        return GridSequenceIterator(self)

    def close(self) -> None:
        """Method for finalizing ``gmsh``.

        ``gmsh`` will be informed that it should be finalized. Final method to be called
        when the iteration is going to be stopped.

        """
        if hasattr(self, "_gmsh"):
            self._gmsh.finalize()
            # Also inform the GmshWriter class that gmsh is no longer initialized.
            pp.fracs.gmsh_interface.GmshWriter.gmsh_initialized = False

    def _generate(self, counter: int) -> pp.MixedDimensionalGrid:
        """Which kind of grid refinement that is to be used is selected by this method.

        Based on the refinement mode, this method will decide if nested or unstructured
        refinement should be used.

        See also:
            1. :meth:`_prepare_nested` and :meth:`_generate_nested` for more about
                generation of nested refinement.
            2. :meth:`_generate_unstructured` for more about generation of
                unstructured refinement.

        Parameters:
            counter: Refinement counter.

        Returns:
            The refined mixed dimensional grid.

        """
        if self._refinement_mode == "nested":
            return self._generate_nested(counter)
        elif self._refinement_mode == "unstructured":
            return self._generate_unstructured(counter)

    def _prepare_nested(self) -> None:
        """Prepare nested refinement.

        Nested refinement is in this method prepared by initializing ``gmsh`` and
        generating the first grid.

        """
        # Operate on a deep copy of the network to avoid that fractures etc. are
        # unintentionally modified during operations.
        net = self._network.copy()

        file_name = "gmsh_convergence"
        # Generate data in a format suitable for defining a model in gmsh
        gmsh_data = net.prepare_for_gmsh(
            self._mesh_parameters,
            **self._grid_parameters,
        )
        # Initialize gmsh model with the relevant data. The data will be accessible in
        # gmsh.model.mesh (that is the way the Gmsh Python API works)
        writer = pp.fracs.gmsh_interface.GmshWriter(gmsh_data)
        # Generate the first grid
        # Do not finalize gmsh; this should be done by a call to self.close()
        writer.generate(file_name=file_name, finalize=False, clear_gmsh=False)
        self._out_file = file_name
        self._gmsh = gmsh

    def _generate_nested(self, counter: int) -> pp.MixedDimensionalGrid:
        """Generates nested refinement.

        A refined grid, based on the previous grid, is generated. Original cell faces
        will be kept, and the refinement happens locally to the cells instead of
        creating a grid from scratch.

        Parameters:
            counter: Refinement counter.

        Returns:
            The refined mixed dimensional grid.

        """
        out_file = Path(self._out_file)
        out_file = out_file.parent / out_file.stem
        out_file_name = f"{out_file}_{counter}.msh"

        # The first mesh is already done. Start refining all subsequent meshes.
        self._gmsh.model.mesh.refine()  # Refine the mesh

        self._gmsh.write(out_file_name)  # Write the result to '.msh' file

        mdg = pp.fracture_importer.dfm_from_gmsh(out_file_name, self.dim)
        pp.contact_conditions.set_projections(mdg)
        return mdg

    def _generate_unstructured(self, counter: int) -> pp.MixedDimensionalGrid:
        """Generate unstructured refinement.

        Refinement is done from "scratch" every time. Instead of refining already
        existing grid, this method will create a new grid that is more refined than the
        previous one.

        Parameters:
              counter: Refinement counter.

        Returns:
              The refined mixed dimensional grid.

        """
        net = self._network.copy()
        mesh_args = self._mesh_parameters[counter]
        grid_param = self._grid_parameters
        mdg = net.mesh(mesh_args, **grid_param)
        return mdg

    def _set_parameters(self, param: dict) -> None:
        """Method for setting parameters.

        Mode of refinement, number of refinement, mesh parameters and the grid
        parameters are saved as attributes.

        Parameters:
            param: Parameter dictionary.

        """
        self._refinement_mode = param["mode"]

        self._num_refinements = param["num_refinements"]

        self._mesh_parameters = param["mesh_param"]

        self._grid_parameters = param.get("grid_param", {})
