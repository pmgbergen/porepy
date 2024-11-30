"""This module contains various methods to refine a grid.

It furthermore contains classes to define sequences of refined grids, and a factory
class to generate sets of refined grids.

"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Literal, Optional

import gmsh
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.grids.grid import Grid
from porepy.grids.simplex import TriangleGrid
from porepy.grids.structured import TensorGrid
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data

logger = logging.getLogger(__name__)


def distort_grid_1d(
    g: pp.Grid, ratio: float = 0.1, fixed_nodes: Optional[np.ndarray] = None
) -> pp.Grid:
    """Randomly distort internal nodes in a 1d grid.

    The boundary nodes are left untouched, and the perturbations will not perturb the
    topology of the mesh.

    Parameters:
         g: The grid that will be perturbed. Modifications will happen in place.
         ratio: ``default=0.1``

            Perturbation ratio. A node can be moved at most half the distance in towards
            any of its neighboring nodes. The ratio will multiply the chosen distortion.
            Should be less than 1 to preserve grid topology.
         fixed_nodes: ``default=None``

            Index of nodes to keep fixed under distortion. Boundary nodes will always be
            fixed, even if not explicitly included as fixed_node.

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

    # Cell-node relation. Enforce csc format, since this is heavily used below. 
    cell_nodes = g.cell_nodes().tocsc()
    nodes, cells, _ = sparse_array_to_row_col_data(cell_nodes)

    # Every cell will contribute (ratio - 1) new nodes
    num_new_nodes = (ratio - 1) * g.num_cells + g.num_nodes
    x = np.zeros((3, num_new_nodes))
    # Coordinates for splitting of cells
    theta = np.arange(1, ratio) / float(ratio)
    pos = 0
    shift = 0

    offset_new_indices = 0

    # Array that indicates whether an item in the cell-node relation represents a node
    # not listed before (e.g. whether this is the first or second occurrence of the
    # cell). EK: This does not work if the cells are not linearly ordered.
    _, first_occ = np.unique(cell_nodes.indices, return_index=True)
    if_add = np.zeros(cell_nodes.indices.size, dtype=bool)
    if_add[first_occ] = True

    # Holder for the new indices.
    new_indices: list[np.ndarray] = []

    # Template array of node indices for refined cells.
    indices_template = np.vstack((np.arange(ratio), np.arange(ratio) + 1)).ravel("F")
    indices_template = np.repeat(np.arange(ratio), 2)
    nd = np.r_[np.diff(cell_nodes.indices)[1::2], 0]

    old_2_new_nodes = {}

    # The refinement must work also for grids that are not equidistant. While it might
    # be possible to do the refinemnet in a vectorized fashion, looping over the cells
    # and refining them one by one is a simpler solution.

    # TODO: To cover permuted indices of nodes and cells, we will need a more general
    # way of tracking nodes (in the coarse grid) that have been added to the new grid.
    # To this end, make a dictionary of old-to-new nodes and use this to define the
    # indices in the assignment to new_indices. The new nodes (refined ones) can still
    # be done as now, nodes also in the coarse grid have the current implementation if
    # if_add_loc is true for the relevant item, and fetch the index from the dictionary
    # otherwise. What to do with the mysterious shift variable is still unclear (note
    # that the current implementation is probably incorrect even for unpermuted
    # indices), it will take some thinking of how many nodes have really been added,
    # what happens to the start and end note etc.

    # Loop over all old cells and refine them.
    for c in np.arange(g.num_cells):
        # Find start and end nodes of the old cell. First the location in the sparse
        # storage.
        loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
        # Then the actual indices. This will be correct also if the nodes are not
        # ordered linearly (start+1 != end).
        start, end = cell_nodes.indices[loc]

        # Flags for whether this is the first occurrences of the nodes of the old cell.
        # If so, they should be added to the new node array
        if_add_loc = if_add[loc]

        # Local cell-node (thus cell-face) relations of the new grid. This will give a
        # linear ordering of the new nodes.
        #offset_new_indices = new_indices[-1][-1]
        
        loc_new_ind = []

        # Add coordinate of the startpoint to the node array if relevant
        if if_add_loc[0]:
            x[:, pos] = g.nodes[:, start]
            old_2_new_nodes[start] = pos
            loc_new_ind.append(pos)
            pos += 1
        else:
            loc_new_ind.append(old_2_new_nodes[start])

        loc_new_ind += list(pos + indices_template[:-2])
        
        # Add coordinates of the internal nodes. The coordinates added here run from
        # closest to the start node to closest to the end node. Reshape is needed since
        # we may add more than one node at a time (in contrast to the assignments in the
        # if-statements above and below).
        x[:, pos : (pos + ratio - 1)] = g.nodes[:, start].reshape((-1, 1)) * (1 - theta) + g.nodes[
            :, end
        ].reshape((-1, 1)) * theta
        # Shift the position
        pos += ratio - 1
        shift += ratio + (2 - np.sum(if_add_loc) * (1 - nd[c])) - nd[c]

        # Add coordinate of the endpoint of this interval if relevant
        if if_add_loc[-1]:
            x[:, pos] = g.nodes[:, end]
            old_2_new_nodes[end] = pos
            loc_new_ind.append(pos)
            pos += 1
        else:
            loc_new_ind.append(old_2_new_nodes[end])

        new_indices.append(np.array(loc_new_ind))          
        debug = True
        #assert shift == offset_new_indices

    # For 1d grids, there is a 1-1 relation between faces and nodes
    face_nodes = sps.identity(x.shape[1], format="csc")

    # The grid is 1d, so the face indices are the same as the node indices stored in
    # new_indices.
    cell_face_ind = np.hstack(new_indices)

    # The signs are -1 for the first node of each cell, and 1 for the second. Other
    # possibilities would be possible, but this should work.
    # Find the first occurrence of each node.
    _, first_occurrences = np.unique(cell_face_ind, return_index=True)
    # Initialize the signs array with -1, and set the first occurrences to 1.
    signs = np.full(cell_face_ind.size, -1)
    signs[first_occurrences] = 1

    # We have the cell-face relation. The index pointer assigns two indices to each
    # cell, again, this is 1d.s
    cell_faces = sps.csc_matrix(
        (
            signs,
            cell_face_ind,
            np.arange(0, cell_face_ind.size + 1, 2),
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
    binom = ((1, 0), (2, 1), (0, 2))

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
            np.ravel_multi_index(equal.T, dims=loc_n.shape)  # type: ignore
        ]

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


def mdg_refinement(
    mdg: pp.MixedDimensionalGrid,
    mdg_ref: pp.MixedDimensionalGrid,
    tol: float = 1e-8,
    mode: Literal["nested"] = "nested",
) -> None:
    """Wrapper for ``'coarse_fine_cell_mapping'`` in the grid data dictionaries
    to construct mapping for grids in mixed-dimensional grid.

    Adds a data dictionary entry to each grid in mdg.
    The key is ``'coarse_fine_cell_mapping'``.

    Currently, only nested refinement is supported; more general cases are also
    possible.

    ..rubric:: Acknowledgement

        The code was contributed by Haakon Ervik.

    Note:
        No node prop is added to the reference grids in mdg_ref.

    Parameters:
        mdg: Coarse mixed-dimensional grid.
        mdg_ref: Refined mixed-dimensional grid.
        tol: ``default=1e-8``

            Tolerance for point_in_poly* -methods
        mode: ``default='nested'``

            Refinement mode. ``'nested'`` corresponds to refinement by splitting.

    """

    # Check that both md-grids have the same dimension.
    assert mdg_ref.dim_max() == mdg.dim_max()

    # Check that both mdgs have the same number of subdomains.
    subdomains = mdg.subdomains()
    subdomains_ref = mdg_ref.subdomains()
    assert len(subdomains) == len(subdomains_ref)

    # Check that both mdgs have the same bounding box.
    bmin, bmax = pp.domain.mdg_minmax_coordinates(mdg)
    bmin_ref, bmax_ref = pp.domain.mdg_minmax_coordinates(mdg_ref)
    np.testing.assert_almost_equal(bmin, bmin_ref, decimal=8)
    np.testing.assert_almost_equal(bmax, bmax_ref, decimal=8)

    # Strongly check that grids refer to same domain.
    for i in np.arange(len(subdomains)):
        sd, sd_ref = subdomains[i], subdomains_ref[i]
        assert sd.id == sd_ref

        # Compute the mapping for this subdomain-pair, and assign the result to the node
        # of the coarse mdg.
        if mode == "nested":
            mapping = structured_refinement(sd, sd_ref, point_in_poly_tol=tol)
        else:
            raise NotImplementedError("Unknown refinement mode")

        mdg.subdomain_data(sd)["coarse_fine_cell_mapping"] = mapping


def structured_refinement(
    g: pp.Grid, g_ref: pp.Grid, point_in_poly_tol: float = 1e-8
) -> sps.csc_matrix:
    """Construct a mapping between cells of a grid and its refined version.

    Assuming a regular and a refined mesh, where the refinement is executed by
    splitting. I.e. a cell in the refined grid is completely contained within a cell in
    the coarse grid.

    .. rubric:: Acknowledgement

        The code was contributed by Haakon Ervik.

    Parameters:
        g: Coarse grid.
        g_ref: Refined grid.
        point_in_poly_tol: ``default=1e-8``

            Tolerance for :func:`~porepy.geometry_property_checks.point_in_polyhedron`.

    Raises:
        AssertionError: If ``g_ref`` has more cells than ``g`` or if they are of unequal
            dimension.
        AssertionError: Depending on the dimension of ``g``, a certain number of nodes
            must be available.
        AssertionError: If any cell in the finer grid is not contained in exactly one
            cell of the coarser grid.

    Returns:
        Column major sparse matrix mapping from coarse to fine cells.

    """
    # This method creates a mapping from fine to coarse cells by creating a matrix 'M'
    # where the rows represent the fine cells while the columns represent the coarse
    # cells. In practice this means that for an array 'p', of known values on a coarse
    # grid, by applying the mapping
    #     q = M * p
    # each value in a coarse cell will now be transferred to all the fine cells
    # contained within the coarse cell.

    # The procedure for creating this mapping relies on two main assumptions.
    #     1. Each fine cell is fully contained inside exactly one coarse cell.
    #     2. Each cell can be characterised as a simplex.
    #         I.e. Every node except one defines every face of the object.

    # The first assumption implies that the problem of assessing if a fine cell is
    # contained within a coarse cell is reduced to assessing if the center of a fine
    # cell is contained within the coarse cell.

    # The second assumption implies that a cell in any dimension (1D, 2D, 3D) will be
    # a convex object. This way, we can use existing algorithms in PorePy to find if
    # a point is inside a polygon (2D) or polyhedron (3D). (The 1D case is trivial)

    # The general algorithm is as follows (refer to start of for-loop in the code):
    # 1. Make a list of (pointers to) untested cell centers called 'test_cells_ptr'.
    # 2. Iterate through all coarse cells. Now, consider one of these:
    # 3. For all untested cell centers (defined by 'test_cells_ptr'), check if they
    #     are inside the coarse cell.
    # 4. Those that pass (is inside the coarse cell) add to the mapping, then remove
    #     those point from the list of untested cell centers.
    # 5. Assemble the mapping.

    if g.dim == 0:
        mapping = sps.csc_matrix((np.ones(1), ([0], [0])))
        return mapping
    assert g.num_cells < g_ref.num_cells, "Wrong order of input grids"
    assert g.dim == g_ref.dim, "Grids must be of same dimension"

    # 1. Step: Create a list of tuples pointing to the (start, end) index of the nodes
    # of each cell on the coarse grid.
    cell_nodes = g.cell_nodes()
    # start/end row pointers for each column
    nodes_of_cell_ptr = zip(cell_nodes.indptr[:-1], cell_nodes.indptr[1:])

    # 2. Step: Initialize a sps.csc_matrix mapping fine cells to coarse cells.
    indptr = np.array([0])
    indices = np.empty(0)

    cells_ref = g_ref.cell_centers.copy()  # Cell centers in fine grid
    test_cells_ptr = np.arange(g_ref.num_cells)  # Pointer to cell centers
    nodes = g.nodes.copy()

    # 3. Step: If the grids are in 1D or 2D, we can simplify the calculation by rotating
    # the coordinate system to local coordinates. For example, a 2D grid embedded in 3D
    # would be "rotated" so that each coordinate is of the form (x, y, 0).
    if g.dim == 1:
        # Rotate coarse nodes and fine cell centers to align with the x-axis.
        tangent = pp.map_geometry.compute_tangent(nodes)
        reference = np.array([1.0, 0.0, 0.0])
        R = pp.map_geometry.project_line_matrix(nodes, tangent, reference=reference)
        nodes = R.dot(nodes)[0, :]
        cells_ref = R.dot(cells_ref)[0, :]

    elif g.dim == 2:  # Pre-processing for efficiency
        # Rotate coarse nodes and fine cell centers to the xy-plane.
        R = pp.map_geometry.project_plane_matrix(nodes, check_planar=False)
        nodes = np.dot(R, nodes)[:2, :]
        cells_ref = np.dot(R, cells_ref)[:2, :]

    # 4. Step: Loop through every coarse cell
    for st, nd in nodes_of_cell_ptr:
        nodes_idx = cell_nodes.indices[st:nd]
        num_nodes = nodes_idx.size

        # 5. Step: for the appropriate grid dimension, test all cell centers not already
        # found to be inside some other coarse cell. 'in_poly' is a boolean array that
        # is True for the points inside and False for the points not inside.
        if g.dim == 1:
            # In 1D, we use a numpy method to check which coarse cell the fine points
            # are inside.
            assert num_nodes == 2, "We assume a 'cell' in 1D is defined by two points."
            line = np.sort(nodes[nodes_idx])
            test_points = cells_ref[test_cells_ptr]
            in_poly = np.searchsorted(line, test_points, side="left") == 1

        elif g.dim == 2:
            assert num_nodes == 3, "We assume simplexes in 2D (i.e. 3 nodes)"
            polygon = nodes[:, nodes_idx]
            test_points = cells_ref[:, test_cells_ptr]
            in_poly = pp.geometry_property_checks.point_in_polygon(
                poly=polygon, p=test_points
            )

        elif g.dim == 3:
            # Make polyhedron from node coordinates.
            # Polyhedron defined as a list of nodes defining its (convex) faces.
            # Assumes simplexes: Every node except one defines every face.
            assert num_nodes == 4, "We assume simplexes in 3D (i.e. 4 nodes)"
            node_coords = nodes[:, nodes_idx]

            ids = np.arange(num_nodes)
            polyhedron: np.ndarray = np.array(
                [node_coords[:, ids != i] for i in np.arange(num_nodes)]
            )
            # Test only points not inside another polyhedron.
            test_points = cells_ref[:, test_cells_ptr]
            in_poly = pp.geometry_property_checks.point_in_polyhedron(
                polyhedron=polyhedron, test_points=test_points, tol=point_in_poly_tol
            )

        else:
            logger.warning("A grid of dimension %d encountered. Skip!", g.dim)
            continue

        # 6. Step: Update pointer to which cell centers to use as test points.
        in_poly_ids = test_cells_ptr[in_poly]  # id of cells inside this polyhedron
        # Keep only cells not inside this polyhedron
        test_cells_ptr = test_cells_ptr[~in_poly]

        # Update mapping
        indices = np.append(indices, in_poly_ids)
        indptr = np.append(indptr, indptr[-1] + in_poly_ids.size)

    # 7. Step: assemble the sparse matrix with the mapping.
    data = np.ones(indices.size)
    coarse_fine = sps.csc_matrix((data, indices, indptr))

    assert (
        indices.size == g_ref.num_cells
    ), "Every fine cell should be inside exactly one coarse cell"
    return coarse_fine


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

    def __init__(self, network: pp.fracture_network, params: dict) -> None:
        self._network = network.copy()
        self._counter: int = 0
        self._set_parameters(params)

        self.dim: int
        """Dimension of the fracture network."""

        if isinstance(network, FractureNetwork2d):
            self.dim = 2
        elif isinstance(network, FractureNetwork3d):
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
            self._gmsh.finalize()  # type: ignore
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

        Raises:
            ValueError if the refinement mode is not recognized.

        Returns:
            The refined mixed dimensional grid.

        """
        if self._refinement_mode == "nested":
            return self._generate_nested(counter)
        elif self._refinement_mode == "unstructured":
            return self._generate_unstructured(counter)
        else:
            raise ValueError(f"Unknown refinement mode {self._refinement_mode}")

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
        # Generate the first grid.
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
        pp.set_local_coordinate_projections(mdg)
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
