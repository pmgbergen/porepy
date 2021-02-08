#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various methods to refine a grid.

Created on Sat Nov 11 17:06:37 2017

"""
import abc
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import gmsh
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids.grid import Grid
from porepy.grids.simplex import TriangleGrid
from porepy.grids.structured import TensorGrid

module_sections = ["grids", "gridding"]


@pp.time_logger(sections=module_sections)
def distort_grid_1d(
    g: pp.Grid, ratio: Optional[float] = 0.1, fixed_nodes: Optional[np.ndarray] = None
) -> pp.Grid:
    """Randomly distort internal nodes in a 1d grid.

    The boundary nodes are left untouched.

    The perturbations will not perturb the topology of the mesh.

    Parameters:
         g (pp.grid): To be perturbed. Modifications will happen in place.
         ratio (int, optional, defaults to 0.1): Perturbation ratio. A node can be
              moved at most half the distance in towards any of its
              neighboring nodes. The ratio will multiply the chosen
              distortion. Should be less than 1 to preserve grid topology.
         fixed_nodes (np.array, optional): Index of nodes to keep fixed under
             distortion. Boundary nodes will always be fixed, even if not
             expli)itly included as fixed_node

    Returns:
         grid: With distorted nodes

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


@pp.time_logger(sections=module_sections)
def refine_grid_1d(g: pp.Grid, ratio: int = 2) -> pp.Grid:
    """Refine cells in a 1d grid.

    Parameters:
        g (pp.Grid): A 1d grid, to be refined.
        ratio (int):

    Returns:
        grid: New grid, with finer cells.

    """

    # Implementation note: The main part of the function is the construction of
    # the new cell-face relation. Since the grid is 1d, nodes and faces are
    # equivalent, and notation used mostly refers to nodes instead of faces.

    # Cell-node relation
    cell_nodes = g.cell_nodes()
    nodes, cells, _ = sps.find(cell_nodes)

    # Every cell will contribute (ratio - 1) new nodes
    num_new_nodes = (ratio - 1) * g.num_cells + g.num_nodes
    x = np.zeros((3, num_new_nodes))
    # Cooridates for splitting of cells
    theta = np.arange(1, ratio) / float(ratio)
    pos = 0
    shift = 0

    # Array that indicates whether an item in the cell-node relation represents
    # a node not listed before (e.g. whether this is the first or second
    # occurence of the cell)
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

        # Flags for whether this is the first occurences of the the nodes of
        # the old cell. If so, they should be added to the new node array
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


@pp.time_logger(sections=module_sections)
def refine_triangle_grid(g: pp.TriangleGrid) -> Tuple[pp.TriangleGrid, np.ndarray]:
    """Uniform refinement of triangle grid, all cells are split into four
    subcells by combining existing nodes and face centrers.

    Implementation note: It should be fairly straighforward to extend the
    function to 3D simplex grids as well. The loop over face combinations
    extends straightforwardly, but obtaining the node in the corner defined
    by faces may be a bit tricky.

    Parameters:
        g (pp.TriangleGrid). To be refined.

    Returns:
        TriangleGrid: New grid, with nd+2 times as many cells as g.
        np.array: Mapping from new to old cells.

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

    # Holder for new tessalation.
    new_tri = np.empty(shape=(nd + 1, g.num_cells, nd + 2), dtype=int)

    # Loop over combinations
    for ti, b in enumerate(binom):
        # Find face-nodes of these faces. Each column corresponds to a single cell.
        # There should be one duplicate in each column (since this is 2D)
        loc_n = np.vstack((fn[:, cf[b[0]]], fn[:, cf[b[1]]]))
        # Find the duplicate: First sort along column, then take diff along column
        # and look for zero.
        # Implementation note: To extend to 3D, also require that np.gradient
        # is zero (that function should be able to do this).
        loc_n.sort(axis=0)
        equal = np.argwhere(np.diff(loc_n, axis=0) == 0)
        # equal is now 2xnum_cells. To pick out the right elements, consider
        # the raveled index, and construct the corresponding raveled array
        equal_n = loc_n.ravel()[np.ravel_multi_index(equal.T, loc_n.shape)]

        # Define node combination. Both nodes associated with a face have their
        # offset adjusted.
        new_tri[:, :, ti] = np.vstack((equal_n, offset + cf[b[0]], offset + cf[b[1]]))

    # Create final triangle by combining faces only
    new_tri[:, :, -1] = offset + cf

    # Reshape into 2d array
    new_tri = new_tri.reshape((nd + 1, (nd + 2) * g.num_cells))

    # The new grid inherits the history of the old one.
    name = g.name.copy()
    name.append("Refinement")

    # Also create mapping from refined to parent cells
    parent = np.tile(np.arange(g.num_cells), g.dim + 2)

    return TriangleGrid(new_nodes, tri=new_tri, name=name), parent


@pp.time_logger(sections=module_sections)
def remesh_1d(g_old: pp.Grid, num_nodes: int, tol: Optional[float] = 1e-6) -> pp.Grid:
    """Create a new 1d mesh covering the same domain as an old one.

    The new grid is equispaced, and there is no guarantee that the nodes in
    the old and new grids are coincinding. Use with care, in particular for
    grids with internal boundaries.

    Parameters:
        g_old (pp.Grid): 1d grid to be replaced.
        num_nodes (int): Number of nodes in the new grid.
        tol (double, optional): Tolerance used to compare node coornidates
            (for mapping of boundary conditions). Defaults to 1e-6.

    Returns:
        pp.Grid: New grid.

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

    # map the tags from the old grid to the new one
    # normally the tags are given at faces/point that are fixed the 1d mesh
    # we use this assumption to proceed.
    for f_old in np.arange(g_old.num_faces):
        # detect in the new grid which face is geometrically the same (upon a tolerance)
        # as in the old grid
        dist = pp.distances.point_pointset(g_old.face_centers[:, f_old], g.face_centers)
        f_new = np.where(dist < tol)[0]

        # if you find a match transfer all the tags from the face in the old grid to
        # the face in the new grid
        if f_new.size:
            if f_new.size != 1:
                raise ValueError(
                    "It cannot be more than one face, something went wrong"
                )
            for tag in pp.utils.tags.standard_face_tags():
                g.tags[tag][f_new] = g_old.tags[tag][f_old]

    g.update_boundary_node_tag()

    return g


class GridSequenceIterator:
    @pp.time_logger(sections=module_sections)
    def __init__(self, factory):
        self._factory = factory
        self._counter = 0

    @pp.time_logger(sections=module_sections)
    def __next__(self):
        if self._counter >= self._factory._num_refinements:
            self._factory.close()
            raise StopIteration()

        else:
            gb = self._factory._generate(self._counter)
            self._counter += 1
            return gb


class GridSequenceFactory(abc.ABC):
    """Factory class to generate a set of refined grids.

    To define new refinement types, inherit from this class and override the _generate()
    method.

    The class can be used in (for now) two ways: Either by nested or unstructured
    refinement. This is set by setting the parameter 'mode' to 'nested' or
    'unstructured', respectively.

    The number of refinemnet is set by the parameter 'num_refinements'.

    Mesh sizes is set by the standard FractureNetwork.mesh() arguments 'mesh_size_fracs'
    and 'mesh_size_bound'. These are set by the parameter 'mesh_param'. If the mode is
    'nested', 'mesh_param' should be a single dictionary; subsequent mesh refinements
    are then set by nested refinement. If the mode is 'unstructured', mesh_params should
    be a list, where each list element is a dictionary with mesh size parameters to be
    passed to the FractureNetwork.

    Further argument, such as fractures that are really constraints, can be given by
    params['grid_params']. These are passed on the FractureNetwork.mesh(), essentially
    as **kwargs.

    Acknowledgement: The design idea and the majority of the code was contributed by
        Haakon Ervik.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(
        self, network: Union[pp.FractureNetwork2d, pp.FractureNetwork3d], params: Dict
    ) -> None:
        """
        Construct a GridSequenceFactory.

        Args:
            network (Union[pp.FractureNetwork2d, pp.FractureNetwork3d]): Define domain
                to be discretized.
            params (Dict): Parameter dictionary. See class documentation for details.

        """
        self._network = network.copy()
        self._counter = 0
        self._set_parameters(params)

        if isinstance(network, pp.FractureNetwork2d):
            self.dim = 2
        elif isinstance(network, pp.FractureNetwork3d):
            self.dim = 3
        else:
            raise ValueError("Invalid type for fracture network")

        if self._refinement_mode == "nested":
            self._prepare_nested()

    @pp.time_logger(sections=module_sections)
    def __iter__(self):
        return GridSequenceIterator(self)

    @pp.time_logger(sections=module_sections)
    def close(self):
        if hasattr(self, "_gmsh"):
            self._gmsh.finalize()

    @pp.time_logger(sections=module_sections)
    def _generate(self, counter):
        if self._refinement_mode == "nested":
            return self._generate_nested(counter)
        elif self._refinement_mode == "unstructured":
            return self._generate_unstructured(counter)

    @pp.time_logger(sections=module_sections)
    def _prepare_nested(self):
        # Operate on a deep copy of the network to avoid that fractures etc. are
        # unintentionally modified during operations.
        net = self._network.copy()

        file_name = "gmsh_convergence"
        # Generate data in a format suitable for defining a model in gmsh
        gmsh_data = net.prepare_for_gmsh(
            self._mesh_parameters,
            **self._grid_parameters,
        )
        # Initialize gmsh model with the relevant data. The data will be accessible
        # in gmsh.model.mesh (that is the way the Gmsh Python API works)
        pp.fracs.gmsh_interface.GmshWriter(gmsh_data)
        # Generate the first grid
        gmsh.model.mesh.generate(dim=self.dim)
        self._out_file = file_name
        self._gmsh = gmsh

    @pp.time_logger(sections=module_sections)
    def _generate_nested(self, counter: int):
        out_file = Path(self._out_file)
        out_file = out_file.parent / out_file.stem
        out_file_name = f"{out_file}_{counter}.msh"

        # The first mesh is already done. Start refining all subsequent meshes.
        self._gmsh.model.mesh.refine()  # Refine the mesh

        self._gmsh.write(out_file_name)  # Write the result to '.msh' file

        gb = pp.fracture_importer.dfm_from_gmsh(out_file_name, self.dim)
        pp.contact_conditions.set_projections(gb)
        return gb

    @pp.time_logger(sections=module_sections)
    def _generate_unstructured(self, counter: int):
        net = self._network.copy()
        mesh_args = self._mesh_parameters[counter]
        grid_param = self._grid_parameters
        gb = net.mesh(mesh_args, **grid_param)
        return gb

    @pp.time_logger(sections=module_sections)
    def _set_parameters(self, param):
        self._refinement_mode = param["mode"]

        self._num_refinements = param["num_refinements"]

        self._mesh_parameters = param["mesh_param"]
        self._grid_parameters = param.get("grid_param", {})
