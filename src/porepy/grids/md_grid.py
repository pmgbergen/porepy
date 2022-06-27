"""
Module to store the grid hierarchy formed by a set of fractures and their
intersections in the form of a MixedDimensionalGrid.




"""
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.grids import mortar_grid
from porepy.utils import setmembership


class MixedDimensionalGrid:
    """
    Container for the hierarchy of grids formed by fractures and their
    intersections.

    The mixed-dimensional grid can be considered a graph, with the fixed-dimensional grids
    defining nodes, and interfaces edges. The terminology in the implementation,
    and in method names, are strongly marked by this graph thinking.

    Connections between the grids are defined as a tuple, with first the highest
    and then the lowest-dimensional neighboring grid.

    Attributes:
        _subdomain_data (dictionary): For each subdomain grid, a dictionary with
            grids as keys, and data associated with the grid as values.
        _interface_data (dictionary): For each interface grid, a dictionary with
            the edges as keys, and data associated with the grid as values.

    """

    def __init__(self) -> None:
        self._subdomain_data: Dict[pp.Grid, Dict] = {}
        self._interface_data: Dict[pp.MortarGrid, Dict] = {}
        self._interface_to_subdomains: Dict[pp.MortarGrid, Tuple[pp.Grid, pp.Grid]]
        self.name = "Grid Tree"

    def __contains__(self, key: Any) -> bool:
        """Overload __contains__.

        Args:
            key (object): Object to be tested.

        Return:
            True if either key is a pp.Grid, and key is among the nodes of the graph
                representation of this md-grid, *or* key is a 2-tuple, with both items
                in self._subdomain_data and the tuple is in self._interface_data.

        """
        if isinstance(key, pp.Grid):
            return key in self._subdomain_data
        elif isinstance(key, pp.MortarGrid):
            return key in self._interface_data

        # Everything else is not in self.
        return False

    # --------- Iterators -------------------------

    def subdomains(
        self, return_data: bool = False, dim: Optional[int] = None
    ) -> Generator[Tuple[pp.Grid, Tuple[pp.Grid, Dict]], None, None]:
        """Iterator over the subdomains in the MixedDimensionalGrid.

        Optionally, the iterator can filter subdomains on dimension. Also, the data
        dictionary belonging to the subdomain can be returned.

        Args:
            return_data (boolean, optional): If True, the data dictionary of the
                subdomain will be returned together with the subdomain grids. Defaults
                to False.
            dim (int, optional): If provided, only subdomains of the specified dimension
                will be returned.

        Yields:
            pp.Grid: Grid associated with the current node.
            dict: Data dictionary associated with the subdomain. Only returned if
                return_data = True.

        """
        for grid in self._subdomain_data:
            # Filter on dimension if requested.
            if dim is not None and dim != grid.dim:
                continue

            if return_data:
                yield grid, self._subdomain_data[grid]
            else:
                yield grid

    def interfaces(
        self, return_data: bool = False, dim: Optional[int] = None
    ) -> Generator[pp.MortarGrid, None, None]:
        """
        Iterator over the MortarGrids belonging to interfaces in the
        MixedDimensionalGrid.

        Optionally, the iterator can filter interfaces on dimension. Also, the data
        dictionary belonging to the interface can be returned.

        Args:
            return_data (boolean, optional): If True, the data dictionary of the
                subdomain will be returned together with the interface grids. Defaults
                to False.
            dim (int, optional): If provided, only interfaces  of the specified
                dimension will be returned.

        Yields:
            MortarGrid: Grid on the the interface.
            dict: Data dictionary associated with the subdomain. Only returned if
                return_data = True.

        """
        for edge in self._interface_data:
            # Filter on dimension if requested.
            if dim is not None and dim != edge.dim:
                continue

            if return_data:
                yield edge, self._interface_data[edge]
            else:
                yield edge

    # ---------- Navigate within the graph --------

    def subdomains_of_interface(self, intf: pp.MortarGrid) -> Tuple[pp.Grid, pp.Grid]:
        """Obtain the subdomains of an interface.

        The nodes will be given in decending order with respect their
        dimension. If the edge is between grids of the same dimension, the node
        ordering (as defined by assign_node_ordering()) is used. If no ordering
        of nodes exists, assign_node_ordering() will be called by this method.

        Args:
            intf, pp.MortarGrid: The

        Returns:
            pp.Grid: The primary subdomain neighboring this interface. Normally, this
                will be the subdomain with the higher dimension.
            pp.Grid: The secondary subdomain neighboring this interface. Normally, this
                will be the subdomain with the lowerer dimension.

        """
        subdomains = self._interface_to_subdomains[intf]
        if subdomains[0].dim == subdomains[1].dim:
            if not (
                "node_number" in self._subdomain_data[subdomains[0]]
                and "node_number" in self._subdomain_data[subdomains[1]]
            ):
                self.assign_node_ordering()

            subdomain_indexes = [
                self.subdomain_data(grid, "node_number") for grid in subdomains
            ]
            if subdomain_indexes[0] < subdomain_indexes[1]:
                return subdomain_indexes[0], subdomain_indexes[1]
            else:
                return subdomain_indexes[1], subdomain_indexes[0]

        elif subdomains[0].dim > subdomains[1].dim:
            return subdomains[0], subdomains[1]
        else:
            return subdomains[1], subdomains[0]

    def subdomain_pair_to_interface(
        self, subdomain_pair: Tuple[pp.Grid, pp.Grid]
    ) -> pp.MortarGrid:
        """Get an interface from a subdomain"""

        reverse_dict = {v: k for k, v in self._interface_to_subdomains.items()}
        if subdomain_pair in reverse_dict:
            return reverse_dict[subdomain_pair]
        elif subdomain_pair[::-1] in reverse_dict:
            return reverse_dict[subdomain_pair[::-1]]
        else:
            raise KeyError("Unknown subdomain pair")

    def interfaces_of_subdomain(
        self, node: pp.Grid
    ) -> Generator[pp.MortarGrid, None, None]:
        """Iterator over the edges of the specific node.

        Args:
            node: A node in the graph.

        Yields:
            edge (Tuple[pp.Grid, pp.Grid]):
                The edge (pair of grids) associated with an edge.

        """
        for edge in self._interface_data:
            node_pair = self._interface_to_subdomains[edge]
            if node_pair[0] == node or node_pair[1] == node:
                yield edge

    def neighboring_subdomains(
        self, node: pp.Grid, only_higher: bool = False, only_lower: bool = False
    ) -> List[pp.Grid]:
        """Get neighbors of a node in the graph.

        Optionally, return only neighbors corresponding to higher or lower
        dimension. At most one of only_higher and only_lower can be True.

        Args:
            node (pp.Grid): node in the graph.
            only_higher (boolean, optional): Consider only the higher
                dimensional neighbors. Defaults to False.
            only_lower (boolean, optional): Consider only the lower dimensional
                neighbors. Defaults to False.

        Return:
            np.ndarray: Neighbors of node 'node'

        Raises:
            ValueError if both only_higher and only_lower is True.

        """

        neigh = []
        for mg, edge in self._interface_to_subdomains.items():
            if edge[0] == node:
                neigh.append(edge[1])
            elif edge[1] == node:
                neigh.append(edge[0])

        if not only_higher and not only_lower:
            return neigh
        elif only_higher and only_lower:
            raise ValueError("Cannot return both only higher and only lower")
        elif only_higher:
            # Find the neighbours that are higher dimensional
            return [g for g in neigh if g.dim > node.dim]
        else:
            # Find the neighbours that are higher dimensional
            return [g for g in neigh if g.dim < node.dim]

    # ------------ Getters for node and edge properties

    def subdomain_data(self, grid: pp.Grid) -> Any:
        """Getter for a subdomain property of the MixedDimensionalGrid.

        Args:
            grid (pp.Grid): The grid associated with the node.

        Returns:
            object: A dictionary with keys and properties.

        """
        return self._subdomain_data[grid]

    def interface_data(self, edge: pp.MortarGrid) -> Dict:
        """Getter for an edge properties of the bucket.

        Args:
            edge (Tuple of pp.Grid): The two grids making up the edge.

        Returns:
            object: The properties.

        Raises:
            KeyError if the two grids do not form an edge.

        """
        return self._interface_data[edge]

    # ------------ Add new nodes and edges ----------

    def add_subdomains(self, new_grids: Union[pp.Grid, Iterable[pp.Grid]]) -> None:
        """
        Add grids to the hierarchy.

        Args:
            new_grids (pp.Grid or Iterable of pp.Grid): The grids to be added. None of
                these should have been added previously.

        Raises:
            ValueError if a grid is already present in the bucket

        """
        if isinstance(new_grids, pp.Grid):
            ng = [new_grids]
        else:
            ng = new_grids  # type: ignore

        if any([i is j for i in ng for j in list(self._subdomain_data)]):
            raise ValueError("Grid already defined in MixedDimensionalGrid")

        for grid in ng:
            # Add the grid to the list of subdomains with an empty data dictionary.
            self._subdomain_data[grid] = {}

    def add_interface(
        self,
        mg: pp.MortarGrid,
        grids: Tuple[pp.Grid, pp.Grid],
        primary_secondary_map: sps.spmatrix,
    ) -> None:
        """
        Add an interface to the MixedDimensionalGrid.

        FIXME: Subdomain order should be primary, secondary. Applies to, and will have
        consequences for, the entire code base. Primary is the grid with the highest
        dimension (in the rare cases with an equi-dimensional coupling, the ordering
        is determined from the argument to this function).

        If the grids have different dimensions, the coupling will be added with
        the higher-dimensional grid as the first node. For equal dimensions,
        the ordering of the nodes is the same as in input grids.

        Args:
            mg (pp.MortarGrid): MortarGrid for this
            grids (Tuple, len==2). Grids to be connected. Order is arbitrary.
            primary_secondary_map (sps.spmatrix): Identity mapping between cells in the
                higher-dimensional grid and faces in the lower-dimensional
                grid. No assumptions are made on the type of the object at this
                stage. In the grids[0].dim = grids[1].dim case, the mapping is
                from faces of the first grid to faces of the second one. In the
                co-dimension 2 case, it maps from cells to cells.

        Raises:
            ValueError if the edge is not specified by exactly two grids
            ValueError if the edge already exists.
            ValueError if the two grids are not between zero and two dimensions apart.

        """
        if len(grids) != 2:
            raise ValueError("An edge should be specified by exactly two grids")

        if mg in self._interface_data:
            raise ValueError("Cannot add existing edge")

        # The data dictionary is initialized with a single field, defining the mapping
        # between grid quantities in the primary and secondary neighboring subdomains
        # (for a co-dimension 1 coupling, this will be between faces in the
        # higher-dimensional grid and cells in the lower-dimensional grid).
        data = {"face_cells": primary_secondary_map}

        # Add the interface to the list, with
        self._interface_data[mg] = data

        # The higher-dimensional grid is the first node of the edge.

        if grids[0].dim - 1 == grids[1].dim:
            node_pair = (grids[0], grids[1])
        elif grids[0].dim == grids[1].dim - 1:
            node_pair = (grids[1], grids[0])
        elif grids[0].dim - 2 == grids[1].dim:
            node_pair = (grids[0], grids[1])
        elif grids[0].dim == grids[1].dim - 2:
            node_pair = (grids[1], grids[0])
        elif grids[0].dim == grids[1].dim:
            node_pair = (grids[0], grids[1])
        else:
            raise ValueError("Grid dimension mismatch")

        self._interface_to_subdomains[mg] = node_pair

    # --------- Remove and update nodes

    def remove_subdomain(self, node: pp.Grid) -> None:
        """
        Remove subdomain, and related interfaces, from the grid bucket.

        Args:
           node : the subdomain to be removed

        """
        # Delete from the subdomain lits.
        del self._subdomain_data[node]

        # Find interfaces to be removed, store in a list.
        edges_to_remove: List[pp.MortarGrid] = []
        for mg in self.interfaces():
            edge = self._interface_to_subdomains[mg]
            if edge[0] == node or edge[1] == node:
                edges_to_remove.append(mg)

        # Remove interfaces.
        for mg in edges_to_remove:
            del self._interface_data[mg]
            del self._interface_to_subdomains[mg]

    def update_subdomains(self, mapping: Dict[pp.Grid, pp.Grid]) -> None:
        """
        Update the grids giving old and new values. The interfaces are updated
        accordingly.

        FIXME: This is in reality a subset of self.replace_subdomains. Should we merge?
        EK FIX
        Args:
            mapping: A dictionary with the old grid as keys and new grid as values.

        """
        for old, new in mapping.items():
            data = self._subdomain_data[old]
            self._subdomain_data[new] = data
            del self._subdomain_data[old]

            # Loop over edges, look for edges which has the node to be replaced as
            # one of its neighbors
            for mg, edge in self._interface_to_subdomains.items():
                # If we find a hit, overwrite the old edge information.
                if edge[0] == old:
                    self._interface_to_subdomains[mg] = (new, edge[1])
                elif edge[1] == old:
                    self._interface_to_subdomains[mg] = (edge[0], new)

    # ---------- Functionality related to ordering of nodes

    def assign_subdomain_ordering(self, overwrite_existing: bool = True) -> None:
        """
        Assign an ordering of the nodes in the graph, stored as the attribute
        'node_number'.

        The ordering starts with grids of highest dimension. The ordering
        within each dimension is determined by an iterator over the graph, and
        can in principle change between two calls to this function.

        If an ordering covering all nodes in the graph already exist, but does
        not coincide with the new ordering, a warning is issued. If the optional
        parameter overwrite_existing is set to False, no update is performed if
        a node ordering already exists.

        Args:
            overwrite_existing (bool, optional): If True (default), any existing node
                ordering will be overwritten.

        """

        # Check whether 'node_number' is defined for the grids already.
        ordering_exists = True
        for _, data in self._subdomain_data:
            if "node_number" not in data:
                ordering_exists = False
        if ordering_exists and not overwrite_existing:
            return

        counter = 0
        # Loop over grids in decreasing dimensions
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            for grid in self.subdomains_of_dimension(dim):
                data = self._subdomain_data[grid]
                # Get old value, issue warning if not equal to the new one.
                num = data.get("node_number", -1)
                if ordering_exists and num != counter:
                    warnings.warn("Order of graph nodes has changed")
                # Assign new value
                data["node_number"] = counter
                counter += 1

        counter = 0
        for _, data in self.interfaces():
            data["edge_number"] = counter
            counter += 1

    def update_node_ordering(self, removed_number: int) -> None:
        """
        Uppdate an existing ordering of the nodes in the graph, stored as the attribute
        'node_number'.
        Intended for keeping the node ordering after removing a node from the bucket.
        In this way, the edge sorting will not be disturbed by the removal, but no gaps
        in are created.

        Parameter:
            removed_number (int): node_number of the removed grid.

        """

        # Loop over grids in decreasing dimensions
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            for g in self.subdomains_of_dimension(dim):
                if "node_number" not in self._subdomain_data[g]:
                    # It is not clear how severe this case is. For the moment,
                    # we give a warning, and hope the user knows what to do
                    warnings.warn("Tried to update node ordering where none exists")
                    # No point in continuing with this node.
                    continue

                # Obtain the old node number
                n = self._subdomain_data[g]
                old_number = n.get("node_number", -1)
                # And replace it if it is higher than the removed one
                if old_number > removed_number:
                    n["node_number"] = old_number - 1

    def sort_subdomain_data(self, nodes: List[pp.Grid]) -> List[pp.Grid]:
        """
        Sort nodes according to node number.

        Args:
            nodes: List of graph nodes.

        Returns:
            sorted_subdomain_data: The same nodes, sorted.

        """
        assert all(["node_number" in self._subdomain_data[g] for g in nodes])

        return sorted(nodes, key=lambda n: self.subdomain_data(n, "node_number"))

    # ------------- Miscellaneous functions ---------

    def compute_geometry(self) -> None:
        """Compute geometric quantities for the grids."""
        for grid in self.subdomains():
            grid.compute_geometry()

        for mg in self.interfaces():
            mg.compute_geometry()

    def copy(self) -> "MixedDimensionalGrid":
        """Make a shallow copy of the grid bucket. The underlying grids are not copied.

        Return:
            pp.MixedDimensionalGrid: Copy of this MixedDimensionalGrid.

        """
        gb_copy: pp.MixedDimensionalGrid = MixedDimensionalGrid()
        gb_copy._subdomain_data = self._subdomain_data.copy()
        gb_copy._interface_data = self._interface_data.copy()
        gb_copy._interface_to_subdomains = self._interface_to_subdomains.copy()
        return gb_copy

    def replace_subdomains_and_interfaces(
        self,
        g_map: Optional[Dict[pp.Grid, pp.Grid]] = None,
        mg_map: Optional[
            Dict[pp.MortarGrid, Dict[mortar_grid.MortarSides, pp.Grid]]
        ] = None,
        tol: float = 1e-6,
    ) -> None:
        """Replace grids and / or mortar grids in the mixed-dimensional grid.

        FIXME: Improve documentation.

        Args:
            gb (MixedDimensionalGrid): To be updated.
            g_map (dictionary): Mapping between the old and new grids. Keys are
                the old grids, and values are the new grids.
            mg_map (dictionary): Mapping between the old mortar grid and new
                (side grids of the) mortar grid. Keys are the old mortar grids,
                and values are dictionaries with side grid number as keys, and
                side grids as values.

        """
        # The operation is carried out in three steps:
        # 1) Update the mortar grid. This will also update projections to and from the
        #    mortar grids so that they are valid for the new mortar grid, but not with
        #    respect to any changes in the subdomain grids.
        # 2) Update the subdomain grids with respect to nodes in the MixedDimensionalGrid.
        # 3) Update the subdomain grids with respect to the interfaces in the MixedDimensionalGrid.
        #    This will update projections to and from mortar grids.
        # Thus, if both the mortar grid and an adjacent subdomain grids are updated, the
        # projections will be updated twice.

        if mg_map is None:
            mg_map = {}

        # refine the mortar grids when specified
        for mg_old, mg_new in mg_map.items():
            mg_old.update_mortar(mg_new, tol)

        # update the grid bucket considering the new grids instead of the old one
        # valid only for physical grids and not for mortar grids
        if g_map is not None:
            self.update_subdomains(g_map)
        else:
            g_map = {}

        # refine the grids when specified
        for g_old, g_new in g_map.items():
            for mg in self.interfaces_of_subdomain(g_new):
                if mg.dim == g_new.dim:
                    # update the mortar grid of the same dimension
                    mg.update_secondary(g_new, tol)
                else:  # g_new.dim == mg.dim + 1
                    # FIXME: Will this also work for co-dimension 2 couplings?
                    # EK believes so.
                    mg.update_primary(g_new, g_old, tol)

    # ----------- Apply functions to nodes and edges

    def apply_function_to_subdomains(
        self, fct: Callable[[pp.Grid, Dict], Any]
    ) -> np.ndarray:
        """
        Loop on all the nodes and evaluate a function on each of them.

        Parameter:
            fct: function to evaluate. It takes a grid and the related data and
                returns a scalar.

        Returns:
            values: vector containing the function evaluated on each node,
                ordered by 'node_number'.

        """
        values = np.empty(self.num_subdomains())
        for g, d in self._subdomain_data.items():
            values[d["node_number"]] = fct(g, d)
        return values

    def apply_function_to_interfaces(
        self, fct: Callable[[pp.Grid, pp.Grid, Dict, Dict, Dict], Any]
    ) -> sps.spmatrix:
        """
        Loop on all the edges and evaluate a function on each of them.

        Parameter:
            fct: function to evaluate. It returns a scalar and takes: the higher
                and lower dimensional grids, the higher and lower dimensional
                data, the global data.

        Returns:
            matrix: sparse strict upper triangular matrix containing the function
                evaluated on each edge (pair of nodes), ordered by their
                relative 'node_number'.

        """
        i = np.zeros(len(self._interface_data), dtype=int)
        j = np.zeros(i.size, dtype=int)
        values = np.zeros(i.size)

        # Loop over the edges of the graph (pair of connected nodes)
        idx = 0
        for mg in self.interfaces():
            data = self.interface_data(mg)

            g_l, g_h = self.subdomains_of_interface(mg)
            data_l, data_h = self.subdomain_data(g_l), self.subdomain_data(g_h)

            i[idx] = self.subdomain_data(g_l, "node_number")
            j[idx] = self.subdomain_data(g_h, "node_number")
            values[idx] = fct(g_h, g_l, data_h, data_l, data)
            idx += 1

        # Upper triangular matrix
        return sps.coo_matrix(
            (values, (i, j)),
            (self.num_graph_subdomain_data(), self.num_graph_subdomain_data()),
        )

    # ---- Methods for getting information on the bucket, or its components ----

    def diameter(
        self, cond: Callable[[Union[pp.Grid, Tuple[pp.Grid, pp.Grid]]], bool] = None
    ) -> float:
        """
        Compute the grid bucket diameter (mesh size), considering a loop on all
        the grids.  It is possible to specify a condition based on the grid to
        select some of them.

        Parameter:
            cond: optional, predicate with a grid or a tuple of grids (an edge)
                as input.

        Return:
            diameter: the diameter of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        diam_g = [
            np.amax(g.cell_diameters()) for g in self._subdomain_data.keys() if cond(g)
        ]

        diam_mg = [
            np.amax(mg.cell_diameters()) for mg in self._interface_data if cond(mg)
        ]

        return np.amax(np.hstack((diam_g, diam_mg)))

    #    def size(self) -> int:
    #        """
    #        Returns:
    #            int: Number of mono-dimensional grids and interfaces in the bucket.
    #
    #        """
    #        return self.num_subdomains() + self.num_interfaces()

    def dim_min(self) -> int:
        """
        Returns:
            int: Minimum dimension of the grids present in the hierarchy.

        """
        return np.min([grid.dim for grid in self.subdomains()])

    def dim_max(self) -> int:
        """
        Returns:
            int: Maximum dimension of the grids present in the hierarchy.

        """
        return np.max([grid.dim for grid in self.subdomains()])

    """
    def all_dims(self) -> np.ndarray:
        ""
        Returns:
            np.array: Active dimensions of the grids present in the hierarchy.

        ""
        return np.unique([grid.dim for grid in self.subdomains()])
    """

    def num_subdomain_cells(self, cond: Callable[[pp.Grid], bool] = None) -> int:
        """
        Compute the total number of cells of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            num_cells (int): the total number of cells of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.sum(  # type: ignore
            [grid.num_cells for grid in self.subdomains() if cond(grid)], dtype=int
        )

    def num_interface_cells(self, cond: Callable[[pp.Grid], bool] = None) -> int:
        """
        Compute the total number of mortar cells of the grid bucket, considering
        a loop on all mortar grids. It is possible to specify a condition based
        on the grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            num_cells (int): the total number of mortar cells of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.sum(  # type: ignore
            [mg.num_cells for mg in self.interfaces() if cond(mg)],
            dtype=int,
        )

    def num_subdomains(self):
        """
        Return the total number of nodes (physical meshes) in the graph.

        Returns:
            int: Number of nodes in the graph.

        """
        return len(self._subdomain_data)

    def num_interfaces(self) -> int:
        """
        Return the total number of edge in the graph.

        Returns:
            int: Number of edges in the graph.

        """
        return len(self._interface_data)

    def __str__(self) -> str:
        max_dim = self.subdomains_of_dimension(self.dim_max())
        num_subdomain_data = 0
        num_cells = 0
        for g in max_dim:
            num_subdomain_data += g.num_subdomain_data
            num_cells += g.num_cells
        s = (
            "Mixed-dimensional grid. \n"
            f"Maximum dimension present: {self.dim_max()} \n"
            f"Minimum dimension present: {self.dim_min()} \n"
        )
        s += "Size of highest dimensional grid: Cells: " + str(num_cells)
        s += ". Nodes: " + str(num_subdomain_data) + "\n"
        s += "In lower dimensions: \n"
        for dim in range(self.dim_max() - 1, self.dim_min() - 1, -1):
            gl = self.subdomains_of_dimension(dim)
            num_subdomain_data = 0
            num_cells = 0
            for g in gl:
                num_subdomain_data += g.num_subdomain_data
                num_cells += g.num_cells

            s += (
                f"{len(gl)} grids of dimension {dim}, with in total "
                f"{num_cells} cells and {num_subdomain_data} nodes. \n"
            )

        s += f"Total number of interfaces: {self.num_subdomains()}\n"
        for dim in range(self.dim_max(), self.dim_min(), -1):
            num_e = 0
            for mg in self.interfaces():
                if mg.dim == dim:
                    num_e += 1

            s += f"{num_e} interfaces between grids of dimension {dim} and {dim-1}\n"
        return s

    def __repr__(self) -> str:
        s = (
            f"Mixed-dimensional containing {self.num_subdomains() } grids and "
            f"{self.num_interfaces()} interfaces\n"
            f"Maximum dimension present: {self.dim_max()} \n"
            f"Minimum dimension present: {self.dim_min()} \n"
        )

        if self.num_subdomains() > 0:
            for dim in range(self.dim_max(), self.dim_min() - 1, -1):
                gl = self.subdomains_of_dimension(dim)
                nc = 0
                for g in gl:
                    nc += g.num_cells
                s += (
                    f"{len(gl)} grids of dimension {dim}" f" with in total {nc} cells\n"
                )
        if self.num_interfaces() > 0:
            for dim in range(self.dim_max(), self.dim_min(), -1):
                num_e = 0
                num_mc = 0
                for mg in self.interfaces():
                    if mg.dim == dim:
                        num_e += 1
                        num_mc += mg.num_cells

                s += (
                    f"{num_e} interfaces between grids of dimension {dim} and {dim-1}"
                    f" with in total {num_mc} mortar cells\n"
                )

        return s
