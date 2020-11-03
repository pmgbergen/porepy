"""
Module to store the grid hiearchy formed by a set of fractures and their
intersections in the form of a GridBucket.

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
from porepy.utils import setmembership


class GridBucket:
    """
    Container for the hierarchy of grids formed by fractures and their
    intersections.

    The mixed-dimensional grid can be considered a graph, with the fixed-dimensional grids
    defining nodes, and interfaces edges. The terminology in the implementation,
    and in method names, are strongly marked by this graph thinking.

    Connections between the grids are defined as a tuple, with first the highest
    and the the lowest-dimensional neighboring grid.

    Attributes:
        _nodes (dictionary): For each subdomain grid, a dictionary with
            grids as keys, and data associated with the grid as values.
        _edges (dictionary): For each interface grid, a dictionary with
            the edges as keys, and data associated with the grid as values.

    """

    def __init__(self) -> None:
        self._nodes: Dict[pp.Grid, Dict] = {}
        self._edges: Dict[Tuple[pp.Grid, pp.Grid], Dict] = {}
        self.name = "grid bucket"

    def __contains__(self, key: Any) -> bool:
        """Overload __contains__.

        Parameters:
            key (object): Object to be tested.

        Return:
            True if either key is a pp.Grid, and key is among the nodes of the graph
                representation of this md-grid, *or* key is a 2-tuple, with both items
                in self._nodes and the tuple is in self._edges.

        """
        if isinstance(key, pp.Grid):
            return key in self._nodes
        elif isinstance(key, tuple):
            if (
                len(key) == 2
                and isinstance(key[0], pp.Grid)
                and isinstance(key[1], pp.Grid)
            ):
                for edge, _ in self.edges():
                    if (edge[0] == key[0] and edge[1] == key[1]) or (
                        edge[1] == key[0] and edge[0] == key[1]
                    ):
                        return True
                # None of the edges were a match
                return False

        # Everything else is not in self.
        return False

    # --------- Iterators -------------------------

    def __iter__(self) -> Generator[Tuple[pp.Grid, Dict], None, None]:
        """Iterator over the nodes in the GridBucket.

        Yields:
            grid (pp.Grid): The grid associated with a node.
            data (dict): The dictionary storing all information in this node.

        """
        for grid, data in self._nodes.items():
            yield grid, data

    def nodes(self) -> Generator[Tuple[pp.Grid, Dict], None, None]:
        """Iterator over the nodes in the GridBucket.

        Identical functionality to self.__iter__(), but kept for consistency
        with self.edges()

        Yields:
            grid (pp.Grid): Grid associated with the current node.
            data (dict): The dictionary storing all information in this node.

        """
        for grid, data in self._nodes.items():
            yield grid, data

    def edges(self) -> Generator[Tuple[Tuple[pp.Grid, pp.Grid], Dict], None, None]:
        """
        Iterator over the edges in the GridBucket

        Yields:
            edge (Tuple[pp.Grid, pp.Grid]):
                Grid pair associated with the current edge.
            data (dict):
                The dictionary storing all information in this edge..

        """
        for edge, data in self._edges.items():
            yield edge, data

    # ---------- Navigate within the graph --------

    def nodes_of_edge(self, edge: Tuple[pp.Grid, pp.Grid]) -> Tuple[pp.Grid, pp.Grid]:
        """Obtain the vertices of an edge.

        The nodes will be given in ascending order with respect their
        dimension. If the edge is between grids of the same dimension, the node
        ordering (as defined by assign_node_ordering()) is used. If no ordering
        of nodes exists, assign_node_ordering() will be called by this method.

        Parameters:
            edge (Tuple[pp.Grid, pp.Grid]):
                An edge in the graph.

        Returns:
            pp.Grid: The first vertex of the edge.
            pp.Grid: The second vertex of the edge.

        """

        if edge[0].dim == edge[1].dim:
            if not (
                "node_number" in self._nodes[edge[0]]
                and "node_number" in self._nodes[edge[1]]
            ):
                self.assign_node_ordering()

            node_indexes = [self.node_props(grid, "node_number") for grid in edge]
            if node_indexes[0] < node_indexes[1]:
                return edge[0], edge[1]
            else:
                return edge[1], edge[0]

        elif edge[0].dim < edge[1].dim:
            return edge[0], edge[1]
        else:
            return edge[1], edge[0]

    def edges_of_node(
        self, node
    ) -> Generator[Tuple[Tuple[pp.Grid, pp.Grid], Dict], None, None]:
        """Iterator over the edges of the specific node.

        Parameters:
            node: A node in the graph.

        Yields:
            edge (Tuple[pp.Grid, pp.Grid]):
                The edge (pair of grids) associated with an edge.
            data (dict):
                The dictionary storing all information in this edge.

        """
        for edge, _ in self._edges.items():
            if edge[0] == node or edge[1] == node:
                yield edge, self.edge_props(edge)

    def node_neighbors(
        self, node: pp.Grid, only_higher: bool = False, only_lower: bool = False
    ) -> np.ndarray:
        """Get neighbors of a node in the graph.

        Optionally, return only neighbors corresponding to higher or lower
        dimension. At most one of only_higher and only_lower can be True.

        Parameters:
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
        for edge in self._edges.keys():
            if edge[0] == node:
                neigh.append(edge[1])
            elif edge[1] == node:
                neigh.append(edge[0])

        neigh = np.array(neigh)

        if not only_higher and not only_lower:
            return neigh
        elif only_higher and only_lower:
            raise ValueError("Cannot return both only higher and only lower")
        elif only_higher:
            # Find the neighbours that are higher dimensional
            is_high = np.array([w.dim > node.dim for w in neigh])
            return neigh[is_high]
        else:
            # Find the neighbours that are higher dimensional
            is_low = np.array([w.dim < node.dim for w in neigh])
            return neigh[is_low]

    # ------------ Getters for grids

    def get_grids(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """Obtain the grids, optionally filtered by a specified condition.

        Example:
        grid = self.gb.get_grids(lambda grid: grid.dim == dim)

        Parameters:
            cond: Predicate to select a grid. If None is given (default), all
                grids will be returned.

        Returns:
            grids: np.array of the grids.

        """
        if cond is None:
            cond = lambda grid: True

        return np.array([grid for grid, _ in self if cond(grid)])

    def grids_of_dimension(self, dim: int) -> np.ndarray:
        """Get all grids in the bucket of a specific dimension.

        Returns:
            grids (np.ndarray): Array of grids of the specified dimension

        """

        return self.get_grids(lambda grid: grid.dim == dim)

    def get_mortar_grids(
        self, cond: Callable[[pp.Grid], bool] = None, name: str = "mortar_grid"
    ) -> np.ndarray:
        """Obtain the mortar grids, optionally filtered by a specified condition.

        Example:
        grid = self.gb.get_mortar_grids(lambda grid: grid.dim == dim)

        Parameters:
            cond: Predicate to select a grid. If None is given (default), all
                grids will be returned.
            name (str, optional): key to identify mortar grids on edges.

        Returns:
            grids (np.ndarray): Array of the grids.

        """
        if cond is None:
            cond = lambda grid: True
        return np.array([data[name] for _, data in self.edges() if cond(data[name])])

    # ----------- Adders for node and edge properties (introduce keywords)

    def add_node_props(
        self,
        keys: Union[Any, List[Any]],
        grids: Union[pp.Grid, List[pp.Grid]] = None,
        overwrite: bool = True,
    ) -> None:
        """Add a new property to existing nodes in the graph.

        Properties can be added either to all nodes, or to selected nodes as
        specified by their grid. In the former case, all nodes will be assigned
        the property.

        The property gets value None, other values must be assigned later.

        By default, existing keys are overwritten.
        If existing keys should be kept, set overwrite to False.

        Parameters:
            keys (object or list of object): Key to the property to be handled.
            grids (list of pp.Grid, optional): Nodes to be assigned values.
                Defaults to None, in which case all nodes are assigned the same
                value.
            overwrite (bool, optional): Whether to overwrite existing keys.
                If false, any existing key on any grid will not be overwritten.

        Raises:
            ValueError if the key is 'node_number', this is reserved for other
                purposes. See self.assign_node_ordering() for details.

        """

        # Check that the key is not 'node_number' - this is reserved
        if "node_number" in keys:
            raise ValueError("Node number is a reserved key, stay away")

        # Do some checks of parameters first
        if grids is not None and not isinstance(grids, list):
            grids = [grids]
        if grids is None:
            grids = list(self._nodes)

        keys = [keys] if isinstance(keys, str) else list(keys)

        for key in keys:
            for grid in grids:
                data = self._nodes[grid]
                if overwrite or key not in data:
                    data[key] = None

    def add_edge_props(
        self,
        keys: Union[Any, List[Any]],
        edges: List[Tuple[pp.Grid, pp.Grid]] = None,
        overwrite: bool = True,
    ) -> None:
        """Associate a property with an edge.

        Properties can be added either to all edges, or to selected edges as
        specified by their grid pair. In the former case, all edges will be
        assigned the property, with None as the default option.

        By default, existing keys are overwritten.
        If existing keys should be kept, set overwrite to False.

        Parameters:
            keys (object or list of object): Key to the property to be handled.
            edges (list of 2-tuple of pp.Grid, optional): Grid pairs
                defining the edges to be assigned. values. Defaults to None, in
                which case all edges are assigned the same value.
            overwrite (bool, optional): Whether to overwrite existing keys.
                If false, any existing key on any grid-pair will not be overwritten.

        Raises:
            KeyError if a grid pair is not an existing edge in the grid.

        """
        keys = [keys] if isinstance(keys, str) else list(keys)
        if edges is None:
            edges = list(self._edges)

        for key in keys:
            for edge in edges:
                data = self.edge_props(edge)
                if overwrite or key not in data:
                    data[key] = None

    # ------------ Getters for node and edge properties

    def has_nodes_prop(self, grids: Iterable[pp.Grid], key: Any) -> List[bool]:
        """Test if a key exists in the data dictionary related to each of a list of nodes.

        Note: the property may contain None but the outcome of the test is
        still true.

        Parameters:
            grids (list of pp.Grid): The grids associated with the nodes.
            key (object): Key for the property to be tested.

        Returns:
            object: The tested property.

        """
        found = []
        for grid in grids:
            found.append(key in self._nodes[grid])
        return found

    def node_props(self, grid: pp.Grid, key: Any = None) -> Any:
        """Getter for a node property of the bucket.

        Parameters:
            grid (pp.Grid): The grid associated with the node.
            key (object, optional): Keys for the properties to be retrieved.
                If key is None (default) the entire data dictionary for the
                node is returned.

        Returns:
            object: A dictionary with keys and properties.

        """
        if key is None:
            return self._nodes[grid]
        else:
            return self._nodes[grid][key]

    def edge_props(self, edge: Tuple[pp.Grid, pp.Grid], key: Any = None) -> Dict:
        """Getter for an edge properties of the bucket.

        Parameters:
            edge (Tuple of pp.Grid): The two grids making up the edge.
            key (object, optional): Keys for the properties to be retrieved.
                If key is None (default) the entire data dictionary for the
                edge is returned.

        Returns:
            object: The properties.

        Raises:
            KeyError if the two grids do not form an edge.

        """
        if tuple(edge) in list(self._edges.keys()):
            if key is None:
                return self._edges[edge]
            else:
                return self._edges[edge][key]
        elif tuple(edge[::-1]) in list(self._edges.keys()):
            if key is None:
                return self._edges[(edge[1], edge[0])]
            else:
                return self._edges[(edge[1], edge[0])][key]
        else:
            raise KeyError("Unknown edge")

    # ------------- Setters for edge and grid properties

    def set_node_prop(self, grid: pp.Grid, key: Any, val: Any) -> None:
        """Set the value of a property of a given node.

        Values can also be set by accessing the data dictionary of the node
        directly.

        No checks are performed on whether the property has been added to
        the node. If the property has not been added, it will implicitly be
        generated.

        Parameters:
            grid (pp.Grid): Grid identifying the node.
            key (object): Key identifying the field to add.
            val (Any): Value to be added.

        """
        self._nodes[grid][key] = val

    def set_edge_prop(self, edge: Tuple[pp.Grid, pp.Grid], key: Any, val: Any) -> None:
        """Set the value of a property of a given edge.

        Values can also be set by accessing the data dictionary of the edge
        directly.

        No checks are performed on whether the property has been added to
        the edge. If the property has not been added, it will implicitly be
        generated.

        Parameters:
            edge (2-tuple of grids): Grid pair identifying the edge.
            key (object): Key identifying the field to add.
            val (Any): Value to be added.

        Raises:
            KeyError if the two grids do not form an edge.

        """
        if tuple(edge) in list(self._edges.keys()):
            self._edges[(edge[0], edge[1])][key] = val
        elif tuple(edge[::-1]) in list(self._edges.keys()):
            self._edges[(edge[1], edge[0])][key] = val

        else:
            raise KeyError("Unknown edge")

    # ------------ Removers for nodes properties ----------

    def remove_node_props(
        self, keys: Union[Any, List[Any]], grids: Union[pp.Grid, List[pp.Grid]] = None
    ) -> None:
        """Remove property to existing nodes in the graph.

        Properties can be removed either to all nodes, or to selected nodes as
        specified by their grid. In the former case, to all nodes the property
        will be removed.

        Parameters:
            keys (object or list of object): Key to the property to be handled.
            grids (list of pp.Grid, optional): Nodes to be removed the values.
                Defaults to None, in which case the property is removed from
                all nodes.

        Raises:
            ValueError if the key is 'node_number', this is reserved for other
                purposes. See self.assign_node_ordering() for details.

        """

        # Check that the key is not 'node_number' - this is reserved
        if "node_number" in keys:
            raise ValueError("Node number is a reserved key, stay away")

        # Do some checks of parameters first
        if grids is not None and not isinstance(grids, list):
            grids = [grids]
        if grids is None:
            grids = list(self._nodes)

        keys = [keys] if isinstance(keys, str) else list(keys)

        for key in keys:
            for grid in grids:
                data = self._nodes[grid]
                if key in data:
                    del data[key]

    def remove_edge_props(
        self,
        keys: Union[Any, List[Any]],
        edges: Optional[
            Union[Tuple[pp.Grid, pp.Grid], List[Tuple[pp.Grid, pp.Grid]]]
        ] = None,
    ) -> None:
        """
        Remove property to existing edges in the graph.

        Properties can be removed either to all edges, or to selected edges as
        specified by their grid pair. In the former case, to all edges the property
        will be removed.

        Parameters:
            keys (object or list of objects): Key to the property to be handled.
            edges (List of tuples of pp.Grid, optional): Edges to be removed the
                values. Defaults to None, in which case the property is removed
                from all edges.

        Raises:
            ValueError if the key is 'edge_number', this is reserved for other
                purposes. See self.assign_node_ordering() for details.

        """
        keys = list(keys)

        # Check that the key is not 'edge_number' - this is reserved
        if "edge_number" in keys:
            raise ValueError("Edge number is a reserved key, stay away")

        # Check if a single edge is passed. If so, turn it into a list.
        if edges is None:
            edges_processed: List[Tuple[pp.Grid, pp.Grid]] = list(self._edges)
        elif (
            isinstance(edges, tuple)
            and len(edges) == 2
            and all(isinstance(e, pp.Grid) for e in edges)
        ):
            edges_processed = [edges]
        else:
            edges_processed = edges  # type: ignore

        for key in keys:
            for edge in edges_processed:
                data = self.edge_props(edge)
                if key in data:
                    del data[key]

    # ------------ Add new nodes and edges ----------

    def add_nodes(self, new_grids: Union[pp.Grid, Iterable[pp.Grid]]) -> None:
        """
        Add grids to the hierarchy.

        The grids are added to self.grids.

        Parameters:
            new_grids (pp.Grid or Iterable of pp.Grid): The grids to be added. None of
                these should have been added previously.

        Raises:
            ValueError if a grid is already present in the bucket

        """
        if isinstance(new_grids, pp.Grid):
            new_grids = [new_grids]

        if np.any([i is j for i in new_grids for j in list(self._nodes.keys())]):
            raise ValueError("Grid already defined in bucket")
        for grid in new_grids:
            self._nodes[grid] = {}

    def add_edge(self, grids: List[pp.Grid], face_cells: sps.spmatrix) -> None:
        """
        Add an edge in the graph.

        If the grids have different dimensions, the coupling will be added with
        the higher-dimensional grid as the first node. For equal dimensions,
        the ordering of the nodes is the same as in input grids.

        Parameters:
            grids (list, len==2). Grids to be connected. Order is arbitrary.
            face_cells (sps.spmatrix): Identity mapping between cells in the
                higher-dimensional grid and faces in the lower-dimensional
                grid. No assumptions are made on the type of the object at this
                stage. In the grids[0].dim = grids[1].dim case, the mapping is
                from faces of the first grid to faces of the second one.

        Raises:
            ValueError if the edge is not specified by exactly two grids
            ValueError if the edge already exists.
            ValueError if the two grids are not either one dimension apart, or of the
                same dimension.

        """
        if len(grids) != 2:
            raise ValueError("An edge should be specified by exactly two grids")

        if tuple(grids) in list(self._edges.keys()) or tuple(grids[::-1]) in list(
            self._edges.keys()
        ):
            raise ValueError("Cannot add existing edge")

        data = {"face_cells": face_cells}

        # The higher-dimensional grid is the first node of the edge.
        if grids[0].dim - 1 == grids[1].dim:
            self._edges[(grids[0], grids[1])] = data
        elif grids[0].dim == grids[1].dim - 1:
            self._edges[(grids[1], grids[0])] = data
        elif grids[0].dim == grids[1].dim:
            self._edges[(grids[0], grids[1])] = data
        else:
            raise ValueError("Grid dimension mismatch")

    # --------- Remove and update nodes

    def remove_node(self, node: pp.Grid) -> None:
        """
        Remove node, and related edges, from the grid bucket.

        Parameters:
           node : the node to be removed

        """

        del self._nodes[node]

        edges_to_remove = []
        for edge, _ in self.edges():
            if edge[0] == node or edge[1] == node:
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            del self._edges[edge]

    def remove_nodes(self, cond: Callable[[pp.Grid], bool]) -> None:
        """
        Remove nodes, and related edges, from the grid bucket subject to a
        conditions. The latter takes as input a grid.

        Parameters:
            cond: predicate to select the grids to remove.

        """
        if cond is None:
            cond = lambda g: True

        edges_to_remove = []

        for grid in self._nodes.keys():
            if cond(grid):
                del self._nodes[grid]
                for edge in self._edges.keys():
                    if edge[0] == grid or edge[1] == grid:
                        edges_to_remove.append(edge)

        for edge in edges_to_remove:
            del self._edges[edge]

    def update_nodes(self, mapping: Dict[pp.Grid, pp.Grid]) -> None:
        """
        Update the grids giving old and new values. The edges are updated
        accordingly.

        Parameters:
            mapping: A dictionary with the old grid as keys and new grid as values.

        """
        for old, new in mapping.items():
            data = self._nodes[old]
            self._nodes[new] = data
            del self._nodes[old]
            new_dict = {}

            for edge, data in self._edges.items():
                if edge[0] == old:
                    new_dict[(new, edge[1])] = data
                elif edge[1] == old:
                    new_dict[(edge[0], new)] = data
                else:
                    new_dict[edge] = data

            self._edges = new_dict

    def eliminate_node(self, node: pp.Grid) -> List[pp.Grid]:
        """
        Remove the node (and the edges it partakes in) and add new direct
        connections (gb edges) between each of the neighbor pairs. A node with
        n_neighbors neighbours gives rise to 1 + 2 + ... + n_neighbors-1 new edges.

        """
        # Identify neighbors
        neighbors: List[pp.Grid] = self.sort_multiple_nodes(self.node_neighbors(node))

        n_neighbors = len(neighbors)

        # Add an edge between each neighbor pair
        for i in range(n_neighbors - 1):
            g0 = neighbors[i]
            for j in range(i + 1, n_neighbors):
                g1 = neighbors[j]
                cell_cells = self._find_shared_face(g0, g1, node)
                self.add_edge([g0, g1], cell_cells)

        # Remove the node and update the ordering of the remaining nodes
        node_number = self.node_props(node, "node_number")
        self.remove_node(node)
        self.update_node_ordering(node_number)

        return neighbors

    def duplicate_without_dimension(
        self, dim: int
    ) -> Tuple["GridBucket", Dict[str, Dict]]:
        """
        Remove all the nodes of dimension dim and add new edges between their
        neighbors by calls to remove_node.

        Parameters:
            dim (int): Dimension for which all grids should be removed.

        Returns:
            pp.GridBucket: Copy of this GridBucket, with all grids of dimension dim
                removed, and new edges between the neighbors of removed grids.
            Dict: Information on removed grids. Keys:
                "eliminated_nodes": List of all grids that were removed.
                "neighbours": List with neighbors of the eliminated grids. Sorted in
                    the same order as eliminated_nodes. Node ordering is updated.
                "neigbours_old": Same as neighbours, but with the original node
                    ordering.

        """

        gb_copy = self.copy()
        grids_of_dim = gb_copy.grids_of_dimension(dim)
        grids_of_dim_old = self.grids_of_dimension(dim)
        # The node numbers are copied for each grid, so they can be used to
        # make sure we use the same grids (g and g_old) below.
        nn_new = [gb_copy.node_props(g, "node_number") for g in grids_of_dim]
        nn_old = [self.node_props(g, "node_number") for g in grids_of_dim_old]
        _, old_in_new = setmembership.ismember_rows(
            np.array(nn_new), np.array(nn_old), sort=False
        )
        neighbours_dict = {}
        neighbours_dict_old = {}
        eliminated_nodes = {}
        for i, g in enumerate(grids_of_dim):
            # Eliminate the node and add new gb edges:
            neighbours = gb_copy.eliminate_node(g)
            # Keep track of which nodes were connected to each of the eliminated
            # nodes. Note that the key is the node number in the old gb, whereas
            # the neighbours lists refer to the copy grids.
            g_old = grids_of_dim_old[old_in_new[i]]
            neighbours_dict[i] = neighbours
            neighbours_dict_old[i] = self.node_neighbors(g_old)
            eliminated_nodes[i] = g_old

        elimination_data = {
            "neighbours": neighbours_dict,
            "neighbours_old": neighbours_dict_old,
            "eliminated_nodes": eliminated_nodes,
        }

        return gb_copy, elimination_data  # type: ignore

    # ---------- Functionality related to ordering of nodes

    def assign_node_ordering(self, overwrite_existing: bool = True) -> None:
        """
        Assign an ordering of the nodes in the graph, stored as the attribute
        'node_number'.

        The ordering starts with grids of highest dimension. The ordering
        within each dimension is determined by an iterator over the graph, and
        can in principle change between two calls to this function.

        If an ordering covering all nodes in the graph already exist, but does
        not coincide with the new ordering, a warning is issued. If the optional
        parameter overwrite_existing is set to False, no update is performed if
        an node ordering already exists.

        Parameters:
            overwrite_existing (bool, optional): If True (default), any existing node
                ordering will be overwritten.

        """

        # Check whether 'node_number' is defined for the grids already.
        ordering_exists = True
        for _, data in self:
            if "node_number" not in data.keys():
                ordering_exists = False
        if ordering_exists and not overwrite_existing:
            return

        counter = 0
        # Loop over grids in decreasing dimensions
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            for grid in self.grids_of_dimension(dim):
                data = self._nodes[grid]
                # Get old value, issue warning if not equal to the new one.
                num = data.get("node_number", -1)
                if ordering_exists and num != counter:
                    warnings.warn("Order of graph nodes has changed")
                # Assign new value
                data["node_number"] = counter
                counter += 1

        self.add_edge_props("edge_number")
        counter = 0
        for _, data in self.edges():
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
            for g in self.grids_of_dimension(dim):
                if not self.has_nodes_prop([g], "node_number"):
                    # It is not clear how severe this case is. For the moment,
                    # we give a warning, and hope the user knows what to do
                    warnings.warn("Tried to update node ordering where none exists")
                    # No point in continuing with this node.
                    continue

                # Obtain the old node number
                n = self._nodes[g]
                old_number = n.get("node_number", -1)
                # And replace it if it is higher than the removed one
                if old_number > removed_number:
                    n["node_number"] = old_number - 1

    def sort_multiple_nodes(self, nodes: List[pp.Grid]) -> List[pp.Grid]:
        """
        Sort all the nodes according to node number.

        Parameters:
            nodes: List of graph nodes.

        Returns:
            sorted_nodes: The same nodes, sorted.

        """
        assert self.has_nodes_prop(nodes, "node_number")

        return sorted(nodes, key=lambda n: self.node_props(n, "node_number"))

    # ------------- Miscellaneous functions ---------

    def target_2_source_nodes(self, g_src, g_trg):
        """
        Find the local node mapping from a source grid to a target grid.

        target_2_source_nodes(..) returns the mapping from g_src -> g_trg such
        that g_trg.nodes[:, map] == g_src.nodes. E.g., if the target grid is the
        highest dim grid, target_2_source_nodes will equal the global node
        numbering.

        """
        node_source = np.atleast_2d(g_src.global_point_ind)
        node_target = np.atleast_2d(g_trg.global_point_ind)
        _, trg_2_src_nodes = setmembership.ismember_rows(
            node_source.astype(np.int32), node_target.astype(np.int32)
        )
        return trg_2_src_nodes

    def compute_geometry(self) -> None:
        """Compute geometric quantities for the grids."""
        for grid, _ in self:
            grid.compute_geometry()

        for _, data in self.edges():
            if "mortar_grid" in data.keys():
                data["mortar_grid"].compute_geometry()

    def copy(self) -> "GridBucket":
        """Make a shallow copy of the grid bucket. The underlying grids are not copied.

        Return:
            pp.GridBucket: Copy of this GridBucket.

        """
        gb_copy: pp.GridBucket = GridBucket()
        gb_copy._nodes = self._nodes.copy()
        gb_copy._edges = self._edges.copy()
        return gb_copy

    def replace_grids(
        self,
        g_map: Optional[Dict[pp.Grid, pp.Grid]] = None,
        mg_map: Optional[Dict[pp.MortarGrid, Dict[int, pp.Grid]]] = None,
        tol: float = 1e-6,
    ) -> None:
        """Replace grids and / or mortar grids in the mixed-dimensional grid.

        Parameters:
            gb (GridBucket): To be updated.
            g_map (dictionary): Mapping between the old and new grids. Keys are
                the old grids, and values are the new grids.
            mg_map (dictionary): Mapping between the old mortar grid and new
                (side grids of the) mortar grid. Keys are the old mortar grids,
                and values are dictionaries with side grid number as keys, and
                side grids as values.

        """
        if mg_map is None:
            mg_map = {}

        # refine the mortar grids when specified
        for mg_old, mg_new in mg_map.items():
            mg_old.update_mortar(mg_new, tol)

        # update the grid bucket considering the new grids instead of the old one
        # valid only for physical grids and not for mortar grids
        if g_map is not None:
            self.update_nodes(g_map)
        else:
            g_map = {}

        # refine the grids when specified
        for g_old, g_new in g_map.items():
            for _, d in self.edges_of_node(g_new):
                mg = d["mortar_grid"]
                if mg.dim == g_new.dim:
                    # update the mortar grid of the same dimension
                    mg.update_secondary(g_new, tol)
                else:  # g_new.dim == mg.dim + 1
                    mg.update_primary(g_new, g_old, tol)

    def _find_shared_face(self, g0: pp.Grid, g1: pp.Grid, g_l: pp.Grid) -> np.ndarray:
        """
        Given two nd grids meeting at a (n-1)d node (to be removed), find which two
        faces meet at the intersection (one from each grid) and build the connection
        matrix cell_cells.
        In face_cells, the cells of the lower dimensional grid correspond to the first
        axis and the faces of the higher dimensional grid to the second axis.
        Furthermore, grids are sorted with the lower-dimensional first and the
        higher-dimesnional last. To be consistent with this, the grid corresponding
        to the first axis of cell_cells is the first grid of the node sorting.

        Parameters:
            g0 and g1: The two nd grids.
            g_l: The (n-1)d grid to be removed.
        Returns: The np array cell_cells (g0.num_cells x g1.num_cells), the nd-nd
            equivalent of the face_cells matrix. Cell_cells identifies connections
            ("faces") between all permutations of cells of the two grids which were
            initially connected to the lower_dim_node.
            Example: g_l is connected to cells 0 and 1 of g0 (vertical)
            and 1 and 2 of g1 (horizontal) as follows:

                |
            _ _ . _ _
                |

            Cell_cells: [ [ 0, 1, 1, 0],
                          [ 0, 1, 1, 0] ]
            connects both cell 0 and 1 of g0 (read along first dimension) to
            cells 1 and 2 of g1 (second dimension of the array).
        """
        # Sort nodes according to node_number
        g0, g1 = self.nodes_of_edge((g0, g1))

        # Identify the faces connecting the neighbors to the grid to be removed
        fc1 = self.edge_props((g0, g_l))
        fc2 = self.edge_props((g1, g_l))
        _, faces_1 = fc1["face_cells"].nonzero()
        _, faces_2 = fc2["face_cells"].nonzero()
        # Extract the corresponding cells
        _, cells_1 = g0.cell_faces[faces_1].nonzero()
        _, cells_2 = g1.cell_faces[faces_2].nonzero()

        # Connect the two remaining grid through the cell_cells matrix,
        # to be placed as a face_cells substitute. The ordering is consistent
        # with the face_cells wrt the sorting of the nodes.
        cell_cells = np.zeros((g0.num_cells, g1.num_cells), dtype=bool)
        rows = np.tile(cells_1, (cells_2.size, 1))
        cols = np.tile(cells_2, (cells_1.size, 1)).T
        cell_cells[rows, cols] = True

        return cell_cells

    # ----------- Apply functions to nodes and edges

    def apply_function_to_nodes(
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
        values = np.empty(self.num_graph_nodes())
        for g, d in self:
            values[d["node_number"]] = fct(g, d)
        return values

    def apply_function_to_edges(
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
        i = np.zeros(len(self._edges), dtype=int)
        j = np.zeros(i.size, dtype=int)
        values = np.zeros(i.size)

        # Loop over the edges of the graph (pair of connected nodes)
        idx = 0
        for e, data in self.edges():
            g_l, g_h = self.nodes_of_edge(e)
            data_l, data_h = self.node_props(g_l), self.node_props(g_h)

            i[idx] = self.node_props(g_l, "node_number")
            j[idx] = self.node_props(g_h, "node_number")
            values[idx] = fct(g_h, g_l, data_h, data_l, data)
            idx += 1

        # Upper triangular matrix
        return sps.coo_matrix(
            (values, (i, j)), (self.num_graph_nodes(), self.num_graph_nodes())
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
        diam_g = [np.amax(g.cell_diameters()) for g in self._nodes.keys() if cond(g)]

        diam_mg = [
            np.amax(d["mortar_grid"].cell_diameters())
            for e, d in self.edges()
            if cond(e) and d.get("mortar_grid")
        ]

        return np.amax(np.hstack((diam_g, diam_mg)))

    def bounding_box(
        self, as_dict: bool = False
    ) -> Union[Dict[str, float], Tuple[float, float]]:
        """
        Return the bounding box of the grid bucket.
        """
        c_0s = np.empty((3, self.num_graph_nodes()))
        c_1s = np.empty((3, self.num_graph_nodes()))

        for i, grid in enumerate(self._nodes.keys()):
            c_0s[:, i], c_1s[:, i] = grid.bounding_box()

        min_vals = np.amin(c_0s, axis=1)
        max_vals = np.amax(c_1s, axis=1)

        if as_dict:
            return {
                "xmin": min_vals[0],
                "xmax": max_vals[0],
                "ymin": min_vals[1],
                "ymax": max_vals[1],
                "zmin": min_vals[2],
                "zmax": max_vals[2],
            }
        else:
            return min_vals, max_vals

    def size(self) -> int:
        """
        Returns:
            int: Number of mono-dimensional grids and interfaces in the bucket.

        """
        return self.num_graph_nodes() + self.num_graph_edges()

    def dim_min(self) -> int:
        """
        Returns:
            int: Minimum dimension of the grids present in the hierarchy.

        """
        return np.amin([grid.dim for grid, _ in self])

    def dim_max(self) -> int:
        """
        Returns:
            int: Maximum dimension of the grids present in the hierarchy.

        """
        return np.amax([grid.dim for grid, _ in self])

    def all_dims(self) -> np.array:
        """
        Returns:
            np.array: Active dimensions of the grids present in the hierarchy.

        """
        return np.unique([grid.dim for grid, _ in self])

    def cell_volumes(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """
        Get the cell volumes of all cells of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            cell_volumes (np.ndarray): The volume of all cells in the GridBucket

        """
        if cond is None:
            cond = lambda g: True
        return np.hstack(
            [grid.cell_volumes for grid, _ in self._nodes.items() if cond(grid)]
        )

    def face_centers(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """
        Get the face centers of all faces of the grid bucket, considering a loop
        on all the graph nodes.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            face_centers (ndArray): The face centers of all faces in the GridBucket.
        """
        if cond is None:
            cond = lambda g: True
        return np.hstack(
            [grid.face_centers for grid, _ in self._nodes.items() if cond(grid)]
        )

    def cell_centers(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """
        Get the cell centers of all cells of the grid bucket, considering a loop
        on all the graph nodes.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            cell_centers (ndArray): The cell centers of all cells in the GridBucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.hstack(
            [grid.cell_centers for grid, _ in self._nodes.items() if cond(grid)]
        )

    def cell_volumes_mortar(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """
        Get the cell volumes of all mortar cellse of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            cell_volumes (np.ndarray): The cell volumes of all mortar cells in the
                GridBucket.

        """
        if cond is None:
            cond = lambda g: True
        if self.num_mortar_cells(cond) == 0:
            return np.array([])
        return np.hstack(
            [
                data["mortar_grid"].cell_volumes
                for edge, data in self.edges()
                if cond(data["mortar_grid"])
            ]
        )

    def num_cells(self, cond: Callable[[pp.Grid], bool] = None) -> int:
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
        return np.sum(
            [grid.num_cells for grid in self._nodes.keys() if cond(grid)], dtype=np.int
        )

    def num_mortar_cells(self, cond: Callable[[pp.Grid], bool] = None) -> int:
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
        return np.sum(
            [
                data["mortar_grid"].num_cells
                for _, data in self.edges()
                if data.get("mortar_grid") and cond(data["mortar_grid"])
            ],
            dtype=np.int,
        )

    def num_faces(self, cond: Callable[[pp.Grid], bool] = None) -> int:
        """
        Compute the total number of faces of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            num_faces (int): the total number of faces of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.sum([grid.num_faces for grid in self._nodes.keys() if cond(grid)])

    def num_nodes(self, cond: Callable[[pp.Grid], bool] = None) -> int:
        """
        Compute the total number of nodes of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            num_nodes (int): the total number of nodes of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.sum([grid.num_nodes for grid in self._nodes.keys() if cond(grid)])

    def num_graph_nodes(self):
        """
        Return the total number of nodes (physical meshes) in the graph.

        Returns:
            int: Number of nodes in the graph.

        """
        return len(self._nodes)

    def num_graph_edges(self) -> int:
        """
        Return the total number of edge in the graph.

        Returns:
            int: Number of edges in the graph.

        """
        return len(self._edges)

    def num_nodes_edges(self) -> int:
        """
        Return the total number of nodes (physical meshes) plus the total number
        of edges in the graph. It is the size of the graph.

        Returns:
            int: Number of nodes and edges in the graph.

        """
        return self.num_graph_nodes() + self.num_graph_edges()

    def __str__(self) -> str:
        max_dim = self.grids_of_dimension(self.dim_max())
        num_nodes = 0
        num_cells = 0
        for g in max_dim:
            num_nodes += g.num_nodes
            num_cells += g.num_cells
        s = (
            "Mixed dimensional grid. \n"
            f"Maximum dimension present: {self.dim_max()} \n"
            f"Minimum dimension present: {self.dim_min()} \n"
        )
        s += "Size of highest dimensional grid: Cells: " + str(num_cells)
        s += ". Nodes: " + str(num_nodes) + "\n"
        s += "In lower dimensions: \n"
        for dim in range(self.dim_max() - 1, self.dim_min() - 1, -1):
            gl = self.grids_of_dimension(dim)
            num_nodes = 0
            num_cells = 0
            for g in gl:
                num_nodes += g.num_nodes
                num_cells += g.num_cells

            s += (
                f"{len(gl)} grids of dimension {dim}, with in total "
                f"{num_cells} cells and {num_nodes} nodes. \n"
            )

        s += f"Total number of interfaces: {self.num_graph_edges()}\n"
        for dim in range(self.dim_max(), self.dim_min(), -1):
            num_e = 0
            for e, _ in self.edges():
                if e[0].dim == dim:
                    num_e += 1

            s += f"{num_e} interfaces between grids of dimension {dim} and {dim-1}\n"
        return s

    def __repr__(self) -> str:
        s = (
            f"Grid bucket containing {self.num_graph_nodes() } grids and "
            f"{self.num_graph_edges()} interfaces\n"
            f"Maximum dimension present: {self.dim_max()} \n"
            f"Minimum dimension present: {self.dim_min()} \n"
        )

        if self.num_graph_nodes() > 0:
            for dim in range(self.dim_max(), self.dim_min() - 1, -1):
                gl = self.grids_of_dimension(dim)
                nc = 0
                for g in gl:
                    nc += g.num_cells
                s += (
                    f"{len(gl)} grids of dimension {dim}" f" with in total {nc} cells\n"
                )
        if self.num_graph_edges() > 0:
            for dim in range(self.dim_max(), self.dim_min(), -1):
                num_e = 0
                num_mc = 0
                for e, d in self.edges():
                    if e[0].dim == dim:
                        num_e += 1
                        num_mc += d["mortar_grid"].num_cells

                s += (
                    f"{num_e} interfaces between grids of dimension {dim} and {dim-1}"
                    f" with in total {num_mc} mortar cells\n"
                )

        return s
