"""
Module to store the grid hierarchy formed by a set of fractures and their
intersections in the form of a GridTree.

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


class GridTree:
    """
    Container for the hierarchy of grids formed by fractures and their
    intersections.

    The mixed-dimensional grid can be considered a graph, with the fixed-dimensional grids
    defining nodes, and interfaces edges. The terminology in the implementation,
    and in method names, are strongly marked by this graph thinking.

    Connections between the grids are defined as a tuple, with first the highest
    and then the lowest-dimensional neighboring grid.

    Attributes:
        _nodes (dictionary): For each subdomain grid, a dictionary with
            grids as keys, and data associated with the grid as values.
        _edges (dictionary): For each interface grid, a dictionary with
            the edges as keys, and data associated with the grid as values.

    """

    def __init__(self) -> None:
        self._nodes: Dict[pp.Grid, Dict] = {}
        self._edge_data: Dict[pp.MortarGrid, Dict] = {}
        self._edge_to_node: Dict[pp.MortarGrid, Tuple[pp.Grid, pp.Grid]]
        self.name = "Grid Tree"

    def __contains__(self, key: Any) -> bool:
        """Overload __contains__.

        Parameters:
            key (object): Object to be tested.

        Return:
            True if either key is a pp.Grid, and key is among the nodes of the graph
                representation of this md-grid, *or* key is a 2-tuple, with both items
                in self._nodes and the tuple is in self._edge_data.

        """
        if isinstance(key, pp.Grid):
            return key in self._nodes
        elif isinstance(key, pp.MortarGrid):
            return key in self._edge_data

        # Everything else is not in self.
        return False

    # --------- Iterators -------------------------

    def subdomains(self, dim: Optional[int] = None) -> Generator[pp.Grid, None, None]:
        """Iterator over the subdomains in the GridTree.

        Optionally, the iterator can filter subdomains on dimension.

        Parameters:
            dim (int, optional): If provided, only subdomains of the specified dimension
                will be returned.

        Yields:
            grid (pp.Grid): Grid associated with the current node.

        """
        for grid in self._nodes:
            # Filter on dimension if requested.
            if dim is not None and dim != grid.dim:
                continue

            yield grid

    def interfaces(
        self, dim: Optional[int] = None
    ) -> Generator[pp.MortarGrid, None, None]:
        """
        Iterator over the MortarGrids belonging to interfaces in the GridTree.

        Optionally, the iterator can filter interfaces on dimension.

        Parameters:
            dim (int, optional): If provided, only interfaces  of the specified
                dimension will be returned.

        Yields:
            interface (pp.MortarGrid):

        """
        for edge in self._edge_data:
            # Filter on dimension if requested.
            if dim is not None and dim != edge.dim:
                continue

            yield edge

    # ---------- Navigate within the graph --------

    def subdomains_of_interface(self, mg: pp.MortarGrid) -> Tuple[pp.Grid, pp.Grid]:
        """Obtain the subdomains of an interface.

        FIXME: Consider introducing only_higher, only_lower keywords.

        The nodes will be given in ascending order with respect their
        dimension. If the edge is between grids of the same dimension, the node
        ordering (as defined by assign_node_ordering()) is used. If no ordering
        of nodes exists, assign_node_ordering() will be called by this method.

        Parameters:
            mg, pp.MortarGrid:
                An edge in the graph.

        Returns:
            pp.Grid: The first vertex of the edge.
            pp.Grid: The second vertex of the edge.

        """
        edge = self._edge_to_node[mg]
        if edge[0].dim == edge[1].dim:
            if not (
                "node_number" in self._nodes[edge[0]]
                and "node_number" in self._nodes[edge[1]]
            ):
                self.assign_node_ordering()

            node_indexes = [self.subdomain_data(grid, "node_number") for grid in edge]
            if node_indexes[0] < node_indexes[1]:
                return edge[0], edge[1]
            else:
                return edge[1], edge[0]

        elif edge[0].dim < edge[1].dim:
            return edge[0], edge[1]
        else:
            return edge[1], edge[0]

    def subdomain_pair_to_interface(
        self, node_pair: Tuple[pp.Grid, pp.Grid]
    ) -> pp.MortarGrid:
        reverse_dict = {v: k for k, v in self._edge_to_node.items()}
        if node_pair in reverse_dict:
            return reverse_dict[node_pair]
        elif node_pair[::-1] in reverse_dict:
            return reverse_dict[node_pair[::-1]]
        else:
            raise KeyError("Unknown subdomain pair")

    def interfaces_of_subdomain(
        self, node: pp.Grid
    ) -> Generator[pp.MortarGrid, None, None]:
        """Iterator over the edges of the specific node.

        FIXME: Consider introducing only_higher, only_lower keywords.

        Parameters:
            node: A node in the graph.

        Yields:
            edge (Tuple[pp.Grid, pp.Grid]):
                The edge (pair of grids) associated with an edge.

        """
        for edge in self._edge_data:
            node_pair = self._edge_to_node[edge]
            if node_pair[0] == node or node_pair[1] == node:
                yield edge

    def neighboring_subdomains(
        self, node: pp.Grid, only_higher: bool = False, only_lower: bool = False
    ) -> List[pp.Grid]:
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
        for mg, edge in self._edge_to_node.items():
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

    # ----------- Adders for node and edge properties (introduce keywords)

    def add_subdomain_properties(
        self,
        keys: Union[Any, List[Any]],
        grids: Union[pp.Grid, List[pp.Grid]] = None,
        overwrite: bool = True,
    ) -> None:
        """Add a new property to existing nodes in the graph.

        FIXME: Update documentation

        FIXME: Should we delete this

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

    def add_interface_properties(
        self,
        keys: Union[Any, List[Any]],
        edges: List[Tuple[pp.Grid, pp.Grid]] = None,
        overwrite: bool = True,
    ) -> None:
        """Associate a property with an edge.

        FIXME: Should we delete this?

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
                data = self.interface_data(edge)
                if overwrite or key not in data:
                    data[key] = None

    # ------------ Getters for node and edge properties

    def is_property_of_subdomain(
        self, grids: Iterable[pp.Grid], key: Any
    ) -> List[bool]:
        """Test if a key exists in the data dictionary related to each of a list of nodes.

        Note: the property may contain None but the outcome of the test is
        still true.

        FIXME: This is not used in the core of porepy. Delete?

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

    def subdomain_data(self, grid: pp.Grid, key: Any = None) -> Any:
        """Getter for a subdomain property of the GridTree.

        FIXME: Should we delete the key and simply return the full data dictionary?

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

    def interface_data(self, edge: pp.MortarGrid, key: Any = None) -> Dict:
        """Getter for an edge properties of the bucket.

        FIXME: Should we delete the key and simply return the full data dictionary?

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
        if edge in self._edge_data:
            if key is None:
                return self._edge_data[edge]
            else:
                return self._edge_data[edge][key]

        else:
            raise KeyError("Unknown edge")

    # ------------- Setters for edge and grid properties

    def set_subdomain_property(self, grid: pp.Grid, key: Any, val: Any) -> None:
        """Set the value of a property of a given subdomain.

        Values can also be set by accessing the data dictionary of the subdomain
        directly.

        FIXME: Delete this one?

        No checks are performed on whether the property has been added to
        the subdomain. If the property has not been added, it will implicitly be
        generated.

        Parameters:
            grid (pp.Grid): Grid identifying the subdomain.
            key (object): Key identifying the field to add.
            val (Any): Value to be added.

        """
        self._nodes[grid][key] = val

    def set_interface_property(
        self, edge: Tuple[pp.Grid, pp.Grid], key: Any, val: Any
    ) -> None:
        """Set the value of a property of a given interface.

        Values can also be set by accessing the data dictionary of the interface
        directly.

        No checks are performed on whether the property has been added to
        the edge. If the property has not been added, it will implicitly be
        generated.

        FIXME: Can we delete this?

        Parameters:
            edge (2-tuple of grids): Grid pair identifying the interface.
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

    def remove_subdomain_property(
        self, keys: Union[Any, List[Any]], grids: Union[pp.Grid, List[pp.Grid]] = None
    ) -> None:
        """Remove property to existing subdomain in the GridTree.

        FIXME: Property or properties? Applies to several methods.

        FIXME: Can we delete this?

        Properties can be removed either to all subdomains, or to selected
        subdomains as specified by their grid. In the former case, to all nodes
        the property will be removed.

        Parameters:
            keys (object or list of object): Key to the property to be handled.
            grids (list of pp.Grid, optional): Subdomains to have the values
                removed. Defaults to None, in which case the property is
                removed from all nodes.

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

    def remove_interface_property(
        self,
        keys: Union[Any, List[Any]],
        edges: Optional[Union[Tuple[pp.Grid, pp.Grid]]] = None,
    ) -> None:
        """
        Remove property to existing edges in the graph.

        FIXME: Can we delete this?

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
                data = self.interface_data(edge)
                if key in data:
                    del data[key]

    # ------------ Add new nodes and edges ----------

    def add_subdomains(self, new_grids: Union[pp.Grid, Iterable[pp.Grid]]) -> None:
        """
        Add grids to the hierarchy.

        Parameters:
            new_grids (pp.Grid or Iterable of pp.Grid): The grids to be added. None of
                these should have been added previously.

        Raises:
            ValueError if a grid is already present in the bucket

        """
        if isinstance(new_grids, pp.Grid):
            ng = [new_grids]
        else:
            ng = new_grids  # type: ignore

        if any([i is j for i in ng for j in list(self._nodes.keys())]):
            raise ValueError("Grid already defined in bucket")

        for grid in ng:
            # Add the grid to the list of subdomains with an empty data dictionary.
            self._nodes[grid] = {}

    def add_interface(
        self,
        mg: pp.MortarGrid,
        grids: Tuple[pp.Grid, pp.Grid],
        primary_secondary_map: sps.spmatrix,
    ) -> None:
        """
        Add an interface to the GridTree.

        If the grids have different dimensions, the coupling will be added with
        the higher-dimensional grid as the first node. For equal dimensions,
        the ordering of the nodes is the same as in input grids.

        Parameters:
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

        if mg in self._edge_data:
            raise ValueError("Cannot add existing edge")

        # The data dictionary is initialized with a single field, defining the mapping
        # between grid quantities in the primary and secondary neighboring subdomains
        # (for a co-dimension 1 coupling, this will be between faces in the
        # higher-dimensional grid and cells in the lower-dimensional grid).
        data = {"face_cells": primary_secondary_map}

        # Add the interface to the list, with
        self._edge_data[mg] = data

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

        self._edge_to_node[mg] = node_pair

    # --------- Remove and update nodes

    def remove_subdomain(self, node: pp.Grid) -> None:
        """
        Remove subdomain, and related interfaces, from the grid bucket.

        Parameters:
           node : the subdomain to be removed

        """
        # Delete from the subdomain lits.
        del self._nodes[node]

        # Find interfaces to be removed, store in a list.
        edges_to_remove: List[pp.MortarGrid] = []
        for mg in self.interfaces():
            edge = self._edge_to_node[mg]
            if edge[0] == node or edge[1] == node:
                edges_to_remove.append(mg)

        # Remove interfaces.
        for mg in edges_to_remove:
            del self._edge_data[mg]
            del self._edge_to_node[mg]

    def update_subdomains(self, mapping: Dict[pp.Grid, pp.Grid]) -> None:
        """
        Update the grids giving old and new values. The interfaces are updated
        accordingly.

        FIXME: This is in reality a subset of self.replace_subdomains. Should we merge?

        Parameters:
            mapping: A dictionary with the old grid as keys and new grid as values.

        """
        for old, new in mapping.items():
            data = self._nodes[old]
            self._nodes[new] = data
            del self._nodes[old]

            # Loop over edges, look for edges which has the node to be replaced as
            # one of its neighbors
            for mg, edge in self._edge_to_node.items():
                # If we find a hit, overwrite the old edge information.
                if edge[0] == old:
                    self._edge_to_node[mg] = (new, edge[1])
                elif edge[1] == old:
                    self._edge_to_node[mg] = (edge[0], new)

    def eliminate_subdomain(self, node: pp.Grid) -> List[pp.Grid]:
        """
        Remove the node (and the edges it partakes in) and add new direct
        connections (gb edges) between each of the neighbor pairs. A node with
        n_neighbors neighbours gives rise to 1 + 2 + ... + n_neighbors-1 new edges.

        FIXME: Can we delete this one?

        """
        # Identify neighbors
        neighbors: List[pp.Grid] = self.sort_nodes(
            self.subdomain_neighbors(node).tolist()
        )

        n_neighbors = len(neighbors)

        # Add an edge between each neighbor pair
        for i in range(n_neighbors - 1):
            g0 = neighbors[i]
            for j in range(i + 1, n_neighbors):
                g1 = neighbors[j]
                cell_cells = self._find_shared_face(g0, g1, node)
                self.add_edge((g0, g1), cell_cells)

        # Remove the node and update the ordering of the remaining nodes
        node_number = self.subdomain_data(node, "node_number")
        self.remove_node(node)
        self.update_node_ordering(node_number)

        return neighbors

    def duplicate_without_dimension(
        self, dim: int
    ) -> Tuple["GridTree", Dict[str, Dict]]:
        """
        Remove all the nodes of dimension dim and add new edges between their
        neighbors by calls to remove_node.

        FIXME: Can we delete this one?

        Parameters:
            dim (int): Dimension for which all grids should be removed.

        Returns:
            pp.GridTree: Copy of this GridTree, with all grids of dimension dim
                removed, and new edges between the neighbors of removed grids.
            Dict: Information on removed grids. Keys:
                "eliminated_nodes": List of all grids that were removed.
                "neighbours": List with neighbors of the eliminated grids. Sorted in
                    the same order as eliminated_nodes. Node ordering is updated.
                "neighbours_old": Same as neighbours, but with the original node
                    ordering.

        """

        gb_copy = self.copy()
        grids_of_dim = gb_copy.subdomains_of_dimension(dim)
        grids_of_dim_old = self.subdomains_of_dimension(dim)
        # The node numbers are copied for each grid, so they can be used to
        # make sure we use the same grids (g and g_old) below.
        nn_new = [gb_copy.subdomain_data(g, "node_number") for g in grids_of_dim]
        nn_old = [self.subdomain_data(g, "node_number") for g in grids_of_dim_old]
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
            neighbours_dict_old[i] = self.subdomain_neighbors(g_old)
            eliminated_nodes[i] = g_old

        elimination_data = {
            "neighbours": neighbours_dict,
            "neighbours_old": neighbours_dict_old,
            "eliminated_nodes": eliminated_nodes,
        }

        return gb_copy, elimination_data  # type: ignore

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

        Parameters:
            overwrite_existing (bool, optional): If True (default), any existing node
                ordering will be overwritten.

        """

        # Check whether 'node_number' is defined for the grids already.
        ordering_exists = True
        for _, data in self._nodes:
            if "node_number" not in data:
                ordering_exists = False
        if ordering_exists and not overwrite_existing:
            return

        counter = 0
        # Loop over grids in decreasing dimensions
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            for grid in self.subdomains_of_dimension(dim):
                data = self._nodes[grid]
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
                if "node_number" not in self._nodes[g]:
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

    def sort_nodes(self, nodes: List[pp.Grid]) -> List[pp.Grid]:
        """
        Sort nodes according to node number.

        Parameters:
            nodes: List of graph nodes.

        Returns:
            sorted_nodes: The same nodes, sorted.

        """
        assert all(["node_number" in self._nodes[g] for g in nodes])

        return sorted(nodes, key=lambda n: self.subdomain_data(n, "node_number"))

    # ------------- Miscellaneous functions ---------

    def target_2_source_nodes(self, g_src, g_trg):
        """
        Find the local node mapping from a source grid to a target grid.

        FIXME: Can we delete this one? It is not used in the core of PorePy.

        target_2_source_nodes(..) returns the mapping from g_src -> g_trg such
        that g_trg.nodes[:, map] == g_src.nodes. E.g., if the target grid is the
        highest dim grid, target_2_source_nodes will equal the global node
        numbering.

        """
        node_source = np.atleast_2d(g_src.global_point_ind)
        node_target = np.atleast_2d(g_trg.global_point_ind)
        _, trg_2_src_nodes = setmembership.ismember_rows(
            node_source.astype(np.int32_), node_target.astype(int)
        )
        return trg_2_src_nodes

    def compute_geometry(self) -> None:
        """Compute geometric quantities for the grids."""
        for grid in self.subdomains():
            grid.compute_geometry()

        for mg in self.interfaces():
            mg.compute_geometry()

    def copy(self) -> "GridTree":
        """Make a shallow copy of the grid bucket. The underlying grids are not copied.

        Return:
            pp.GridTree: Copy of this GridTree.

        """
        gb_copy: pp.GridTree = GridTree()
        gb_copy._nodes = self._nodes.copy()
        gb_copy._edge_data = self._edge_data.copy()
        gb_copy._edge_to_node = self._edge_to_node.copy()
        return gb_copy

    def replace_subdomains(
        self,
        g_map: Optional[Dict[pp.Grid, pp.Grid]] = None,
        mg_map: Optional[
            Dict[pp.MortarGrid, Dict[mortar_grid.MortarSides, pp.Grid]]
        ] = None,
        tol: float = 1e-6,
    ) -> None:
        """Replace grids and / or mortar grids in the mixed-dimensional grid.

        FIXME: Should be replace_subdomains_and_interfaces?

        Parameters:
            gb (GridTree): To be updated.
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
        # 2) Update the subdomain grids with respect to nodes in the GridTree.
        # 3) Update the subdomain grids with respect to the interfaces in the GridTree.
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

    def _find_shared_face(self, g0: pp.Grid, g1: pp.Grid, g_l: pp.Grid) -> np.ndarray:
        """
        Given two nd grids meeting at a (n-1)d node (to be removed), find which two
        faces meet at the intersection (one from each grid) and build the connection
        matrix cell_cells.
        In face_cells, the cells of the lower dimensional grid correspond to the first
        axis and the faces of the higher dimensional grid to the second axis.
        Furthermore, grids are sorted with the lower-dimensional first and the
        higher-dimensional last. To be consistent with this, the grid corresponding
        to the first axis of cell_cells is the first grid of the node sorting.

        FIXME: This can be deleted if eliminate_subdomains is deleted.

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
        fc1 = self.interface_data((g0, g_l))
        fc2 = self.interface_data((g1, g_l))
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

    def apply_function_to_subdomains(
        self, fct: Callable[[pp.Grid, Dict], Any]
    ) -> np.ndarray:
        """
        Loop on all the nodes and evaluate a function on each of them.

        FIXME: can we delete this one?

        Parameter:
            fct: function to evaluate. It takes a grid and the related data and
                returns a scalar.

        Returns:
            values: vector containing the function evaluated on each node,
                ordered by 'node_number'.

        """
        values = np.empty(self.num_subdomains())
        for g, d in self._nodes.items():
            values[d["node_number"]] = fct(g, d)
        return values

    def apply_function_to_interfaces(
        self, fct: Callable[[pp.Grid, pp.Grid, Dict, Dict, Dict], Any]
    ) -> sps.spmatrix:
        """
        Loop on all the edges and evaluate a function on each of them.

        FIXME: Can we delete this?

        Parameter:
            fct: function to evaluate. It returns a scalar and takes: the higher
                and lower dimensional grids, the higher and lower dimensional
                data, the global data.

        Returns:
            matrix: sparse strict upper triangular matrix containing the function
                evaluated on each edge (pair of nodes), ordered by their
                relative 'node_number'.

        """
        i = np.zeros(len(self._edge_data), dtype=int)
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

        diam_mg = [np.amax(mg.cell_diameters()) for mg in self._edge_data if cond(mg)]

        return np.amax(np.hstack((diam_g, diam_mg)))

    def bounding_box(
        self, as_dict: bool = False
    ) -> Union[Dict[str, float], Tuple[float, float]]:
        """
        Return the bounding box of the grid bucket.
        """
        c_0s = np.empty((3, self.num_subdomains()))
        c_1s = np.empty((3, self.num_subdomains()))

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
            return min_vals, max_vals  # type: ignore

    def size(self) -> int:
        """
        Returns:
            int: Number of mono-dimensional grids and interfaces in the bucket.

        """
        return self.num_subdomains() + self.num_interfaces()

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

    def all_dims(self) -> np.ndarray:
        """
        Returns:
            np.array: Active dimensions of the grids present in the hierarchy.

        """
        return np.unique([grid.dim for grid in self.subdomains()])

    def cell_volumes(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """
        Get the cell volumes of all cells of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            cell_volumes (np.ndarray): The volume of all cells in the GridTree

        """
        if cond is None:
            cond = lambda g: True
        return np.hstack(
            [grid.cell_volumes for grid in self.subdomains() if cond(grid)]
        )

    def face_centers(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """
        Get the face centers of all faces of the grid bucket, considering a loop
        on all the graph nodes.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            face_centers (ndArray): The face centers of all faces in the GridTree.
        """
        if cond is None:
            cond = lambda g: True
        return np.hstack(
            [grid.face_centers for grid in self.subdomains() if cond(grid)]
        )

    def cell_centers(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """
        Get the cell centers of all cells of the grid bucket, considering a loop
        on all the graph nodes.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            cell_centers (np.ndarray): The cell centers of all cells in the GridTree.

        """
        if cond is None:
            cond = lambda g: True
        return np.hstack(
            [grid.cell_centers for grid, _ in self._nodes.items() if cond(grid)]
        )

    def cell_volumes_mortar(self, cond: Callable[[pp.Grid], bool] = None) -> np.ndarray:
        """
        Get the cell volumes of all mortar cells of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            cell_volumes (np.ndarray): The cell volumes of all mortar cells in the
                GridTree.

        """
        if cond is None:
            cond = lambda g: True
        if self.num_mortar_cells(cond) == 0:
            return np.array([])
        return np.hstack([mg.cell_volumes for mg in self.interfaces()() if cond(mg)])

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
        return np.sum(  # type: ignore
            [grid.num_cells for grid in self.subdomains() if cond(grid)], dtype=int
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
        return np.sum(  # type: ignore
            [mg.num_cells for mg in self.interfaces() if cond(mg)],
            dtype=int,
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

        return np.sum(  # type: ignore
            [grid.num_faces for grid in self.subdomains() if cond(grid)]
        )

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

        return np.sum(  # type: ignore
            [grid.num_nodes for grid in self.subdomains() if cond(grid)]
        )

    def num_subdomains(self):
        """
        Return the total number of nodes (physical meshes) in the graph.

        Returns:
            int: Number of nodes in the graph.

        """
        return len(self._nodes)

    def num_interfaces(self) -> int:
        """
        Return the total number of edge in the graph.

        Returns:
            int: Number of edges in the graph.

        """
        return len(self._edge_data)

    def num_subdomains_and_interfaces(self) -> int:
        """
        Return the total number of nodes (physical meshes) plus the total number
        of edges in the graph. It is the size of the graph.

        Returns:
            int: Number of nodes and edges in the graph.

        """
        return self.num_subdomains() + self.num_interfaces()

    def __str__(self) -> str:
        max_dim = self.subdomains_of_dimension(self.dim_max())
        num_nodes = 0
        num_cells = 0
        for g in max_dim:
            num_nodes += g.num_nodes
            num_cells += g.num_cells
        s = (
            "Mixed-dimensional grid. \n"
            f"Maximum dimension present: {self.dim_max()} \n"
            f"Minimum dimension present: {self.dim_min()} \n"
        )
        s += "Size of highest dimensional grid: Cells: " + str(num_cells)
        s += ". Nodes: " + str(num_nodes) + "\n"
        s += "In lower dimensions: \n"
        for dim in range(self.dim_max() - 1, self.dim_min() - 1, -1):
            gl = self.subdomains_of_dimension(dim)
            num_nodes = 0
            num_cells = 0
            for g in gl:
                num_nodes += g.num_nodes
                num_cells += g.num_cells

            s += (
                f"{len(gl)} grids of dimension {dim}, with in total "
                f"{num_cells} cells and {num_nodes} nodes. \n"
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
