"""
Module to store the grid hiearchy formed by a set of fractures and their
intersections in the form of a GridBucket.

"""

import warnings
from scipy import sparse as sps
import numpy as np
import networkx

from porepy.utils import setmembership
from porepy.numerics.mixed_dim import condensation
from porepy.params.data import Parameters


class GridBucket(object):
    """
    Container for the hiererchy of grids formed by fractures and their
    intersection.

    The information is stored in a graph borrowed from networkx, and the
    GridBucket is to a large degree a wrapper around the graph. Each grid
    defines a node in the graph, while edges are defined by grid pairs that
    have a connection.

    To all nodes and vertexes, there is associated a dictionary (courtesy
    networkx) that can store any type of data. Thus the GridBucket can double
    as a data storage and management tool.

    Attributes:
        graph (networkx.Graph): The of the grid. See above for further
            description.

    """

    def __init__(self):
        self.graph = networkx.Graph(directed=False)
        self.name = "grid bucket"

    # --------- Iterators -------------------------

    def __iter__(self):
        """
        Iterator over the nodes in the GridBucket.

        Yields:
            core.grid.nodes: The grid associated with a node.
            data: The dictionary storing all information in this node.

        """
        for g in self.graph:
            data = self.graph.node[g]
            yield g, data

    def nodes(self):
        """ Iterator over the nodes in the GridBucket.

        Identical functionality to self.__iter__(), but kept for consistency
        with self.edges()

        Yields:
            g: Grid associated with the current node.
            data: The dictionary storing all information in this node.

        """
        for g in self.graph:
            data = self.graph.node[g]
            yield g, data

    def edges(self):
        """
        Iterator over the edges in the GridBucket

        Yields:
            e: Grid pair associated with the current edge.
            data: The dictionary storing all information in this edge..

        """
        for e in self.graph.edges():
            yield e, self.edge_props(e)

    # ---------- Navigate within the graph --------

    def nodes_of_edge(self, e):
        """
        Obtain the vertices of an edge.

        The nodes will be given in ascending order with respect their
        dimension. If the edge is between grids of the same dimension, the node
        ordering (as defined by assign_node_ordering()) is used. If no ordering
        of nodes exists, assign_node_ordering() will be called by this method.

        Parameters:
            e: An edge in the graph.

        Returns:
            dictionary: The first vertex of the edge.
            dictionary: The second vertex of the edge.

        """

        if e[0].dim == e[1].dim:
            if not (
                "node_number" in self.graph.node[e[0]]
                and "node_number" in self.graph.node[e[1]]
            ):
                self.assign_node_ordering()

            node_indexes = [self.node_props(g, "node_number") for g in e]
            if node_indexes[0] < node_indexes[1]:
                return e[0], e[1]
            else:
                return e[1], e[0]

        elif e[0].dim < e[1].dim:
            return e[0], e[1]
        else:
            return e[1], e[0]

    def edges_of_node(self, n):
        """
        Iterator over the edges of the specific node.

        Parameters:
            n: A node in the graph.

        Yields:
            core.grid.edges: The edge (pair of grids) associated with an edge.
            data: The dictionary storing all information in this edge.
            object: A dictionary with keys and properties.

        """
        for e in self.graph.edges([n]):
            yield e, self.edge_props(e)

    def node_neighbors(self, node, only_higher=False, only_lower=False):
        """
        Get neighbors of a node in the graph.

        Optionally, return only neighbors corresponding to higher or lower
        dimension. At most one of only_higher and only_lower can be True.

        Parameters:
            node (Grid): node in the graph.
            only_higher (boolean, optional): Consider only the higher
                dimensional neighbors. Defaults to False.
            only_lower (boolean, optional): Consider only the lower dimensional
                neighbors. Defaults to False.

        Return:
            list of networkx.node: Neighbors of node 'node'

        Raises:
            ValueError if both only_higher and only_lower is True.

        """

        neigh = np.array([n for n in self.graph.neighbors(node)])

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

    def get_grids(self, cond=None):
        """
        Obtain the grids, optionally filtered by a specified condition.

        Example:
        g = self.gb.get_grids(lambda g: g.dim == dim)

        Parameters:
            cond: Predicate to select a grid. If None is given (default), all
                grids will be returned.

        Returns:
            grids: np.array of the grids.

        """
        if cond is None:
            cond = lambda g: True

        return np.array([g for g, _ in self if cond(g)])

    def grids_of_dimension(self, dim):
        """
        Get all grids in the bucket of a specific dimension.
        Returns:
            list: Of grids of the specified dimension

        """

        return self.get_grids(lambda g: g.dim == dim)

    def get_mortar_grids(self, cond=None, name="mortar_grid"):
        """
        Obtain the mortar grids, optionally filtered by a specified condition.

        Example:
        g = self.gb.get_mortar_grids(lambda g: g.dim == dim)

        Parameters:
            cond: Predicate to select a grid. If None is given (default), all
                grids will be returned.

        Returns:
            grids: np.array of the grids.

        """
        if cond is None:
            cond = lambda g: True
        return np.array([d[name] for _, d in self.edges() if cond(d[name])])

    # ----------- Adders for node and edge properties (introduce keywords)

    def add_node_props(self, keys, g=None):
        """
        Add a new property to existing nodes in the graph.

        Properties can be added either to all nodes, or to selected nodes as
        specified by their grid. In the former case, all nodes will be assigned
        the property.

        The property gets value None, other values must be assigned later.

        No tests are done on whether the key already exist, values are simply
        overwritten.

        Parameters:
            keys (object): Key to the property to be handled.
            g (list of grids.grid, optional): Nodes to be assigned values.
                Defaults to None, in which case all nodes are assigned the same
                value.

        Raises:
            ValueError if the key is 'node_number', this is reserved for other
                purposes. See self.assign_node_ordering() for details.

        """

        # Check that the key is not 'node_number' - this is reserved
        if "node_number" in keys:
            raise ValueError("Node number is a reserved key, stay away")

        # Do some checks of parameters first
        if g is not None and not isinstance(g, list):
            g = [g]

        for key in np.atleast_1d(keys):
            if g is None:
                networkx.set_node_attributes(self.graph, name=key, values=None)
            else:
                for h, n in self:
                    if h in g:
                        n[key] = None

    def add_edge_props(self, keys, grid_pairs=None):
        """
        Associate a property with an edge.

        Properties can be added either to all edges, or to selected edges as
        specified by their grid pair. In the former case, all edges will be
        assigned the property, with None as the default option.

        No tests are done on whether the key already exist, values are simply
        overwritten.

        Parameters:
            key (object): Key to the property to be handled.
            grid_pairs (list of list of core.grids.grid, optional): Grid pairs
                defining the edges to be assigned. values. Defaults to None, in
                which case all edges are assigned the same value.

        Raises:
            KeyError if a grid pair is not an existing edge in the grid.

        """
        for key in np.atleast_1d(keys):
            if grid_pairs is None:
                networkx.set_edge_attributes(self.graph, name=key, values=None)
            else:
                for gp in grid_pairs:
                    if tuple(gp) in self.graph.edges():
                        self.graph.adj[gp[0]][gp[1]][key] = None
                    elif tuple(gp[::-1]) in self.graph.edges():
                        self.graph.adj[gp[1]][gp[0]][key] = None
                    else:
                        raise KeyError(
                            "Cannot assign property to undefined\
                                         edge"
                        )

    # ------------ Getters for node and edge properties

    def has_nodes_prop(self, grids, key):
        """
        Test if a key exists for a node property of the bucket, for several nodes.
        Note: the property may contain None but the outcome of the test is
        still true.

        Parameters:
            grids (core.grids.grid): The grids associated with the nodes.
            key (object): Key for the property to be tested.

        Returns:
            object: The tested property.

        """
        return tuple([key in self.graph.node[g] for g in grids])

    def node_props(self, g, key=None):
        """
        Getter for a node property of the bucket.

        Parameters:
            grid (grid): The grid associated with the node.
            key (object, optional): Keys for the properties to be retrieved.
                If key is None (default) the entire data dictionary for the
                node is returned.

        Returns:
            object: A dictionary with keys and properties.

        """
        if key is None:
            return self.graph.node[g]
        else:
            return self.graph.node[g][key]

    def edge_props(self, gp, key=None):
        """
        Getter for an edge properties of the bucket.

        Parameters:
            grids_pair (list of core.grids.grid): The two grids making up the
                edge.
            key (object, optional): Keys for the properties to be retrieved.
                If key is None (default) the entire data dictionary for the
                edge is returned.

        Returns:
            object: The properties.

        Raises:
            KeyError if the two grids do not form an edge.

        """
        if tuple(gp) in self.graph.edges():
            if key is None:
                return self.graph.adj[gp[0]][gp[1]]
            else:
                return self.graph.adj[gp[0]][gp[1]][key]
        elif tuple(gp[::-1]) in self.graph.edges():
            if key is None:
                return self.graph.adj[gp[1]][gp[0]]
            else:
                return self.graph.adj[gp[1]][gp[0]][key]
        else:
            raise KeyError("Unknown edge")

    # ------------- Setters for edge and grid properties

    def set_node_prop(self, g, key, val):
        """ Set the value of a property of a given node.

        Values can also be set by accessing the data dictionary of the node
        directly.

        No checks are performed on whether the property has been added to
        the node. If the property has not been added, it will implicitly be
        generated.

        Parameters:
            g (grid): Grid identifying the node.
            key (object): Key identifying the field to add.
            val: Value to be added.

        """
        self.graph.node[g][key] = val

    def set_edge_prop(self, gp, key, val):
        """ Set the value of a property of a given edge.

        Values can also be set by accessing the data dictionary of the edge
        directly.

        No checks are performed on whether the property has been added to
        the edge. If the property has not been added, it will implicitly be
        generated.

        Parameters:
            g (grid): Grid identifying the node.
            key (object): Key identifying the field to add.
            val: Value to be added.

        Raises:
            KeyError if the two grids do not form an edge.

        """
        if tuple(gp) in self.graph.edges():
            self.graph.adj[gp[0]][gp[1]][key] = val
        elif tuple(gp[::-1]) in self.graph.edges():
            self.graph.adj[gp[1]][gp[0]][key] = val
        else:
            raise KeyError("Unknown edge")

    # ------------ Removers for nodes properties ----------

    def remove_node_props(self, keys, g=None):
        """
        Remove property to existing nodes in the graph.

        Properties can be removed either to all nodes, or to selected nodes as
        specified by their grid. In the former case, to all nodes the property
        will be removed.

        Parameters:
            keys (object): Key to the property to be handled.
            g (list of grids.grid, optional): Nodes to be removed the values.
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
        if g is not None and not isinstance(g, list):
            g = [g]

        for key in np.atleast_1d(keys):
            if g is None:
                for _, d in self:
                    del d[key]
            else:
                for h, d in self:
                    if h in g:
                        del d[key]

    def remove_edge_props(self, keys, e=None):
        """
        Remove property to existing edges in the graph.

        Properties can be removed either to all edges, or to selected edges as
        specified by their grid pair. In the former case, to all edges the property
        will be removed.

        Parameters:
            keys (object): Key to the property to be handled.
            e (list of pair of grids.grid, optional): Edges to be removed the
                values. Defaults to None, in which case the property is removed
                from all edges.

        Raises:
            ValueError if the key is 'edge_number', this is reserved for other
                purposes. See self.assign_node_ordering() for details.

        """

        # Check that the key is not 'edge_number' - this is reserved
        if "edge_number" in keys:
            raise ValueError("Edge number is a reserved key, stay away")

        # Do some checks of parameters first
        if e is not None and not any(isinstance(el, list) for el in e):
            e = [e]

        for key in np.atleast_1d(keys):
            if e is None:
                for _, d in self.edges():
                    del d[key]
            else:
                for ed, d in self.edges():
                    if ed in e:
                        del d[key]

    # ------------ Add new nodes and edges ----------

    def add_nodes(self, new_grids):
        """
        Add grids to the hierarchy.

        The grids are added to self.grids.

        Parameters:
            grids (iterable, list?): The grids to be added. None of these
                should have been added previously.

        Returns:
            np.ndarray, dtype object: Numpy array of vertexes, that are
                identified with the grid hierarchy. Same order as input grid.

        Raises:
            ValueError if a grid is already present in the bucket

        """
        new_grids = np.atleast_1d(new_grids)
        if np.any([i is j for i in new_grids for j in self.graph]):
            raise ValueError("Grid already defined in bucket")
        [self.graph.add_node(g) for g in new_grids]

    def add_edge(self, grids, face_cells):
        """
        Add an edge in the graph.

        If the grids have different dimensions, the coupling will be added with
        the higher-dimensional grid as the first node. For equal dimensions,
        the ordering of the nodes is the same as in input grids.

        Parameters:
            grids (list, len==2). Grids to be connected. Order is arbitrary.
            face_cells (object): Identity mapping between cells in the
                higher-dimensional grid and faces in the lower-dimensional
                grid. No assumptions are made on the type of the object at this
                stage. In the grids[0].dim = grids[1].dim case, the mapping is
                from faces of the first grid to faces of the second one.

        Raises:
            ValueError if the edge already exists.
            ValueError if the two grids are not one dimension apart.

        """
        assert np.asarray(grids).size == 2

        if (
            tuple(grids) in self.graph.edges()
            or tuple(grids[::-1]) in self.graph.edges()
        ):
            raise ValueError("Cannot add existing edge")

        # The higher-dimensional grid is the first node of the edge.
        if grids[0].dim - 1 == grids[1].dim:
            self.graph.add_edge(*grids, face_cells=face_cells)
        elif grids[0].dim == grids[1].dim - 1:
            self.graph.add_edge(*grids[::-1], face_cells=face_cells)
        elif grids[0].dim == grids[1].dim:
            self.graph.add_edge(*grids, face_cells=face_cells)
        else:
            raise ValueError("Grid dimension mismatch")

    # --------- Remove and update nodes

    def remove_node(self, node):
        """
        Remove node, and related edges, from the grid bucket.

        Parameters:
           node : the node to be removed

        """

        self.graph.remove_node(node)

    def remove_nodes(self, cond):
        """
        Remove nodes, and related edges, from the grid bucket subject to a
        conditions. The latter takes as input a grid.

        Parameters:
            cond: predicate to select the grids to remove.

        """
        if cond is None:
            cond = lambda g: True

        self.graph.remove_nodes_from([g for g in self.graph if cond(g)])

    def update_nodes(self, new, old):
        """
        Update the grids giving the old and new values. The edges are updated
        accordingly.

        Parameters:
            new: List of the new grids. Can be a single element
            old: List of the old grids, Can be a single element

        """
        new = np.atleast_1d(new)
        old = np.atleast_1d(old)
        assert new.size == old.size

        networkx.relabel_nodes(self.graph, dict(zip(new, old)), False)

    def eliminate_node(self, node):
        """
        Remove the node (and the edges it partakes in) and add new direct
        connections (gb edges) between each of the neighbor pairs. A node with
        n_neighbors neighbours gives rise to 1 + 2 + ... + n_neighbors-1 new edges.

        """
        # Identify neighbors
        neighbors = self.sort_multiple_nodes(self.node_neighbors(node))

        n_neighbors = len(neighbors)

        # Add an edge between each neighbor pair
        for i in range(n_neighbors - 1):
            g0 = neighbors[i]
            for j in range(i + 1, n_neighbors):
                g1 = neighbors[j]
                cell_cells = self.find_shared_face(g0, g1, node)
                self.add_edge([g0, g1], cell_cells)

        # Remove the node and update the ordering of the remaining nodes
        node_number = self.node_props(node, "node_number")
        self.remove_node(node)
        self.update_node_ordering(node_number)

        return neighbors

    def duplicate_without_dimension(self, dim):
        """
        Remove all the nodes of dimension dim and add new edges between their
        neighbors by calls to remove_node.

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

        return gb_copy, elimination_data

    # ---------- Functionality related to ordering of nodes

    def assign_node_ordering(self, overwrite_existing=True):
        """
        Assign an ordering of the nodes in the graph, stored as the attribute
        'node_number'.

        The intended use is to define the block structure of a discretization
        on the grid hierarchy.

        The ordering starts with grids of highest dimension. The ordering
        within each dimension is determined by an iterator over the graph, and
        can in principle change between two calls to this function.

        If an ordering covering all nodes in the graph already exist, but does
        not coincide with the new ordering, a warning is issued. If the optional
        parameter overwrite_existing is set to False, no update is performed if
        an node ordering already exists.
        """

        # Check whether 'node_number' is defined for the grids already.
        ordering_exists = True
        for _, n in self:
            if not "node_number" in n.keys():
                ordering_exists = False
        if ordering_exists and not overwrite_existing:
            return

        counter = 0
        # Loop over grids in decreasing dimensions
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            for g in self.grids_of_dimension(dim):
                n = self.graph.node[g]
                # Get old value, issue warning if not equal to the new one.
                num = n.get("node_number", -1)
                if ordering_exists and num != counter:
                    warnings.warn("Order of graph nodes has changed")
                # Assign new value
                n["node_number"] = counter
                counter += 1

        self.add_edge_props("edge_number")
        counter = 0
        for e, d in self.edges():
            d["edge_number"] = counter
            counter += 1

    def update_node_ordering(self, removed_number):
        """
        Uppdate an existing ordering of the nodes in the graph, stored as the attribute
        'node_number'.
        Intended for keeping the node ordering after removing a node from the bucket. In
        this way, the edge sorting will not be disturbed by the removal, but no gaps in are
        created.

        Parameter:
            removed_number: node_number of the removed grid.

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
                n = self.graph.node[g]
                old_number = n.get("node_number", -1)
                # And replace it if it is higher than the removed one
                if old_number > removed_number:
                    n["node_number"] = old_number - 1

    def sort_multiple_nodes(self, nodes):
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

    def cell_global2loc(self):
        """
        Create a global to local cell-mapping.

        cell_global2loc(..) add a keyword cell_global2local to each node in the
        GridBucket which is a sparse matrix R which restrict the global cell
        number, to local cell number. I.e., R*global_cell_vector
        equals the local cell ordering of that node. For the GridBucket:

                                   0 1 4 2 3
                                   - - x - -

        where - represent 1D cells and x a 0D cell, and the numbers above is the
        global cell ordering, cell_global2loc will give out the two matrices
        R_1 = [[1,0,0,0,0],
               [0,1,0,0,0],
               [0,0,1,0,0],
               [0,0,0,1,0]]
        R_0 = [[0,0,0,0,1]]

        If the GridBucket has mortar grids on the edges, a corresponding
        restriction from global mortar cells to local mortar cells will be
        made.
        """

        # Create node restriction
        self.add_node_props("cell_global2loc")
        for g, d in self:
            pos_i = d["node_number"]
            mat = np.empty(self.num_graph_nodes(), dtype=np.object)
            # first initial empty matrix
            for g_j, d_j in self:
                pos_j = d_j["node_number"]
                mat[pos_j] = sps.coo_matrix((g.num_cells, g_j.num_cells))

            # overwrite the local matrix for grid g
            mat[pos_i] = sps.eye(g.num_cells)
            d["cell_global2loc"] = sps.hstack(mat, "csr")

        # create mortar restriction
        for _, d in self.edges():
            if not d.get("mortar_grid"):
                continue
            gm = d["mortar_grid"]
            pos_i = d["edge_number"]
            mat = np.empty(self.num_graph_edges(), dtype=np.object)
            # first initial empty matrix
            for _, d_j in self.edges():
                gm_j = d_j["mortar_grid"]
                pos_j = d_j["edge_number"]
                mat[pos_j] = sps.coo_matrix((gm.num_cells, gm_j.num_cells))

            # overwrite the local matrix for grid g
            mat[pos_i] = sps.eye(gm.num_cells)

            d["cell_global2loc"] = sps.hstack(mat, "csr")

    def compute_geometry(self):
        """Compute geometric quantities for the grids.
        """

        [g.compute_geometry() for g, _ in self]
        [
            d["mortar_grid"].compute_geometry()
            for _, d in self.edges()
            if d.get("mortar_grid")
        ]

    def copy(self):
        """Make a copy of the grid bucket utilizing the built-in copy function
        of networkx.

        """
        gb_copy = GridBucket()
        gb_copy.graph = self.graph.copy()
        return gb_copy

    def find_shared_face(self, g0, g1, g_l):
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
        g0, g1 = self.nodes_of_edge([g0, g1])

        # Identify the faces connecting the neighbors to the grid to be removed
        fc1 = self.edge_props([g0, g_l])
        fc2 = self.edge_props([g1, g_l])
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

    def apply_function_to_nodes(self, fct):
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

    def apply_function_to_edges(self, fct):
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
        i = np.zeros(self.graph.number_of_edges(), dtype=int)
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

    def apply_function(self, fct_nodes, fct_edges):
        """
        Loop on all the nodes and edges and evaluate a function on each of them.

        Parameter:
            fct_nodes: function to evaluate. It takes a grid and the related data
                and returns a scalar.

            fct_edges: function to evaluate. It returns a scalar and takes: the
                higher and lower dimensional grids, the higher and lower
                dimensional data, the global data.

        Returns:
            matrix: sparse triangular matrix containing the function
                evaluated on each edge (pair of nodes) and node, ordered by their
                relative 'node_number'. The diagonal contains the node
                evaluation.

        """
        matrix = self.apply_function_to_edges(fct_edges)
        matrix.setdiag(self.apply_function_to_nodes(fct_nodes))
        return matrix

    # ---- Methods for getting information on the bucket, or its components ----

    def diameter(self, cond=None):
        """
        Compute the grid bucket diameter (mesh size), considering a loop on all
        the grids.  It is possible to specify a condition based on the grid to
        select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            diameter: the diameter of the grid bucket.
        """
        if cond is None:
            cond = lambda g: True
        diam_g = [np.amax(g.cell_diameters()) for g in self.graph if cond(g)]

        diam_mg = [
            np.amax(d["mortar_grid"].cell_diameters())
            for e, d in self.edges()
            if cond(e) and d.get("mortar_grid")
        ]

        return np.amax(np.hstack((diam_g, diam_mg)))

    def bounding_box(self, as_dict=False):
        """
        Return the bounding box of the grid bucket.
        """
        c_0s = np.empty((3, self.num_graph_nodes()))
        c_1s = np.empty((3, self.num_graph_nodes()))

        for i, g in enumerate(self.graph):
            c_0s[:, i], c_1s[:, i] = g.bounding_box()

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

    def size(self):
        """
        Returns:
            int: Number of mono-dimensional grids in the bucket.

        """
        return self.num_graph_nodes() + self.num_graph_edges()

    def dim_min(self):
        """
        Returns:
            int: Minimum dimension of the grids present in the hierarchy.

        """
        return np.amin([g.dim for g, _ in self])

    def dim_max(self):
        """
        Returns:
            int: Maximum dimension of the grids present in the hierarchy.

        """
        return np.amax([g.dim for g, _ in self])

    def all_dims(self):
        """
        Returns:
            int: Active dimensions of the grids present in the hierarchy.

        """
        return np.unique([g.dim for g, _ in self])

    def num_cells(self, cond=None):
        """
        Compute the total number of cells of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            num_cells: the total number of cells of the grid bucket.
        """
        if cond is None:
            cond = lambda g: True
        return np.sum([g.num_cells for g in self.graph if cond(g)])

    def num_mortar_cells(self, cond=None):
        """
        Compute the total number of mortar cells of the grid bucket, considering
        a loop on all mortar grids. It is possible to specify a condition based
        on the grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            num_cells: the total number of cells of the grid bucket.
        """
        if cond is None:
            cond = lambda g: True
        return np.sum(
            [
                d["mortar_grid"].num_cells
                for _, d in self.edges()
                if d.get("mortar_grid") and cond(d["mortar_grid"])
            ]
        )

    def num_faces(self, cond=None):
        """
        Compute the total number of faces of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            num_faces: the total number of faces of the grid bucket.
        """
        if cond is None:
            cond = lambda g: True
        return np.sum([g.num_faces for g in self.graph if cond(g)])

    def num_nodes(self, cond=None):
        """
        Compute the total number of nodes of the grid bucket, considering a loop
        on all the grids.  It is possible to specify a condition based on the
        grid to select some of them.

        Parameter:
            cond: optional, predicate with a grid as input.

        Return:
            num_nodes: the total number of nodes of the grid bucket.
        """
        if cond is None:
            cond = lambda g: True
        return np.sum([g.num_nodes for g in self.graph if cond(g)])

    def num_graph_nodes(self):
        """
        Return the total number of nodes (physical meshes) in the graph.

        Returns:
            int: Number of nodes in the graph.

        """
        return self.graph.number_of_nodes()

    def num_graph_edges(self):
        """
        Return the total number of edge in the graph.

        Returns:
            int: Number of edges in the graph.

        """
        return self.graph.number_of_edges()

    def num_nodes_edges(self):
        """
        Return the total number of nodes (physical meshes) plus the total number
        of edges in the graph. It is the size of the graph.

        Returns:
            int: Number of nodes and edges in the graph.

        """
        return self.num_graph_nodes() + self.num_graph_edges()

    def __str__(self):
        max_dim = self.grids_of_dimension(self.dim_max())
        num_nodes = 0
        num_cells = 0
        for g in max_dim:
            num_nodes += g.num_nodes
            num_cells += g.num_cells
        s = "Mixed dimensional grid. \n"
        s += "Maximum dimension " + str(self.dim_max()) + "\n"
        s += "Minimum dimension " + str(self.dim_min()) + "\n"
        s += "Size of highest dimensional grid: Cells: " + str(num_cells)
        s += ". Nodes: " + str(num_nodes) + "\n"
        s += "In lower dimensions: \n"
        for dim in range(self.dim_max() - 1, self.dim_min() - 1, -1):
            gl = self.grids_of_dimension(dim)
            s += str(len(gl)) + " grids of dimension " + str(dim) + "\n"
        return s

    def __repr__(self):
        s = "Grid bucket containing " + str(self.num_graph_nodes()) + " grids:\n"
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            gl = self.grids_of_dimension(dim)
            s += str(len(gl)) + " grids of dimension " + str(dim) + "\n"
        return s
