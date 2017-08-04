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

#------------------------------------------------------------------------------#

    def __iter__(self):
        """
        Iterator over the grid bucket.

        Yields:
            core.grid.nodes: The grid associated with a node.
            data: The dictionary storing all information in this node.

        """
        for g in self.graph:
            data = self.graph.node[g]
            yield g, data

#------------------------------------------------------------------------------#

    def size(self):
        """
        Returns:
            int: Number of nodes in the grid.

        """
        return self.graph.number_of_nodes()

#-----------------------

    def dim_max(self):
        """
        Returns:
            int: Maximum dimension of the grids present in the hierarchy.

        """
        return np.amax([g.dim for g, _ in self])

#------------------------------------------------------------------------------#

    def dim_min(self):
        """
        Returns:
            int: Minimum dimension of the grids present in the hierarchy.

        """
        return np.amin([g.dim for g, _ in self])

#------------------------------------------------------------------------------#

    def nodes(self):
        """
        Yields:
            An iterator over the graph nodes.

        """
        return self.graph.nodes_iter()

#------------------------------------------------------------------------------#

    def edges(self):
        """
        Yields:
            An iterator over the graph edges.

        """
        return self.graph.edges_iter()

#------------------------------------------------------------------------------#

    def sorted_nodes_of_edge(self, e):
        """
        Obtain the vertices of an edge, in ascending order with respect their
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
            if not self.has_nodes_prop(e, 'node_number'):
                self.assign_node_ordering()

            node_indexes = self.nodes_prop(e, 'node_number')
            if node_indexes[0] < node_indexes[1]:
                return e[0], e[1]
            return e[1], e[0]

        elif e[0].dim < e[1].dim:
            return e[0], e[1]
        else:
            return e[1], e[0]

#------------------------------------------------------------------------------#

    def sort_multiple_nodes(self, nodes): 
        """ 
        Sort all the nodes according to node number. 
 
        Parameters: 
            nodes: List of graph nodes. 
 
        Returns: 
            sorted_nodes: The same nodes, sorted. 
 
        """ 
        assert self.has_nodes_prop(nodes,'node_number') 
         
        return sorted(nodes, key = lambda n: self.node_prop( n, 'node_number')) 
 
#------------------------------------------------------------------------------# 
    def nodes_of_edge(self, e):
        """
        Obtain the vertices of an edge.

        Parameters:
            e: An edge in the graph.

        Returns:
            dictionary: The first vertex of the edge.
            dictionary: The second vertex of the edge.

        """
        return e[0], e[1]

#------------------------------------------------------------------------------#

    def node_neighbors(self, n):
        """
        Return:
            list of networkx.node: Neighbors of node n

        """
        return self.graph.neighbors(n)

#------------------------------------------------------------------------------#

    def add_node_prop(self, key, g=None, prop=None):
        """
        Add a new property to existing nodes in the graph.

        Properties can be added either to all nodes, or to selected nodes as
        specified by their grid. In the former case, all nodes will be assigned
        the property, with None as the default option.

        No tests are done on whether the key already exist, values are simply
        overwritten.

        Parameters:
            prop (optional, defaults to None): Property to be added.
            key (object): Key to the property to be handled.
            g (list of core.grids.grid, optional): Nodes to be assigned values.
                Defaults to None, in which case all nodes are assigned the same
                value.
            prop (list, or single object, optional): Value to be assigned.
                Should be either a list with the same length as g, or a single
                item which is assigned to all nodes. Defaults to None.

        Raises:
            ValueError if the key is 'node_number', this is reserved for other
                purposes. See self.assign_node_ordering() for details.

        """

        # Check that the key is not 'node_number' - this is reserved
        if key == 'node_number':
            raise ValueError('Node number is a reserved key, stay away')

        # Do some checks of parameters first
        if g is None:
            assert prop is None or not isinstance(prop, list)
        else:
            assert len(g) == len(prop)

        # Set default value.
        networkx.set_node_attributes(self.graph, key, None)

        if prop is not None:
            if g is None:
                for _, n in self:
                    n[key] = prop
            else:
                for v, p in zip(g, prop):
                    self.graph.node[v][key] = p

#------------------------------------------------------------------------------#

    def add_node_props(self, keys):
        """
        Add new properties to existing nodes in the graph.

        Properties are be added to all nodes.

        No tests are done on whether the key already exist, values are simply
        overwritten.

        Parameters:
            keys (object): Keys to the properties to be handled.

        Raises:
            ValueError if the key is 'node_number', this is reserved for other
                purposes. See self.assign_node_ordering() for details.

        """
        [self.add_node_prop(key) for key in np.atleast_1d(keys)]

#------------------------------------------------------------------------------#

    def add_edge_prop(self, key, grid_pairs=None, prop=None):
        """
        Associate a property with an edge.

        Properties can be added either to all edges, or to selected edges as
        specified by their grid pair. In the former case, all edges will be
        assigned the property, with None as the default option.

        No tests are done on whether the key already exist, values are simply
        overwritten.

        Parameters:
            prop (optional, defaults to None): Property to be added.
            key (object): Key to the property to be handled.
            g (list of list of core.grids.grid, optional): Grid pairs defining
                the edges to be assigned. values. Defaults to None, in which
                case all edges are assigned the same value.
            prop (list, or single object, optional): Value to be assigned.
                Should be either a list with the same length as grid_pairs, or
                a single item which is assigned to all nodes. Defaults to
                None.

        Raises:
            KeyError if a grid pair is not an existing edge in the grid.

        """

        # Do some checks of parameters first
        if grid_pairs is None:
            assert prop is None or not isinstance(prop, list)
        else:
            assert len(grid_pairs) == len(prop)

        # networkx.set_edge_attributes(self.graph, key, None)

        if prop is not None:
            for gp, p in zip(grid_pairs, prop):
                if tuple(gp) in self.graph.edges():
                    self.graph.edge[gp[0]][gp[1]][key] = p
                elif tuple(gp[::-1]) in self.graph.edges():
                    self.graph.edge[gp[1]][gp[0]][key] = p
                else:
                    raise KeyError('Cannot assign property to undefined\
                                     edge')

#------------------------------------------------------------------------------#

    def add_edge_props(self, keys):
        """
        Add new propertiy to existing edges in the graph.

        Properties can be added either to all edges.

        No tests are done on whether the key already exist, values are simply
        overwritten.

        Parameters:
            keys (object): Keys to the properties to be handled.

        Raises:
            KeyError if a grid pair is not an existing edge in the grid.

        """
        [self.add_edge_prop(key) for key in np.atleast_1d(keys)]

#------------------------------------------------------------------------------#

    def node_prop(self, g, key):
        """
        Getter for a node property of the bucket.

        Parameters:
            grid (core.grids.grid): The grid associated with the node.
            key (object): Key for the property to be retrieved.

        Returns:
            object: The property.
        """
        return self.graph.node[g][key]

#------------------------------------------------------------------------------#

    def has_nodes_prop(self, gs, key):
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
        return tuple([key in self.graph.node[g] for g in gs])

#------------------------------------------------------------------------------#

    def nodes_prop(self, gs, key):
        """
        Getter for a node property of the bucket, for several nodes.

        Parameters:
            grids (core.grids.grid): The grids associated with the nodes.
            key (object): Key for the property to be retrieved.

        Returns:
            object: The property.

        """
        return tuple([self.graph.node[g][key] for g in gs])

#------------------------------------------------------------------------------#

    def node_props(self, g):
        """
        Getter for a node property of the bucket.

        Parameters:
            grid (core.grids.grid): The grid associated with the node.

        Returns:
            object: A dictionary with keys and properties.

        """
        return self.graph.node[g]

#------------------------------------------------------------------------------#

    def node_props_of_keys(self, g, keys):
        """
        Getter for a node property of the bucket.

        Parameters:
            grid (core.grids.grid): The grid associated with the node.
            keys (object): Key for the property to be retrieved.

        Returns:
            object: A dictionary with key and property.

        """
        return {key: self.graph.node[g][key] for key in keys}

#------------------------------------------------------------------------------#

    def edge_prop(self, grid_pairs, key):
        """
        Getter for an edge property of the bucket.

        Parameters:
            grid_pairs (list of core.grids.grid): The two grids making up the
                edge.
            key (object): Key for the property to be retrieved.

        Returns:
            object: The property.

        Raises:
            KeyError if the two grids do not form an edge.

        """
        prop_list = []
        for gp in np.atleast_2d(grid_pairs):
            if tuple(gp) in self.graph.edges():
                prop_list.append(self.graph.edge[gp[0]][gp[1]][key])
            elif tuple(gp[::-1]) in self.graph.edges():                
                prop_list.append(self.graph.edge[gp[1]][gp[0]][key])
            else:
                raise KeyError('Unknown edge')
        return prop_list
#------------------------------------------------------------------------------#

    def edge_props(self, gp):
        """
        Getter for an edge properties of the bucket.

        Parameters:
            grids_pair (list of core.grids.grid): The two grids making up the
                edge.
            key (object): Keys for the properties to be retrieved.

        Returns:
            object: The properties.

        Raises:
            KeyError if the two grids do not form an edge.

        """
        if tuple(gp) in self.graph.edges():
            return self.graph.edge[gp[0]][gp[1]]
        elif tuple(gp[::-1]) in self.graph.edges():
            return self.graph.edge[gp[1]][gp[0]]
        else:
            raise KeyError('Unknown edge')

#------------------------------------------------------------------------------#

    def edges_props(self):
        """
        Iterator over the edges of the grid bucket.

        Yields:
            core.grid.edges: The edge (pair of grids) associated with an edge.
            data: The dictionary storing all information in this edge.

        """
        for e in self.graph.edges():
            yield e, self.edge_props(e)

#------------------------------------------------------------------------------#

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

        """
        assert not np.any([i is j for i in new_grids for j in self.graph])

        for g in new_grids:
            self.graph.add_node(g)

#------------------------------------------------------------------------------#

    def remove_node(self, node):
        """
        Remove node, and related edges, from the grid bucket.
        Parameters:
           node : the node to be removed

        """

        self.graph.remove_node(node)

#------------------------------------------------------------------------------#

    def remove_nodes(self, cond):
        """
        Remove nodes, and related edges, from the grid bucket subject to a
        conditions. The latter takes as input a grid.

        Parameters:
            cond: predicate to select the grids to remove.

        """

        self.graph.remove_nodes_from([g for g in self.graph if cond(g)])

#------------------------------------------------------------------------------#

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
            ValueError if the two grids are not one dimension apart

        """
        assert np.asarray(grids).size == 2
        # Check that the connection does not already exist
        assert not (tuple(grids) in self.graph.edges()
                    or tuple(grids[::-1]) in self.graph.edges())

        # The higher-dimensional grid is the first node of the edge.
        if grids[0].dim - 1 == grids[1].dim:
            self.graph.add_edge(*grids, face_cells=face_cells)
        elif grids[0].dim == grids[1].dim - 1:
            self.graph.add_edge(*grids[::-1], face_cells=face_cells)
        elif grids[0].dim == grids[1].dim:
            self.graph.add_edge(*grids, face_cells=face_cells)
        else:
            raise ValueError('Grid dimension mismatch')

#------------------------------------------------------------------------------#

    def grids_of_dimension(self, dim):
        """
        Get all grids in the bucket of a specific dimension.

        Returns:
            list: Of grids of the specified dimension

        """
        return [g for g in self.graph.nodes() if g.dim == dim]

#------------------------------------------------------------------------------#

    def assign_node_ordering(self):
        """
        Assign an ordering of the nodes in the graph, stored as the attribute
        'node_number'.

        The intended use is to define the block structure of a discretization
        on the grid hierarchy.

        The ordering starts with grids of highest dimension. The ordering
        within each dimension is determined by an iterator over the graph, and
        can in principle change between two calls to this function.

        If an ordering covering all nodes in the graph already exist, but does
        not coincide with the new ordering, a warning is issued.

        """

        # Check whether 'node_number' is defined for the grids already.
        ordering_exists = True
        for _, n in self:
            if not 'node_number' in n.keys():
                ordering_exists = False

        counter = 0
        # Loop over grids in decreasing dimensions
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            for g in self.grids_of_dimension(dim):
                n = self.graph.node[g]
                # Get old value, issue warning if not equal to the new one.
                num = n.get('node_number', -1)
                if ordering_exists and num != counter:
                    warnings.warn('Order of graph nodes has changed')
                # Assign new value
                n['node_number'] = counter
                counter += 1
#------------------------------------------------------------------------------#

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
                if not self.has_nodes_prop([g], 'node_number'):
                    # It is not clear how severe this case is. For the moment,
                    # we give a warning, and hope the user knows what to do
                    warnings.warn(
                        'Tried to update node ordering where none exists')
                    # No point in continuing with this node.
                    continue

                # Obtain the old node number
                n = self.graph.node[g]
                old_number = n.get('node_number', -1)
                # And replace it if it is higher than the removed one
                if old_number > removed_number:
                    n['node_number'] = old_number - 1


#------------------------------------------------------------------------------#

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
            node_source.astype(np.int32), node_target.astype(np.int32))
        return trg_2_src_nodes

#------------------------------------------------------------------------------#

    def compute_geometry(self, is_embedded=True, is_starshaped=False):
        """Compute geometric quantities for the grids.

        Note: the flag "is_embedded" is True by default.
        """

        [g.compute_geometry(is_embedded=is_embedded,
                            is_starshaped=is_starshaped) for g, _ in self]

#------------------------------------------------------------------------------#

    def copy(self):
        """Make a copy of the grid bucket utilizing the built-in copy function
        of networkx.

        """
        gb_copy = GridBucket()
        gb_copy.graph = self.graph.copy()
        return gb_copy

#------------------------------------------------------------------------------#

    def duplicate_without_dimension(self, dim, compute_new_fluxes=False):
        """
        Remove all the nodes of dimension dim and add new edges between their
        neighbors by calls to remove_node.

        """
        gb_copy = self.copy()
        grids_of_dim=gb_copy.grids_of_dimension(dim)
        gb_copy.add_edge_prop('cell_cells')
        for g in grids_of_dim:
            gb_copy.eliminate_node(g, compute_new_fluxes)

        return gb_copy

#------------------------------------------------------------------------------#

    def find_shared_face(self, n1, n2, zero_d_node):
        """
        Given two 1d grids meeting at a 0d node (to be removed), find which two
        faces meet at the intersection (one from each grid) and build the connection
        matrix cell_cells.
        The lower dimensional node (corresponding to the first dimension, cells,
        in face_cells) is first in gb.sorted_nodes_of_edge. To be consistent with
        this, the grid corresponding to the first dimension of cell_cells should
        be the first grid of the node sorting.

        Parameters:
            n1 and n2: The two 1d grids.
            zero_d_node: The 0d grid to be removed.
        Returns: The sparse matrix face_faces (n1.num_faces x n2.num_faces, with n1
            and n2 sorted) and cell_cells (n1.num_cells x n2.num_cells), the 1d-1d 
            equivalent of the face_cells matrix.
        """
        # Sort nodes according to node_number
        n1, n2 = self.sorted_nodes_of_edge([n1, n2])

        # Identify the faces connecting the neighbors to the grid to be removed
        fc1 = self.edge_props([n1, zero_d_node])
        fc2 = self.edge_props([n2, zero_d_node])
        _, face_number_1 = fc1['face_cells'].nonzero()
        _, face_number_2 = fc2['face_cells'].nonzero()
        _,cells_1 = n1.cell_faces[face_number_1].nonzero()
        _,cells_2 = n2.cell_faces[face_number_2].nonzero()
        # Connect the two remaining grid through the cell_cells matrix,
        # to be placed as a face_cells
        # substitute.
        face_faces = sps.csc_matrix(
            (np.array([True]*face_number_1.shape[0]), (face_number_1, face_number_2)),
            (n1.num_faces, n2.num_faces))
        # NB ORDERING!        
        cell_cells = np.zeros((n2.num_cells, n1.num_cells), dtype=bool)
        rows = np.tile(cells_2,(cells_1.size,1))
        cols = np.tile(cells_1,(cells_2.size,1)).T
        cell_cells[rows,cols] = True
        
        return face_faces, cell_cells

#------------------------------------------------------------------------------#

    def eliminate_node(self, node, compute_new_fluxes=False):
        """
        Remove the node (and the edges it partakes in) and add new direct
        connections (gb edges) between each of the neighbor pairs. A 0d node
        with n_neighbors gives rise to 1 + 2 + ... + n_neighbors-1 new edges.

        """
        # Identify neighbors
        neighbors = self.sort_multiple_nodes( self.node_neighbors(node) )
        
        n_neighbors = len(neighbors)
        
        # Add an edge between each neighbor pair
        for i in range(n_neighbors - 1):
            n1 = neighbors[i]
            for j in range(i + 1, n_neighbors):
                n2 = neighbors[j]
                face_faces, cell_cells = self.find_shared_face(n1, n2, node)
                self.add_edge([n1, n2], cell_cells)
                if compute_new_fluxes:
                    self.add_edge_prop('param', [[n1,n2]],[Parameters(n2)])
        if compute_new_fluxes:
            condensation.new_coupling_fluxes(self, node, neighbors)
        
        # Remove the node and update the ordering of the remaining nodes
        node_number = self.node_prop(node, 'node_number')
        self.remove_node(node)
        self.update_node_ordering(node_number)
        
#------------------------------------------------------------------------------#

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
        values = np.empty(self.size())
        for g, d in self:
            values[d['node_number']] = fct(g, d)
        return values

#------------------------------------------------------------------------------#

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
        for e, data in self.edges_props():
            g_l, g_h = self.sorted_nodes_of_edge(e)
            data_l, data_h = self.node_props(g_l), self.node_props(g_h)

            i[idx], j[idx] = self.nodes_prop([g_l, g_h], 'node_number')
            values[idx] = fct(g_h, g_l, data_h, data_l, data)
            idx += 1

        # Upper triangular matrix
        return sps.coo_matrix((values, (i, j)), (self.size(), self.size()))

#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#

    def __str__(self):
        max_dim = self.grids_of_dimension(self.dim_max())
        num_nodes = 0
        num_cells = 0
        for g in max_dim:
            num_nodes += g.num_nodes
            num_cells += g.num_cells
        s = 'Mixed dimensional grid. \n'
        s += 'Maximum dimension ' + str(self.dim_max()) + '\n'
        s += 'Minimum dimension ' + str(self.dim_min()) + '\n'
        s += 'Size of highest dimensional grid: Cells: ' + str(num_cells)
        s += '. Nodes: ' + str(num_nodes) + '\n'
        s += 'In lower dimensions: \n'
        for dim in range(self.dim_max() - 1, self.dim_min() - 1, -1):
            gl = self.grids_of_dimension(dim)
            s += str(len(gl)) + ' grids of dimension ' + str(dim) + '\n'
        return s

#------------------------------------------------------------------------------#

    def __repr__(self):
        s = 'Grid bucket containing ' + str(gb.size) + ' grids:\n'
        num = 0
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            gl = self.grids_of_dimension(dim)
            s += str(len(gl)) + ' grids of dimension ' + str(dim) + '\n'
        return s
