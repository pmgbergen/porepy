"""
Module to store the grid hiearchy formed by a set of fractures and their
intersections in the form of a GridBucket.

"""
from scipy import sparse as sps
import numpy as np
import networkx
import warnings

from utils import setmembership


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
        return np.amax([g.dim for g,_ in self])

#------------------------------------------------------------------------------#

    def dim_min(self):
        """
        Returns:
            int: Minimum dimension of the grids present in the hierarchy.

        """
        return np.amin([g.dim for g,_ in self])

#------------------------------------------------------------------------------#

    def nodes(self):
        """
        Yields:
            An iterator over the graph nodes.

        """
        return self.graph.nodes()

#------------------------------------------------------------------------------#

    def edges(self):
        """
        Yields:
            An iterator over the graph edges.

        """
        return self.graph.edges()

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

        """

        # Do some checks of parameters first
        if g is None:
            assert prop is None or not isinstance(prop, list)
        else:
            assert len(g) == len(prop)

        # Set default value. 
        networkx.set_node_attributes(self.graph, key, None)

        if prop is not None:
            if g is None:
                for grid, n in self:
                    n[key] = prop
            else:
                for v, p in zip(g, prop):
                    self.graph.node[v][key] = p

#------------------------------------------------------------------------------#

    def add_edge_prop(self, key, grid_pairs, prop):
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

        networkx.set_edge_attributes(self.graph, key, None)

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

    def get_node_prop(self, grid, key):
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

    def get_edge_prop(self, grid_pairs, key):
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
        for gp in grid_pairs:
            if tuple(gp) in self.graph.edges():
                return self.graph.edge[gp[0]][gp[1]][key]
            elif tuple(gp[::-1]) in self.graph.edges():
                return self.graph.edge[gp[1]][gp[0]][key]
            else:
                raise KeyError('Unknown edge')

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
        assert not np.any([ i is j for i in new_grids for j in self.graph])

        for g in new_grids:
            self.graph.add_node(g)

#------------------------------------------------------------------------------#

    def add_edge(self, grids, face_cells):
        """
        Add an edge in the graph, based on the higher and lower-dimensional
        grid.

        The coupling will be added with the higher-dimensional grid as the
        first node.

        NOTE: If we are interested in couplings between grids of the same
        dimension (e.g. in a domain-decomposition setting), we would need to
        loosen assertions in this function. We would also need to reinterpret
        face_cells.

        Parameters:
            grids (list, len==2). Grids to be connected. Order is arbitrary.
            face_cells (object): Identity mapping between cells in the
                higher-dimensional grid and faces in the lower-dimensional
                grid. No assumptions are made on the type of the object at this
                stage.

        Raises:
            ValueError if the two grids are not one dimension apart

        """
        assert np.asarray(grids).size == 2
        # Check that the connection does not already exist
        assert not (tuple(grids) in self.graph.edges()
                    or tuple(grids[::-1]) in self.graph.edges())

        # The higher-dimensional grid is the first node of the edge.
        if grids[0].dim-1 == grids[1].dim:
            self.graph.add_edge(*grids, face_cells=face_cells)
        elif grids[0].dim == grids[1].dim-1:

            self.graph.add_edge(*grids[::-1], face_cells=face_cells)
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
        for dim in range(self.dim_max(), self.dim_min()-1, -1):
            for g in self.grids_of_dimension(dim):
                n = self.graph.node[g]
                # Get old value, issue warning if not equal to the new one.
                num = n.get('node_number', -1)
                if ordering_exists and num != counter:
                    warnings.warn('Order of graph nodes has changed')
                # Assign new value
                n['node_number'] = counter
                counter += 1

    def target_2_source_nodes(self, g_src, g_trg):
        """
        Runar did this.

        """
        node_source = g_src.global_point_ind
        node_target = g_trg.global_point_ind
        node_source = node_source[None,:]
        node_target = node_target[None,:]
        _, trg_2_src_nodes = setmembership.ismember_rows(node_source, node_target)
        return trg_2_src_nodes

