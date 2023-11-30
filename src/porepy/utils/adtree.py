""" Module containing the implementation of an alternating digital tree (ADT)
for 3D geometric searching and intersection problems.

See the following works: 10.1002/nme.1620310102 and 10.1201/9781420050349 section 14.

Let [a, b) be the interval that contains the points that we want to insert
in a binary tree. We then divide the interval into two equal parts: all
the points in the interval [a, (a+b)/2) will be inserted in the left
subtree, while the points in the interval [(a+b)/2, b ) will be placed
in the right subtree. The reasoning is then iterated to the next
level for each subtree.

                    [0, 1)
                    /    \
                   /      \
                  /        \
           [0, 0.5)         [0.5, 1)
           /   |              |     \
          /    |              |      \
         /     |              |       \
[0, 0.25)   [0.25, 0.5)   [0.5, 0.75)  [0.75, 1)

When inserting the following nodes
A = 0.1, B = 0.6, C = 0.7, D = 0.8, E = 0.2 and F = 0.1
in the tree, we get the following

                      A
                    /    \
                   /      \
                 E          B
                / \        / \
               /   \      /   \
              F   -1     C     D

The first element A is added as the root. The second element B check if its
coordinate (in this case is a single value) is smaller than 0.5. If so, it goes
on the left part of the tree starting from the root otherwise on the right part.
Now, since B = 0.6 it goes on the right part. Now are with the node C,
we need to check as before if its coordinate is smaller than 0.5 (so it goes on the left)
or bigger than 0.5 (so it goes on the right). Since it is 0.7 it goes on the right and being
already taken by B we need to go one level down. We check now if its coordinate is
smaller (left) or bigger (right) than 0.75. Since it's smaller we proceed on the left
part and being empty we add it. The insertion is not related to the parent but to
which level and coordinate a node has. For the multi-dimension case we alternate
the dimension by each level, so first we check the abscissa
(again with left and right decided as before) and then the
ordinate and so on. We detail a bit more here in the sequel.

In the multi-dimensional case, the ADT is organized in the same way, but
the subdivision takes place alternately for the various coordinates:
if the structure must contain n-dimensional points, at the i-th level
of the tree the subdivision is carried out with respect to the j-th
coordinate, where j is the remainder of the i / n division.
We immediately observe that the n-dimensional "points", the structure
contains true points in 2 or 3 dimensions, and rectangles or
parallelepipeds, which can be represented by "points" in 4 and 6
dimensions, with the writing (xmin, ymin, zmin, xmax, ymax, zmax).
Other geometrical objects are represented by their bounding box.
To avoid floating point problems, all the "points" are rescaled in [0, 1].

A search in the tree gives a list of all possible nodes that may
intersect the given one.

"""
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.utils.array_operations import sparse_array_to_row_col_data


class ADTNode:
    """
    Simple bookkeeping class that contains the basic information of a tree node.

    Attributes:
        key (Any): any key related to the node
        box (np.ndarray): the bounding box associated to the node
        child (list): list of identification of right and left children, if a children is not
            present is marked as -1.
        parent (int): identification of the parent node, marked -1 if not present (the root
            of a tree)

    """

    def __init__(self, key: Any, box: np.ndarray) -> None:
        """Initialize the node.
        The physical dimension associated to the node represents the dimension of the object.
        For a 3d element is 3, for 2d elements is 2, for 1d elements is 1, for 0d elements
        is 1. The latter can be seen as the degenerate case of a 1d element.

        Parameters:
            key (Any): The key associated to the node
            box (np.ndarray): The bounding box of the node
        """
        self.key: Any = key
        self.box: np.ndarray = np.atleast_1d(np.asarray(box))

        self.child: List[int] = [-1, -1]
        self.parent: int = -1

    def __str__(self) -> str:
        """Implementation of __str__"""
        s = (
            "Node with key: "
            + str(self.key)
            + "\nChild nodes: "
            + str(self.child)
            + "\nParent node: "
            + str(self.parent)
            + "\nBounding box: "
            + str(self.box)
        )
        return s

    def __repr__(self) -> str:
        """Implementation of __repr__"""
        s = (
            "key: "
            + repr(self.key)
            + " left child: "
            + repr(self.child[0])
            + " right child: "
            + repr(self.child[0])
            + " parent: "
            + repr(self.parent)
            + " box: "
            + repr(self.box)
        )
        return s


class ADTree:
    """
    ADT structure, it is possible to fill the tree by giving a PorePy grid and then search for
    possible intersections. The implementation does not include some features, like removing a
    node, that are not used so far. Possible extensions in the future.

    Attributes:
        tree_dim (int): search dimension of the tree, typically (e.g., when a pp.Grid is
            given) the double of the phys_dim
        phys_dim (int): physical dimension of nodes in the tree, e.g., a 2d grid will have
            phys_dim = 2
        nodes (list): the list of nodes as ADTNode
        region_min (float): to scale the bounding box of all the elements in [0, 1]^phys_dim
            we need the minimum corner point of the all region
        delta (float): a parameter to scale and get all the bounding box of the elements in
            [0, 1]^phys_dim
        LEFT (int): define the index of the left child, being equal to 0
        RIGHT (int): define the index of the right child, being equal to 1
    """

    # Pre-defined ordering of left and right node
    LEFT: int = 0
    RIGHT: int = 1

    def __init__(self, tree_dim: int, phys_dim: int) -> None:
        """Initialize the tree, if the grid is given then the tree is filled.

        Parameters:
            tree_dim (np.ndarray, optional): Set the tree dimension (typically the double of
                the physical dimension)
            tree_dim (np.ndarray, optional): Set the physical dimension
        """
        self.tree_dim: int = tree_dim
        self.phys_dim: int = phys_dim

        self.nodes: List[ADTNode] = []
        self.region_min: float = 0.0
        self.delta: float = 1.0

    def __str__(self) -> str:
        """Implementation of __str__"""
        s = (
            "Tree search dimension: "
            + str(self.tree_dim)
            + "\nPhysical dimension: "
            + str(self.phys_dim)
            + "\nNumber of nodes: "
            + str(len(self.nodes))
            + "\nFor the geometrical scaling in [0, 1], the region minimum: "
            + str(self.region_min)
            + " and delta "
            + str(self.delta)
        )
        return s

    def __repr__(self) -> str:
        """Implementation of __repr__"""

        s = (
            "Search dimension: "
            + str(self.tree_dim)
            + " physical dimension: "
            + str(self.phys_dim)
            + " the region minimum: "
            + str(self.region_min)
            + " delta: "
            + str(self.delta)
            + " number of nodes: "
            + str(len(self.nodes))
            + " list of nodes:\n"
        )

        s += "\n".join(
            ["node " + str(idn) + " " + repr(n) for idn, n in enumerate(self.nodes)]
        )

        return s

    def add_node(self, node: ADTNode) -> None:
        """Add a new node to the tree. We traverse the tree as previously specified and
        assign the new node accordingly.

        Parameters:
            node (ADTNode): the new node to be added.

        """
        # When the tree is empty just add the node as root
        if len(self.nodes) == 0:
            self.nodes.append(node)
            return

        # Current level for the dimension to check
        level = 0
        # Get the first node id, the position in the self.nodes list
        next_node_id = 0
        box = node.box.copy()
        while next_node_id != -1:
            current_node_id = next_node_id
            # Get the current search dimension
            search_dim = level % self.tree_dim
            # We take advantage of the fact that we are only descending
            # the tree. We then recursively multiply by 2 the coordinate.
            box[search_dim] *= 2.0
            # Check if the new node go on the right or on the left
            if box[search_dim] < 1.0:
                edge = self.LEFT
            else:
                edge = self.RIGHT
                box[search_dim] -= 1.0
            # Take the new node
            next_node_id = self.nodes[current_node_id].child[edge]
            level += 1

        # The correct position has been found, add informations to its parent
        self.nodes[current_node_id].child[edge] = len(self.nodes)

        # Add the new node to the list
        node.parent = current_node_id
        self.nodes.append(node)

    def search(self, node: ADTNode, tol: float = 2.0e-6) -> np.ndarray:
        """Search all possible nodes in the tree that might intersect with the input node.
        The node is not added to the tree.

        Parameters:
            node (ADTNode): Input node
            tol (float, optional): Geometrical tolerance to avoid floating point problems

        Returns:
            nodes (np.ndarray): Sorted, by id, list of nodes id that might intersect the node
        """

        # Enlarge the region to avoid floating point problems
        box = node.box.copy()
        box[: self.phys_dim] = self._scale(box[: self.phys_dim]) - tol
        box[self.phys_dim :] = self._scale(box[self.phys_dim :]) + tol

        # Origin of the sub-tree
        origin = np.zeros(self.tree_dim, dtype=float)

        level = 0
        node_id = 0
        # List of possible intersections
        found: List[int] = []
        # List of right parts of the tree that need to be checked.
        # Each element contains the node_id (position in self.nodes)
        # the (shifted) origin at the current level and the current
        # level
        stack: List[Tuple[int, np.ndarray, int]] = []

        # To traverse a non-empty binary tree in preorder,
        # perform the following operations recursively at each node,
        # starting with the root node:
        # > visit the root;
        # > traverse the left subtree;
        # > traverse the right subtree.
        while node_id != -1:
            # check the left part
            next_node_id = -1
            while True:
                # Intersect with the search box, add in case
                if self._box_intersect(box, self.nodes[node_id].box):
                    found.append(node_id)
                # Go to the left part of the sub-tree, and check if
                # it intersects the box. The right part is considered later.
                next_node_id = self.nodes[node_id].child[self.LEFT]
                if next_node_id != -1:
                    search_dim = level % self.tree_dim
                    if search_dim < self.phys_dim:
                        if origin[search_dim] > box[search_dim + self.phys_dim]:
                            next_node_id = -1
                    else:
                        delta = self._delta(level, self.tree_dim)
                        if origin[search_dim] + delta < box[search_dim - self.phys_dim]:
                            next_node_id = -1
                # Left sub-tree is neither null nor external, push info onto the stack
                if next_node_id != -1:
                    stack.append((node_id, origin.copy(), level))
                    node_id = next_node_id
                    level += 1
                else:
                    break

            # Check the right part that are stored in the stack
            level += 1
            node_id = self.nodes[node_id].child[self.RIGHT]
            while True:
                while node_id == -1:
                    if len(stack) == 0:
                        return np.sort(found)
                    info = stack.pop()
                    node_id = self.nodes[info[0]].child[self.RIGHT]
                    origin = info[1]
                    level = info[2] + 1

                # Check if the subtree intersects the box. Otherwise set right_link = -1,
                # pop new node from the stack and adjourn level.
                #
                # lev-1 is the level of the parent node, which directs the search.
                search_dim = (level - 1) % self.tree_dim
                delta = self._delta(level - 1, self.tree_dim)
                # reset origin (we have just traversed the tree to the right)
                origin[search_dim] += delta
                if search_dim < self.phys_dim:
                    if origin[search_dim] > box[search_dim + self.phys_dim]:
                        node_id = -1
                else:
                    if origin[search_dim] + delta < box[search_dim - self.phys_dim]:
                        node_id = -1

                if node_id != -1:
                    break

        # nothing is found, return an empty array
        return np.empty(0)

    def from_grid(self, g: pp.Grid, only_cells: Optional[np.ndarray] = None) -> None:
        """Function that constructs the tree from a grid by adding one cell at a time.

        Parameters:
            g (pp.Grid): The grid to be used to construct the tree
            only_cells (np.ndarray, optional): Consider only a portion of the cells
                due to some a-priori estimates of the searching elements.

        """
        self.g = g
        # Get the geometrical information cell-to-nodes
        g_cell_nodes = self.g.cell_nodes()
        g_nodes, g_cells, _ = sparse_array_to_row_col_data(g_cell_nodes)

        # select which cells to add to the tree
        if only_cells is not None:
            which_cells = np.asarray(only_cells)

            # to compute the min and max of the region, and get a more balanced tree,
            # we need to consider only the cells involved
            which_nodes = np.empty(0, dtype=int)
            for c in which_cells:
                loc = slice(g_cell_nodes.indptr[c], g_cell_nodes.indptr[c + 1])
                which_nodes = np.append(which_nodes, g_nodes[loc])
            which_nodes = np.unique(which_nodes)

        else:
            # if only_cells is not specified consider all the cells and nodes of the mesh
            which_cells = np.arange(self.g.num_cells)
            which_nodes = np.arange(self.g.num_nodes)

        # data for the normalization of the points
        self.region_min = self.g.nodes[: self.phys_dim, which_nodes].min(axis=1)
        region_max = self.g.nodes[: self.phys_dim, which_nodes].max(axis=1)
        self.delta = 1.0 / (region_max - self.region_min)

        # loop over the considered cells of the grid
        for c in which_cells:
            # get the nodes of the current cell
            loc = slice(g_cell_nodes.indptr[c], g_cell_nodes.indptr[c + 1])
            c_nodes = self.g.nodes[: self.phys_dim, g_nodes[loc]]
            # compute the scaled bounding box
            c_min = self._scale(c_nodes.min(axis=1))
            c_max = self._scale(c_nodes.max(axis=1))

            # create the new node
            new_node = ADTNode(c, np.hstack((c_min, c_max)))
            self.add_node(new_node)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        """Scale the input point to be in the interval [0, 1]

        Parameters:
            x (np.ndarray): the point to be scaled

        Returns:
            (np.ndarray): the scaled point
        """
        return self.delta * (x - self.region_min)

    def _box_intersect(self, box1: np.ndarray, box2: np.ndarray) -> bool:
        """Check if two boxes intersect.

        Parameters:
            box1 (np.ndarray): the first box
            box2 (np.ndarray): the second box

        Returns:
            (bool): if the two boxes intersect
        """
        return not (
            np.any(box1[: self.phys_dim] > box2[self.phys_dim :])
            or np.any(box1[self.phys_dim :] < box2[: self.phys_dim])
        )

    def _delta(self, level: int, dim: int) -> float:
        """Compute portion of the [0, 1]^self.dim at the current level

        Parameters:
            level (int): current level

        Returns:
            (float): current portion of the interval according to the level
        """
        return np.prod(0.5 * np.ones(int(level / dim) + 1, dtype=float))
