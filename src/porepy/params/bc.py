# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:30:22 2016

@author: eke001
"""
import numpy as np
import warnings


class BoundaryCondition(object):

    """ Class to store information on boundary conditions.

    The BCs are specified by face number, and can have type Dirichlet or
    Neumann (Robin may be included later). For details on default values etc.,
    see constructor.

    Attributes:
        num_faces (int): Number of faces in the grid
        dim (int): Dimension of the boundary. One less than the dimension of
            the grid.
        is_neu (np.ndarray boolean, size g.num_faces): Element i is true if
            face i has been assigned a Neumann condition. Tacitly assumes that
            the face is on the boundary. Should be false for internal faces, as
            well as Dirichlet faces.
        is_dir (np.ndarary, boolean, size g.num_faces): Element i is true if
            face i has been assigned a Neumann condition.

    """

    def __init__(self, g, faces=None, cond=None):
        """Constructor for BoundaryConditions.

        The conditions are specified by face numbers. Faces that do not get an
        explicit condition will have Neumann conditions assigned.

        Parameters:
            g (grid): For which boundary conditions are set.
            faces (np.ndarray): Faces for which conditions are assigned.
            cond (list of str): Conditions on the faces, in the same order as
                used in faces. Should be as long as faces.

        Example:
            # Assign Dirichlet condititons on the left side of a grid; implicit
            # Neumann conditions on the rest
            g = CartGrid([2, 2])
            west_face = bc.face_on_side(g, 'west')
            bound_cond = BoundaryCondition(g, faces=west_face, cond=['dir',
                                                                     'dir'])

        """

        self.num_faces = g.num_faces
        self.dim = g.dim - 1

        self.bc_type = "scalar"

        # Find boundary faces
        bf = g.get_all_boundary_faces()

        # Keep track of internal boundaries
        self.is_internal = g.tags["fracture_faces"]

        self.is_neu = np.zeros(self.num_faces, dtype=bool)
        self.is_dir = np.zeros(self.num_faces, dtype=bool)

        # By default, all faces are Neumann.
        self.is_neu[bf] = True

        if faces is not None:
            # Validate arguments
            assert cond is not None
            if faces.dtype == bool:
                if faces.size != self.num_faces:
                    raise ValueError(
                        """When giving logical faces, the size of
                                        array must match number of faces"""
                    )
                faces = np.argwhere(faces)
            if not np.all(np.in1d(faces, bf)):
                raise ValueError(
                    "Give boundary condition only on the \
                                 boundary"
                )
            domain_boundary_and_tips = np.argwhere(
                np.logical_or(g.tags["domain_boundary_faces"], g.tags["tip_faces"])
            )
            if not np.all(np.in1d(faces, domain_boundary_and_tips)):
                warnings.warn(
                    "You are now specifying conditions on internal \
                              boundaries. Be very careful!"
                )
            if isinstance(cond, str):
                cond = [cond] * faces.size
            if faces.size != len(cond):
                raise ValueError("One BC per face")

            for l in np.arange(faces.size):
                s = cond[l]
                if s.lower() == "neu":
                    pass  # Neumann is already default
                elif s.lower() == "dir":
                    self.is_dir[faces[l]] = True
                    self.is_neu[faces[l]] = False
                else:
                    raise ValueError("Boundary should be Dirichlet or Neumann")

class BoundaryConditionNode(object):

    """ Class to store information on boundary conditions for nodal numerical
        schemes.

    The BCs are specified by node number, and can have type Dirichlet or
    Neumann (Robin may be included later). For details on default values etc.,
    see constructor.

    Attributes:
        num_nodes (int): Number of nodes in the grid
        dim (int): Dimension of the boundary. One less than the dimension of
            the grid.
        is_neu (np.ndarray boolean, size g.num_nodes): Element i is true if
            node i has been assigned a Neumann condition. Tacitly assumes that
            the node is on the boundary. Should be false for internal nodes, as
            well as Dirichlet nodes.
        is_dir (np.ndarary, boolean, size g.num_nodes): Element i is true if
            node i has been assigned a Neumann condition.

    """

    def __init__(self, g, nodes=None, cond=None):
        """Constructor for BoundaryConditionsNode.

        The conditions are specified by node numbers. Nodes that do not get an
        explicit condition will have Neumann conditions assigned.

        Parameters:
            g (grid): For which boundary conditions are set.
            nodes (np.ndarray): Nodes for which conditions are assigned.
            cond (list of str): Conditions on the nodes, in the same order as
                used in nodes. Should be as long as nodes.

        """

        self.num_nodes = g.num_nodes
        self.dim = g.dim - 1

        self.bc_type = "scalar"

        # Find boundary nodes
        bn = g.get_all_boundary_nodes()

        # Keep track of internal boundaries
        self.is_internal = g.tags["fracture_nodes"]

        self.is_neu = np.zeros(self.num_nodes, dtype=np.bool)
        self.is_dir = np.zeros(self.num_nodes, dtype=np.bool)

        # By default, all nodes are Neumann.
        self.is_neu[bn] = True

        if nodes is not None:
            # Validate arguments
            assert cond is not None
            if nodes.dtype == bool:
                if nodes.size != self.num_nodes:
                    raise ValueError(
                        """When giving logical nodes, the size of
                                        array must match number of nodes"""
                    )
                nodes = np.argwhere(nodes)
            if not np.all(np.in1d(nodes, bn)):
                raise ValueError(
                    "Give boundary condition only on the \
                                 boundary"
                )
            domain_boundary_and_tips = np.argwhere(
                np.logical_or(g.tags["domain_boundary_nodes"], g.tags["tip_nodes"])
            )
            if not np.all(np.in1d(nodes, domain_boundary_and_tips)):
                warnings.warn(
                    "You are now specifying conditions on internal \
                              boundaries. Be very careful!"
                )
            if isinstance(cond, str):
                cond = [cond] * nodes.size
            if nodes.size != len(cond):
                raise ValueError("One BC per node")

            for l in np.arange(nodes.size):
                s = cond[l]
                if s.lower() == "neu":
                    pass  # Neumann is already default
                elif s.lower() == "dir":
                    self.is_dir[nodes[l]] = True
                    self.is_neu[nodes[l]] = False
                else:
                    raise ValueError("Boundary should be Dirichlet or Neumann")


class BoundaryConditionVectorial(object):

    """ Class to store information on boundary conditions.

        The BCs are specified by face number and assigned to the single
        component, and can have type Dirichlet or
        Neumann (Robin may be included later).
        NOTE: Currently works for boundary faces aligned with the coordinate
        system.

        For description of attributes, parameters and constructors,
        refer to the above class BoundaryCondition.

        NOTE: g.dim > 1 for the procedure to make sense

    """

    def __init__(self, g, faces=None, cond=None):

        self.num_faces = g.num_faces
        self.dim = g.dim

        self.bc_type = "vectorial"

        # Find boundary faces
        self.bf = g.get_all_boundary_faces()

        self.is_neu = np.zeros((g.dim, self.num_faces), dtype=bool)
        self.is_dir = np.zeros((g.dim, self.num_faces), dtype=bool)

        self.is_neu[:, self.bf] = True
        self.set_bc(faces, cond)

    def set_bc(self, faces, cond):

        if faces is not None:
            # Validate arguments
            assert cond is not None
            if faces.dtype == bool:
                if faces.size != self.num_faces:
                    raise ValueError(
                        """When giving logical faces, the size of
                                        array must match number of faces"""
                    )
                faces = np.argwhere(faces)

            if not np.all(np.in1d(faces, self.bf)):
                raise ValueError("Give boundary condition only on the boundary")
            if isinstance(cond, str):
                cond = [cond] * faces.size
            if faces.size != len(cond):
                raise ValueError(str(self.dim) + " BC per face")

            for j in np.arange(faces.size):
                s = cond[j]
                if s.lower() == "neu":
                    pass  # Neumann is already default
                elif s.lower() == "dir":
                    self.is_dir[:, faces[j]] = True
                    self.is_neu[:, faces[j]] = False
                elif s.lower() == "dir_x":
                    self.is_dir[0, faces[j]] = True
                    self.is_neu[0, faces[j]] = False
                    self.is_dir[1, faces[j]] = False
                    self.is_neu[1, faces[j]] = True
                    if self.dim == 3:
                        self.is_dir[2, faces[j]] = False
                        self.is_neu[2, faces[j]] = True
                elif s.lower() == "dir_y":
                    self.is_dir[0, faces[j]] = False
                    self.is_dir[1, faces[j]] = True
                    self.is_neu[0, faces[j]] = True
                    self.is_neu[1, faces[j]] = False
                    if self.dim == 3:
                        self.is_dir[2, faces[j]] = False
                        self.is_neu[2, faces[j]] = True
                elif s.lower() == "dir_xy":
                    self.is_dir[0, faces[j]] = True
                    self.is_neu[0, faces[j]] = False
                    self.is_dir[1, faces[j]] = True
                    self.is_neu[1, faces[j]] = False
                    if self.dim == 3:
                        self.is_dir[2, faces[j]] = False
                        self.is_neu[2, faces[j]] = True
                elif s.lower() == "dir_z":
                    self.is_dir[0, faces[j]] = False
                    self.is_dir[1, faces[j]] = False
                    self.is_dir[2, faces[j]] = True
                    self.is_neu[0, faces[j]] = True
                    self.is_neu[1, faces[j]] = True
                    self.is_neu[2, faces[j]] = False
                else:
                    raise ValueError("Boundary should be Dirichlet or Neumann")


def face_on_side(g, side, tol=1e-8):
    """ Find faces on specified sides of a grid.

    It is assumed that the grid forms a box in 2d or 3d.

    The faces are specified by one of two type of keywords: (xmin / west),
    (xmax / east), (ymin / south), (ymax / north), (zmin, bottom),
    (zmax / top).

    Parameters:
        g (grid): For which we want to find faces.
        side (str, or list of str): Sides for which we want to find the
            boundary faces.
        tol (double, optional): Geometric tolerance for deciding whether a face
            lays on the boundary. Defaults to 1e-8.

    Returns:
        list of lists: Outer list has one element per element in side (same
            ordering). Inner list contains global indices of faces laying on
            that side.

    """
    if isinstance(side, str):
        side = [side]

    faces = []
    for s in side:
        s = s.lower().strip()
        if s == "west" or s == "xmin":
            xm = g.nodes[0].min()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[0] - xm) < tol)))
        elif s == "east" or s == "xmax":
            xm = g.nodes[0].max()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[0] - xm) < tol)))
        elif s == "south" or s == "ymin":
            xm = g.nodes[1].min()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[1] - xm) < tol)))
        elif s == "north" or s == "ymax":
            xm = g.nodes[1].max()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[1] - xm) < tol)))
        elif s == "bottom" or s == "bot" or s == "zmin":
            xm = g.nodes[2].min()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[2] - xm) < tol)))
        elif s == "top" or s == "zmax":
            xm = g.nodes[2].max()
            faces.append(np.squeeze(np.where(np.abs(g.face_centers[2] - xm) < tol)))
        else:
            raise ValueError("Unknow face side")
    return faces
