# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:30:22 2016

@author: eke001
"""
import warnings

import numpy as np

import porepy as pp

module_sections = ["parameters"]


class AbstractBoundaryCondition(object):
    """
    This is an abstract class that include the shared functionality of the
    boundary conditions
    """

    @pp.time_logger(sections=module_sections)
    def copy(self):
        """
        Create a deep copy of the boundary condition.

        Returns:
            bc: A deep copy of self. All attributes will also be copied.

        """
        # We don't call the init since we don't have access to the grid.
        # Maybe this is bad style.
        bc = BoundaryCondition.__new__(BoundaryCondition)
        bc.bc_type = self.bc_type
        bc.is_neu = self.is_neu
        bc.is_dir = self.is_dir
        bc.is_rob = self.is_rob
        bc.basis = self.basis
        bc.robin_weight = self.robin_weight
        bc.num_faces = self.num_faces
        bc.dim = self.dim
        bc.is_internal = self.is_internal
        bc.bf = self.bf
        return bc


class BoundaryCondition(AbstractBoundaryCondition):

    """Class to store information on boundary conditions for problems of a single
    variable.

    The BCs are specified by face number, and can have type Dirichlet, Neumann
    or Robin. For details on default values etc. see constructor.

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
        is_rob (np.ndarray, boolean, size g.num_faces): Element i is true if
            face i has been assigned a Robin condition.
    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, g, faces=None, cond=None):
        """Constructor for BoundaryCondition.

        The conditions are specified by face numbers. Faces that do not get an
        explicit condition will have Neumann conditions assigned.

        Parameters:
            g (grid): For which boundary conditions are set.
            faces (np.ndarray): Faces for which conditions are assigned.
            cond (list of str): Conditions on the faces, in the same order as
                used in faces. Should be as long as faces. The list elements
                should be one of "dir", "neu", "rob".

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
        self.bf = g.get_all_boundary_faces()

        # Keep track of internal boundaries
        self.is_internal = g.tags["fracture_faces"]

        self.is_neu = np.zeros(self.num_faces, dtype=bool)
        self.is_dir = np.zeros(self.num_faces, dtype=bool)
        self.is_rob = np.zeros(self.num_faces, dtype=bool)

        # By default, all faces are Neumann.
        self.is_neu[self.bf] = True

        # Set robin weight
        self.robin_weight = np.ones(g.num_faces)
        # Basis is mostly here to be consistent with vectorial. If changing the
        # basis to -1 it should be possible to define innflow as positive, but this
        # has not been tested
        self.basis = np.ones(g.num_faces)

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

            for ind in np.arange(faces.size):
                s = cond[ind]
                if s.lower() == "neu":
                    pass  # Neumann is already default
                elif s.lower() == "dir":
                    self.is_dir[faces[ind]] = True
                    self.is_neu[faces[ind]] = False
                    self.is_rob[faces[ind]] = False
                elif s.lower() == "rob":
                    self.is_dir[faces[ind]] = False
                    self.is_neu[faces[ind]] = False
                    self.is_rob[faces[ind]] = True
                else:
                    raise ValueError("Boundary should be Dirichlet, Neumann or Robin")

    @pp.time_logger(sections=module_sections)
    def __repr__(self) -> str:
        num_cond = self.is_neu.sum() + self.is_dir.sum() + self.is_rob.sum()
        s = (
            f"Boundary condition for scalar problem in {self.dim + 1} dimensions\n"
            f"Grid has {self.num_faces} faces.\n"
            f"Conditions set for {num_cond} faces, out of which "
            f"{self.is_internal.sum()} are internal boundaries.\n"
            f"Number of faces with Dirichlet conditions: {self.is_dir.sum()} \n"
            f"Number of faces with Neumann conditions: {self.is_neu.sum()} \n"
            f"Number of faces with Robin conditions: {self.is_rob.sum()} \n"
        )

        bc_sum = self.is_neu + self.is_dir + self.is_rob
        if np.any(bc_sum) > 1:
            s += "Conflicting boundary conditions set on {np.sum(bc_sum > 1)} faces.\n"

        not_bound = np.setdiff1d(np.arange(self.num_faces), self.bf)
        if np.any(self.is_dir[not_bound]):
            s += f"Dirichlet conditions set on {self.is_dir[not_bound].sum()}"
            s += " non-boundary faces.\n"
        if np.any(self.is_neu[not_bound]):
            s += f"Neumann conditions set on {self.is_neu[not_bound].sum()}"
            s += " non-boundary faces.\n"
        if np.any(self.is_rob[not_bound]):
            s += f"Robin conditions set on {self.is_rob[not_bound].sum()} "
            s += "non-boundary faces.\n"

        return s


class BoundaryConditionVectorial(AbstractBoundaryCondition):
    """
    Class to store information on boundary conditions for problems with vector variables
    (e.g. momentuum conservation).

    The BCs are specified by face number and assigned to the single
    component, and can have type Dirichlet,
    Neumann or Robin.

    The Robin condition is defined by
        sigma*n + alpha * u = G
    where alpha is defined by the attribute self.robin_weight

    The boundary conditions are applied in the basis given by the attribute
    self.basis (defaults to the coordinate system). The basis is defined face-wise,
    and the boundary condition should be given in the coordinates of these basis.

    For description of attributes, parameters and constructors,
    refer to the above class BoundaryCondition.

    NOTE: g.dim > 1 for the procedure to make sense

    Attributes:
        num_faces (int): Number of faces in the grid
        dim (int): Dimension of the boundary. One less than the dimension of
            the grid.
        is_neu (np.ndarray boolean, size g.dim x g.num_faces): Element i is true if
            face i has been assigned a Neumann condition. Tacitly assumes that
            the face is on the boundary. Should be false for internal faces, as
            well as Dirichlet faces.
        is_dir (np.ndarary, boolean, size g.dim x g.num_faces): Element i is true if
            face i has been assigned a Neumann condition.
        is_rob (np.ndarray, boolean, size g.dim x g.num_faces): Element i is true if
            face i has been assigned a Robin condition.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, g, faces=None, cond=None):
        """Constructor for BoundaryConditionVectorial.

        The conditions are specified by face numbers. Faces that do not get an
        explicit condition will have Neumann conditions assigned.

        Parameters:
            g (grid): For which boundary conditions are set.
            faces (np.ndarray): Faces for which conditions are assigned.
            cond (list of str): Conditions on the faces, in the same order as
                used in faces. Should be as long as faces. To set uniform condition
                in all spatial directions for a face, use 'dir', 'neu', or 'rob'.

            NOTE: For more general combinations of boundary conditions, it is
            recommended to first construct a BoundaryConditionVectorial object,
            and then access the attributes is_dir, is_neu, is_rob to set the
            conditions.

        Example:
            # Assign Dirichlet condititons on the left side of a grid; implicit
            # Neumann conditions on the rest
            g = pp.CartGrid([2, 2])
            west_face = pp.bc.face_on_side(g, 'west')
            bound_cond = pp.BoundaryConditionVectorial(g, faces=west_face, cond=['dir',
                                                                                 'dir'])

        Example:
            Assign Dirichlet condition in the x-direction, Robin in the z-direction.
            g = pp.CartGrid([2, 2, 2])
            bc = pp.BoundaryConditionVectorial(g)
            target_face = 0
            bc.is_neu[[0, 2], target_face] = False
            bc.is_dir[0, target_face] = True
            bc.is_rob[2, target_face] = True

        """

        self.num_faces = g.num_faces
        self.dim = g.dim

        self.bc_type = "vectorial"

        # Keep track of internal boundaries
        self.is_internal = g.tags["fracture_faces"]
        # Find boundary faces
        self.bf = g.get_all_boundary_faces()

        self.is_neu = np.zeros((g.dim, self.num_faces), dtype=bool)
        self.is_dir = np.zeros((g.dim, self.num_faces), dtype=bool)
        self.is_rob = np.zeros((g.dim, self.num_faces), dtype=bool)

        self.is_neu[:, self.bf] = True
        self.set_bc(faces, cond)

        #  Default robin weights
        r_w = np.tile(np.eye(g.dim), (1, g.num_faces))
        self.robin_weight = np.reshape(r_w, (g.dim, g.dim, g.num_faces), "F")
        basis = np.tile(np.eye(g.dim), (1, g.num_faces))
        self.basis = np.reshape(basis, (g.dim, g.dim, g.num_faces), "F")

    @pp.time_logger(sections=module_sections)
    def __repr__(self) -> str:
        s = (
            f"Boundary condition for vectorial problem in {self.dim} dimensions\n"
            f"Conditions set for {self.bf.size} faces, out of which "
            f"{self.is_internal.sum()} are internal boundaries.\n"
        )

        only_neu = np.sum(np.all(self.is_neu, axis=0))
        only_dir = np.sum(np.all(self.is_dir, axis=0))
        only_rob = np.sum(np.all(self.is_rob, axis=0))

        neu_or_dir = (
            np.sum(np.all(np.logical_or(self.is_dir, self.is_neu), axis=0))
            - only_dir
            - only_neu
        )

        s += (
            f"Number of faces with all Dirichlet conditions: {only_dir} \n"
            f"Number of faces with all Neumann conditions: {only_neu} \n"
            f"Number of faces with all Robin conditions: {only_rob} \n"
            f"Number of faces with combination of Dirichlet and Neumann {neu_or_dir}\n"
        )

        bc_sum = np.sum(self.is_neu + self.is_dir + self.is_rob, axis=0)
        if np.any(bc_sum) > self.dim:
            s += "Conflicting boundary conditions set on {np.sum(bc_sum > 1)} faces.\n"

        not_bound = np.setdiff1d(np.arange(self.num_faces), self.bf)
        if np.any(self.is_dir[:, not_bound]):
            s += (
                f"Dirichlet conditions set on "
                f"{self.is_dir[:, not_bound].any(axis=0).sum()} non-boundary faces.\n"
            )
        if np.any(self.is_neu[:, not_bound]):
            s += (
                f"Neumann conditions set on "
                f"{self.is_neu[:, not_bound].any(axis=0).sum()} non-boundary faces.\n"
            )
        if np.any(self.is_rob[:, not_bound]):
            s += (
                f"Robin conditions set on "
                f"{self.is_rob[:, not_bound].any(axis=0).sum()} non-boundary faces.\n"
            )

        return s

    @pp.time_logger(sections=module_sections)
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
                elif s.lower() == "rob":
                    self.is_rob[:, faces[j]] = True
                    self.is_neu[:, faces[j]] = False
                    self.is_dir[:, faces[j]] = False
                else:
                    raise ValueError(f"Unknown boundary condition {s}")


@pp.time_logger(sections=module_sections)
def face_on_side(g, side, tol=1e-8):
    """Find faces on specified sides of a grid.

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
