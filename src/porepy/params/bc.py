# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:30:22 2016

@author: eke001
"""
import numpy as np
import warnings


class AbstractBoundaryCondition(object):
    """
    This is an abstract class that include the shared functionality of the
    boundary conditions
    """

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
        bc.is_per = self.is_per
        bc.per_map = self.per_map
        return bc


class BoundaryCondition(AbstractBoundaryCondition):

    """ Class to store information on boundary conditions.

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
        is_per (np.ndarray, boolean, size g.num_faces): Element i is true if
            face i has been assigned a periodic boundary condition. Note that
            periodic boundary conditions are non-standard and might not work
            for all discretizations. See also attribute per_map
        per_map (np.ndarray, int, 2 x # periodic faces): Defines the periodic
            faces. Face index per_map[0, i] is periodic with face index
            per_map[1, i].
    """

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
        self.is_per = np.zeros(self.num_faces, dtype=bool)

        # No periodic boundaries by default
        self.per_map = np.zeros((2, 0), dtype=int)

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

            for l in np.arange(faces.size):
                s = cond[l]
                if s.lower() == "neu":
                    pass  # Neumann is already default
                elif s.lower() == "dir":
                    self.is_dir[faces[l]] = True
                    self.is_neu[faces[l]] = False
                    self.is_rob[faces[l]] = False
                    self.is_per[faces[l]] = False
                elif s.lower() == "rob":
                    self.is_dir[faces[l]] = False
                    self.is_neu[faces[l]] = False
                    self.is_rob[faces[l]] = True
                    self.is_per[faces[l]] = False
                elif s.lower() == "per":
                    self.is_dir[faces[l]] = False
                    self.is_neu[faces[l]] = False
                    self.is_rob[faces[l]] = False
                    self.is_per[faces[l]] = True
                else:
                    raise ValueError("Boundary should be Dirichlet, Neumann or Robin")

            if not (self.is_per.sum() % 2) == 0:
                raise ValueError("The number of periodic boundary faces must be even!")

    def __repr__(self) -> str:
        num_cond = (
            self.is_neu.sum()
            + self.is_dir.sum()
            + self.is_rob.sum()
            + self.is_per.sum()
        )
        s = (
            f"Boundary condition for scalar problem in {self.dim + 1} dimensions\n"
            f"Grid has {self.num_faces} faces.\n"
            f"Conditions set for {num_cond} faces, out of which "
            f"{self.is_internal.sum()} are internal boundaries.\n"
            f"Number of faces with Dirichlet conditions: {self.is_dir.sum()} \n"
            f"Number of faces with Neumann conditions: {self.is_neu.sum()} \n"
            f"Number of faces with Robin conditions: {self.is_rob.sum()} \n"
            f"Number of faces with Periodic conditions: {self.is_per.sum()} \n"
        )

        bc_sum = self.is_neu + self.is_dir + self.is_rob + self.is_per
        if np.any(bc_sum) > 1:
            s += "Conflicting boundary conditions set on {np.sum(bc_sum > 1)} faces.\n"

        not_bound = np.setdiff1d(np.arange(self.num_faces), self.bf)
        if np.any(self.is_dir[not_bound]):
            s += f"Dirichlet conditions set on {self.is_dir[not_bound].sum()} non-boundary faces.\n"
        if np.any(self.is_neu[not_bound]):
            s += f"Neumann conditions set on {self.is_neu[not_bound].sum()} non-boundary faces.\n"
        if np.any(self.is_rob[not_bound]):
            s += f"Robin conditions set on {self.is_rob[not_bound].sum()} non-boundary faces.\n"
        if np.any(self.is_per[not_bound]):
            s += f"Periodic conditions set on {self.is_per[not_bound].sum()} non-boundary faces.\n"

        return s

    def set_periodic_map(self, per_map: np.ndarray):
        """
        Set the index map between periodic boundary faces. The mapping assumes
        a one to one mapping between the periodic boundary faces (i.e., matching
        faces).

        Parameters
        per_map (np.ndarray, int, 2 x # periodic faces): Defines the periodic
            faces. Face index per_map[0, i] is periodic with face index
            per_map[1, i]. The given map is stored to the attribute per_map
        """
        map_shape = (2, self.is_per.sum() // 2)
        if not np.array_equal(per_map.shape, map_shape):
            raise ValueError(
                """Periodic map has wrong size. Given array size is: {},
                              but should be: {}""".format(
                    per_map.shape, map_shape
                )
            )
        self.per_map = per_map
