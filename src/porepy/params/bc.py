# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 14:30:22 2016

@author: eke001
"""

import numpy as np

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

        self.bc_type = 'scalar'

        # Find boundary faces
        bf = g.get_boundary_faces()

        self.is_neu = np.zeros(self.num_faces, dtype=bool)
        self.is_dir = np.zeros(self.num_faces, dtype=bool)

        # By default, all faces are Neumann.
        self.is_neu[bf] = True

        if faces is not None:
            # Validate arguments
            assert cond is not None
            if not np.all(np.in1d(faces, bf)):
                raise ValueError('Give boundary condition only on the boundary')
            if faces.size != len(cond):
                raise ValueError('One BC per face')

            for l in np.arange(faces.size):
                s = cond[l]
                if s.lower() == 'neu':
                    pass  # Neumann is already default
                elif s.lower() == 'dir':
                    self.is_dir[faces[l]] = True
                    self.is_neu[faces[l]] = False
                else:
                    raise ValueError('Boundary should be Dirichlet or Neumann')

                
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

        self.bc_type = 'vectorial'

        # Find boundary faces
        bf = g.get_boundary_faces()

        self.is_neu = np.zeros((g.dim, self.num_faces), dtype=bool)
        self.is_dir = np.zeros((g.dim, self.num_faces), dtype=bool)

        self.is_neu[:, bf] = True

        if faces is not None:
            # Validate arguments
            assert cond is not None
            if not np.all(np.in1d(faces, bf)):
                raise ValueError('Give boundary condition only on the boundary')
            if faces.size != len(cond):
                raise ValueError('One BC per face')

            for j in np.arange(faces.size):
                s = cond[j]
                if s.lower() == 'neu':
                    pass  # Neumann is already default
                elif s.lower() == 'dir':
                    self.is_dir[:, faces[j]] = True
                    self.is_neu[:, faces[j]] = False
                elif s.lower() == 'dir_x':
                    self.is_dir[0, faces[j]] = True
                    self.is_neu[0, faces[j]] = False
                    self.is_dir[1, faces[j]] = False
                    self.is_neu[1, faces[j]] = True
                    if self.dim == 3:
                        self.is_dir[2, faces[j]] = False
                        self.is_neu[2, faces[j]]= True
                elif s.lower() == 'dir_y':
                    self.is_dir[0, faces[j]] = False
                    self.is_dir[1, faces[j]] = True
                    self.is_neu[0, faces[j]] = True
                    self.is_neu[1, faces[j]] = False
                    if self.dim == 3:
                        self.is_dir[2, faces[j]] = False
                        self.is_neu[2, faces[j]]= True
                elif s.lower() == 'dir_z':
                    self.is_dir[0, faces[j]] = False
                    self.is_dir[1, faces[j]] = False
                    self.is_dir[2, faces[j]] = True
                    self.is_neu[0, faces[j]] = True
                    self.is_neu[1, faces[j]] = True
                    self.is_neu[2, faces[j]] = False
                else:
                    raise ValueError('Boundary should be Dirichlet or Neumann')


