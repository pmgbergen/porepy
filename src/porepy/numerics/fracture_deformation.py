"""
Module for initializing, solve, and save a fracture deformation law.
"""
import numpy as np
import scipy.sparse as sps
import time
import logging

from porepy.grids.grid_bucket import GridBucket
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.viz.exporter import Exporter


# Module-wide logger
logger = logging.getLogger(__name__)


class FrictionSlipModel:
    """
    Class for solving a frictional slip problem: T_s <= mu * (T_n -p)
    

    Parameters in Init:
    gb: (Grid) a Grid Object.
    data: (dictionary) Should contain a Parameter class with the keyword
        'Param'
    physics: (string): defaults to 'slip'

    Functions:
    solve(): Calls reassemble and solves the linear system.
             Returns: new slip if T_s > mu * (T_n - p)
             Sets attributes: self.x, self.is_slipping, self.d_n
    step(): Same as solve
    normal_shear_traction(): project the traction into the corresponding
                             normal and shear components
    grid(): Returns: the Grid or GridBucket
    data(): Returns: Data dictionary
    fracture_dilation(slip_distance): Returns: the amount of dilation for given
                                               slip distance
    slip_distance(): saves the slip distance to the data dictionary and
                     returns it
    aperture_change(): saves the aperture change to the data dictionary and
                       returns it
    mu(faces): returns: the coefficient of friction
    gamma(): returns: the numerical step length parameter
    save(): calls split('pressure'). Then export the pressure to a vtk file to the
            folder kwargs['folder_name'] with file name
            kwargs['file_name'], default values are 'results' for the folder and
            physics for the file name.
    """

    def __init__(self, gb, data, physics="slip", **kwargs):
        self.physics = physics
        if isinstance(gb, GridBucket):
            raise ValueError("FrictionSlip excpected a Grid, not a GridBucket")

        self._gb = gb
        self._data = data

        file_name = kwargs.get("file_name", physics)
        folder_name = kwargs.get("folder_name", "results")

        tic = time.time()
        logger.info("Create exporter")
        self.exporter = Exporter(self._gb, file_name, folder_name)
        logger.info("Elapsed time: " + str(time.time() - tic))

        self.x = np.zeros((3, gb.num_faces))
        self.d_n = np.zeros(gb.num_faces)

        self.is_slipping = np.zeros(gb.num_faces, dtype=np.bool)

        self.slip_name = "slip_distance"
        self.aperture_name = "aperture_change"

    def solve(self):
        """ Linearize and solve corresponding system

        First, the function calculate if the slip-criterion is satisfied for
        each face: T_s <= mu * (T_n - p).
        If this is violated, the fracture is slipping. It estimates the slip
        in direction of shear traction as:
        d += T_s - mu(T_n - p) * sqrt(face_area) / G.

        Stores this result in self.x which is a ndarray of dimension
        (3, number of faces). Also updates the ndarray self.is_slipping to
        True for any face that violated the slip-criterion.

        Requires the following keywords in the data dictionary:
        'face_pressure': (ndarray) size equal number of faces in the grid.
                         Only the pressure on the fracture faces are used, and
                         should be equivalent to the pressure in the
                        pressure in the corresponding lower dimensional cells.
        'traction': (ndarray) size (3, number_of_faces). This should be the 
                    area scaled traction on each face.
        'rock': a Rock Object with shear stiffness Rock.MU defined.

        Returns:
            new_slip (bool) returns True if the slip vector was violated for
                     any faces
        """
        assert self._gb.dim == 3, "only support for 3D (yet)"

        frac_faces = self._gb.frac_pairs
        fi = frac_faces[1]
        fi_left = frac_faces[0]
        T_n, T_s, n, t = self.normal_shear_traction(fi)

        assert np.all(T_s > -1e-10)
        assert np.all(T_n < 0), "Must have a normal force on the fracture"

        # we find the effective normal stress on the fracture face.
        # Here we need to multiply T_n with -1 as we want the absolute value,
        # and all the normal tractions should be negative.
        sigma_n = -T_n - self._data["face_pressure"][fi]
        #        assert np.all(sigma_n > 0 )

        # new slip are fracture faces slipping in this iteration
        new_slip = (
            T_s - self.mu(fi, self.is_slipping[fi]) * sigma_n
            > 1e-5 * self._data["rock"].MU
        )

        self.is_slipping[fi] = self.is_slipping[fi] | new_slip

        # calculate the shear stiffness
        shear_stiffness = np.sqrt(self._gb.face_areas[fi]) / (self._data["rock"].MU)

        # calculate aproximated slip distance
        excess_shear = np.abs(T_s) - self.mu(fi, self.is_slipping[fi]) * sigma_n
        slip_d = excess_shear * shear_stiffness * self.gamma() * new_slip

        # We also add the values to the left cells so that when we average the
        # face values to obtain a cell value, it will equal the face value

        slip_vec = -t * slip_d - n * self.fracture_dilation(slip_d, fi)

        self.d_n[fi] += self.fracture_dilation(slip_d, fi)
        self.d_n[fi_left] += self.fracture_dilation(slip_d, fi_left)

        assert np.all(self.d_n[fi] > -1e-6)

        self.x[:, fi] += slip_vec
        self.x[:, fi_left] -= slip_vec

        return new_slip

    def normal_shear_traction(self, faces=None):
        """
        Project the traction vector into the normal and tangential components
        as seen from the fractures. 
        Requires that the data dictionary has keyword:
        traction:  (ndarray) size (3, number of faces). giving the area
                   weighted traction on each face.
        
        Returns:
        --------
        T_n:  (ndarray) size (number of fracture_cells) the normal traction on
              each fracture.
        T_s:  (ndarray) size (number of fracture_cells) the shear traction on
              each fracture.
        normals: (ndarray) size (3, number of fracture_cells) normal vector,
            i.e., the direction of normal traction
        tangents: (ndarray) size (3, number of fracture_cells) tangential
            vector, i.e., the direction of shear traction
        """

        if faces is None:
            frac_faces = self._gb.frac_pairs
            fi = frac_faces[1]
        else:
            fi = faces

        assert self._gb.dim == 3
        T = self._data["traction"].copy()
        T = T / self._gb.face_areas

        sgn = sign_of_faces(self._gb, fi)
        # sgn_test = g.cell_faces[fi, ci]

        T = sgn * T[:, fi]
        normals = sgn * self._gb.face_normals[:, fi] / self._gb.face_areas[fi]
        assert np.allclose(np.sqrt(np.sum(normals ** 2, axis=0)), 1)

        T_n = np.sum(T * normals, axis=0)
        tangents = T - T_n * normals
        T_s = np.sqrt(np.sum(tangents ** 2, axis=0))
        tangents = tangents / np.sqrt(np.sum(tangents ** 2, axis=0))
        assert np.allclose(np.sqrt(np.sum(tangents ** 2, axis=0)), 1)
        assert np.allclose(T, T_n * normals + T_s * tangents)
        # Sanity check:
        frac_faces = self._gb.frac_pairs
        trac = self._data["traction"].copy()
        fi_left = frac_faces[0]
        sgn_left = sign_of_faces(self._gb, fi_left)
        sgn_right = sign_of_faces(self._gb, fi)
        T_left = sgn_left * trac.reshape((3, -1), order="F")[:, fi_left]
        T_right = sgn_right * trac.reshape((3, -1), order="F")[:, fi]
        assert np.allclose(T_left, -T_right)

        # TESTING DONE

        return T_n, T_s, normals, tangents

    def fracture_dilation(self, distance, _):
        """
        defines the fracture dilation as a function of slip distance
        Parameters:
        ----------
        distance: (ndarray) the slip distances

        Returns:
        ---------
        dilation: (ndarray) the corresponding normal displacement of fractures.
        """

        phi = 1 * np.pi / 180
        return distance * np.tan(phi)

    def mu(self, faces, slip_faces=[]):
        """
        Coefficient of friction.
        Parameters:
        ----------
        faces: (ndarray) indexes of fracture faces
        slip_faces: (ndarray) optional, defaults to []. Indexes of faces that
                    are slipping ( will be given a dynamic friciton).
        returns:
        mu: (ndarray) the coefficient of each fracture face.
        """
        mu_d = 0.55
        mu_ = 0.6 * np.ones(faces.size)
        mu_[slip_faces] = mu_d
        return mu_

    def gamma(self):
        """
        Numerical step length parameter. Defines of far a fracture violating 
        the slip-condition should slip.
        """
        return 2

    def step(self):
        """
        calls self.solve()
        """
        return self.solve()

    def grid(self):
        """
        returns model grid
        """
        return self._gb

    def data(self):
        """
        returns data
        """
        return self._data

    def slip_distance(self, slip_name="slip_distance"):
        """
        Save the slip distance to the data dictionary. The slip distance
        will be saved as a (3, self.grid().num_faces) array
        Parameters: 
        -----------
        slip_name:    (string) Defaults to 'slip_distance'. Defines the
                               keyword for the saved slip distance in the data
                               dictionary
        Returns:
        --------
        d:  (ndarray) the slip distance as a (3, self.grid().num_faces) array
        """
        self.slip_name = slip_name
        self._data[self.slip_name] = self.x
        return self.x

    def aperture_change(self, aperture_name="aperture_change"):
        """
        Save the aperture change to the data dictionary. The aperture change
        will be saved as a (self.grid().num_faces) array
        Parameters: 
        -----------
        slip_name:    (string) Defaults to 'aperture_name'. Defines the
                               keyword for the saved aperture change in the data
                               dictionary
        Returns:
        --------
        d:  (ndarray) the change in aperture as a (self.grid().num_faces) array
        """
        self.aperture_name = aperture_name
        self._data[self.aperture_name] = self.d_n
        return self.d_n

    def save(self, variables=None, save_every=None):
        """
        Save the result as vtk. 

        Parameters:
        ----------
        variables: (list) Optional, defaults to None. If None, only the grid
            will be exported. A list of strings where each element defines a
            keyword in the data dictionary to be saved.
        time_step: (float) optinal, defaults to None. The time step of the
            variable(s) that is saved
        """
        if variables is None:
            self.exporter.write_vtk()
        else:
            variables = {k: self._data[k] for k in variables if k in self._data}
            self.exporter.write_vtk(variables)


# ------------------------------------------------------------------------------#
class FrictionSlipDataAssigner:
    """
    Class for setting data to a slip problem:
    T_s <= mu (T_n - p)
    This class creates a Parameter object and assigns the data to this object
    by calling FricitonSlipDataAssigner's functions.

    To change the default values, create a class that inherits from
    FrictionSlipDataAssigner. Then overload the values you whish to change.

    Parameters in Init:
    gb: (Grid) a grid object 
    data: (dictionary) Dictionary which Parameter will be added to with keyword
          'param'
    physics: (string): defaults to 'mechanics'

    Functions that assign data to Parameter class:
        bc(): defaults to neumann boundary condition
             Returns: (Object) boundary condition
        bc_val(): defaults to 0
             returns: (ndarray) boundary condition values
        stress_tensor(): defaults to 1
             returns: (tensor.FourthOrder) Stress tensor

    Utility functions:
        grid(): returns: the grid

    """

    def __init__(self, g, data, physics="slip"):
        self._g = g
        self._data = data
        self.physics = physics
        self._set_data()

    def data(self):
        return self._data

    def grid(self):
        return self._g

    def _set_data(self):
        if "param" not in self._data:
            self._data["param"] = Parameters(self.grid())


# -----------------------------------------------------------------------------#


def sign_of_faces(g, faces):
    """
    returns the sign of faces as defined by g.cell_faces. 
    Parameters:
    g: (Grid Object)
    faces: (ndarray) indices of faces that you want to know the sign for. The 
           faces must be boundary faces.

    Returns:
    sgn: (ndarray) the sign of the faces
    """

    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn = sps.find(g.cell_faces[faces[IA], :])
    assert fi.size == faces.size, "sign of internal faces does not make sense"
    I = np.argsort(fi)
    sgn = sgn[I]
    sgn = sgn[IC]
    return sgn
