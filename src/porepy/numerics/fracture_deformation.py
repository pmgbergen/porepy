'''
Module for initializing, solve, and save a fracture deformation law.
'''
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



class FrictionSlip():
    '''
    Class for solving a frictional slip problem:
    

    Parameters in Init:
    gb: (Grid /GridBucket) a grid or grid bucket object. If gb = GridBucket
        a Parameter class should be added to each grid bucket data node with
        keyword 'param'.
    data: (dictionary) Defaults to None. Only used if gb is a Grid. Should
          contain a Parameter class with the keyword 'Param'
    physics: (string): defaults to 'flow'

    Functions:
    solve(): Calls reassemble and solves the linear system.
             Returns: the pressure p.
             Sets attributes: self.x
    step(): Same as solve, but without reassemble of the matrices
    reassemble(): Assembles the lhs matrix and rhs array.
            Returns: lhs, rhs.
            Sets attributes: self.lhs, self.rhs
    source_disc(): Defines the discretization of the source term.
            Returns Source discretization object
    flux_disc(): Defines the discretization of the flux term.
            Returns Flux discretization object (E.g., Tpfa)
    grid(): Returns: the Grid or GridBucket
    data(): Returns: Data dictionary
    split(name): Assignes the solution self.x to the data dictionary at each
                 node in the GridBucket.
                 Parameters:
                    name: (string) The keyword assigned to the pressure
    discharge(): Calls split('p'). Then calculate the discharges over each
                 face in the grids and between edges in the GridBucket
    save(): calls split('p'). Then export the pressure to a vtk file to the
            folder kwargs['folder_name'] with file name
            kwargs['file_name'], default values are 'results' for the folder and
            physics for the file name.
    '''

    def __init__(self, gb, data, physics='slip', **kwargs):
        self.physics = physics
        if isinstance(gb, GridBucket):
            raise ValueError('FrictionSlip excpected a Grid, not a GridBucket')

        self._gb = gb
        self._data = data

        file_name = kwargs.get('file_name', physics)
        folder_name = kwargs.get('folder_name', 'results')

        tic = time.time()
        logger.info('Create exporter')
        self.exporter = Exporter(self._gb, file_name, folder_name)
        logger.info('Elapsed time: ' + str(time.time() - tic))

        self.x = np.zeros((3, gb.num_faces))
        self.is_slipping = np.zeros(gb.num_faces, dtype=np.bool)
        self.slip_name = 'slip_distance'

    def solve(self):
        """ Reassemble and solve linear system.

        After the funtion has been called, the attributes lhs and rhs are
        updated according to the parameter states. Also, the attribute x
        gives the pressure given the current state.

        TODO: Provide an option to save solver information if multiple
        systems are to be solved with the same left hand side.

        The function attempts to set up the best linear solver based on the
        system size. The setup and parameter choices here are still
        experimental.

        Parameters:
            max_direct (int): Maximum number of unknowns where a direct solver
                is applied. If a direct solver can be applied this is usually
                the most efficient option. However, if the system size is
                too large compared to available memory, a direct solver becomes
                extremely slow.
            callback (boolean, optional): If True iteration information will be
                output when an iterative solver is applied (system size larger
                than max_direct)

        Returns:
            np.array: Pressure state.

        """
        assert self._gb.dim == 3, 'only support for 3D (yet)'

        frac_faces = self._gb.frac_pairs
        fi = frac_faces[1]
        T_n, T_s, n, t = self.normal_shear_traction(fi)

        assert np.all(T_s > -1e-10)
        assert np.all(T_n < 0), 'Must have a normal force on the fracture'
        # we find the effective normal stress on the fracture face.
        # Here we need to multiply T_n with -1 as we want the absolute value,
        # and all the normal tractions should be negative.
        p = self._data['face_pressure'][fi]
        sigma_n = -T_n - p

        new_slip = T_s - self.mu(fi, self.is_slipping[fi]) * sigma_n >1e-5 * self._data['param'].shear_modulus
       
        self.is_slipping[fi] = self.is_slipping[fi] | new_slip
        excess_shear = np.abs(T_s) - self.mu(fi, self.is_slipping[fi]) * sigma_n

        shear_stiffness = np.sqrt(self._gb.face_areas[fi]) / (self._data['param'].shear_modulus)
        slip_d = excess_shear * shear_stiffness * self.gamma * new_slip

        slip_vec = t * slip_d + n * self.fracture_dialation(slip_d)

        self.x[:, fi] += slip_vec

        return new_slip

    def normal_shear_traction(self, faces=None):
        if faces is None:
            frac_faces = self._gb.frac_pairs
            fi = frac_faces[1]
        else:
            fi = faces

        assert self._gb.dim == 3
        T = self._data['traction'].copy()
        T_area = np.ones((self._gb.dim, self._gb.num_faces))
        T_area *= self._gb.face_areas
        T = T / T_area.ravel('F')

        sgn = sign_of_faces(self._gb, fi)
        #sgn_test = g.cell_faces[fi, ci]

        T = T.reshape((3, -1), order='F')
        T = sgn * T[:, fi]
        normals = sgn * self._gb.face_normals[:, fi] / self._gb.face_areas[fi]
        assert np.allclose(np.sqrt(np.sum(normals**2, axis=0)), 1)

        # T_b = np.zeros(T.shape)
        # sigma = self.background_stress()
        # for i in range(normals.shape[1]):
        #     T_b[:, i] = np.dot(normals[:, i], sigma)

        T_n = np.sum(T * normals, axis=0)
        tangents = T - T_n * normals
        T_s = np.sqrt(np.sum(tangents**2, axis=0))

        tangents = tangents / np.sqrt(np.sum(tangents**2, axis=0))
        assert np.allclose(T, T_n * normals + T_s * tangents)
        # TESTING
        frac_faces = self._gb.frac_pairs
        trac = self._data['traction'].copy()
        fi_left = frac_faces[0]
        sgn_left = sign_of_faces(self._gb, fi_left)
        sgn_right = sign_of_faces(self._gb, fi)
        T_left = sgn_left * \
            trac.reshape((3, -1), order='F')[:, fi_left]
        T_right = sgn_right * \
            trac.reshape((3, -1), order='F')[:, fi]
        if not np.allclose(T_left, -T_right):#, atol=1e-6 * np.max(T_left)):
            import pdb
            pdb.set_trace()

        assert np.allclose(T_left, -T_right)#, atol=1e-6 * np.max(np.abs(T_left)))

        # TESTING DONE

        return T_n, T_s, normals, tangents

    def fracture_dialation(self, distance):
        phi = 3 * np.pi / 180
        return distance * np.tan(phi)

    def mu(self, faces, slip_faces=[]):
        mu_d = 0.4
        mu_ = 0.6 * np.ones(faces.size)
        mu_[slip_faces] = mu_d
        return mu_

    def gamma(self):
        return 3

    def step(self):
        return self.solve()

    def grid(self):
        return self._gb

    def data(self):
        return self._data

    def slip_distance(self, slip_name='slip_distance'):
        self.slip_name = slip_name
        self._data[self.slip_name] = self.x

    def save(self, variables=None, save_every=None):
        if variables is None:
            self.exporter.write_vtk()
        else:
            variables = {k: self._data[k] for k in variables \
                         if k in self._data}
            self.exporter.write_vtk(variables)


#------------------------------------------------------------------------------#


def sign_of_faces(g, faces):
    IA = np.argsort(faces)
    IC = np.argsort(IA)

    fi, _, sgn = sps.find(g.cell_faces[faces[IA], :])
    assert fi.size == faces.size, 'sign of internal faces does not make sense'
    I = np.argsort(fi)
    sgn = sgn[I]
    sgn = sgn[IC]
    return sgn
