""" Contains class for storing data / parameters associated with a grid.
"""
import numpy as np


class Data(object):
    """ Class to store all physical parameters used by solvers.

    The intention is to provide a unified way of passing around parameters, and
    also circumvent the issue of using a solver for multiple physical
    processes (e.g. different types of boundary conditions in multi-physics
    applications).

    List of possible attributes (implemented as properties, see respective
    getter methods for further description):
        aperture (fracture width)
        perm (permeability tensor)
        conductivity (heat conductivity tensor)
        stiffness (Elastic stiffness tensor)
        bc: bc_flow, bc_transport, bc_mech (boundary condition object)
        sources: source_flow, source_transport (cell-wise sources)

    Solvers will access data as needed. If a solver inquires for unassigned
    data, this will result in a runtime error.

    """

    def __init__(self, g):
        """ Initialize Data object.

        """
        self._num_cells = g.num_cells
        self._num_faces = g.num_faces

    def __repr__(self):
        s = 'Data object for grid with ' + str(self._num_cells)
        s += ' cells and ' + str(self._num_faces) + ' faces \n'
        s += 'Assigned attributes / properties: \n'
        s += str(list(self.__dict__.keys()))
        return s

#------------------ Aperture -----------------

    def _get_aperture(self):
        """ double or array_like
        Cell-wise quantity representing fracture aperture (really, height of
        surpressed dimensions). Set as either a np.ndarray, or a scalar
        (uniform value. Always
        returned as np.ndarray.
        """
        if isinstance(self._aperture, np.ndarray):
            # Hope that the user did not initialize as array with wrong size
            return self._aperture
        else:
            return self._aperture * np.ones(self._num_cells)
    def _set_aperture(self, val):
        if (isinstance(val, np.ndarray) and np.any(val<0)) or val < 0:
            raise ValueError('Negative aperture')
        self._aperture = val

    aperture = property(_get_aperture, _set_aperture)

#------------------- Sources ---------------------------------------------

    def sources(self, obj):
        """ Pick out solver-specific source.

        Discretization methods should access this method.

        Parameters:

        obj : Solver
            Discretization object. Should have attribute 'physics'.

        Returns:

        second order tensor
            Volume source if obj.physics equals 'flow' or 'pressure',
            Heat source if obj.physics equals 'heat' or 'transport'.

        """
        if obj.physics.lower().strip() in ['flow', 'pressure']:
            return self._source_flow
        elif obj.physics.lower().strip() in ['heat', 'transport']:
            return self._source_transport

    def _get_source_flow(self):
        """ array_like
        Cell-wise quantity representing the volume source in a cell. Represent
        total in/outflow in the cell (integrated over the cell volume).
        Solvers should rather access the function source().
        """
        return self._source_flow
    def _set_source_flow(self, arr):
        self._source_flow = arr
    source_flow = property(_get_source_flow, _set_source_flow)

    def _get_source_transport(self):
        """ array_like
        Cell-wise quantity representing the concentration / temperature source
        in a cell. Represent total in/outflow in the cell (integrated over the
        cell volume).
        Solvers should rather access the function source().
        """
        return self._source_transport
    def _set_source_transport(self, arr):
        self._source_transport = arr
    source_transport= property(_get_source_transport, _set_source_transport)

#-------------------- Permeability and conductivity ---------------------

    def tensor(self, obj):
        """ Pick out solver-specific second order tensor.

        Discretization methods for second order elliptic equations should
        access this method (instead of perm or conductivity).

        Parameters:

        obj : Solver
            Discretization object. Should have attribute 'physics'.

        Returns:

        second order tensor
            Permeability if obj.physics equals 'flow' or 'pressure',
            conductivity if obj.physics equals 'heat' or 'transport'.

        """
        if obj.physics.lower().strip() in ['flow', 'pressure']:
            return self.perm
        elif obj.physics.lower().strip() in ['heat', 'transport']:
            return self.conductivity

    def _get_perm(self):
        """ tensor.SecondOrder
        Cell wise permeability, represented as a second order tensor.
        Solvers should rather access tensor().
        """
        return self._perm
    def _set_perm(self, ten):
        self._perm = ten

    perm = property(_get_perm, _set_perm)

    def _get_conductivity(self):
        """ tensor.SecondOrder
        Cell wise conductivity, represented as a second order tensor.
        Solvers should rather access tensor().
        """
        return self._conductivity
    def _set_conductivity(self, ten):
        self._conductivity = ten

    conductivity = property(_get_conductivity, _set_conductivity)

#--------------------- Stiffness -------------------------------------

    def _stiffness_getter(self):
        """ Stiffness matrix, defined as fourth order tensor
        """
        return self._stiffness
    def _stiffness_setter(self, val):
        self._stiffness = val
    stiffness = property(_stiffness_getter, _stifness_setter)

#--------------------- Boundary conditions ----------------------------

    def bc(self, obj):
        """ Pick out solver specific boundary condition object

        Discretization methods in need of boundary conditions should access
        this method (instead of properties bc_flow, bc_transport etc).

        Parameters:

        obj: Solver
            Discretization object. Should have attribute 'physics'

        Returns:

        bc.BoundaryCondition:
            bc_flow if obj.physics equals 'flow' or 'pressure'
            bc_transport if obj.physics equals 'heat' or 'transport'
            bc_mechanics if obj.physcis equals 'mechanics', 'mech' or
                'elasticity'

        """
        if obj.physics.lower().strip() in ['flow', 'pressure']:
            return self.bc_flow
        elif obj.physics.lower().strip() in ['heat', 'transport']:
            return self.bc_transport
        elif obj.physcis.lower().strip() in ['mechanics', 'elasticity', 'mech']:
            return self.bc_mech
#---

    def _get_bc_flow(self):
        """ Boundary condition for flow problem.
        Solvers should rather access bc().
        """
        return self._bc_flow
    def _set_bc_flow(self, bnd):
        self._bc_flow = bnd
    bc_flow = property(_get_bc_flow, _set_bc_flow)

#---

    def _get_bc_transport(self):
        """ Boundary condition for transport problem.
        Solvers should rather access bc().
        """
        return self._bc_transport
    def _set_bc_transport(self, bnd):
        self._bc_transport = bnd
    bc_transport = property(_get_bc_transport, _set_bc_transport)

#---

    def _get_bc_mech(self):
        """ Boundary condition for mechanics problem.
        Solvers should rather access bc().
        """
        return self._bc_mech
    def _set_bc_mech(self, bnd):
        self._bc_mech = bnd
    bc_mech = property(_get_bc_mech, _set_bc_mech)

