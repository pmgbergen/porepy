import numpy as np
import scipy.sparse as sps

from porepy.grids import structured
from porepy.numerics.parabolic import *
from porepy.numerics.fv import tpfa, mass_matrix, fvutils
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.params.data import Parameters
from porepy.params import tensor
from porepy.params import bc
from porepy.viz.exporter import export_vtk, export_pvd


class SlightlyCompressible(ParabolicProblem):
    '''
    Inherits from ParabolicProblem
    This class solves equations of the type:
    phi *c_p dp/dt  - \nabla K \nabla p = q

    Init:
    - gb (Grid/GridBucket) Grid or grid bucket for the problem
    - physics (string) Physics key word. See Parameters class for valid physics

    functions:
    discharge(): computes the discharges and saves it in the grid bucket as 'p'
    Also see functions from ParabolicProblem

    Example:
    # We create a problem with standard data

    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['problem'] = SlightlyCompressibleData(g, d)
    problem = SlightlyCompressible(gb)
    problem.solve()
   '''

    def __init__(self, gb, physics='flow'):
        ParabolicProblem.__init__(self, gb, physics)

    def space_disc(self):
        return self.diffusive_disc(), self.source_disc()

    def time_disc(self):
        """
        Returns the time discretization.
        """
        class TimeDisc(mass_matrix.MassMatrix):
            def __init__(self, deltaT):
                self.deltaT = deltaT

            def matrix_rhs(self, g, data):
                lhs, rhs = mass_matrix.MassMatrix.matrix_rhs(self, g, data)
                return lhs * data['compressibility'], rhs * data['compressibility']
        single_dim_discr = TimeDisc(self.time_step())
        multi_dim_discr = Coupler(single_dim_discr)
        return multi_dim_discr

    def discharge(self):
        self.diffusive_disc().split(self.grid(), 'p', self._solver.p)
        fvutils.compute_discharges(self.grid())


class SlightlyCompressibleData(ParabolicData):
    '''
    Inherits from ParabolicData
    Base class for assigning valid data for a slighly compressible problem.
    Init:
    - g    (Grid) Grid that data should correspond to
    - d    (dictionary) data dictionary that data will be assigned to
    - physics (string) Physics key word. See Parameters class for valid physics

    Functions:
        compressibility: (float) the compressibility of the fluid
        permeability: (tensor.SecondOrder) The permeability tensor for the rock.
                      Setting the permeability is equivalent to setting
                      the ParabolicData.diffusivity() function. 
    Example:
    # We set an inflow and outflow boundary condition by overloading the
    # bc_val term
    class ExampleData(SlightlyCompressibleData):
        def __init__(g, d):
            SlightlyCompressibleData.__init__(self, g, d)
        def bc_val(self):
            left = self.grid().nodes[0] < 1e-6
            right = self.grid().nodes[0] > 1 - 1e-6
            val = np.zeros(g.num_faces)
            val[left] = 1
            val[right] = -1
            return val
    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['problem'] = ExampleData(g, d)
    '''

    def __init__(self, g, data, physics='flow'):
        ParabolicData.__init__(self, g, data, physics)

    def _set_data(self):
        ParabolicData._set_data(self)
        self.data()['compressibility'] = self.compressibility()

    def compressibility(self):
        return 1.0

    def permeability(self):
        kxx = np.ones(self.grid().num_cells)
        return tensor.SecondOrder(self.grid().dim, kxx)

    def diffusivity(self):
        return self.permeability()
