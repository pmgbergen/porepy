import numpy as np
import scipy.sparse as sps

from porepy.grids import structured
from porepy.numerics.pdeproblem import *
from porepy.numerics.fv import tpfa, mass_matrix, fvutils
from porepy.numerics.mixed_dim.coupler import Coupler
from porepy.params.data import Parameters
from porepy.params import tensor
from porepy.params import bc
from porepy.viz.exporter import export_vtk, export_pvd


class SlightlyCompressible(PdeProblem):
    def __init__(self, physics='flow'):
        PdeProblem.__init__(self, physics)

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


class SlightlyCompressibleData(PdeProblemData):
    def __init__(self, g, data, physics='flow'):
        PdeProblemData.__init__(self, g, data, physics)

    def _set_data(self):
        PdeProblemData._set_data(self)
        self.data()['compressibility'] = self.compressibility()

    def compressibility(self):
        return 1.0

    def permeability(self):
        kxx = np.ones(self.grid().num_cells)
        return tensor.SecondOrder(self.grid().dim, kxx)

    def diffusivity(self):
        return self.permeability()
