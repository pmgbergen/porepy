import numpy as np
import scipy.sparse as sps

from porepy.grids import structured
from porepy.params.data import Parameters
from porepy.params import tensor
from porepy.params import bc


class Implicit():
    """
    Class for solving slightly compressible flow using backward Euler.
    We solve the equation:
    c_p * phi * (p^k+1 - p^k)/dt - nabla * K * grad(p^k+1) = f^k+1.
    """

    def __init__(self, problem):
        """
        Parameters:
        ----------
        problem: a problem class. Must have the attributes
            problem.grid()
            problem.porosity()
            problem.compressibility()
            problem.flux_disc()
            problem.permeability()
            problem.time_step()
            problem.end_time()
            problem.bc()
            problem.bc_val(t)
            problem.initial_pressure()
            problem.source(t)
        """
        g = problem.grid()
        por = problem.porosity()
        cell_vol = g.cell_volumes
        c_p = problem.compressibility()
        dt = problem.time_step()
        T = problem.end_time()
        flux_disc = problem.flux_disc()
        param = Parameters(g)
        data = dict()
        param.set_tensor(flux_disc, problem.permeability())
        param.set_source(flux_disc, problem.source(dt))
        param.set_bc(flux_disc, problem.bc())
        param.set_bc_val(flux_disc, problem.bc_val(dt))
        data['param'] = param
        data['pressure'] = []
        data['times'] = []
        lhs_flux, rhs_flux = flux_disc.matrix_rhs(g, data)
        p0 = problem.initial_pressure()
        p = p0
        data['pressure'].append(p)
        data['times'].append(0.0)

        self.problem = problem
        self.g = g
        self.data = data
        self.por = por
        self.dt = dt
        self.T = T
        self.flux_disc = flux_disc
        self.p0 = p0
        self.p = p
        self.lhs_flux = lhs_flux
        self.rhs_flux = rhs_flux

        # First initial empty lhs and rhs, then initialize them throug
        # reassemble
        self.lhs = []
        self.rhs = []
        self.reassemble()

        self.parameters = {'store_results': False, 'verbose': False}

    def solve(self):
        """
        Solve problem.
        """
        t = self.dt
        while t < self.T + 1e-14:
            if self.parameters['verbose']:
                print('solving time step: ', t)
            self.step()
            self.update(t)
            self.reassemble()
            t += self.dt
        return self.data

    def step(self):
        """
        Take one time step
        """
        self.p = sps.linalg.spsolve(self.lhs, self.rhs)

    def update(self, t):
        """
        update parameters for next time step
        """
        param = self.data['param']
        param.set_source(self.flux_disc, self.problem.source(t + self.dt))
        param.set_bc_val(self.flux_disc, self.problem.bc_val(t + self.dt))
        self.p0 = self.p
        self.reassemble()
        # Store result
        if self.parameters['store_results'] == True:
            self.data['pressure'].append(self.p)
            self.data['times'].append(t)

    def reassemble(self):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system. 
        """
        bound_flux = self.data['bound_flux']
        param = self.data['param']
        por = self.problem.porosity()
        cell_vol = self.g.cell_volumes
        c_p = self.problem.compressibility()
        I = np.eye(self.flux_disc.ndof(self.g))

        self.lhs = cell_vol * por * c_p * I + self.dt * self.lhs_flux
        self.rhs_flux = self.dt * self.flux_disc.rhs(
            self.g, bound_flux, param.bc_val_flow, param.source_flow)
        self.rhs = cell_vol * por * c_p * self.p0 + self.dt * self.rhs_flux
