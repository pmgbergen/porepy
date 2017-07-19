import numpy as np
import scipy.sparse as sps

from porepy.grids import structured
from porepy.grids.grid_bucket import GridBucket
from porepy.params.data import Parameters
from porepy.params import tensor
from porepy.params import bc
from porepy.numerics.mixed_dim.solver import Solver


class Implicit():
    """
    Class for solving slightly compressible flow using backward Euler.
    We solve the equation:
    c_p * phi * (p^k+1 - p^k)/dt - nabla * K * grad(p^k+1) = f^k+1.

    To solve a slightly compressible flow problem please see the
    porepy.numerics.compressible.problems.SlightlyCompressible class.
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
        # Get data
        g = problem.grid()

        dt = problem.time_step()
        T = problem.end_time()
        data = problem.data()
        data['pressure'] = []
        data['times'] = []

        problem.update(dt)

        flux_disc = problem.flux_disc()
        time_disc = problem.time_disc()

        p0 = problem.initial_pressure()
        p = p0
        data['pressure'].append(p)
        data['times'].append(0.0)

        self.problem = problem
        self.g = g
        self.data = data
        self.dt = dt
        self.T = T
        self.flux_disc = flux_disc
        self.time_disc = time_disc
        self.p0 = p0
        self.p = p
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
        self.problem.update(t + self.dt)
        self.p0 = self.p
        # Store result
        if self.parameters['store_results'] == True:
            self.data['pressure'].append(self.p)
            self.data['times'].append(t)

    def reassemble(self):
        """
        reassemble matrices. This must be called between every time step to
        update the rhs of the system.
        """
        lhs_flux, rhs_flux = self._discretize(self.flux_disc)
        lhs_time, rhs_time = self._discretize(self.time_disc)

        self.lhs = lhs_time + lhs_flux
        self.rhs = lhs_time * self.p0 + rhs_flux + rhs_time

    def _discretize(self, discr):
        if isinstance(self.g, GridBucket):
            lhs, rhs = discr.matrix_rhs(self.g)
        else:
            lhs, rhs = discr.matrix_rhs(self.g, self.data)
        return lhs, rhs


if __name__ == '__main__':
    from porepy.fracs import meshing
    from porepy.params.data import Parameters
    from porepy.numerics.fv.tpfa import Tpfa
    from porepy.numerics.fv.tpfa_coupling import TpfaCoupling
    from porepy.numerics.mixed_dim.coupler import Coupler

    T = 0.1
    dt = 0.001

    frac = np.array([[1, 3], [2, 2]])
    gb = meshing.cart_grid([frac], [12, 12], physdims=[4, 4])
    gb.assign_node_ordering()
    gb.add_node_props(['param'])
    for g, d in gb:
        params = Parameters(g)
        params._compressibility = 1
        if g.dim != 2:
            continue
        bc_val = np.zeros(g.num_cells)
        left = np.isclose(g.face_centers[0], 0)
        bc_val[left] = 0.1 * g.face_areas[left]
        params.set_bc_val('flow', bc_val)
        d['param'] = params
        d['time_step'] = dt
        d['pressure'] = np.zeros(g.num_cells)

    darcy_discr = Tpfa('Flow')
    solver = Compressible(darcy_discr, 'Flow')
    coupling_conditions = TpfaCoupling(solver)
    solver = Coupler(solver, coupling_conditions)

    t = dt
    pressures = np.zeros(solver.ndof(gb))
    times = [0.0]
    while t < T:
        matrix, rhs = solver.matrix_rhs(gb)
        p = sps.linalg.spsolve(matrix, rhs)
        solver.split(gb, 'pressure', p)
        pressures.append(p)
        times.append(t)
        t += dt
