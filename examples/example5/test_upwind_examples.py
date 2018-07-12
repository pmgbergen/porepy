"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import numpy as np
import unittest

import scipy.sparse as sps

from porepy.grids import structured
from porepy.params.bc import BoundaryCondition
from porepy.params import tensor
from porepy.params.data import Parameters

from porepy.numerics.vem import vem_dual, vem_source
from porepy.numerics.fv.transport import upwind
from porepy.numerics.fv import mass_matrix
from porepy.viz.exporter import Exporter

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_upwind_example0(self, if_export=False):
        #######################
        # Simple 2d upwind problem with explicit Euler scheme in time
        #######################
        T = 1
        Nx, Ny = 4, 1
        g = structured.CartGrid([Nx, Ny], [1, 1])
        g.compute_geometry()

        advect = upwind.Upwind("transport")
        param = Parameters(g)
        dis = advect.discharge(g, [1, 0, 0])

        b_faces = g.get_all_boundary_faces()
        bc = BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
        param.set_bc("transport", bc)
        param.set_bc_val("transport", bc_val)

        data = {"param": param, "discharge": dis}
        data["deltaT"] = advect.cfl(g, data)

        U, rhs = advect.matrix_rhs(g, data)
        OF = advect.outflow(g, data)
        M, _ = mass_matrix.MassMatrix().matrix_rhs(g, data)

        conc = np.zeros(g.num_cells)

        M_minus_U = M - U
        invM, _ = mass_matrix.InvMassMatrix().matrix_rhs(g, data)

        # Loop over the time
        Nt = int(T / data["deltaT"])
        time = np.empty(Nt)
        folder = "example0"
        production = np.zeros(Nt)
        save = Exporter(g, "conc_EE", folder)
        for i in np.arange(Nt):

            # Update the solution
            production[i] = np.sum(OF.dot(conc))
            conc = invM.dot((M_minus_U).dot(conc) + rhs)
            time[i] = data["deltaT"] * i
            if if_export:
                save.write_vtk({"conc": conc}, time_step=i)

        if if_export:
            save.write_pvd(time)

        known = 1.09375
        assert np.sum(production) == known

    # ------------------------------------------------------------------------------#

    def test_upwind_example1(self, if_export=False):
        #######################
        # Simple 2d upwind problem with implicit Euler scheme in time
        #######################
        T = 1
        Nx, Ny = 10, 1
        g = structured.CartGrid([Nx, Ny], [1, 1])
        g.compute_geometry()

        advect = upwind.Upwind("transport")
        param = Parameters(g)
        dis = advect.discharge(g, [1, 0, 0])

        b_faces = g.get_all_boundary_faces()
        bc = BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
        param.set_bc("transport", bc)
        param.set_bc_val("transport", bc_val)

        data = {"param": param, "discharge": dis}
        data["deltaT"] = advect.cfl(g, data)

        U, rhs = advect.matrix_rhs(g, data)
        M, _ = mass_matrix.MassMatrix().matrix_rhs(g, data)

        conc = np.zeros(g.num_cells)

        # Perform an LU factorization to speedup the solver
        IE_solver = sps.linalg.factorized((M + U).tocsc())

        # Loop over the time
        Nt = int(T / data["deltaT"])
        time = np.empty(Nt)
        folder = "example1"
        save = Exporter(g, "conc_IE", folder)
        for i in np.arange(Nt):

            # Update the solution
            # Backward and forward substitution to solve the system
            conc = IE_solver(M.dot(conc) + rhs)
            time[i] = data["deltaT"] * i
            if if_export:
                save.write_vtk({"conc": conc}, time_step=i)

        if if_export:
            save.write_pvd(time)

        known = np.array(
            [
                0.99969927,
                0.99769441,
                0.99067741,
                0.97352474,
                0.94064879,
                0.88804726,
                0.81498958,
                0.72453722,
                0.62277832,
                0.51725056,
            ]
        )
        assert np.allclose(conc, known)

    # ------------------------------------------------------------------------------#

    def test_upwind_example2(self, if_export=False):
        #######################
        # Simple 2d upwind problem with explicit Euler scheme in time coupled with
        # a Darcy problem
        #######################
        T = 2
        Nx, Ny = 10, 10
        folder = "example2"

        def funp_ex(pt):
            return -np.sin(pt[0]) * np.sin(pt[1]) - pt[0]

        g = structured.CartGrid([Nx, Ny], [1, 1])
        g.compute_geometry()

        param = Parameters(g)

        # Permeability
        perm = tensor.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Boundaries
        b_faces = g.get_all_boundary_faces()
        bc = BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.zeros(g.num_faces)
        bc_val[b_faces] = funp_ex(g.face_centers[:, b_faces])
        param.set_bc("flow", bc)
        param.set_bc_val("flow", bc_val)

        # Darcy solver
        data = {"param": param}
        solver = vem_dual.DualVEM("flow")
        D_flow, b_flow = solver.matrix_rhs(g, data)

        solver_source = vem_source.DualSource("flow")
        D_source, b_source = solver_source.matrix_rhs(g, data)

        up = sps.linalg.spsolve(D_flow + D_source, b_flow + b_source)

        p, u = solver.extract_p(g, up), solver.extract_u(g, up)
        P0u = solver.project_u(g, u, data)

        save = Exporter(g, "darcy", folder)

        if if_export:
            save.write_vtk({"pressure": p, "P0u": P0u})

        # Discharge
        dis = u

        # Boundaries
        bc = BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
        bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
        param.set_bc("transport", bc)
        param.set_bc_val("transport", bc_val)

        data = {"param": param, "discharge": dis}

        # Advect solver
        advect = upwind.Upwind("transport")

        U, rhs = advect.matrix_rhs(g, data)

        data["deltaT"] = advect.cfl(g, data)
        M, _ = mass_matrix.MassMatrix().matrix_rhs(g, data)

        conc = np.zeros(g.num_cells)
        M_minus_U = M - U
        invM, _ = mass_matrix.InvMassMatrix().matrix_rhs(g, data)

        # Loop over the time
        Nt = int(T / data["deltaT"])
        time = np.empty(Nt)
        save.change_name("conc_darcy")
        for i in np.arange(Nt):

            # Update the solution
            conc = invM.dot((M_minus_U).dot(conc) + rhs)
            time[i] = data["deltaT"] * i
            if if_export:
                save.write_vtk({"conc": conc}, time_step=i)

        if if_export:
            save.write_pvd(time)

        known = np.array(
            [
                9.63168200e-01,
                8.64054875e-01,
                7.25390695e-01,
                5.72228235e-01,
                4.25640080e-01,
                2.99387331e-01,
                1.99574336e-01,
                1.26276876e-01,
                7.59011550e-02,
                4.33431230e-02,
                3.30416807e-02,
                1.13058617e-01,
                2.05372538e-01,
                2.78382057e-01,
                3.14035373e-01,
                3.09920132e-01,
                2.75024694e-01,
                2.23163145e-01,
                1.67386939e-01,
                1.16897527e-01,
                1.06111312e-03,
                1.11951850e-02,
                3.87907727e-02,
                8.38516119e-02,
                1.36617802e-01,
                1.82773271e-01,
                2.10446545e-01,
                2.14651936e-01,
                1.97681518e-01,
                1.66549151e-01,
                3.20751341e-05,
                9.85780113e-04,
                6.07062715e-03,
                1.99393042e-02,
                4.53237556e-02,
                8.00799828e-02,
                1.17199623e-01,
                1.47761481e-01,
                1.64729339e-01,
                1.65390555e-01,
                9.18585872e-07,
                8.08267622e-05,
                8.47227168e-04,
                4.08879583e-03,
                1.26336029e-02,
                2.88705048e-02,
                5.27841497e-02,
                8.10459333e-02,
                1.07956484e-01,
                1.27665318e-01,
                2.51295298e-08,
                6.29844122e-06,
                1.09361990e-04,
                7.56743783e-04,
                3.11384414e-03,
                9.04446601e-03,
                2.03443897e-02,
                3.75208816e-02,
                5.89595194e-02,
                8.11457277e-02,
                6.63498510e-10,
                4.73075468e-07,
                1.33728945e-05,
                1.30243418e-04,
                7.01905707e-04,
                2.55272292e-03,
                6.96686157e-03,
                1.52290448e-02,
                2.78607282e-02,
                4.40402650e-02,
                1.71197497e-11,
                3.47118057e-08,
                1.57974045e-06,
                2.13489614e-05,
                1.48634295e-04,
                6.68104990e-04,
                2.18444135e-03,
                5.58646819e-03,
                1.17334966e-02,
                2.09744728e-02,
                4.37822313e-13,
                2.52373622e-09,
                1.83589660e-07,
                3.40553325e-06,
                3.02948532e-05,
                1.66504215e-04,
                6.45119867e-04,
                1.90731440e-03,
                4.53436628e-03,
                8.99977737e-03,
                1.12627412e-14,
                1.84486857e-10,
                2.13562387e-08,
                5.39492977e-07,
                6.08223906e-06,
                4.05535296e-05,
                1.84731221e-04,
                6.25871542e-04,
                1.66459389e-03,
                3.59980231e-03,
            ]
        )

        assert np.allclose(conc, known)


# ------------------------------------------------------------------------------#
