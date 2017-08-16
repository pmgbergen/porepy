"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import sys
import getopt
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import time
import traceback
import logging
from inspect import isfunction, getmembers

from porepy.grids import structured, simplex
from porepy.params.bc import BoundaryCondition
from porepy.params import tensor
from porepy.params.data import Parameters

from porepy.utils.errors import error
from porepy.utils import comp_geom as cg
from porepy.numerics.vem import dual
from porepy.numerics.fv.transport import upwind
from porepy.numerics.fv import mass_matrix
from porepy.viz.exporter import export_vtk, export_pvd

#------------------------------------------------------------------------------#

def upwind_example0(**kwargs):
    #######################
    # Simple 2d upwind problem with explicit Euler scheme in time
    #######################
    T = 1
    Nx, Ny = 10, 1
    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    advect = upwind.Upwind("transport")
    param = Parameters(g)
    param.set_discharge(advect.discharge(g, [1, 0, 0]))

    b_faces = g.get_boundary_faces()
    bc = BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
    param.set_bc("transport", bc)
    param.set_bc_val("transport", bc_val)

    data = {'param': param}
    data['deltaT'] = advect.cfl(g, data)

    U, rhs = advect.matrix_rhs(g, data)
    M, _ = mass_matrix.MassMatrix().matrix_rhs(g, data)

    conc = np.zeros(g.num_cells)

    M_minus_U = M - U
    invM, _ = mass_matrix.InvMassMatrix().matrix_rhs(g, data)

    # Loop over the time
    Nt = int(T / data['deltaT'])
    time = np.empty(Nt)
    folder = 'example0'
    for i in np.arange( Nt ):

        # Update the solution
        conc = invM.dot((M_minus_U).dot(conc) + rhs)
        time[i] = data['deltaT']*i
        export_vtk(g, "conc_EE", {"conc": conc}, time_step=i, folder=folder)

    export_pvd(g, "conc_EE", time, folder=folder)

#------------------------------------------------------------------------------#

def upwind_example1(**kwargs):
    #######################
    # Simple 2d upwind problem with implicit Euler scheme in time
    #######################
    T = 1
    Nx, Ny = 10, 1
    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    advect = upwind.Upwind("transport")
    param = Parameters(g)
    param.set_discharge(advect.discharge(g, [1, 0, 0]))

    b_faces = g.get_boundary_faces()
    bc = BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
    param.set_bc("transport", bc)
    param.set_bc_val("transport", bc_val)

    data = {'param': param}
    data['deltaT'] = advect.cfl(g, data)

    U, rhs = advect.matrix_rhs(g, data)
    M, _ = mass_matrix.MassMatrix().matrix_rhs(g, data)

    conc = np.zeros(g.num_cells)

    # Perform an LU factorization to speedup the solver
    IE_solver = sps.linalg.factorized((M + U).tocsc())

    # Loop over the time
    Nt = int(T / data['deltaT'])
    time = np.empty(Nt)
    folder = 'example1'
    for i in np.arange( Nt ):

        # Update the solution
        # Backward and forward substitution to solve the system
        conc = IE_solver(M.dot(conc) + rhs)
        time[i] = data['deltaT']*i
        export_vtk(g, "conc_IE", {"conc": conc}, time_step=i, folder=folder)

    export_pvd(g, "conc_IE", time, folder=folder)

#------------------------------------------------------------------------------#

def upwind_example2(**kwargs):
    #######################
    # Simple 2d upwind problem with explicit Euler scheme in time coupled with
    # a Darcy problem
    #######################
    T = 2
    Nx, Ny = 10, 10
    def funp_ex(pt): return -np.sin(pt[0])*np.sin(pt[1])-pt[0]

    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    param = Parameters(g)

    # Permeability
    perm = tensor.SecondOrder(g.dim, kxx=np.ones(g.num_cells))
    param.set_tensor("flow", perm)

    # Source term
    param.set_source("flow", np.zeros(g.num_cells))

    # Boundaries
    b_faces = g.get_boundary_faces()
    bc = BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bc_val = np.zeros(g.num_faces)
    bc_val[b_faces] = funp_ex(g.face_centers[:, b_faces])
    param.set_bc("flow", bc)
    param.set_bc_val("flow", bc_val)

    # Darcy solver
    solver = dual.DualVEM("flow")
    data = {'param': param}
    D, rhs = solver.matrix_rhs(g, data)

    up = sps.linalg.spsolve(D, rhs)

    p, u = solver.extract_p(g, up), solver.extract_u(g, up)
    P0u = solver.project_u(g, u, data)
    export_vtk(g, "darcy", {"p": p, "P0u": P0u})

    # Discharge
    param.set_discharge(u)

    # Boundaries
    bc = BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bc_val = np.hstack(([1], np.zeros(g.num_faces - 1)))
    param.set_bc("transport", bc)
    param.set_bc_val("transport", bc_val)

    data = {'param': param}

    # Advect solver
    advect = upwind.Upwind("transport")

    U, rhs = advect.matrix_rhs(g, data)

    data['deltaT'] = advect.cfl(g, data)
    M, _ = mass_matrix.MassMatrix().matrix_rhs(g, data)

    conc = np.zeros(g.num_cells)
    M_minus_U = M - U
    invM, _ = mass_matrix.InvMassMatrix().matrix_rhs(g, data)

    # Loop over the time
    Nt = int(T / data['deltaT'])
    time = np.empty(Nt)
    folder = 'example2'
    for i in np.arange( Nt ):

        # Update the solution
        conc = invM.dot((M_minus_U).dot(conc) + rhs)
        time[i] = data['deltaT']*i
        export_vtk(g, "conc_darcy", {"conc": conc}, time_step=i, folder=folder)

    export_pvd(g, "conc_darcy", time, folder=folder)

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    # If invoked as main, run all tests
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'v:',['verbose='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    verbose = True
    # process options
    for o, a in opts:
        if o in ('-v', '--verbose'):
            verbose = bool(a)

    success_counter = 0
    failure_counter = 0

    time_tot = time.time()

    functions_list = [o for o in getmembers(
        sys.modules[__name__]) if isfunction(o[1])]

    for f in functions_list:
        func = f
        if func[0] == 'isfunction' or func[0] == 'getmembers' or \
           func[0] == 'spsolve' or func[0] == 'plot_grid' or \
           func[0] == 'export_pvd' or func[0] == 'export_vtk':
            continue
        if verbose:
            print('Running ' + func[0])

        time_loc = time.time()
        try:
            func[1]()

        except Exception as exp:
            print('\n')
            print(' ************** FAILURE **********')
            print('Example ' + func[0] + ' failed')
            print(exp)
            logging.error(traceback.format_exc())
            failure_counter += 1
            continue

        # If we have made it this far, this is a success
        success_counter += 1
    #################################
    # Summary
    #
    print('\n')
    print(' --- ')
    print('Ran in total ' + str(success_counter + failure_counter) + ' tests,'
          + ' out of which ' + str(failure_counter) + ' failed.')
    print('Total elapsed time is ' + str(time.time() - time_tot) + ' seconds')
    print('\n')

#------------------------------------------------------------------------------#
