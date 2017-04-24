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

from core.grids import structured, simplex
from core.bc import bc
from core.constit import second_order_tensor
from utils.errors import error

from vem import dual
import compgeom.basics as cg
from fvdiscr.transport import upwind
from fvdiscr import mass_matrix

from viz.exporter import export_vtk, export_pvd

#------------------------------------------------------------------------------#

def upwind_example0(**kwargs):
    #######################
    # Simple 2d upwind problem with explicit Euler scheme in time
    #######################
    T = 1
    Nx, Ny = 10, 1
    g = structured.CartGrid( [Nx, Ny], [1, 1] )
    g.compute_geometry()

    advect = upwind.Upwind()
    beta_n = advect.beta_n(g, [1, 0, 0])

    b_faces = g.get_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bnd_val = {'dir': np.hstack(([1], np.zeros(b_faces.size-1)))}
    data = {'beta_n': beta_n, 'bc': bnd, 'bc_val': bnd_val}

    U, rhs = advect.matrix_rhs(g, data)

    data = {'deltaT': advect.cfl(g, data)}
    M, _ = mass_matrix.Mass().matrix_rhs(g, data)

    conc = np.zeros(g.num_cells)

    M_minus_U = M - U
    invM, _ = mass_matrix.InvMass().matrix_rhs(g, data)

    # Loop over the time
    Nt = int(T / data['deltaT'])
    time = np.empty(Nt)
    for i in np.arange( Nt ):

        # Update the solution
        conc = invM.dot((M_minus_U).dot(conc) + rhs)
        time[i] = data['deltaT']*i
        export_vtk(g, "conc_EE", {"conc": conc}, time_step=i)

    export_pvd(g, "conc_EE", time)

#------------------------------------------------------------------------------#

def upwind_example1(**kwargs):
    #######################
    # Simple 2d upwind problem with implicit Euler scheme in time
    #######################
    T = 1
    Nx, Ny = 10, 1
    g = structured.CartGrid( [Nx, Ny], [1, 1] )
    g.compute_geometry()

    advect = upwind.Upwind()
    beta_n = advect.beta_n(g, [1, 0, 0])

    b_faces = g.get_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bnd_val = {'dir': np.hstack(([1], np.zeros(b_faces.size-1)))}
    data = {'beta_n': beta_n, 'bc': bnd, 'bc_val': bnd_val}

    U, rhs = advect.matrix_rhs(g, data)

    data = {'deltaT': 2*advect.cfl(g, data)}
    M, _ = mass_matrix.Mass().matrix_rhs(g, data)

    conc = np.zeros(g.num_cells)

    # Perform an LU factorization to speedup the solver
    IE_solver = sps.linalg.factorized( ( M + U ).tocsc() )

    # Loop over the time
    Nt = int(T / data['deltaT'])
    time = np.empty(Nt)
    for i in np.arange( Nt ):

        # Update the solution
        # Backward and forward substitution to solve the system
        conc = IE_solver(M.dot(conc) + rhs)
        time[i] = data['deltaT']*i
        export_vtk(g, "conc_IE", {"conc": conc}, time_step=i)

    export_pvd(g, "conc_IE", time)

#------------------------------------------------------------------------------#

def upwind_example2(**kwargs):
    #######################
    # Simple 2d upwind problem with explicit Euler scheme in time coupled with
    # a Darcy problem
    #######################
    T = 2
    Nx, Ny = 10, 10
    g = structured.CartGrid( [Nx, Ny], [1, 1] )
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = second_order_tensor.SecondOrderTensor(g.dim, kxx)

    def funp_ex(pt): return -np.sin(pt[0])*np.sin(pt[1])-pt[0]

    f = np.zeros(g.num_cells)

    b_faces = g.get_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bnd_val = {'dir': funp_ex(g.face_centers[:, b_faces])}

    solver = dual.DualVEM()
    data = {'k': perm, 'f': f, 'bc': bnd, 'bc_val': bnd_val}
    D, rhs = solver.matrix_rhs(g, data)

    up = sps.linalg.spsolve(D, rhs)
    beta_n = solver.extractU(g, up)

    u, p = solver.extractU(g, up), solver.extractP(g, up)
    P0u = solver.projectU(g, u, data)
    export_vtk(g, "darcy", {"p": p, "P0u": P0u})

    advect = upwind.Upwind()

    bnd_val = {'dir': np.hstack(([1], np.zeros(b_faces.size-1)))}
    data = {'beta_n': beta_n, 'bc': bnd, 'bc_val': bnd_val}

    U, rhs = advect.matrix_rhs(g, data)

    data = {'deltaT': advect.cfl(g, data)}
    M, _ = mass_matrix.Mass().matrix_rhs(g, data)

    conc = np.zeros(g.num_cells)
    M_minus_U = M - U
    invM, _ = mass_matrix.InvMass().matrix_rhs(g, data)

    # Loop over the time
    Nt = int(T / data['deltaT'])
    time = np.empty(Nt)
    for i in np.arange( Nt ):

        # Update the solution
        conc = invM.dot((M_minus_U).dot(conc) + rhs)
        time[i] = data['deltaT']*i
        export_vtk(g, "conc_darcy", {"conc": conc}, time_step=i)

    export_pvd(g, "conc_darcy", time)

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
