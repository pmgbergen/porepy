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

from core.grids import structured
from core.bc import bc
from core.constit import second_order_tensor
from utils.errors import error
from vem import dual

from viz.plot_grid import plot_grid

#------------------------------------------------------------------------------#

def darcy_dualVEM_example0(**kwargs):
    Nx = Ny = 25
    g = structured.CartGrid( [Nx, Ny], [1,1] )
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = second_order_tensor.SecondOrderTensor(g.dim, kxx)

    f = np.ones(g.num_cells)

    b_faces = g.get_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bnd_val = {'dir': np.zeros(b_faces.size)}

    solver = dual.DualVEM()
    data = {'k': perm, 'f': f, 'bc': bnd, 'bc_val': bnd_val}
    D, rhs = solver.matrix_rhs(g, data)

    up = sps.linalg.spsolve(D, rhs)
    u, p = solver.extractU(g, up), solver.extractP(g, up)
    P0u = solver.projectU(g, u, data)

    if kwargs['visualize']: plot_grid(g, p, P0u)

    return error.norm_L2(g, p), error.norm_L2(g, P0u)

#------------------------------------------------------------------------------#

def darcy_dualVEM_example1(**kwargs):
    Nx = Ny = 25
    g = structured.CartGrid( [Nx, Ny], [1,1] )
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = second_order_tensor.SecondOrderTensor(g.dim, kxx)

    def funP_ex(pt): return np.sin(2*np.pi*pt[0]) * \
                            np.sin(2*np.pi*pt[1])
    def funU_ex(pt): return [\
                          -2*np.pi*np.cos(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1]),
                          -2*np.pi*np.sin(2*np.pi*pt[0])*np.cos(2*np.pi*pt[1]),
                           0]
    def fun(pt): return 8*np.pi**2 * funP_ex(pt)

    f = np.array([fun(pt) for pt in g.cell_centers.T])

    b_faces = g.get_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ['dir']*b_faces.size)
    bnd_val = {'dir': funP_ex(g.face_centers[:, b_faces])}

    solver = dual.DualVEM()
    data = {'k': perm, 'f': f, 'bc': bnd, 'bc_val': bnd_val}
    D, rhs = solver.matrix_rhs(g, data)

    up = sps.linalg.spsolve(D, rhs)
    u, p = solver.extractU(g, up), solver.extractP(g, up)
    P0u = solver.projectU(g, u, data)

    if kwargs['visualize']: plot_grid(g, p, P0u)

    p_ex = error.interpolate(g, funP_ex)
    u_ex = error.interpolate(g, funU_ex)

    return error.error_L2(g, p, p_ex), error.error_L2(g, P0u, u_ex)

#------------------------------------------------------------------------------#

if __name__ == '__main__':
    # If invoked as main, run all tests
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'v:',['verbose=','visualize='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    verbose = 0
    visualize = False
    # process options
    for o, a in opts:
        if o in ('-v', '--verbose'):
            verbose = bool(a)
        elif o == '--visualize':
            visualize = bool(a)

    success_counter = 0
    failure_counter = 0

    time_tot = time.time()

    #######################
    # Simple 2d Darcy problem with known exact solution
    example = 0
    if verbose > 0:
        print('Example', example )
        print('Run 2d Darcy example')
    try:
        time_loc = time.time()
        known = np.array([0.041554943620853595, 0.18738227880674516])
        answer = darcy_dualVEM_example0(visualize=visualize)
        assert np.allclose( answer, known )

        if verbose > 0:
            print('Example', example, 'completed successfully')
            print('Elapsed time ' + str(time.time() - time_loc))
        success_counter += 1
    except Exception as exp:
        print('\n')
        print(' ***** FAILURE ****')
        print('Example', example, 'failed')
        print(exp)
        logging.error(traceback.format_exc())
        failure_counter += 1

    #######################
    # Simple 2d Darcy problem with known exact solution
    example += 1
    if verbose > 0:
       print('Example', example )
       print('Run 2d Darcy example with known solution')
    try:
        time_loc = time.time()
        known = np.array([0.0210718223032, 0.00526933885613])
        answer = darcy_dualVEM_example1(visualize=visualize)
        assert np.allclose( answer, known )

        if verbose > 0:
            print('Example', example, 'completed successfully')
            print('Elapsed time ' + str(time.time() - time_loc))
        success_counter += 1
    except Exception as exp:
        print('\n')
        print(' ***** FAILURE ****')
        print('Example', example, 'failed')
        print(exp)
        logging.error(traceback.format_exc())
        failure_counter += 1


