import numpy as np

from core.grids import structured


def setup_2d():
    grid_list = []

    # Unstructured perturbation
    nx = np.array([2, 2])
    g_cart_unpert = structured.CartGrid(nx)
    g_cart_unpert.compute_geometry()

    grid_list.append(g_cart_unpert)

    # Structured perturbation
    g_cart_spert = structured.CartGrid(nx)
    g_cart_spert.nodes[0, 4] = 1.5
    g_cart_spert.compute_geometry()
    grid_list.append(g_cart_spert)

    # Larger grid, random perturbations
    nx = np.array([3, 3])
    g_cart_rpert = structured.CartGrid(nx)
    dx = 1
    pert = .4
    g_cart_rpert.nodes = g_cart_rpert.nodes + dx * pert * \
                                              (0.5 -
                                               np.random.rand(g_cart_rpert.dim,
                                                              g_cart_rpert.num_nodes))
    g_cart_rpert.compute_geometry()
    grid_list.append(g_cart_rpert)

    return grid_list
