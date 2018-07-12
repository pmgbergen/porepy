import numpy as np

from porepy.grids import structured, simplex
from porepy.utils import mcolon


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
    rand = np.vstack(
        (
            np.random.rand(g_cart_rpert.dim, g_cart_rpert.num_nodes),
            np.repeat(0., g_cart_rpert.num_nodes),
        )
    )
    g_cart_rpert.nodes = g_cart_rpert.nodes + dx * pert * (0.5 - rand)
    # No perturbations of the z-coordinate (which is not active in this case)
    g_cart_rpert.nodes[2, :] = 0
    g_cart_rpert.compute_geometry()
    grid_list.append(g_cart_rpert)

    return grid_list


def perturb(g, rate, dx):
    rand = np.vstack((np.random.rand(g.dim, g.num_nodes), np.repeat(0., g.num_nodes)))
    g.nodes += rate * dx * (rand - 0.5)
    # Ensure there are no perturbations in the z-coordinate
    if g.dim == 2:
        g.nodes[2, :] = 0
    return g


def make_grid(grid, grid_dims, domain, dim):
    if grid.lower() == "cart" or grid.lower() == "cartesian":
        return structured.CartGrid(grid_dims, domain)
    elif (grid.lower() == "simplex" and dim == 2) or grid.lower() == "triangular":
        return simplex.StructuredTriangleGrid(grid_dims, domain)


def grid_sequence(basedim, num_levels, grid_type, pert=0, ref_rate=2, domain=None):
    dim = basedim.shape[0]

    if domain is None:
        domain = np.ones(dim)
    for iter1 in range(num_levels):
        nx = basedim * ref_rate ** iter1
        g = make_grid(grid_type, nx, domain, dim)
        if pert > 0:
            dx = np.max(domain / nx)
            g = perturb(g, pert, dx)
        g.compute_geometry()
        yield g


def grid_sequence_fixed_lines(
    basedim, num_levels, grid_type, pert=0, ref_rate=2, domain=None, subdom_func=None
):
    dim = basedim.shape[0]

    if domain is None:
        domain = np.ones(dim)

    for iter1 in range(num_levels):
        nx = basedim * ref_rate ** iter1
        g = make_grid(grid_type, nx, domain, dim)

        if pert > 0:
            g.compute_geometry()
            old_nodes = g.nodes.copy()
            dx = np.max(domain / nx)
            g = perturb(g, pert, dx)
            if subdom_func is not None:
                # Characteristic function for all cell centers
                xc = g.cell_centers
                chi = subdom_func(xc[0], xc[1])
                #
                chi_face = np.abs(g.cell_faces * chi)
                bnd_face = np.argwhere(chi_face > 0).squeeze(1)
                node_ptr = g.face_nodes.indptr
                node_ind = g.face_nodes.indices
                bnd_nodes = node_ind[
                    mcolon.mcolon(node_ptr[bnd_face], node_ptr[bnd_face + 1])
                ]
                g.nodes[:, bnd_nodes] = old_nodes[:, bnd_nodes]

        g.compute_geometry()
        yield g
