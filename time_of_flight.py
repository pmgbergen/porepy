"""
Module to compute time of flight based on a flux field.

"""
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve


def compute_tof(g, flux, poro, q):
    """
    Compute time of flight using a upstream weighted finite volume method.

    The function is a translation of the corresponding MRST function, albeit
    without the more advanced options.

    Parameters:
        g (core.grids.grid): Grid structure.
        flux (np.array, size num_faces): Flow field used in the computation.
        poro (np.array, size num_cells): Cell-wise porosity
        q (np.array, size num_cells): Combined source terms and flux
        contribution from boundary conditions.

    Returns:
        np.array, size num_cells: Cell-wise time of flight

    """

    # Get neighbors on a dense form (array of two rows)
    neighs = g.cell_face_as_dense()
    # We're only interested in internal faces, boundaries are hanled below
    is_int = np.all(neighs >=0, axis=0)
    int_neigh = neighs[:, is_int]

    # Outflow fluxes are non-positive
    out_flow = np.minimum(f[is_int], 0)
    # Inflow fluxes are non-negative
    in_flow = np.maximum(f[is_int], 0)

    # Find accumulation in each cell.
    # A positive flow is from neigh[0] to neigh[1], so in_flow will
    # add to neigh[1]. Conversely, the accumulation in neigh[0] is
    # the negative of outflow
    accum = np.bincount(np.hstack((int_neigh[1], int_neigh[0])),
                                            weights=np.hstack((in_flow,
                                                               -out_flow)))

    # To consider flow from sources/boundaries to cells, we only need to
    # consider positive sources
    np.clip(q, 0, out=q)

    # The average TOF for cells with sources are set to twice the time it takes
    # to fill the cell. Achieve this by adding twice the sources.
    sources = accum + 2 * q

    nc = g.num_cells

    # Upstream weighting of fluxes taken out of cells
    A = sps.coo_matrix((-in_flow, (int_neigh[1], int_neigh[0])), shape=(nc,
                                                                        nc)) \
                + sps.coo_matrix((out_flow, (int_neigh[0], int_neigh[1])),
                                 shape=(nc, nc))

    # Add accumulation in each cell
    A += sps.dia_matrix((sources, 0), shape=(nc, nc))

    # Pore volume equals porosity times cell volume
    pv = g.cell_volumes * poro

    tof = spsolve(A, pv)
    return tof
