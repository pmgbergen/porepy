import numpy as np
import scipy.sparse as sps

from porepy.ad.forward_mode import Ad_array, initAdArrays


def concatenate(variables, axis=0):
    vals = [var.val for var in variables]
    jacs = np.array([var.jac for var in variables])

    vals_stacked = np.concatenate(vals, axis=axis)
    jacs_stacked = []
    jacs_stacked = sps.vstack(jacs)
    #    for i in range(jacs.shape[1]):
    #        jacs_stacked.append(sps.vstack(jacs[:, i]))

    return Ad_array(vals_stacked, jacs_stacked)
