import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import Ad_array

module_sections = ["assembly", "numerics"]


@pp.time_logger(sections=module_sections)
def concatenate(variables, axis=0):
    vals = [var.val for var in variables]
    jacs = np.array([var.jac for var in variables])

    vals_stacked = np.concatenate(vals, axis=axis)
    jacs_stacked = []
    jacs_stacked = sps.vstack(jacs)

    return Ad_array(vals_stacked, jacs_stacked)
