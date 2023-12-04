import numpy as np
import porepy as pp

import pdb


def rel_perm_quadratic(saturation: np.ndarray) -> np.ndarray:
    """ """

    # linear: ----------------------------------------
    # relative_perm = saturation

    # abs: ---------------------------------------
    # relative_perm = pp.ad.abs(saturation)

    # quadratic: -----------------------------------------------
    relative_perm = saturation**2

    # cubic: -----------------------------------------------
    # relative_perm = saturation**3

    return relative_perm


def rel_perm_brooks_corey(s_c_alpha, rho_alpha):
    """
    formula found in hamon 2018
    """

    def perm(saturation, s_c_alpha=s_c_alpha, rho_alpha=rho_alpha):
        return ((saturation - s_c_alpha) / (1 - s_c_alpha)) ** rho_alpha

    return perm
