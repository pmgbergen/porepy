import numpy as np
import porepy as pp

import pdb


def rel_perm_quadratic(saturation):  # ->... :
    """ """
    # print("\inside rel_perm_brooks_corey")
    # print("saturation = ", saturation)
    # if type(saturation) == pp.ad.AdArray:
    #     saturation.val = np.clip(saturation.val, 0, 1)
    #     print("saturation.val = ", saturation.val)

    # const: ---------------------------------------
    # if type(saturation) == pp.ad.AdArray:
    #     # pdb.set_trace()
    #     relative_perm = 0*saturation
    #     relative_perm.val = 0.5*np.ones(saturation.val.shape[0])
    # else:
    #     relative_perm = 0.5*1

    # linear: ----------------------------------------
    # relative_perm = saturation

    # linear abs: ---------------------------------------
    # relative_perm = pp.ad.abs(saturation)

    # quadratic: -----------------------------------------------
    relative_perm = saturation**2

    # cubic: -----------------------------------------------
    # relative_perm = saturation**3

    return relative_perm


# def rel_perm_quadratic_operator(subdomains, saturation):
#     """ """
#     s = saturation(subdomains)
#     rel_perm = pp.ad.Function(rel_perm_brooks_corey, "rel_perm_brooks_corey_operator")
#     rel_perm = rel_perm_quadratic(s)
#     return rel_perm


# def second_derivative_quadratic(saturation): # replaced by discrete second order derivative in hu
#     """
#     move it? i need the second derivative in hu
#     """
#     return 2 * np.ones(saturation.shape)


def rel_perm_brooks_corey(s_c_alpha, rho_alpha):  # ->... :
    """
    formula found in hamon 2018
    """

    def perm(saturation, s_c_alpha=s_c_alpha, rho_alpha=rho_alpha):
        return ((saturation - s_c_alpha) / (1 - s_c_alpha)) ** rho_alpha

    return perm


# USELESS?
# def rel_perm_brooks_corey_operator(subdomains, saturation):
#     """ """
#     s = saturation(subdomains)
#     rel_perm = pp.ad.Function(rel_perm_brooks_corey, "rel_perm_brooks_corey_operator")
#     rel_perm = rel_perm_brooks_corey_operator(s)
#     return rel_perm
