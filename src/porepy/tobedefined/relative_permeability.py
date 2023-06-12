import numpy as np
import porepy as pp


def rel_perm_brooks_corey(saturation):  # ->... :
    """ """
    relative_perm = saturation**2
    return relative_perm


def rel_perm_brooks_corey_operator(subdomains, saturation):
    """ """
    s = saturation(subdomains)
    rel_perm = pp.ad.Function(rel_perm_brooks_corey, "rel_perm_brooks_corey_operator")
    rel_perm = rel_perm_brooks_corey_operator(s)
    return rel_perm


def second_derivative(saturation):
    """
    move it? i need the second derivative in hu
    """
    return 2 * np.ones(saturation.shape)
