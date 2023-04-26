import porepy as pp


def mobility(saturation, dynamic_viscosity):
    """ """
    mobility = pp.rel_perm_brooks_corey(saturation) / dynamic_viscosity
    return mobility
