import porepy as pp


def mobility(saturation, dynamic_viscosity):
    """ """
    mobility = pp.rel_perm_brooks_corey(saturation) / dynamic_viscosity
    return mobility


def mobility_operator(subdomains, saturation, dynamic_viscosity):
    """ """
    s = saturation(subdomains)
    mobility_operator = pp.ad.Function(mobility, "mobility_operator")
    mobility = mobility_operator(s, dynamic_viscosity)
    return mobility
