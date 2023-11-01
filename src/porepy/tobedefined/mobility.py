import porepy as pp



"""
"""

# useless:
def mobility_quadratic(saturation, dynamic_viscosity):
    """ """
    mob = pp.rel_perm_quadratic(saturation) / dynamic_viscosity
    return mob


# useless:
def mobility_quadratic_operator(subdomains, saturation, dynamic_viscosity):
    """ """
    s = saturation(subdomains)
    mob_operator = pp.ad.Function(mobility_quadratic, "mobility_operator")
    mob = mob_operator(s, pp.ad.Scalar(dynamic_viscosity))
    return mob






def mobility(relative_permeability):
    """ """
    def mob(saturation, dynamic_viscosity, relative_permeability=relative_permeability):
        return relative_permeability(saturation) / dynamic_viscosity
    return mob


def mobility_operator(mobility):
    """ """
    def mob_op(subdomains, saturation, dynamic_viscosity, mobility=mobility):
        s = saturation(subdomains)
        mob_operator = pp.ad.Function(mobility, "mobility_operator")
        mob = mob_operator(s, pp.ad.Scalar(dynamic_viscosity))
        return mob
    return mob_op
